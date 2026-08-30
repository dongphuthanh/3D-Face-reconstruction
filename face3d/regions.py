"""Named regions of the face, as soft masks in UV space.

The parametric texture layer needs to know where the lips are, where the brows
are, and what counts as plain skin. FLAME ships no such labelling, but the
MediaPipe landmark embedding already on disk does: 105 points embedded on the
surface, each carrying its MediaPipe index, and MediaPipe's canonical face mesh
has published index sets for every feature. All 20 outer-lip, 20 inner-lip and
20 eyebrow landmarks are present, so the regions that matter here are fully
covered. The nose is only half covered, which is why there is no nose region.

Masks are SOFT, and that is not decoration. A hard region boundary composited
into a texture is a visible outline on the face -- the same mistake as the
polygon edge that face_texel_mask left across the forehead, and as the
face-shaped patch the first texture generator painted. Everything here fades
out over a few millimetres of surface.

Distances are geodesic-ish only in the sense that they are Euclidean on the
neutral mesh, which is close enough at these scales: 1 FLAME unit is 1 metre,
so a 10 mm lip radius is 0.010.
"""

import numpy as np
import torch

from .albedo import CACHE_DIR

# MediaPipe canonical face-mesh index sets. Only groups fully covered by the
# 105 embedded landmarks are listed; a partly covered group would give a mask
# with a hole in it and no error to say so.
GROUPS = {
    "lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269,
             267, 0, 37, 39, 40, 185,
             78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311,
             312, 13, 82, 81, 80, 191],
    "brows": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336,
              46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
    "eyes": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386,
             385, 384, 398,
             33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160,
             161, 246],
}

# Radius at which each region reaches full strength, and the width of the fade
# beyond it. Metres.
RADII = {"lips": (0.008, 0.006), "brows": (0.009, 0.007), "eyes": (0.010, 0.006)}


def _smoothstep(x, lo, hi):
    t = ((x - lo) / max(hi - lo, 1e-9)).clip(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def texel_positions(flame, static):
    """(R,R,3) position on the neutral mesh for every texel. Topology only."""
    from .render import interpolate

    fid, bary = static[0].cpu(), static[1].cpu()
    v = flame.v_template.detach().cpu().float()[None]
    return interpolate(v, flame.faces.cpu(), fid[None], bary[None])[0].numpy()


def region_masks(flame, embedding, static, device="cpu"):
    """Soft (R,R) masks for each named group, plus `skin`. Cached.

    `skin` is what is left of the fitted face once lips, brows and eyes are
    taken out -- the region a single "skin colour" is actually a fair summary
    of. Averaging colour over a face that still contains lips and eyebrows is
    how you get a skin tone nobody has.
    """
    R = static[0].shape[0]
    cache = CACHE_DIR / f"regions_{R}.npz"
    if cache.exists():
        with np.load(cache) as d:
            return {k: torch.as_tensor(d[k], device=device) for k in d.files}

    P = texel_positions(flame, static)                       # (R,R,3)
    uv_ok = static[2].cpu().numpy()

    with torch.no_grad():
        v, _ = flame(batch_size=1)
        pts = embedding.positions(v, flame.faces)[0].cpu().numpy()   # (105,3)
    mp_idx = list(np.load(
        embedding.path if hasattr(embedding, "path") else
        CACHE_DIR.parent / "mediapipe_landmark_embedding" /
        "mediapipe_landmark_embedding.npz")["landmark_indices"])

    out = {}
    flat = P.reshape(-1, 3)
    for name, ids in GROUPS.items():
        rows = [i for i, m in enumerate(mp_idx) if m in ids]
        if not rows:
            continue
        L = pts[rows]                                        # (k,3)
        d = np.sqrt(((flat[:, None, :] - L[None, :, :]) ** 2).sum(-1)).min(1)
        r0, fade = RADII[name]
        m = 1.0 - _smoothstep(d.reshape(R, R), r0, r0 + fade)
        out[name] = (m * uv_ok).astype(np.float32)

    # Skin: the fitted face minus the features. face_texel_mask is already the
    # "region the photometric loss optimised", which is the right outer bound.
    from .albedo import face_texel_mask
    with np.load(CACHE_DIR / "flame_texture_256_50.npz") as d:
        vt, ft = d["vt"].astype(np.float32), d["ft"].astype(np.int64)
    face = face_texel_mask(flame, embedding, vt, ft, resolution=R)
    feat = np.zeros((R, R), np.float32)
    for k in out:
        feat = np.maximum(feat, out[k])
    out["skin"] = (face * (1.0 - feat)).astype(np.float32)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return {k: torch.as_tensor(v, device=device) for k, v in out.items()}


def crease_map(cache_dir, static, resolution=256, limit=1500):
    """Where faces are commonly darker at high frequency. (R,R) in 0-1. Cached.

    Derived from the corpus rather than authored, because an authored wrinkle
    map is the one part of a parametric skin layer that is pure art direction
    and this project has no artist. Averaging the DARKENING half of the
    high-frequency residual over many faces leaves the places creases land in
    most people -- nasolabial folds, forehead lines, the lash line, the lip
    seam, crow's feet -- and averages away whatever was specific to any one of
    them.

    It is a shape, not an amount. How much of it a given face gets is a
    predicted parameter.
    """
    import pathlib

    out = CACHE_DIR / f"crease_{resolution}.npy"
    if out.exists():
        return np.load(out)

    cache_dir = pathlib.Path(cache_dir)
    alb = np.load(cache_dir / "albedo.npy", mmap_mode="r")
    wgt = np.load(cache_dir / "weight.npy", mmap_mode="r")
    n = min(len(alb), limit)

    acc = np.zeros((resolution, resolution), np.float64)
    den = np.zeros((resolution, resolution), np.float64)
    from PIL import Image, ImageFilter
    for i in range(n):
        a = alb[i].astype(np.float32) / 255.0
        w = wgt[i].astype(np.float32) / 255.0
        lo = np.asarray(Image.fromarray((a * 255).astype(np.uint8))
                        .filter(ImageFilter.GaussianBlur(4))).astype(np.float32) / 255.0
        hf = (a - lo).mean(-1)
        acc += np.clip(-hf, 0, None) * w          # darkening only
        den += w
    m = acc / np.clip(den, 1e-6, None)
    m = m / max(m.max(), 1e-6)
    np.save(out, m.astype(np.float32))
    return m.astype(np.float32)
