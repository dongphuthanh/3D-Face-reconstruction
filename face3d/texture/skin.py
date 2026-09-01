"""A parametric skin layer: named numbers -> texture, differentiably.

Same idea as face3d/render/eyes.py, which is the proof this works in this project: six
numbers (iris RGB, sclera RGB) drive a procedurally generated eye, and the
predicted iris tracks the photographed one at r = +0.822. This does the rest of
the face -- skin tone, lip colour, brow colour, freckles, creases -- as eleven.

Why bother when a learned decoder already produces a texture. Three reasons,
and only the third is about quality:

  NAMEABLE   A slider called "lip colour" is a thing a person can move. A 128-
             vector is not. This is what a game's character creator actually
             is: a curated parametric space where every combination is valid by
             construction, rather than a fit that happens to land somewhere
             plausible.

  BOUNDED    Every parameter is a sigmoid into a fixed range, so there is no
             setting of them that produces something that is not a face.

  SHARP      An L1-trained decoder is blurry, and not by accident: the
             minimiser of expected L1 over a distribution of plausible faces IS
             a blurred face. A procedural lip boundary or freckle field has no
             such pressure on it. This is the one thing the learned layer
             cannot be fixed into doing.

What it cannot do is identity. Eleven named numbers will never encode this
person's particular brow shape or mole. That is what the learned residual
underneath is still for; this composites on top of it.

REGIONAL OPERATIONS ARE THE HAZARD HERE, and the same one that has bitten this
project three times: a hard-edged mask composited into a texture is a visible
outline on the face. Lips and brows are genuinely local and use soft masks that
fade over several millimetres. Everything else is applied GLOBALLY even when it
is measured locally -- the skin tone is measured on the skin region and then
shifted over the whole head, because shifting only inside the region would draw
its boundary.
"""

import numpy as np
import torch

from ..render.albedo import CACHE_DIR

# skin RGB, lip RGB, brow RGB, freckle amount, crease amount
N_PARAMS = 11

# How much of the way toward the requested colour each region is moved. These
# scale a SHIFT, not a blend -- see compose().
LIP_STRENGTH = 0.9
BROW_STRENGTH = 0.9

# Ceilings on the multiplicative darkening effects.
FRECKLE_MAX = 0.22
CREASE_MAX = 0.45


def freckle_noise(resolution=256, scale=1.6, seed=0):
    """A fixed blotch field, freckle-sized. Cached, never per-subject.

    Deterministic on purpose. Re-randomising per request would make the same
    photograph produce a different face each time it was uploaded, which is the
    kind of thing nobody notices until a user re-runs their own portrait.
    """
    out = CACHE_DIR / f"freckles_{resolution}.npy"
    if out.exists():
        return np.load(out)

    from PIL import Image, ImageFilter
    rng = np.random.default_rng(seed)
    n = rng.random((resolution, resolution)).astype(np.float32)
    img = Image.fromarray((n * 255).astype(np.uint8))
    n = np.asarray(img.filter(ImageFilter.GaussianBlur(scale))).astype(np.float32) / 255.0
    # Re-normalise, then keep only the upper tail: freckles are sparse spots,
    # not a texture covering the whole cheek.
    n = (n - n.mean()) / max(n.std(), 1e-6)
    n = np.clip((n - 1.0) / 1.5, 0.0, 1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, n.astype(np.float32))
    return n.astype(np.float32)


def split(raw):
    """(B,N_PARAMS) unbounded -> named, bounded parameters."""
    s = torch.sigmoid(raw)
    return {
        "skin": s[:, 0:3],
        "lip": s[:, 3:6],
        "brow": s[:, 6:9],
        "freckle": s[:, 9:10],
        "crease": s[:, 10:11],
    }


def region_means(tex, weight, masks):
    """Weighted mean colour of `tex` inside each named region. (B,3) each.

    The supervision target for the colour parameters, and the reason they end
    up meaning what they are called. Trained only through the composition loss,
    "lip colour" would be free to become whatever value happened to reduce the
    error, which is useless as a slider.

    Excluding lips, brows and eyes from `skin` matters: averaged over a face
    that still contains them, "skin colour" comes out as a tone nobody has.
    """
    out = {}
    for name, m in masks.items():
        w = (m[None, None] * weight)                      # (B,1,R,R)
        out[name] = (tex * w).sum((2, 3)) / w.sum((2, 3)).clamp(min=1e-6)
    return out


def compose(base, params, masks, noise, crease):
    """base (B,3,R,R) + parameters -> textured (B,3,R,R).

    Order matters. Tone first, so the regional colours land on top of a base
    already at the right level; then lips and brows, which are local; then the
    two multiplicative detail fields, which must not be shifted afterwards or
    they stop looking like marks on skin and start looking like paint.
    """
    p = params
    out = base

    # Skin tone. MEASURED on the skin region, APPLIED to the whole head -- a
    # shift confined to the region would draw the region's outline.
    m = masks["skin"][None, None]
    cur = (out * m).sum((2, 3), keepdim=True) / m.sum((2, 3), keepdim=True).clamp(min=1e-6)
    out = out + (p["skin"][:, :, None, None] - cur)

    # Lips and brows: SHIFT the region so its mean becomes the requested
    # colour. Not a blend toward that colour, which was the first attempt and
    # was wrong in a way worth recording. The parameter is the region's MEAN,
    # and a brow region is mostly the skin around the hairs -- measured, the
    # region averaged 0.854/0.634/0.524 while the hairs themselves sat at 0.428.
    # Blending the whole region toward that skin-dominated mean erases the
    # hairs: brow contrast fell from 0.208 to 0.153 and the eyebrows went faint.
    # A shift moves the average while leaving every difference within the
    # region intact, so the hairs stay as dark relative to their surroundings as
    # they were.
    for name, key, strength in (("lip", "lips", LIP_STRENGTH),
                                ("brow", "brows", BROW_STRENGTH)):
        m = masks[key][None, None]
        cur = (out * m).sum((2, 3), keepdim=True) / m.sum((2, 3), keepdim=True).clamp(min=1e-6)
        out = out + (p[name][:, :, None, None] - cur) * m * strength

    # Freckles and creases darken; they never lighten. Applied globally rather
    # than through a skin mask, because the skin mask is a polygon and its edge
    # would show. The crease map is already face-shaped and fades on its own.
    out = out * (1 - noise[None, None] * p["freckle"][:, :, None, None] * FRECKLE_MAX)
    out = out * (1 - crease[None, None] * p["crease"][:, :, None, None] * CREASE_MAX)
    return out.clamp(0.0, 1.0)


def settle_unobserved(tex, skin_mask, unobserved, strength=0.92, factor=1.02):
    """Calm the parts of the head no frontal camera could have seen.

    Measured over 30 subjects, the underside of the jaw reached 1.68x the face's
    mean brightness and saturated to near-white on the worst of them. Two causes
    compound there, neither visible from the front:

      The PCA basis is already bright under the chin for some subjects -- 1.29x
      against 0.97x for others -- because the texture space was built from
      photographs and almost nobody photographs under a jaw.

      diffuse_fill then extends the face's correction over that region as an
      ADDITIVE shift, with no way to know the base underneath is already near
      the top of its range. A subject who needs brightening gets it applied
      where there is no headroom, and it clips.

    Capping the level alone was not enough, and the reason is worth keeping: it
    scales the brightness down but preserves the EDGE, and an edge is what reads
    as a patch rather than as shading. The crescent was structure invented for a
    region we have no information about.

    So settle it toward the one colour actually measured on this person -- the
    same argument as harmonise(), keyed on surface orientation instead of a
    polygon so it cannot draw its own outline. The remaining structure is capped
    by luminance, scaled rather than clipped so hue survives.

    Nothing here touches the face: `unobserved` is ~0.09 there and ~0.86 under
    the chin.
    """
    m = skin_mask[None, None]
    ref = (tex * m).sum((2, 3), keepdim=True) / m.sum((2, 3), keepdim=True).clamp(min=1e-6)

    a = (unobserved[None, None] * strength).clamp(0.0, 1.0)
    out = tex * (1 - a) + ref * a

    lum = out.mean(1, keepdim=True)
    ceiling = ref.mean(1, keepdim=True) * factor
    scale = (ceiling / lum.clamp(min=1e-4)).clamp(max=1.0)
    scale = 1.0 + (scale - 1.0) * unobserved[None, None]
    return (out * scale).clamp(0.0, 1.0)


def default_params(batch=1, device="cpu"):
    """Mid-range everything: the identity setting, near enough."""
    raw = torch.zeros(batch, N_PARAMS, device=device)
    return raw
