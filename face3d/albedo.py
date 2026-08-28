"""FLAME texture space as a differentiable albedo model.

FLAME is geometry only. A photometric loss needs surface colour, so this wraps
MPI's texture PCA (mean + 200 directions over a 512x512 UV map) into something
the encoder can drive: albedo coefficients -> texture image -> sampled per pixel.

Note this is a *texture* basis, not strictly albedo: it was built from
photographs, so some lighting is baked in. BFM's albedo (what DECA's published
config uses) is cleaner reflectance from controlled capture. The trade is
licensing — the FLAME texture space is CC BY-NC-SA, i.e. redistributable, which
matters for the on-device story; BFM is a separate research registration.

The raw file is 1.26 GB of float64. Loading that on every run is intolerable, so
the first load writes a downsampled float32 cache and later runs use it.
"""

import pathlib

import numpy as np
import torch
import torch.nn.functional as F

CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "cache"


def _pool(a, factor):
    """Mean-pool the leading two axes by `factor`. Anti-aliases the downsample;
    plain striding would alias the pores and freckles into noise."""
    h, w = a.shape[:2]
    h2, w2 = h // factor, w // factor
    a = a[: h2 * factor, : w2 * factor]
    a = a.reshape(h2, factor, w2, factor, *a.shape[2:])
    return a.mean(axis=(1, 3))


def build_cache(npz_path, resolution=256, n_components=50, out=None):
    """One-time: decompress the full texture space and write a small cache."""
    npz_path = pathlib.Path(npz_path)
    out = out or CACHE_DIR / f"flame_texture_{resolution}_{n_components}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path) as d:
        mean = d["mean"]                                  # (512,512,3)
        dirs = d["tex_dir"][..., :n_components]           # (512,512,3,N)
        vt, ft = d["vt"], d["ft"]

    factor = mean.shape[0] // resolution
    if factor > 1:
        mean, dirs = _pool(mean, factor), _pool(dirs, factor)

    # The file stores channels BGR (OpenCV convention), verified by checking the
    # mean face: skin under neutral light is R > G > B, and the raw arrays give
    # the reverse. Left unswapped this renders a blue face, which the photometric
    # loss would then try to explain with nonsense albedo coefficients rather
    # than failing outright.
    mean = mean[..., ::-1]
    dirs = dirs[:, :, ::-1, :]

    # Stored 0-255; keep them in [0,1] so the render and the target photograph
    # live on the same scale.
    np.savez_compressed(
        out,
        mean=np.ascontiguousarray(mean / 255.0, dtype=np.float32),
        dirs=np.ascontiguousarray(dirs / 255.0, dtype=np.float32),
        vt=vt.astype(np.float32), ft=ft.astype(np.int64))
    return out


class FlameTexture:
    """Differentiable: albedo coefficients -> per-pixel colour."""

    def __init__(self, source, resolution=256, n_components=50, device="cpu"):
        source = pathlib.Path(source)
        if source.is_dir() or source.suffix == ".npz" and "tex_dir" in _keys(source):
            source = build_cache(source, resolution, n_components)
        cache = CACHE_DIR / f"flame_texture_{resolution}_{n_components}.npz"
        if not cache.exists():
            cache = build_cache(source, resolution, n_components)

        with np.load(cache) as d:
            self.mean = torch.as_tensor(d["mean"]).to(device)          # (H,W,3)
            self.dirs = torch.as_tensor(d["dirs"]).to(device)          # (H,W,3,N)
            self.vt = torch.as_tensor(d["vt"]).to(device)              # (Vt,2)
            self.ft = torch.as_tensor(d["ft"]).to(device)              # (F,3)
        self.n_components = self.dirs.shape[-1]
        self.resolution = self.mean.shape[0]

    def texture(self, coef):
        """(B,N) -> (B,3,H,W) for grid_sample."""
        n = min(coef.shape[1], self.n_components)
        tex = self.mean + torch.einsum("hwcn,bn->bhwc", self.dirs[..., :n], coef[:, :n])
        return tex.permute(0, 3, 1, 2)

    def uv_map(self, fid, bary):
        """Per-pixel UV from the rasteriser output. (B,H,W,2) in [0,1].

        Uses ft/vt, not the mesh's own faces: UV seams duplicate vertices, so the
        texture indexing (5118 uv verts) does not match the geometry (5023).
        """
        from .render import interpolate
        B = fid.shape[0]
        vt = self.vt.unsqueeze(0).expand(B, -1, -1)
        return interpolate(vt, self.ft, fid, bary)

    def sample(self, coef, fid, bary):
        """(B,H,W,3) albedo, sampled from the reconstructed texture."""
        uv = self.uv_map(fid, bary)
        # OBJ UVs put v=0 at the bottom; image row 0 is the top.
        grid = torch.stack([uv[..., 0], 1.0 - uv[..., 1]], -1) * 2 - 1
        tex = self.texture(coef)
        out = F.grid_sample(tex, grid, mode="bilinear", align_corners=False)
        return out.permute(0, 2, 3, 1)


def face_texel_mask(flame, embedding, vt, ft, resolution=256, radius=0.045):
    """(H,W) float mask, 1 where a UV texel belongs to the fitted face region.

    Rasterises the skin-mask triangles into UV space. Depends only on FLAME and
    the landmark embedding, not on the person, so build it once and reuse it.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    from .facemask import face_faces, face_region

    keep = face_faces(flame, face_region(flame, embedding, radius=radius))
    keep = keep.cpu().numpy()
    img = Image.new("L", (resolution, resolution), 0)
    draw = ImageDraw.Draw(img)
    for tri in np.asarray(ft)[keep]:
        draw.polygon([(float(vt[i, 0] * resolution),
                       float((1.0 - vt[i, 1]) * resolution)) for i in tri],
                     fill=255)
    return np.asarray(img).astype(np.float32) / 255.0


def harmonise(tex, mask, blur=10.0):
    """Blend the un-fitted region of a baked albedo toward the fitted tone.

    (H,W,3) float in [0,1] -> same. `mask` comes from face_texel_mask().

    Why this is needed. The 50 albedo coefficients drive the WHOLE UV map, but
    the basis directions carry nearly all their energy in the face, because that
    is where the training textures vary. Fitting therefore moves the face and
    leaves the neck and scalp sitting near the basis mean -- a generic
    over-saturated tone that belongs to nobody.

    Measured on one subject: the source photograph's face has saturation 0.223,
    the fitted albedo 0.201 (right, and correctly a little flatter since the
    photo carries shading), and the un-fitted neck 0.282. The neck is 27% more
    saturated than the actual person, and because the eye judges the face
    against its neighbour, a CORRECT face reads as washed out. Rendered, the
    gap widened to 64% -- lighting adds white, and white dilutes saturation.

    So this does not correct the face. It replaces a region nobody optimised
    with the one skin tone that was actually measured. Nothing is invented: the
    fill colour is the fitted region's own mean, and the Gaussian falloff keeps
    the transition seamless rather than trading one hard edge for another.

    The eyeballs survive because they fall INSIDE the skin mask -- worth knowing,
    since painting them skin-coloured would be far worse than the artifact.
    """
    import numpy as np
    from PIL import Image, ImageFilter

    inside = mask > 0.5
    if not inside.any():
        return tex
    target = tex[inside].mean(0)

    soft = Image.fromarray((mask * 255).astype(np.uint8))
    soft = np.asarray(soft.filter(ImageFilter.GaussianBlur(blur)))
    soft = (soft.astype(np.float32) / 255.0)[..., None]
    return tex * soft + target[None, None, :] * (1.0 - soft)


def _keys(p):
    import zipfile
    with zipfile.ZipFile(p) as z:
        return [n[:-4] for n in z.namelist()]
