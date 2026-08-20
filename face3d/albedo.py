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


def _keys(p):
    import zipfile
    with zipfile.ZipFile(p) as z:
        return [n[:-4] for n in z.namelist()]
