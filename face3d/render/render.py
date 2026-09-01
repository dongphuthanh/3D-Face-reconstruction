"""Differentiable rasterisation in pure PyTorch.

Stands in for nvdiffrast, which needs nvcc + MSVC + ninja to build and so is
unavailable on this machine. Deliberately mirrors nvdiffrast's structure so the
two are swappable behind `rasterize` / `interpolate`:

  1. Decide which triangle wins each pixel. This is discrete, so it runs under
     no_grad and produces integer ids.
  2. Recompute barycentric coordinates from vertex positions for the winning
     triangle. This part IS differentiable, and it is where gradients w.r.t.
     geometry come from.

Like nvdiffrast without antialiasing, gradients are correct in triangle
interiors and absent across occlusion boundaries. That is sufficient for a
masked photometric loss on a face; it is not a general-purpose renderer.

Binning strategy: each triangle only touches pixels inside its bounding box, so
candidates are generated per-face over a KxK window rather than testing every
pixel against every face. K is derived from the actual mesh, not guessed.
"""

import torch
import torch.nn.functional as F


def weak_perspective(verts, cam):
    """DECA-style camera. verts (B,V,3), cam (B,3) = (scale, tx, ty) -> (B,V,3).

    Returns x,y in normalised device coords and z kept for depth ordering only.
    """
    s, t = cam[:, :1].unsqueeze(-1), cam[:, 1:].unsqueeze(1)
    xy = verts[..., :2] * s + t
    return torch.cat([xy, verts[..., 2:]], dim=-1)


def _bary(tri, px, py):
    """tri (...,3,2), px/py (...) -> barycentric (...,3). Differentiable in tri."""
    x0, y0 = tri[..., 0, 0], tri[..., 0, 1]
    x1, y1 = tri[..., 1, 0], tri[..., 1, 1]
    x2, y2 = tri[..., 2, 0], tri[..., 2, 1]
    d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    d = torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12), d)
    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / d
    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / d
    return torch.stack([w0, w1, 1.0 - w0 - w1], dim=-1)


@torch.no_grad()
def _assign_faces(verts_px, depth, faces, H, W, K, budget=2_000_000):
    """Z-buffer. Returns (B,H,W) long face ids, -1 where nothing was hit.

    Uses the CUDA kernel in cuda/ when one can be built, falling back to the
    PyTorch path below otherwise. Measured at the training configuration (batch
    8, 224px): 15.3 ms here against 0.26 ms there. See face3d/render/raster_cuda.py;
    FACE3D_NO_CUDA_RASTER=1 forces this path.

    The two are not quite identical, and the CUDA one is MORE correct. K exists
    only to size the candidate tensor below, and it drops triangles in three
    ways a bounding-box kernel does not: span > K falls in no bucket, span == 0
    fails the `span > 0` test, and K < 16 breaks the bucket loop before the K
    bucket is reached. None fires on FLAME geometry -- p99 span is 15.8 px
    against K = 23 -- but they are why a diff between the two paths on synthetic
    input is not automatically a bug in the kernel.

    Depth and face index are packed into one int64 key so a single amin-scatter
    resolves both the winner and the tie-break, instead of two passes. Faces are
    processed in chunks because the candidate tensor is (B, F, K*K, 3) and K
    grows with how large the mesh is drawn; amin-scatter composes across chunks,
    so this changes memory use and nothing else.
    """
    if verts_px.is_cuda:
        from . import raster_cuda
        ext = raster_cuda.extension()
        if ext is not None:
            return ext.assign_faces(verts_px.contiguous().float(),
                                    depth.contiguous().float(),
                                    faces.contiguous().int(),
                                    int(H), int(W)).long()

    B, Fn = verts_px.shape[0], faces.shape[0]
    dev = verts_px.device
    # Global depth range, so keys stay comparable between chunks.
    zmin, zmax = depth.min(), depth.max()
    scale = (1 << 30) / (zmax - zmin + 1e-12)
    BIG = (1 << 30) * Fn + Fn
    buf = torch.full((B, H * W), BIG, dtype=torch.long, device=dev)

    # Bucket faces by how large they are drawn. A single K sized for the biggest
    # triangle makes every small one allocate the same K*K candidates: on a
    # FLAME head the median triangle spans ~2 px while the largest spans ~16, so
    # one K wastes roughly an order of magnitude on 90% of the mesh.
    tri_all = verts_px[:, faces]
    span = (tri_all.max(-2).values - tri_all.min(-2).values).max(-1).values.max(0).values
    del tri_all
    edges = [4, 8, 16, K]
    lo_edge = 0
    for edge in edges:
        if edge > K:
            break
        sel = torch.nonzero((span > lo_edge) & (span <= edge), as_tuple=True)[0]
        lo_edge = edge
        if sel.numel() == 0:
            continue
        k = min(int(edge) + 2, K)

        # Chunk so the candidate tensors stay within a fixed element budget,
        # instead of a fixed face count whose memory then scales with B and K.
        per_face = max(B * k * k, 1)
        chunk = max(1, min(sel.numel(), budget // per_face))

        off = torch.arange(k, device=dev, dtype=verts_px.dtype)
        oy, ox = torch.meshgrid(off, off, indexing="ij")
        ox, oy = ox.reshape(-1), oy.reshape(-1)

        for lo in range(0, sel.numel(), chunk):
            idx = sel[lo:lo + chunk]
            fsub = faces[idx]
            tri = verts_px[:, fsub]                             # (B,f,3,2)
            zt = depth[:, fsub]

            px = tri[..., 0].min(-1).values.floor().unsqueeze(-1) + ox
            py = tri[..., 1].min(-1).values.floor().unsqueeze(-1) + oy

            w = _bary(tri.unsqueeze(2), px + 0.5, py + 0.5)      # (B,f,k*k,3)
            z = (w * zt.unsqueeze(2)).sum(-1)

            ok = (w >= 0).all(-1) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
            if not ok.any():
                continue

            zq = ((z - zmin) * scale).long().clamp(0, 1 << 30)
            fid = idx.view(1, -1, 1).expand_as(zq)
            key = torch.where(ok, zq * Fn + fid, torch.full_like(zq, BIG))
            flat = (py.long() * W + px.long()).clamp(0, H * W - 1)
            buf.scatter_reduce_(1, flat.reshape(B, -1), key.reshape(B, -1),
                                reduce="amin", include_self=True)

    return torch.where(buf == BIG, torch.full_like(buf, -1), buf % Fn).view(B, H, W)


def rasterize(verts_ndc, faces, H, W, K=None):
    """verts_ndc (B,V,3) with xy in [-1,1] -> (face_id (B,H,W), bary (B,H,W,3), mask (B,H,W)).

    Barycentrics are differentiable w.r.t. verts_ndc.
    """
    B = verts_ndc.shape[0]
    # NDC -> pixel centres; y flipped so +y is up in NDC but down in the image
    px_x = (verts_ndc[..., 0] * 0.5 + 0.5) * W
    px_y = (0.5 - verts_ndc[..., 1] * 0.5) * H
    verts_px = torch.stack([px_x, px_y], -1)
    depth = verts_ndc[..., 2]

    if K is None:
        with torch.no_grad():
            tri = verts_px[:, faces]
            span = (tri.max(-2).values - tri.min(-2).values).max()
            K = int(span.ceil().item()) + 2
            K = max(2, min(K, 64))     # 64 is a guard against a degenerate camera

    fid = _assign_faces(verts_px, depth, faces, H, W, K)
    mask = fid >= 0

    ys, xs = torch.meshgrid(torch.arange(H, device=verts_ndc.device, dtype=verts_px.dtype),
                            torch.arange(W, device=verts_ndc.device, dtype=verts_px.dtype),
                            indexing="ij")
    safe = fid.clamp(min=0)
    tri = verts_px[:, faces][torch.arange(B, device=fid.device).view(B, 1, 1), safe]  # (B,H,W,3,2)
    bary = _bary(tri, xs + 0.5, ys + 0.5) * mask.unsqueeze(-1)
    return fid, bary, mask


def interpolate(attrs, faces, fid, bary):
    """attrs (B,V,C) -> (B,H,W,C), zero outside the mask."""
    B, C = attrs.shape[0], attrs.shape[-1]
    safe = fid.clamp(min=0)
    corners = attrs[:, faces][torch.arange(B, device=fid.device).view(B, 1, 1), safe]  # (B,H,W,3,C)
    return (corners * bary.unsqueeze(-1)).sum(-2)


def vertex_normals(verts, faces):
    """Area-weighted vertex normals, (B,V,3)."""
    tri = verts[:, faces]
    fn = torch.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0], dim=-1)
    vn = torch.zeros_like(verts)
    for i in range(3):
        vn.index_add_(1, faces[:, i], fn)
    return F.normalize(vn, dim=-1, eps=1e-8)


def sh_shading(normals, sh):
    """Order-2 spherical harmonics irradiance. normals (B,H,W,3), sh (B,9,3) -> (B,H,W,3).

    The lighting model DECA uses; predicted by the encoder purely so the
    photometric loss has something comparable to the input photo.
    """
    x, y, z = normals.unbind(-1)
    c = [0.282095, 0.488603, 1.092548, 0.315392, 0.546274]
    basis = torch.stack([
        torch.ones_like(x) * c[0], c[1] * y, c[1] * z, c[1] * x,
        c[2] * x * y, c[2] * y * z, c[3] * (3 * z * z - 1),
        c[2] * x * z, c[4] * (x * x - y * y)], dim=-1)          # (B,H,W,9)
    return torch.einsum("bhwk,bkc->bhwc", basis, sh)
