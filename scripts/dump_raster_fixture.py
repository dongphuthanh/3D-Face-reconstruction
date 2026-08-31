"""Dump the rasteriser's inputs and reference output as flat binaries.

For writing a CUDA replacement for face3d.render._assign_faces without needing
PyTorch on the CUDA side. Everything lands in out/raster_fixture/ as raw
little-endian arrays plus a meta.json, so a standalone .cu can fread() them and
diff its face-id buffer against a known-good answer.

Why _assign_faces and not the whole of rasterize: it is 94% of the time (15.3 of
16.3 ms at B=8, 224px on a 5070), and it is the only part that is NOT
differentiable. It runs under no_grad and returns integer ids; rasterize then
recomputes barycentrics from them in PyTorch, which is where the gradients come
from. So a kernel replacing it needs no backward pass, and autograd keeps
working untouched.

What the PyTorch version is doing badly, and what a kernel avoids: it cannot
scatter without materialising candidates, so it builds a (B, F, K*K, 3) tensor
per bucket. Measured at B=8/224: 6.69 M candidate (face, pixel) pairs tested
against 0.46 M pixels of actual triangle area -- 14.4x more work than the
geometry needs -- and 196 MB of peak memory. The median triangle spans 2.4 px
while its bucket pays for K*K.

    python scripts/dump_raster_fixture.py --batch 8 --size 224
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d import assets
from face3d.flame_torch import FlameTorch
from face3d.pipeline import project
from face3d.render import _assign_faces

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--scale", type=float, default=7.0, help="camera scale")
    ap.add_argument("--out", default=str(ROOT / "out" / "raster_fixture"))
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    flame = FlameTorch(assets.model_path("FLAME2023Open/flame2023_Open.pkl"))
    B, S = a.batch, a.size

    # A batch of DIFFERENT faces, not one repeated: identical geometry would
    # hide any bug whose trigger is per-image, and would let a kernel that
    # silently reused image 0 still pass.
    g = torch.Generator().manual_seed(0)
    shape = torch.randn(B, flame.n_shape, generator=g) * 0.8
    expr = torch.randn(B, flame.n_expr, generator=g) * 0.3
    with torch.no_grad():
        verts, _ = flame(shape, expr, None)
        cam = torch.tensor([[a.scale, 0.0, 0.0]]).expand(B, 3)
        ndc = project(verts, cam)

    # Exactly the transform rasterize() applies before calling _assign_faces.
    px_x = (ndc[..., 0] * 0.5 + 0.5) * S
    px_y = (0.5 - ndc[..., 1] * 0.5) * S
    verts_px = torch.stack([px_x, px_y], -1).contiguous()      # (B,V,2)
    depth = ndc[..., 2].contiguous()                           # (B,V)
    faces = flame.faces.contiguous()                           # (F,3) int64

    tri = verts_px[:, faces]
    span = (tri.max(-2).values - tri.min(-2).values).max()
    K = max(2, min(int(span.ceil().item()) + 2, 64))

    with torch.no_grad():
        fid = _assign_faces(verts_px, depth, faces, S, S, K)    # (B,H,W) int64

    def dump(name, arr):
        p = out / name
        arr.tofile(p)
        print(f"  {name:16s} {str(arr.dtype):8s} {arr.shape}  {p.stat().st_size/1e6:.2f} MB")

    dump("verts_px.f32", verts_px.numpy().astype("<f4"))
    dump("depth.f32", depth.numpy().astype("<f4"))
    dump("faces.i32", faces.numpy().astype("<i4"))
    dump("ref_fid.i32", fid.numpy().astype("<i4"))

    meta = dict(
        batch=B, verts=int(flame.n_verts), faces=int(faces.shape[0]),
        height=S, width=S, K=K,
        note=("verts_px is (B,V,2) xy in PIXELS; depth is (B,V), SMALLER is "
              "nearer; faces is (F,3) vertex indices; ref_fid is (B,H,W) with "
              "-1 where no triangle covers the pixel. A pixel belongs to the "
              "triangle with the smallest depth at its CENTRE (x+0.5, y+0.5); "
              "ties break to the LOWER face index."),
        reference=("face3d/render.py::_assign_faces -- keep it as the oracle, "
                   "the ids must match exactly, not approximately"),
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  meta.json\n\nwrote {out}")

    cov = float((fid >= 0).float().mean())
    print(f"coverage {100*cov:.1f}% of pixels; {len(np.unique(fid.numpy()))} distinct ids")


if __name__ == "__main__":
    main()
