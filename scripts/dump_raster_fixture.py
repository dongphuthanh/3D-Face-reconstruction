"""Dump rasteriser inputs + reference output as flat binaries, as a test suite.

For writing a CUDA replacement for face3d.render._assign_faces without needing
PyTorch on the CUDA side. Each case lands in out/raster_fixture/<name>/ as raw
little-endian arrays plus a meta.json, so a standalone .cu can fread() them and
diff its face-id buffer against a known-good answer.

    python scripts/dump_raster_fixture.py          # write every case
    python scripts/dump_raster_fixture.py --case ties

Then, in WSL:  bash cuda/test_raster.sh

WHY _assign_faces AND NOT THE WHOLE RASTERISER. It is 94% of rasterize (15.3 of
16.3 ms at B=8, 224px on a 5070) and rasterize is ~25% of an encoder training
step. It is also the only part that is NOT differentiable: it runs under no_grad
and returns integer ids, and rasterize() recomputes barycentrics from them in
PyTorch, which is where gradients come from. A replacement therefore needs no
backward kernel.

THE CASES exist because one realistic mesh proves only the happy path. Each
targets a specific way this kernel goes wrong, and every one of them is a real
mistake someone makes on a first attempt:

    basic       a real FLAME batch. Eight DIFFERENT faces, so a kernel that
                silently rasterised image 0 into every slot cannot pass.
    tiny        32x32, one triangle. Small enough to print and read by eye.
    edges       vertices landing exactly on pixel centres. The reference counts
                a barycentric of exactly 0 as INSIDE, so `> 0` instead of
                `>= 0` shows up here and almost nowhere else.
    offscreen   geometry pushed past every border, including a triangle that
                encloses the whole frame without any vertex inside it. Clamping
                a bounding box wrongly loses that one entirely.
    degenerate  zero-area and collinear triangles, which drive the |d| < 1e-12
                guard. Divide without it and you get NaNs, and NaN comparisons
                are false, so they vanish silently instead of erroring.
    ties        two layers at EXACTLY the same depth. The winner must be the
                lower face index. Any tie-break that depends on which thread
                arrived first will be nondeterministic and fail here.
    subpixel    triangles smaller than a pixel. Most cover no pixel centre at
                all and must produce nothing.
    large       512px with big triangles: the load-imbalance case, where one
                thread per triangle leaves most of a warp idle.

A LIMIT OF THE ORACLE, worth knowing before trusting a failure. The reference
does not scan a triangle's whole bounding box: it scans a KxK window anchored at
floor(bbox min), with K clamped to 64 in rasterize(). A triangle spanning more
than ~62 px is therefore silently under-rasterised, and a kernel that scans the
real bounding box is MORE correct while reporting `spurious`. Real FLAME
triangles have a p99 span of 15.8 px so this never fires in the pipeline, but
fixtures have to stay inside it or they test the wrong thing.
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


def _flame(batch, size, scale, seed=0):
    """A real head, projected exactly as rasterize() would."""
    flame = FlameTorch(assets.model_path("FLAME2023Open/flame2023_Open.pkl"))
    g = torch.Generator().manual_seed(seed)
    shape = torch.randn(batch, flame.n_shape, generator=g) * 0.8
    expr = torch.randn(batch, flame.n_expr, generator=g) * 0.3
    with torch.no_grad():
        verts, _ = flame(shape, expr, None)
        cam = torch.tensor([[scale, 0.0, 0.0]]).expand(batch, 3)
        ndc = project(verts, cam)
    px = torch.stack([(ndc[..., 0] * 0.5 + 0.5) * size,
                      (0.5 - ndc[..., 1] * 0.5) * size], -1)
    return px.contiguous(), ndc[..., 2].contiguous(), flame.faces.contiguous(), size


def _synth(tris, size):
    """Build a fixture from an explicit list of ((x,y,z) x3) triangles."""
    v, f = [], []
    for t in tris:
        f.append([len(v), len(v) + 1, len(v) + 2])
        v.extend(t)
    v = torch.tensor(v, dtype=torch.float32)
    return (v[None, :, :2].contiguous(), v[None, :, 2].contiguous(),
            torch.tensor(f, dtype=torch.long), size)


def case_basic():
    return _flame(8, 224, 7.0)


def case_large():
    return _flame(2, 512, 9.0, seed=3)


def case_tiny():
    return _synth([[(4.0, 4.0, 0.0), (28.0, 6.0, 0.0), (10.0, 26.0, 0.0)]], 32)


def case_edges():
    # Vertices exactly ON pixel centres, so barycentrics hit exactly 0 along the
    # edges. `> 0` drops that whole boundary; `>= 0` keeps it.
    return _synth([
        [(4.5, 4.5, 0.0), (20.5, 4.5, 0.0), (4.5, 20.5, 0.0)],
        [(20.5, 20.5, 0.1), (20.5, 4.5, 0.1), (4.5, 20.5, 0.1)],
    ], 32)


def case_offscreen():
    # A 16px frame, not 32, and that is forced by the oracle rather than chosen.
    # _assign_faces buckets triangles by span into [4, 8, 16, K] and a triangle
    # with span > K falls into NO bucket and is skipped outright -- not merely
    # under-covered. K is clamped to 64, and no triangle with span <= 64 can
    # enclose a 32px frame, so at 32 the enclosing triangle simply never
    # rasterised and a correct kernel "failed" with 821 spurious pixels.
    # At 16 the enclosing triangle spans 48 and the oracle can represent it.
    return _synth([
        [(-20.0, -20.0, 0.0), (5.0, -15.0, 0.0), (-15.0, 5.0, 0.0)],   # fully outside
        [(-5.0, 4.0, 0.1), (6.0, 3.0, 0.1), (1.0, 12.0, 0.1)],         # straddles left
        [(12.0, 10.0, 0.2), (30.0, 9.0, 0.2), (15.0, 30.0, 0.2)],      # straddles corner
        [(-16.0, -8.0, 0.3), (32.0, -8.0, 0.3), (8.0, 32.0, 0.3)],     # encloses frame
    ], 16)


def case_degenerate():
    return _synth([
        [(5.0, 5.0, 0.0), (5.0, 5.0, 0.0), (5.0, 5.0, 0.0)],             # a point
        [(2.0, 2.0, 0.1), (20.0, 20.0, 0.1), (11.0, 11.0, 0.1)],         # collinear
        [(4.0, 24.0, 0.2), (26.0, 24.0, 0.2), (4.0, 24.0, 0.2)],         # zero height
        [(6.0, 6.0, 0.3), (26.0, 8.0, 0.3), (12.0, 28.0, 0.3)],          # a real one
    ], 32)


def case_ties():
    # Same geometry, same depth, three times. The reference must pick the LOWEST
    # face index every time, and so must the kernel -- deterministically.
    tri = [(4.0, 4.0, 0.5), (28.0, 6.0, 0.5), (10.0, 26.0, 0.5)]
    return _synth([tri, tri, tri], 32)


def case_subpixel():
    # Well under a pixel. Most cover no pixel CENTRE and must draw nothing;
    # a kernel that fills the bounding box regardless will report `spurious`.
    tris = []
    for i in range(12):
        x, y = 3.0 + i * 2.3, 5.0 + (i % 5) * 4.1
        tris.append([(x, y, 0.1 * i), (x + 0.3, y, 0.1 * i), (x, y + 0.3, 0.1 * i)])
    return _synth(tris, 32)


CASES = {
    "basic": case_basic, "tiny": case_tiny, "edges": case_edges,
    "offscreen": case_offscreen, "degenerate": case_degenerate,
    "ties": case_ties, "subpixel": case_subpixel, "large": case_large,
}


def write_case(name, out_root):
    verts_px, depth, faces, S = CASES[name]()
    B = verts_px.shape[0]

    tri = verts_px[:, faces]
    span = (tri.max(-2).values - tri.min(-2).values).max()
    K = max(2, min(int(span.ceil().item()) + 2, 64))

    with torch.no_grad():
        fid = _assign_faces(verts_px, depth, faces, S, S, K)

    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    verts_px.numpy().astype("<f4").tofile(out / "verts_px.f32")
    depth.numpy().astype("<f4").tofile(out / "depth.f32")
    faces.numpy().astype("<i4").tofile(out / "faces.i32")
    fid.numpy().astype("<i4").tofile(out / "ref_fid.i32")
    (out / "meta.json").write_text(json.dumps(dict(
        case=name, batch=B, verts=int(verts_px.shape[1]),
        faces=int(faces.shape[0]), height=S, width=S, K=K,
        note=("verts_px (B,V,2) xy in PIXELS, y DOWN, may fall outside the "
              "frame; depth (B,V), SMALLER is nearer; faces (F,3) vertex "
              "indices, SHARED across the batch; ref_fid (B,H,W), -1 where no "
              "triangle covers the pixel centre (x+0.5, y+0.5). Ties in depth "
              "break to the LOWER face index."),
        oracle="face3d/render.py::_assign_faces",
    ), indent=2))

    cov = float((fid >= 0).float().mean())
    print(f"  {name:11s} B={B} F={faces.shape[0]:5d} {S}x{S}  K={K:2d}  "
          f"coverage {100*cov:5.1f}%")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CASES) + ["all"], default="all")
    ap.add_argument("--out", default=str(ROOT / "out" / "raster_fixture"))
    a = ap.parse_args()

    root = pathlib.Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    names = sorted(CASES) if a.case == "all" else [a.case]
    print(f"writing {len(names)} case(s) to {root}")
    for n in names:
        write_case(n, root)
    print("\nrun them:  bash cuda/test_raster.sh")


if __name__ == "__main__":
    main()
