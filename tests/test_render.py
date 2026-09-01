"""Verify the pure-torch rasteriser: correctness, gradients, speed."""

import sys, pathlib, time
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d.geometry.flame_torch import FlameTorch
from face3d import assets
from face3d.render.render import rasterize, interpolate, vertex_normals, sh_shading

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "out"; OUT.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


MODEL = assets.model_path_or_skip()
REAL = not assets.is_fixture(MODEL)
flame = FlameTorch(MODEL).to(DEV)
faces = flame.faces
H = W = 224


def to_ndc(v):
    """Centre the head and fit it to the viewport."""
    c = (v.amax(1, keepdim=True) + v.amin(1, keepdim=True)) / 2
    r = (v.amax(1, keepdim=True) - v.amin(1, keepdim=True)).max() / 2
    n = (v - c) / r * 0.9
    return torch.cat([n[..., :2], -n[..., 2:]], -1)   # negate z: the amin z-buffer keeps the smallest, i.e. nearest


print(f"=== rasteriser ({DEV}) ===")
v, _ = flame(batch_size=1)
ndc = to_ndc(v)
fid, bary, mask = rasterize(ndc, faces, H, W)

cov = mask.float().mean().item()
check("mesh covers a plausible pixel fraction", 0.15 < cov < 0.75, f"{cov*100:.1f}% of frame")
check("face ids in range", int(fid.max()) < len(faces) and int(fid[mask].min()) >= 0)
check("barycentrics sum to 1 inside mask",
      float((bary[mask].sum(-1) - 1).abs().max()) < 1e-4)
check("barycentrics non-negative inside mask", float(bary[mask].min()) > -1e-4)
check("background is empty", float(bary[~mask].abs().max()) == 0.0)

# Depth ordering: the nearest surface point on a front-facing head is the nose
# tip, so the argmax of interpolated depth must land near the centre. Probing a
# fixed off-centre column instead would sample background and prove nothing.
z = interpolate(ndc[..., 2:], faces, fid, bary)[..., 0]
# NDC z is negated in to_ndc so the amin z-buffer keeps the nearest surface;
# "nearest" is therefore the MINIMUM z here, not the maximum.
zm = torch.where(mask, z, torch.full_like(z, 1e9))
idx = int(zm[0].argmin())
r, c = idx // W, idx % W
dist = ((r - H / 2) ** 2 + (c - W / 2) ** 2) ** 0.5 / W
check("nearest surface point is the nose (near frame centre)", dist < 0.15,
      f"nearest pixel at (row {r}, col {c}), {dist*100:.1f}% of width from centre")

# Occlusion: the far side of the head must never win a pixel. The most distant
# visible surface should stay well in front of the mesh's rearmost vertex.
vis = z[mask]
depth_span = float(ndc[..., 2].max() - ndc[..., 2].min())
# A broken or absent z-buffer shows up as visible depth reaching the rearmost
# vertex. The open neck boundary legitimately exposes some fairly deep surface,
# so the margin only has to be wide enough to separate those two cases.
check("no rear-facing surface wins a pixel",
      float(vis.max()) < float(ndc[..., 2].max()) - 0.1 * depth_span,
      f"farthest visible z {float(vis.max()):.3f} vs mesh rear z {float(ndc[...,2].max()):.3f}")

print("\n=== gradients through geometry ===")
vg = v.clone().requires_grad_(True)
fid2, bary2, mask2 = rasterize(to_ndc(vg), faces, H, W)
n = vertex_normals(vg, faces)
img = interpolate(n, faces, fid2, bary2)
img.pow(2).sum().backward()
g = vg.grad
check("d(image)/d(verts) is finite", bool(torch.isfinite(g).all()))
check("d(image)/d(verts) is non-zero", float(g.abs().max()) > 0,
      f"|g|max {float(g.abs().max()):.4e}")
touched = (g.abs().sum(-1) > 0).float().mean().item()
check("gradient reaches most vertices", touched > 0.4, f"{touched*100:.0f}% of vertices")

# The gradient must survive back to FLAME coefficients — that is the whole chain.
sh = torch.zeros(1, flame.n_shape, device=DEV, requires_grad=True)
vv, _ = flame(sh)
f3, b3, m3 = rasterize(to_ndc(vv), faces, H, W)
interpolate(vertex_normals(vv, faces), faces, f3, b3).pow(2).sum().backward()
check("gradient reaches FLAME shape coefficients", float(sh.grad.abs().max()) > 0,
      f"|g|max {float(sh.grad.abs().max()):.4e}")

print("\n=== shaded render ===")
V, _ = flame(batch_size=3)
V[1] = flame(expr=torch.zeros(1, flame.n_expr, device=DEV).index_fill_(1, torch.tensor([0], device=DEV), 2.0))[0][0]
p = torch.zeros(1, flame.n_joints * 3, device=DEV); p[0, 6] = 0.3
V[2] = flame(pose=p)[0][0]
nd = to_ndc(V)
f4, b4, m4 = rasterize(nd, faces, H, W)
nrm = interpolate(vertex_normals(V, faces), faces, f4, b4)
nrm = torch.nn.functional.normalize(nrm, dim=-1, eps=1e-8)
shc = torch.zeros(3, 9, 3, device=DEV); shc[:, 0] = 0.9; shc[:, 2] = 0.6; shc[:, 3] = 0.4
rgb = (sh_shading(nrm, shc) * m4.unsqueeze(-1)).clamp(0, 1)

from PIL import Image
strip = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
Image.fromarray(np.concatenate(list(strip), axis=1)).save(OUT / "render_test.png")
print(f"  wrote render_test.png  ({strip.shape[0]} views)")

print("\n=== speed (224x224) ===")
for B in (1, 4, 8):
    Vb, _ = flame(batch_size=B)
    nb = to_ndc(Vb)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        fi, ba, ma = rasterize(nb, faces, H, W)
        interpolate(vertex_normals(Vb, faces), faces, fi, ba)
    if DEV == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 20 * 1000
    print(f"       B={B:<2d} {dt:6.1f} ms  ({B/dt*1000:6.0f} img/s)")
if DEV == "cuda":
    print(f"       peak VRAM: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")

print("\n" + "=" * 52)
print("ALL CHECKS PASSED" if not FAILURES else f"FAILURES: {FAILURES}")
sys.exit(1 if FAILURES else 0)
