"""Can gradient descent invert the chain? Three objectives, increasing realism.

Renders a head with known coefficients and optimises from zero to recover them.
No encoder, no dataset — a failure is unambiguously in the differentiable chain.

Three stages, because "the fit is imperfect" has two very different causes and
they must not be confused:

  A. geometric   — loss directly on 3D vertices. Well-posed. If this does not
                   converge to ~0, FLAME or autograd is broken.
  B. image only  — shading of an untextured face under one light. Known to be
                   ill-posed (bas-relief ambiguity), so partial recovery here is
                   a property of the problem, not a defect in the code.
  C. image + sparse 2D landmarks — the conditioning fix real pipelines use.
                   Quantifies what the landmark term is actually buying.
"""

import sys, pathlib
import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d.flame_torch import FlameTorch
from face3d import assets
from face3d.render import rasterize, interpolate, vertex_normals, sh_shading

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
H = W = 160
NS, NE = 10, 5
FAILURES = []

torch.manual_seed(0)
MODEL = assets.model_path_or_skip()
if assets.is_fixture(MODEL):
    # Thresholds below are in millimetres against a real head; the
    # synthetic fixture has no meaningful scale to compare against.
    print("SKIP — needs the real FLAME model, not the CI fixture")
    sys.exit(0)
flame = FlameTorch(MODEL).to(DEV)
faces = flame.faces
SH = torch.zeros(1, 9, 3, device=DEV); SH[:, 0] = 0.9; SH[:, 2] = 0.6; SH[:, 3] = 0.4

# Stand-in for the FLAME landmark embedding, which is a separate MPI download we
# do not have yet. Spreading points over the front of the face approximates what
# a real 68-point set constrains; it is not a substitute for the real indices.
with torch.no_grad():
    v_ref, _ = flame(batch_size=1)
    front = (v_ref[0, :, 2] > v_ref[0, :, 2].median()) & (v_ref[0, :, 1] > v_ref[0, :, 1].min() + 0.12)
    LMK = torch.where(front)[0][torch.randperm(int(front.sum()), device=DEV)[:68]]


def geo(shape, expr, pose):
    s = torch.cat([shape, torch.zeros(1, flame.n_shape - NS, device=DEV)], 1)
    e = torch.cat([expr, torch.zeros(1, flame.n_expr - NE, device=DEV)], 1)
    return flame(s, e, pose)[0]


def to_ndc(v):
    c = (v.amax(1, keepdim=True) + v.amin(1, keepdim=True)) / 2
    n = (v - c) / 0.18 * 0.9
    return torch.cat([n[..., :2], -n[..., 2:]], -1)


def render(v):
    ndc = to_ndc(v)
    fid, bary, mask = rasterize(ndc, faces, H, W)
    nrm = Fn.normalize(interpolate(vertex_normals(v, faces), faces, fid, bary), dim=-1, eps=1e-8)
    return (sh_shading(nrm, SH) * mask.unsqueeze(-1)).clamp(0, 1), ndc


gt_shape = torch.zeros(1, NS, device=DEV); gt_shape[0, :4] = torch.tensor([1.5, -1.2, 0.8, 0.6], device=DEV)
gt_expr = torch.zeros(1, NE, device=DEV);  gt_expr[0, :2] = torch.tensor([1.4, -0.9], device=DEV)
gt_pose = torch.zeros(1, flame.n_joints * 3, device=DEV); gt_pose[0, 6] = 0.22
with torch.no_grad():
    v_gt = geo(gt_shape, gt_expr, gt_pose)
    img_gt, ndc_gt = render(v_gt)
    lmk_gt = ndc_gt[:, LMK, :2]


def fit(mode, iters=400, lr=0.02):
    shape = torch.zeros(1, NS, device=DEV, requires_grad=True)
    expr = torch.zeros(1, NE, device=DEV, requires_grad=True)
    pose = torch.zeros(1, flame.n_joints * 3, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([shape, expr, pose], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    first = None
    for _ in range(iters):
        opt.zero_grad()
        v = geo(shape, expr, pose)
        if mode == "geometric":
            loss = ((v - v_gt) ** 2).sum(-1).mean() * 1e3
        else:
            img, ndc = render(v)
            loss = Fn.mse_loss(img, img_gt)
            if mode == "image+landmarks":
                loss = loss + 2.0 * Fn.mse_loss(ndc[:, LMK, :2], lmk_gt)
        loss.backward(); opt.step(); sched.step()
        first = first if first is not None else loss.item()
    with torch.no_grad():
        v_fit = geo(shape, expr, pose)
        err = (v_gt - v_fit).norm(dim=-1).mean().item() * 1000
        ds = (shape - gt_shape).abs().max().item()
        dj = abs(pose[0, 6].item() - gt_pose[0, 6].item())
    return dict(loss0=first, loss1=loss.item(), err=err, dshape=ds, djaw=dj,
                v=v_fit.detach())


print(f"=== inverting the chain: {NS} shape + {NE} expr + jaw, {H}x{W}, {DEV} ===\n")
res = {}
for mode in ("geometric", "image only", "image+landmarks"):
    r = fit(mode)
    res[mode] = r
    print(f"  {mode:<16s} loss {r['loss0']:.2e} -> {r['loss1']:.2e} "
          f"({r['loss0']/r['loss1']:6.0f}x)   vertex err {r['err']:7.3f} mm   "
          f"max|dshape| {r['dshape']:.3f}   |djaw| {r['djaw']:.4f}")

print()
check = lambda n, ok, d="": (print(f"  [{'PASS' if ok else 'FAIL'}] {n}{'  — ' + d if d else ''}"),
                             None if ok else FAILURES.append(n))
# Threshold rationale: published NoW median errors sit around 1.1 mm, so the
# solver's own residual has to be small *relative to what will be measured*.
# 0.5 mm keeps it an order of magnitude below the signal; demanding ~0 would
# just be testing how long Adam was allowed to anneal.
check("geometric fit converges (chain + autograd are correct)",
      res["geometric"]["err"] < 0.5, f"{res['geometric']['err']:.4f} mm, well under the ~1.1 mm NoW scale")
# Stated relatively, not as an absolute bound. This quantity measures ~19.8 mm
# with roughly +/-0.3 run-to-run spread (GPU scatter is not bitwise
# deterministic), so an upper bound at 20.0 mm failed about a third of the time
# -- a flaky test that trains everyone to ignore a red suite. The claim being
# made is that shading alone is far worse than shading plus landmarks.
check("image-only fit is much worse than image+landmarks (expected ambiguity)",
      res["image only"]["err"] > 3 * res["image+landmarks"]["err"],
      f"{res['image only']['err']:.2f} mm vs {res['image+landmarks']['err']:.2f} mm "
      f"({res['image only']['err'] / res['image+landmarks']['err']:.1f}x)")
check("landmarks materially improve the photometric fit",
      res["image+landmarks"]["err"] < 0.6 * res["image only"]["err"],
      f"{res['image+landmarks']['err']:.2f} mm vs {res['image only']['err']:.2f} mm")

from PIL import Image
with torch.no_grad():
    tiles = [img_gt[0]] + [render(res[m]["v"])[0][0] for m in
                           ("geometric", "image only", "image+landmarks")]
Image.fromarray((torch.cat(tiles, 1).cpu().numpy() * 255).astype(np.uint8)).save(OUT / "fit_synthetic.png")
print("\n  wrote fit_synthetic.png  (target | geometric | image-only | image+landmarks)")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED" if not FAILURES else f"FAILURES: {FAILURES}")
sys.exit(1 if FAILURES else 0)
