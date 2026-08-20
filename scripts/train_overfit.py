"""Story C1's acceptance criterion: the training loop overfits a single batch.

Data is synthetic â€” FLAME renders its own targets â€” because the real ingredients
(landmark embedding, detector, albedo basis, segmentation, ArcFace) are not on
disk yet. That is a feature for this test: with ground-truth parameters known
exactly, "the loss went down" can be replaced by "the parameters were recovered",
which is a far stronger statement. Swapping in real photos later changes the
data loader and nothing else in this file.
"""

import sys, os, pathlib, time
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d.flame_torch import FlameTorch
from face3d import assets
from face3d.encoder import ResNetEncoder
from face3d.params import FlameParams
from face3d.pipeline import FaceRenderer, project
from face3d.losses import photometric_loss, landmark_loss, regularization

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "out"; OUT.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, SIZE = 8, 224
# Overridable so the run can be shortened in CI: `python train_overfit.py 300 3e-4`
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 3e-4
NS, NE = 100, 50

torch.manual_seed(0)
MODEL = assets.model_path_or_skip()
if assets.is_fixture(MODEL):
    # Thresholds below are in millimetres against a real head; the
    # synthetic fixture has no meaningful scale to compare against.
    print("SKIP — needs the real FLAME model, not the CI fixture")
    sys.exit(0)
flame = FlameTorch(MODEL).to(DEV)

# Placeholder for the FLAME landmark embedding (a separate MPI download). Real
# landmarks are anatomically chosen; these are spread over the front of the face
# so the term is at least well-conditioned. Swap the indices, not the code.
with torch.no_grad():
    vt, _ = flame(batch_size=1)
    front = (vt[0, :, 2] > vt[0, :, 2].median()) & (vt[0, :, 1] > vt[0, :, 1].min() + 0.12)
    LMK = torch.where(front)[0][torch.randperm(int(front.sum()), device=DEV)[:68]]

renderer = FaceRenderer(flame, lmk_idx=LMK, image_size=SIZE)


def random_params(n, gen):
    z = lambda *s: torch.zeros(*s, device=DEV)
    shape = z(n, NS); shape[:, :20] = torch.randn(n, 20, generator=gen, device=DEV) * 1.2
    expr = z(n, NE);  expr[:, :10] = torch.randn(n, 10, generator=gen, device=DEV) * 1.0
    pose = z(n, 15)
    pose[:, 0:3] = torch.randn(n, 3, generator=gen, device=DEV) * 0.08     # global
    pose[:, 6] = torch.rand(n, generator=gen, device=DEV) * 0.25           # jaw open
    cam = torch.stack([5.6 + torch.randn(n, generator=gen, device=DEV) * 0.2,
                       torch.randn(n, generator=gen, device=DEV) * 0.05,
                       0.16 + torch.randn(n, generator=gen, device=DEV) * 0.05], 1)
    light = z(n, 9, 3); light[:, 0] = 0.7; light[:, 2] = 0.4; light[:, 3] = 0.3
    light += torch.randn(n, 9, 3, generator=gen, device=DEV) * 0.05
    return FlameParams(shape=shape, expr=expr, pose=pose, cam=cam, light=light)


gen = torch.Generator(device=DEV).manual_seed(7)
gt = random_params(B, gen)
with torch.no_grad():
    out = renderer(gt)
    target_img = out["image"]
    target_lmk = renderer.landmarks(out["verts"], gt.cam)
    v_gt = out["verts"]

PRETRAINED = os.environ.get("FACE3D_PRETRAINED", "1") == "1"
enc = ResNetEncoder(n_shape=NS, n_expr=NE, arch="resnet50", pretrained=PRETRAINED).to(DEV)
opt = torch.optim.Adam(enc.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)

img_in = target_img.permute(0, 3, 1, 2).contiguous()   # (B,3,H,W)
print(f"=== overfitting a batch of {B} at {SIZE}px on {DEV} ===")
print(f"    encoder: resnet50 ({'ImageNet-pretrained' if PRETRAINED else 'random init'}) -> {enc.n_out} params  "
      f"({sum(p.numel() for p in enc.parameters())/1e6:.1f} M weights)\n")

hist, t0, err0 = [], time.perf_counter(), None
for it in range(ITERS + 1):
    opt.zero_grad()
    pred = enc.predict(img_in)
    verts, _ = renderer.geometry(pred)
    img, mask = renderer.render(verts, pred)
    lmk = project(verts, pred.cam)[:, LMK, :2]

    l_pho = photometric_loss(img, target_img, mask | (target_img.sum(-1) > 0))
    l_lmk = landmark_loss(lmk, target_lmk)
    l_reg = regularization(pred)
    loss = l_pho + 5.0 * l_lmk + l_reg
    loss.backward(); opt.step(); sched.step()
    hist.append(loss.item())

    if it % (ITERS // 8) == 0 or it == ITERS:
        with torch.no_grad():
            v_err = (verts - v_gt).norm(dim=-1).mean().item() * 1000
        err0 = v_err if err0 is None else err0
        print(f"  it {it:3d}  loss {loss.item():.4f}  (pho {l_pho.item():.4f}  "
              f"lmk {l_lmk.item():.4f}  reg {l_reg.item():.4f})   vertex err {v_err:6.2f} mm")

dt = time.perf_counter() - t0
with torch.no_grad():
    pred = enc.predict(img_in)
    verts, _ = renderer.geometry(pred)
    v_err = (verts - v_gt).norm(dim=-1).mean().item() * 1000
    jaw_err = (pred.jaw[:, 0] - gt.jaw[:, 0]).abs().mean().item()

print(f"\n  {ITERS} iters in {dt:.0f}s  ({dt/ITERS*1000:.0f} ms/iter)")
print(f"  loss {hist[0]:.4f} -> {hist[-1]:.4f}  ({hist[0]/hist[-1]:.0f}x)")
print(f"  final vertex error {v_err:.2f} mm   jaw error {jaw_err:.4f} rad")
if DEV == "cuda":
    print(f"  peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

from PIL import Image
with torch.no_grad():
    img, _ = renderer.render(verts, pred)
    rows = [torch.cat(list(target_img[:4]), 1), torch.cat(list(img[:4]), 1)]
Image.fromarray((torch.cat(rows, 0).cpu().numpy() * 255).astype(np.uint8)).save(OUT / "overfit.png")
print("  wrote overfit.png  (top: targets, bottom: encoder output)")

# Threshold rationale: scripts/diag_overfit_floor.py optimises these same
# parameters directly under identical losses and data, reaching 4.79 mm.
# NOTE: that is a reference point, NOT a floor. A pretrained encoder reaches
# 3.73 mm — better than direct optimisation — which shows the direct run was
# itself under-converged (it was still improving when its schedule annealed),
# and that pretrained features act as a useful implicit prior. Treat 4.79 mm as
# "what a naive direct fit achieves", and re-measure it whenever the loss mix
# changes rather than trusting this constant.
REFERENCE_MM = 4.79

# Two tiers, because one threshold cannot serve both callers. CI runs a few
# hundred iterations to check the loop still learns; the real run gets a full
# budget and is judged on convergence. Applying the convergence bar to the short
# run would just encode "300 iterations is not 2000", which tests nothing.
learned = hist[0] / hist[-1] > 10 and v_err < 0.6 * err0
converged = v_err < 1.5 * REFERENCE_MM
strict = ITERS >= 2000
ok = (learned and converged) if strict else learned
print("\n" + "=" * 60)
print("PASS â€” loop overfits a single batch (C1)" if ok
      else f"FAIL â€” {hist[0]/hist[-1]:.1f}x drop, {v_err:.2f} mm")
sys.exit(0 if ok else 1)
