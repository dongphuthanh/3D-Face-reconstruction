"""Diagnostic: is the encoder underfitting, or are the losses the floor?

train_overfit.py stalls around 9 mm. Two very different explanations:
  (a) the loss terms cannot determine the parameters any better than that, or
  (b) they can, and the randomly-initialised ResNet just has not got there.

Optimising the parameters DIRECTLY under identical losses, data and resolution
separates them. Direct optimisation has no representational limit, so whatever
error it reaches is the floor these losses impose. If the encoder is far above
that floor, the problem is optimisation, not the objective.
"""

import sys, pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from face3d.geometry.flame_torch import FlameTorch
from face3d import assets
from face3d.geometry.params import FlameParams
from face3d.render.pipeline import FaceRenderer, project
from face3d.learn.losses import photometric_loss, landmark_loss, regularization

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, SIZE, NS, NE = 8, 224, 100, 50

# Reproduce train_overfit.py's batch exactly: same seeds, same construction.
torch.manual_seed(0)
MODEL = assets.model_path_or_skip()
if assets.is_fixture(MODEL):
    # Thresholds below are in millimetres against a real head; the
    # synthetic fixture has no meaningful scale to compare against.
    print("SKIP — needs the real FLAME model, not the CI fixture")
    sys.exit(0)
flame = FlameTorch(MODEL).to(DEV)
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
    pose[:, 0:3] = torch.randn(n, 3, generator=gen, device=DEV) * 0.08
    pose[:, 6] = torch.rand(n, generator=gen, device=DEV) * 0.25
    cam = torch.stack([5.6 + torch.randn(n, generator=gen, device=DEV) * 0.2,
                       torch.randn(n, generator=gen, device=DEV) * 0.05,
                       0.16 + torch.randn(n, generator=gen, device=DEV) * 0.05], 1)
    light = z(n, 9, 3); light[:, 0] = 0.7; light[:, 2] = 0.4; light[:, 3] = 0.3
    light += torch.randn(n, 9, 3, generator=gen, device=DEV) * 0.05
    return FlameParams(shape=shape, expr=expr, pose=pose, cam=cam, light=light)


gen = torch.Generator(device=DEV).manual_seed(7)
gt = random_params(B, gen)
with torch.no_grad():
    o = renderer(gt)
    target_img, v_gt = o["image"], o["verts"]
    target_lmk = renderer.landmarks(v_gt, gt.cam)

# Free variables initialised at the mean face, exactly where the encoder starts.
shape = torch.zeros(B, NS, device=DEV, requires_grad=True)
expr = torch.zeros(B, NE, device=DEV, requires_grad=True)
pose = torch.zeros(B, 15, device=DEV, requires_grad=True)
cam = torch.tensor([[5.6, 0.0, 0.16]], device=DEV).repeat(B, 1).requires_grad_(True)
light = torch.zeros(B, 9, 3, device=DEV); light[:, 0] = 0.7
light = light.requires_grad_(True)

opt = torch.optim.Adam([shape, expr, pose, cam, light], lr=0.02)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1500)

print(f"=== direct parameter optimisation, identical losses ({B} imgs, {SIZE}px) ===")
for it in range(1501):
    opt.zero_grad()
    p = FlameParams(shape=shape, expr=expr, pose=pose, cam=cam, light=light)
    verts, _ = renderer.geometry(p)
    img, mask = renderer.render(verts, p)
    lmk = project(verts, cam)[:, LMK, :2]
    loss = (photometric_loss(img, target_img, mask | (target_img.sum(-1) > 0))
            + 5.0 * landmark_loss(lmk, target_lmk) + regularization(p))
    loss.backward(); opt.step(); sched.step()
    if it % 250 == 0:
        with torch.no_grad():
            e = (verts - v_gt).norm(dim=-1).mean().item() * 1000
            j = (pose[:, 6] - gt.pose[:, 6]).abs().mean().item()
        print(f"  it {it:4d}  loss {loss.item():.4f}   vertex err {e:6.2f} mm   jaw err {j:.4f} rad")

with torch.no_grad():
    verts, _ = renderer.geometry(FlameParams(shape=shape, expr=expr, pose=pose, cam=cam, light=light))
    floor = (verts - v_gt).norm(dim=-1).mean().item() * 1000
    jaw = (pose[:, 6] - gt.pose[:, 6]).abs().mean().item()

print(f"\n  loss floor for these terms: {floor:.2f} mm  (jaw {jaw:.4f} rad)")
print(f"  encoder reached 9.24 mm (jaw 0.5194 rad)")
print(f"\n  => {'ENCODER IS UNDERFITTING' if floor < 0.6 * 9.24 else 'LOSSES ARE THE BINDING CONSTRAINT'}")
