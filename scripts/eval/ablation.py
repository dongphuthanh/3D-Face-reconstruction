"""Ablation over the loss ingredients — the shape story C4 needs.

Each row optimises FLAME parameters directly from zero against a synthetic
target with known ground truth, under identical data, seeds and iteration
budget. Only the loss ingredients change. Direct optimisation is used rather
than a trained encoder so the numbers measure the *objective*, not how well a
network happened to converge.

Reported error is mean per-vertex distance to the known ground truth, which the
image-space loss never sees.
"""

import sys, pathlib, time
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from face3d.geometry.flame_torch import FlameTorch
from face3d import assets
from face3d.render.albedo import FlameTexture, CACHE_DIR
from face3d.geometry.landmarks import LandmarkEmbedding
from face3d.geometry.params import FlameParams
from face3d.render.pipeline import FaceRenderer, project
from face3d.learn.losses import photometric_loss, landmark_loss, regularization

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, SIZE, NS, NE, NA, ITERS = 8, 224, 100, 50, 50, 1200

MODEL = assets.model_path_or_skip()
if assets.is_fixture(MODEL):
    print("SKIP — needs the real FLAME model"); sys.exit(0)
tex_cache = CACHE_DIR / "flame_texture_256_50.npz"
lmk_npz = ROOT / "mediapipe_landmark_embedding" / "mediapipe_landmark_embedding.npz"
for p, what in [(tex_cache, "texture cache"), (lmk_npz, "landmark embedding")]:
    if not p.exists():
        print(f"SKIP — missing {what}: {p}"); sys.exit(0)

torch.manual_seed(0)
flame = FlameTorch(MODEL).to(DEV)
tex = FlameTexture(tex_cache, device=DEV)
real_lmk = LandmarkEmbedding(lmk_npz, device=DEV)

# The stand-in used before the real embedding arrived: random front-facing
# vertices. Kept as an ablation row to show what the real asset is worth.
with torch.no_grad():
    vt0, _ = flame(batch_size=1)
    front = (vt0[0, :, 2] > vt0[0, :, 2].median()) & (vt0[0, :, 1] > vt0[0, :, 1].min() + 0.12)
    fake_lmk = torch.where(front)[0][torch.randperm(int(front.sum()), device=DEV)[:105]]

z = lambda *s: torch.zeros(*s, device=DEV)
g = torch.Generator(device=DEV).manual_seed(7)
gt = FlameParams(
    shape=torch.cat([torch.randn(B, 20, generator=g, device=DEV) * 1.2, z(B, NS - 20)], 1),
    expr=torch.cat([torch.randn(B, 10, generator=g, device=DEV) * 1.0, z(B, NE - 10)], 1),
    pose=z(B, 15), cam=torch.stack([
        5.6 + torch.randn(B, generator=g, device=DEV) * 0.2,
        torch.randn(B, generator=g, device=DEV) * 0.05,
        0.16 + torch.randn(B, generator=g, device=DEV) * 0.05], 1),
    light=z(B, 9, 3), albedo=torch.randn(B, NA, generator=g, device=DEV) * 1.2)
gt.pose[:, 6] = torch.rand(B, generator=g, device=DEV) * 0.25
gt.light[:, 0] = 0.9; gt.light[:, 2] = 0.35; gt.light[:, 3] = 0.25


# The target is ALWAYS rendered with texture, because a real photograph always
# has one. Rendering the target with whatever configuration is being fitted
# would let each configuration invert its own output — which measures
# self-consistency, not ability to explain a real face.
TARGET = FaceRenderer(flame, lmk_idx=real_lmk, image_size=SIZE, texture=tex)
with torch.no_grad():
    V_GT, _ = TARGET.geometry(gt)
    IMG_GT, _ = TARGET.render(V_GT, gt)


def run(name, use_texture, lmk, w_lmk=5.0):
    r = FaceRenderer(flame, lmk_idx=lmk, image_size=SIZE,
                     texture=tex if use_texture else None)
    v_gt, img_gt = V_GT, IMG_GT
    with torch.no_grad():
        lmk_gt = r.landmarks(v_gt, gt.cam) if lmk is not None else None

    p = {k: v for k, v in dict(
        shape=z(B, NS), expr=z(B, NE), pose=z(B, 15),
        cam=torch.tensor([[5.6, 0.0, 0.16]], device=DEV).repeat(B, 1),
        light=z(B, 9, 3), albedo=z(B, NA)).items()}
    p["light"][:, 0] = 0.9
    for v in p.values():
        v.requires_grad_(True)
    opt = torch.optim.Adam(list(p.values()), lr=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        opt.zero_grad()
        par = FlameParams(**p)
        verts, _ = r.geometry(par)
        img, mask = r.render(verts, par)
        loss = photometric_loss(img, img_gt, mask | (img_gt.sum(-1) > 0)) + regularization(par)
        if lmk is not None:
            loss = loss + w_lmk * landmark_loss(r.landmarks(verts, p["cam"]), lmk_gt)
        loss.backward(); opt.step(); sched.step()

    with torch.no_grad():
        verts, _ = r.geometry(FlameParams(**p))
        err = (verts - v_gt).norm(dim=-1).mean().item() * 1000
        jaw = (p["pose"][:, 6] - gt.pose[:, 6]).abs().mean().item()
    print(f"  {name:<42s} {err:7.2f} mm   jaw {jaw:.4f} rad   ({time.perf_counter()-t0:.0f}s)")
    return err


print(f"=== loss ablation: {B} images, {SIZE}px, {ITERS} iters, direct optimisation ===")
print(f"    free parameters: {NS} shape + {NE} expr + 15 pose + 3 cam + 27 light + {NA} albedo\n")
res = {}
print("    target is always textured; only the fitted model varies")

res["grey"]     = run("grey albedo,    no landmarks",   False, None)
res["grey+lmk"] = run("grey albedo,    real landmarks", False, real_lmk)
res["tex"]      = run("texture albedo, no landmarks",   True, None)
res["tex+lmk"]  = run("texture albedo, real landmarks", True, real_lmk)

print("")
print(f"  landmarks, grey model:    {res['grey']:6.2f} -> {res['grey+lmk']:5.2f} mm"
      f"  ({res['grey']/max(res['grey+lmk'],1e-9):.1f}x)")
print(f"  landmarks, texture model: {res['tex']:6.2f} -> {res['tex+lmk']:5.2f} mm"
      f"  ({res['tex']/max(res['tex+lmk'],1e-9):.1f}x)")
print(f"  albedo, no landmarks:     {res['grey']:6.2f} -> {res['tex']:5.2f} mm")
print(f"  albedo, with landmarks:   {res['grey+lmk']:6.2f} -> {res['tex+lmk']:5.2f} mm")
