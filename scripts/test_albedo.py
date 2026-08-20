"""Verify the FLAME texture space renders and stays differentiable."""

import sys, pathlib
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d.flame_torch import FlameTorch
from face3d import assets
from face3d.albedo import FlameTexture, CACHE_DIR
from face3d.params import FlameParams
from face3d.pipeline import FaceRenderer

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "out"; OUT.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAILURES = []

cache = CACHE_DIR / "flame_texture_256_50.npz"
if not cache.exists():
    print("SKIP — no texture cache. Run: python -c \"from face3d.albedo import "
          "build_cache; build_cache('TextureSpace/FLAME_texture.npz')\"")
    sys.exit(0)

MODEL = assets.model_path_or_skip()
if assets.is_fixture(MODEL):
    print("SKIP — the texture space is defined against real FLAME topology")
    sys.exit(0)


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


flame = FlameTorch(MODEL).to(DEV)
tex = FlameTexture(cache, device=DEV)
r = FaceRenderer(flame, image_size=224, texture=tex)

print(f"=== FLAME texture space ({tex.resolution}px, {tex.n_components} components) ===")
check("uv faces match mesh faces", tex.ft.shape[0] == flame.n_faces,
      f"{tex.ft.shape[0]} vs {flame.n_faces}")
check("uv verts exceed mesh verts (seams duplicate)", tex.vt.shape[0] > flame.n_verts,
      f"{tex.vt.shape[0]} uv verts vs {flame.n_verts} mesh verts")

B = 4
z = lambda *s: torch.zeros(*s, device=DEV)
cam = torch.tensor([[5.6, 0.0, 0.16]], device=DEV).repeat(B, 1)
light = z(B, 9, 3); light[:, 0] = 0.9; light[:, 2] = 0.35; light[:, 3] = 0.25
alb = z(B, 50)
torch.manual_seed(0)
alb[1:] = torch.randn(B - 1, 50, device=DEV) * 1.5      # vary identity
shape = z(B, flame.n_shape)
shape[:, :10] = torch.randn(B, 10, device=DEV) * 1.0

p = FlameParams(shape=shape, expr=z(B, flame.n_expr), pose=z(B, 15),
                cam=cam, light=light, albedo=alb)
out = r(p)
img = out["image"]

skin = img[out["mask"]]
check("rendered pixels are skin-toned, not grey",
      float(skin[:, 0].mean()) > float(skin[:, 2].mean()) * 1.15,
      f"R {float(skin[:,0].mean()):.3f} > B {float(skin[:,2].mean()):.3f}")
check("albedo coefficients change the face",
      float((img[1] - img[0]).abs().mean()) > 0.01,
      f"mean abs diff {float((img[1]-img[0]).abs().mean()):.4f}")
check("no NaN in the render", bool(torch.isfinite(img).all()))

a = alb.clone().requires_grad_(True)
p2 = FlameParams(shape=shape, expr=z(B, flame.n_expr), pose=z(B, 15),
                 cam=cam, light=light, albedo=a)
r(p2)["image"].pow(2).sum().backward()
check("gradients reach albedo coefficients",
      bool(torch.isfinite(a.grad).all()) and float(a.grad.abs().max()) > 0,
      f"|g|max {float(a.grad.abs().max()):.3e}")

from PIL import Image
Image.fromarray((torch.cat(list(img), 1).detach().cpu().numpy() * 255).astype(np.uint8)
                ).save(OUT / "albedo_test.png")
print("  wrote albedo_test.png")

print("\n" + "=" * 56)
print("ALL CHECKS PASSED" if not FAILURES else f"FAILURES: {FAILURES}")
sys.exit(1 if FAILURES else 0)
