"""Stories B1/B2: the Encoder boundary and the FlameParams contract.

B1's acceptance is "a second encoder can be added without touching callers", so
the test defines a throwaway encoder that shares no code with ResNetEncoder and
pushes it through the same pipeline. If that needs an edit anywhere else, the
abstraction has not actually been built.
"""

import sys, pathlib
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d.geometry.flame_torch import FlameTorch
from face3d import assets
from face3d.learn.encoder import Encoder, ResNetEncoder
from face3d.geometry.params import FlameParams
from face3d.render.pipeline import FaceRenderer

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def raises(fn, exc=Exception):
    try:
        fn(); return False
    except exc:
        return True


flame = FlameTorch(assets.model_path_or_skip()).to(DEV)
renderer = FaceRenderer(flame, image_size=64)
img = torch.rand(2, 3, 64, 64, device=DEV)

print("=== FlameParams contract ===")
mk = lambda **kw: FlameParams(**{**dict(
    shape=torch.zeros(2, 100), expr=torch.zeros(2, 50), pose=torch.zeros(2, 15),
    cam=torch.zeros(2, 3), light=torch.zeros(2, 9, 3)), **kw})
p = mk()
check("valid params construct", p.batch_size == 2)
check("jaw is joint 2", p.jaw.shape == (2, 3))
check("batch mismatch is rejected", raises(lambda: mk(expr=torch.zeros(3, 50)), ValueError))
check("wrong pose width is rejected", raises(lambda: mk(pose=torch.zeros(2, 6)), ValueError))
check("wrong light shape is rejected", raises(lambda: mk(light=torch.zeros(2, 3, 9)), ValueError))

padded = p.pad_to(300, 100)
check("pad_to widens to the full basis",
      padded.shape.shape == (2, 300) and padded.expr.shape == (2, 100))
check("pad_to preserves leading coefficients",
      bool(torch.equal(padded.shape[:, :100], p.shape)))
check("pad_to zero-fills the tail", float(padded.shape[:, 100:].abs().max()) == 0.0)
check("pad_to refuses to truncate", raises(lambda: p.pad_to(50, 50), ValueError))

print("\n=== ResNetEncoder ===")
# Size the encoder to whatever model is loaded. Hardcoding 100/50 works against
# real FLAME (300/100, so pad_to widens) but not against the CI fixture, where
# pad_to correctly refuses to truncate.
enc = ResNetEncoder(n_shape=min(100, flame.n_shape),
                    n_expr=min(50, flame.n_expr)).to(DEV)
out = enc.predict(img)
check("satisfies the Encoder protocol", isinstance(enc, Encoder))
check("returns FlameParams", isinstance(out, FlameParams))
check("neck and eye joints are left at rest",
      float(out.pose[:, 3:6].abs().max()) == 0.0 and float(out.pose[:, 9:].abs().max()) == 0.0)
check("initialises near the mean face", float(out.shape.abs().max()) < 0.1,
      f"max |coeff| {float(out.shape.abs().max()):.4f} before training")
check("initial camera frames the head", 5.0 < float(out.cam[0, 0]) < 6.5,
      f"scale {float(out.cam[0,0]):.2f}")
# A zero-initialised head would pass every check above and still starve the
# backbone of gradient, so this is checked explicitly rather than assumed.
out.shape.sum().backward()
n_grad = sum(1 for q in enc.trunk.parameters() if q.grad is not None and q.grad.abs().sum() > 0)
check("gradients reach the trunk at initialisation", n_grad > 0,
      f"{n_grad} trunk tensors with non-zero grad")

print("\n=== a second, unrelated encoder ===")

class ConstantEncoder(nn.Module):
    """Shares no code with ResNetEncoder — the point is that nothing downstream cares."""
    n_shape, n_expr = min(20, flame.n_shape), min(10, flame.n_expr)

    def predict(self, image):
        B = image.shape[0]
        d = image.device
        light = torch.zeros(B, 9, 3, device=d); light[:, 0] = 0.7
        cam = torch.zeros(B, 3, device=d); cam[:, 0] = 5.6; cam[:, 2] = 0.16
        return FlameParams(shape=torch.zeros(B, self.n_shape, device=d),
                           expr=torch.zeros(B, self.n_expr, device=d),
                           pose=torch.zeros(B, 15, device=d), cam=cam, light=light)

alt = ConstantEncoder()
check("satisfies the Encoder protocol", isinstance(alt, Encoder))
r1 = renderer(enc.predict(img).detach())
r2 = renderer(alt.predict(img))
check("both encoders drive the same renderer unmodified",
      r1["image"].shape == r2["image"].shape == (2, 64, 64, 3))
check("differing basis widths are absorbed by pad_to",
      r2["verts"].shape == (2, flame.n_verts, 3),
      f"encoder emitted {alt.n_shape}+{alt.n_expr}, FLAME wants {flame.n_shape}+{flame.n_expr}")

print("\n=== landmark embedding is required, not silently faked ===")
check("renderer refuses landmarks without an embedding",
      raises(lambda: renderer.landmarks(r2["verts"], alt.predict(img).cam), RuntimeError))

print("\n" + "=" * 60)
print("ALL CHECKS PASSED" if not FAILURES else f"FAILURES: {FAILURES}")
sys.exit(1 if FAILURES else 0)
