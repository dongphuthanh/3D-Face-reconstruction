"""Verify the torch FLAME port against the numpy reference.

The numpy version is already smoke-tested, so it is the oracle. What matters
here is not just that values match but that gradients exist, are finite, and
reach the coefficients — because that is the only property the training loop
actually depends on.
"""

import sys, pathlib, time
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d.flame_np import FlameModel
from face3d import assets
from face3d.flame_torch import FlameTorch

PKL = assets.model_path_or_skip()
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


print("=== parity: torch vs numpy reference ===")
ref = FlameModel(PKL)
m64 = FlameTorch(PKL, dtype=torch.float64)

rng = np.random.default_rng(0)
worst = 0.0
for trial in range(5):
    s = rng.normal(0, 1.5, ref.n_shape)
    e = rng.normal(0, 1.5, ref.n_expr)
    p = rng.normal(0, 0.15, ref.n_joints * 3)
    v_np, _ = ref(s, e, p)
    v_t, _ = m64(*[torch.tensor(x, dtype=torch.float64)[None] for x in (s, e, p)])
    worst = max(worst, float(torch.abs(v_t[0] - torch.tensor(v_np)).max()))
check("torch matches numpy on random params", worst < 1e-10, f"max abs diff {worst:.2e} m")

v0, _ = m64()
check("zero params reproduce v_template",
      float((v0[0] - m64.v_template).abs().max()) < 1e-12)

print("\n=== batching ===")
S = torch.tensor(rng.normal(0, 1.5, (4, ref.n_shape)), dtype=torch.float64)
P = torch.tensor(rng.normal(0, 0.15, (4, ref.n_joints * 3)), dtype=torch.float64)
vb, _ = m64(S, None, P)
singles = torch.cat([m64(S[i:i+1], None, P[i:i+1])[0] for i in range(4)])
check("batched == looped", float((vb - singles).abs().max()) < 1e-12, f"B=4")

print("\n=== frozen-by-construction ===")
n_par = sum(p.numel() for p in m64.parameters())
check("model.parameters() is empty", n_par == 0, f"{n_par} trainable tensors")
check("basis registered as buffers", len(list(m64.buffers())) == 6)

print("\n=== gradients ===")
m32 = FlameTorch(PKL)
sh = torch.zeros(1, ref.n_shape, requires_grad=True)
ex = torch.zeros(1, ref.n_expr, requires_grad=True)
po = torch.zeros(1, ref.n_joints * 3, requires_grad=True)   # the theta->0 edge case
v, _ = m32(sh, ex, po)
# a random linear functional, not .sum(): summing all vertices lets symmetric
# deformations cancel and understates the true gradient magnitude
torch.manual_seed(0)
(v * torch.randn_like(v)).sum().backward()
for name, t in (("shape", sh), ("expr", ex), ("pose", po)):
    g = t.grad
    check(f"d(verts)/d({name}) is finite at zero pose",
          bool(torch.isfinite(g).all()), f"|g|max {float(g.abs().max()):.4f}")
    check(f"d(verts)/d({name}) is non-zero at zero pose", float(g.abs().max()) > 1e-6,
          f"|g|max {float(g.abs().max()):.4e}")

# The regression that motivated the rodrigues rewrite: verify the analytic pose
# gradient at exactly zero equals the finite-difference one, rather than 0.
m64g = FlameTorch(PKL, dtype=torch.float64)
pz = torch.zeros(1, ref.n_joints * 3, dtype=torch.float64, requires_grad=True)
torch.manual_seed(1)
w = torch.randn(1, ref.n_verts, 3, dtype=torch.float64)
(m64g(None, None, pz)[0] * w).sum().backward()
analytic = pz.grad.clone()
numeric = torch.zeros_like(analytic)
h = 1e-6
for i in range(ref.n_joints * 3):
    d = torch.zeros(1, ref.n_joints * 3, dtype=torch.float64); d[0, i] = h
    fp = (m64g(None, None, d)[0] * w).sum()
    fm = (m64g(None, None, -d)[0] * w).sum()
    numeric[0, i] = (fp - fm) / (2 * h)
rel = float((analytic - numeric).abs().max() / numeric.abs().max())
check("pose gradient at zero pose matches finite differences", rel < 1e-6,
      f"rel err {rel:.2e}, |analytic|max {float(analytic.abs().max()):.3f}")

# analytic vs numerical, on a slice small enough for gradcheck
sub = FlameTorch(PKL, dtype=torch.float64)
x = torch.zeros(1, 6, dtype=torch.float64, requires_grad=True)
def f(x):
    s = torch.cat([x, torch.zeros(1, sub.n_shape - 6, dtype=torch.float64)], 1)
    pose = torch.zeros(1, sub.n_joints * 3, dtype=torch.float64)
    pose = pose + 0.1
    return sub(s, None, pose)[0][:, ::400]
check("gradcheck (analytic == numerical)", torch.autograd.gradcheck(f, (x,), eps=1e-6, atol=1e-6))

print("\n=== cuda ===")
if torch.cuda.is_available():
    g = FlameTorch(PKL).cuda()
    Sc = torch.tensor(rng.normal(0, 1.5, (8, ref.n_shape)), dtype=torch.float32).cuda()
    vg, _ = g(Sc)
    vc, _ = m32(Sc.cpu())
    check("cuda matches cpu", float((vg.cpu() - vc).abs().max()) < 2e-6,
          f"max abs diff {float((vg.cpu()-vc).abs().max()):.2e} m (fp32)")
    for B in (1, 8, 32):
        Sb = torch.randn(B, ref.n_shape, device="cuda")
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(50):
            g(Sb)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 50 * 1000
        print(f"       B={B:<3d} {dt:6.2f} ms/forward  ({B/dt*1000:8.0f} meshes/s)")
    print(f"       peak VRAM at B=32: {torch.cuda.max_memory_allocated()/1e6:.1f} MB")
else:
    check("cuda available", False)

print("\n" + "=" * 52)
print("ALL CHECKS PASSED" if not FAILURES else f"FAILURES: {FAILURES}")
sys.exit(1 if FAILURES else 0)
