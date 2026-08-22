"""The identity loss must reach geometry and nothing else.

DECA's `id_shape_only`. If the gradient can also reach the 27 light and 50
albedo coefficients, the cheapest way to make a recognition network agree is to
repaint the face, and the term stops supervising the thing we added it for.
Detaching happens inside FaceRenderer.shade, three call-frames from the loss,
which is exactly the kind of plumbing that regresses without anyone noticing --
the run still trains, the number still falls, and the geometry does not improve.

Also pinned here: detaching must NOT change the geometry gradient. If those
differ, the split into raster_pass + shade dropped a path.
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d import assets
from face3d.albedo import CACHE_DIR, FlameTexture
from face3d.flame_torch import FlameTorch
from face3d.losses import make_overlay
from face3d.params import FlameParams
from face3d.pipeline import FaceRenderer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, SIZE = 2, 112


def gradients(renderer, leaves, detach):
    shape, expr, pose, light, albedo = leaves
    for t in leaves:
        t.grad = None
    cam = torch.tensor([[7.0, 0.0, 0.0]], device=DEV).repeat(B, 1)
    p = FlameParams(shape=shape, expr=expr, pose=pose, cam=cam,
                    light=light + 0.5, albedo=albedo)
    verts, _ = renderer.geometry(p)
    fid, bary, mask, normals = renderer.raster_pass(verts, p)
    img = renderer.shade(fid, bary, mask, normals, p, detach_appearance=detach)
    target = torch.zeros(B, SIZE, SIZE, 3, device=DEV)
    ((make_overlay(img, target, mask) - target) ** 2).mean().backward()
    return [0.0 if t.grad is None else t.grad.abs().max().item() for t in leaves]


def main():
    flame = FlameTorch(assets.model_path_or_skip()).to(DEV)
    cache = CACHE_DIR / "flame_texture_256_50.npz"
    if not cache.exists():
        print("SKIP - no texture cache; albedo gradient would be vacuously zero")
        return 0
    renderer = FaceRenderer(flame, image_size=SIZE,
                            texture=FlameTexture(cache, device=DEV))

    leaf = lambda *s: torch.zeros(*s, device=DEV, requires_grad=True)
    leaves = [leaf(B, 100), leaf(B, 50), leaf(B, 15), leaf(B, 9, 3), leaf(B, 50)]
    names = ["shape", "expr", "pose", "light", "albedo"]

    det = gradients(renderer, leaves, detach=True)
    full = gradients(renderer, leaves, detach=False)
    print(f"{'param':8s} {'detached':>12s} {'full':>12s}")
    for n, d, f in zip(names, det, full):
        print(f"{n:8s} {d:12.3e} {f:12.3e}")

    d = dict(zip(names, det))
    f = dict(zip(names, full))
    assert d["light"] == 0.0, f"light gradient leaked: {d['light']:.3e}"
    assert d["albedo"] == 0.0, f"albedo gradient leaked: {d['albedo']:.3e}"
    assert f["light"] > 0 and f["albedo"] > 0, "undetached render has no appearance path"
    for n in ("shape", "pose"):
        assert d[n] > 0, f"{n} gradient lost"
        # Loose on purpose: the claim is "detaching dropped no path back to
        # geometry", not bitwise equality. Both routes accumulate over ~10k
        # triangles in float32 and land ~1e-4 relative apart; a tolerance
        # tight enough to catch that noise fails at random, which is worse
        # than not testing it. A severed path shows up as orders of magnitude.
        assert abs(d[n] - f[n]) <= 1e-12 + 1e-3 * f[n], (
            f"{n} gradient changed with detach: {d[n]:.6e} vs {f[n]:.6e}")

    # render() must still agree with the split it now delegates to.
    p = FlameParams(shape=leaves[0].detach(), expr=leaves[1].detach(),
                    pose=leaves[2].detach(), albedo=leaves[4].detach(),
                    light=leaves[3].detach() + 0.5,
                    cam=torch.tensor([[7.0, 0.0, 0.0]], device=DEV).repeat(B, 1))
    verts, _ = renderer.geometry(p)
    a_img, a_mask = renderer.render(verts, p)
    fid, bary, mask, normals = renderer.raster_pass(verts, p)
    b_img = renderer.shade(fid, bary, mask, normals, p)
    # Same reasoning: ~2e-7 of float32 drift between two runs of identical
    # arithmetic, so atol must sit above that. The mask is integer and must
    # match exactly.
    assert torch.equal(a_mask, mask), "render() and raster_pass disagree on coverage"
    delta = (a_img - b_img).abs().max().item()
    assert delta < 1e-5, f"render() and raster_pass+shade disagree by {delta:.3e}"

    print("\nOK: identity path reaches geometry only; render() unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
