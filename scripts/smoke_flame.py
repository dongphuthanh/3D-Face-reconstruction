"""FLAME smoke test — run before DECA exists.

Answers exactly one question: does the head model load and deform correctly on
its own? Every failure here would otherwise surface later as "DECA is broken."
"""

import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d.flame_np import FlameModel, save_obj

OUT = pathlib.Path(__file__).resolve().parents[1] / "out"
OUT.mkdir(exist_ok=True)
JAW = 2          # FLAME joints: 0 global, 1 neck, 2 jaw, 3 L eye, 4 R eye
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def region_split(v):
    """Split vertices into lower-face and upper-face by height, for locality tests."""
    y = v[:, 1]
    lo, hi = np.percentile(y, 15), np.percentile(y, 85)
    return y < lo, y > hi


def run(tag, path):
    print(f"\n=== {tag} :: {pathlib.Path(path).name} ===")
    m = FlameModel(path)
    print(f"  {m}")
    if m.meta:
        print(f"  expression metadata: {m.meta}")

    # 1. Identity: zero shape + zero pose must reproduce the template exactly.
    #    This is the real test of the LBS/rest-pose maths, not a formality.
    v0, J = m(None, None, None)
    err = np.abs(v0 - m.v_template).max()
    check("zero params reproduce v_template", err < 1e-9, f"max abs err {err:.2e} m")
    check("vertex count 5023", m.n_verts == 5023, str(m.n_verts))
    check("face count 9976", len(m.faces) == 9976, str(len(m.faces)))
    check("faces index in range", m.faces.min() == 0 and m.faces.max() == m.n_verts - 1)
    check("skinning weights sum to 1", np.allclose(m.weights.sum(1), 1.0))

    bbox = v0.max(0) - v0.min(0)
    check("head bbox is human-scale (metres)",
          0.12 < bbox[1] < 0.35, f"W×H×D = {bbox[0]:.3f} × {bbox[1]:.3f} × {bbox[2]:.3f}")
    save_obj(OUT / f"{tag}_neutral.obj", v0, m.faces)

    # 2. Shape coefficient must actually move the face, and symmetrically-ish.
    s = np.zeros(m.n_shape); s[0] = 2.0
    v_s, _ = m(s, None, None)
    d = np.linalg.norm(v_s - v0, axis=1)
    check("shape[0]=+2 deforms the mesh", d.max() > 1e-3,
          f"max {d.max()*1000:.1f} mm, mean {d.mean()*1000:.1f} mm")
    save_obj(OUT / f"{tag}_shape0_p2.obj", v_s, m.faces)

    s[0] = -2.0
    v_sn, _ = m(s, None, None)
    check("shape[0]=-2 moves the opposite way",
          np.dot((v_s - v0).ravel(), (v_sn - v0).ravel()) < 0)

    # 3. Expression basis must be live and distinct from the shape basis.
    e = np.zeros(m.n_expr); e[0] = 2.0
    v_e, _ = m(None, e, None)
    de = np.linalg.norm(v_e - v0, axis=1)
    check("expr[0]=+2 deforms the mesh", de.max() > 1e-3,
          f"max {de.max()*1000:.1f} mm")
    cos = np.dot((v_e - v0).ravel(), (v_s - v0).ravel()) / (
        np.linalg.norm(v_e - v0) * np.linalg.norm(v_s - v0))
    check("expression is not a shape direction", abs(cos) < 0.9, f"cos = {cos:+.3f}")
    save_obj(OUT / f"{tag}_expr0_p2.obj", v_e, m.faces)

    # 4. Jaw rotation must be spatially local — the chin moves, the forehead does not.
    #    This is what actually proves LBS + the joint hierarchy are wired correctly.
    p = np.zeros(m.n_joints * 3); p[JAW * 3] = 0.3    # ~17 deg open
    v_j, _ = m(None, None, p)
    dj = np.linalg.norm(v_j - v0, axis=1)
    lower, upper = region_split(v0)
    check("jaw pose opens the mouth", dj.max() > 5e-3, f"max {dj.max()*1000:.1f} mm")
    check("jaw motion is local to the lower face",
          dj[lower].mean() > 5 * dj[upper].mean(),
          f"lower {dj[lower].mean()*1000:.2f} mm vs upper {dj[upper].mean()*1000:.3f} mm")
    save_obj(OUT / f"{tag}_jaw_open.obj", v_j, m.faces)

    return m, v0, v_s, v_e, v_j


def render(tag, m, frames):
    """Orthographic front view, flat-shaded, painter's algorithm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    fig, axes = plt.subplots(1, len(frames), figsize=(3.0 * len(frames), 3.6))
    for ax, (label, v) in zip(np.atleast_1d(axes), frames):
        tri = v[m.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        shade = np.clip(n @ np.array([0.3, 0.35, 0.89]), 0.05, 1.0)
        order = np.argsort(tri[:, :, 2].mean(1))
        ax.add_collection(PolyCollection(
            tri[order][:, :, :2], facecolors=plt.cm.bone(0.25 + 0.75 * shade[order]),
            edgecolors="none"))
        c, r = v.mean(0), np.abs(v - v.mean(0)).max() * 1.05
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(label, fontsize=9)
    fig.suptitle(tag, fontsize=10)
    fig.tight_layout()
    p = OUT / f"{tag}_preview.png"
    fig.savefig(p, dpi=110, facecolor="white"); plt.close(fig)
    print(f"  wrote {p.name}")


if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parents[1]
    targets = [("flame2020", root / "FLAME2020" / "generic_model.pkl"),
               ("flame2023open", root / "FLAME2023Open" / "flame2023_Open.pkl")]
    for tag, path in targets:
        if not path.exists():
            print(f"\n=== {tag} :: SKIPPED, not found at {path}")
            FAILURES.append(f"{tag} missing")
            continue
        m, v0, v_s, v_e, v_j = run(tag, path)
        render(tag, m, [("neutral", v0), ("shape[0]=+2", v_s),
                        ("expr[0]=+2", v_e), ("jaw open", v_j)])

    print("\n" + "=" * 52)
    print("ALL CHECKS PASSED" if not FAILURES else f"FAILURES: {FAILURES}")
    sys.exit(1 if FAILURES else 0)
