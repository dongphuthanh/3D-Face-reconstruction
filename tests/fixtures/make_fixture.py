"""Generate a synthetic model in FLAME's pickle format, for CI.

Contains no MPI data: an icosphere for topology, random orthonormal bases, and
joints placed by hand. It exercises every code path the real model does — LBS,
the joint hierarchy, blendshapes, rasterisation, gradients — at ~1/25th the
vertex count, so the whole suite runs on a CPU runner in seconds.

What it deliberately cannot check is anything about FLAME's actual geometry.
Those assertions (5023 verts, 9976 faces, human-scale bbox, jaw locality) stay
in smoke_flame.py and only run where the real model is present.
"""

import pathlib
import pickle

import numpy as np


def icosphere(subdiv=2):
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],
                  [11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],
                  [3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]])
    for _ in range(subdiv):
        mid, nf = {}, []
        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                mid[k] = len(v_list)
                v_list.append((v_list[a] + v_list[b]) / 2)
            return mid[k]
        v_list = list(v)
        for a, b, c in f:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        v, f = np.array(v_list), np.array(nf)
    return v / np.linalg.norm(v, axis=1, keepdims=True), f


def main():
    rng = np.random.default_rng(0)
    v, f = icosphere(2)
    v = v * np.array([0.09, 0.14, 0.10])          # roughly head-proportioned
    V, N_SHAPE, N_EXPR, J = len(v), 20, 10, 5

    # Smooth, low-frequency deformation directions, so blendshapes look like
    # shape changes rather than noise.
    def basis(n, scale):
        b = rng.normal(0, 1, (V, 3, n))
        b += np.roll(b, 1, axis=0) + np.roll(b, -1, axis=0)
        return b / np.abs(b).max() * scale

    # Joints down the vertical axis: root, neck, jaw, two eyes.
    joints = np.array([[0, -0.05, 0], [0, -0.02, 0], [0, 0.01, 0.02],
                       [-0.03, 0.05, 0.05], [0.03, 0.05, 0.05]])
    J_reg = np.zeros((J, V))
    for j, jp in enumerate(joints):
        w = np.exp(-((v - jp) ** 2).sum(1) / 0.002)
        J_reg[j] = w / w.sum()

    # Skinning weights from distance to each joint, normalised per vertex.
    d = np.linalg.norm(v[:, None, :] - joints[None], axis=2)
    w = np.exp(-d ** 2 / 0.01)
    w /= w.sum(1, keepdims=True)

    model = {
        "v_template": v,
        "f": f.astype(np.uint32),
        "shapedirs": basis(N_SHAPE + N_EXPR, 0.02),
        "posedirs": basis((J - 1) * 9, 0.004),
        "J_regressor": J_reg,
        "weights": w,
        "kintree_table": np.array([[-1, 0, 1, 1, 1], [0, 1, 2, 3, 4]], np.int64),
        "bs_style": "lbs",
        "bs_type": "lrotmin",
        # Declared, not inferred — see FlameModel on why the count is read
        "n_expr": N_EXPR,
    }
    out = pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "tiny_head.pkl"
    with open(p, "wb") as fh:
        pickle.dump(model, fh, protocol=2)
    print(f"wrote {p.relative_to(p.parents[2])}  "
          f"({V} verts, {len(f)} faces, {p.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
