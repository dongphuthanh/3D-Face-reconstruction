"""Minimal NumPy FLAME: load the MPI pickle and evaluate the mesh.

Deliberately dependency-light (numpy only). The point is to isolate "is the
model loading and posing correctly" from any question about torch, CUDA, or an
encoder checkpoint. A torch port comes later; the numpy path stays as the
reference the torch version is diffed against.
"""

import pickle
import numpy as np

# FLAME's pickles are Python 2 and reference chumpy, which we do not install.
# chumpy arrays carry their numeric payload in `.r`; a stub that captures
# __setstate__ is enough to recover it.
class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"_s": state})


class _ShimUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            if module.startswith("numpy"):
                raise
            return _Stub


def _as_array(v):
    if isinstance(v, np.ndarray):
        return v
    for attr in ("r", "x", "_data"):
        inner = getattr(v, attr, None)
        if isinstance(inner, np.ndarray):
            return inner
    if hasattr(v, "toarray"):          # scipy sparse (J_regressor)
        return v.toarray()
    raise TypeError(f"cannot coerce {type(v).__name__} to ndarray")


def _rodrigues(axis_angle):
    """(J,3) axis-angle -> (J,3,3) rotation matrices."""
    theta = np.linalg.norm(axis_angle, axis=1, keepdims=True)
    # guard the theta -> 0 branch; the limit is the identity
    safe = np.where(theta < 1e-8, 1.0, theta)
    k = axis_angle / safe
    K = np.zeros((len(axis_angle), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    I = np.eye(3)[None]
    s, c = np.sin(theta)[..., None], np.cos(theta)[..., None]
    R = I + s * K + (1 - c) * (K @ K)
    return np.where((theta < 1e-8)[..., None], I, R)


class FlameModel:
    """FLAME as a pure function: (shape, expression, pose) -> vertices."""

    def __init__(self, path):
        with open(path, "rb") as f:
            d = _ShimUnpickler(f, encoding="latin1").load()

        self.v_template = _as_array(d["v_template"]).astype(np.float64)
        self.shapedirs = _as_array(d["shapedirs"]).astype(np.float64)
        self.posedirs = _as_array(d["posedirs"]).astype(np.float64)
        self.J_regressor = _as_array(d["J_regressor"]).astype(np.float64)
        self.weights = _as_array(d["weights"]).astype(np.float64)
        self.faces = _as_array(d["f"]).astype(np.int64)
        self.parents = _as_array(d["kintree_table"]).astype(np.int64)[0].copy()
        self.parents[0] = -1
        self.bs_style = d.get("bs_style")
        self.bs_type = d.get("bs_type")
        self.meta = d.get("supr_expression_metadata")

        self.n_verts = self.v_template.shape[0]
        self.n_joints = self.weights.shape[1]
        # FLAME convention: the trailing coefficients of shapedirs are the
        # expression basis. Every MPI release so far uses 100, but the count is
        # read from the file when it says so rather than assumed — hardcoding it
        # makes n_shape go negative for any model with a different width, and
        # the resulting error points at tensor shapes rather than at the cause.
        meta = d.get("supr_expression_metadata") or {}
        self.n_expr = int(d.get("n_expr", meta.get("n_expr", 100)))
        self.n_shape = self.shapedirs.shape[2] - self.n_expr
        if self.n_shape <= 0:
            raise ValueError(
                f"model has {self.shapedirs.shape[2]} blendshape directions but "
                f"n_expr={self.n_expr}, leaving {self.n_shape} for identity. "
                f"Set an 'n_expr' key in the model file.")

    def __repr__(self):
        return (f"FlameModel(verts={self.n_verts}, faces={len(self.faces)}, "
                f"joints={self.n_joints}, shape={self.n_shape}, expr={self.n_expr}, "
                f"bs_type={self.bs_type!r})")

    def __call__(self, shape=None, expr=None, pose=None):
        """shape:(n_shape,) expr:(n_expr,) pose:(n_joints*3,) axis-angle. All optional."""
        shape = np.zeros(self.n_shape) if shape is None else np.asarray(shape, float)
        expr = np.zeros(self.n_expr) if expr is None else np.asarray(expr, float)
        pose = np.zeros(self.n_joints * 3) if pose is None else np.asarray(pose, float)

        betas = np.concatenate([shape, expr])
        v_shaped = self.v_template + self.shapedirs @ betas

        R = _rodrigues(pose.reshape(self.n_joints, 3))
        # 'lrotmin': pose feature is the non-root rotations minus identity
        pose_feature = (R[1:] - np.eye(3)[None]).reshape(-1)
        v_posed = v_shaped + self.posedirs @ pose_feature

        J = self.J_regressor @ v_shaped
        return self._lbs(v_posed, J, R), J

    def _lbs(self, v_posed, J, R):
        def homo(rot, t):
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = rot, t
            return M

        G = np.zeros((self.n_joints, 4, 4))
        G[0] = homo(R[0], J[0])
        for i in range(1, self.n_joints):
            G[i] = G[self.parents[i]] @ homo(R[i], J[i] - J[self.parents[i]])

        # subtract the rest-pose transform so a zero pose is the identity map
        J_homo = np.concatenate([J, np.zeros((self.n_joints, 1))], axis=1)
        G = G - np.einsum("jab,jb->ja", G, J_homo)[:, :, None] * np.eye(4)[:, 3]

        T = np.einsum("vj,jab->vab", self.weights, G)
        v_homo = np.concatenate([v_posed, np.ones((self.n_verts, 1))], axis=1)
        return np.einsum("vab,vb->va", T, v_homo)[:, :3]


def save_obj(path, verts, faces):
    with open(path, "w") as f:
        f.write("\n".join(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts))
        f.write("\n")
        f.write("\n".join(f"f {a} {b} {c}" for a, b, c in (faces + 1)))
        f.write("\n")
