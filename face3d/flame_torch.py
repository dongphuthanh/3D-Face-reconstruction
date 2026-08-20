"""FLAME as a differentiable torch module.

Design note: every FLAME array is registered as a *buffer*, never a Parameter.
Decision 1 says FLAME is frozen; making that structural rather than a convention
means `model.parameters()` is empty and no optimizer can touch the basis by
accident. Gradients still flow *through* the module to whatever produced the
coefficients — which is the entire point.
"""

import numpy as np
import torch
import torch.nn as nn

from .flame_np import FlameModel as _NumpyFlame


def batch_rodrigues(axis_angle):
    """(B,J,3) axis-angle -> (B,J,3,3), smooth and correctly differentiable at theta=0.

    Uses R = I + A*K + B*K@K with K = skew(r) *unnormalised*, A = sin(t)/t,
    B = (1-cos t)/t^2. Naively selecting an identity branch for small angles
    (torch.where(t < eps, I, R)) returns the right value but a zero gradient,
    because the chosen branch is a constant. Zero pose is the initialisation of
    every training run, so that silently freezes all pose parameters.

    Here the linear term A*skew(r) carries the gradient through the origin
    (A -> 1), and only the scalar coefficients switch to a Taylor branch. theta
    is computed via a masked sqrt so the discarded branch never produces a NaN
    that could poison the gradient through torch.where.
    """
    r = axis_angle
    t2 = (r * r).sum(-1, keepdim=True)
    small = t2 < 1e-8
    # evaluate the trig branch at t=1 where the Taylor branch will be taken,
    # so the unused side is finite and NaN-free
    t2_safe = torch.where(small, torch.ones_like(t2), t2)
    t = torch.sqrt(t2_safe)
    # Every denominator must use t2_safe, never t2: torch.where still evaluates
    # the discarded branch, and inf * 0 = NaN would poison the gradient.
    A = torch.where(small, 1 - t2 / 6 + t2 * t2 / 120, torch.sin(t) / t)
    B = torch.where(small, 0.5 - t2 / 24 + t2 * t2 / 720, (1 - torch.cos(t)) / t2_safe)

    rx, ry, rz = r.unbind(-1)
    O = torch.zeros_like(rx)
    K = torch.stack([O, -rz, ry, rz, O, -rx, -ry, rx, O], dim=-1).reshape(*r.shape[:-1], 3, 3)
    I = torch.eye(3, dtype=r.dtype, device=r.device).expand_as(K)
    return I + A.unsqueeze(-1) * K + B.unsqueeze(-1) * (K @ K)


class FlameTorch(nn.Module):
    def __init__(self, pkl_path, dtype=torch.float32):
        super().__init__()
        m = _NumpyFlame(pkl_path)
        self.n_verts, self.n_joints = m.n_verts, m.n_joints
        self.n_shape, self.n_expr = m.n_shape, m.n_expr
        self.parents = m.parents.tolist()

        def buf(name, arr):
            self.register_buffer(name, torch.as_tensor(np.ascontiguousarray(arr), dtype=dtype))

        buf("v_template", m.v_template)
        # flatten the 3D basis to (V*3, n_coeff) so blending is one matmul
        buf("shapedirs", m.shapedirs.reshape(-1, m.shapedirs.shape[2]))
        buf("posedirs", m.posedirs.reshape(-1, m.posedirs.shape[2]))
        buf("J_regressor", m.J_regressor)
        buf("weights", m.weights)
        self.register_buffer("faces", torch.as_tensor(m.faces, dtype=torch.long))

    @property
    def n_faces(self):
        return self.faces.shape[0]

    def forward(self, shape=None, expr=None, pose=None, batch_size=None):
        """shape:(B,n_shape) expr:(B,n_expr) pose:(B,J*3). Returns verts (B,V,3), joints (B,J,3)."""
        given = [t for t in (shape, expr, pose) if t is not None]
        B = batch_size or (given[0].shape[0] if given else 1)
        dev, dt = self.v_template.device, self.v_template.dtype
        z = lambda n: torch.zeros(B, n, device=dev, dtype=dt)
        shape = z(self.n_shape) if shape is None else shape
        expr = z(self.n_expr) if expr is None else expr
        pose = z(self.n_joints * 3) if pose is None else pose

        betas = torch.cat([shape, expr], dim=1)
        v_shaped = self.v_template + (betas @ self.shapedirs.T).view(B, self.n_verts, 3)

        R = batch_rodrigues(pose.view(B, self.n_joints, 3))
        I = torch.eye(3, device=dev, dtype=dt)
        pose_feature = (R[:, 1:] - I).reshape(B, -1)          # 'lrotmin'
        v_posed = v_shaped + (pose_feature @ self.posedirs.T).view(B, self.n_verts, 3)

        J = torch.einsum("jv,bvc->bjc", self.J_regressor, v_shaped)
        return self._lbs(v_posed, J, R), J

    def _lbs(self, v_posed, J, R):
        B, dev, dt = J.shape[0], J.device, J.dtype

        rel = J.clone()
        rel[:, 1:] = J[:, 1:] - J[:, self.parents[1:]]
        G_local = torch.zeros(B, self.n_joints, 4, 4, device=dev, dtype=dt)
        G_local[..., :3, :3] = R
        G_local[..., :3, 3] = rel
        G_local[..., 3, 3] = 1.0

        Gs = [G_local[:, 0]]
        for i in range(1, self.n_joints):
            Gs.append(Gs[self.parents[i]] @ G_local[:, i])
        G = torch.stack(Gs, dim=1)

        # Cancel the rest-pose transform so a zero pose is exactly the identity.
        J_homo = torch.cat([J, torch.zeros(B, self.n_joints, 1, device=dev, dtype=dt)], -1)
        offset = torch.einsum("bjnm,bjm->bjn", G, J_homo)
        G = G.clone()
        G[..., 3] = G[..., 3] - offset

        T = torch.einsum("vj,bjnm->bvnm", self.weights, G)
        v_homo = torch.cat([v_posed, torch.ones(B, self.n_verts, 1, device=dev, dtype=dt)], -1)
        return torch.einsum("bvnm,bvm->bvn", T, v_homo)[..., :3]
