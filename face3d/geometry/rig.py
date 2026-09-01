"""FLAME -> riggable asset: armature and named morph targets (B4, B5).

FLAME already has everything an engine needs, in a form no engine understands.
This translates it:

  armature       FLAME's five joints (global, neck, jaw, and both eyes) with
                 their rest positions from J_regressor, and the linear blend
                 skinning weights it already ships.
  morph targets  The expression basis, as per-vertex position deltas from the
                 subject's own neutral mesh -- not from FLAME's mean face. A
                 blendshape has to deform *this* head, so the deltas are
                 evaluated at this subject's shape.

Naming matters for the consumer: glTF morph targets are addressed by index, and
`extras.targetNames` is the only place a human-readable name can live. Without
it a downstream artist sees 50 anonymous sliders.
"""

import numpy as np
import torch

JOINT_NAMES = ["root", "neck", "jaw", "eye_left", "eye_right"]


def rest_joints(flame, shape=None):
    """(J,3) joint positions for a subject, from FLAME's own joint regressor."""
    with torch.no_grad():
        n_shape = flame.n_shape
        s = torch.zeros(1, n_shape, device=flame.v_template.device)
        if shape is not None:
            k = min(shape.shape[-1], n_shape)
            s[0, :k] = torch.as_tensor(shape).reshape(-1)[:k]
        v_shaped = flame.v_template + (
            torch.cat([s, torch.zeros(1, flame.n_expr, device=s.device)], 1)
            @ flame.shapedirs.T).view(1, flame.n_verts, 3)
        return torch.einsum("jv,bvc->bjc", flame.J_regressor, v_shaped)[0]


def expression_targets(flame, shape=None, n_targets=None, amplitude=1.0):
    """(M,V,3) morph deltas plus names.

    Deltas are relative to the subject's neutral mesh, so applying weight 1.0 to
    target k reproduces expression coefficient k at `amplitude`. Evaluating them
    on the mean face instead would make every exported head animate like the
    average person.
    """
    dev = flame.v_template.device
    n_expr = flame.n_expr if n_targets is None else min(n_targets, flame.n_expr)

    with torch.no_grad():
        s = torch.zeros(1, flame.n_shape, device=dev)
        if shape is not None:
            k = min(shape.shape[-1], flame.n_shape)
            s[0, :k] = torch.as_tensor(shape, device=dev).reshape(-1)[:k]
        neutral, _ = flame(s, torch.zeros(1, flame.n_expr, device=dev),
                           torch.zeros(1, flame.n_joints * 3, device=dev))

        deltas = []
        for i in range(n_expr):
            e = torch.zeros(1, flame.n_expr, device=dev)
            e[0, i] = amplitude
            v, _ = flame(s, e, torch.zeros(1, flame.n_joints * 3, device=dev))
            deltas.append((v - neutral)[0].cpu().numpy())

    names = [f"expr_{i:02d}" for i in range(n_expr)]
    return np.stack(deltas).astype(np.float32), names, neutral[0].cpu().numpy()


def jaw_target(flame, shape=None, angle=0.35):
    """A jaw-open morph, for consumers that drive blendshapes but not skeletons.

    Jaw is a joint rotation in FLAME, so it is not part of the expression basis;
    exposing it as a morph target as well means a mouth can be opened without
    touching the armature.
    """
    dev = flame.v_template.device
    with torch.no_grad():
        s = torch.zeros(1, flame.n_shape, device=dev)
        if shape is not None:
            k = min(shape.shape[-1], flame.n_shape)
            s[0, :k] = torch.as_tensor(shape, device=dev).reshape(-1)[:k]
        z = torch.zeros(1, flame.n_expr, device=dev)
        pose = torch.zeros(1, flame.n_joints * 3, device=dev)
        neutral, _ = flame(s, z, pose)
        pose[0, 6] = angle                      # joint 2 (jaw), x rotation
        opened, _ = flame(s, z, pose)
    return (opened - neutral)[0].cpu().numpy().astype(np.float32)
