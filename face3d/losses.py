"""Self-supervised loss terms — story C1.

Four terms, per the plan: landmark, photometric, identity, regularisation.
Each is a plain function so the ablation table (C4) is a matter of switching
weights, not editing the training loop.
"""

import torch
import torch.nn.functional as F


def landmark_loss(pred, target, weights=None):
    """pred/target (B,L,2) in NDC. L1 is standard here — it tolerates the
    outliers a 2D detector produces on profile and occluded faces."""
    d = (pred - target).abs().sum(-1)
    if weights is not None:
        d = d * weights
    return d.mean()


def photometric_loss(rendered, target, mask):
    """Masked L1 between render and photo. rendered/target (B,H,W,3), mask (B,H,W).

    The mask is what keeps hair, glasses and hands out of the loss (story C2);
    without it the encoder distorts geometry to explain pixels that are not face.
    Normalising by mask area stops large faces from dominating small ones.
    """
    m = mask.unsqueeze(-1).float()
    return ((rendered - target).abs() * m).sum() / (m.sum() * 3 + 1e-8)


def identity_loss(embed_fn, rendered, target):
    """Cosine distance between face-recognition embeddings of render and photo.

    embed_fn is an ArcFace-style network, frozen and training-time only — it is
    never shipped, so its licence does not propagate to the exported weights.
    """
    a = F.normalize(embed_fn(rendered), dim=-1)
    b = F.normalize(embed_fn(target), dim=-1)
    return (1 - (a * b).sum(-1)).mean()


def shape_consistency_loss(shape, group_id):
    """Images of the same person must yield the same identity coefficients.

    This is the term that requires an identity-labelled corpus; FFHQ cannot
    supply it. Without it the encoder is free to absorb expression and lighting
    into identity, which is the failure mode the whole method exists to avoid.
    """
    loss, n = shape.new_zeros(()), 0
    for g in torch.unique(group_id):
        s = shape[group_id == g]
        if s.shape[0] < 2:
            continue
        loss = loss + ((s - s.mean(0, keepdim=True)) ** 2).sum(-1).mean()
        n += 1
    return loss / max(n, 1)


def regularization(params, w_shape=1e-4, w_expr=1e-4, w_pose=1e-4):
    """Keep coefficients near the basis mean. FLAME's bases are PCA over scans,
    so an L2 penalty is a proper Gaussian prior, not an arbitrary shrinkage."""
    return (w_shape * (params.shape ** 2).sum(-1).mean()
            + w_expr * (params.expr ** 2).sum(-1).mean()
            + w_pose * (params.pose ** 2).sum(-1).mean())
