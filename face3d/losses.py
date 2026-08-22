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


class IdentityLoss(torch.nn.Module):
    """Cosine distance between face-recognition features of render and photo.

    DECA's `id` term, weight 0.2. It is the only loss that supervises geometry
    with a signal explicitly about *who* the person is -- the channel every
    other term is bad at. Swapping shape moves the landmarks 1.95 px against
    pose's 17.13, and the photometric term mostly sees shading.

    Two details from their implementation are easy to miss and both matter.

    Overlay, not bare render. The rendered face is composited INTO the
    photograph over the face region. A recognition network fed a bare render --
    grey background, no hair, no neck -- produces features dominated by those
    absences rather than by facial geometry. Compositing means the only
    difference between the two inputs is the face itself.

    Albedo and light detached by the caller. Gradient then reaches shape alone
    (`id_shape_only` in their config), so the term cannot be satisfied by
    repainting the texture instead of fixing the geometry.

    The network is frozen and training-time only; it is never shipped, so its
    licence does not propagate to the exported encoder.
    """

    def __init__(self, device="cuda"):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1
        self.net = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        for q in self.net.parameters():
            q.requires_grad = False

    def features(self, img_bhwc):
        # (B,H,W,3) in [0,1] -> (B,3,160,160) in [-1,1], facenet's expected range
        x = img_bhwc.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
        return self.net(x * 2.0 - 1.0)

    def forward(self, overlay, target):
        a = self.features(overlay)
        b = self.features(target)
        return (1.0 - F.cosine_similarity(a, b, dim=1)).mean()


def make_overlay(render, target, mask):
    """Composite the render into the photograph over the face region.

    render/target (B,H,W,3), mask (B,H,W). Gradient flows through `render`
    only; the photograph is a constant.
    """
    m = mask.unsqueeze(-1).to(render.dtype)
    return render * m + target.detach() * (1 - m)


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


def regularization(params, w_shape=1e-4, w_expr=1e-4, w_pose=1e-4, w_light=1.0):
    """Keep coefficients near the basis mean. FLAME's bases are PCA over scans,
    so an L2 penalty on shape and expression is a proper Gaussian prior.

    Lighting is a different case and was missing entirely. The encoder gets 27
    unconstrained spherical-harmonic coefficients, and unconstrained lighting
    can explain away shading that should have come from geometry -- the
    photometric term gets satisfied by inventing a light rather than by getting
    the shape right. DECA regularises this at weight 1.0, the strongest in
    their config.
    """
    reg = (w_shape * (params.shape ** 2).sum(-1).mean()
           + w_expr * (params.expr ** 2).sum(-1).mean()
           + w_pose * (params.pose ** 2).sum(-1).mean())
    if w_light:
        # Deviation of each SH band from its own channel mean: penalises
        # coloured and strongly directional light without touching brightness.
        light = params.light                       # (B, 9, 3)
        reg = reg + w_light * ((light - light.mean(-1, keepdim=True)) ** 2).mean()
    return reg
