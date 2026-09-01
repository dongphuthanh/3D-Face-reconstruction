"""Paired-view augmentation for the shape-consistency loss.

The measured failure of the first training run was that shape coefficients moved
with pose, lighting and crop framing -- within-subject spread 3.719 against
between-subject 1.531. Shape-consistency punishes exactly that, but the textbook
form needs several photographs of the same person, which FFHQ cannot supply.

Two augmented views of one image give the same constraint against nuisance
factors, with no identity labels and no new data. It is strictly weaker than
true multi-image consistency -- it cannot cover genuine expression change,
different cameras or ageing -- but it targets the axes that were measured to
fail.

Geometric augmentation has to move the landmarks too. Applying an affine to the
pixels and leaving the targets alone would train the encoder against silently
wrong 2D positions, which looks like a plain accuracy problem rather than a bug.

Coordinate care: landmark NDC here is y-up (built as 1 - 2v at ingest), while
grid_sample's sampling grid is y-down. The two differ by a reflection F, and
conjugating a rotation by it negates the angle: F R(t) F = R(-t). Composed with
the inversion needed for grid_sample, the two negations cancel -- see
_inverse_grid_matrix. Getting this wrong breaks rotation only, leaving
translation and scale correct, so test rotation specifically.
"""

import torch
import torch.nn.functional as F


def sample_params(n, device, max_rot=0.26, scale=(0.9, 1.1), max_shift=0.08):
    """Random affine parameters. max_rot is radians (~15 degrees)."""
    r = lambda lo, hi: torch.empty(n, device=device).uniform_(lo, hi)
    return {
        "theta": r(-max_rot, max_rot),
        "scale": r(*scale),
        "tx": r(-max_shift, max_shift),
        "ty": r(-max_shift, max_shift),
    }


def _forward_matrix(p):
    """(B,2,3) mapping landmark NDC -> augmented landmark NDC (y-up)."""
    c, s, sc = torch.cos(p["theta"]), torch.sin(p["theta"]), p["scale"]
    z = torch.zeros_like(c)
    m = torch.stack([sc * c, -sc * s, p["tx"],
                     sc * s, sc * c, p["ty"]], dim=-1)
    return m.reshape(-1, 2, 3)


def _inverse_grid_matrix(p):
    """(B,2,3) for affine_grid: output pixel -> input sample point, y-down.

    Derivation, with F = diag(1,-1) the y-up/y-down reflection and forward
    p' = sc*R(t)*p + t:

        q_in = F * (1/sc) * R(-t) * (F*q_out - t)
             = (1/sc) * [F R(-t) F] * q_out  -  (1/sc) * F R(-t) * t

    and F R(-t) F = R(+t). So the linear part is R(+t)/sc, NOT R(-t)/sc --
    the reflection turns the inverse rotation back into a forward one. Getting
    this wrong leaves translation and scale correct and only rotation broken,
    which is exactly what the detector comparison reported.
    """
    inv_sc = 1.0 / p["scale"]
    c, s = torch.cos(p["theta"]), torch.sin(p["theta"])
    a11, a12 = inv_sc * c, -inv_sc * s
    a21, a22 = inv_sc * s, inv_sc * c
    # b = -(1/sc) * F R(-t) * t, where F R(-t) = [[c, s], [s, -c]]
    tx, ty = p["tx"], p["ty"]
    b1 = -inv_sc * (c * tx + s * ty)
    b2 = -inv_sc * (s * tx - c * ty)
    return torch.stack([a11, a12, b1, a21, a22, b2], dim=-1).reshape(-1, 2, 3)


def affine_view(img, lmk, p):
    """img (B,3,H,W), lmk (B,L,2) y-up NDC -> augmented image and landmarks."""
    # padding_mode="border", not "zeros". Zero padding fills the corners of a
    # rotated view with exact black, which is a large shift in low-level image
    # statistics. BatchNorm's running estimates then track the augmented
    # distribution and misnormalise clean images at eval time -- measured as a
    # predicted camera scale of 9.2 against 6.9, i.e. the face rendered
    # off-frame, purely from switching train() to eval() on identical weights.
    grid = F.affine_grid(_inverse_grid_matrix(p), img.shape, align_corners=False)
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="border",
                        align_corners=False)
    M = _forward_matrix(p)
    ones = torch.ones(*lmk.shape[:2], 1, device=lmk.device, dtype=lmk.dtype)
    homo = torch.cat([lmk, ones], dim=-1)
    return out, torch.einsum("bij,blj->bli", M, homo)


def photometric_jitter(img, brightness=0.25, contrast=0.25, gamma=(0.8, 1.25),
                       noise=0.02):
    """Colour-space only: leaves geometry, so landmarks are untouched."""
    B = img.shape[0]
    d = img.device
    r = lambda lo, hi: torch.empty(B, 1, 1, 1, device=d).uniform_(lo, hi)
    x = img * r(1 - brightness, 1 + brightness)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * r(1 - contrast, 1 + contrast) + mean
    x = x.clamp(1e-4, 1.0) ** r(*gamma)
    if noise:
        x = x + torch.randn_like(x) * noise
    return x.clamp(0, 1)


def scale_jitter(img, lmk, lo, hi):
    """DECA's crop-scale augmentation, applied as a zoom on an already-made crop.

    They re-crop the source photograph at a bounding-box multiplier drawn from
    [1.4, 1.8]. Our crops are baked at ingest with a fixed margin of 1.6, so the
    equivalent is a zoom of 1.6/hi to 1.6/lo about the crop centre -- for their
    range, roughly 0.89 to 1.14.

    Scale only. DECA sets trans_scale to 0 and never rotates, and in-plane
    rotation is what put black corners into the training distribution here
    before and shifted BatchNorm's running statistics off the eval distribution.
    There is no reason to reintroduce it for an effect they did not use.

    Every image is jittered independently, including the K views of one
    identity, so the shape swap has to survive a framing change as well as a
    pose change.
    """
    n = img.shape[0]
    z = torch.zeros(n, device=img.device)
    p = {"theta": z, "tx": z, "ty": z,
         "scale": torch.empty(n, device=img.device).uniform_(lo, hi)}
    return affine_view(img, lmk, p)


def two_views(img, lmk, weak_strong=True):
    """Return (2B,3,H,W) and (2B,L,2): view A stacked above view B.

    Weak/strong pairing rather than two equally-augmented views. View A gets
    only mild colour jitter and no geometric change; view B gets the full
    affine plus jitter. Two reasons:

      * Every training step then still shows BatchNorm a geometrically clean
        image, so its running statistics stay usable at eval time. Two strongly
        augmented views drift the statistics away from the inference
        distribution entirely.
      * "Your answer on the clean image and on the perturbed one must agree" is
        the stronger and more standard constraint -- it anchors consistency to
        the distribution the model will actually be asked about, instead of
        letting both views drift together.

    The halves index the same source images, so the consistency loss is simply
    `first_half - second_half`.
    """
    if weak_strong:
        a_img = photometric_jitter(img, brightness=0.1, contrast=0.1,
                                   gamma=(0.95, 1.05), noise=0.0)
        a_lmk = lmk
    else:
        a_img, a_lmk = affine_view(img, lmk, sample_params(img.shape[0], img.device))
        a_img = photometric_jitter(a_img)

    b_img, b_lmk = affine_view(img, lmk, sample_params(img.shape[0], img.device))
    b_img = photometric_jitter(b_img)
    return torch.cat([a_img, b_img], 0), torch.cat([a_lmk, b_lmk], 0)


def consistency_loss(shape):
    """shape (2B,N) from two_views -> mean squared disagreement between halves.

    DEPRECATED for training. This is the collapse-prone formulation: it is
    minimised perfectly by predicting a constant, and measured doing exactly
    that -- within-subject shape spread fell 3x but between-subject fell 4.4x,
    driving the identity ratio from 0.42 down to 0.29. Kept because the tests
    and the earlier ablation reference it. Use swap_shape() instead.
    """
    a, b = shape.chunk(2, dim=0)
    return (a - b).pow(2).sum(-1).mean()


def swap_shape(params, k=2, blocked=False):
    """Exchange shape between images of the same identity, everything else kept.

    This is DECA's shape-consistency mechanism. Two layouts are supported:

      halves   (B*2, ...) as `two_views` emits, first half paired with second.
      blocked  (B*K, ...) as `IdentityPairs` emits, identity g occupying rows
               g*K..g*K+K-1. Shape is permuted WITHIN each block, which is what
               DECA does -- with K=4 a single image's shape must explain three
               other views of that face, not just one.

    Why swapping rather than penalising `(shape_A - shape_B)^2`: the distance
    penalty is a separate term from reconstruction, so shape can collapse to a
    constant (satisfying it perfectly) while pose, expression and camera
    compensate in the reconstruction. DECA's own comment on this is blunt --
    the L2 form "encourage s0, s1 is close in l2 space, but not really ensure
    shape will be close". We measured that collapse: ratio 0.42 -> 0.29.

    Swapping couples the two into one render. Everything except shape stays
    specific to its own image, so another image's shape is the only free
    variable left to explain this one's pixels and landmarks.
    """
    import torch as _t
    from dataclasses import replace

    s = params.shape
    if not blocked:
        a, b = s.chunk(2, dim=0)
        return replace(params, shape=_t.cat([b, a], dim=0))

    n = s.shape[0]
    assert n % k == 0, f"batch {n} is not a multiple of k={k}"
    idx = _t.arange(n, device=s.device).view(-1, k)
    # A derangement per block where possible, so no image keeps its own shape.
    perm = _t.stack([row.roll(1) for row in idx]).reshape(-1)
    return replace(params, shape=s[perm])
