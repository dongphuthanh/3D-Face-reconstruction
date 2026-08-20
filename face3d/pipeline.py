"""Composes FLAME + camera + rasteriser into one differentiable image formation.

This is the forward model the self-supervised losses are defined against:
    FlameParams -> mesh -> projected landmarks + rendered image
Keeping it in one place means the training loop and the eval harness cannot
drift apart in how they project or shade.
"""

import torch
import torch.nn.functional as F

from .params import FlameParams
from .render import rasterize, interpolate, vertex_normals, sh_shading


def project(verts, cam):
    """Weak-perspective projection to NDC. verts (B,V,3), cam (B,3)=(s,tx,ty).

    z is negated so the amin z-buffer in `rasterize` keeps the nearest surface.
    """
    s = cam[:, :1].unsqueeze(-1)
    t = cam[:, 1:].unsqueeze(1)
    xy = verts[..., :2] * s + t
    z = -verts[..., 2:] * s
    return torch.cat([xy, z], dim=-1)


class FaceRenderer:
    """FlameParams -> (image, mask, landmarks). Holds no state beyond the model."""

    def __init__(self, flame, lmk_idx=None, image_size=224, texture=None,
                 face_keep=None):
        self.flame = flame
        self.lmk_idx = lmk_idx
        self.size = image_size
        self.texture = texture          # face3d.albedo.FlameTexture, or None
        # Boolean (F,) over triangles. When set, the returned mask covers only
        # face skin rather than the whole silhouette -- see face3d.facemask.
        self.face_keep = face_keep

    def geometry(self, params: FlameParams):
        p = params.pad_to(self.flame.n_shape, self.flame.n_expr)
        verts, joints = self.flame(p.shape, p.expr, p.pose)
        return verts, joints

    def landmarks(self, verts, cam):
        """(B,L,2) in NDC.

        Accepts either a LandmarkEmbedding (barycentric, the real MPI asset) or
        a plain tensor of vertex indices (a stand-in).
        """
        if self.lmk_idx is None:
            raise RuntimeError(
                "no landmark embedding set — download the FLAME landmark embedding "
                "from the MPI portal, or pass lmk_idx explicitly")
        if hasattr(self.lmk_idx, "positions"):
            pts = self.lmk_idx.positions(verts, self.flame.faces)
            # Project the embedded points themselves, not their host vertices.
            s, t = cam[:, :1].unsqueeze(-1), cam[:, 1:].unsqueeze(1)
            return pts[..., :2] * s + t
        return project(verts, cam)[:, self.lmk_idx, :2]

    def render(self, verts, params: FlameParams, albedo=None):
        """Returns (image (B,H,W,3), mask (B,H,W)).

        Albedo comes from `self.texture` when one is attached and the params
        carry coefficients. Without it the surface is flat grey, which makes
        every rendered face an identical mannequin — the photometric loss can
        then only see silhouette and shading, not identity.
        """
        H = W = self.size
        ndc = project(verts, params.cam)
        fid, bary, mask = rasterize(ndc, self.flame.faces, H, W)
        if self.face_keep is not None:
            # Restrict to skin. The scalp is hair in nearly every photograph and
            # the neck stub is geometry FLAME invents, so scoring either against
            # a skin albedo model is noise the encoder pays for in shape.
            mask = mask & self.face_keep[fid.clamp(min=0)]
        n = interpolate(vertex_normals(verts, self.flame.faces), self.flame.faces, fid, bary)
        n = F.normalize(n, dim=-1, eps=1e-8)
        shaded = sh_shading(n, params.light)
        if albedo is None:
            if self.texture is not None and params.albedo is not None:
                albedo = self.texture.sample(params.albedo, fid, bary)
            else:
                albedo = shaded.new_full((1, 1, 1, 3), 0.6)
        return (shaded * albedo * mask.unsqueeze(-1)).clamp(0, 1), mask

    def __call__(self, params: FlameParams, albedo=None):
        verts, _ = self.geometry(params)
        img, mask = self.render(verts, params, albedo)
        return dict(verts=verts, image=img, mask=mask)
