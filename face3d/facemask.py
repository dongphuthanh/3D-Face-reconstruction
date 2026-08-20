"""Which FLAME vertices are face skin — story C2, mesh side.

The photometric loss compares a render to a photograph over the rendered
silhouette, which on FLAME covers scalp, ears and a large neck and shoulder
stub. None of those are skin the albedo model can explain: the scalp is hair in
almost every photograph, and the neck stub is geometry FLAME invents below the
jaw. The encoder is therefore penalised for failing to reproduce hair, and it
pays for that by distorting shape -- the one channel measured to have learned
nothing.

MPI ships FLAME_masks.pkl with exactly these regions, but it is a separate
download. This derives an equivalent from what is already on disk: the 105
MediaPipe landmarks embedded on the surface mark out the face, so a vertex
belongs to the face region if it is close to that set and faces forward.

Deliberately conservative. Including background is a real cost; excluding a
little genuine cheek is not.
"""

import numpy as np
import torch


def face_region(flame, embedding, radius=0.045, front_normal=0.0):
    """Boolean (V,) mask over FLAME vertices: True for face skin.

    radius        metres from the nearest embedded landmark
    front_normal  minimum z of the vertex normal on the neutral mesh, so the
                  back of the head is excluded even where it comes close to a
                  landmark in Euclidean terms
    """
    from .render import vertex_normals

    with torch.no_grad():
        v, _ = flame(batch_size=1)
        lmk = embedding.positions(v, flame.faces)[0]          # (L,3)
        n = vertex_normals(v, flame.faces)[0]                 # (V,3)
        d = torch.cdist(v[0], lmk).min(dim=1).values          # (V,)
        return (d < radius) & (n[:, 2] > front_normal)


def face_faces(flame, vertex_mask, require_all=True):
    """Triangles to keep. require_all=True drops any triangle touching a
    non-face vertex, which keeps the boundary inside the face rather than
    letting it spill onto the neck."""
    tri = vertex_mask[flame.faces]
    return tri.all(-1) if require_all else tri.any(-1)


def render_mask(fid, face_keep):
    """Per-pixel mask from a rasterised face-id buffer. fid is -1 outside the
    mesh, so clamp before indexing and re-apply the background."""
    hit = fid >= 0
    return hit & face_keep[fid.clamp(min=0)]
