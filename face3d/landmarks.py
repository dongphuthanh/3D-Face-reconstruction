"""FLAME landmark embedding.

MPI ships landmarks as a barycentric embedding rather than vertex indices: each
landmark is a point *inside* a triangle, given by a face index and three
weights. That is strictly better than snapping to the nearest vertex — anatomical
landmarks rarely sit exactly on one — and it stays differentiable in the vertex
positions, which is what the landmark loss needs.

The mediapipe variant embeds 105 points of MediaPipe's FaceMesh into FLAME's
surface, so the detector to pair it with is MediaPipe FaceMesh, not FAN.
"""

import pathlib

import numpy as np
import torch


class LandmarkEmbedding:
    def __init__(self, npz_path, device="cpu"):
        with np.load(pathlib.Path(npz_path)) as d:
            self.face_idx = torch.as_tensor(
                d["lmk_face_idx"].astype(np.int64).ravel()).to(device)
            self.b_coords = torch.as_tensor(
                d["lmk_b_coords"].astype(np.float32)).to(device)
            key = "landmark_indices" if "landmark_indices" in d else None
            self.detector_idx = (torch.as_tensor(d[key].astype(np.int64).ravel()).to(device)
                                 if key else None)
        self.n = self.face_idx.shape[0]

    def __len__(self):
        return self.n

    def positions(self, verts, faces):
        """verts (B,V,3), faces (F,3) -> (B,L,3). Differentiable in verts."""
        tri = verts[:, faces[self.face_idx]]              # (B,L,3,3)
        return (tri * self.b_coords.unsqueeze(0).unsqueeze(-1)).sum(2)

    def validate(self, n_faces, n_verts):
        """Cheap structural checks; a mismatched embedding silently misplaces
        every landmark and shows up only as a stubbornly high loss."""
        problems = []
        if int(self.face_idx.max()) >= n_faces:
            problems.append(f"face index {int(self.face_idx.max())} >= {n_faces} faces")
        if not torch.allclose(self.b_coords.sum(-1), torch.ones(self.n), atol=1e-5):
            problems.append("barycentric weights do not sum to 1")
        if float(self.b_coords.min()) < -1e-6:
            problems.append("negative barycentric weight")
        return problems
