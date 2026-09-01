"""Export a rigged, blendshape-driven head as GLB — stories B4/B5/B6.

Takes either a photograph (encoder path) or the FLAME mean face, and writes a
glTF 2.0 binary containing a skinned mesh, a neck/jaw/eye armature, and the
expression basis as named morph targets.
"""

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from face3d import assets
from face3d.geometry.facemask import eye_faces
from face3d.geometry.flame_torch import FlameTorch
from face3d.export.gltf import build_gltf, write_glb, write_gltf
from face3d.geometry.rig import JOINT_NAMES, expression_targets, jaw_target, rest_joints
from webapp.pipeline import NoFaceFound, Reconstructor

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def uv_from_texture_space():
    """FLAME's UV layout, if the texture space has been cached.

    vt has more entries than the mesh has vertices because UV seams duplicate
    them. glTF requires one UV per vertex, so each vertex takes the UV of the
    first corner that references it -- correct everywhere except exactly on a
    seam, where the texture will be slightly wrong for one triangle.
    """
    from face3d.render.albedo import CACHE_DIR
    cache = CACHE_DIR / "flame_texture_256_50.npz"
    if not cache.exists():
        return None
    with np.load(cache) as d:
        return d["vt"].astype(np.float32), d["ft"].astype(np.int64)


def write(a, gltf, blob, flame, n_targets):
    """Write the GLB (and optionally the .gltf), then report what is in it."""
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_glb(out, gltf, blob)
    print(f"  wrote {out.name}  ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"  {flame.n_verts} verts, {flame.n_faces} tris, {len(JOINT_NAMES)} joints, "
          f"{n_targets} morph targets")
    if a.also_gltf:
        g = out.with_suffix(".gltf")
        write_gltf(g, gltf, blob)
        print(f"  wrote {g.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="", help="photograph; omit for the mean face")
    ap.add_argument("--checkpoint", default=str(Reconstructor.CHECKPOINT))
    ap.add_argument("--flame-model", default=Reconstructor.FLAME_MODEL,
                    help="MUST match the checkpoint. The two bases share "
                         "topology, so a mismatch renders quiet nonsense "
                         "rather than raising")
    ap.add_argument("--out", default=str(ROOT / "out" / "head.glb"))
    ap.add_argument("--targets", type=int, default=20,
                    help="expression morph targets to export. All 100 is valid "
                         "glTF but many real-time engines cap far lower")
    ap.add_argument("--no-project", action="store_true",
                    help="bake the PCA albedo only, instead of sampling the "
                         "photograph into UV space (face3d/texture/project.py)")
    ap.add_argument("--also-gltf", action="store_true")
    a = ap.parse_args()

    # The photograph path is Reconstructor's job, verbatim -- this script used
    # to carry its own copy of it, and the copy had drifted. It cropped and
    # encoded correctly but resolved FLAME through assets.model_path_or_skip(),
    # which returns FLAME 2020 whatever the checkpoint was trained on, and it
    # still defaulted to a checkpoint (runs/id_swap) that no longer exists.
    if a.image:
        ckpt = pathlib.Path(a.checkpoint)
        if not ckpt.exists():
            print(f"SKIP - no checkpoint at {ckpt}")
            sys.exit(0)
        rec = Reconstructor(checkpoint=ckpt, device=DEV,
                            flame_model=a.flame_model,
                            project=not a.no_project)
        try:
            gltf, blob = rec.build(pathlib.Path(a.image).read_bytes(),
                                   targets=a.targets)
        except NoFaceFound:
            print("no face found in the image")
            sys.exit(1)
        print(f"encoded {a.image}")
        write(a, gltf, blob, rec.flame, a.targets + 1)
        return

    # No photograph, so nothing to encode and nothing to project. The mean face
    # gets the mean albedo, i.e. zero coefficients, rather than no texture at
    # all -- a grey head is not a useful default.
    flame = FlameTorch(assets.model_path(a.flame_model)).to(DEV)
    shape = None
    albedo = torch.zeros(50)
    from face3d.render.eyes import default_eye_params
    _i, _s = default_eye_params(1)
    eye = torch.cat([_i, _s], 1)[0]
    print("no --image: exporting the FLAME mean face")

    deltas, names, neutral = expression_targets(flame, shape, n_targets=a.targets)
    deltas = np.concatenate([deltas, jaw_target(flame, shape)[None]], 0)
    names = names + ["jaw_open"]
    joints = rest_joints(flame, shape).cpu().numpy()

    uv = None
    got = uv_from_texture_space()
    if got is not None:
        vt, ft = got
        uv = np.zeros((flame.n_verts, 2), np.float32)
        f = flame.faces.cpu().numpy()
        for corner in range(3):                    # first writer wins per vertex
            uv[f[:, corner]] = vt[ft[:, corner]]
        # OBJ puts v=0 at the bottom of the image, glTF at the top.
        uv[:, 1] = 1.0 - uv[:, 1]

    # Bake the predicted albedo into the UV map the mesh already carries.
    # Without this the GLB is geometry only: correct, riggable, and grey.
    tex_png = None
    if uv is not None and albedo is not None:
        from io import BytesIO

        from PIL import Image

        from face3d.render.albedo import (CACHE_DIR as TEX_CACHE, FlameTexture,
                                   face_texel_mask, harmonise)
        from face3d.geometry.landmarks import LandmarkEmbedding
        cache = TEX_CACHE / "flame_texture_256_50.npz"
        if cache.exists():
            ft_tex = FlameTexture(cache, device=DEV).attach_eyes(flame)
            with torch.no_grad():
                t = ft_tex.texture(albedo[None].to(DEV),
                                   eye=eye[None].to(DEV))[0]       # (3,H,W)
            arr = t.permute(1, 2, 0).clamp(0, 1).cpu().numpy()

            # Replace the un-fitted neck and scalp with the fitted face's own
            # tone. The basis mean sits 27% more saturated than the subject, and
            # beside it a correctly fitted face reads as washed out.
            emb_lm = LandmarkEmbedding(
                ROOT / "mediapipe_landmark_embedding" /
                "mediapipe_landmark_embedding.npz", device=DEV)
            with np.load(cache) as d:
                m = face_texel_mask(flame, emb_lm, d["vt"].astype(np.float32),
                                    d["ft"].astype(np.int64))
            arr = harmonise(arr, m)
            img = (arr * 255).astype(np.uint8)
            b = BytesIO()
            Image.fromarray(img).save(b, format="PNG")
            tex_png = b.getvalue()

    gltf, blob = build_gltf(
        verts=neutral, faces=flame.faces.cpu().numpy(), joints=joints,
        parents=flame.parents, skin_weights=flame.weights.cpu().numpy(),
        joint_names=JOINT_NAMES, morph_targets=deltas, morph_names=names,
        uv=uv, name="face3d_head", texture_png=tex_png,
        eye_mask=eye_faces(flame).cpu().numpy())

    write(a, gltf, blob, flame, len(names))


if __name__ == "__main__":
    main()
