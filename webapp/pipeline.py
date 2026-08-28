"""Photograph bytes -> GLB bytes. No HTTP anywhere in this file.

Keeping the domain logic free of web concerns means you can test the hard part
from a plain Python prompt:

    from webapp.pipeline import Reconstructor
    r = Reconstructor()
    glb = r.reconstruct(open("photo.jpg", "rb").read())
    open("out/test.glb", "wb").write(glb)
    # then: node scripts/validate_glb.js out/test.glb

When the web layer later misbehaves you will already know the model half is fine.

Reference: scripts/export_head.py does all of this in order, as a CLI.
"""

import io
import pathlib
import threading

import numpy as np
import torch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]

from face3d import assets
from face3d.albedo import CACHE_DIR, FlameTexture, face_texel_mask, harmonise
from face3d.detect import FaceDetector, crop_square
from face3d.encoder import ResNetEncoder
from face3d.facemask import eye_faces
from face3d.flame_torch import FlameTorch
from face3d.gltf import build_gltf, glb_bytes
from face3d.landmarks import LandmarkEmbedding
from face3d.rig import JOINT_NAMES, expression_targets, jaw_target, rest_joints

# Must match training. Every corpus and the NoW evaluation crop at 1.6, and a
# mismatch is a distribution shift, not a cosmetic difference. See MODEL_CARD.md.
CROP_MARGIN = 1.6

# How many expression blendshapes to export. FLAME has 100; most real-time
# engines choke well before that, and the tail components move vertices by
# fractions of a millimetre. 20 + jaw_open is a reasonable rig.
MORPH_TARGETS = 20


class NoFaceFound(Exception):
    """No face in the image. A USER error, not a server fault -- the caller
    should turn this into a 422, never a 500."""


class Reconstructor:
    """Built ONCE per process, reused for every request.

    Loading takes ~4 seconds; running takes ~40 milliseconds. Anything that
    constructs this per request is 100x slower than it needs to be, and that is
    the most common mistake in ML web services.

    deca_open is the default: the only licence-clean checkpoint (FLAME 2023
    Open + DigiFace + FFHQ, no CelebA, no FLAME 2020), the only one whose
    exported meshes may be redistributed, and the only one with predicted eye
    colour. It gives up nothing measurable to be all three.

    THE CHECKPOINT AND THE FLAME BASIS MUST MATCH, and nothing enforces that
    automatically. face3d/assets.py resolves FLAME by searching CANDIDATES,
    which lists FLAME2020 FIRST, so leaving it implicit would silently drive
    deca_open's weights through the 2020 basis. The two bases share topology and
    dimensions, so nothing would raise -- but their axes are rotated, retaining
    only 81% of each other's energy, and the output would be quiet nonsense.
    Hence flame_model is explicit here rather than inherited from list order.
    """

    # Matched pair. Change both or neither.
    CHECKPOINT = ROOT / "runs" / "deca_open" / "encoder.pt"
    FLAME_MODEL = "FLAME2023Open/flame2023_Open.pkl"

    def __init__(self, checkpoint=None, device="cpu", flame_model=None):
        self.device = device
        checkpoint = pathlib.Path(checkpoint or self.CHECKPOINT)
        flame_model = flame_model or self.FLAME_MODEL

        # FLAME: the frozen statistical face model. Given (shape, expression,
        # pose) coefficients it produces 5023 vertices. Nothing here is trained
        # -- it is a fixed basis the encoder learns to drive.
        flame_path = assets.model_path(flame_model)
        if flame_path is None:
            raise FileNotFoundError(
                f"FLAME basis {flame_model!r} not found. It must match the "
                f"checkpoint: {checkpoint.parent.name} was trained on it, and a "
                f"mismatch produces plausible-looking nonsense rather than an "
                f"error.")
        print(f"    FLAME: {flame_path.name}   checkpoint: {checkpoint.parent.name}",
              flush=True)
        self.flame = FlameTorch(flame_path).to(device)

        # The encoder: ResNet-50 trunk + one linear head emitting 186 numbers
        # (100 shape, 50 expression, 6 pose, 3 camera, 27 light, 50 albedo).
        # pretrained=False because we are about to overwrite every weight with
        # the checkpoint; downloading ImageNet weights first would just be slow.
        self.enc = ResNetEncoder(n_shape=100, n_expr=50,
                                 pretrained=False).to(device)
        self.enc.load_state_dict(
            torch.load(checkpoint, map_location=device)["model"])

        # .eval() switches BatchNorm to its running statistics and disables
        # dropout. Forgetting it makes results depend on batch composition --
        # a single-image request would normalise against itself and produce
        # garbage that still looks plausible.
        self.enc.eval()

        # The albedo basis: 50 PCA components over face textures, which turn
        # the predicted albedo coefficients into a 256x256 UV colour map.
        # Optional -- without it the GLB is geometry with no material.
        cache = CACHE_DIR / "flame_texture_256_50.npz"
        self.texture = FlameTexture(cache, device=device) if cache.exists() else None
        if self.texture is not None:
            self.texture.attach_eyes(self.flame)
        self.uv = self._uv_layout(cache)

        # Which UV texels the photometric loss actually optimised. Depends only
        # on FLAME and the landmark embedding, so it is built once here rather
        # than per request. See harmonise() for why it is needed.
        self.face_mask = None
        if cache.exists():
            emb = LandmarkEmbedding(
                ROOT / "mediapipe_landmark_embedding" /
                "mediapipe_landmark_embedding.npz", device=device)
            with np.load(cache) as d:
                self.face_mask = face_texel_mask(
                    self.flame, emb, d["vt"].astype(np.float32),
                    d["ft"].astype(np.int64))

        # MediaPipe graphs are stateful and NOT safe to share across threads.
        # FastAPI runs sync handlers in a worker threadpool, so two concurrent
        # requests would otherwise call detect() on one object at the same time.
        # threading.local() gives each thread its own; ingest_ffhq.py needed the
        # same fix for the same reason.
        # Eyeball triangles, so they can be exported as their own primitive
        # with a glossy material. Depends only on FLAME, so compute it once.
        self.eye_mask = eye_faces(self.flame).cpu().numpy()

        self._local = threading.local()

    def _detector(self):
        """One FaceDetector per thread, created lazily on first use."""
        if not hasattr(self._local, "det"):
            # blendshapes=False: MediaPipe can also regress 52 ARKit-style
            # expression scores, which we do not use and which cost time.
            self._local.det = FaceDetector(blendshapes=False)
        return self._local.det

    def _uv_layout(self, cache):
        """One UV coordinate per vertex, in glTF's orientation. (V,2) or None.

        The texture cache stores FLAME's UV unwrap as `vt` (the UV coordinates)
        and `ft` (which UV belongs to each triangle corner). That is OBJ's
        convention, and it does not line up with glTF's in two ways.

        1. vt has 5118 entries but the mesh has 5023 vertices. UV seams -- the
           cuts where the unwrap is split open -- duplicate a vertex so its two
           sides can land in different parts of the texture. glTF allows exactly
           one UV per vertex, so each vertex takes the UV of the first triangle
           corner that references it. Wrong only for triangles sitting exactly
           on a seam, which is a handful and invisible.

        2. OBJ puts v=0 at the BOTTOM of the image; glTF puts it at the TOP.
           Miss the flip and the head loads with its face upside down -- and
           still passes the validator, because the file is well-formed. This is
           the kind of bug you find by looking, never by testing return codes.
        """
        if not cache.exists():
            return None
        with np.load(cache) as d:
            vt = d["vt"].astype(np.float32)      # (5118, 2) UV coordinates
            ft = d["ft"].astype(np.int64)        # (9976, 3) UV index per corner

        faces = self.flame.faces.cpu().numpy()   # (9976, 3) vertex index per corner
        uv = np.zeros((self.flame.n_verts, 2), np.float32)
        for corner in range(3):
            # Scatter: for every triangle, vertex faces[:,c] gets UV vt[ft[:,c]].
            # Later writes overwrite earlier ones, so effectively "last writer
            # wins" -- which is fine, since only seam vertices disagree.
            uv[faces[:, corner]] = vt[ft[:, corner]]

        uv[:, 1] = 1.0 - uv[:, 1]                # OBJ bottom-up -> glTF top-down
        return uv

    def _bake_texture(self, albedo, eye=None):
        """50 albedo coefficients -> PNG bytes, entirely in memory.

        FlameTexture.texture() evaluates mean + sum(coeff_i * basis_i) to give a
        (1,3,256,256) image in [0,1]. We convert to 8-bit RGB and encode as PNG
        so it can be embedded directly in the GLB's binary chunk, keeping the
        asset a single self-contained file with no external references.
        """
        if self.texture is None or self.uv is None:
            return None
        with torch.no_grad():
            # albedo is (50,); texture() wants a batch, hence [None] -> (1,50).
            tex = self.texture.texture(albedo[None].to(self.device),
                                       eye=eye[None].to(self.device))[0]

        # (3,H,W) float [0,1] -> (H,W,3) uint8, which is what PIL expects.
        arr = tex.permute(1, 2, 0).clamp(0, 1).cpu().numpy()

        # Replace the un-fitted neck and scalp with the fitted face's own tone.
        # Without this the basis mean shows through at 27% higher saturation
        # than the subject, and a correct face reads as washed out beside it.
        if self.face_mask is not None:
            arr = harmonise(arr, self.face_mask)

        arr = (arr * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    def reconstruct(self, image_bytes: bytes) -> bytes:
        """Photograph -> GLB. Nothing touches the filesystem.

        Uploaded faces are biometric data under GDPR and BIPA. The easiest way
        to honour "do not retain it" is to have no code path that could.
        """
        # --- 1. decode -------------------------------------------------------
        # convert("RGB") normalises away greyscale, palettes and RGBA. Without
        # it a PNG with an alpha channel arrives as (H,W,4) and every later
        # shape assumption breaks.
        img = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

        # --- 2. find the face ------------------------------------------------
        # res["norm"] is (478,2) landmarks in normalised [0,1] image coords.
        res = self._detector().detect(img)
        if res is None:
            raise NoFaceFound("no face detected in the image")

        # --- 3. crop ---------------------------------------------------------
        # A square crop centred on the landmark bounding box, expanded by
        # CROP_MARGIN and resized to 224x224 -- exactly what the encoder saw in
        # training. Passing margin explicitly rather than relying on the default
        # so a reader can see it matches without checking the signature.
        crop, _ = crop_square(img, res["norm"], size=224, margin=CROP_MARGIN)

        # --- 4. to a tensor --------------------------------------------------
        # ascontiguousarray because crop_square can hand back a view with
        # negative strides, which torch.from_numpy refuses.
        # (224,224,3) uint8 -> permute to (3,224,224) -> [None] adds the batch
        # dimension -> (1,3,224,224) float in [0,1]. The encoder applies its own
        # ImageNet mean/std normalisation internally, so do not do it here.
        x = torch.from_numpy(np.ascontiguousarray(crop))
        x = x.to(self.device).permute(2, 0, 1)[None].float() / 255.0

        # --- 5. predict ------------------------------------------------------
        # no_grad() stops autograd building a graph we will never backward
        # through: less memory, faster.
        with torch.no_grad():
            # calibrate=True multiplies shape by SHAPE_CALIBRATION (0.40).
            # NOT optional: the raw prediction scores 1.5485 mm on NoW, WORSE
            # than emitting the FLAME mean face (1.3554). The shrinkage is the
            # only reason the model beats a constant mesh.
            p = self.enc.predict(x, calibrate=True)

        # From here on we are building the ASSET, not running the model, so
        # everything moves to numpy. [0] drops the batch dimension.
        shape = p.shape[0].cpu().numpy()          # (100,) identity coefficients

        # --- 6. blendshapes --------------------------------------------------
        # expression_targets returns, for THIS person's identity:
        #   deltas  (20, 5023, 3) per-vertex offsets, one per expression
        #   names   the target names, expr_00 .. expr_19
        #   neutral (5023, 3) the rest mesh the deltas are offsets FROM
        # glTF morph targets are offsets, not absolute positions, which is why
        # the neutral mesh and the deltas travel together.
        deltas, names, neutral = expression_targets(self.flame, shape,
                                                    n_targets=MORPH_TARGETS)

        # The jaw is a JOINT in FLAME, not an expression component, so opening
        # the mouth is not reachable through the expression basis alone. We bake
        # a jaw rotation into an extra blendshape so the rig can open its mouth
        # without the consumer needing to pose the skeleton. [None] makes it
        # (1,5023,3) so it concatenates onto the 20 expression deltas.
        deltas = np.concatenate([deltas, jaw_target(self.flame, shape)[None]], 0)
        names = names + ["jaw_open"]

        # --- 7. skeleton -----------------------------------------------------
        # Joint positions depend on identity: a larger head puts the neck joint
        # somewhere different. (5,3) for root/neck/jaw/eye_left/eye_right.
        joints = rest_joints(self.flame, shape).cpu().numpy()

        # --- 8. texture ------------------------------------------------------
        tex_png = self._bake_texture(p.albedo[0], p.eye[0])

        # --- 9. assemble ------------------------------------------------------
        # build_gltf returns (json_dict, binary_blob); glb_bytes packs them into
        # the GLB container with the 4-byte chunk padding the spec requires.
        # skin_weights (5023,5) says how much each joint influences each vertex;
        # build_gltf keeps the top 4 because glTF's JOINTS_0 is a vec4.
        gltf, blob = build_gltf(
            verts=neutral,
            faces=self.flame.faces.cpu().numpy(),
            joints=joints,
            parents=self.flame.parents,
            skin_weights=self.flame.weights.cpu().numpy(),
            joint_names=JOINT_NAMES,
            morph_targets=deltas,
            morph_names=names,
            uv=self.uv,
            name="face3d_head",
            texture_png=tex_png,
            # Split the eyeballs into their own primitive so they can be wet
            # and glossy while the skin stays matte.
            eye_mask=self.eye_mask,
        )
        return glb_bytes(gltf, blob)
