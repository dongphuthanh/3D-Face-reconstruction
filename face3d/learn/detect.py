"""2D face landmarks from MediaPipe, aligned to the FLAME embedding.

The pairing that matters: MPI's landmark embedding places 105 *MediaPipe*
vertices on FLAME's surface, and this detector emits MediaPipe's 478 points.
Selecting the embedding's `landmark_indices` from the detection gives a 1:1
correspondence with `LandmarkEmbedding.positions()` — which is exactly what the
landmark loss needs, and why the detector must be MediaPipe and not FAN.

MediaPipe 1.0 removed the legacy `mp.solutions.face_mesh` API that most FLAME-era
code uses; this wraps the Tasks API instead.
"""

import pathlib

import numpy as np

MODEL = pathlib.Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"


class FaceDetector:
    """Wraps FaceLandmarker. Not thread-safe; MediaPipe graphs are stateful."""

    def __init__(self, model_path=MODEL, num_faces=1, blendshapes=False,
                 min_confidence=0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        model_path = pathlib.Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model bundle missing: {model_path}\n"
                f"  curl -sSL -o {model_path} https://storage.googleapis.com/"
                f"mediapipe-models/face_landmarker/face_landmarker/float16/1/"
                f"face_landmarker.task")

        self._mp = mp
        self._blendshapes = blendshapes
        self._landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=num_faces,
                min_face_detection_confidence=min_confidence,
                output_face_blendshapes=blendshapes))

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def detect(self, image):
        """image: HxWx3 uint8 RGB. Returns None when no face is found.

        Returning None rather than raising is deliberate: "no face" is an
        expected input in this pipeline (story D8), not an error condition.
        """
        arr = np.ascontiguousarray(image, dtype=np.uint8)
        res = self._landmarker.detect(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=arr))
        if not res.face_landmarks:
            return None

        h, w = arr.shape[:2]
        lm = res.face_landmarks[0]
        # MediaPipe returns x,y normalised to the image and z in a
        # roughly-comparable unit; only x,y are used for the landmark loss.
        norm = np.array([[p.x, p.y] for p in lm], dtype=np.float32)
        out = {"norm": norm, "pixel": norm * np.array([w, h], np.float32),
               "n_faces": len(res.face_landmarks)}
        if self._blendshapes and res.face_blendshapes:
            out["blendshapes"] = {b.category_name: b.score
                                  for b in res.face_blendshapes[0]}
        return out


def select_embedding_points(norm_xy, embedding_indices):
    """Pick the points the FLAME embedding was built on. (478,2) -> (105,2)."""
    idx = np.asarray(embedding_indices, dtype=np.int64).ravel()
    if idx.max() >= norm_xy.shape[0]:
        raise ValueError(f"embedding wants index {idx.max()} but the detector "
                         f"returned only {norm_xy.shape[0]} landmarks")
    return norm_xy[idx]


def crop_square(image, norm_xy, size=224, margin=1.6):
    """Square crop around the detected face, plus the transform to undo it.

    Returns (crop uint8, to_ndc) where to_ndc maps *original* normalised image
    coordinates into the crop's [-1,1] NDC frame. The landmark loss compares
    against the mesh rendered in that frame, so the mapping has to travel with
    the crop rather than be recomputed later.
    """
    from PIL import Image

    h, w = image.shape[:2]
    px = norm_xy * np.array([w, h], np.float32)
    lo, hi = px.min(0), px.max(0)
    centre = (lo + hi) / 2
    half = float((hi - lo).max()) * margin / 2

    x0, y0 = centre - half, centre + half
    box = (int(round(centre[0] - half)), int(round(centre[1] - half)),
           int(round(centre[0] + half)), int(round(centre[1] + half)))
    crop = Image.fromarray(image).crop(box).resize((size, size), Image.BILINEAR)

    def to_ndc(n_xy):
        p = np.asarray(n_xy, np.float32) * np.array([w, h], np.float32)
        u = (p - np.array([box[0], box[1]], np.float32)) / (2 * half)   # [0,1]
        return np.stack([u[..., 0] * 2 - 1, 1 - u[..., 1] * 2], -1)     # y up

    return np.asarray(crop), to_ndc
