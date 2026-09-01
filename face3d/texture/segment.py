"""Semantic segmentation of a portrait, via MediaPipe's selfie multiclass model.

Used to decide which pixels of a photograph may be sampled into the texture.
The class boundary falls exactly where the texture projection needs it:
eyebrows and lips are FACE_SKIN and survive; spectacle frames and headdresses
are OTHER, and hair fallen across a forehead is HAIR, so none of them are
painted onto the face. Auditing 14 FFHQ portraits, 8 carried an occluder of
some kind, which was the dominant artefact before this gate existed.
"""

import pathlib

import numpy as np

MODEL = (pathlib.Path(__file__).resolve().parents[2] / "models" /
         "selfie_multiclass_256x256.tflite")

# Category ids in MediaPipe's selfie multiclass segmenter.
BACKGROUND, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHER = range(6)

class Segmenter:
    """Wraps MediaPipe's selfie multiclass segmenter. NOT thread-safe.

    Same constraint as FaceDetector, for the same reason: MediaPipe graphs hold
    state across calls, so two threads sharing one instance corrupt each other.
    Callers that serve concurrent requests need one per thread.
    """

    def __init__(self, model_path=MODEL):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        model_path = pathlib.Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe segmenter missing: {model_path}\n"
                f"  curl -sSL -o {model_path} https://storage.googleapis.com/"
                f"mediapipe-models/image_segmenter/selfie_multiclass_256x256/"
                f"float32/latest/selfie_multiclass_256x256.tflite")

        self._mp = mp
        self._seg = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                output_category_mask=True))

    def categories(self, image):
        """(H,W) uint8 category ids for an RGB uint8 image."""
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(image))
        # numpy_view() gives (H,W,1); squeeze so callers can index with it.
        return np.squeeze(self._seg.segment(img).category_mask.numpy_view())

    def hair(self, image):
        return self.categories(image) == HAIR

    def close(self):
        self._seg.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
