"""Dataset over the ingested FFHQ crops.

Landmarks are cached in NDC at ingest time so the detector does not run during
training -- MediaPipe would otherwise dominate step time and cannot be batched
on the GPU.
"""

import pathlib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class FFHQCrops(Dataset):
    def __init__(self, root, size=224, split="train", val_frac=0.1, seed=0):
        root = pathlib.Path(root)
        cache = root / f"landmarks_{size}.npz"
        if not cache.exists():
            raise FileNotFoundError(
                f"{cache} not found - run scripts/ingest_ffhq.py first")
        z = np.load(cache)
        keys = np.array([str(k) for k in z["keys"]])
        lmk = z["landmarks"].astype(np.float32)

        # Fixed permutation from a fixed seed, so the held-out set stays the
        # same across runs.
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(keys))
        n_val = max(1, int(len(keys) * val_frac))
        idx = order[n_val:] if split == "train" else order[:n_val]

        self.keys, self.lmk = keys[idx], lmk[idx]
        self.dir = root / "crops"
        self.size = size

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        im = Image.open(self.dir / f"{self.keys[i]}.jpg").convert("RGB")
        # np.array (not asarray): PIL hands back a read-only buffer, and
        # torch.from_numpy on it warns on every single sample.
        arr = np.array(im, dtype=np.uint8)
        img = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        # Square crops can extend past the original image, which PIL pads black.
        # Those pixels are not evidence about the face and must not enter the
        # photometric loss.
        valid = torch.from_numpy((arr.sum(-1) > 0).astype(np.float32))
        return {"image": img, "landmarks": torch.from_numpy(self.lmk[i]),
                "valid": valid, "key": self.keys[i]}
