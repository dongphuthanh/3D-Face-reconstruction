"""Datasets over the ingested crops.

Landmarks are cached in NDC at ingest time so the detector does not run during
training -- MediaPipe would otherwise dominate step time and cannot be batched
on the GPU.

Two datasets, because they supply different things:

  FFHQCrops     real photographs, one per person. Carries the landmark and
                photometric terms, and keeps the model anchored on the real
                image distribution it will be asked about.
  IdentityPairs identity-grouped renders. Returns two DIFFERENT images of the
                same subject, which is the constraint the shape swap needs and
                the one thing no amount of augmentation could substitute for.
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


class IdentityPairs(Dataset):
    """Two different images of one subject, batched so halves match by identity.

    Item i returns a pair. Collated into a batch of B pairs, the default
    collate gives tensors of shape (B, 2, ...); `flatten_pairs` reshapes those
    into the (2B, ...) layout that swap_shape expects, with the two halves
    aligned by identity. That is the same layout `two_views` produces, so the
    swap loss needs no change -- only its meaning does. Augmented views only
    ever taught invariance to the augmentation; these are genuinely different
    images of one face.
    """

    def __init__(self, root, size=224, split="train", val_frac=0.05, seed=0,
                 min_images=2):
        root = pathlib.Path(root)
        cache = root / f"landmarks_{size}.npz"
        if not cache.exists():
            raise FileNotFoundError(
                f"{cache} not found - run scripts/ingest_digiface.py first")
        z = np.load(cache)
        keys = np.array([str(k) for k in z["keys"]])
        lmk = z["landmarks"].astype(np.float32)
        subj = np.array([str(s) for s in z["subject"]])

        groups = {}
        for i, s in enumerate(subj):
            groups.setdefault(s, []).append(i)
        ids = sorted(s for s, idx in groups.items() if len(idx) >= min_images)

        # Split by identity, not by image: the same person must never appear in
        # both train and val, or the held-out score measures memorisation.
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(ids))
        n_val = max(1, int(len(ids) * val_frac))
        chosen = order[n_val:] if split == "train" else order[:n_val]

        self.groups = [np.array(groups[ids[i]]) for i in chosen]
        self.dir = root / "crops"
        self.cache = cache
        self.rng = np.random.default_rng(seed + 1)

        # Keys and landmarks are NOT stored on the instance. On Windows,
        # DataLoader workers are spawned and the dataset is pickled through a
        # pipe; at 110k identities that array is 444 MB and the pipe rejects it
        # with OSError [Errno 22], which reads as a mysterious spawn failure
        # rather than a size limit. Each worker loads its own copy lazily
        # instead, so only paths and index arrays cross the pipe.
        self._keys = None
        self._lmk = None

    def _arrays(self):
        if self._lmk is None:
            z = np.load(self.cache)
            self._keys = np.array([str(k) for k in z["keys"]])
            self._lmk = z["landmarks"].astype(np.float32)
        return self._keys, self._lmk

    @property
    def keys(self):
        return self._arrays()[0]

    @property
    def lmk(self):
        return self._arrays()[1]

    def __len__(self):
        return len(self.groups)

    def _load(self, i):
        keys, lmk = self._arrays()
        im = Image.open(self.dir / f"{keys[i]}.jpg").convert("RGB")
        arr = np.array(im, dtype=np.uint8)
        return (torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0,
                torch.from_numpy(lmk[i]),
                torch.from_numpy((arr.sum(-1) > 0).astype(np.float32)))

    def __getitem__(self, k):
        g = self.groups[k]
        a, b = self.rng.choice(len(g), size=2, replace=False)
        ia, ib = self._load(g[a]), self._load(g[b])
        return {"image": torch.stack([ia[0], ib[0]]),
                "landmarks": torch.stack([ia[1], ib[1]]),
                "valid": torch.stack([ia[2], ib[2]])}


def flatten_pairs(batch):
    """(B,2,...) -> (2B,...) with view A stacked above view B, halves aligned."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == 2:
            out[k] = torch.cat([v[:, 0], v[:, 1]], dim=0)
        else:
            out[k] = v
    return out
