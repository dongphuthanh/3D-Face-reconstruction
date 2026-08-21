"""Ingest Arc2Face identity groups, reading the archives remotely.

Arc2Face is WebFace42M restored to 448px: 21M images across 1M identities,
grouped by identity, in 35 archives of about 28 GB each. Downloading one to use
a few percent of it would be absurd, so face3d/remotezip.py reads the central
directory over HTTP range requests and fetches only the wanted members. The
index for a 28 GB archive reads in about 8 seconds.

Why this over DigiFace: 30k identities per archive against DigiFace's 10k
total, 448px against 112px, and real photographs with real pose variation --
profiles and three-quarter views, which is exactly where MediaPipe failed on
NoW and where FFHQ (99.9% frontal) taught nothing.

Provenance is the cost. Arc2Face derives from WebFace42M, which is scraped.
CC BY-NC-SA 4.0 covers the authors' restoration work, not the underlying
images, and ShareAlike may bind derivative models. Record it in the licence
register accordingly.

Members of one identity are contiguous in the archive, so iterating in archive
order means the 8 MB read cache serves many identities per HTTP request.
"""

import argparse
import collections
import io
import pathlib
import sys
import threading
import time
import concurrent.futures as cf

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from face3d.detect import FaceDetector, crop_square, select_embedding_points
from face3d.remotezip import open_remote

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARC = ROOT / "data" / "arc2face"
BASE = "https://huggingface.co/datasets/FoivosPar/Arc2Face/resolve/main"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/0_0.zip", help="archive within the repo")
    ap.add_argument("--identities", type=int, default=10000)
    ap.add_argument("--per-identity", type=int, default=6)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--margin", type=float, default=1.15,
                    help="crop margin. Sources are already tightly framed, so "
                         "the 1.6 used for FFHQ would pad with large black bands")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(ARC / "crops"))
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    emb = np.load(ROOT / "mediapipe_landmark_embedding" /
                  "mediapipe_landmark_embedding.npz")
    need = emb["landmark_indices"].astype(int).ravel()

    url = f"{BASE}/{a.part}"
    print(f"=== Arc2Face ingest: {a.part} ===")
    t0 = time.time()
    index = open_remote(url)
    groups = collections.defaultdict(list)
    for n in index.namelist():
        if n.lower().endswith((".jpg", ".jpeg", ".png")):
            groups[n.split("/")[0]].append(n)
    ids = sorted(groups)[: a.identities]
    print(f"    index: {len(groups)} identities, taking {len(ids)} "
          f"x {a.per_identity} ({time.time() - t0:.0f}s)")

    cache = out.parent / f"landmarks_{a.size}.npz"
    have = {}
    if cache.exists():
        z = np.load(cache)
        have = {str(k): (v, str(s)) for k, v, s in
                zip(z["keys"], z["landmarks"], z["subject"])}
        print(f"    reusing {len(have)} cached records")

    # One archive handle and one detector per worker: neither the HTTP reader
    # nor a MediaPipe graph is safe to share across threads.
    local = threading.local()

    def worker_state():
        if not hasattr(local, "z"):
            local.z = open_remote(url)
            local.det = FaceDetector(blendshapes=False)
        return local.z, local.det

    lock = threading.Lock()
    stats = collections.Counter()
    records = []

    def do_identity(sid):
        z, det = worker_state()
        got = []
        for n in sorted(groups[sid])[: a.per_identity + 3]:
            if len(got) >= a.per_identity:
                break
            key = f"{sid}_{pathlib.Path(n).stem}"
            if key in have:
                lm, s = have[key]
                got.append((key, lm, s))
                continue
            try:
                with z.open(n) as fh:
                    im = np.array(Image.open(io.BytesIO(fh.read())).convert("RGB"),
                                  np.uint8)
            except Exception:
                with lock:
                    stats["read_error"] += 1
                continue
            r = det.detect(im)
            if r is None:
                with lock:
                    stats["no_face"] += 1
                continue
            pts = select_embedding_points(r["norm"], need)
            crop, to_ndc = crop_square(im, r["norm"], size=a.size, margin=a.margin)
            Image.fromarray(crop).save(out / f"{key}.jpg", quality=95)
            got.append((key, to_ndc(pts).astype(np.float32), sid))
        # An identity with a single usable image cannot form a pair, so it
        # contributes nothing to the swap loss and is dropped.
        if len(got) < 2:
            with lock:
                stats["too_few"] += 1
            return []
        return got

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, got in enumerate(ex.map(do_identity, ids), 1):
            records.extend(got)
            if i % 500 == 0 or i == len(ids):
                el = time.time() - t0
                print(f"  {i:6d}/{len(ids)} ids  {len(records):7d} images  "
                      f"no-face={stats['no_face']} thin={stats['too_few']}  "
                      f"{el:.0f}s  ({i / max(el, 1):.1f} id/s)", flush=True)

    if records:
        np.savez_compressed(
            cache,
            keys=np.array([r[0] for r in records]),
            landmarks=np.stack([r[1] for r in records]),
            subject=np.array([r[2] for r in records]))
        n_ids = len({r[2] for r in records})
        print(f"\n  wrote {cache.name}  ({len(records)} images, {n_ids} identities)")
    print(f"  no-face {stats['no_face']} | too-few-images {stats['too_few']} | "
          f"read errors {stats['read_error']}")


if __name__ == "__main__":
    main()
