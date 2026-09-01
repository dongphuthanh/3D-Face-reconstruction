"""Ingest CelebA: 202k real photographs of 10,177 labelled identities.

Why this corpus. scripts/eval/diag_corpus_identity.py measured DigiFace's
between-subject facenet distance at 0.76 against 0.92-0.98 for every real
corpus -- its 100k synthetic identities are less distinct from each other than
real people are, which caps the between-subject shape spread an encoder trained
on it can reach. CelebA is the widest measured at 0.98.

Its role is the MIX corpus, replacing FFHQ's 17,981 images. Over three epochs
the mix draws ~623k images, so FFHQ repeats ~35x and CelebA ~3x. The identity
and photometric losses both run on mix batches, so this is the real-photograph
diet those two terms see.

What it is NOT good for, and this is deliberate: CelebA's within-subject
distance is 0.33, no better than DigiFace, because one celebrity spans years,
makeup and lighting. Some of that variation is age and weight, which genuinely
change 3D shape -- so a CelebA identity is not one fixed geometry. Identity
labels are ingested because they are free to keep, but the swap and consistency
losses should stay on DigiFace, where an identity IS one asset.

Margin. The images are the ALIGNED 178x218 crops, already tight on the face, so
a 1.6 crop runs off the edge and PIL pads it black: 3.2% of area on average,
above 5% for 22% of images. Ingested at 1.6 anyway. Every other corpus and the
NoW eval crop use 1.6, and an odd face scale is the failure this project has
actually been bitten by -- eval cam scale 9.2 against train 6.9 -- while a few
per cent of black at the corners sits outside the skin mask.

Licence. Non-commercial research only, no redistribution, copies allowed only
for internal use at a single site. The identity annotations are formally
released on request; the flwrlabs mirror ships them. Source images are
"obtained from the Internet", the same scraped-provenance trade recorded for
Arc2Face. Only this recipe is tracked; data/ and data/_hfcache/ are ignored.

Resumable: existing crops are skipped, and each shard's landmark records are
written before the next is fetched. Parquet shards are deleted after use --
all 19 are 9.5 GB and only the crops are needed afterwards.
"""

import argparse, io, pathlib, sys, threading, time
import concurrent.futures as cf

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from face3d.learn.detect import FaceDetector, select_embedding_points, crop_square

ROOT = pathlib.Path(__file__).resolve().parents[2]
CELEBA = ROOT / "data" / "celeba"
REPO = "flwrlabs/celeba"
N_SHARDS = 19


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--margin", type=float, default=1.6,
                    help="see the module docstring before changing this")
    ap.add_argument("--shards", type=int, default=N_SHARDS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keep-parquet", action="store_true",
                    help="do not delete each shard after processing (9.5 GB total)")
    a = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    out = CELEBA / "crops"
    out.mkdir(parents=True, exist_ok=True)
    cache = CELEBA / f"landmarks_{a.size}.npz"

    emb = np.load(ROOT / "mediapipe_landmark_embedding" /
                  "mediapipe_landmark_embedding.npz")
    need = emb["landmark_indices"].astype(int).ravel()

    have = {}
    if cache.exists():
        z = np.load(cache)
        have = {str(k): (l, str(s)) for k, l, s
                in zip(z["keys"], z["landmarks"], z["subject"])}
        print(f"    reusing {len(have)} cached landmark records")

    # MediaPipe graphs are stateful; one detector per worker thread.
    local = threading.local()

    def detector():
        if not hasattr(local, "det"):
            local.det = FaceDetector(blendshapes=False)
        return local.det

    lock = threading.Lock()
    tally = {"ok": 0, "nodet": 0, "err": 0, "skip": 0}

    def work(item):
        key, subject, raw = item
        dst = out / f"{key}.jpg"
        if key in have:
            with lock:
                tally["skip"] += 1
            return (key, have[key][0], subject)
        try:
            if dst.exists():
                # Crop on disk but no landmark record: re-detect on the crop.
                # It is already face-centred, so normalised coords are NDC.
                im = np.asarray(Image.open(dst).convert("RGB"))
                res = detector().detect(im)
                if res is None:
                    with lock:
                        tally["nodet"] += 1
                    return None
                pts = select_embedding_points(res["norm"], need)
                ndc = np.stack([pts[:, 0] * 2 - 1, 1 - pts[:, 1] * 2], -1)
                with lock:
                    tally["skip"] += 1
                return (key, ndc.astype(np.float32), subject)

            im = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
            res = detector().detect(im)
            if res is None:
                with lock:
                    tally["nodet"] += 1
                return None
            pts = select_embedding_points(res["norm"], need)
            crop, to_ndc = crop_square(im, res["norm"], size=a.size,
                                       margin=a.margin)
            Image.fromarray(crop).save(dst, quality=95)
            with lock:
                tally["ok"] += 1
            return (key, to_ndc(pts).astype(np.float32), subject)
        except Exception:
            with lock:
                tally["err"] += 1
            return None

    print(f"=== CelebA ingest: {a.shards} shards -> {a.size}px, margin {a.margin} ===")
    records = {k: (v[0], v[1]) for k, v in have.items()}
    t0 = time.time()

    for shard in range(a.shards):
        name = f"img_align+identity+attr/train-{shard:05d}-of-{N_SHARDS:05d}.parquet"
        path = pathlib.Path(hf_hub_download(REPO, name, repo_type="dataset",
                                            cache_dir=str(ROOT / "data" / "_hfcache")))
        tab = pq.ParquetFile(path).read(columns=["image", "celeb_id"])
        ids = tab.column("celeb_id").to_pylist()
        imgs = tab.column("image").to_pylist()

        items = []
        for i, cid in enumerate(ids):
            rec = imgs[i]
            raw = rec["bytes"] if isinstance(rec, dict) else rec
            items.append((f"{cid:05d}_{shard:02d}{i:05d}", str(cid), raw))
        del tab, imgs

        with cf.ThreadPoolExecutor(a.workers) as ex:
            for r in ex.map(work, items):
                if r is not None:
                    records[r[0]] = (r[1], r[2])
        del items

        # Write after every shard: an interrupted ingest keeps its progress.
        keys = sorted(records)
        np.savez_compressed(
            cache,
            keys=np.array(keys),
            landmarks=np.stack([records[k][0] for k in keys]).astype(np.float32),
            subject=np.array([records[k][1] for k in keys]))

        if not a.keep_parquet:
            path.unlink(missing_ok=True)

        el = time.time() - t0
        print(f"  shard {shard:2d}/{a.shards}  {len(records):6d} records  "
              f"ok={tally['ok']} skip={tally['skip']} nodet={tally['nodet']} "
              f"err={tally['err']}  [{el:.0f}s]", flush=True)

    subs = len({v[1] for v in records.values()})
    print(f"\n{len(records)} images across {subs} identities -> {cache}")


if __name__ == "__main__":
    main()
