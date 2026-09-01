"""Ingest an FFHQ subset: download, detect, crop, cache landmarks.

Only permissively-licensed images are taken. FFHQ as a whole is CC BY-NC-SA,
but 58% of its images are individually CC BY 2.0, Public Domain, CC0 or US
Government Work. Since a fine-tune needs far fewer than 70k images, filtering
costs nothing and keeps the provenance of the trained encoder clean -- which is
what the on-device story in plan section 9 depends on.

Images are downloaded at 1024px, cropped around the detected face and written at
`--size`; the original is never kept. Bandwidth is the cost here, not disk.

Resumable: anything already cropped is skipped.
"""

import argparse, json, pathlib, sys, time, threading
import concurrent.futures as cf

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from face3d.io.gdrive import download as gdrive_download
from face3d.learn.detect import FaceDetector, select_embedding_points, crop_square

ROOT = pathlib.Path(__file__).resolve().parents[2]
FFHQ = ROOT / "data" / "ffhq"

PERMISSIVE = {
    "Attribution License",
    "Public Domain Mark",
    "Public Domain Dedication (CC0)",
    "United States Government Work",
}


def select(meta_path, limit, permissive_only=True):
    with open(meta_path) as f:
        d = json.load(f)
    out = []
    for k in sorted(d, key=int):
        e = d[k]
        lic = e.get("metadata", {}).get("license", "")
        if permissive_only and lic not in PERMISSIVE:
            continue
        out.append((k, e["image"], lic))
        if len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--all-licences", action="store_true")
    ap.add_argument("--out", default=str(FFHQ / "crops"))
    a = ap.parse_args()

    meta = FFHQ / "ffhq-dataset-v2.json"
    if not meta.exists():
        print(f"SKIP - metadata missing: {meta}"); sys.exit(0)

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items = select(meta, a.limit, not a.all_licences)
    print(f"=== FFHQ ingest: {len(items)} images "
          f"({'all licences' if a.all_licences else 'permissive only'}) -> {a.size}px ===")

    emb = np.load(ROOT / "mediapipe_landmark_embedding" /
                  "mediapipe_landmark_embedding.npz")
    need = emb["landmark_indices"].astype(int).ravel()

    # One detector per worker thread: MediaPipe graphs are stateful and not
    # safe to share across threads.
    local = threading.local()

    def detector():
        if not hasattr(local, "det"):
            local.det = FaceDetector(blendshapes=False)
        return local.det

    cache = out.parent / f"landmarks_{a.size}.npz"
    have, have_lic = {}, {}
    if cache.exists():
        z = np.load(cache)
        have = {str(k): v for k, v in zip(z["keys"], z["landmarks"])}
        have_lic = {str(k): str(v) for k, v in zip(z["keys"], z["licenses"])}
        print(f"    reusing {len(have)} cached landmark records")

    lock = threading.Lock()
    done = {"ok": 0, "nodet": 0, "err": 0, "skip": 0, "bytes": 0}
    records = []

    def work(item):
        key, spec, lic = item
        dst = out / f"{key}.jpg"
        if dst.exists():
            # A previously-downloaded crop still needs a landmark record, or a
            # resumed run silently produces fewer landmarks than crops. Re-detect
            # on the crop itself: it is already face-centred, so normalised
            # coordinates map straight to NDC.
            with lock:
                done["skip"] += 1
            if key in have:
                return (key, have[key], have_lic.get(key, lic))
            im = np.asarray(Image.open(dst).convert("RGB"))
            res = detector().detect(im)
            if res is None:
                return None
            pts = select_embedding_points(res["norm"], need)
            ndc = np.stack([pts[:, 0] * 2 - 1, 1 - pts[:, 1] * 2], -1)
            return (key, ndc.astype(np.float32), lic)
        tmp = out / f".{key}.png"
        try:
            fid = spec["file_url"].rsplit("id=", 1)[1]
            gdrive_download(fid, tmp, expected_size=spec["file_size"],
                            expected_md5=spec["file_md5"])
            im = np.asarray(Image.open(tmp).convert("RGB"))
            res = detector().detect(im)
            if res is None:
                with lock:
                    done["nodet"] += 1
                return None
            pts = select_embedding_points(res["norm"], need)
            crop, to_ndc = crop_square(im, res["norm"], size=a.size)
            Image.fromarray(crop).save(dst, quality=95)
            with lock:
                done["ok"] += 1
                done["bytes"] += spec["file_size"]
            return (key, to_ndc(pts).astype(np.float32), lic)
        except Exception as e:
            with lock:
                done["err"] += 1
            return ("ERR", key, f"{type(e).__name__}: {e}")
        finally:
            tmp.unlink(missing_ok=True)

    t0 = time.time()
    errors = []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(work, items), 1):
            if r is not None:
                (errors if r[0] == "ERR" else records).append(r)
            if i % 50 == 0 or i == len(items):
                el = time.time() - t0
                print(f"  {i:5d}/{len(items)}  ok={done['ok']} no-face={done['nodet']} "
                      f"err={done['err']} skip={done['skip']}  "
                      f"{done['bytes']/1e6:.0f} MB  {el:.0f}s", flush=True)

    if records:
        np.savez_compressed(
            cache,
            keys=np.array([r[0] for r in records]),
            landmarks=np.stack([r[1] for r in records]),
            licenses=np.array([r[2] for r in records]))
        print(f"\n  wrote landmarks_{a.size}.npz  "
              f"({len(records)} x {records[0][1].shape})")
    n = done["ok"] + done["nodet"]
    if n:
        print(f"  detection rate: {done['ok']}/{n} = {done['ok']/n*100:.1f}%")
    for e in errors[:5]:
        print(f"  ERROR {e[1]}: {e[2]}")


if __name__ == "__main__":
    main()
