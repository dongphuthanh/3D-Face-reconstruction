"""Fetch CelebA's shape-relevant attributes and split identities by them.

CelebA has no timestamps -- the images are scraped celebrity photos with no
capture date -- so pairs cannot be filtered by time directly. What it does
annotate is the two axes that actually matter here: age (`Young`) and weight
(`Chubby`, `Double_Chin`). Those are the reasons a CelebA identity is not one
fixed geometry, and they are what the swap loss must not be asked to explain.

The filter works by REFINING the subject label rather than dropping images: a
celebrity photographed young and slim becomes a different subject from the same
celebrity photographed older and heavier. IdentityPairs then groups on the
refined label and never pairs across the boundary, with no change to the
dataset class.

Writes landmarks_224_swap.npz beside the ingest cache, same schema, so it is a
drop-in for --identity-data via a directory that symlinks or copies the crops.

Only celeb_id and the attribute columns are read; images are never decoded.
Shards are deleted after use.
"""

import argparse, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

ROOT = pathlib.Path(__file__).resolve().parents[2]
CELEBA = ROOT / "data" / "celeba"
REPO = "flwrlabs/celeba"
N_SHARDS = 19

# Attributes that change 3D FACE SHAPE. Hair attributes (Bald, Gray_Hair,
# Receding_Hairline) correlate with age but are not geometry, and every extra
# attribute halves the group sizes the swap loss depends on.
SHAPE_ATTRS = ["Young", "Chubby", "Double_Chin"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--attrs", nargs="*", default=SHAPE_ATTRS)
    ap.add_argument("--min-images", type=int, default=2,
                    help="drop refined subjects with fewer images than this")
    ap.add_argument("--keep-parquet", action="store_true")
    a = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    cache = CELEBA / f"landmarks_{a.size}.npz"
    if not cache.exists():
        print(f"SKIP - {cache} not found; run scripts/data/ingest_celeba.py first")
        sys.exit(0)
    z = np.load(cache)
    keys = np.array([str(k) for k in z["keys"]])
    lmk = z["landmarks"].astype(np.float32)
    subj = np.array([str(s) for s in z["subject"]])
    print(f"    {len(keys)} ingested crops, {len(set(subj))} identities")

    print(f"=== reading {a.attrs} from {N_SHARDS} shards (no images) ===")
    sig = {}
    for shard in range(N_SHARDS):
        name = f"img_align+identity+attr/train-{shard:05d}-of-{N_SHARDS:05d}.parquet"
        path = pathlib.Path(hf_hub_download(REPO, name, repo_type="dataset",
                                            cache_dir=str(ROOT / "data" / "_hfcache")))
        tab = pq.ParquetFile(path).read(columns=["celeb_id"] + a.attrs)
        ids = tab.column("celeb_id").to_pylist()
        cols = [tab.column(c).to_pylist() for c in a.attrs]
        for i, cid in enumerate(ids):
            # Same key construction as ingest_celeba.py.
            sig[f"{cid:05d}_{shard:02d}{i:05d}"] = "".join(
                "1" if cols[j][i] else "0" for j in range(len(a.attrs)))
        del tab, cols
        if not a.keep_parquet:
            path.unlink(missing_ok=True)
        print(f"  shard {shard:2d}/{N_SHARDS}  {len(sig)} signatures", flush=True)

    missing = [k for k in keys if k not in sig]
    if missing:
        print(f"    WARNING {len(missing)} crops have no attribute row; dropped")
    keep = np.array([k in sig for k in keys])
    keys, lmk, subj = keys[keep], lmk[keep], subj[keep]
    refined = np.array([f"{s}_{sig[k]}" for k, s in zip(keys, subj)])

    # Drop refined subjects too small to pair.
    counts = {}
    for s in refined:
        counts[s] = counts.get(s, 0) + 1
    ok = np.array([counts[s] >= a.min_images for s in refined])

    out = CELEBA / f"landmarks_{a.size}_swap.npz"
    np.savez_compressed(out, keys=keys[ok], landmarks=lmk[ok],
                        subject=refined[ok])

    n_before, n_after = len(set(subj)), len(set(refined[ok]))
    sizes = np.array([counts[s] for s in set(refined[ok])])
    print(f"\n{n_before} identities -> {n_after} refined subjects "
          f"({ok.sum()} of {len(keys)} images kept)")
    print(f"    images per subject: median {np.median(sizes):.0f}  "
          f"mean {sizes.mean():.1f}  max {sizes.max()}")
    for k in (2, 4):
        print(f"    subjects with >= {k} images: {(sizes >= k).sum()}")
    print(f"    -> {out}")


if __name__ == "__main__":
    main()
