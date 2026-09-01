"""Ingest a DigiFace-1M subset: identity-grouped faces for the shape swap.

Why this dataset. Every attempt to make the encoder learn identity from
single images has failed -- distance-penalty consistency collapsed the
representation, the DECA-style swap had no leverage through the landmark term,
and the C2 mask changed nothing. The one thing measured to help was more data,
and the one constraint never actually supplied was "these photographs are the
same person". DigiFace provides that: 2,000 subjects at 72 renders each,
varying pose, expression, lighting and accessories.

It is synthetic, so there is a domain gap to real photographs -- and no consent,
privacy or biometric exposure, which is why it clears the project's standing
rule against scraped face data outright.

Licence: Research Use of Data Agreement v1.0, non-commercial. Note it treats
models trained on the data as "Results", which are exempt from the Data
redistribution restrictions -- so the trained encoder may be shipped, unlike
anything derived from DECA.

Only a slice is taken by default: the swap loss needs several images per
identity, not every image of every identity.
"""

import argparse
import collections
import pathlib
import sys
import time
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from face3d.learn.detect import FaceDetector, crop_square, select_embedding_points

ROOT = pathlib.Path(__file__).resolve().parents[2]
DIGI = ROOT / "data" / "digiface"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", nargs="*", default=None,
                    help="one or more part archives; default is every "
                         "subjects_*_imgs.zip present. The 72-image parts give "
                         "10k identities, the 5-image parts another 100k")
    ap.add_argument("--subjects", type=int, default=1500)
    ap.add_argument("--per-subject", type=int, default=8)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out", default=str(DIGI / "crops"))
    ap.add_argument("--verify", action="store_true",
                    help="CRC-check every archive member first (slow on 2.7 GB, "
                         "but catches a truncated download before ingesting)")
    a = ap.parse_args()

    archives = ([pathlib.Path(x) for x in a.archive] if a.archive
                else sorted(DIGI.glob("subjects_*_imgs.zip")))
    archives = [x for x in archives if x.exists()]
    if not archives:
        print(f"SKIP - no archives found in {DIGI}")
        sys.exit(0)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    emb = np.load(ROOT / "mediapipe_landmark_embedding" /
                  "mediapipe_landmark_embedding.npz")
    need = emb["landmark_indices"].astype(int).ravel()

    # Each part is ~2.7 GB over plain HTTP and one dropped mid-stream already.
    # A truncated zip still opens and still lists entries; it fails only when a
    # member near the end is read, which would be an hour into the ingest.
    # Check each central directory up front instead.
    zips, by_subject = {}, collections.defaultdict(list)
    for arc in archives:
        try:
            z = zipfile.ZipFile(arc)
            if a.verify and z.testzip() is not None:
                print(f"SKIP - {arc.name} is corrupt; resume with `curl -C -`")
                sys.exit(0)
        except zipfile.BadZipFile as e:
            print(f"SKIP - {arc.name} is not a readable zip ({e}); the download "
                  f"likely truncated. Resume with `curl -C -`.")
            sys.exit(0)
        zips[arc.name] = z
        for n in z.namelist():
            if n.lower().endswith(".png"):
                # Subject ids repeat across parts, so qualify with the part name.
                by_subject[(arc.name, n.split("/")[0])].append(n)

    subjects = sorted(by_subject)[: a.subjects]
    print(f"=== DigiFace ingest: {len(subjects)} subjects x {a.per_subject} images "
          f"from {len(archives)} archive(s) ===")

    # Reuse any previous ingest so parts can be added incrementally rather than
    # each run silently replacing the last one's landmarks.
    cache = out.parent / f"landmarks_{a.size}.npz"
    have = {}
    if cache.exists():
        zc = np.load(cache)
        have = {str(k): (v, str(s)) for k, v, s in
                zip(zc["keys"], zc["landmarks"], zc["subject"])}
        print(f"    reusing {len(have)} cached records")

    det = FaceDetector(blendshapes=False)
    keys, lmks, sids = [], [], []
    ok = nodet = 0
    t0 = time.time()

    for si, s in enumerate(subjects, 1):
        arc_name, subj_id = s
        z = zips[arc_name]
        part = arc_name.split("_")[1]
        # Sort by the numeric stem so the chosen images are a stable subset
        # rather than whatever order the archive happens to list.
        names = sorted(by_subject[s], key=lambda n: int(pathlib.Path(n).stem))
        taken = 0
        for n in names:
            if taken >= a.per_subject:
                break
            key = f"{part}_{subj_id}_{pathlib.Path(n).stem}"
            if key in have:
                lm, sid = have[key]
                keys.append(key); lmks.append(lm); sids.append(sid)
                taken += 1; ok += 1
                continue
            with z.open(n) as fh:
                im = np.array(Image.open(fh).convert("RGB"), dtype=np.uint8)
            res = det.detect(im)
            if res is None:
                nodet += 1
                continue
            pts = select_embedding_points(res["norm"], need)
            crop, to_ndc = crop_square(im, res["norm"], size=a.size)
            Image.fromarray(crop).save(out / f"{key}.jpg", quality=95)
            keys.append(key)
            lmks.append(to_ndc(pts).astype(np.float32))
            sids.append(f"{part}_{subj_id}")
            taken += 1
            ok += 1
        if si % 100 == 0 or si == len(subjects):
            print(f"  {si:4d}/{len(subjects)} subjects  ok={ok} no-face={nodet}  "
                  f"{time.time() - t0:.0f}s", flush=True)

    det.close()
    if keys:
        # subject_id travels with the landmarks: it is the whole point of this
        # dataset and the swap loss cannot be formed without it.
        np.savez_compressed(cache,
                            keys=np.array(keys), landmarks=np.stack(lmks),
                            subject=np.array(sids))
        print(f"\n  wrote landmarks_{a.size}.npz  ({len(keys)} images, "
              f"{len(set(sids))} identities)")
    if ok + nodet:
        print(f"  detection rate: {ok}/{ok + nodet} = {ok / (ok + nodet) * 100:.1f}%")


if __name__ == "__main__":
    main()
