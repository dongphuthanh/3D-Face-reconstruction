"""How recoverable is identity in a corpus, before spending GPU on it?

The identity loss scores geometry with InceptionResnetV1/VGGFace2, a network
trained on real photographs. DigiFace is synthetic 112px CG, so the term could
in principle have been near-vacuous on ~97% of our batches. It is not:

    corpus     nature                  within  between  ratio    AUC
    digiface   synthetic 112px CG      0.3225   0.7620   2.36   0.971
    celeba     real, aligned 178x218   0.3320   0.9814   2.96   0.974
    celeba16   same, margin 1.6        0.3506   0.9765   2.78   0.976
    arc2face   real, restored 448px    0.2714   0.9611   3.54   0.998
    NoW        real photographs        0.2020   0.9192   4.55   0.999

    within  = mean cosine distance between images of ONE subject
    between = mean cosine distance between images of DIFFERENT subjects
    ratio   = between / within;  AUC = verification AUC, 0.5 is chance

DigiFace identities are cleanly separable (AUC 0.971), so --w-id 0.2 is doing
real work -- which independently corroborates the identity ratio jumping
0.92 -> 1.15 when that term was added.

Read the two columns separately; they say different things.

BETWEEN is identity diversity, and it is where DigiFace is genuinely deficient:
0.76 against 0.92-0.98 for every real corpus. Its 100k identities are less
distinct FROM EACH OTHER than real people are, because a parametric generator
samples a narrower region of face space. That caps the between-subject shape
spread any encoder trained on it can reach, and is a candidate explanation for
the identity ratio plateauing at 1.15-1.18. CelebA is the widest measured.

WITHIN is how much one subject's images vary, and it is NOT simply "worse when
higher". NoW's 0.20 is one capture session. CelebA's 0.33 is the same celebrity
across years, makeup and lighting -- hard variation the shape must stay
constant across, which is what the swap loss exists to exploit. The catch is
that it also spans age and weight, which genuinely change 3D shape, so a
CelebA identity is NOT one fixed geometry the way a DigiFace identity is.

Margin sensitivity is small: CelebA at 1.15 and 1.6 differ by 6% in ratio and
0.002 in AUC. Corpora ingested at different margins (arc2face 1.15, digiface
1.6) are therefore still comparable here.

Use this as an acceptance test on a candidate corpus before training on it.
Note what it cannot do: Arc2Face scores well here and still failed to transfer
to NoW (in-domain ratio 1.06, NoW 0.55-0.63, no shrinkage dip), because blind
face restoration makes identity MORE recoverable than reality. A good score
here is necessary, not sufficient.

Corpora are cropped identically (MediaPipe landmarks) and capped at the same
subject/image counts. NoW images are drawn round-robin across its four capture
categories: taking the first N of a flat sorted list draws them all from one
session and understates within-subject scatter (ratio 5.42 rather than 4.55).

    python scripts/eval/diag_corpus_identity.py [digiface] [celeba] [arc2face] [NoW]
"""
import pathlib, sys, warnings
import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from facenet_pytorch import InceptionResnetV1
from face3d.learn.detect import FaceDetector, crop_square

DEV = "cuda"
N_SUBJ, N_IMG = 100, 6
rng = np.random.default_rng(0)

net = InceptionResnetV1(pretrained="vggface2").eval().to(DEV)


@torch.no_grad()
def embed(crops):
    """list of HxWx3 uint8 -> (N,512) L2-normalised."""
    out = []
    for i in range(0, len(crops), 32):
        x = torch.from_numpy(np.stack(crops[i:i + 32])).to(DEV)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(160, 160), mode="bilinear",
                                            align_corners=False)
        e = net(x * 2.0 - 1.0)
        out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
    return np.concatenate(out)


def stats(emb, labels):
    """between/within cosine distance and verification AUC."""
    d = 1.0 - emb @ emb.T
    n = len(labels)
    iu = np.triu_indices(n, k=1)
    same = labels[iu[0]] == labels[iu[1]]
    dist = d[iu]
    within, between = dist[same], dist[~same]
    # AUC = P(a same-pair distance < a different-pair distance), via rank sum
    order = np.argsort(np.concatenate([within, between]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(len(order))
    r_same = ranks[:len(within)].sum()
    auc = 1.0 - (r_same - len(within) * (len(within) - 1) / 2) / (len(within) * len(between))
    return within.mean(), between.mean(), between.mean() / within.mean(), auc, len(within), len(between)


def from_ingest(name):
    """DigiFace / Arc2Face: crops are already cached on disk."""
    root = ROOT / "data" / name
    z = np.load(root / "landmarks_224.npz")
    keys = np.array([str(k) for k in z["keys"]])
    subj = np.array([str(s) for s in z["subject"]])
    groups = {}
    for i, s in enumerate(subj):
        groups.setdefault(s, []).append(i)
    ids = sorted(s for s, v in groups.items() if len(v) >= 2)
    pick = rng.permutation(len(ids))[:N_SUBJ]
    crops, labels = [], []
    for j in pick:
        idx = groups[ids[j]][:N_IMG]
        for i in idx:
            # keys carry no extension; the crops are written as .jpg
            p = root / "crops" / (keys[i] + ".jpg")
            if not p.exists():
                continue
            crops.append(np.asarray(Image.open(p).convert("RGB").resize((224, 224))))
            labels.append(j)
    return crops, np.array(labels)


def from_celeba(margin=1.6, shard=0):
    """CelebA: real, identity-labelled, un-restored. Streamed from the parquet
    mirror that carries celeb_id (the official identity file is request-gated).

    Shards are sorted by celeb_id, so one covers enough identities. The images
    are the ALIGNED 178x218 crops, i.e. already tight on the face like
    Arc2Face -- cropping them at 1.6 pads with black, so the margin is a
    parameter here and the caller runs both.
    """
    import io
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "flwrlabs/celeba",
        f"img_align+identity+attr/train-{shard:05d}-of-00019.parquet",
        repo_type="dataset", cache_dir=str(ROOT / "data" / "_hfcache"))
    tab = pq.ParquetFile(path).read(columns=["image", "celeb_id"])
    ids = tab.column("celeb_id").to_pylist()
    imgs = tab.column("image").to_pylist()

    groups = {}
    for i, cid in enumerate(ids):
        groups.setdefault(cid, []).append(i)
    usable = [c for c, v in groups.items() if len(v) >= 2][:N_SUBJ]

    det = FaceDetector()
    crops, labels, miss = [], [], 0
    for j, cid in enumerate(usable):
        for i in groups[cid][:N_IMG]:
            rec = imgs[i]
            raw = rec["bytes"] if isinstance(rec, dict) else rec
            im = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
            res = det.detect(im)
            if not res:
                miss += 1
                continue
            c, _ = crop_square(im, res["norm"], size=224, margin=margin)
            crops.append(c)
            labels.append(j)
    det.close()
    print(f"  (CelebA margin {margin}: {len(usable)} ids, {miss} undetected)")
    return crops, np.array(labels)


def from_now():
    """NoW: full photographs, cropped with the same detector and margin."""
    base = ROOT / "NoW_Dataset" / "final_release_version" / "iphone_pictures"
    det = FaceDetector()
    subs = sorted(p for p in base.iterdir() if p.is_dir())[:N_SUBJ]
    crops, labels, miss = [], [], 0
    for j, s in enumerate(subs):
        # Round-robin across the four capture categories (neutral, expressions,
        # occlusions, selfie). Taking the first N of a flat sorted list draws
        # them all from one category -- same session, same light, same pose
        # family -- which understates within-subject scatter and flatters the
        # ceiling this corpus is here to provide.
        cats = sorted(p for p in s.iterdir() if p.is_dir())
        pools = [sorted(c.rglob("*.jpg")) for c in cats] or [sorted(s.rglob("*.jpg"))]
        imgs, r = [], 0
        while len(imgs) < N_IMG and any(len(p) > r for p in pools):
            for p in pools:
                if r < len(p) and len(imgs) < N_IMG:
                    imgs.append(p[r])
            r += 1
        for p in imgs:
            im = np.asarray(Image.open(p).convert("RGB"))
            res = det.detect(im)
            if not res:
                miss += 1
                continue
            c, _ = crop_square(im, res["norm"], size=224, margin=1.6)
            crops.append(c)
            labels.append(j)
    det.close()
    print(f"  (NoW: {miss} images with no detected face, skipped)")
    return crops, np.array(labels)


print(f"{N_SUBJ} subjects x up to {N_IMG} images, facenet InceptionResnetV1/vggface2\n")
print(f"{'corpus':12s} {'nature':22s} {'within':>8s} {'between':>8s} "
      f"{'ratio':>7s} {'AUC':>7s}   pairs")
rows = []
for name, nature, loader in (
        ("digiface", "synthetic 112px CG", lambda: from_ingest("digiface")),
        ("arc2face", "real, restored 448px", lambda: from_ingest("arc2face")),
        # celeba streams from parquet; celeba_disk reads what ingest_celeba.py
        # actually wrote. They should agree -- if they do not, the ingest has
        # mismatched crops to subject labels, which training would not notice.
        ("celeba_disk", "ingested crops, m1.6", lambda: from_ingest("celeba")),
        ("celeba", "real, aligned 178x218", lambda: from_celeba(1.15)),
        ("celeba16", "same, margin 1.6", lambda: from_celeba(1.6)),
        ("NoW", "real photographs", from_now)):
    if len(sys.argv) > 1 and name not in sys.argv[1:]:
        continue
    try:
        crops, labels = loader()
    except Exception as e:                       # a missing corpus must not kill the rest
        print(f"{name:12s} {nature:22s}  SKIPPED: {type(e).__name__}: {e}")
        continue
    if len(crops) < 20:
        print(f"{name:12s} {nature:22s}  SKIPPED: only {len(crops)} crops")
        continue
    w, b, r, auc, nw, nb = stats(embed(crops), labels)
    print(f"{name:12s} {nature:22s} {w:8.4f} {b:8.4f} {r:7.3f} {auc:7.3f}   "
          f"{nw} same / {nb} diff")
    rows.append((name, r, auc))

print()
for name, r, auc in rows:
    verdict = ("identity is NOT recoverable -- the id loss is near-vacuous here"
               if auc < 0.75 else
               "weakly separable" if auc < 0.90 else "cleanly separable")
    print(f"  {name:10s} AUC {auc:.3f}  ratio {r:.2f}   {verdict}")
