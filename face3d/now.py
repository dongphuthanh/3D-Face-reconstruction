"""NoW benchmark harness — story A4.

The metric itself is the official `now_evaluation` code, run through the
Dockerfile it ships. Reimplementing scan-to-mesh distance would produce numbers
that are not comparable to published ones, which defeats the purpose.

What this module does is everything around it: predict, write predictions in the
layout the metric expects, emit the 7 alignment landmarks, and invoke the image.

The 7 landmarks
---------------
NoW rigidly aligns each prediction to the scan using 7 corresponding points, and
ships only a *picture* of where they are — no FLAME indices. They are recovered
here from the MediaPipe landmark embedding, using canonical FaceMesh indices,
and validated against the ground-truth .pp files by Procrustes:

    correct ordering   7.44 mm RMS   (FLAME mean face vs 20 real faces)
    mirrored ordering 19.00 mm RMS
    random ordering   35.67 mm RMS

The 7.44 mm is identity variation, not correspondence error. See
scripts/validate_now_landmarks.py.
"""

import os
import pathlib
import re
import shutil
import subprocess

import numpy as np
import torch

# Slots into the 105-point MediaPipe embedding, in the order NoW expects:
# right eye outer, right eye inner, left eye inner, left eye outer,
# nose tip, mouth right, mouth left.
NOW_LMK_SLOTS = [37, 38, 22, 21, 57, 72, 90]

DOCKER_CANDIDATES = [
    r"C:\Program Files\Docker\Docker\resources\bin\docker.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin\docker.exe"),
    "/usr/bin/docker",
]


def find_docker():
    """Docker Desktop installs per-user and does not always land on PATH."""
    p = shutil.which("docker")
    if p:
        return p
    for c in DOCKER_CANDIDATES:
        if pathlib.Path(c).exists():
            return c
    return None


def load_pp(path):
    """MeshLab picked-points file -> (N,3)."""
    txt = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    return np.array([[float(x), float(y), float(z)] for x, y, z in re.findall(
        r'x="([-\d.eE]+)"\s+y="([-\d.eE]+)"\s+z="([-\d.eE]+)"', txt)], dtype=np.float32)


def image_list(root, split="validation"):
    """Relative image paths for a split, as 'subject/category/IMG.jpg'."""
    f = pathlib.Path(root) / f"imagepaths{split}.txt"
    return [l.strip() for l in f.read_text().splitlines() if l.strip()]


def write_prediction(pred_root, rel_image, verts, faces, lmk7):
    """Write one prediction in the layout compute_error.py globs for.

    `<pred_root>/<subject>/<category>/<stem>.obj` plus a matching `.npy` of
    exactly 7 landmarks — the metric skips any prediction missing either.
    """
    parts = pathlib.Path(rel_image).parts
    subject, category, stem = parts[-3], parts[-2], pathlib.Path(parts[-1]).stem
    d = pathlib.Path(pred_root) / subject / category
    d.mkdir(parents=True, exist_ok=True)

    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64) + 1
    with open(d / f"{stem}.obj", "w") as fh:
        fh.write("\n".join(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v))
        fh.write("\n")
        fh.write("\n".join(f"f {a} {b} {c}" for a, b, c in f))
        fh.write("\n")
    np.save(d / f"{stem}.npy", np.asarray(lmk7, dtype=np.float32).reshape(7, 3))


def landmarks_7(embedding, verts, faces):
    """(B,V,3) -> (B,7,3) in NoW's landmark order."""
    return embedding.positions(verts, faces)[:, NOW_LMK_SLOTS]


def run_docker_eval(dataset_root, pred_root, image="noweval", nproc=None,
                    docker_bin=None, timeout=7200, imgs_list=None, extra=()):
    """Invoke the official metric. Returns (returncode, stdout+stderr).

    Note the container writes results into the predictions mount, not the
    dataset mount — the dataset is only ever read.
    """
    docker_bin = docker_bin or find_docker()
    if docker_bin is None:
        raise RuntimeError("docker not found; see face3d.now.DOCKER_CANDIDATES")

    ds = pathlib.Path(dataset_root).resolve()
    pr = pathlib.Path(pred_root).resolve()
    pr.mkdir(parents=True, exist_ok=True)

    cmd = [docker_bin, "run", "--ipc", "host", "--rm",
           "-v", f"{ds}:/dataset", "-v", f"{pr}:/preds", image]
    if nproc:
        cmd += ["--nproc", str(nproc)]
    if imgs_list:
        # Path as seen inside the container. Without this the metric walks the
        # full 352-image split and skips whatever has no prediction.
        cmd += ["--imgs_list", f"/preds/{pathlib.Path(imgs_list).name}"]
    cmd += list(extra)

    env = dict(os.environ)
    # The credential helper lives beside docker.exe; without it on PATH the
    # daemon call fails with a confusing "error getting credentials".
    env["PATH"] = str(pathlib.Path(docker_bin).parent) + os.pathsep + env.get("PATH", "")
    # errors="replace": the container emits bytes the Windows console codec
    # cannot decode, and the default strict decoding kills the reader thread
    # after the evaluation has already succeeded.
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env,
                       encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")
