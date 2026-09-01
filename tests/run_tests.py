"""Run every check in dependency order. This is what CI will call (story A7)."""

import subprocess, sys, pathlib, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Ordered cheapest-first and by dependency: a broken FLAME loader should fail
# before anything spends GPU time on the renderer.
SUITES = [
    ("flame loads and deforms", "tests/smoke_flame.py", []),
    ("torch port parity + gradients", "tests/test_flame_torch.py", []),
    ("rasteriser", "tests/test_render.py", []),
    ("encoder / params interface", "tests/test_interface.py", []),
    ("albedo / texture space", "tests/test_albedo.py", []),
    ("mediapipe detector", "tests/test_detect.py", []),
    ("paired-view augmentation", "tests/test_augment.py", []),
    ("identity loss gradient routing", "tests/test_identity_loss.py", []),
    ("eye/lip closure landmarks", "tests/test_closure_loss.py", []),
    ("gltf export + validator", "tests/test_export.py", []),
    ("chain inverts", "tests/test_fit_synthetic.py", []),
    # not a test file, but its acceptance criterion is one: the loop must
    # still be able to drive a single batch to convergence.
    ("overfits a batch", "scripts/train/train_overfit.py", ["300", "3e-4"]),
]

if __name__ == "__main__":
    only = sys.argv[1:] or None
    rows, failed = [], 0
    for label, script, args in SUITES:
        if only and not any(o in script for o in only):
            continue
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, str(ROOT / script), *args],
                           capture_output=True, text=True)
        dt = time.perf_counter() - t0
        ok = r.returncode == 0
        failed += (not ok)
        rows.append((label, ok, dt))
        print(f"[{'PASS' if ok else 'FAIL'}] {label:<32s} {dt:6.1f}s")
        if not ok:
            print("\n".join("       " + l for l in r.stdout.strip().splitlines()[-15:]))

    print("\n" + "=" * 56)
    print(f"{len(rows) - failed}/{len(rows)} suites passed"
          + ("" if failed else "  —  all green"))
    sys.exit(1 if failed else 0)
