"""Run every check in dependency order. This is what CI will call (story A7)."""

import subprocess, sys, pathlib, time

HERE = pathlib.Path(__file__).resolve().parent
# Ordered cheapest-first and by dependency: a broken FLAME loader should fail
# before anything spends GPU time on the renderer.
SUITES = [
    ("flame loads and deforms", "smoke_flame.py", []),
    ("torch port parity + gradients", "test_flame_torch.py", []),
    ("rasteriser", "test_render.py", []),
    ("encoder / params interface", "test_interface.py", []),
    ("albedo / texture space", "test_albedo.py", []),
    ("mediapipe detector", "test_detect.py", []),
    ("paired-view augmentation", "test_augment.py", []),
    ("chain inverts", "test_fit_synthetic.py", []),
    ("overfits a batch", "train_overfit.py", ["300", "3e-4"]),   # short run for CI
]

if __name__ == "__main__":
    only = sys.argv[1:] or None
    rows, failed = [], 0
    for label, script, args in SUITES:
        if only and not any(o in script for o in only):
            continue
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, str(HERE / script), *args],
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
