#!/usr/bin/env bash
# Run the CUDA rasteriser against every fixture case. PASS/FAIL, no interpretation
# needed.
#
#   bash cuda/test_raster.sh
#
# Regenerate the fixtures first, from Windows:
#   python scripts/dump_raster_fixture.py
#
# Exit code is 0 only if every case matches the PyTorch reference EXACTLY.

set -u
cd "$(dirname "$0")"
export PATH=/usr/local/cuda/bin:$PATH

FIX="${1:-../out/raster_fixture}"

if ! command -v nvcc >/dev/null; then
    echo "nvcc not found. Try: export PATH=/usr/local/cuda/bin:\$PATH"
    exit 2
fi
if [ ! -d "$FIX" ]; then
    echo "no fixtures at $FIX -- run: python scripts/dump_raster_fixture.py"
    exit 2
fi

echo "building..."
if ! nvcc -O3 -arch=sm_120 -o raster raster.cu 2>build.log; then
    echo "BUILD FAILED"; grep -E "error" build.log | head -20; exit 1
fi

pass=0; fail=0; failed=""
printf '\n%-12s %10s  %s\n' CASE TIME RESULT
printf -- '---------------------------------------------------------------\n'
for dir in "$FIX"/*/; do
    name=$(basename "$dir")
    [ -f "$dir/meta.json" ] || continue
    out=$(./raster "$dir" 2>&1)
    ms=$(echo "$out" | grep -oE '[0-9.]+ ms / call' | head -1 | cut -d' ' -f1)
    if echo "$out" | grep -q "EXACT MATCH"; then
        printf '%-12s %8s ms  PASS\n' "$name" "${ms:-?}"
        pass=$((pass+1))
    else
        detail=$(echo "$out" | grep -E "^mismatched" | head -1)
        printf '%-12s %8s ms  FAIL   %s\n' "$name" "${ms:-?}" "$detail"
        fail=$((fail+1)); failed="$failed $name"
    fi
done

printf -- '---------------------------------------------------------------\n'
if [ "$fail" -eq 0 ] && [ "$pass" -gt 0 ]; then
    echo "$pass/$pass PASS"
    exit 0
fi
echo "$pass passed, $fail FAILED:$failed"
cat <<'EOF'

  missed    a covered pixel left empty   -> bounding box, or `>` where the
                                            reference uses `>=`
  spurious  an empty pixel filled        -> inside test too loose, or you are
                                            filling the bbox without testing
  swapped   right pixel, wrong triangle  -> depth compare or tie-break

  A few `swapped` between triangles at nearly equal depth is float rounding.
  Anything else is logic. Re-read face3d/render.py::_assign_faces -- that code
  is the definition, not the comments in raster.cu.
EOF
exit 1
