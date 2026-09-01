#!/usr/bin/env bash
# Run the CUDA rasteriser against every fixture case. PASS/FAIL, no interpretation
# needed.
#
#   bash cuda/test_raster.sh
#
# Regenerate the fixtures first, from Windows:
#   python tests/fixtures/dump_raster_fixture.py
#
# Exit code is 0 only if every case matches the PyTorch reference EXACTLY.
#
# KEEP THIS FILE LF-ONLY. Rewriting it from a Windows editor can introduce CRLF,
# and bash then reports `$'\r': command not found` on every line.

set -u
cd "$(dirname "$0")"
export PATH=/usr/local/cuda/bin:$PATH

FIX="${1:-../out/raster_fixture}"

if ! command -v nvcc >/dev/null; then
    echo "nvcc not found. Try: export PATH=/usr/local/cuda/bin:\$PATH"
    exit 2
fi
if [ ! -d "$FIX" ]; then
    echo "no fixtures at $FIX -- run: python tests/fixtures/dump_raster_fixture.py"
    exit 2
fi

echo "building..."
# On Windows, CUB pulls in CCCL, which refuses to compile against MSVC's
# traditional preprocessor; -lineinfo is for Nsight source correlation.
FLAGS="-O3 -std=c++17 -arch=sm_120 -lineinfo"  # CUB requires C++17
EXE=./raster
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*)
    FLAGS="$FLAGS -Xcompiler /Zc:preprocessor"
    # nvcc writes raster.exe here, and a stale ELF `raster` from a WSL
    # build shadows it -- bash picks that up and dies on Exec format error.
    EXE=./raster.exe ;;
esac

if ! nvcc $FLAGS -o "$EXE" raster.cu 2>build.log; then
    echo "BUILD FAILED"; grep -E "error" build.log | head -20; exit 1
fi

pass=0; fail=0; failed=""
printf '\n%-12s %8s %8s   %s\n' CASE KERNEL TOTAL RESULT
printf -- '----------------------------------------------------------------\n'
for dir in "$FIX"/*/; do
    name=$(basename "$dir")
    [ -f "$dir/meta.json" ] || continue
    out=$("$EXE" "$dir" 2>&1)
    # Read the per-phase table. `kern` is the rasteriser alone; `ms` is the whole
    # pipeline including keys, sort, fill and unpack. Optimising the first at the
    # expense of the second looks like a win and is not one, so both are shown.
    ms=$(echo "$out"   | awk '$1=="TOTAL"        {print $2}')
    kern=$(echo "$out" | awk '$1=="assign_faces" {print $2}')
    if echo "$out" | grep -q "EXACT MATCH"; then
        printf '%-12s %8s %8s   PASS\n' "$name" "${kern:-?}" "${ms:-?}"
        pass=$((pass+1))
    else
        detail=$(echo "$out" | grep -E "^mismatched" | head -1)
        printf '%-12s %8s %8s   FAIL  %s\n' "$name" "${kern:-?}" "${ms:-?}" "$detail"
        fail=$((fail+1)); failed="$failed $name"
    fi
done

printf -- '----------------------------------------------------------------\n'
if [ "$fail" -eq 0 ] && [ "$pass" -gt 0 ]; then
    echo "$pass/$pass PASS      (times in ms)"
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
  Anything else is logic. Re-read face3d/render/render.py::_assign_faces -- that code
  is the definition, not the comments in raster.cu.
EOF
exit 1
