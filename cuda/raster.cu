// A CUDA replacement for face3d/render.py::_assign_faces.
//
//   export PATH=/usr/local/cuda/bin:$PATH
//   nvcc -O3 -arch=sm_120 -o raster raster.cu
//   ./raster "/mnt/c/Users/ADMIN/Documents/3D Face Project/out/raster_fixture"
//
// -arch=sm_120 because the 5070 is Blackwell. Build for the wrong arch and it
// still compiles, then fails at launch with "no kernel image is available".
//
// WHAT THIS IS FOR. The PyTorch version cannot scatter without materialising
// candidates, so it builds a (B, F, K*K, 3) tensor per size bucket. At B=8,
// 224px that is 6.69 M candidate (face, pixel) pairs tested against 0.46 M
// pixels of actual triangle area -- 14.4x more work than the geometry needs --
// and 196 MB of peak memory. The median triangle spans 2.4 px while its whole
// bucket pays for K*K. A kernel just loops the bounding box.
//
// It is 94% of rasterize (15.3 of 16.3 ms) and rasterize is ~25% of an encoder
// training step, against a 48.5 ms ResNet-50 fwd+bwd.
//
// WHY THIS ONE IS WORTH DOING FIRST: it is the only part of the rasteriser that
// is not differentiable. It runs under no_grad and returns integer ids;
// rasterize() recomputes barycentrics from those ids in PyTorch, and that is
// where gradients come from. So there is NO BACKWARD KERNEL to write -- the
// hard half of a differentiable rasteriser -- and autograd keeps working.
//
// THE RULES, from the reference implementation. Match them exactly; ids must be
// identical, not close.
//   - A pixel belongs to the triangle with the smallest depth at its CENTRE,
//     (x + 0.5, y + 0.5).
//   - Depth: SMALLER is nearer.
//   - Ties break to the LOWER face index.
//   - -1 where no triangle covers the pixel.
//   - Barycentrics >= 0 on all three edges counts as inside (the reference
//     tests w >= 0, so it keeps exact-edge pixels).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define TILE_DIM 16;
#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); } } while (0)

template <typename T>
static std::vector<T> slurp(const char *dir, const char *name) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<T> v(n / sizeof(T));
    if (fread(v.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    return v;
}

#include "raster_kernel.cuh"

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : ".";

    // meta.json is small and regular; parse the four numbers we need rather
    // than pulling in a JSON library.
    char path[1024]; snprintf(path, sizeof(path), "%s/meta.json", dir);
    FILE *mf = fopen(path, "rb");
    if (!mf) { fprintf(stderr, "no meta.json in %s\n", dir); return 1; }
    char js[4096] = {0}; fread(js, 1, sizeof(js) - 1, mf); fclose(mf);
    auto grab = [&](const char *k) {
        const char *p = strstr(js, k); if (!p) { fprintf(stderr, "missing %s\n", k); exit(1); }
        p = strchr(p, ':'); return atoi(p + 1);
    };
    int B = grab("\"batch\""), V = grab("\"verts\""), F = grab("\"faces\"");
    int H = grab("\"height\""), W = grab("\"width\"");

    auto h_px = slurp<float>(dir, "verts_px.f32");
    auto h_z  = slurp<float>(dir, "depth.f32");
    auto h_f  = slurp<int>(dir, "faces.i32");
    auto h_ref= slurp<int>(dir, "ref_fid.i32");
    printf("B=%d V=%d F=%d %dx%d\n", B, V, F, H, W);

    float zmin = h_z[0], zmax = h_z[0];
    for (float z : h_z) { zmin = z < zmin ? z : zmin; zmax = z > zmax ? z : zmax; }
    // float32 throughout, because the reference is: depth is a float32
    // tensor there, so zmin/zmax/scale all are too. Computing this in
    // double would quantise depth differently and break exact ties.
    float scale = (float)(1 << 30) / (zmax - zmin + 1e-12f);
    unsigned long long BIG = (unsigned long long)((1LL << 30) * (long long)F + F);

    float *d_px, *d_z; int *d_f, *d_fid; unsigned long long *d_buf; unsigned *key, *key_out; int *val;
    long long npix = (long long)B * H * W;
    CHECK(cudaMalloc(&key, B * F * sizeof(unsigned)));
    CHECK(cudaMalloc(&key_out, B * F * sizeof(unsigned)));
    CHECK(cudaMalloc(&val, B * F * sizeof(int)));
    CHECK(cudaMalloc(&d_px, h_px.size() * 4));
    CHECK(cudaMalloc(&d_z,  h_z.size() * 4));
    CHECK(cudaMalloc(&d_f,  h_f.size() * 4));
    CHECK(cudaMalloc(&d_buf, npix * 8));
    CHECK(cudaMalloc(&d_fid, npix * 4));
    CHECK(cudaMemcpy(d_px, h_px.data(), h_px.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_z,  h_z.data(),  h_z.size() * 4,  cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_f,  h_f.data(),  h_f.size() * 4,  cudaMemcpyHostToDevice));

    int *d_order;
    CHECK(cudaMalloc(&d_order, (size_t)B * F * sizeof(int)));
    std::vector<int> h_order(B * F);
    for (int i = 0; i < B * F; i++) h_order[i] = i;
    CHECK(cudaMemcpy(d_order, h_order.data(), h_order.size() * 4, cudaMemcpyHostToDevice));
    // One lambda per phase, so each can be timed on its own. When you add the
    // keys kernel and the sort, give them lambdas here too and put them in
    // `run` -- and in the timing below. A sort you do not measure is a sort
    // you cannot tell is worth it.
    // A bbox area can never exceed H*W, so that many bits always suffice.
    // Radix sort costs one pass per 4-8 bits, so telling CUB the real range
    // instead of the default 32 removes half the passes. Deriving it beats
    // hard-coding: too few bits sorts only the low ones and silently gives a
    // partial order -- harmless for correctness, since min is commutative,
    // but it quietly wastes the whole point of sorting.
    int end_bit = 0;
    while ((1u << end_bit) <= (unsigned)(H * W) && end_bit < 32) end_bit++;
    printf("radix bits: %d (H*W = %d)\n", end_bit, H * W);

    void  *d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
                                key, key_out,
                                val, d_order,
                                (int)(B * F), 0, end_bit);
    CHECK(cudaMalloc(&d_temp, temp_bytes));
    long long nthreads = (long long)B * F;
    auto do_keys = [&]() { keys<<<(int) ((nthreads + 255) / 256),256>>>(d_px, d_f, key, val, B, V, F, H, W);};
    auto do_sort = [&]() {
        size_t bytes = temp_bytes;                        // CUB may modify it; pass a copy
        cub::DeviceRadixSort::SortPairs(d_temp, bytes, key, key_out,
                                        val, d_order, (int)(B * F), 0, end_bit);
    };
    auto do_fill   = [&]() { fill<<<(int)((npix + 255) / 256), 256>>>(d_buf, npix, BIG); };
    auto do_assign = [&]() { assign_faces<<<(int)((nthreads + 255) / 256), 256>>>(
                                 d_px, d_z, d_f, d_buf, B, V, F, H, W, zmin, scale,
                                 d_order); };
    auto do_unpack = [&]() { unpack<<<(int)((npix + 255) / 256), 256>>>(
                                 d_buf, d_fid, npix, BIG, F); };
    auto run = [&]() { do_keys(), do_sort(), do_fill(); do_assign(); do_unpack(); };

    run();
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    std::vector<int> got(npix);
    CHECK(cudaMemcpy(got.data(), d_fid, npix * 4, cudaMemcpyDeviceToHost));

    // Split the mismatches by KIND. They point at different mistakes:
    //   missed   a covered pixel left empty      -> bounding box or inside test
    //   spurious an empty pixel filled           -> inside test too loose
    //   swapped  right pixel, wrong triangle     -> depth compare or tie-break
    // A few `swapped` where the two triangles sit at nearly the same depth is
    // float rounding, not a bug. Anything else is logic.
    long long wrong = 0, covered = 0, missed = 0, spurious = 0, swapped = 0;
    for (long long i = 0; i < npix; i++) {
        if (h_ref[i] >= 0) covered++;
        if (got[i] == h_ref[i]) continue;
        wrong++;
        if (got[i] < 0) missed++;
        else if (h_ref[i] < 0) spurious++;
        else swapped++;
    }
    printf("reference covers %lld of %lld pixels\n", covered, npix);
    printf("mismatched %lld   missed %lld  spurious %lld  swapped %lld\n",
           wrong, missed, spurious, swapped);
    printf(wrong == 0 ? "EXACT MATCH\n" : "NOT MATCHING YET\n");

    cudaEvent_t a, b2; cudaEventCreate(&a); cudaEventCreate(&b2);
    auto bench = [&](const char *label, auto &&fn) {
        for (int i = 0; i < 5; i++) fn();
        CHECK(cudaDeviceSynchronize());
        cudaEventRecord(a);
        for (int i = 0; i < 50; i++) fn();
        cudaEventRecord(b2); CHECK(cudaEventSynchronize(b2));
        float ms; cudaEventElapsedTime(&ms, a, b2);
        printf("  %-16s %7.3f ms\n", label, ms / 50);
        return ms / 50;
    };
    printf("\n");
    // assign_faces is timed as (fill + assign) minus fill, because it needs a
    // freshly initialised buffer to do representative work: run it twice over
    // an already-populated buffer and most of its atomics lose immediately.
    float t_keys   = bench("keys", do_keys);
    float t_sort   = bench("sort", do_sort);
    float t_fill   = bench("fill", do_fill);
    float t_assign = bench("assign_faces", [&]{ do_fill(); do_assign(); }) - t_fill;
    float t_unpack = bench("unpack", do_unpack);
    float total    = t_keys + t_sort + t_fill + t_assign + t_unpack;
    printf("  %-16s %7.3f ms   assign_faces is %.0f%% of it\n",
           "TOTAL", total, 100.0 * t_assign / total);
    printf("  PyTorch _assign_faces on `basic`: 15.3 ms  ->  %.0fx\n",
           15.3 / total);

    return 0;
}
