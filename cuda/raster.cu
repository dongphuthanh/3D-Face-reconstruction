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

// Depth and face id packed into one 64-bit key so a single atomicMin resolves
// the winner AND the tie-break together. The Python version already does this
// for scatter_reduce; here it is what the hardware wants anyway.
__device__ __forceinline__ unsigned long long pack(float z, int fid,
                                                   float zmin, float scale,
                                                   int nfaces) {
    long long q = (long long)((z - zmin) * scale);
    q = q < 0 ? 0 : (q > (1LL << 30) ? (1LL << 30) : q);
    return (unsigned long long)(q * (long long)nfaces + fid);
}

__global__ void assign_faces(const float *__restrict__ verts_px,  // (B,V,2)
                             const float *__restrict__ depth,     // (B,V)
                             const int *__restrict__ faces,       // (F,3)
                             unsigned long long *__restrict__ buf,// (B,H,W)
                             int B, int V, int F, int H, int W,
                             float zmin, float scale) {
    long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= (long long)B * F) return;
    int b = (int)(tid / F);
    int f = (int)(tid % F);

    // ------------------------------------------------------------------
    // YOUR KERNEL GOES HERE.
    //
    // 1. Read the three vertex indices from faces[f*3 + {0,1,2}].
    // 2. Fetch their pixel xy from verts_px + b*V*2, and their depth.
    // 3. Bounding box, clamped to [0,W) x [0,H). floor the min, ceil the max.
    // 4. For each pixel centre (x+0.5, y+0.5) in the box, compute barycentrics
    //    the same way _bary does:
    //        d  = (y1-y2)(x0-x2) + (x2-x1)(y0-y2)
    //        w0 = ((y1-y2)(px-x2) + (x2-x1)(py-y2)) / d
    //        w1 = ((y2-y0)(px-x2) + (x0-x2)(py-y2)) / d
    //        w2 = 1 - w0 - w1
    //    The reference guards |d| < 1e-12 by replacing d with 1e-12.
    // 5. Inside if w0,w1,w2 all >= 0. Depth there is w0*z0 + w1*z1 + w2*z2.
    // 6. atomicMin(&buf[(b*H + y)*W + x], pack(z, f, zmin, scale, F));
    //
    // Start with one thread per triangle -- correctness first. Once it matches,
    // the interesting question is what to do about the long tail: the median
    // triangle covers ~6 pixels and the largest covers ~400, so threads finish
    // wildly out of step. That is where the real speedup is, and where it stops
    // being a beginner exercise.
    // ------------------------------------------------------------------
    int vert0 = faces[f * 3], vert1 = faces[f * 3 + 1], vert2 = faces[f * 3 + 2];
    float x0= verts_px[(b * V + vert0) * 2], y0 = verts_px[(b * V + vert0) * 2 + 1], d0 = depth[b * V + vert0];
    float x1= verts_px[(b * V + vert1) * 2], y1 = verts_px[(b * V + vert1) * 2 + 1], d1 = depth[b * V + vert1];
    float x2= verts_px[(b * V + vert2) * 2], y2 = verts_px[(b * V + vert2) * 2 + 1], d2 = depth[b * V + vert2];
    float d = ((y1 - y2) * (x0 - x2)) + ((x2 - x1) * (y0 - y2));

    int xlo  = max(0,   (int)floorf(fminf(fminf(x0, x1), x2)));
    int xhi  = min(W-1, (int)floorf(fmaxf(fmaxf(x0, x1), x2)));

    int ylo  = max(0,   (int)floorf(fminf(fminf(y0, y1), y2)));
    int yhi  = min(H-1, (int)floorf(fmaxf(fmaxf(y0, y1), y2)));

    for (int y = ylo; y <= yhi; y++) {
        float py = y + 0.5;
        for (int x = xlo; x <= xhi; x++) {
            float px = x + 0.5;
            float w0 = ((y1-y2)*(px-x2) + (x2-x1)*(py-y2)) / d;
            float w1 = ((y2-y0)*(px-x2) + (x0-x2)*(py-y2)) / d;
            float w2 = 1 - w0 - w1;
            if (w0 < 0 || w1 < 0 || w2 < 0) {
                continue;
            };
            float z = w0 * d0 + w1 * d1 + w2 * d2;
            atomicMin(&buf[(b*H + y)*W + x], pack(z, f, zmin, scale, F));
        }
    }


}

__global__ void unpack(const unsigned long long *buf, int *fid, long long n,
                       unsigned long long big, int nfaces) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) fid[i] = (buf[i] == big) ? -1 : (int)(buf[i] % (unsigned long long)nfaces);
}

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

    float *d_px, *d_z; int *d_f, *d_fid; unsigned long long *d_buf;
    long long npix = (long long)B * H * W;
    CHECK(cudaMalloc(&d_px, h_px.size() * 4));
    CHECK(cudaMalloc(&d_z,  h_z.size() * 4));
    CHECK(cudaMalloc(&d_f,  h_f.size() * 4));
    CHECK(cudaMalloc(&d_buf, npix * 8));
    CHECK(cudaMalloc(&d_fid, npix * 4));
    CHECK(cudaMemcpy(d_px, h_px.data(), h_px.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_z,  h_z.data(),  h_z.size() * 4,  cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_f,  h_f.data(),  h_f.size() * 4,  cudaMemcpyHostToDevice));

    auto run = [&]() {
        std::vector<unsigned long long> init(npix, BIG);
        CHECK(cudaMemcpy(d_buf, init.data(), npix * 8, cudaMemcpyHostToDevice));
        long long threads = (long long)B * F;
        assign_faces<<<(threads + 255) / 256, 256>>>(d_px, d_z, d_f, d_buf,
                                                     B, V, F, H, W, zmin, scale);
        unpack<<<(npix + 255) / 256, 256>>>(d_buf, d_fid, npix, BIG, F);
    };

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
    for (int i = 0; i < 3; i++) run();
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for (int i = 0; i < 20; i++) run();
    cudaEventRecord(b2); CHECK(cudaEventSynchronize(b2));
    float ms; cudaEventElapsedTime(&ms, a, b2);
    printf("%.3f ms / call   (PyTorch _assign_faces on this data: 15.3 ms)\n", ms / 20);
    return 0;
}
