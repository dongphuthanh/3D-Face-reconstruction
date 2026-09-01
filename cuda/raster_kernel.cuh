#pragma once

#include <cuda_runtime.h>

__device__ __forceinline__ unsigned long long pack(float z, int fid,
                                                   float zmin, float scale,
                                                   int nfaces) {
    long long q = (long long)((z - zmin) * scale);
    q = q < 0 ? 0 : (q > (1LL << 30) ? (1LL << 30) : q);
    return (unsigned long long)(q * (long long)nfaces + fid);
}

__global__ void keys(const float *__restrict__ verts_px,
                     const int   *__restrict__ faces,
                     unsigned *__restrict__ key, int *__restrict__ val,
                     int B, int V, int F, int H, int W) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)B * F) return;
    int b = (int)(i / F), f = (int)(i % F);
    int vert0 = faces[f * 3], vert1 = faces[f * 3 + 1], vert2 = faces[f * 3 + 2];
    float x0= verts_px[(b * V + vert0) * 2], y0 = verts_px[(b * V + vert0) * 2 + 1];
    float x1= verts_px[(b * V + vert1) * 2], y1 = verts_px[(b * V + vert1) * 2 + 1];
    float x2= verts_px[(b * V + vert2) * 2], y2 = verts_px[(b * V + vert2) * 2 + 1];
    
    int xlo  = max(0,   (int)floorf(fminf(fminf(x0, x1), x2)));
    int xhi  = min(W-1, (int)floorf(fmaxf(fmaxf(x0, x1), x2)));

    int ylo  = max(0,   (int)floorf(fminf(fminf(y0, y1), y2)));
    int yhi  = min(H-1, (int)floorf(fmaxf(fmaxf(y0, y1), y2)));

    int bw = xhi - xlo + 1, bh = yhi - ylo + 1;
    key[i] = (bw > 0 && bh > 0) ? (unsigned)(bw * bh) : 0u;
    val[i] = (int)i;
}


__global__ void assign_faces(const float *__restrict__ verts_px,  // (B,V,2)
                             const float *__restrict__ depth,     // (B,V)
                             const int *__restrict__ faces,       // (F,3)
                             unsigned long long *__restrict__ buf,// (B,H,W)
                             int B, int V, int F, int H, int W,
                             float zmin, float scale,
                            const int *__restrict__ order) {
    long long slot = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (slot >= (long long)B * F) return;
    long long tid = order[slot];
    int b = (int)(tid / F);
    int f = (int)(tid % F);

    int vert0 = faces[f * 3], vert1 = faces[f * 3 + 1], vert2 = faces[f * 3 + 2];
    float x0= verts_px[(b * V + vert0) * 2], y0 = verts_px[(b * V + vert0) * 2 + 1], d0 = depth[b * V + vert0];
    float x1= verts_px[(b * V + vert1) * 2], y1 = verts_px[(b * V + vert1) * 2 + 1], d1 = depth[b * V + vert1];
    float x2= verts_px[(b * V + vert2) * 2], y2 = verts_px[(b * V + vert2) * 2 + 1], d2 = depth[b * V + vert2];
    float d = ((y1 - y2) * (x0 - x2)) + ((x2 - x1) * (y0 - y2));
    if (fabsf(d) < 1e-12f) d = 1e-12f;
    float _d = 1.0f / d;

    int xlo  = max(0,   (int)floorf(fminf(fminf(x0, x1), x2)));
    int xhi  = min(W-1, (int)floorf(fmaxf(fmaxf(x0, x1), x2)));

    int ylo  = max(0,   (int)floorf(fminf(fminf(y0, y1), y2)));
    int yhi  = min(H-1, (int)floorf(fmaxf(fmaxf(y0, y1), y2)));

    for (int y = ylo; y <= yhi; y++) {
        float py = y + 0.5f;
        for (int x = xlo; x <= xhi; x++) {
            float px = x + 0.5f;
            float w0 = ((y1-y2)*(px-x2) + (x2-x1)*(py-y2)) * _d;
            float w1 = ((y2-y0)*(px-x2) + (x0-x2)*(py-y2)) * _d;
            float w2 = 1 - w0 - w1;
            if (w0 < 0 || w1 < 0 || w2 < 0) {
                continue;
            };
            float z = w0 * d0 + w1 * d1 + w2 * d2;
            atomicMin(&buf[(b*H + y)*W + x], pack(z, f, zmin, scale, F));
        }
    }


}


__global__ void fill(unsigned long long *buf, long long n, unsigned long long v) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) buf[i] = v;
}

__global__ void unpack(const unsigned long long *buf, int *fid, long long n,
                       unsigned long long big, int nfaces) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) fid[i] = (buf[i] == big) ? -1 : (int)(buf[i] % (unsigned long long)nfaces);
}
