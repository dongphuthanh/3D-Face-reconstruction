// PyTorch binding for the CUDA rasteriser. Built on demand by
// face3d/raster_cuda.py; see there for why it is JIT-compiled rather than
// shipped as a wheel.
//
// The kernels themselves live in raster_kernel.cuh and are shared with the
// standalone harness, so the thing benchmarked in cuda/raster.cu and the thing
// used in training are the same code.
//
// This deliberately does NOT include the bbox-area sort. Measured, the sort
// wins 0.260 -> 0.201 ms on the training configuration and LOSES on 512px
// (0.554 -> 0.598), and it pulls in CUB. Land the plain kernel first, which is
// already ~59x the PyTorch path, then decide whether the sort earns its
// complexity.

#include <torch/extension.h>
#include <cuda_runtime.h>

#include "raster_kernel.cuh"

// verts_px (B,V,2) float32, depth (B,V) float32, faces (F,3) int32 -- all CUDA,
// all contiguous. Returns (B,H,W) int32 face ids, -1 where nothing is covered.
//
// K is absent on purpose. The PyTorch reference needs it to size its candidate
// tensor, and that parameter is the source of three latent bugs there: a
// triangle whose span exceeds K is dropped entirely, one with span exactly 0 is
// dropped by the `span > 0` bucket test, and when K < 16 the bucket loop breaks
// before it reaches the K bucket. A kernel scanning the real bounding box has
// none of those failure modes, so it is not merely faster but strictly more
// correct. That also means its output can DIFFER from the reference on
// pathological input -- which never occurs on FLAME geometry (p99 triangle span
// is 15.8 px against a K of 23) but is worth knowing before trusting a diff.
torch::Tensor assign_faces_cuda(torch::Tensor verts_px,
                                torch::Tensor depth,
                                torch::Tensor faces,
                                int64_t H, int64_t W) {
    TORCH_CHECK(verts_px.is_cuda() && depth.is_cuda() && faces.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(verts_px.scalar_type() == torch::kFloat32, "verts_px must be float32");
    TORCH_CHECK(depth.scalar_type() == torch::kFloat32, "depth must be float32");
    TORCH_CHECK(faces.scalar_type() == torch::kInt32, "faces must be int32");
    TORCH_CHECK(verts_px.dim() == 3 && verts_px.size(2) == 2, "verts_px must be (B,V,2)");
    TORCH_CHECK(depth.dim() == 2, "depth must be (B,V)");
    TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3, "faces must be (F,3)");
    TORCH_CHECK(verts_px.size(0) == depth.size(0) && verts_px.size(1) == depth.size(1),
                "verts_px and depth disagree about B or V");

    verts_px = verts_px.contiguous();
    depth    = depth.contiguous();
    faces    = faces.contiguous();

    const int B = (int)verts_px.size(0);
    const int V = (int)verts_px.size(1);
    const int F = (int)faces.size(0);
    const long long npix = (long long)B * H * W;

    // Depth range over the WHOLE batch, matching the reference: keys from
    // different images have to stay comparable.
    auto mn = depth.min().item<float>();
    auto mx = depth.max().item<float>();
    const float scale = (float)(1 << 30) / (mx - mn + 1e-12f);
    const unsigned long long BIG =
        (unsigned long long)((1LL << 30) * (long long)F + F);

    auto opts_u64 = torch::TensorOptions().dtype(torch::kInt64).device(verts_px.device());
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(verts_px.device());
    auto buf = torch::empty({(long long)npix}, opts_u64);
    auto fid = torch::empty({B, (long long)H, (long long)W}, opts_i32);

    // The identity order. Passing it rather than special-casing keeps one code
    // path in the kernel; the sort can be dropped in later without touching it.
    auto order = torch::arange((long long)B * F, opts_i32);

    auto *d_buf = reinterpret_cast<unsigned long long *>(buf.data_ptr<int64_t>());
    const int threads = 256;
    auto stream = at::cuda::getCurrentCUDAStream();

    fill<<<(int)((npix + threads - 1) / threads), threads, 0, stream>>>(
        d_buf, npix, BIG);
    assign_faces<<<(int)(((long long)B * F + threads - 1) / threads), threads, 0, stream>>>(
        verts_px.data_ptr<float>(), depth.data_ptr<float>(),
        faces.data_ptr<int>(), d_buf, B, V, F, (int)H, (int)W, mn, scale,
        order.data_ptr<int>());
    unpack<<<(int)((npix + threads - 1) / threads), threads, 0, stream>>>(
        d_buf, fid.data_ptr<int>(), npix, BIG, F);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return fid;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("assign_faces", &assign_faces_cuda,
          "z-buffer triangle assignment (CUDA)");
}
