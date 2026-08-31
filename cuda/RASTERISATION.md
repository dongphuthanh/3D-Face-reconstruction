# How this rasteriser works

The maths behind `face3d/render.py::_assign_faces`, which `cuda/raster.cu`
replaces. Written to be read before writing the kernel.

The reference implementation is the definition. Where this document and the code
disagree, the code is right.

---

## 1. The problem

We have a triangle mesh whose vertices have already been projected to pixel
coordinates, and a grid of pixels. For every pixel we want one number: **which
triangle is visible there**, or `-1` if none.

    in    verts_px  (B, V, 2)   xy in pixels, y downward
          depth     (B, V)      smaller = nearer
          faces     (F, 3)      vertex indices, shared across the batch
    out   fid       (B, H, W)   winning triangle index, or -1

Two sub-problems, and they are independent:

1. **Coverage** — does triangle *f* contain the centre of pixel *(x, y)*?
2. **Visibility** — of the triangles that do, which is nearest?

Everything below is those two.

---

## 2. Barycentric coordinates

A point **P** inside triangle **P₀P₁P₂** can be written as a weighted average of
its corners:

    P = w₀·P₀ + w₁·P₁ + w₂·P₂        with   w₀ + w₁ + w₂ = 1

The weights **(w₀, w₁, w₂)** are the barycentric coordinates of P. They are the
natural coordinate system of a triangle: independent of where the triangle sits
or how it is rotated, and the same weights work for interpolating *any* quantity
stored at the corners — depth, colour, normals, UVs.

### Deriving them

Take P₂ as the origin. Let

    a = P₀ − P₂        b = P₁ − P₂        p = P − P₂

Substituting into the definition and using w₂ = 1 − w₀ − w₁:

    p = w₀·a + w₁·b

Two equations, two unknowns. Solve with the 2D cross product
`cross(u, v) = uₓvᵧ − uᵧvₓ`, which is the signed area of the parallelogram
spanned by u and v:

    w₀ = cross(p, b) / cross(a, b)
    w₁ = cross(a, p) / cross(a, b)
    w₂ = 1 − w₀ − w₁

That is exactly what `_bary` computes, with `d = cross(a, b)`:

    d  = (y₁−y₂)(x₀−x₂) + (x₂−x₁)(y₀−y₂)
    w₀ = ((y₁−y₂)(pₓ−x₂) + (x₂−x₁)(pᵧ−y₂)) / d
    w₁ = ((y₂−y₀)(pₓ−x₂) + (x₀−x₂)(pᵧ−y₂)) / d

### What they mean geometrically

`cross(a, b)` is twice the signed area of the whole triangle, and each numerator
is twice the signed area of a sub-triangle. So:

    w₀ = area(P, P₁, P₂) / area(P₀, P₁, P₂)
    w₁ = area(P₀, P, P₂) / area(P₀, P₁, P₂)
    w₂ = area(P₀, P₁, P) / area(P₀, P₁, P₂)

w₀ is "how much of the triangle is on the far side of P from P₀". At P₀ itself
it is 1; on the opposite edge it is 0; outside that edge it goes negative. Which
gives the coverage test for free.

### The inside test

    inside  ⟺  w₀ ≥ 0 and w₁ ≥ 0 and w₂ ≥ 0

Three conditions, one per edge. Negative means "outside that edge".

**`≥`, not `>`.** A barycentric of exactly 0 means the point sits precisely on an
edge, and the reference counts that as inside. Using `>` loses a thin scatter of
pixels — usually invisible on a real mesh, which is why the `edges` fixture puts
vertices exactly on pixel centres to force the issue.

**Winding does not matter.** Flip the vertex order and `d` changes sign, but so
does every numerator, so the ratios are unchanged. Interior points have all
three weights positive either way. This rasteriser therefore draws back-facing
triangles too — there is no backface culling — and the far side of the head is
rasterised and then thrown away by the depth test. If you ever add culling
(`d < 0` → skip), the reference will disagree with you.

### The degenerate case

If the three vertices are collinear the triangle has zero area, `d = 0`, and
every weight is a division by zero. The reference clamps:

    if |d| < 1e-12:  d = 1e-12

Without that you get `inf` and `NaN`. And `NaN >= 0` is **false**, so a NaN
triangle silently covers nothing rather than raising — a bug that hides. The
`degenerate` fixture contains a point, a collinear triple and a zero-height
triangle for exactly this.

---

## 3. Depth, and why interpolation is simple here

The depth at P is the barycentric interpolation of the corner depths:

    z(P) = w₀·z₀ + w₁·z₁ + w₂·z₂

**This is exact in this pipeline, and that is not generally true.** Under a real
perspective projection you must interpolate `1/w` and `z/w` and divide —
"perspective-correct interpolation" — because the projection is a division and
does not commute with linear interpolation. Interpolating z linearly in screen
space under perspective is a classic and very visible bug.

We are safe because `face3d/pipeline.py::project` is *weak perspective*:

    xy = v_xy · s + t
    z  = −v_z · s

Scale and translate. No divide by depth. An **affine** map, and affine maps
commute with linear interpolation, so screen-space linear interpolation is
exactly right. If the camera ever becomes a true perspective one, this section
stops being true.

The negation of z is deliberate: it makes **smaller = nearer**, so visibility is
a `min`.

---

## 4. Visibility: the z-buffer

For each pixel keep the smallest depth seen so far, and the triangle that
produced it. Classic z-buffer, one pass over all triangles, no sorting.

The subtlety is **ties**. Two triangles at exactly the same depth must resolve
the same way every run, or the output is nondeterministic — and on a GPU,
thousands of threads write concurrently in an order that changes run to run. The
reference breaks ties toward the **lower face index**.

### The packed key

Rather than a depth buffer plus an id buffer plus a lock, pack both into one
64-bit integer, ordered so that a single `min` resolves depth first and index
second:

    q   = (z − z_min) · scale        quantised to 30 bits, clamped
    key = q · F + f                  F = number of faces

The high bits are depth, the low bits are the face index. Comparing keys
compares depth first; equal depths fall through to comparing `f`, and the lower
index wins. Exactly the required rule.

`z_min` and `scale` are computed over the whole batch so keys from different
pixels remain comparable, and 2³⁰ leaves headroom below the 63 bits available
after `· F`.

The empty value is `BIG = 2³⁰·F + F`, larger than any real key, so an untouched
pixel unpacks to `-1`.

Python does this to make one `scatter_reduce(reduce="amin")` work. In CUDA it is
what the hardware wants anyway: **`atomicMin` on `unsigned long long`**, which
is a single instruction and needs no lock. Determinism comes free — `min` is
commutative and associative, so the answer does not depend on arrival order.

---

## 5. Pseudocode

The reference, and what your kernel must reproduce:

    zmin, zmax = min(depth), max(depth)          # over the whole batch
    scale = 2^30 / (zmax - zmin + 1e-12)
    BIG   = 2^30 * F + F
    buf[b][y][x] = BIG   for all b, y, x

    for b in 0 .. B-1:
      for f in 0 .. F-1:                          # ← parallel over (b, f)
        i0, i1, i2 = faces[f]
        (x0,y0), (x1,y1), (x2,y2) = verts_px[b][i0], [i1], [i2]
        z0, z1, z2               = depth[b][i0],   [i1], [i2]

        xlo = max(0,   floor(min(x0,x1,x2)))      # bounding box, clipped
        xhi = min(W-1, floor(max(x0,x1,x2)))
        ylo = max(0,   floor(min(y0,y1,y2)))
        yhi = min(H-1, floor(max(y0,y1,y2)))

        d = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2)
        if |d| < 1e-12: d = 1e-12

        for y in ylo .. yhi:
          for x in xlo .. xhi:
            px, py = x + 0.5, y + 0.5             # pixel CENTRE
            w0 = ((y1-y2)*(px-x2) + (x2-x1)*(py-y2)) / d
            w1 = ((y2-y0)*(px-x2) + (x0-x2)*(py-y2)) / d
            w2 = 1 - w0 - w1
            if w0 < 0 or w1 < 0 or w2 < 0: continue

            z = w0*z0 + w1*z1 + w2*z2
            q = clamp(int((z - zmin) * scale), 0, 2^30)
            atomicMin(&buf[b][y][x], q * F + f)

    fid[b][y][x] = -1 if buf == BIG else buf % F

Note the bounding box is clipped, not clamped-and-skipped: a triangle can be
partly or entirely outside the frame, and one in the `offscreen` fixture
*encloses* the whole frame with no vertex inside it. Get the clipping wrong and
that one vanishes.

---

## 6. Why the PyTorch version is slow, and where yours won't be

PyTorch has no way to write "for each pixel in this triangle's box" — it can
only scatter from a materialised tensor. So it buckets triangles by size, and
for each bucket builds a `(B, F, K², 3)` tensor of candidate pixels: every
triangle in the bucket pays for K² pixels whether it covers them or not.

Measured on this mesh at B=8, 224²:

| | |
|---|---|
| candidate (face, pixel) pairs tested | 6.69 M |
| pixels of actual triangle area | 0.46 M |
| **wasted work** | **14.4×** |
| peak memory | 196 MB |
| time | 15.3 ms |

A kernel loops the real bounding box, so it does the 0.46 M and allocates
nothing. That alone is most of the win.

## 7. Where it gets interesting

One thread per triangle is the right first version, and it is not the end.

    triangle span, pixels:   median 2.4    mean 3.9    p99 15.8    max 20.0

The median triangle covers about 6 pixels; the largest covers around 400. Threads
in a warp run in lockstep, so a warp finishes when its *slowest* lane does — 31
lanes idle while one grinds through a big triangle. That is the load imbalance,
and it is the real optimisation problem here.

Directions, roughly in order of how much they'd teach you:

- **Persistent threads with a work queue** — threads pull triangles from an
  atomic counter instead of owning one, so a thread that finishes early takes
  more work.
- **Two-phase binning** — first pass assigns triangles to screen tiles, second
  pass rasterises one tile per block with the depth buffer in shared memory.
  This is roughly what real GPU rasterisers do, and it turns global atomics into
  shared-memory ones.
- **One warp per triangle** — lanes cover different pixels of the same triangle.
  Simple, and it helps the large triangles at the cost of the small ones.

Measure before choosing. The empty-kernel floor here is ~0.3 ms, which is the
memset and unpack; that is the budget you are working against, not zero.
