import math
import time
from PIL import Image, ImageFilter, ImageChops

class ThinPlateSpline2D:
    def __init__(self, src_pts, dst_pts, reg=1e-3):
        self.pts = dst_pts
        N = len(dst_pts)
        def U(r):
            if r < 1e-9:
                return 0.0
            return r * r * math.log(r)

        K = [[0.0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i != j:
                    dx = dst_pts[i][0] - dst_pts[j][0]
                    dy = dst_pts[i][1] - dst_pts[j][1]
                    K[i][j] = U(math.hypot(dx, dy))

        L = [[0.0] * (N + 3) for _ in range(N + 3)]
        for i in range(N):
            for j in range(N):
                L[i][j] = K[i][j]
            L[i][N] = 1.0
            L[i][N + 1] = dst_pts[i][0]
            L[i][N + 2] = dst_pts[i][1]
            L[N][i] = 1.0
            L[N + 1][i] = dst_pts[i][0]
            L[N + 2][i] = dst_pts[i][1]
            L[i][i] += reg

        Yx = [src_pts[i][0] for i in range(N)] + [0.0, 0.0, 0.0]
        Yy = [src_pts[i][1] for i in range(N)] + [0.0, 0.0, 0.0]

        def solve_linear(A, b):
            M = [row[:] for row in A]
            v = b[:]
            n = len(v)
            for i in range(n):
                max_row = i
                for k in range(i + 1, n):
                    if abs(M[k][i]) > abs(M[max_row][i]):
                        max_row = k
                M[i], M[max_row] = M[max_row], M[i]
                v[i], v[max_row] = v[max_row], v[i]
                pivot = M[i][i]
                if abs(pivot) < 1e-12:
                    continue
                for k in range(i + 1, n):
                    factor = M[k][i] / pivot
                    for j in range(i, n):
                        M[k][j] -= factor * M[i][j]
                    v[k] -= factor * v[i]
            res = [0.0] * n
            for i in range(n - 1, -1, -1):
                s = v[i]
                for j in range(i + 1, n):
                    s -= M[i][j] * res[j]
                res[i] = s / M[i][i]
            return res

        self.Wx = solve_linear(L, Yx)
        self.Wy = solve_linear(L, Yy)
        self.N = N

    def transform_point(self, x, y):
        def U(r):
            if r < 1e-9:
                return 0.0
            return r * r * math.log(r)
        ux = self.Wx[self.N] + self.Wx[self.N + 1] * x + self.Wx[self.N + 2] * y
        uy = self.Wy[self.N] + self.Wy[self.N + 1] * x + self.Wy[self.N + 2] * y
        for i in range(self.N):
            dx = x - self.pts[i][0]
            dy = y - self.pts[i][1]
            r = math.hypot(dx, dy)
            if r > 1e-9:
                u_r = U(r)
                ux += self.Wx[i] * u_r
                uy += self.Wy[i] * u_r
        return (ux, uy)

def warp_tps_dense(im, src_pts, dst_pts, w=240, h=300, step=8):
    """Warps image using a continuous Thin Plate Spline grid with bilinear resampling."""
    tps = ThinPlateSpline2D(src_pts, dst_pts)
    grid_pts = {}
    for y in range(0, h + step, step):
        for x in range(0, w + step, step):
            gx = min(w, x)
            gy = min(h, y)
            grid_pts[(gx, gy)] = tps.transform_point(gx, gy)

    mesh_data = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            x1 = min(w, x + step)
            y1 = min(h, y + step)
            bbox = (x, y, x1, y1)
            u0, v0 = grid_pts[(x, y)]
            u1, v1 = grid_pts[(x, y1)]
            u2, v2 = grid_pts[(x1, y1)]
            u3, v3 = grid_pts[(x1, y)]
            quad = (u0, v0, u1, v1, u2, v2, u3, v3)
            mesh_data.append((bbox, quad))

    return im.transform((w, h), Image.Transform.MESH, mesh_data, Image.Resampling.BILINEAR)

def laplacian_blend_images(im_a, im_b, t):
    """
    Multiresolution frequency-band blending:
      - Low frequency (lighting, broad tone) blends smoothly with t.
      - High frequency (sharp edges of eyes, lips, pores) uses a sigmoidal handover.
    """
    g1_a = im_a.filter(ImageFilter.GaussianBlur(3.0))
    g1_b = im_b.filter(ImageFilter.GaussianBlur(3.0))
    mid_blend = Image.blend(g1_a, g1_b, (1.0 - math.cos(t * math.pi)) / 2.0)

    # Sigmoid handover for high-frequency micro-features
    t_high = 1.0 / (1.0 + math.exp(-14.0 * (t - 0.5)))
    high_blend = Image.blend(im_a, im_b, t_high)

    return Image.blend(mid_blend, high_blend, 0.5)
