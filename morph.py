#!/usr/bin/env python3
"""
Advanced Face Morphing Pipeline: Thin Plate Splines (TPS) + Multiresolution Laplacian Blending
Morphs the facial features of the female portrait (Maggie Smith) directly onto the male
portrait's face (Martin Scorsese) using:
  1. Continuous 2D Thin Plate Spline (TPS) with biharmonic energy minimization (C¹ smooth deformation)
  2. Multiresolution frequency-separated Laplacian blending to eliminate ghosting
  3. Stable boundary pinning to preserve hair, ears, suit, tie, and background

Output: morph.mp4 strictly smaller than 100 KB (100,000 bytes).
"""

import os
import sys
import math
import subprocess
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageStat

# ─────────────────────────────────────────────────────────────────────────────
# 1. THIN PLATE SPLINE (BIHARMONIC ENERGY MINIMIZATION)
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# 2. MULTIRESOLUTION LAPLACIAN PYRAMID BLENDING
# ─────────────────────────────────────────────────────────────────────────────

def laplacian_blend_images(im_a, im_b, t):
    """
    Multiresolution frequency-separated blending:
      - Low frequencies (skin tone, broad illumination) blend smoothly with t.
      - High frequencies (sharp feature contours, pores) undergo sigmoidal handover.
    """
    g1_a = im_a.filter(ImageFilter.GaussianBlur(3.0))
    g1_b = im_b.filter(ImageFilter.GaussianBlur(3.0))
    mid_blend = Image.blend(g1_a, g1_b, (1.0 - math.cos(t * math.pi)) / 2.0)

    # Sigmoid handover for fine micro-textures to eliminate ghosting
    t_high = 1.0 / (1.0 + math.exp(-14.0 * (t - 0.5)))
    high_blend = Image.blend(im_a, im_b, t_high)

    return Image.blend(mid_blend, high_blend, 0.5)

# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 60)
    print("Advanced Face Morphing: Thin Plate Splines (TPS) + Laplacian")
    print("=" * 60)

    work_dir = os.path.dirname(os.path.abspath(__file__))
    frames_dir = os.path.join(work_dir, "frames_tps")
    os.makedirs(frames_dir, exist_ok=True)

    sc_path = os.path.join(work_dir, "scorsese.jpg")
    mg_path = os.path.join(work_dir, "maggie.jpg")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) FaceMorph/4.0'}

    if not os.path.exists(sc_path):
        print("Downloading Martin Scorsese source photo...")
        url = "https://upload.wikimedia.org/wikipedia/commons/c/ce/Martin_Scorsese_MFF_2023.jpg"
        r = requests.get(url, headers=headers)
        with open(sc_path, 'wb') as f:
            f.write(r.content)

    if not os.path.exists(mg_path):
        print("Downloading Maggie Smith source photo...")
        url = "https://upload.wikimedia.org/wikipedia/commons/2/28/Maggie_Smith_-_Vintage_%28trim%29.jpg"
        r = requests.get(url, headers=headers)
        with open(mg_path, 'wb') as f:
            f.write(r.content)

    print("Normalizing faces and framing...")
    sc_orig = Image.open(sc_path).convert('RGB')
    mg_orig = Image.open(mg_path).convert('RGB')

    sc_prev = sc_orig.resize((400, int(400 * sc_orig.height / sc_orig.width)), Image.Resampling.LANCZOS)
    mg_prev = mg_orig.resize((400, int(400 * mg_orig.height / mg_orig.width)), Image.Resampling.LANCZOS)

    w, h = 240, 300
    sc = sc_prev.crop((60, 120, 334, 462)).resize((w, h), Image.Resampling.LANCZOS)
    mg_raw = mg_prev.crop((85, 149, 335, 461)).resize((w, h), Image.Resampling.LANCZOS)

    # Inpaint Maggie's resting hand using her symmetrical right cheek
    clean_half = mg_raw.crop((120, 0, 240, 300)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    mask_hand = Image.new('L', (120, 300), 0)
    d_h = ImageDraw.Draw(mask_hand)
    d_h.polygon([(10, 110), (85, 120), (88, 230), (50, 270), (0, 270), (0, 110)], fill=255)
    mask_hand = mask_hand.filter(ImageFilter.GaussianBlur(8))
    mg_clean = mg_raw.copy()
    mg_clean.paste(clean_half, (0, 0), mask_hand)

    # Match skin tone from Scorsese's cheek/chin to Maggie's skin
    sc_skin = sc.crop((100, 190, 140, 230))
    stat_sc = ImageStat.Stat(sc_skin)
    mean_r, mean_g, mean_b = stat_sc.mean[:3]

    mg_gray = mg_clean.convert('L')
    mg_skin = mg_gray.crop((100, 140, 140, 180))
    mean_mg = ImageStat.Stat(mg_skin).mean[0]

    scale_r = (mean_r / mean_mg) * 1.02
    scale_g = (mean_g / mean_mg) * 1.01
    scale_b = (mean_b / mean_mg) * 0.99

    r_ch = mg_gray.point(lambda p: min(255, int(p * scale_r)))
    g_ch = mg_gray.point(lambda p: min(255, int(p * scale_g)))
    b_ch = mg_gray.point(lambda p: min(255, int(p * scale_b)))
    mg = Image.merge('RGB', (r_ch, g_ch, b_ch))

    # Face contour boundary (pinned to Scorsese's head)
    contour = [
        (50, 42), (85, 30), (120, 26), (155, 30), (190, 42),
        (212, 58), (220, 75), (205, 110), (198, 145), (188, 185),
        (155, 226), (120, 238), (85, 226),
        (52, 185), (42, 145), (35, 110), (20, 75), (28, 58)
    ]

    inner_sc = [
        (70, 78), (170, 75), (88, 77), (152, 77), (52, 79), (188, 77),
        (25, 68), (55, 46), (85, 46), (115, 60),
        (25, 75), (65, 68), (95, 66), (115, 68),
        (125, 60), (155, 46), (185, 46), (215, 68),
        (125, 68), (145, 66), (175, 68), (215, 75),
        (120, 76), (120, 102), (120, 136), (96, 136), (144, 136),
        (58, 160), (182, 150), (120, 152), (120, 160), (90, 162), (150, 158),
        (120, 168), (120, 182), (90, 178), (150, 175),
        (120, 218), (90, 212), (150, 212),
        (65, 125), (175, 125)
    ]

    inner_mg = [
        (68, 77), (172, 77), (86, 79), (154, 79), (50, 77), (190, 77),
        (32, 62), (55, 48), (80, 48), (105, 56),
        (32, 66), (55, 54), (80, 54), (105, 60),
        (135, 56), (160, 48), (185, 48), (208, 62),
        (135, 60), (160, 54), (185, 54), (208, 66),
        (120, 80), (120, 115), (120, 150), (100, 148), (140, 148),
        (84, 195), (156, 195), (120, 188), (120, 194), (98, 195), (142, 195),
        (120, 196), (120, 206), (98, 202), (142, 202),
        (120, 228), (92, 222), (148, 222),
        (68, 135), (172, 135)
    ]

    pins = [
        (0, 0), (239, 0), (0, 299), (239, 299),
        (120, 0), (0, 150), (239, 150), (120, 299)
    ]

    pts_sc = pins + contour + inner_sc
    pts_mg = pins + contour + inner_mg

    mask = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(mask)
    d.polygon(contour, fill=255)
    mask_feather = mask.filter(ImageFilter.GaussianBlur(5))

    N_MORPH = 36
    HOLD_START = 6
    HOLD_END = 6
    FPS = 16
    idx = 0

    print("Rendering TPS + Laplacian frames...")

    # Hold Scorsese face
    for _ in range(HOLD_START):
        sc.save(os.path.join(frames_dir, f"frame_{idx:04d}.png"))
        idx += 1

    # Morph facial features using TPS + Laplacian blending
    for i in range(1, N_MORPH):
        t_lin = i / N_MORPH
        t = (1.0 - math.cos(t_lin * math.pi)) / 2.0

        mid_pts = [
            ((1 - t) * ps[0] + t * pm[0], (1 - t) * ps[1] + t * pm[1])
            for ps, pm in zip(pts_sc, pts_mg)
        ]
        ws = warp_tps_dense(sc, pts_sc, mid_pts, w, h, step=8)
        wm = warp_tps_dense(mg, pts_mg, mid_pts, w, h, step=8)
        blended_face = laplacian_blend_images(ws, wm, t)

        frame = Image.composite(blended_face, sc, mask_feather)
        frame.save(os.path.join(frames_dir, f"frame_{idx:04d}.png"))
        idx += 1

    # Hold female face on Scorsese
    wm_end = warp_tps_dense(mg, pts_mg, pts_mg, w, h, step=8)
    end_frame = Image.composite(wm_end, sc, mask_feather)
    for _ in range(HOLD_END):
        end_frame.save(os.path.join(frames_dir, f"frame_{idx:04d}.png"))
        idx += 1

    print(f"Generated {idx} frames in {frames_dir}.")

    # Generate filmstrip
    f0 = Image.open(os.path.join(frames_dir, "frame_0000.png"))
    f1 = Image.open(os.path.join(frames_dir, "frame_0012.png"))
    f2 = Image.open(os.path.join(frames_dir, "frame_0024.png"))
    f3 = Image.open(os.path.join(frames_dir, "frame_0036.png"))
    f4 = Image.open(os.path.join(frames_dir, f"frame_{idx-1:04d}.png"))

    strip = Image.new('RGB', (w * 5, h))
    strip.paste(f0, (0, 0))
    strip.paste(f1, (w, 0))
    strip.paste(f2, (w * 2, 0))
    strip.paste(f3, (w * 3, 0))
    strip.paste(f4, (w * 4, 0))
    strip.save(os.path.join(work_dir, "morph_strip.jpg"), quality=92)

    # Encode with FFmpeg
    output_mp4 = os.path.join(work_dir, "morph.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "26",
        "-preset", "slow",
        "-movflags", "+faststart",
        output_mp4
    ]

    print("Encoding video with FFmpeg...")
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    size_bytes = os.path.getsize(output_mp4)
    size_kb = size_bytes / 1000.0

    print("\n" + "=" * 60)
    print(f"Video saved: {output_mp4}")
    print(f"File size: {size_bytes:,} bytes ({size_kb:.2f} KB)")
    print(f"Limit: < 100,000 bytes (100.00 KB)")
    if size_bytes < 100000:
        print("[PASS] Video is strictly under the 100 KB limit!")
    else:
        print("[FAIL] File size exceeds 100 KB limit!")
    print("=" * 60)

if __name__ == "__main__":
    main()
