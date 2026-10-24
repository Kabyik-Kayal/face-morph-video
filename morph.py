#!/usr/bin/env python3
"""
True Face Morphing Pipeline: Martin Scorsese -> Maggie Smith
Performs genuine dense Delaunay mesh deformation and warping so that:
  - Scorsese's wide open smile physically closes and narrows into Maggie's subtle lips
  - The thick, bushy eyebrows visibly thin out and arch upward
  - The bulbous nose tip elongates downward and narrows
  - The square jaw contracts into an elegant oval chin
  - The colors and textures blend synchronously with the shape deformation

Output: morph.mp4 strictly smaller than 100 KB (100,000 bytes).
"""

import os
import sys
import math
import subprocess
import requests
from PIL import Image, ImageDraw, ImageFilter

# ─────────────────────────────────────────────────────────────────────────────
# 1. GEOMETRY & DELAUNAY TRIANGULATION (BOWYER-WATSON)
# ─────────────────────────────────────────────────────────────────────────────

class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

def circumcircle_contains(p1, p2, p3, p):
    """Returns True if point p lies inside the circumcircle of triangle (p1, p2, p3)."""
    d = 2 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))
    if abs(d) < 1e-9:
        return False
    ux = ((p1.x**2 + p1.y**2) * (p2.y - p3.y) + (p2.x**2 + p2.y**2) * (p3.y - p1.y) + (p3.x**2 + p3.y**2) * (p1.y - p2.y)) / d
    uy = ((p1.x**2 + p1.y**2) * (p3.x - p2.x) + (p2.x**2 + p2.y**2) * (p1.x - p3.x) + (p3.x**2 + p3.y**2) * (p2.x - p1.x)) / d
    r2 = (p1.x - ux)**2 + (p1.y - uy)**2
    dist2 = (p.x - ux)**2 + (p.y - uy)**2
    return dist2 <= r2

def delaunay_triangulation(points):
    """Bowyer-Watson algorithm for Delaunay Triangulation in 2D."""
    pts = [Point(x, y) for x, y in points]
    min_x = min(p.x for p in pts) - 100
    max_x = max(p.x for p in pts) + 100
    min_y = min(p.y for p in pts) - 100
    max_y = max(p.y for p in pts) + 100

    st1 = Point(min_x - 1000, min_y - 1000)
    st2 = Point(max_x + 1000, min_y - 1000)
    st3 = Point((min_x + max_x) / 2, max_y + 3000)
    all_pts = pts + [st1, st2, st3]
    st_indices = {len(pts), len(pts) + 1, len(pts) + 2}

    triangles = [(len(pts), len(pts) + 1, len(pts) + 2)]

    for i, p in enumerate(pts):
        bad_triangles = []
        for tri in triangles:
            p1, p2, p3 = all_pts[tri[0]], all_pts[tri[1]], all_pts[tri[2]]
            if circumcircle_contains(p1, p2, p3, p):
                bad_triangles.append(tri)

        polygon = []
        for tri in bad_triangles:
            edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            for edge in edges:
                shared = False
                for other in bad_triangles:
                    if other == tri:
                        continue
                    other_edges = [(other[0], other[1]), (other[1], other[2]), (other[2], other[0])]
                    if edge in other_edges or (edge[1], edge[0]) in other_edges:
                        shared = True
                        break
                if not shared:
                    polygon.append(edge)

        for tri in bad_triangles:
            triangles.remove(tri)

        for edge in polygon:
            triangles.append((edge[0], edge[1], i))

    return [tri for tri in triangles if not (tri[0] in st_indices or tri[1] in st_indices or tri[2] in st_indices)]

# ─────────────────────────────────────────────────────────────────────────────
# 2. PIECEWISE AFFINE WARPING
# ─────────────────────────────────────────────────────────────────────────────

def solve_affine(src_tri, dst_tri):
    """
    Computes affine transformation matrix mapping destination triangle
    coordinates (x, y) to source (u, v):
        u = a*x + b*y + c
        v = d*x + e*y + f
    """
    (x0, y0), (x1, y1), (x2, y2) = dst_tri
    (u0, v0), (u1, v1), (u2, v2) = src_tri
    det = x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1)
    if abs(det) < 1e-6:
        return None
    a = (u0 * (y1 - y2) + u1 * (y2 - y0) + u2 * (y0 - y1)) / det
    b = (u0 * (x2 - x1) + u1 * (x0 - x2) + u2 * (x1 - x0)) / det
    c = (u0 * (x1 * y2 - x2 * y1) + u1 * (x2 * y0 - x0 * y2) + u2 * (x0 * y1 - x1 * y0)) / det

    d = (v0 * (y1 - y2) + v1 * (y2 - y0) + v2 * (y0 - y1)) / det
    e = (v0 * (x2 - x1) + v1 * (x0 - x2) + v2 * (x1 - x0)) / det
    f = (v0 * (x1 * y2 - x2 * y1) + v1 * (x2 * y0 - x0 * y2) + v2 * (x0 * y1 - x1 * y0)) / det
    return (a, b, c, d, e, f)

def warp_image_mesh(src_im, src_pts, dst_pts, triangles, width, height):
    """Warps source image to destination mesh using triangular affine patches."""
    dst_canvas = Image.new('RGB', (width, height), (0, 0, 0))
    for i, j, k in triangles:
        s_tri = [src_pts[i], src_pts[j], src_pts[k]]
        d_tri = [dst_pts[i], dst_pts[j], dst_pts[k]]

        min_x = max(0, int(math.floor(min(p[0] for p in d_tri))))
        max_x = min(width, int(math.ceil(max(p[0] for p in d_tri))) + 1)
        min_y = max(0, int(math.floor(min(p[1] for p in d_tri))))
        max_y = min(height, int(math.ceil(max(p[1] for p in d_tri))) + 1)

        tw = max_x - min_x
        th = max_y - min_y
        if tw <= 0 or th <= 0:
            continue

        local_d_tri = [(p[0] - min_x, p[1] - min_y) for p in d_tri]
        aff = solve_affine(s_tri, local_d_tri)
        if aff is None:
            continue

        tile = src_im.transform((tw, th), Image.Transform.AFFINE, aff, Image.Resampling.BILINEAR)
        mask = Image.new('L', (tw, th), 0)
        ImageDraw.Draw(mask).polygon(local_d_tri, fill=255, outline=255, width=1)
        dst_canvas.paste(tile, (min_x, min_y), mask)
    return dst_canvas

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
    print("Genuine Face Morphing: Martin Scorsese -> Maggie Smith")
    print("=" * 60)

    work_dir = os.path.dirname(os.path.abspath(__file__))
    frames_dir = os.path.join(work_dir, "frames_v2")
    os.makedirs(frames_dir, exist_ok=True)

    sc_path = os.path.join(work_dir, "scorsese.jpg")
    mg_path = os.path.join(work_dir, "maggie.jpg")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) FaceMorph/2.0'}

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

    # Scale-matched crop into 240x300
    im_s = sc_prev.crop((60, 120, 334, 462)).resize((240, 300), Image.Resampling.LANCZOS)
    im_m_raw = mg_prev.crop((85, 149, 335, 461)).resize((240, 300), Image.Resampling.LANCZOS)

    # Inpaint Maggie's resting hand by mirroring symmetrical right cheek
    clean_half = im_m_raw.crop((120, 0, 240, 300)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    mask = Image.new('L', (120, 300), 0)
    d = ImageDraw.Draw(mask)
    d.polygon([(10, 110), (85, 120), (88, 230), (50, 270), (0, 270), (0, 110)], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(8))
    im_m = im_m_raw.copy()
    im_m.paste(clean_half, (0, 0), mask)

    # Vignette background to eliminate harsh contrast
    bg_mask = Image.new('L', (240, 300), 255)
    d_bg = ImageDraw.Draw(bg_mask)
    d_bg.ellipse([(15, 0), (225, 290)], fill=0)
    bg_mask = bg_mask.filter(ImageFilter.GaussianBlur(15))
    dark_bg = Image.new('RGB', (240, 300), (15, 18, 22))
    im_m = Image.composite(dark_bg, im_m, bg_mask)

    # Subtle warm color grade to unify color spaces
    mg_l = im_m.convert('L')
    r_ch = mg_l.point(lambda p: min(255, int(p * 1.08)))
    g_ch = mg_l.point(lambda p: min(255, int(p * 0.96)))
    b_ch = mg_l.point(lambda p: min(255, int(p * 0.88)))
    im_m = Image.merge('RGB', (r_ch, g_ch, b_ch))

    im_s.save(os.path.join(work_dir, "scorsese_crop.jpg"))
    im_m.save(os.path.join(work_dir, "maggie_crop.jpg"))

    # Dense 82-point anatomical landmarks (tracking distinct feature deformations)
    pts_sc = [
        # Left eye
        (70, 78), (88, 77), (52, 79), (70, 71), (70, 85),
        # Right eye
        (170, 75), (152, 77), (188, 77), (170, 69), (170, 83),
        # Left brow top (thick bushy brow)
        (30, 68), (55, 48), (85, 48), (115, 60),
        # Left brow bottom
        (35, 75), (65, 68), (95, 66), (115, 68),
        # Right brow top
        (125, 60), (155, 48), (185, 48), (210, 68),
        # Right brow bottom
        (125, 68), (145, 66), (175, 68), (205, 75),
        # Nose
        (120, 76), (120, 102), (120, 136), (96, 136), (144, 136), (120, 146),
        # Mouth & Teeth (wide open grin)
        (58, 160), (182, 150), (120, 152), (120, 160), (90, 162), (150, 158),
        (120, 168), (120, 182), (90, 178), (150, 175),
        # Chin & Jaw (square, wide, stubby)
        (120, 222), (88, 216), (152, 216), (48, 185), (192, 185), (38, 135), (202, 135),
        # Cheeks & Temples
        (45, 110), (195, 110), (65, 125), (175, 125),
        # Forehead & Hair
        (120, 30), (65, 25), (175, 25), (120, 0), (50, 5), (190, 5), (15, 60), (225, 60),
        # Neck & Suit
        (70, 255), (170, 255), (120, 260), (30, 290), (210, 290),
        # Outer Boundary Pins
        (0, 0), (60, 0), (120, 0), (180, 0), (239, 0),
        (0, 75), (239, 75),
        (0, 150), (239, 150),
        (0, 225), (239, 225),
        (0, 299), (60, 299), (120, 299), (180, 299), (239, 299)
    ]

    pts_mg = [
        # Left eye
        (68, 77), (86, 79), (50, 77), (68, 70), (68, 85),
        # Right eye
        (172, 77), (154, 79), (190, 77), (172, 70), (172, 85),
        # Left brow top (thin arched brow)
        (35, 62), (55, 48), (80, 48), (105, 56),
        # Left brow bottom
        (35, 66), (55, 54), (80, 54), (105, 60),
        # Right brow top
        (135, 56), (160, 48), (185, 48), (205, 62),
        # Right brow bottom
        (135, 60), (160, 54), (185, 54), (205, 66),
        # Nose (slender, 14px lower tip!)
        (120, 80), (120, 115), (120, 150), (100, 148), (140, 148), (120, 158),
        # Mouth & Teeth (closed lips, 42px lower, 52px narrower!)
        (84, 202), (156, 202), (120, 194), (120, 200), (98, 201), (142, 201),
        (120, 202), (120, 212), (98, 208), (142, 208),
        # Chin & Jaw (46px lower, elegant oval!)
        (120, 268), (92, 252), (148, 252), (58, 225), (182, 225), (35, 145), (195, 145),
        # Cheeks & Temples
        (42, 110), (198, 110), (68, 140), (172, 140),
        # Forehead & Hair
        (120, 30), (65, 25), (175, 25), (120, 0), (45, 5), (195, 5), (10, 60), (230, 60),
        # Neck & Suit / Scarf
        (65, 275), (175, 275), (120, 285), (20, 290), (220, 290),
        # Outer Boundary Pins
        (0, 0), (60, 0), (120, 0), (180, 0), (239, 0),
        (0, 75), (239, 75),
        (0, 150), (239, 150),
        (0, 225), (239, 225),
        (0, 299), (60, 299), (120, 299), (180, 299), (239, 299)
    ]

    avg_pts = [((ps[0] + pm[0]) / 2, (ps[1] + pm[1]) / 2) for ps, pm in zip(pts_sc, pts_mg)]
    triangles = delaunay_triangulation(avg_pts)
    print(f"Constructed {len(triangles)} Delaunay triangles across {len(avg_pts)} landmarks.")

    N_MORPH = 36
    HOLD_START = 6
    HOLD_END = 6
    FPS = 16

    print("Generating morph frames with genuine feature deformation...")
    idx = 0

    for _ in range(HOLD_START):
        im_s.save(os.path.join(frames_dir, f"frame_{idx:04d}.png"))
        idx += 1

    for i in range(1, N_MORPH):
        t_lin = i / N_MORPH
        # Cosine ease-in ease-out
        t = (1.0 - math.cos(t_lin * math.pi)) / 2.0

        mid_pts = [
            ((1 - t) * ps[0] + t * pm[0], (1 - t) * ps[1] + t * pm[1])
            for ps, pm in zip(pts_sc, pts_mg)
        ]
        ws = warp_image_mesh(im_s, pts_sc, mid_pts, triangles, 240, 300)
        wm = warp_image_mesh(im_m, pts_mg, mid_pts, triangles, 240, 300)
        blended = Image.blend(ws, wm, t)
        blended.save(os.path.join(frames_dir, f"frame_{idx:04d}.png"))
        idx += 1

    for _ in range(HOLD_END):
        im_m.save(os.path.join(frames_dir, f"frame_{idx:04d}.png"))
        idx += 1

    print(f"Generated {idx} total frames.")

    output_mp4 = os.path.join(work_dir, "morph.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "27",
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
