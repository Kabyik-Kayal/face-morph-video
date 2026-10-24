#!/usr/bin/env python3
"""
Face-Only Morphing Pipeline: Female Face onto Male Face (Maggie Smith -> Martin Scorsese)
Morphs the facial features (eyes, eyebrows, nose, mouth, cheeks, chin, skin)
of the female portrait directly onto the male portrait's face, keeping the male's
hair, ears, neck, suit, tie, and background completely stable and anchored.

Output: morph.mp4 strictly smaller than 100 KB (100,000 bytes).
"""

import os
import sys
import math
import subprocess
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageStat

# ─────────────────────────────────────────────────────────────────────────────
# 1. GEOMETRY & DELAUNAY TRIANGULATION (BOWYER-WATSON)
# ─────────────────────────────────────────────────────────────────────────────

class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

def circumcircle_contains(p1, p2, p3, p):
    d = 2 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))
    if abs(d) < 1e-9:
        return False
    ux = ((p1.x**2 + p1.y**2) * (p2.y - p3.y) + (p2.x**2 + p2.y**2) * (p3.y - p1.y) + (p3.x**2 + p3.y**2) * (p1.y - p2.y)) / d
    uy = ((p1.x**2 + p1.y**2) * (p3.x - p2.x) + (p2.x**2 + p2.y**2) * (p1.x - p3.x) + (p3.x**2 + p3.y**2) * (p2.x - p1.x)) / d
    r2 = (p1.x - ux)**2 + (p1.y - uy)**2
    dist2 = (p.x - ux)**2 + (p.y - uy)**2
    return dist2 <= r2

def delaunay_triangulation(points):
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
    print("Face-Only Morph: Maggie Smith Face -> Martin Scorsese Head")
    print("=" * 60)

    work_dir = os.path.dirname(os.path.abspath(__file__))
    frames_dir = os.path.join(work_dir, "frames_face_only")
    os.makedirs(frames_dir, exist_ok=True)

    sc_path = os.path.join(work_dir, "scorsese.jpg")
    mg_path = os.path.join(work_dir, "maggie.jpg")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) FaceMorph/3.0'}

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

    # Face boundary contour (Anchored strictly to Martin Scorsese's face outline)
    contour = [
        (50, 42), (85, 30), (120, 26), (155, 30), (190, 42),
        (212, 58), (220, 75), (205, 110), (198, 145), (188, 185),
        (155, 226), (120, 238), (85, 226),
        (52, 185), (42, 145), (35, 110), (20, 75), (28, 58)
    ]

    # Interior facial landmarks for Scorsese
    inner_sc = [
        # Left eye
        (70, 78), (88, 77), (52, 79), (70, 71), (70, 85),
        # Right eye
        (170, 75), (152, 77), (188, 77), (170, 69), (170, 83),
        # Left brow top (thick bushy brow)
        (25, 68), (55, 46), (85, 46), (115, 60),
        # Left brow bottom
        (25, 75), (65, 68), (95, 66), (115, 68),
        # Right brow top
        (125, 60), (155, 46), (185, 46), (215, 68),
        # Right brow bottom
        (125, 68), (145, 66), (175, 68), (215, 75),
        # Nose
        (120, 76), (120, 102), (120, 136), (96, 136), (144, 136), (120, 146),
        # Mouth & Teeth (wide open grin)
        (58, 160), (182, 150), (120, 152), (120, 160), (90, 162), (150, 158),
        (120, 168), (120, 182), (90, 178), (150, 175),
        # Chin interior
        (120, 218), (90, 212), (150, 212),
        # Cheeks
        (65, 125), (175, 125)
    ]

    # Corresponding interior facial landmarks for Maggie
    inner_mg = [
        # Left eye
        (68, 77), (86, 79), (50, 77), (68, 70), (68, 85),
        # Right eye
        (172, 77), (154, 79), (190, 77), (172, 70), (172, 85),
        # Left brow top (thin arched brow)
        (32, 62), (55, 48), (80, 48), (105, 56),
        # Left brow bottom
        (32, 66), (55, 54), (80, 54), (105, 60),
        # Right brow top
        (135, 56), (160, 48), (185, 48), (208, 62),
        # Right brow bottom
        (135, 60), (160, 54), (185, 54), (208, 66),
        # Nose (slender, lower tip)
        (120, 80), (120, 115), (120, 150), (100, 148), (140, 148), (120, 158),
        # Mouth & Teeth (closed lips, lower, narrower)
        (84, 195), (156, 195), (120, 188), (120, 194), (98, 195), (142, 195),
        (120, 196), (120, 206), (98, 202), (142, 202),
        # Chin interior
        (120, 228), (92, 222), (148, 222),
        # Cheeks
        (68, 135), (172, 135)
    ]

    # Outer canvas boundary pins (remain completely fixed)
    pins = [
        (0, 0), (60, 0), (120, 0), (180, 0), (239, 0),
        (0, 75), (239, 75), (0, 150), (239, 150), (0, 225), (239, 225),
        (0, 299), (60, 299), (120, 299), (180, 299), (239, 299)
    ]

    pts_sc = pins + contour + inner_sc
    pts_mg = pins + contour + inner_mg

    avg_pts = [((ps[0] + pm[0]) / 2, (ps[1] + pm[1]) / 2) for ps, pm in zip(pts_sc, pts_mg)]
    triangles = delaunay_triangulation(avg_pts)
    print(f"Constructed {len(triangles)} Delaunay triangles across {len(avg_pts)} landmarks.")

    # Soft mask along the face contour for seamless compositing onto Scorsese
    mask = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(mask)
    d.polygon(contour, fill=255)
    mask_feather = mask.filter(ImageFilter.GaussianBlur(5))

    N_MORPH = 36
    HOLD_START = 6
    HOLD_END = 6
    FPS = 16
    idx = 0

    print("Rendering face-only morph sequence...")

    # Hold Scorsese face
    for _ in range(HOLD_START):
        sc.save(os.path.join(frames_dir, f"frame_{idx:04d}.png"))
        idx += 1

    # Morph facial features inside the face boundary
    for i in range(1, N_MORPH):
        t_lin = i / N_MORPH
        t = (1.0 - math.cos(t_lin * math.pi)) / 2.0

        mid_pts = [
            ((1 - t) * ps[0] + t * pm[0], (1 - t) * ps[1] + t * pm[1])
            for ps, pm in zip(pts_sc, pts_mg)
        ]
        ws = warp_image_mesh(sc, pts_sc, mid_pts, triangles, w, h)
        wm = warp_image_mesh(mg, pts_mg, mid_pts, triangles, w, h)
        blended_face = Image.blend(ws, wm, t)

        # Composite morphed face onto Martin Scorsese's body/background
        frame = Image.composite(blended_face, sc, mask_feather)
        frame.save(os.path.join(frames_dir, f"frame_{idx:04d}.png"))
        idx += 1

    # Hold female face on Scorsese
    wm_end = warp_image_mesh(mg, pts_mg, pts_mg, triangles, w, h)
    end_frame = Image.composite(wm_end, sc, mask_feather)
    for _ in range(HOLD_END):
        end_frame.save(os.path.join(frames_dir, f"frame_{idx:04d}.png"))
        idx += 1

    print(f"Generated {idx} frames in {frames_dir}.")

    # Generate updated filmstrip
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
