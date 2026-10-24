#!/usr/bin/env python3
"""
Face Morphing Pipeline: Martin Scorsese to Maggie Smith
Creates an ultra-smooth, high quality face morph video encoded under 100 KB (100,000 bytes).
"""

import os
import sys
import math
import subprocess
import requests
from PIL import Image, ImageDraw

# ─────────────────────────────────────────────────────────────────────────────
# 1. GEOMETRY & DELAUNAY TRIANGULATION
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
    """
    Bowyer-Watson algorithm for Delaunay Triangulation in 2D.
    Takes a list of (x, y) tuples and returns a list of (i, j, k) triangle vertex indices.
    """
    pts = [Point(x, y) for x, y in points]
    min_x = min(p.x for p in pts) - 100
    max_x = max(p.x for p in pts) + 100
    min_y = min(p.y for p in pts) - 100
    max_y = max(p.y for p in pts) + 100

    # Initial super-triangle bounding all points
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

    # Remove triangles containing super-triangle vertices
    return [tri for tri in triangles if not (tri[0] in st_indices or tri[1] in st_indices or tri[2] in st_indices)]

# ─────────────────────────────────────────────────────────────────────────────
# 2. PIECEWISE AFFINE WARPING
# ─────────────────────────────────────────────────────────────────────────────

def solve_affine(src_tri, dst_tri):
    """
    Computes affine transformation matrix parameters (a, b, c, d, e, f)
    mapping destination triangle coordinates (x, y) back to source (u, v):
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
# 3. FACE ALIGNMENT BY EYES
# ─────────────────────────────────────────────────────────────────────────────

def get_affine_align(eye_l, eye_r, target_l, target_r):
    """Similarity transform mapping target canonical eye positions to source image coordinates."""
    dx_t = target_r[0] - target_l[0]
    dy_t = target_r[1] - target_l[1]
    dist_t = math.hypot(dx_t, dy_t)
    ang_t = math.atan2(dy_t, dx_t)

    dx_s = eye_r[0] - eye_l[0]
    dy_s = eye_r[1] - eye_l[1]
    dist_s = math.hypot(dx_s, dy_s)
    ang_s = math.atan2(dy_s, dx_s)

    scale = dist_s / dist_t
    theta = ang_s - ang_t

    cos_t = scale * math.cos(theta)
    sin_t = scale * math.sin(theta)

    a = cos_t
    b = -sin_t
    c = eye_l[0] - cos_t * target_l[0] + sin_t * target_l[1]
    d = sin_t
    e = cos_t
    f = eye_l[1] - sin_t * target_l[0] - cos_t * target_l[1]
    return (a, b, c, d, e, f)

# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 60)
    print("Face Morphing Pipeline: Martin Scorsese -> Maggie Smith")
    print("=" * 60)

    work_dir = os.path.dirname(os.path.abspath(__file__))
    frames_dir = os.path.join(work_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    sc_path = os.path.join(work_dir, "scorsese.jpg")
    mg_path = os.path.join(work_dir, "maggie.jpg")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) FaceMorph/1.0'}

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

    print("Loading and normalizing images...")
    s_orig = Image.open(sc_path).convert('RGB')
    m_orig = Image.open(mg_path).convert('RGB')

    # Resize previews for fast canonical extraction
    s_preview = s_orig.resize((400, int(400 * s_orig.height / s_orig.width)), Image.Resampling.LANCZOS)
    m_preview = m_orig.resize((400, int(400 * m_orig.height / m_orig.width)), Image.Resampling.LANCZOS)

    CANVAS_W, CANVAS_H = 240, 280
    T_EYE_L = (80, 115)
    T_EYE_R = (160, 115)

    # Eye positions in preview images:
    # Scorsese: Left Eye (140, 217), Right Eye (254, 206)
    # Maggie:   Left Eye (158, 232), Right Eye (262, 232)
    aff_s = get_affine_align((140, 217), (254, 206), T_EYE_L, T_EYE_R)
    aff_m = get_affine_align((158, 232), (262, 232), T_EYE_L, T_EYE_R)

    im_s = s_preview.transform((CANVAS_W, CANVAS_H), Image.Transform.AFFINE, aff_s, Image.Resampling.BICUBIC)
    im_m = m_preview.transform((CANVAS_W, CANVAS_H), Image.Transform.AFFINE, aff_m, Image.Resampling.BICUBIC)

    im_s.save(os.path.join(work_dir, "scorsese_aligned.jpg"))
    im_m.save(os.path.join(work_dir, "maggie_aligned.jpg"))

    # Canonical 63-point landmark geometry (aligned coordinate space)
    landmarks_s = [
        # Eyes
        (80, 115), (97, 114), (61, 116), (80, 107), (80, 122),
        (160, 115), (143, 114), (179, 116), (160, 107), (160, 122),
        # Brows
        (50, 95), (78, 86), (108, 92),
        (132, 92), (162, 86), (190, 95),
        # Nose
        (120, 112), (120, 130), (120, 150), (102, 153), (138, 153), (120, 160),
        # Mouth
        (85, 168), (155, 166), (120, 168), (120, 174), (120, 186),
        # Chin & Jaw
        (120, 218), (100, 212), (140, 212), (60, 185), (180, 185),
        # Cheeks & Temples
        (55, 145), (185, 145), (48, 115), (192, 115),
        # Forehead & Hair
        (120, 60), (70, 55), (170, 55),
        (120, 15), (55, 25), (185, 25),
        (30, 90), (210, 90),
        # Neck & Collar
        (70, 245), (170, 245), (120, 255),
        # Outer boundary anchors (prevent edge distortions)
        (0, 0), (60, 0), (120, 0), (180, 0), (239, 0),
        (0, 70), (239, 70),
        (0, 140), (239, 140),
        (0, 210), (239, 210),
        (0, 279), (60, 279), (120, 279), (180, 279), (239, 279)
    ]

    landmarks_m = [
        # Eyes
        (80, 115), (98, 117), (62, 114), (80, 107), (80, 122),
        (160, 115), (142, 117), (178, 114), (160, 107), (160, 122),
        # Brows
        (52, 97), (80, 84), (105, 94),
        (135, 94), (160, 84), (188, 97),
        # Nose
        (120, 114), (120, 136), (120, 168), (105, 168), (135, 168), (120, 175),
        # Mouth
        (93, 198), (147, 198), (120, 196), (120, 200), (120, 208),
        # Chin & Jaw
        (120, 240), (100, 232), (140, 232), (65, 205), (175, 205),
        # Cheeks & Temples
        (60, 155), (180, 155), (46, 115), (194, 115),
        # Forehead & Hair
        (120, 60), (70, 55), (170, 55),
        (120, 15), (50, 20), (190, 20),
        (20, 90), (220, 90),
        # Neck & Collar
        (70, 255), (170, 255), (120, 260),
        # Outer boundary anchors
        (0, 0), (60, 0), (120, 0), (180, 0), (239, 0),
        (0, 70), (239, 70),
        (0, 140), (239, 140),
        (0, 210), (239, 210),
        (0, 279), (60, 279), (120, 279), (180, 279), (239, 279)
    ]

    # Compute Delaunay Triangulation on average landmark coordinates
    avg_pts = [((ps[0] + pm[0]) / 2, (ps[1] + pm[1]) / 2) for ps, pm in zip(landmarks_s, landmarks_m)]
    triangles = delaunay_triangulation(avg_pts)
    print(f"Constructed {len(triangles)} Delaunay triangles across {len(avg_pts)} landmark vertices.")

    N_MORPH = 40      # 40 smooth morphing steps
    HOLD_START = 6   # Hold initial face for viewer context
    HOLD_END = 6     # Hold target face
    FPS = 18

    print(f"Rendering {HOLD_START + N_MORPH + HOLD_END} frames with non-linear cosine easing...")
    frame_idx = 0

    # Hold start: Scorsese
    for _ in range(HOLD_START):
        im_s.save(os.path.join(frames_dir, f"frame_{frame_idx:04d}.png"))
        frame_idx += 1

    # Morph transition with smooth cosine easing
    for i in range(1, N_MORPH):
        linear_t = i / N_MORPH
        ease_t = (1.0 - math.cos(linear_t * math.pi)) / 2.0

        mid_pts = [
            ((1 - ease_t) * ps[0] + ease_t * pm[0], (1 - ease_t) * ps[1] + ease_t * pm[1])
            for ps, pm in zip(landmarks_s, landmarks_m)
        ]
        warped_s = warp_image_mesh(im_s, landmarks_s, mid_pts, triangles, CANVAS_W, CANVAS_H)
        warped_m = warp_image_mesh(im_m, landmarks_m, mid_pts, triangles, CANVAS_W, CANVAS_H)
        blended = Image.blend(warped_s, warped_m, ease_t)
        blended.save(os.path.join(frames_dir, f"frame_{frame_idx:04d}.png"))
        frame_idx += 1

    # Hold end: Maggie
    for _ in range(HOLD_END):
        im_m.save(os.path.join(frames_dir, f"frame_{frame_idx:04d}.png"))
        frame_idx += 1

    total_frames = frame_idx
    print(f"Successfully generated {total_frames} frames.")

    # Encode with FFmpeg
    output_mp4 = os.path.join(work_dir, "morph.mp4")
    ffmpeg_cmd = [
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

    print("Encoding MP4 video with FFmpeg...")
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    size_bytes = os.path.getsize(output_mp4)
    size_kb = size_bytes / 1000.0

    print("\n" + "=" * 60)
    print(f"Video generated: {output_mp4}")
    print(f"File size: {size_bytes:,} bytes ({size_kb:.2f} KB)")
    print(f"Target limit: < 100,000 bytes (100.00 KB)")
    if size_bytes < 100000:
        print("[PASS] File is well within the 100 KB limit!")
    else:
        print("[FAIL] File exceeds 100 KB limit! Increase CRF.")
    print("=" * 60)

if __name__ == "__main__":
    main()
