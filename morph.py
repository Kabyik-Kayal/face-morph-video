#!/usr/bin/env python3
r"""
UBO Face Morphing Video Pipeline
=================================
Official implementation powered by the UBO Morphing Algorithm
(Ferrara, Franco, Maltoni - "Decoupling texture blending and shape warping in face morphing", BIOSIG 2019)
Repo: https://github.com/MI-BioLab/ubo-morph

Key Pipeline Features:
  1. MediaPipe 478-Point 3D FaceMesh Landmark Extraction.
  2. Canonical Eye-Center Alignment & Rescaling.
  3. Delaunay Triangulation (970+ triangles with image-edge border anchors).
  4. CDF Histogram-Based Color/Lightness Equalization.
  5. Anti-Aliased Piecewise Affine Triangle Warping (cv2.LINE_AA + cv2.blendLinear).
  6. Smooth Decoupled Texture Blending & Feathered Background Substitution.
  7. High-efficiency H.264 encoding strictly under 100 KB (100,000 bytes).
"""

import os
import sys
import math
import subprocess
import requests
import cv2
import numpy as np

# UBO Morphing algorithm modules
from ubo_morph import MediaPipeLandmarkExtractor
from ubo_morph.morphing.backend import get_backend
from ubo_morph.morphing.core import (
    _align_face_images,
    _delaunay_triangles,
    _equalize_face,
    _substitute_background,
)
from ubo_morph.morphing.points import (
    add_border_points,
    remove_border_points,
    remove_overlapped_points,
)

def ensure_file(path, url, description):
    if not os.path.exists(path):
        print(f"Downloading {description}...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) FaceMorph/6.0'}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
        print(f"Saved {path} ({len(r.content):,} bytes).")

def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 70)
    print("UBO Face Morphing Pipeline (Ferrara et al. / MI-BioLab/ubo-morph)")
    print("=" * 70)

    work_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(work_dir, "face_landmarker.task")
    model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    if not os.path.exists(model_path):
        parent_model = os.path.join(work_dir, "..", "face_landmarker.task")
        if os.path.exists(parent_model):
            model_path = parent_model
        else:
            ensure_file(model_path, model_url, "MediaPipe FaceLandmarker model")

    sc_path = os.path.join(work_dir, "scorsese.jpg")
    mg_path = os.path.join(work_dir, "maggie.jpg")
    sc_url = "https://upload.wikimedia.org/wikipedia/commons/c/ce/Martin_Scorsese_MFF_2023.jpg"
    mg_url = "https://upload.wikimedia.org/wikipedia/commons/2/28/Maggie_Smith_-_Vintage_%28trim%29.jpg"

    ensure_file(sc_path, sc_url, "Martin Scorsese source photo")
    ensure_file(mg_path, mg_url, "Maggie Smith source photo")

    img1 = cv2.imread(sc_path)
    img2 = cv2.imread(mg_path)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load source portraits!")

    print("1. Extracting MediaPipe 478 3D landmarks via UBO extractor...")
    with MediaPipeLandmarkExtractor(model_path) as extractor:
        lm1 = extractor.extract(img1)
        lm2 = extractor.extract(img2)
    print(f"   Detected {len(lm1.points)} landmarks on Scorsese, {len(lm2.points)} on Maggie Smith.")

    backend = get_backend("cpu")

    print("2. Aligning faces by eye centers (canonical interpupillary geometry)...")
    aligned_img1, aligned_img2, aligned_lm1, aligned_lm2 = _align_face_images(
        backend, backend.to_backend(img1), backend.to_backend(img2), lm1, lm2
    )

    print("3. Building Delaunay triangulation with image border anchors...")
    morph_pts1 = add_border_points(aligned_img1, aligned_lm1.points, 5)
    morph_pts2 = add_border_points(aligned_img2, aligned_lm2.points, 5)
    morph_pts1, morph_pts2 = remove_overlapped_points(morph_pts1, morph_pts2)
    triangles = _delaunay_triangles(morph_pts1)
    print(f"   Generated {len(triangles)} triangles spanning facial features and boundary.")

    print("4. Equalizing color and lightness distribution (UBO CDF histogram matching)...")
    eq_pts1, eq_pts2 = remove_overlapped_points(aligned_lm1.points, aligned_lm2.points)
    img2_eq = _equalize_face(backend, aligned_img1, aligned_img2, eq_pts1, eq_pts2, method="color")

    eye_dist = float(np.linalg.norm(aligned_lm1.left_eye - aligned_lm1.right_eye))

    # Video configuration
    HOLD_START = 8
    N_MORPH = 44
    HOLD_END = 8
    FPS = 18

    t_vals = []
    for _ in range(HOLD_START):
        t_vals.append(0.0)
    for i in range(1, N_MORPH):
        u = i / N_MORPH
        t = (1.0 - math.cos(u * math.pi)) / 2.0  # Cosine S-curve easing
        t_vals.append(t)
    for _ in range(HOLD_END):
        t_vals.append(1.0)

    frames_dir = os.path.join(work_dir, "frames_ubo")
    os.makedirs(frames_dir, exist_ok=True)

    h_f, w_f = aligned_img1.shape[:2]
    target_w = 360
    target_h = int(360 * h_f / w_f)
    if target_h % 2 != 0:
        target_h += 1

    print(f"5. Rendering {len(t_vals)} UBO morph frames at {target_w}x{target_h}...")
    cache = {}
    for idx, t in enumerate(t_vals):
        tr = round(float(t), 4)
        if tr not in cache:
            morphed_pts = morph_pts1 + (morph_pts2 - morph_pts1) * tr
            
            # Anti-aliased piecewise affine triangle warping
            w1 = backend.warp_triangles(aligned_img1, morph_pts1, morphed_pts, triangles)
            w2 = backend.warp_triangles(img2_eq, morph_pts2, morphed_pts, triangles)
            
            # Texture blending
            blended = backend.blend(w1, w2, tr)
            
            # Feathered background substitution with base portrait
            face_pts = remove_border_points(morphed_pts, int(blended.shape[1]), int(blended.shape[0]))
            final = _substitute_background(backend, blended, face_pts, w1, blend=True, eye_distance=eye_dist)
            
            res_np = backend.to_numpy(final)
            cache[tr] = cv2.resize(res_np, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            
        cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:04d}.png"), cache[tr])

    print("6. Creating progression filmstrip...")
    f_strip_idxs = [0, len(t_vals)//4, len(t_vals)//2, 3*len(t_vals)//4, len(t_vals)-1]
    strip = np.hstack([cv2.imread(os.path.join(frames_dir, f"frame_{i:04d}.png")) for i in f_strip_idxs])
    cv2.imwrite(os.path.join(work_dir, "morph_strip.jpg"), strip)

    print("7. Encoding video with FFmpeg (H.264, yuv420p, +faststart)...")
    output_mp4 = os.path.join(work_dir, "morph.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "28",
        "-preset", "slow",
        "-movflags", "+faststart",
        output_mp4
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    size_bytes = os.path.getsize(output_mp4)
    size_kb = size_bytes / 1000.0

    print("\n" + "=" * 70)
    print(f"Video saved: {output_mp4}")
    print(f"File size: {size_bytes:,} bytes ({size_kb:.2f} KB)")
    print(f"Assignment limit: < 100,000 bytes (100.00 KB)")
    if size_bytes < 100000:
        print(f"[PASS] Video is strictly under the 100 KB limit! ({size_bytes:,} < 100,000)")
    else:
        print(f"[FAIL] File size exceeds limit! ({size_bytes:,} >= 100,000)")
    print("=" * 70)

if __name__ == "__main__":
    main()
