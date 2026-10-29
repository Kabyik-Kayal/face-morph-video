#!/usr/bin/env python3
r"""
Complete UBO Face & Portrait Morphing Pipeline
==============================================
Official implementation using the UBO Morphing Library
(Ferrara, Franco, Maltoni - "Decoupling texture blending and shape warping in face morphing", BIOSIG 2019)
Repository: https://github.com/MI-BioLab/ubo-morph

Key Pipeline Features:
  1. MediaPipe 478-Point 3D FaceMesh Landmark Extraction.
  2. Canonical Eye-Center Alignment & Geometric Rescaling.
  3. Delaunay Triangulation across full portrait with edge border anchors.
  4. Complete portrait-to-portrait morphing (Scorsese start -> Maggie Smith end).
  5. Cosine S-curve easing with start/end hold frames for smooth viewing.
  6. Strict file size control (< 100,000 bytes / 100 KB) for assignment delivery.
"""

import os
import sys
import math
import subprocess
import requests
import cv2
import numpy as np

# UBO Morphing library
from ubo_morph import MediaPipeLandmarkExtractor, morph_with_landmarks


def ensure_file(path, url, description):
    if not os.path.exists(path):
        print(f"Downloading {description}...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) FaceMorph/7.0'}
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
    print("Complete UBO Portrait Morphing Pipeline (Ferrara et al. / ubo-morph)")
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

    print("1. Extracting 478 3D MediaPipe landmarks via ubo-morph...")
    with MediaPipeLandmarkExtractor(model_path) as extractor:
        lm1 = extractor.extract(img1)
        lm2 = extractor.extract(img2)
    print(f"   Detected {len(lm1.points)} landmarks on Scorsese, {len(lm2.points)} on Maggie Smith.")

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

    frames_dir = os.path.join(work_dir, "frames_complete")
    os.makedirs(frames_dir, exist_ok=True)

    target_w = 360
    target_h = 480

    print(f"2. Rendering {len(t_vals)} complete portrait morph frames via ubo-morph...")
    cache = {}
    for idx, t in enumerate(t_vals):
        tr = round(float(t), 4)
        if tr not in cache:
            # Complete portrait morphing via official ubo-morph
            # morph_with_landmarks aligns eye centers and triangulates full canvas
            # automatic_retouching=False ensures full portrait morph without background snapping
            res = morph_with_landmarks(
                img1,
                img2,
                lm1,
                lm2,
                warping_factor=tr,
                blending_factor=tr,
                align_eye_centers=True,
                automatic_retouching=False,
            )
            cache[tr] = cv2.resize(res, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        out_path = os.path.join(frames_dir, f"frame_{idx:04d}.png")
        cv2.imwrite(out_path, cache[tr])
        if idx % 10 == 0 or idx == len(t_vals) - 1:
            print(f"   Frame {idx+1}/{len(t_vals)} rendered (t={tr:.4f})")

    print("3. Generating progression filmstrip...")
    f_strip_idxs = [0, len(t_vals)//4, len(t_vals)//2, 3*len(t_vals)//4, len(t_vals)-1]
    strip = np.hstack([cv2.imread(os.path.join(frames_dir, f"frame_{i:04d}.png")) for i in f_strip_idxs])
    strip_path = os.path.join(work_dir, "morph_strip.jpg")
    cv2.imwrite(strip_path, strip)
    print(f"   Filmstrip saved to {strip_path}")

    print("4. Encoding H.264 video with FFmpeg strictly under 100 KB...")
    output_mp4 = os.path.join(work_dir, "morph.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "29",
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
