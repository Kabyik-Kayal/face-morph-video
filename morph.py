#!/usr/bin/env python3
"""
High-Fidelity Face Morphing Pipeline
====================================
Morphs Maggie Smith's face seamlessly onto Martin Scorsese's portrait using:
  1. MediaPipe 478-Landmark 3D Mesh for sub-pixel facial feature tracking.
  2. Procrustes / Affine Similarity Normalization to a 384x480 canvas.
  3. Reinhard Color & Luminance Calibration in CIELAB color space.
  4. Continuous 2D Thin Plate Spline (TPS) with biharmonic energy minimization (C1 smooth).
  5. 3-Band Multiresolution Laplacian Pyramid Blending (eliminating ghosting and double teeth).
  6. Soft Anatomical Face-Oval Masking onto Martin Scorsese's stable head, hair, ears, suit, and background.

Output: morph.mp4 strictly smaller than 100 KB (100,000 bytes).
"""

import os
import sys
import math
import subprocess
import requests
import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def ensure_file(path, url, description):
    if not os.path.exists(path):
        print(f"Downloading {description}...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) FaceMorph/5.0'}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
        print(f"Saved {path} ({len(r.content):,} bytes).")

def setup_detector(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
    return vision.FaceLandmarker.create_from_options(options)

def get_landmarks(img_bgr, detector):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_img)
    if not result.face_landmarks:
        raise ValueError("No face detected by MediaPipe FaceLandmarker!")
    return np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]], dtype=np.float32)

def reinhard_tone_transfer(src_bgr, ref_bgr, mask_src=None, mask_ref=None):
    """
    Transfers color, luminance, and contrast distribution from ref to src in CIELAB space.
    Calibrates Maggie's vintage tone to match Scorsese's warm red-carpet lighting.
    """
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    if mask_src is not None and mask_ref is not None:
        src_vals = src_lab[mask_src > 0]
        ref_vals = ref_lab[mask_ref > 0]
    else:
        src_vals = src_lab.reshape(-1, 3)
        ref_vals = ref_lab.reshape(-1, 3)

    src_mean = np.mean(src_vals, axis=0)
    src_std = np.std(src_vals, axis=0) + 1e-6

    ref_mean = np.mean(ref_vals, axis=0)
    ref_std = np.std(ref_vals, axis=0) + 1e-6

    out_lab = (src_lab - src_mean) * (ref_std / src_std) + ref_mean
    out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)

def warp_tps(img, src_pts, dst_pts, w, h, step=8):
    r"""
    Computes global C1 continuous Thin Plate Spline deformation field.
    Solves biharmonic energy minimization: min \iint (\nabla^2 f)^2 dx dy.
    """
    rbf = RBFInterpolator(dst_pts, src_pts, kernel='thin_plate_spline', smoothing=0.08)
    gx, gy = np.meshgrid(np.arange(0, w + step, step), np.arange(0, h + step, step))
    coords = np.column_stack([gx.ravel(), gy.ravel()])
    mapped = rbf(coords)
    map_x = cv2.resize(mapped[:, 0].reshape(gy.shape).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    map_y = cv2.resize(mapped[:, 1].reshape(gy.shape).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)

def laplacian_blend(img_a, img_b, t, sigma1=2.5, sigma2=6.0):
    """
    Multiresolution 3-band Laplacian pyramid blending:
      - Low frequency (L2): smooth cosine lighting & skin tone transition.
      - Mid frequency (L1): facial contours, smile lines & muscle structure.
      - High frequency (L0): steep sigmoid handover for sharp features (teeth, pupils, eyelashes).
    """
    fa = img_a.astype(np.float32) / 255.0
    fb = img_b.astype(np.float32) / 255.0

    g1_a = cv2.GaussianBlur(fa, (0, 0), sigma1)
    g1_b = cv2.GaussianBlur(fb, (0, 0), sigma1)
    g2_a = cv2.GaussianBlur(g1_a, (0, 0), sigma2)
    g2_b = cv2.GaussianBlur(g1_b, (0, 0), sigma2)

    l0_a = fa - g1_a  # High freq: micro-edges, teeth, pores
    l0_b = fb - g1_b
    l1_a = g1_a - g2_a  # Mid freq: facial folds, contours
    l1_b = g1_b - g2_b
    l2_a = g2_a  # Low freq: ambient lighting, skin undertone
    l2_b = g2_b

    t_cos = (1.0 - math.cos(t * math.pi)) / 2.0
    t_high = 1.0 / (1.0 + math.exp(-16.0 * (t - 0.5)))

    b0 = (1.0 - t_high) * l0_a + t_high * l0_b
    b1 = (1.0 - t_cos) * l1_a + t_cos * l1_b
    b2 = (1.0 - t_cos) * l2_a + t_cos * l2_b

    out = b0 + b1 + b2
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)

def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 65)
    print("High-Fidelity Face Morphing: 478 Mesh + TPS + 3-Band Laplacian")
    print("=" * 65)

    work_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(work_dir, "face_landmarker.task")
    model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    if not os.path.exists(model_path):
        parent_model = os.path.join(work_dir, "..", "face_landmarker.task")
        if os.path.exists(parent_model):
            model_path = parent_model
        else:
            ensure_file(model_path, model_url, "MediaPipe FaceLandmarker model")

    detector = setup_detector(model_path)

    sc_path = os.path.join(work_dir, "scorsese.jpg")
    mg_path = os.path.join(work_dir, "maggie.jpg")
    sc_url = "https://upload.wikimedia.org/wikipedia/commons/c/ce/Martin_Scorsese_MFF_2023.jpg"
    mg_url = "https://upload.wikimedia.org/wikipedia/commons/2/28/Maggie_Smith_-_Vintage_%28trim%29.jpg"

    ensure_file(sc_path, sc_url, "Martin Scorsese source photo")
    ensure_file(mg_path, mg_url, "Maggie Smith source photo")

    sc_orig = cv2.imread(sc_path)
    mg_orig = cv2.imread(mg_path)
    if sc_orig is None or mg_orig is None:
        raise FileNotFoundError("Source images could not be loaded!")

    h_sc, w_sc = sc_orig.shape[:2]
    h_mg, w_mg = mg_orig.shape[:2]

    print("Detecting sub-pixel landmarks on source portraits...")
    lm_sc_raw = get_landmarks(sc_orig, detector)
    lm_mg_raw = get_landmarks(mg_orig, detector)

    # Compute eye centers for Procrustes similarity alignment
    eye_l_sc = (lm_sc_raw[33] + lm_sc_raw[133]) / 2.0
    eye_r_sc = (lm_sc_raw[362] + lm_sc_raw[263]) / 2.0
    eye_l_mg = (lm_mg_raw[33] + lm_mg_raw[133]) / 2.0
    eye_r_mg = (lm_mg_raw[362] + lm_mg_raw[263]) / 2.0

    # Canvas dimensions: 384x480 (4:5 portrait ratio, divisible by 16 for H.264 macroblocks)
    W_OUT, H_OUT = 384, 480
    dst_eyes = np.float32([[146.0, 182.0], [238.0, 182.0]])

    M_sc = cv2.estimateAffinePartial2D(np.float32([eye_l_sc, eye_r_sc]), dst_eyes)[0]
    M_mg = cv2.estimateAffinePartial2D(np.float32([eye_l_mg, eye_r_mg]), dst_eyes)[0]

    sc_norm = cv2.warpAffine(sc_orig, M_sc, (W_OUT, H_OUT), flags=cv2.INTER_LANCZOS4)
    mg_norm = cv2.warpAffine(mg_orig, M_mg, (W_OUT, H_OUT), flags=cv2.INTER_LANCZOS4)

    print("Extracting calibrated landmarks on normalized frames...")
    p_sc = get_landmarks(sc_norm, detector)
    p_mg = get_landmarks(mg_norm, detector)

    # Face Oval landmarks (MediaPipe)
    OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    # Oval masks for skin tone calibration
    mask_sc_oval = np.zeros((H_OUT, W_OUT), dtype=np.uint8)
    mask_mg_oval = np.zeros((H_OUT, W_OUT), dtype=np.uint8)
    cv2.fillPoly(mask_sc_oval, [p_sc[OVAL].astype(np.int32)], 255)
    cv2.fillPoly(mask_mg_oval, [p_mg[OVAL].astype(np.int32)], 255)

    print("Calibrating tone and luminance via Reinhard CIELAB transfer...")
    mg_norm = reinhard_tone_transfer(mg_norm, sc_norm, mask_mg_oval, mask_sc_oval)

    # Select comprehensive anatomical landmarks
    LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    PUPILS = [468, 473]
    NOSE = [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 19, 94, 218, 438, 220, 440]
    LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    CHEEKS_FOREHEAD = [116, 123, 147, 213, 138, 345, 352, 376, 433, 367, 151, 9, 8, 108, 337]

    key_idxs = list(dict.fromkeys(OVAL + LEFT_EYEBROW + RIGHT_EYEBROW + LEFT_EYE + RIGHT_EYE + PUPILS + NOSE + LIPS_OUTER + LIPS_INNER + CHEEKS_FOREHEAD))

    # Boundary pins ensure hair, ears, suit, collar, and background remain 100% stationary
    w, h = W_OUT, H_OUT
    pins = np.array([
        [0, 0], [w-1, 0], [0, h-1], [w-1, h-1],
        [w//4, 0], [w//2, 0], [3*w//4, 0],
        [0, h//4], [0, h//2], [0, 3*h//4],
        [w-1, h//4], [w-1, h//2], [w-1, 3*h//4],
        [w//4, h-1], [w//2, h-1], [3*w//4, h-1],
        # Hair anchors above forehead
        [90, 50], [192, 40], [290, 50],
        # Ear anchors
        [45, 180], [340, 180],
        # Neck anchors
        [90, 420], [192, 430], [290, 420]
    ], dtype=np.float32)

    pts_sc = np.vstack([p_sc[key_idxs], pins])
    pts_mg = np.vstack([p_mg[key_idxs], pins])

    # End state: Maggie's inner facial features mapped onto Scorsese's boundary
    pts_mg_in_sc = pts_sc.copy()
    for i, k in enumerate(key_idxs):
        if k not in OVAL:
            pts_mg_in_sc[i] = p_mg[k]

    frames_dir = os.path.join(work_dir, "frames_hq")
    os.makedirs(frames_dir, exist_ok=True)

    # Frame sequence: 8 hold start + 44 morph + 10 hold end = 62 frames @ 18 fps (3.44s)
    N_MORPH = 44
    HOLD_START = 8
    HOLD_END = 10
    FPS = 18
    idx = 0

    print(f"Rendering {HOLD_START + N_MORPH + HOLD_END} morph frames...")

    # 1. Hold Scorsese portrait
    for _ in range(HOLD_START):
        cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:04d}.png"), sc_norm)
        idx += 1

    # Feathered face oval mask inside Scorsese's face boundary
    oval_pts_sc = p_sc[OVAL].astype(np.int32)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(face_mask, [oval_pts_sc], 255)
    face_mask_feather = cv2.GaussianBlur(face_mask, (23, 23), 9).astype(np.float32) / 255.0
    face_mask_feather = face_mask_feather[:, :, None]

    # 2. Smoothly morph face features
    for i in range(1, N_MORPH):
        u = i / N_MORPH
        t = (1.0 - math.cos(u * math.pi)) / 2.0  # Cosine S-curve easing

        mid_pts = (1.0 - t) * pts_sc + t * pts_mg_in_sc

        ws = warp_tps(sc_norm, pts_sc, mid_pts, w, h, step=8)
        wm = warp_tps(mg_norm, pts_mg, mid_pts, w, h, step=8)

        blended = laplacian_blend(ws, wm, t)

        frame = (blended * face_mask_feather + sc_norm * (1.0 - face_mask_feather)).astype(np.uint8)
        cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:04d}.png"), frame)
        idx += 1

    # 3. Hold final female face on male portrait
    wm_end = warp_tps(mg_norm, pts_mg, pts_mg_in_sc, w, h, step=8)
    final_end = (wm_end * face_mask_feather + sc_norm * (1.0 - face_mask_feather)).astype(np.uint8)
    for _ in range(HOLD_END):
        cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:04d}.png"), final_end)
        idx += 1

    print(f"Generated {idx} frames in {frames_dir}.")

    # Generate filmstrip
    f0 = cv2.imread(os.path.join(frames_dir, "frame_0000.png"))
    f1 = cv2.imread(os.path.join(frames_dir, f"frame_{HOLD_START + N_MORPH//4:04d}.png"))
    f2 = cv2.imread(os.path.join(frames_dir, f"frame_{HOLD_START + N_MORPH//2:04d}.png"))
    f3 = cv2.imread(os.path.join(frames_dir, f"frame_{HOLD_START + 3*N_MORPH//4:04d}.png"))
    f4 = cv2.imread(os.path.join(frames_dir, f"frame_{idx-1:04d}.png"))

    strip = np.hstack([f0, f1, f2, f3, f4])
    cv2.imwrite(os.path.join(work_dir, "morph_strip.jpg"), strip)

    # Encode with FFmpeg (CRF 27 guarantees high definition strictly under 100 KB)
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

    print("\n" + "=" * 65)
    print(f"Video saved: {output_mp4}")
    print(f"File size: {size_bytes:,} bytes ({size_kb:.2f} KB)")
    print(f"Limit: < 100,000 bytes (100.00 KB)")
    if size_bytes < 100000:
        print(f"[PASS] Video is strictly under the 100 KB limit! ({size_bytes:,} < 100,000)")
    else:
        print(f"[FAIL] File size exceeds 100 KB limit! ({size_bytes:,} >= 100,000)")
    print("=" * 65)

if __name__ == "__main__":
    main()
