# Official UBO Face & Portrait Morph: Ferrara et al. (MI-BioLab/ubo-morph)

> **Assignment Question 7:** Face & portrait morphing of Martin Scorsese into Maggie Smith using the official **UBO Morphing Algorithm** from Ferrara, Franco, & Maltoni (*"Decoupling texture blending and shape warping in face morphing"*, IEEE BIOSIG 2019).  
> **Upstream Library:** [https://github.com/MI-BioLab/ubo-morph](https://github.com/MI-BioLab/ubo-morph)  
> **Direct Public Video URL:** [https://raw.githubusercontent.com/Kabyik-Kayal/face-morph-video/main/morph.mp4](https://raw.githubusercontent.com/Kabyik-Kayal/face-morph-video/main/morph.mp4)  
> **Interactive Web Viewer:** [https://kabyik-kayal.github.io/face-morph-video/](https://kabyik-kayal.github.io/face-morph-video/)

---

## Deliverables & Constraints Verification

| Metric | Required | Achieved | Status |
| :--- | :--- | :--- | :--- |
| **Direct HTTPS URL** | Publicly accessible | `https://raw.githubusercontent.com/Kabyik-Kayal/face-morph-video/main/morph.mp4` | Verified |
| **File Size** | **< 100,000 bytes** | **91,358 bytes** (~91.4 KB) | **PASS** (< 100 KB limit) |
| **Algorithm Base** | Official UBO Morph | [MI-BioLab/ubo-morph](https://github.com/MI-BioLab/ubo-morph) | Ferrara et al. (2019) |
| **Portrait Transition** | Complete portrait ending | **100% Scorsese &rarr; 100% Maggie Smith** (full face, hair, clothing, background) | Complete & Authentic |
| **Container & Codec** | Playable in browser | MP4 (`H.264 / AVC`, `yuv420p`, progressive) | 100% Browser Compatible |
| **Fast Web Start** | Smooth streaming | `+faststart` (moov atom at file start) | Instant playback |
| **Resolution** | Crisp & detailed | **360 x 480 px** | Clean portrait framing |
| **Landmark Tracking** | Sub-pixel accuracy | **MediaPipe 478-Point 3D FaceMesh** | Pluggable UBO extractor |
| **Eye Alignment** | Canonical interpupillary distance | UBO `_align_face_images` | Zero eye jitter |
| **Warping Engine** | Anti-aliased mesh | Delaunay (970 triangles, `cv2.LINE_AA` + `blendLinear`) | Zero triangle seams |
| **Pacing** | Natural & seamless | Cosine Ease-In Ease-Out (`18 fps`, 3.28s) | Smooth S-curve |

---

## Morph Progression Filmstrip

![Face Morph Progression Filmstrip](morph_strip.jpg)

*Progression (left to right):*
1. **0% (Frame 0):** 100% Martin Scorsese complete portrait (glasses, open smile, suit, tie, dark festival backdrop).
2. **25% (Frame 14):** Early UBO morphing (brows adjusting, smile narrowing, facial and torso contours smoothly adapting).
3. **50% (Frame 29):** Midpoint — Canonical 50/50 UBO morphed face and portrait with balanced texture blending.
4. **75% (Frame 44):** Late UBO morphing (Maggie Smith's facial features and vintage hair resolving into sharp focus).
5. **100% (Frame 58):** 100% Maggie Smith complete portrait (vintage hair, features, black dress, light background).

---

## Methodology & Architecture

```mermaid
flowchart TD
    subgraph S1 ["1. Preprocessing & Alignment"]
        A["Scorsese & Maggie High-Res Photos"] --> B["MediaPipe 478 3D Landmark Extractor"]
        B --> C["UBO Eye-Center Alignment<br/>(_align_face_images)"]
        C --> D["Canonical Aligned Portraits"]
    end

    subgraph S2 ["2. UBO Triangulation & Anchoring"]
        D --> E["Image Border Anchors (5 pts/edge)<br/>+ Overlap Removal"]
        E --> F["Delaunay Triangulation (970 Triangles)<br/>(_delaunay_triangles)"]
    end

    subgraph S3 ["3. Anti-Aliased Mesh Warping"]
        F --> G["Piecewise Affine Triangle Warping<br/>(backend.warp_triangles with LINE_AA)"]
        G --> H["Decoupled Texture Blending<br/>(backend.blend with Cosine Easing)"]
    end

    subgraph S4 ["4. Complete Output Encoding"]
        H --> I["Full Portrait-to-Portrait Transition<br/>(Scorsese 100% &rarr; Maggie Smith 100%)"]
        I --> J["FFmpeg H.264 &lt; 100 KB Encoding<br/>(CRF 29, yuv420p, +faststart)"]
    end
```

---

## Technical Details: The UBO Morphing Algorithm

The pipeline is implemented using the official UBO face morphing algorithm (`ubo-morph`):
* **Decoupling Texture and Shape:** Unlike naive morphs that conflate pixel cross-fades with geometric distortion, UBO morphing treats shape warping and texture blending as independent parameters.
* **Canonical Eye-Center Alignment:** Source images are scaled and translated such that the interpupillary line is horizontally centered at standard eye distance, eliminating rotation or scale discrepancies.
* **Anti-Aliased Triangle Blending:** Triangles are rasterized with `cv2.LINE_AA` and composited using `cv2.blendLinear`, preventing black seam lines or tears characteristic of basic Delaunay implementations.
* **Full Portrait Continuity:** By morphing across the full Delaunay mesh with border anchors (`automatic_retouching=False`), the entire portrait transitions seamlessly from Martin Scorsese into Maggie Smith without artificial edge cuts or background jumps.

---

## Local Reproduction

### Prerequisites
* Python 3.10+
* `ffmpeg` in system PATH
* Requirements: `ubo-morph`, `mediapipe`, `opencv-contrib-python`, `numpy`, `tqdm`, `requests`

```bash
git clone https://github.com/Kabyik-Kayal/face-morph-video.git
cd face-morph-video
pip install -r requirements.txt
python morph.py
```

Output video is generated as `morph.mp4` (~91.4 KB).

---

## Citation

```bibtex
@inproceedings{ferrara2019decoupling,
  title={Decoupling texture blending and shape warping in face morphing},
  author={Ferrara, Matteo and Franco, Annalisa and Maltoni, Davide},
  booktitle={2019 International Conference of the Biometrics Special Interest Group (BIOSIG)},
  pages={1--5},
  year={2019},
  organization={IEEE}
}
```
