
# 🧾 Stimulus Generation Procedure – FLARE Dot Task

This document describes the process implemented in the `batch_dot_generator.py` script used to generate dot-overlaid Mooney and grayscale images for the FLARE project. It includes segmentation via SAM, controlled dot placement, and full output logging.

---

## 1. 🔧 Script Configuration

### Directories:
- `stimuli/original/` — RGB images used for manual segmentation (clicks)
- `stimuli/gray/` — Disambiguated grayscale versions (same filenames with `_gray`)
- `stimuli/mooney/` — Mooney-style versions (same filenames with `_mooney`)
- `stimuli/output_dots/` — Output images with dots + metadata

### Parameters:
- `num_dots = 6`: number of dots per condition (`on` and `off`) for each image type
- `circle_radius_ratio = 0.5`: radius of the invisible circle (as a proportion of the image size)
- `dot_radius = 15`: visual dot radius in pixels

---

## 2. 🧠 Segmentation with SAM

- The user manually clicks on the object(s) within the RGB version of each image.
- The SAM model (`sam_vit_h_4b8939.pth`) generates binary masks for each click.
- The resulting mask is saved as `image_mask.png`.
- A masked object with **transparent background** is also saved as `image_object.png`.

---

## 3. 🔴 Dot Placement Logic

### 3.1 Circle Generation
- A fixed number (`sampling_points = 36`) of candidate dot positions are generated evenly around a circle centered in the image.

### 3.2 Classification
- Each candidate dot is classified as `on` or `off` based on whether it falls within the object mask.

### 3.3 Filtering Criteria

#### ✅ Applied to ALL Dots:
- `filter_near_edges`: ensures dots are not too close to image borders (based on `dot_radius + margin`)

#### ✅ Additional for `on` (inside object):
- `filter_near_mask_edges`: removes dots too close to the **edge of the object** (based on distance transform of the object)

#### ✅ Additional for `off` (outside object):
- `filter_away_from_object`: removes dots too close to the **object** (based on inverse distance transform)

---

## 4. 🎯 Dot Sampling

- After filtering, 12 dots are randomly sampled:
  - 6 for Mooney (3 on + 3 off)
  - 6 for Gray (3 on + 3 off)

- Dot locations are selected using `np.random.choice` to ensure spatial diversity (rather than even angular spacing).

---

## 5. 🖼 Preview and Manual Correction

- If not enough valid dot positions are found, a slider appears to allow manual adjustment of the circle radius.
- Valid candidate positions are shown for visual inspection.

---

## 6. 💾 Output Files Per Image

For each processed image:

### 🗃 Output Images:
- `image_mooney_dot_on1.jpg` … `on3`
- `image_mooney_dot_off1.jpg` … `off3`
- `image_gray_dot_on1.jpg` … `on3`
- `image_gray_dot_off1.jpg` … `off3`

### 🖼 Also saved:
- `image_mask.png` — binary mask from SAM
- `image_object.png` — RGBA image with only the segmented object

### 🧾 Metadata:
- `metadata.json`: includes click points, circle radius, and coordinates of all dots
- `master_log.csv`: master file aggregating dot coordinates for all images

---

## 7. 🧪 Reprocessing & Control

- The script can be run with a `--force` flag to reprocess already-completed images.
- Process is idempotent and logs progress at each step.
