
# Stimulus Construction for the FLARE Dot Task

This document describes the computational procedure used in the FLARE project to construct visual stimuli involving Mooney and grayscale images with controlled dot overlays. The method integrates segmentation using the Segment Anything Model (SAM), systematic dot placement along an invisible circular path, and spatial filtering to ensure robust control over dot locations. This semi-automated procedure enables reproducible generation of stimuli for investigating perceptual and abstract visual processing in behavioral and neuroimaging experiments.

---

## Materials and Setup

### Image Categories

Each stimulus comprises three types of image:
- **Original RGB**: full-color images used for segmentation.
- **Grayscale ("gray")**: gray scale versions of the original images
- **Mooney**: two-tone ambiguous images derived from the grayscale ones.

Images are named consistently and stored in structured folders.

### Environment and Models

The procedure utilizes the Segment Anything Model (SAM, `vit_h` checkpoint). Segmentation is driven by user-provided clicks on the RGB image, which are then used to isolate the object of interest. A binary mask and a transparent-background RGBA image of the object are generated and saved for each trial.

---

## Procedure

### Manual Segmentation

Participants click on the central object(s) in the original RGB image. These coordinates are passed to SAM to produce a segmentation mask that isolates the foreground object.

### Dot Sampling Circle

An invisible circle is constructed **at the center of each image**, with a radius **proportional** to the image size (`circle_radius_ratio`). A fixed number of sample positions (e.g., 36) are generated evenly around this circle.

### Classification and Filtering

Each candidate position is classified as:
- **ON**: within the object mask.
- **OFF**: outside the object mask.

To ensure perceptual clarity and avoid confounds:
- Dots too close to the **image borders** are removed.
- ON dots too close to the **object edge** are removed using a distance transform.
- OFF dots too close to the **object edge** are likewise excluded using an inverse distance transform.

### Dot Sampling

From the filtered positions, twelve dots are randomly sampled:
- 6 for the Mooney image (3 ON, 3 OFF)
- 6 for the Grayscale image (3 ON, 3 OFF)

This sampling avoids forced angular separation, allowing for natural variability while preserving balance across conditions.

---

## Visualization and Interaction

If too few valid dot positions remain after filtering, a visual slider allows the user to adjust the radius of the dot circle. A live preview updates accordingly, with valid candidate positions visualized in yellow.

---

## Outputs

For each input image, the following are generated:

- `*_gray_dot_on*.jpg`, `*_gray_dot_off*.jpg`
- `*_mooney_dot_on*.jpg`, `*_mooney_dot_off*.jpg`
- `*_mask.png`: binary segmentation mask
- `*_object.png`: masked object with transparent background

### Metadata:
- A `metadata.json` file with click points, dot coordinates, and radius.
- A `master_log.csv` containing all dot positions across stimuli for downstream analysis or reconstruction.

---

## Reproducibility and Scaling

The procedure is fully scriptable and compatible with batch processing. An optional `--force` flag allows users to overwrite existing outputs if needed. All parameters (e.g., dot size, number, distance filters) are easily configurable, and outputs are logged consistently for reproducibility.

---

## Conclusion

This pipeline provides a robust, semi-automated method for producing visual stimuli with precise dot-based manipulations. It supports experimental designs requiring strong control over perceptual features, such as those used in ambiguity resolution, insight generation, or attentional guidance paradigms.

