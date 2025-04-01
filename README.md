# FLARE Dot Task Stimuli

This repository contains scripts and conda environment specifications used for generating stimuli (Mooney images with dots) for the FLARE project's "dot task".

# Dot Stimulus Generator 🧠🔴

This tool allows you to semi-automatically generate visual stimuli using:
- Click-based segmentation (SAM)
- Dot placement on/off objects
- Mooney and grayscale variants
- Metadata export in JSON + CSV

## 📂 Folder Structure
```
stimuli/
├── original/      # RGB images used for segmentation
├── gray/          # Disambiguated grayscale images
├── mooney/        # Mooney-style versions
├── output_dots/   # Generated stimuli + metadata
models/
└── sam_vit_h_4b8939.pth  # Segment Anything model
```
## Before running, download the following
### 🔗 Model Weights

Download the SAM model checkpoint from the official repo and place it in the `models/` folder:

- [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
- File: `sam_vit_h_4b8939.pth`

Or use this direct link:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P models/
```

- Add it to folder: `models/`



### 📷 Original Images
- Download the original images from [THINGS database](https://osf.io/jum2f/files/osfstorage/670d66e48092b2004c2ecbfe).
- Download [THINGS-Mooney](https://github.com/wobc/things-mooney)
- Place your original images in the `stimuli/original/` folder.
- Place the Mooney images in the `stimuli/mooney/` folder.
- Place the grayscale images in the `stimuli/gray/` folder.


## ▶️ Usage
```bash
python batch_dot_generator.py
python batch_dot_generator.py --force
```



## 📋 Output
- Stimulus images with red dots
- `metadata.json` per image
- `master_log.csv` (all dots across images)
- Final mask as `.png`

## 📦 Requirements
```bash
# Create the environment from file
conda env create -f environment.yml
conda activate dotstim-env
```

## ✨ Features
- Slider-based dot rescue when not enough positions found
- JSON+CSV metadata
- Skips completed images unless `--force` is used