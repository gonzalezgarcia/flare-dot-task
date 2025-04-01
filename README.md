# FLARE Dot Task Stimuli

This repository contains scripts and conda environment specifications used for generating stimuli (Mooney images with dots) for the FLARE project's "dot task".

## Repository Structure
- **environment/**: Contains conda environment specification (`dot_task_env.yml`)
- **scripts/**: Python scripts for stimulus creation
- **stimuli/**: Folder structure to store stimulus images and related masks

## Setup Instructions
Create the conda environment using:
```bash
conda env create -f environment/dot_task_env.yml
conda activate dot_task_env
