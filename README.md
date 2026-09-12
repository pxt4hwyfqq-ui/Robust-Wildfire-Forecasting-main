# Robust Wildfire Forecasting under Partial Observability: From Reconstruction to Prediction

Official code repository for the paper:

**Robust Wildfire Forecasting under Partial Observability: From Reconstruction to Prediction**  
**Chen Yang, Mehdi Zafari, Ziheng Duan, and A. Lee Swindlehurst**

This repository contains the code for a two-stage learning framework for wildfire forecasting under partial observability. The framework first reconstructs plausible fire maps from corrupted historical observations and then performs next-day wildfire spread prediction using the recovered sequences.

## Overview

Satellite-derived wildfire observations are often incomplete due to cloud cover, smoke obscuration, and sensor artifacts. This repository implements the two-stage framework proposed in our paper to address this domain gap:

- **Stage I: Morphological Reconstruction**
  - Reconstructs plausible complete fire maps from partially observed inputs
  - Implemented with multiple architectures:
    - `MaskUNet`
    - `MaskCVAE`
    - `MaskViT`
    - `MaskD3PM`

- **Stage II: Spatiotemporal Prediction**
  - Predicts the next-day active fire map from reconstructed multi-day sequences
  - Implemented using a U-TAE-style spatiotemporal forecasting model

## Repository Structure

```text
.
├── data-preprocessing/
├── stage-1-reconstruction/
├── stage-2-forecasting/
└── requirements.txt
```

### Folder descriptions

- **data-preprocessing/**  
  Scripts for loading the raw WildfireSpreadTS (WSTS) dataset, preprocessing channels, applying cropping, feature engineering, normalization, and simulating partial observability masks.

- **stage-1-reconstruction/**  
  Code for training and evaluating the Stage I reconstruction models under pixel-wise and block-wise masking.

- **stage-2-forecasting/**  
  Code for training and evaluating the Stage II forecasting model using clean, corrupted, or reconstructed historical inputs.

---

## Dataset

This work uses the **WildfireSpreadTS (WSTS)** dataset, which is publicly available online:

**Dataset link:**  
https://zenodo.org/records/8006177

Please download the dataset from Zenodo and place it in the appropriate local data directory before running the preprocessing or training scripts.

### WSTS summary

The WSTS dataset is a large-scale multi-modal wildfire benchmark covering wildfire events in the Western United States from 2018 to 2021. Each sample includes 23 channels combining:

- VIIRS-based remote sensing observations
- meteorological variables
- topographic information
- land cover
- forecast variables
- active fire maps

In our pipeline, the original 23-channel data is transformed into a **42-channel model-ready representation** through:
- cyclical encoding of angular variables
- one-hot expansion of land-cover categories
- z-score normalization of continuous channels

We use:
- **5-day historical windows**
- **64 × 64 spatial crops**
- corruption settings with both **pixel-wise** and **block-wise** masking

---

## Installation

We recommend using **Python 3.10**.

### Create environment

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
or with Conda:
```bash
conda create -n wildfire-po python=3.10
conda activate wildfire-po
pip install -r requirements.txt
```

---

## Pipeline

### 1. Data preprocessing

The preprocessing stage prepares the WSTS data for reconstruction and forecasting. This includes:
- extracting fixed-length 5-day sequences
- cropping around active fire regions
- transforming 23 raw channels into 42 processed features
- generating corrupted fire maps for partial observability experiments


### 2. Stage I: Reconstruction

Train one of the reconstruction models to recover corrupted fire observations.

Supported models:
- `MaskUNet`
- `MaskCVAE`
- `MaskViT`
- `MaskD3PM`

### 3. Stage II: Forecasting

Train the spatiotemporal forecasting model using reconstructed or clean historical fire sequences.

---

## Experimental Settings

The paper evaluates the framework under:

- **two corruption types**
  - pixel-wise masking
  - block-wise masking

- **eight corruption levels**
  - 10%, 20%, ..., 80%

- **four fire scenarios**
  - FIRE CONTINUES
  - FIRE EXTINGUISHED
  - NEW FIRE WITH NO HISTORY
  - NO FIRE

- **leave-one-year-out cross-validation**
  - across four years of WSTS data

---

## Method Summary

The proposed framework decouples wildfire forecasting under partial observability into two steps:

1. Recover a plausible clean historical fire sequence from corrupted observations
2. Predict future wildfire spread from the reconstructed sequence

This decomposition helps reduce the domain gap between clean training data and degraded deployment-time observations.

---

## Citation

If you use this repository, please cite:

```bibtex
@article{yang2026robustwildfire,
 title={Robust Wildfire Forecasting under Partial Observability: From Reconstruction to Prediction}, 
 author={Chen Yang and Mehdi Zafari and Ziheng Duan and A. Lee Swindlehurst},
 year={2026},
 eprint={2603.09042},
 archivePrefix={arXiv},
 primaryClass={eess.IV},
 url={https://arxiv.org/abs/2603.09042}, 
}
```

---

## Acknowledgment

This repository is based on experiments conducted using the publicly available **WildfireSpreadTS (WSTS)** dataset. We thank the authors and maintainers of the dataset for making this benchmark available to the research community.

---

## Contact

For questions about the paper, codebase, or implementation details, please contact the authors through their institutional affiliations listed in the paper.

You may also open an issue in this repository for bug reports, questions, or suggestions related to the code.

---

© 2026 LS Wireless Group • University of California, Irvine • All rights reserved.