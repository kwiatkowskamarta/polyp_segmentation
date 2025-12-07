# polyp_segmentation: Polyp Segmentation using U-Net (PyTorch)


## Project Overview
Colorectal cancer is a leading cause of cancer-related mortality. Early detection of polyps during colonoscopy is critical for prevention.

This project implements a U-Net deep learning pipeline to automatically segment polyps from colonoscopy images. Built entirely in Python (PyTorch), the system features a modular codebase designed for reproducibility and efficiency on standard hardware.


## Key Features
* Modular Architecture: Clean separation of dataset logic (`src/dataset.py`) and model definition (`src/model.py`).
* Resource Optimized: Input resolution tuned to `128x128` to enable efficient training on CPU-only environments without sacrificing architectural integrity.
* Reproducible Pipeline: Automated data fetching scripts handle the Kvasir-SEG dataset acquisition and preprocessing.
* Medical Metrics: Evaluates performance using the Dice Coefficient (F1-Score), the standard metric for biomedical segmentation.

## Repository Structure
```text
polyp-segmentation/
├── data/                  # Dataset storage (Ignored by Git)
│   └── Kvasir-SEG/        # Raw images and masks
├── src/
│   ├── __init__.py
│   ├── dataset.py         # Custom PyTorch Dataset class
│   └── model.py           # U-Net Architecture implementation
├── results/               # Generated visualization of predictions
├── download_data.py       # Auto-fetch script for Kvasir-SEG
├── train.py               # Main training loop with logging
├── evaluate.py            # Inference script + Visualization
├── .gitignore             # Git configuration
└── README.md              # Project Documentation
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/kwiatkowskamarta/polyp_segmentation.git
cd polyp_segmentation
```
### 2. Set up Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
```bash
pip install torch torchvision opencv-python matplotlib tqdm requests
```

## Usage
### Data Acquisition
Run the automated script to fetch the Kvasir-SEG dataset from Simula Research Laboratory. This script handles SSL verification and unzipping automatically.
```bash
python download_data.py
```
### Training
Initiate the training loop. By default, this is configured for 3 Epochs on 128x128 images to ensure quick validation on CPU environments.
```bash
python train.py
```
### Evaluation & Visualization
Run inference on the validation set to calculate the Dice Score and generate comparison images.
```bash
python evaluate.py
```

