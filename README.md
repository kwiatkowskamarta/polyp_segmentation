# polyp_segmentation: Polyp Segmentation using U-Net (PyTorch)


## Project Overview
Colorectal cancer is a leading cause of cancer-related mortality. Early detection of polyps during colonoscopy is critical for prevention.

This project implements a **U-Net** deep learning pipeline to automatically segment polyps from colonoscopy images. Built entirely in **Python (PyTorch)**, the system features a modular codebase designed for reproducibility and efficiency on standard hardware.


## Key Features
* Modular Architecture:** Clean separation of dataset logic (`src/dataset.py`) and model definition (`src/model.py`).
* **Resource Optimized: Input resolution tuned to `128x128` to enable efficient training on CPU-only environments without sacrificing architectural integrity.
* Reproducible Pipeline: Automated data fetching scripts handle the Kvasir-SEG dataset acquisition and preprocessing.
* Medical Metrics: Evaluates performance using the Dice Coefficient (F1-Score), the standard metric for biomedical segmentation.

## Repository Structure
```text
├── data/                  # Dataset storage (auto-generated)
├── src/
│   ├── dataset.py         # Custom PyTorch Dataset with preprocessing
│   └── model.py           # U-Net Architecture Implementation
├── results/               # Generated prediction visualizations
├── download_data.py       # Automated data fetching script
├── train.py               # Training loop with validation
├── evaluate.py            # Inference and Dice Score calculation
└── README.md              # Project documentation

## Results
The model was trained for 3 epochs on a CPU environment with `128x128` resolution.

* Training Loss: Converged to 0.39
* Validation Dice Score: 0.57 (Intersection over Union measure)
* Inference Speed: ~50ms per image (CPU)

This performance demonstrates the model's ability to learn semantic features of polyps even under constrained computational resources.

