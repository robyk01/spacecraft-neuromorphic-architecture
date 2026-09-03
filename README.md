# Spacecraft Neuromorphic Architecture

Research and prototyping repository for onboard satellite image processing and land cover classification using BrainChip Akida neuromorphic hardware (SNN).

This environment serves as the R&D playground for the Synapse satellite payload subsystem, focused on developing low-power, noise-robust spiking neural network models compiled directly to hardware binaries.

---

## Overview

The pipeline processes high-resolution satellite imagery acquired during flight, preprocess it, slices it into 64x64 patches, and classifies land cover across 10 EuroSAT classes.

Key objectives:
- Train CNN baselines optimized for integer quantization (bounded ReLU6 activations).
- Apply Quantization-Aware Training (QAT) to map weights and activations to 4-bit / 8-bit integer synapses.
- Compile models via `cnn2snn` into Akida hardware binaries (`.fbz`) for on-chip inference on the satellite payload.


---

## Key Contributions (Chronological Order)

### 1. Flight Inference Pipeline (`infer.py`, `infer_all.py`)
- Implemented high-resolution image preprocessing: boundary ceiling calculations, zero-padding for edge mismatches, and 64x64 patch grid slicing (`slice_image`, `prepare_flight_batch`).
- Built the on-device Akida hardware inference runtime and batch evaluation engine (`infer_all.py`) with per-class accuracy and latency tracking.
- Eliminated CPU thermal bottlenecks on the flight computer by replacing PNG disk serialization with in-memory NumPy binary transfers (batch.npy) and transitioning from patch-by-patch driver calls to single-shot batch inference on the Akida NPU.

### 2. Dataset Synthesis & Degradation Modeling (`data_augment.ipynb`)
- Built the synthetic distorted and mixed test image datasets for validation (`test_images_distorted/`, `test_images_mixed/`).
- Modeled realistic spaceborne physical disturbances: Rayleigh scattering (haze), solar glare saturation, spacecraft attitude motion blur, optical defocus blur, partial cloud covers, and radiation/sensor Gaussian noise.

### 3. Lightweight Clarity Filter
- Implemented a lightweight CPU pre-filter in Python to drop unclassifiable, zero-signal patches before routing to the Akida NPU:
  - Variance of Laplacian for motion/defocus blur.
  - HSV saturation and brightness thresholds for atmospheric haze.
  - Quadrant-based intensity and variance checks for dense cloud decks.
- Shifted filtering overhead away from the neural network, improving mixed dataset accuracy from 70.20% to 75.72%.

### 4. Stochastic On-the-Fly Noise Injection & Retraining
- Translated degradation operations into pure TensorFlow tensor math and integrated stochastic, multi-layer noise injection directly into the `tf.data` training stream (`dataset.py`).
- Retrained the baseline CNN and executed Quantization-Aware Training (QAT) to bake intrinsic optical robustness directly into the 4-bit SNN synaptic weights.
- Compiled and exported the final flight model (`new_attuned.fbz`), reaching 81.26% on mixed images and 67.13% on raw distorted images.

---

## Repository Structure

```
├── docs/                     # Architecture, software, and hardware documentation
├── src/
│   ├── core/                 # Modular ML pipeline
│   │   ├── akida_model.py    # Akida conversion, hardware mapping, and SNN validation
│   │   ├── config.py         # Dataclass configurations (Dataset, Model, Train, Quantize)
│   │   ├── dataset.py        # TFDS EuroSAT loader, splits, and noise augmentations
│   │   ├── evaluation.py     # Metrics calculation and confusion matrix generation
│   │   ├── keras_model.py    # Baseline CNN architecture definition
│   │   ├── paths.py          # Artifact and checkpoint path resolution
│   │   ├── quantization.py   # QuantizeML calibration and QAT fine-tuning
│   │   └── training.py       # Compilation and training loop
│   ├── data/                 # Datasets, mock test batches, and exported models
│   ├── inference/
│   │   ├── infer.py          # Single-image patch slicer and Akida inference runner
│   │   └── infer_all.py      # Batch benchmark runner with clarity filter
│   ├── land_classification.py# Standalone training script
│   └── requirements.txt      # Dependency specifications
└── README.md
```

---

## Setup & Execution

### Installation
Python 3.9 - 3.11 is required for Akida MetaTF compatibility.

```bash
pip install -r src/requirements.txt
```

### Running Model Training & Akida Compilation
```bash
python src/train.py --run-name eurosat_robust_v1 --epochs 30 --show-prediction
```

### Running Test Inference
```bash
python src/inference/infer_all.py
```
