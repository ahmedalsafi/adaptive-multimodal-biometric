# Adaptive Multimodal Biometric Identification

**Paper:** Adaptive Multimodal Biometric Identification: A Lightweight CNN Approach with Confidence-Based Score Fusion  
**Journal:** International Journal of Intelligent Engineering and Systems (IJIES) — Paper ID: 20262767  
**Authors:** Ahmed Al-Safi, Yaghoub Farjami — University of Qom

---

## Repository Contents

| File | Description |
|------|-------------|
| `biometric_system_complete.ipynb` | Complete pipeline: training, evaluation, all tables and figures |
| `requirements.txt` | Required Python packages |

---

## Quick Start

1. Open `biometric_system_complete.ipynb` in Google Colab
2. Mount Google Drive and set correct data paths
3. Set `TRAIN = False` to reproduce paper results (recommended)
4. Set `TRAIN = True` to retrain both models from scratch (~100 epochs each)
5. Run all cells (Runtime → Run all)

---

## Dataset

**MULBv1** — 176 subjects, face and hand images.

| Split | Samples |
|-------|---------|
| Train | 2,464 face + 2,464 hand |
| Validation | 528 face + 352 hand |
| Test | 528 face + 528 hand (paired) |

---

## CNN Architecture

Both face and hand models share the same lightweight architecture:
- Conv2D(32) → MaxPool(4×4)
- Conv2D(16) → MaxPool(2×2)
- Conv2D(8)  → MaxPool(2×2)
- Flatten → Dense(512) → Dropout(0.5) → Dense(256) → Dropout(0.5) → Dense(176, softmax)
- **592,776 parameters (~2.26 MiB)**
- Optimizer: Adam (lr=0.001) | Loss: categorical cross-entropy | Epochs: 100 | Batch: 32

---

## Key Results

| System | Rank-1 Accuracy | Error Rate | Hand Invocations |
|--------|----------------|-----------|-----------------|
| Face Only | 96.40% | 3.60% | — |
| Hand Only | 99.24% | 0.76% | — |
| Traditional Fusion | 100.00% | 0.00% | 1,056/1,056 |
| **Proposed Adaptive** | **99.43%** | **0.57%** | **21/528 (3.98%)** |

**Computational savings: 48.01%** at confidence threshold τ = 0.95  
**AUC:** 0.9999 (macro-averaged one-vs-rest, 176 classes)

---

## Threshold Selection

The confidence threshold τ = 0.95 was selected on the **validation set** using the Kneedle method (geometric knee detection), confirmed stable by bootstrap resampling (1,000 resamples). The threshold was never tuned on the test set.

---

## Runtime Benchmarks

Measured on Intel Core i7-8650U @ 1.90 GHz, 8 GB RAM, CPU-only, TensorFlow 2.21.0.

| Format | File Size | Adaptive Avg | Speedup |
|--------|-----------|-------------|---------|
| Keras (float32) | 2.30 MiB | 1.86 ms | 1.89× |
| TFLite (float32) | 2.27 MiB | 1.87 ms | 1.86× |
| TFLite (quantized) | 586 KiB | 1.66 ms | 1.86× |

---

## Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- tensorflow==2.21.0
- opencv-python
- numpy
- scikit-learn
- matplotlib
- scipy

---

## Pre-trained Models

Pre-trained models are available on request from the corresponding author:
- `face_model_final_fixedfeb.keras` — Face recognition model (96.40% accuracy)
- `hand_model_final_v2.keras` — Hand recognition model (99.24% accuracy)

---

## Notes

- All images are pre-processed and stored in the split folders
- Face images: grayscale, 200×200
- Hand images: pre-processed (Otsu thresholding + morphological filtering), grayscale, 200×200
- The notebook runs in ~15-20 minutes with `TRAIN = False`
