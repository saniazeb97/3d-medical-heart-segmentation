# 3D Heart Segmentation (UNet + Augmentations)

This repository contains a complete and reproducible pipeline for **3D medical image segmentation** using a **3D U-Net**, including:

- Dataset loading & preprocessing  
- Resampling 
- Geometric & intensity augmentations  
- Sliding-window inference  
- Evaluation (Dice, HD95, Sensitivity, Precision)  
- Visualizations & overlay results  

The code works **locally**, **on Google Colab**, and **inside containers**.

---

## 1. Installation

### **Option A — Local Machine**

Clone:
```bash
git clone https://github.com/saniazeb97/3D-Medical-Heart-Segmentation.git
cd heart-segmentation
```

Install:
```bash
pip install -r requirements.txt
```

---

### **Option B — Google Colab**

```python
!git clone https://github.com/<your-username>/heart-segmentation.git
%cd heart-segmentation
!pip install -r requirements.txt
```

---

## 2. Dataset Structure

```
heart_dataset/
├── train/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Images are .nii or .nii.gz files.

---

## 3. Environment Variables (optional)

```bash
export DATA_ROOT=/path/to/heart_dataset
export RESULTS_DIR=results
```

---

## 4. Training

Baseline:
```bash
python -m src.train --name baseline
```

Augmented:
```bash
python -m src.train --augment --name augmented
```

---

## 5. Inference & Evaluation

```bash
python -m src.inference
```

Outputs stored in:
```
results/                
├── logs/
├── predictions/
├── augmented_images/
├──plots/
└── checkpoints/
```

---

## 6. Visualizations

Run all visualizations:

```bash
python -m src.visualize
```

Generated files:
```
results/plots/training_curves_baseline.png
results/plots/training_curves_augmented.png
results/augmented_images/aug_examples_train.png
results/predictions/pred_vs_gt_baseline_*.png
results/predictions/pred_vs_gt_augmented_*.png
```

---

## 7. Repository Structure

```
.
├── README.md
├── report
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── io.py          # NIfTI IO, pair matching, resampling
│   │   ├── preprocess.py  # normalization, padding, pair preprocessing
│   │   ├── patches.py     # patch sampling utilities
│   │   └── pipelines.py   # tf.data pipelines
│   ├── augmentations.py   # geometric + intensity augmentations
│   ├── model.py           # 3D U-Net with residual blocks
│   ├── losses.py          # Dice, combo loss, LR scheduler
│   ├── metrics.py         # HD95, sensitivity, precision
│   ├── train.py           # Training entry point (baseline / augmented)
│   ├── inference.py       # Sliding-window inference + evaluation
│   └── plots.py   # plotting utilities 
│   └── visualize.py       # Generate Original vs Augmented slices + Training curves +  Test predictions
└── results/               # Created at runtime (metrics, predictions, curves)
    ├── logs/
    ├── predictions/
    ├── augmented_images/
    ├── plots/
    └── checkpoints/

```



