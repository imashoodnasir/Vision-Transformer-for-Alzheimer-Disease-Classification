
# DA-ViT: Deformable Attention Vision Transformer for Alzheimer’s Disease Classification from MRI Scans

This repository provides a modular and explainable deep learning pipeline to classify Alzheimer's disease stages using MRI scans via a **Deformable Multi-Head Self-Attention (DA-ViT)** Vision Transformer model.

---

## 🧠 Project Overview

DA-ViT enhances traditional Vision Transformers by incorporating deformable attention, allowing dynamic spatial focus on non-uniform brain regions affected by Alzheimer's disease. This project includes:

- Patch-based ViT backbone
- Deformable MHSA implementation
- End-to-end training and evaluation
- Bayesian hyperparameter optimization
- Attention-based explainability module

---

## 📁 Directory Structure

```
.
├── preprocess_data.py         # Dataset loading and preprocessing
├── vit_patch_embedding.py     # Image to patch embedding
├── deformable_mhsa.py         # Deformable multi-head attention
├── da_vit_model.py            # Full DA-ViT model definition
├── train.py                   # Training loop
├── evaluate.py                # Evaluation and confusion matrix / ROC
├── bayesian_opt.py            # Bayesian optimization
├── xai_module.py              # Grad-CAM / attention visualization
├── best_model.pth             # Trained model weights (if saved)
├── confusion_matrix.png       # Output: Confusion matrix visualization
├── roc_curves.png             # Output: ROC-AUC curves
└── attention_overlay.png      # Output: Attention heatmap
```

---

## 📦 Installation & Requirements

```bash
pip install torch torchvision scikit-learn matplotlib seaborn bayesian-optimization
```

---

## 🧪 Training the Model

```bash
python train.py
```

You can configure `epochs`, `lr`, `batch_size` inside `train.py`.

---

## 🔍 Evaluate and Visualize Performance

```bash
python evaluate.py
```

Generates:
- `confusion_matrix.png`
- `roc_curves.png`
- Full classification report

---

## 🤖 Hyperparameter Optimization with Bayesian Search

```bash
python bayesian_opt.py
```

Tunes:
- Learning rate
- Number of transformer blocks
- MLP expansion ratio

---

## 🔬 Visualize Attention Maps (XAI)

```bash
python xai_module.py
```

Generates:
- `attention_overlay.png` showing where the model focuses in the MRI image

---

## 📊 Metrics Tracked

- Accuracy, Precision, Recall, F1
- ROC-AUC for each class
- Confusion Matrix
- Attention Heatmap Overlay

---

## 📚 Acknowledgments

MRI data used from ADNI, OASIS, or Kaggle-based preprocessed AD datasets.

DA-ViT is inspired by ViT and deformable attention from DETR/Swin Transformer architectures.

---

## 🔒 License

For academic use only. Please cite the corresponding paper if this work is used in your research.
