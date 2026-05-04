# 🧠 CNN Dementia Classification

A **Convolutional Neural Network (CNN)** trained to classify brain MRI images as either **Healthy** or **Mild Dementia** using TensorFlow and Keras.

> **89% test accuracy · AUC: 0.96 · Binary Classification · Deep Learning**

---

## 📌 Project Overview

This project applies deep learning to the early detection of dementia using MRI scans. A custom CNN was designed and trained to distinguish between healthy patients and those with mild dementia. The model uses convolutional and pooling layers to extract spatial features from grayscale MRI images, with dropout regularization and early stopping to prevent overfitting.

---

## 📊 Results

| Metric | Healthy (0) | Mild Dementia (1) |
|---|---|---|
| Precision | 0.9121 | 0.8686 |
| Recall | 0.8715 | 0.9100 |
| F1-Score | 0.8913 | 0.8888 |
| **Overall Accuracy** | **0.8901** | |
| **AUC** | **0.9588** | |

### Training Curves

<p align="center">
  <img src="images/Accuracy.png" width="45%" alt="Accuracy Curve"/>
  <img src="images/loss.png" width="45%" alt="Loss Curve"/>
</p>

### Evaluation

<p align="center">
  <img src="images/cnf_matrix.png" width="45%" alt="Confusion Matrix"/>
  <img src="images/auc.png" width="45%" alt="ROC Curve"/>
</p>

<p align="center">
  <img src="images/eval_metrics.png" width="60%" alt="Metric Comparison by Class"/>
</p>

---

## 🏗️ Model Architecture

The CNN consists of the following layers:

```
Input (180x180 grayscale MRI image)
  │
  ├── Conv2D(32, 3x3, ReLU) → MaxPooling2D(2x2)
  ├── Conv2D(64, 3x3, ReLU) → MaxPooling2D(2x2)
  ├── Conv2D(128, 3x3, ReLU) → MaxPooling2D(2x2)
  │
  ├── Flatten
  ├── Dense(128, ReLU) → Dropout(0.5)
  ├── Dense(64, ReLU)  → Dropout(0.5)
  └── Dense(1, Sigmoid)  ← Binary output
```

**Compiled with:**
- Loss: Binary Cross-Entropy
- Optimizer: Adam
- Early Stopping: patience=3, monitors `val_loss`, restores best weights

---

## 📁 Dataset

**Alzheimer Disease Augmented MRI Dataset** — sourced from Kaggle.

🔗 [View Dataset on Kaggle](https://www.kaggle.com/datasets/ashrafulhossenakash/alzheimer-disease-dataset/data)

The dataset is organized into `train/`, `val/`, and `test/` folders with two class subdirectories:
- `Healthy/` (Class 0)
- `MildDementia/` (Class 1)

Images were preprocessed using TensorFlow:
- Resized to **180×180**
- Converted to **grayscale**
- Pixel values **normalized to [0, 1]**
- Training set **shuffled** per epoch; val/test sets kept static for consistent evaluation

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **scikit-learn** — metrics, ROC, confusion matrix
- **NumPy / Pandas**
- **Matplotlib**

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/cnn-dementia-classification.git
cd cnn-dementia-classification
```

### 2. Install dependencies
```bash
pip install tensorflow scikit-learn numpy pandas matplotlib
```

### 3. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/ashrafulhossenakash/alzheimer-disease-dataset/data) and organize it as:
```
data/
├── train/
│   ├── Healthy/
│   └── MildDementia/
├── val/
│   ├── Healthy/
│   └── MildDementia/
└── test/
    ├── Healthy/
    └── MildDementia/
```

### 4. Update the data paths in the notebook
Open `CNN_Model.ipynb` and update `train_dir`, `val_dir`, and `test_dir` to point to your local dataset.

### 5. Run the notebook
```bash
jupyter notebook CNN_Model.ipynb
```

---

## 📂 Repository Structure

```
cnn-dementia-classification/
├── CNN_Model.ipynb        # Main model notebook
├── images/
│   ├── Accuracy.png
│   ├── loss.png
│   ├── cnf_matrix.png
│   ├── auc.png
│   └── eval_metrics.png
└── README.md
```

---

## 📄 License

This project is for educational purposes. Dataset is subject to [Kaggle's terms of use](https://www.kaggle.com/terms).
