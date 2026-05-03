# 🧠 Brain Tumor Classification using CNN

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Project Overview

This project focuses on **brain tumor classification from MRI images** using Convolutional Neural Networks (CNNs).
The goal is to automatically detect the **presence or absence of a tumor** (Yes/No) from grayscale medical images.

---

## 📊 Dataset & Preprocessing

* Dataset: Brain MRI images (grayscale)
* Classes:

  * ✅ Tumor (Yes)
  * ❌ No Tumor (No)

### 🔀 Data Split

* Training: 70%
* Validation: 15%
* Test: 15%

### 🛠️ Preprocessing Steps

* Image resizing
* Normalization
* Data loading & batching

---

## 🧠 Model Architecture

The CNN model includes:

* Convolutional layers
* Max Pooling
* Fully Connected layers
* (Optional enhancements):

  * Residual Blocks
  * Depthwise Separable Convolutions

### 📈 Training Details

* Epochs: ~30
* Loss Function: CrossEntropyLoss
* Optimizer: Adam / SGD

Training and validation **loss & accuracy curves** are used to analyze model performance and detect overfitting.

---

## ⚙️ Optimization & Regularization

### 🔽 Learning Rate Scheduling

* Cosine Annealing
* ReduceLROnPlateau

### 🛡️ Regularization Techniques

* Early Stopping
* L2 Regularization
* Dropout

These techniques help improve generalization and reduce overfitting.

---

## 🚀 Transfer Learning

Pretrained models used:

* ResNet18
* SqueezeNet

### 🔧 Strategy

1. Freeze pretrained layers
2. Train classifier layers
3. Fine-tune entire network

---

## 📊 Results

The model is evaluated using:

* ✅ Accuracy
* 🎯 Precision
* 🔁 Recall
* ⚖️ F1-Score
* 📉 Confusion Matrix
* 📈 ROC Curve

> 📌 Results show improved performance after applying learning rate scheduling and regularization techniques, with further gains using transfer learning.

---


## 📌 Future Improvements

* Use larger and more diverse datasets
* Apply advanced architectures (e.g., EfficientNet, Vision Transformers)
* Hyperparameter tuning
* Deploy model as a web application



