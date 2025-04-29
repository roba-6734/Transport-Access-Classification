



# 🚀 Transportation Access Classifier

Welcome to the **Transportation Access Classifier** project — an efficient image classification system built using **TensorFlow** and **MobileNetV2**. This project applies transfer learning on a custom dataset to classify transportation access scenarios.

---

## 📌 Project Overview

- **Model Architecture:** MobileNetV2 (pre-trained on ImageNet)
- **Task:** Multi-class image classification
- **Dataset:** Custom dataset (organized in `train/` and `validation/` folders)
- **Frameworks:** TensorFlow, Keras
- **Platform:** CPU (no GPU acceleration used)

---

## 🧠 Objectives

- Apply transfer learning with a lightweight architecture
- Use image data augmentation for improved generalization
- Train and evaluate the model effectively
- Visualize training history and save the best model

---

## 🛠️ Key Steps

1. **Data Preparation**
   - Load training and validation datasets using `ImageDataGenerator`
   - Apply data augmentation to training data

2. **Model Construction**
   - Load MobileNetV2 without the top classification layer
   - Add custom dense layers for classification
   - Freeze base layers for transfer learning

3. **Training**
   - Compile model using Adam optimizer and categorical crossentropy
   - Train with validation monitoring
   - Use callbacks like `ModelCheckpoint` to save best-performing model

4. **Evaluation**
   - Evaluate accuracy and loss on validation data
   - Plot training and validation performance

---

## 📁 Directory Structure

```
project/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── validation/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── model_training.ipynb
```

---

## 💾 Saving and Using the Model

- The model is saved using `model.save('transportation_classifier.h5')`
- You can load it later with:
```python
from tensorflow.keras.models import load_model
model = load_model('transportation_classifier.h5')
```

---

## 📊 Visualization

- Training and validation accuracy/loss plots help monitor model behavior.
- You can visualize overfitting, underfitting, or ideal training dynamics.

---

## ✅ Requirements

- Python ≥ 3.10
- TensorFlow ≥ 2.0
- Keras (included with TensorFlow)
- NumPy, Matplotlib

Install dependencies with:
```bash
pip install tensorflow matplotlib
```

---

## 📬 Dataset

The dataset should be organized in the following structure:
```
dataset/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── validation/
│   ├── class_1/
│   ├── class_2/
│   └── ...
```



