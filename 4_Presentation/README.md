# Image Classification: Failure Mode Detection

**Course:** Machine Vision with TensorFlow  
**Date:** 22.01.2026  
**Authors:** Niklas Kohlmann, Johannes Münderlein, Aadip Thapaliya  

---

## 1. Introduction

This project focuses on automated defect detection and classification of microelectronic components using Convolutional Neural Networks (CNNs) and transfer learning.

### Objective

- Determine whether a component is functioning.
- If defective, classify the specific type of defect.

### Data Source

Dataset and challenge provided via:

https://challengedata.ens.fr/participants/challenges/157/

The task involves classification of grayscale images of microelectronic components from a French-Chinese electronics company.

---

## 2. Literature Review

We reviewed relevant literature in industrial defect detection and CNN-based classification.

| Paper | Objective | Approach | Key Findings | Relevance |
|--------|------------|------------|---------------|------------|
| A Comprehensive Review of CNNs for Defect Detection | Industrial defect detection | CNNs, transfer learning | Overview of models & best practices | Methodological foundation |
| Image-based defect detection in lithium-ion battery electrodes | Battery electrode defects | CNNs, transfer learning | Up to 0.99 F1 score | Strong similarity to our task |
| Neural Network for SEM Image Recognition | SEM image classification | CNNs, transfer learning | 85–95% accuracy | Comparable grayscale image data |

Typical ImageNet model accuracies: **75–85%**

---

## 3. Dataset Characteristics

- ~8300 grayscale images
- Resolution range: ~500x500 px to 1200x1200 px
- 8-bit grayscale
- Labels provided via CSV file
- Images linked by filename

### Available Features

- `Image` – raw image data  
- `Lib` – type of component (die)  
- `Window` – year  

No manual feature engineering was performed.

### Preprocessing

- Normalization
- Resolution downsampling
- Optional resizing depending on model architecture

---

## 4. Image Classes

| Class | Description | Number of Images |
|--------|------------|------------------|
| 0_GOOD | Fully functioning component | 1235 |
| 1_Flat loop | Bridge laying flat instead of arching | 71 |
| 2_White lift-off | Bridge arch appears brighter | 270 |
| 3_Black lift-off | Bridge arch appears darker | 104 |
| 4_Missing | Parts of bridge missing | 6472 |
| 5_Short circuit MOS | Unwanted electrical contact | 126 |
| 6_Drift | Other faulty / damaged parts | ~55 (separate dataset) |

### Observations

- Highly imbalanced dataset
- Several classes are visually very similar
- Drift class not present in training set

---

## 5. Baseline Model

### Architecture

- 1 Convolutional layer
- MaxPooling
- 1 Dense layer
- Output layer

### Training Setup

- Image size: 128x128 px
- No data augmentation
- Balanced dataset

### Performance

- Precision: 0.85
- Recall: 0.85
- F1-score: 0.85

Despite simplicity, performance was surprisingly strong.

---

## 6. Model Architectures Compared

| Model | Architecture | Parameters |
|--------|--------------|------------|
| Simple CNN 01 | 1 Conv + 1 Dense | 89,719,878 |
| Simple CNN 02 | 2 Conv + 1 Dense | 43,674,886 |
| Simple CNN 03 | 3 Conv + 2 Dense | 20,171,846 |
| InceptionV3 Transfer | InceptionV3 + dropout + classifier | 21,815,078 |
| EfficientNetV2S Transfer | EfficientNetV2S + dropout + classifier | 20,339,046 |

---

## 7. Evaluation Strategy

### Requirements

- No defective part should be classified as functioning.
- No functioning part should be classified as defective.

### Metric Priorities

- For class `0_GOOD`: Precision is most important.
- For defect classes: Recall is most important.
- Main evaluation metric: **F1 score**

### Training Controls

- Early stopping
- Monitoring `val_loss` and `val_F1`
- Full classification reports analyzed

---

## 8. Best Performing Model

### Architecture

InceptionV3 Transfer Learning

- ImageNet pretrained weights
- Target resolution: 299x299 px
- Full dataset (unbalanced)
- Class weights applied
- Mild data augmentation

### Multi-Phase Training

1. Train classification head only  
2. Unfreeze last 30 layers  
3. Unfreeze last 100 layers  

Learning rate schedule:
- High LR, few epochs (initial)
- Medium training phase
- Low LR, longer training with patience

---

## 9. Final Results (Best Model)

| Label | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|----------|
| 0_GOOD | 0.99 | 0.99 | 0.99 | 238 |
| 1_Flat loop | 0.85 | 0.73 | 0.79 | 15 |
| 2_White lift-off | 0.93 | 0.97 | 0.95 | 57 |
| 3_Black lift-off | 0.87 | 1.00 | 0.93 | 13 |
| 4_Missing | 1.00 | 0.99 | 0.99 | 1316 |
| 5_Short circuit MOS | 0.89 | 0.94 | 0.91 | 17 |

Overall accuracy: **0.99**

Macro average F1-score: **0.93**  
Weighted average F1-score: **0.99**

---

## 10. Model Comparison (F1 Scores)

| Model | F1 Score |
|--------|----------|
| S-CNN 01 | 0.84 |
| S-CNN 02 | 0.90 |
| S-CNN 03 | 0.79 |
| InceptionV3 (feature extraction) | 0.72 |
| InceptionV3 (multi-phase fine-tuning) | 0.93 |
| EfficientNetV2S (multi-phase) | 0.85 |

Multi-phase transfer learning with InceptionV3 performed best.

---

## 11. Drift Class Handling

### Problem

- No drift images in training data
- Only appear in test sets
- No true labels available

### Approach

1. Manually labeled drift images
2. Combined labels 0–5 into "regular"
3. Trained binary classifier:
   - Drift vs Regular
4. Transfer learning from best-performing model

### Result

Drift F1-score: **0.4**

Drift detection remains challenging.

---

## 12. Challenges

- Augmentation reduced performance
- Transfer learning required multi-phase training
- Early stopping sometimes triggered unpredictably
- Validation metrics initially too optimistic
- Low performance on:
  - Drift class
  - Flat loop class

---

## 13. Discussion

- When can model performance be trusted?
- Does performance generalize to independent datasets?
- Results strongly depend on:
  - Hyperparameters
  - Training methodology
  - Dataset balance

Computer Vision remains a strong and practical application of machine learning.

---

## 14. Conclusion & Outlook

### Achievements

- Successful defect classification using CNNs
- High F1-score with transfer learning
- Multi-phase fine-tuning significantly improved performance

### Future Work

- Use transformer-based architectures
- Automated hyperparameter optimization
- Custom loss functions
- Improved drift detection methods
- Methods for low-data regimes

---

## Thank You

Any questions?
