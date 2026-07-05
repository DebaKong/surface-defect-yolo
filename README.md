# 🔍 Surface Defect Detection using YOLOv8

## 📌 Project Overview

This project implements an end-to-end **industrial surface defect detection system** based on **YOLOv8**.  
It focuses on detecting and classifying surface defects in the **NEU-DET dataset**, which contains 6 types of steel surface defects.

The pipeline includes:
- Data preprocessing (VOC XML → YOLO format)
- Dataset cleaning and validation
- Object detection model training (YOLOv8)
- Model evaluation (Precision / Recall / mAP)
- Error analysis (False Positive / False Negative / Class Confusion)
- Inference visualization

---

## 📊 Dataset

**NEU-DET (Steel Surface Defect Dataset)**

Classes:
- crazing
- inclusion
- patches
- pitted_surface
- rolled-in_scale
- scratches

Total:
- 1800 images (train + validation)

---

## ⚙️ Pipeline

### 1. Data Preprocessing
- Parsed XML annotations
- Converted bounding boxes into YOLO format
- Normalized coordinates to [0,1]
- Verified label consistency via visualization

### 2. Data Validation
- Checked label format correctness
- Verified bounding box ranges
- Visual inspection of random samples

### 3. Model Training
- Model: YOLOv8n (Ultralytics)
- Epochs: 20
- Image size: 640
- Training environment: Kaggle GPU

### 4. Evaluation Metrics
- Precision
- Recall
- F1-score
- mAP@0.5
- Confusion Matrix

### 5. Error Analysis
Performed detailed analysis on:
- False Negatives (missed detections)
- False Positives (incorrect detections)
- Class-level confusion patterns

---

## 📈 Results

The model achieves stable convergence on validation set with balanced precision and recall performance.

Key observations:
- Strong performance on high-contrast defects
- Weak performance on low-texture defect types (e.g., crazing)
- Some confusion between visually similar defect classes

---

## 🔍 Inference Demo

Example inference output:

- Input: raw steel surface image  
- Output: bounding boxes with class labels + confidence scores  

YOLOv8 successfully detects multiple defect types in a single image.

---

## 📷 Visualizations

- Training loss curves
- Precision-Recall curves
- Confusion matrix
- Prediction visualizations

---

## 🧠 Key Insights

- Data quality validation is critical before training
- Small object defects require higher resolution or stronger augmentation
- Class imbalance affects detection performance
- Error analysis is essential for model improvement

---

## 🚀 Tech Stack

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Matplotlib
- Kaggle GPU

---

## 📁 Project Structure
surface-defect-yolo/
│
├── dataset/
├── scripts/
│ ├── prepare-and-predict-neu-surface-defect-database.ipynb
│ 
├── runs/
├── data.yaml
├── README.md

---

## 🔥 Future Improvements

- Compare YOLOv8n vs YOLOv8s performance
- Apply data augmentation ablation study
- Improve detection of low-contrast defects
- Deploy model as a web demo (Streamlit)

---

## 📌 Author Notes

This project demonstrates a complete **computer vision pipeline** for industrial defect detection, covering data engineering, model training, evaluation, and error analysis.

It is suitable for:
- CV Engineer roles
- Machine Learning Engineer roles
- Deep Learning internships
