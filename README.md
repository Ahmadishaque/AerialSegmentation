# 🛰️ Aerial Semantic Segmentation using Deep Learning

## 📘 Overview
This project tackles **semantic segmentation of aerial imagery** captured via drones, focusing on **scene understanding** across multiple urban and natural landscapes. The work was carried out as part of a **Kaggle competition** aimed at advancing **automated land-cover classification** — a key problem in **remote sensing, urban planning, and environmental monitoring**.

Leveraging **PyTorch** and **Segmentation Models PyTorch (SMP)**, this research-driven implementation explores deep learning architectures for **pixel-level classification** into 12 semantic categories (including roads, vegetation, buildings, and water bodies).

---

## 🎯 Objectives
- Develop a **custom dataset class** for aerial image segmentation.
- Implement **data augmentation** and **visualization pipelines** using Albumentations.
- Evaluate model performance using **Dice Coefficient** and **IoU metrics**.
- Train and validate a **UNet model with a ResNet50 encoder** pre-trained on ImageNet.
- Generate a **submission-ready CSV** for Kaggle evaluation using **Run-Length Encoding (RLE)**.

---

## 🧠 Research Motivation
Semantic segmentation of aerial imagery bridges **computer vision** and **geospatial analysis**, enabling efficient large-scale terrain interpretation. The research intent of this project is to:
- Investigate the **generalization ability** of deep segmentation networks on varying drone perspectives.
- Assess **model robustness** under limited labeled data and high inter-class imbalance.
- Benchmark **UNet-based architectures** for aerial scene segmentation under **resource-constrained** GPU environments.

---

## 🧩 Dataset Description
- **Total images:** 3,269  
- **Number of classes:** 12 (including background)  
- **Image source:** Drone-based aerial captures at various altitudes and scales  
- **Split:** Custom 80:20 train-validation split (stratified by background ratio)

**Semantic Classes:**
`background`, `building`, `road`, `water`, `barren`, `forest`, `agriculture`, `brushland`, `baseball_diamond`, `basketball_court`, `football_field`, `tennis_court`

---

## 🧰 Tech Stack
| Category | Tools & Frameworks |
|-----------|--------------------|
| Programming Language | Python 3.12 |
| Deep Learning | PyTorch, Segmentation Models PyTorch |
| Data Augmentation | Albumentations |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, OpenCV, PIL |
| Environment | Jupyter Notebook (Vocareum Labs) |

---

## 🧪 Methodology

### 1. Data Preparation
- Designed a **custom PyTorch Dataset** class for seamless data loading.
- Applied **extensive augmentations** including flips, rotations, brightness/contrast adjustments, and coarse dropout for generalization.
- Resized all images and masks to a uniform **384×512** resolution.

### 2. Model Architecture
- Implemented **U-Net with ResNet50 encoder**, leveraging **ImageNet pretrained weights** for feature extraction.
- Combined **Cross-Entropy Loss** and **Dice Loss** to mitigate class imbalance and stabilize training.

### 3. Evaluation Metrics
- **Mean Dice Coefficient (DSC)**:  
  \( DSC = \frac{2|X ∩ Y|}{|X| + |Y|} \)
- **Intersection over Union (IoU)** per class for granular analysis.

### 4. Training Setup
- **Optimizer:** Adam (LR = 1e-3)  
- **Scheduler:** OneCycleLR  
- **Batch Size:** 8  
- **Epochs:** 70  
- **Early Stopping:** Applied patience of 15 epochs  

Performance plots include training and validation loss curves, mean Dice evolution, and per-class IoU trajectories.

---

## 📊 Results

| Metric | Validation Score |
|--------|------------------|
| Mean Dice Coefficient | **0.8898** |
| Mean IoU (across 12 classes) | **0.85 (approx.)** |

Sample inference visualizations show strong agreement between predicted and ground-truth segmentation masks, even across complex scenes with multiple object classes.

---

## 📤 Submission Generation
The final predictions were encoded into **Run-Length Encoding (RLE)** format to comply with Kaggle’s submission requirements.

- **Output:** `submission.csv`
- **Format:**

```bash

ImageID,EncodedPixels
9743172603816335085_0,383 2 32639 2 33023 2 ...
9743172603816335085_1,95056 7 95437 11 ...

```
- **Total rows:** 7,776 (12 masks per image)

---

## 💾 Key Learnings
- Importance of **class balancing** in multi-class segmentation.
- **Combined loss functions** significantly improved convergence stability.
- Demonstrated the impact of **data augmentation** on generalization performance.
- Explored **efficient training strategies** for GPU memory optimization.

---

## 🧩 Future Work
- Experiment with **Transformer-based segmentation models (e.g., SegFormer, DeepLabV3+)**.
- Integrate **attention mechanisms** to enhance fine-grained boundary detection.
- Explore **semi-supervised learning** approaches to leverage unlabeled aerial data.

---

## 📎 Kaggle Competition
- **Platform:** Kaggle (Semantic Segmentation Challenge)  
- **Profile Link:** https://www.kaggle.com/ahmadishaque

---

## 👨‍💻 Author

**Ahmad Ishaque Karimi**  
Graduate Student — Data Science & Computer Vision Research  
📧 ahmadishaquekarimi@gmail.com  
🔗 [LinkedIn][[https://www.linkedin.com/in/Ahmadishaque](https://www.linkedin.com/in/ahmadishaquekarimi/)]

---

> “The power of aerial vision lies not in seeing from above, but in understanding the world below.”
