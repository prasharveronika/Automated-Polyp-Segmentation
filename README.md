# Gastric Cancer Segmentation using Deep Learning

A deep learning-based semantic segmentation pipeline for localizing regions of interest in gastrointestinal endoscopy images, with the goal of supporting clinical review.

## Problem

Identifying suspicious regions in endoscopy images can require detailed expert review. This project explores whether a deep learning segmentation model can automatically highlight regions of interest as a potential assistive tool, rather than replacing clinical assessment.

## Approach

* **Architecture:** U-Net with ResNet34 encoder pretrained on ImageNet
* **Implementation:** `segmentation_models`
* **Framework:** TensorFlow / Keras
* **Dataset:** Kvasir-SEG
* **Preprocessing:** Grayscale conversion, resizing to 512×512, mask normalization, and channel replication for 3-channel encoder input
* **Data Split:** Reproducible train/validation/test split using a fixed random seed
* **Loss:** Binary Cross-Entropy
* **Evaluation:** Custom Dice coefficient metric
* **Post-processing:** OpenCV contour detection to identify predicted regions and generate bounding-box visualizations

## Dataset and Experiment

The project was developed and evaluated using a **200-image subset of the Kvasir-SEG dataset** due to local computational constraints.

### Dataset Split

* **Training:** 128 images
* **Validation:** 32 images
* **Testing:** 40 images

The dataset itself is **not included in this repository**.

## Results

Results from the 200-image experiment:

| Metric           |    Result |
| ---------------- | --------: |
| Test Loss        | **0.349** |
| Test Accuracy    | **0.903** |
| Dice Coefficient | **0.585** |

Accuracy is included for completeness. The **Dice coefficient is particularly relevant for segmentation**, as it measures the overlap between predicted and ground-truth regions.

## Sample Output

The model generates:

**Input Image → Ground Truth Mask → Predicted Mask → Predicted Region with Bounding Box**

Sample output images can be added to the `results/` folder.

## Tech Stack

* Python
* TensorFlow
* Keras
* segmentation_models
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn

## Project Structure

```text
gastric-cancer-segmentation/
│
├── train.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── results/
│   └── sample_output.png
│
├── images/        # Local dataset - not uploaded
└── masks/         # Local dataset - not uploaded
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add the dataset

Place the Kvasir-SEG images and masks in the following folders:

```text
images/
masks/
```

The dataset is not included in this repository.

### 3. Run the training script

```bash
python train.py
```

The script loads the dataset, preprocesses the images and masks, trains the U-Net model, evaluates it on the test set, and generates segmentation visualizations.

## Status

**Developed and evaluated on a 200-image subset.**

Further experimentation can be performed with larger subsets when sufficient computational resources are available.

## Note

This project is an academic deep learning project focused on **image segmentation and localization of regions of interest**. It is not intended to provide a medical diagnosis or replace professional clinical assessment.
