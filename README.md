# Automated Polyp Segmentation using Deep Learning

A deep learning-based semantic segmentation pipeline that localizes polyps in gastrointestinal endoscopy images, with the goal of supporting clinical review.

## Problem

A polyp is an abnormal growth on the inner lining of the gastrointestinal tract. Small or flat polyps can be missed during endoscopy, and reviewing images manually is time-consuming and can vary between observers. This project explores whether a deep learning segmentation model can automatically highlight polyp regions as a potential assistive tool, rather than replacing clinical assessment.

## Approach

* **Architecture:** U-Net with ResNet34 encoder pretrained on ImageNet
* **Implementation:** `segmentation_models`
* **Framework:** TensorFlow / Keras
* **Dataset:** [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) (polyp images with ground-truth masks)
* **Preprocessing:** Grayscale conversion, resizing to 256×256, mask normalization and binarization, and channel replication for 3-channel encoder input
* **Data Split:** Reproducible train/validation/test split using a fixed random seed
* **Augmentation:** Random flips and rotations (training set only)
* **Loss:** Binary Cross-Entropy + Dice loss
* **Evaluation:** Dice coefficient, IoU and pixel accuracy
* **Training:** RMSprop optimizer, up to 50 epochs with early stopping and best-weights checkpointing
* **Post-processing:** OpenCV contour detection to identify predicted regions and generate bounding-box visualizations

## Dataset and Experiment

The model was trained and evaluated on the full **Kvasir-SEG dataset (1,000 images)**, using Google Colab with a GPU.

### Dataset Split

* **Training:** 640 images
* **Validation:** 160 images
* **Testing:** 200 images

The dataset itself is **not included in this repository**. Please download it from the official Kvasir-SEG page linked above.

## Results

Results on the held-out test set (200 images, 256×256 input):

| Metric                    |    Result |
| ------------------------- | --------: |
| Dice Coefficient          | **0.779** |
| IoU                       | **0.653** |
| Pixel Accuracy            | **0.944** |
| Test Loss (BCE + Dice)    | **0.390** |

Accuracy is included for completeness. The **Dice coefficient and IoU are the key segmentation metrics**, as they measure the overlap between predicted and ground-truth regions.

## Sample Output

The model generates:

**Input Image → Ground Truth Mask → Predicted Mask → Predicted Region with Bounding Box**

![Sample output](results/sample_output.png)

## Limitations and Future Work

* Trained and evaluated on a single dataset (Kvasir-SEG), with one train/validation/test split.
* Images are resized to 256×256, which may lose fine detail for small polyps.
* The model outputs a mask for every image; it does not classify whether a polyp is present or absent.

Possible improvements: higher input resolution, stronger augmentation, cross-validation, and testing on other polyp datasets.

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
polyp-segmentation-deep-learning/
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

Download Kvasir-SEG and place the images and masks in the following folders:

```text
images/
masks/
```

### 3. Run the training script

```bash
python train.py
```

The script loads the dataset, preprocesses the images and masks, trains the U-Net model, evaluates it on the test set, and generates segmentation visualizations.

## Status

**Trained and evaluated on the full Kvasir-SEG dataset (1,000 images).**

## Note

This project is an academic deep learning project focused on **polyp segmentation and localization of regions of interest** in endoscopy images. It is not intended to provide a medical diagnosis or replace professional clinical assessment.
