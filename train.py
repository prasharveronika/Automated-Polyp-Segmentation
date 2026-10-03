"""
Automated Polyp Segmentation using Deep Learning
U-Net with ResNet34 encoder (ImageNet-pretrained) on Kvasir-SEG.

Expected folders next to this script:
    images/   -> Kvasir-SEG images (*.jpg)
    masks/    -> Kvasir-SEG masks  (*.jpg)
"""

import os
import glob
import random

# ============================================================
# 1. Reproducibility
# ============================================================
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["SM_FRAMEWORK"] = "tf.keras"

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

import segmentation_models as sm

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 2. Configuration
# ============================================================
IMAGE_SIZE = (512, 512)     # use (256, 256) if you run out of memory
MAX_IMAGES = 1000           # Kvasir-SEG has 1000 images

BACKBONE = "resnet34"
N_CLASSES = 1
ACTIVATION = "sigmoid"

EPOCHS = 50                 # early stopping will end training sooner
BATCH_SIZE = 4

IMAGE_PATH = "images/*.jpg"
MASK_PATH = "masks/*.jpg"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

sm.set_framework("tf.keras")
preprocess_input = sm.get_preprocessing(BACKBONE)


# ============================================================
# 3. Metrics and loss
# ============================================================
def dice_metric(y_true, y_pred):
    """Dice coefficient for binary segmentation."""
    smooth = K.epsilon()
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (
        K.sum(y_true) + K.sum(y_pred) + smooth
    )


def bce_dice_loss(y_true, y_pred):
    """Binary cross-entropy + (1 - Dice)."""
    bce = K.mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + (1.0 - dice_metric(y_true, y_pred))


# ============================================================
# 4. Load dataset
# ============================================================
def load_dataset(image_path, mask_path, max_images):
    """
    Images: grayscale -> resized -> 3 channels -> ResNet34 preprocessing
    Masks : grayscale -> resized -> scaled to [0, 1] -> binarized
    """
    image_files = sorted(glob.glob(image_path))
    mask_files = sorted(glob.glob(mask_path))

    num_images = min(len(image_files), len(mask_files), max_images)
    if num_images == 0:
        raise FileNotFoundError(
            "No images/masks found. Put Kvasir-SEG files in 'images/' and 'masks/'."
        )

    images, masks = [], []

    for image_file, mask_file in zip(image_files[:num_images], mask_files[:num_images]):
        image = cv2.imread(image_file)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Warning: could not read {image_file} or {mask_file}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, IMAGE_SIZE)
        mask = cv2.resize(mask, IMAGE_SIZE)

        images.append(image)
        masks.append(mask)

    images = np.array(images).reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    masks = np.array(masks).reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

    # 3 channels for the ResNet34 encoder
    images = np.repeat(images, 3, axis=-1)

    # Normalize and binarize masks (JPEG compression leaves in-between values)
    masks = masks.astype("float32") / 255.0
    masks = (masks > 0.5).astype("float32")

    images = preprocess_input(images)

    return images, masks


X, Y = load_dataset(IMAGE_PATH, MASK_PATH, MAX_IMAGES)

print("\nDataset Information")
print("-------------------")
print("Total images:", len(X))
print("Image shape:", X.shape)
print("Mask shape:", Y.shape)

# ============================================================
# 5. Train / validation / test split (64% / 16% / 20%)
# ============================================================
X_temp, X_test, Y_temp, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=SEED
)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp, test_size=0.20, random_state=SEED
)

print("\nDataset Split")
print("-------------")
print("Training:  ", len(X_train))
print("Validation:", len(X_val))
print("Testing:   ", len(X_test))


# ============================================================
# 6. Data augmentation (training set only)
# ============================================================
def augment(image, mask):
    """Apply the same random flip/rotation to image and mask."""
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)

    combined = tf.concat([image, mask], axis=-1)
    combined = tf.image.random_flip_left_right(combined)
    combined = tf.image.random_flip_up_down(combined)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    combined = tf.image.rot90(combined, k)

    return combined[..., :3], combined[..., 3:]


train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    .shuffle(len(X_train), seed=SEED)
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(BATCH_SIZE)

# ============================================================
# 7. Model: U-Net with ResNet34 encoder
# ============================================================
model = sm.Unet(
    backbone_name=BACKBONE,
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    classes=N_CLASSES,
    activation=ACTIVATION,
    encoder_weights="imagenet",
)

model.compile(
    optimizer="rmsprop",
    loss=bce_dice_loss,
    metrics=["accuracy", dice_metric, sm.metrics.iou_score],
)

# ============================================================
# 8. Train
# ============================================================
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.weights.h5",
        monitor="val_dice_metric",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_dice_metric",
        mode="max",
        patience=5,
        restore_best_weights=True,
    ),
]

print("\nStarting model training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Training curves
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Loss (BCE + Dice)")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["dice_metric"], label="train")
plt.plot(history.history["val_dice_metric"], label="validation")
plt.title("Dice coefficient")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150)
plt.close()

# ============================================================
# 9. Evaluate on the test set
# ============================================================
results = model.evaluate(
    X_test, Y_test, batch_size=BATCH_SIZE, return_dict=True
)

print("\nTest Results")
print("------------")
for name, value in results.items():
    print(f"{name}: {value:.4f}")

# ============================================================
# 10. Predictions and visualization
# ============================================================
predictions = model.predict(X_test, batch_size=BATCH_SIZE)
predictions_binary = (predictions > 0.5).astype(np.float32)

for i in range(min(4, len(X_test))):
    plt.figure(figsize=(16, 4))

    # Input image (normalized for display only)
    display_image = cv2.normalize(
        X_test[i][:, :, 0], None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    plt.subplot(1, 4, 1)
    plt.imshow(display_image, cmap="gray")
    plt.title("Input Image")
    plt.axis("off")

    # Ground truth
    plt.subplot(1, 4, 2)
    plt.imshow(Y_test[i].squeeze(), cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    # Prediction
    mask = predictions_binary[i].squeeze()
    plt.subplot(1, 4, 3)
    plt.imshow(mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    # Bounding boxes from contours
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    image_with_boxes = cv2.cvtColor(display_image, cv2.COLOR_GRAY2RGB)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 2)

    plt.subplot(1, 4, 4)
    plt.imshow(image_with_boxes)
    plt.title("Predicted Region")
    plt.axis("off")

    plt.tight_layout()

    filename = "sample_output.png" if i == 0 else f"sample_output_{i + 1}.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.show()

print(f"\nSaved sample outputs and training curves to '{RESULTS_DIR}/'")
