import os
import glob
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

import segmentation_models as sm


# ============================================================
# 1. Reproducibility
# ============================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["SM_FRAMEWORK"] = "tf.keras"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. Project Configuration
# ============================================================

IMAGE_SIZE = (512, 512)
MAX_IMAGES = 200

BACKBONE = "resnet34"
N_CLASSES = 1
ACTIVATION = "sigmoid"

EPOCHS = 20
BATCH_SIZE = 4

IMAGE_PATH = "images/*.jpg"
MASK_PATH = "masks/*.jpg"


# ============================================================
# 3. Segmentation Models Configuration
# ============================================================

sm.set_framework("tf.keras")

preprocess_input = sm.get_preprocessing(BACKBONE)


# ============================================================
# 4. Dice Coefficient
# ============================================================

def dice_metric(y_true, y_pred):
    """
    Calculates the Dice coefficient for binary segmentation.
    """

    smooth = K.epsilon()

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(y_true * y_pred)

    dice = (
        2.0 * intersection + smooth
    ) / (
        K.sum(y_true) + K.sum(y_pred) + smooth
    )

    return dice


# ============================================================
# 5. Load Dataset
# ============================================================

def load_dataset(image_path, mask_path, max_images=200):
    """
    Loads images and corresponding segmentation masks.

    Images:
        - Converted to grayscale
        - Resized to 512x512

    Masks:
        - Converted to grayscale
        - Resized to 512x512
        - Normalized to [0, 1]
    """

    image_files = sorted(glob.glob(image_path))
    mask_files = sorted(glob.glob(mask_path))

    # Use at most MAX_IMAGES images and masks
    num_images = min(
        len(image_files),
        len(mask_files),
        max_images
    )

    image_files = image_files[:num_images]
    mask_files = mask_files[:num_images]

    images = []
    masks = []

    for image_file, mask_file in zip(image_files, mask_files):

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(image_file)

        if image is None:
            print(f"Warning: Could not read {image_file}")
            continue

        # Convert image to grayscale
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Resize image
        image = cv2.resize(
            image,
            IMAGE_SIZE
        )

        # ----------------------------------------------------
        # Read mask
        # ----------------------------------------------------

        mask = cv2.imread(
            mask_file,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            print(f"Warning: Could not read {mask_file}")
            continue

        # Resize mask
        mask = cv2.resize(
            mask,
            IMAGE_SIZE
        )

        images.append(image)
        masks.append(mask)

    # Convert lists to NumPy arrays
    images = np.array(images)
    masks = np.array(masks)

    # Add channel dimension
    images = images.reshape(
        -1,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        1
    )

    masks = masks.reshape(
        -1,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        1
    )

    # --------------------------------------------------------
    # Convert grayscale images to 3 channels
    # Required for ResNet34
    # --------------------------------------------------------

    images = np.repeat(
        images,
        3,
        axis=-1
    )

    # Normalize masks
    masks = masks.astype(
        "float32"
    ) / 255.0

    # Apply ResNet34 preprocessing
    images = preprocess_input(images)

    return images, masks


# ============================================================
# 6. Load 200-Image Dataset
# ============================================================

X, Y = load_dataset(
    IMAGE_PATH,
    MASK_PATH,
    MAX_IMAGES
)

print("\nDataset Information")
print("-------------------")
print("Total images:", len(X))
print("Image shape:", X.shape)
print("Mask shape:", Y.shape)


# ============================================================
# 7. Train / Validation / Test Split
# ============================================================

# 80% training + validation
# 20% testing

X_temp, X_test, Y_temp, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=SEED
)

# From remaining 80%:
# 80% training
# 20% validation

X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp,
    Y_temp,
    test_size=0.20,
    random_state=SEED
)

print("\nDataset Split")
print("-------------")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))


# ============================================================
# 8. Build U-Net with ResNet34 Encoder
# ============================================================

model = sm.Unet(
    backbone_name=BACKBONE,
    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),
    classes=N_CLASSES,
    activation=ACTIVATION,
    encoder_weights="imagenet"
)


# ============================================================
# 9. Compile Model
# ============================================================

model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        dice_metric
    ]
)


# ============================================================
# 10. Train Model
# ============================================================

print("\nStarting model training...")

history = model.fit(
    X_train,
    Y_train,
    validation_data=(
        X_val,
        Y_val
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)


# ============================================================
# 11. Evaluate Model
# ============================================================

results = model.evaluate(
    X_test,
    Y_test,
    batch_size=BATCH_SIZE
)

test_loss = results[0]
test_accuracy = results[1]
test_dice = results[2]

print("\nTest Results")
print("------------")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test Dice Coefficient:", test_dice)


# ============================================================
# 12. Generate Predictions
# ============================================================

predictions = model.predict(X_test)

# Convert predicted probabilities into binary masks
predictions_binary = (
    predictions > 0.5
).astype(np.float32)


# ============================================================
# 13. Visualize Segmentation Results
# ============================================================

for i in range(min(4, len(X_test))):

    plt.figure(figsize=(16, 4))

    # --------------------------------------------------------
    # Input Image
    # --------------------------------------------------------

    plt.subplot(1, 4, 1)

    input_image = X_test[i][:, :, 0]

    # Normalize image only for visualization
    display_image = cv2.normalize(
        input_image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    plt.imshow(
        display_image,
        cmap="gray"
    )

    plt.title("Input Image")
    plt.axis("off")

    # --------------------------------------------------------
    # Ground Truth Mask
    # --------------------------------------------------------

    plt.subplot(1, 4, 2)

    plt.imshow(
        Y_test[i].squeeze(),
        cmap="gray"
    )

    plt.title("Ground Truth Mask")
    plt.axis("off")

    # --------------------------------------------------------
    # Predicted Mask
    # --------------------------------------------------------

    plt.subplot(1, 4, 3)

    plt.imshow(
        predictions_binary[i].squeeze(),
        cmap="gray"
    )

    plt.title("Predicted Mask")
    plt.axis("off")

    # --------------------------------------------------------
    # Bounding Box
    # --------------------------------------------------------

    plt.subplot(1, 4, 4)

    mask = predictions_binary[i].squeeze()

    mask_uint8 = (
        mask * 255
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Convert grayscale image to RGB
    image_with_boxes = cv2.cvtColor(
        display_image,
        cv2.COLOR_GRAY2RGB
    )

    # Draw bounding boxes
    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        cv2.rectangle(
            image_with_boxes,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    plt.imshow(image_with_boxes)

    plt.title("Predicted Region")
    plt.axis("off")

    plt.tight_layout()
    plt.show()