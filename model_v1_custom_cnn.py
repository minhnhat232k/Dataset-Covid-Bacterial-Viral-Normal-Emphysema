# -*- coding: utf-8 -*-
"""
model_v1_custom_cnn.py
======================
Phiên bản V1 — CNN tuần tự tự thiết kế (Customize TOPSIS Model, 5 lớp)

Nguồn: Customize_TOPSIS_Model_5_Class_Take2_V100_epoch100.ipynb

Kiến trúc:
    5 khối [Conv2D → BatchNorm → MaxPool] → Flatten → Dense(128) ×2 → Dense(5, softmax)
    - Khối 1 dùng ReLU, các khối 2–5 dùng ELU
    - Ghi chú gốc: kernel 8×8 ở khối 1, 3, 5 đã đổi thành 3×3 (để tính GFLOPs)

Huấn luyện:
    Adam (LR mặc định) · Categorical Cross-Entropy · 100 epoch · batch 32
    Không augmentation, không callback dừng sớm.
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ─────────────────────────────────────────────────────────────────────────────
# 1. Cấu hình
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT   = "/content/gdrive/MyDrive/Dataset_5_Layer_Full_Crop_Balance"
SAVE_PATH   = ("/content/gdrive/MyDrive/Models/5Layers/"
               "model_Customize_TOPSIS_Model_5_Class_Take4_V100_Epoch_100_elu.h5")

CLASS_NAMES = ["Covid-19", "Emphysema", "Normal",
               "Pneumonia-Bacterial", "Pneumonia-Viral"]
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE    = 224
CHANNELS    = 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, CHANNELS)
COLOR_MODE  = "rgb"
BATCH_SIZE  = 32
EPOCHS      = 100
SEED        = 123


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dữ liệu
# ─────────────────────────────────────────────────────────────────────────────
def load_split(split):
    return keras.utils.image_dataset_from_directory(
        os.path.join(DATA_ROOT, split),
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode=COLOR_MODE,
        batch_size=BATCH_SIZE,
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=True,
        seed=SEED,
        interpolation="bilinear",
        follow_links=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Kiến trúc
# ─────────────────────────────────────────────────────────────────────────────
def build_model_v1(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    return keras.models.Sequential([
        layers.Input(shape=input_shape),

        # Khối 1  (gốc: kernel 8×8)
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(axis=-1),
        layers.MaxPooling2D(pool_size=(3, 3)),

        # Khối 2
        layers.Conv2D(64, (3, 3), activation="elu", padding="same"),
        layers.BatchNormalization(axis=-1),
        layers.MaxPooling2D(pool_size=(3, 3)),

        # Khối 3  (gốc: kernel 8×8)
        layers.Conv2D(256, (3, 3), activation="elu", padding="same"),
        layers.BatchNormalization(axis=-1),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Khối 4
        layers.Conv2D(32, (4, 4), activation="elu", padding="same"),
        layers.BatchNormalization(axis=-1),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Khối 5  (gốc: kernel 8×8)
        layers.Conv2D(128, (3, 3), activation="elu", padding="same"),
        layers.BatchNormalization(axis=-1),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Classifier
        layers.Flatten(),
        layers.Dense(128, activation="elu"),
        layers.Dense(128, activation="elu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="CustomCNN_v1")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Huấn luyện
# ─────────────────────────────────────────────────────────────────────────────
def main():
    from google.colab import drive
    drive.mount("/content/gdrive")

    train_set = load_split("train")
    val_set   = load_split("val")

    model = build_model_v1()
    model.summary()

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Bản gốc dùng thêm callbacks=[PlotLossesKeras()] (pip install livelossplot)
    history = model.fit(train_set,
                        epochs=EPOCHS,
                        validation_data=val_set)

    model.save(SAVE_PATH)
    print(f"✓ Đã lưu model: {SAVE_PATH}")
    return model, history


if __name__ == "__main__":
    main()
