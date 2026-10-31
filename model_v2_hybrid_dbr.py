# -*- coding: utf-8 -*-
"""
model_v2_hybrid_dbr.py
======================
Phiên bản V2 — Hybrid Depth-Bottleneck Residual Network (Improved X-Ray Classifier)

Nguồn: New_Hybrid_Model_Take2_Improve.ipynb

Thay đổi so với V1:
    - Bỏ CNN tuần tự, chuyển sang mạng residual 4 stage (độ sâu kiểu ResNet-34: [2, 3, 3, 2])
    - Stem kiểu ResNet: Conv 7×7 stride 2 → MaxPool 3×3 stride 2
    - Khối Depth-Bottleneck Residual: 1×1 bottleneck → 3×3 depthwise → ChannelShuffle → 1×1 expand
    - SE attention ở cuối mỗi stage
    - Head: GlobalAveragePooling → Dropout(0.4) → Dense(5) thay cho Flatten + 2 Dense
    - Activation: ReLU

Đặc điểm của V2 (được sửa ở V3):
    - Stride đặt ở Conv 1×1 bottleneck (không phải depthwise)
    - Conv 1×1 không grouped → ChannelShuffle chưa có tác dụng thực sự

Huấn luyện:
    Adam (LR mặc định) · Categorical Cross-Entropy · 100 epoch · batch 32
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ─────────────────────────────────────────────────────────────────────────────
# 1. Cấu hình
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT   = "/content/gdrive/MyDrive/Dataset_5_Layer_Full_Crop_Balance"
SAVE_PATH   = "/content/gdrive/MyDrive/Models/5Layers/new_model_colab_test2_improved.h5"

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

# Kiến trúc
BLOCKS_CONFIG    = [2, 3, 3, 2]   # số khối residual mỗi stage
INITIAL_FILTERS  = 48
BOTTLENECK_RATIO = 4
SHUFFLE_GROUPS   = 2
SE_REDUCTION     = 4
DROPOUT_RATE     = 0.4


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
# 3. Custom layers & blocks
# ─────────────────────────────────────────────────────────────────────────────
class ChannelShuffle(layers.Layer):
    """Xáo trộn kênh (ShuffleNet)."""

    def __init__(self, groups, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups

    def call(self, inputs):
        s = tf.shape(inputs)
        b, h, w, c = s[0], s[1], s[2], s[3]
        cpg = c // self.groups
        x = tf.reshape(inputs, [b, h, w, self.groups, cpg])
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
        return tf.reshape(x, [b, h, w, c])

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"groups": self.groups})
        return config


def se_block(input_tensor, reduction_ratio=4, name_prefix=""):
    """Squeeze-and-Excitation block."""
    c = keras.backend.int_shape(input_tensor)[-1]
    s = layers.GlobalAveragePooling2D(name=f"{name_prefix}_se_avg_pool")(input_tensor)
    s = layers.Dense(c // reduction_ratio, activation="relu",
                     name=f"{name_prefix}_se_dense1")(s)
    s = layers.Dense(c, activation="sigmoid", name=f"{name_prefix}_se_dense2")(s)
    s = layers.Reshape((1, 1, c), name=f"{name_prefix}_se_reshape")(s)
    return layers.multiply([input_tensor, s], name=f"{name_prefix}_se_multiply")


def depth_bottleneck_residual_block(input_tensor, output_filters, bottleneck_filters,
                                    groups=2, project_shortcut=False, stride=1,
                                    block_prefix=""):
    """
    Depth-Bottleneck Residual Block (không dropout bên trong).

        x ─┬─ Conv1×1 (stride) → BN → ReLU
           │  DWConv3×3 → BN → ReLU
           │  ChannelShuffle
           │  Conv1×1 expand → BN
           └─ shortcut (Conv1×1 + BN nếu project) ──► add → ReLU
    """
    shortcut = input_tensor

    # Main path
    x = layers.Conv2D(bottleneck_filters, (1, 1), strides=stride, padding="valid",
                      use_bias=False, name=f"{block_prefix}_conv1x1_bottleneck")(input_tensor)
    x = layers.BatchNormalization(name=f"{block_prefix}_bn_bottleneck")(x)
    x = layers.ReLU(name=f"{block_prefix}_relu_bottleneck")(x)

    x = layers.DepthwiseConv2D((3, 3), strides=1, padding="same", use_bias=False,
                               name=f"{block_prefix}_dwconv3x3")(x)
    x = layers.BatchNormalization(name=f"{block_prefix}_bn_dwconv")(x)
    x = layers.ReLU(name=f"{block_prefix}_relu_dwconv")(x)

    if groups > 1:
        x = ChannelShuffle(groups=groups, name=f"{block_prefix}_channel_shuffle")(x)

    x = layers.Conv2D(output_filters, (1, 1), strides=1, padding="valid",
                      use_bias=False, name=f"{block_prefix}_conv1x1_expand")(x)
    x = layers.BatchNormalization(name=f"{block_prefix}_bn_expand")(x)

    # Shortcut
    if project_shortcut:
        shortcut = layers.Conv2D(output_filters, (1, 1), strides=stride, padding="valid",
                                 use_bias=False, name=f"{block_prefix}_shortcut_conv")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{block_prefix}_shortcut_bn")(shortcut)

    x = layers.add([x, shortcut], name=f"{block_prefix}_add")
    return layers.ReLU(name=f"{block_prefix}_final_relu")(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Kiến trúc
# ─────────────────────────────────────────────────────────────────────────────
def build_model_v2(input_shape=INPUT_SHAPE,
                   num_classes=NUM_CLASSES,
                   num_blocks_per_stage=BLOCKS_CONFIG,
                   initial_filters=INITIAL_FILTERS,
                   bottleneck_ratio=BOTTLENECK_RATIO,
                   shuffle_groups=SHUFFLE_GROUPS,
                   se_reduction=SE_REDUCTION,
                   final_dropout_rate=DROPOUT_RATE):
    """
    Stem (224→56) → Stage1 96f → Stage2 192f (/2) → Stage3 384f (/2) → Stage4 768f (/2)
    → SE cuối mỗi stage → GAP → Dropout → Dense(softmax)
    """
    img_input = layers.Input(shape=input_shape, name="input_image")

    # Stem
    x = layers.Conv2D(initial_filters, (7, 7), strides=(2, 2), padding="same",
                      use_bias=False, name="conv1_stem")(img_input)
    x = layers.BatchNormalization(name="bn1_stem")(x)
    x = layers.ReLU(name="relu1_stem")(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1_stem")(x)

    # Stages
    for i, num_blocks in enumerate(num_blocks_per_stage):
        stage_idx = i + 1
        output_filters = initial_filters * (2 ** stage_idx)
        bottleneck_filters = output_filters // bottleneck_ratio
        downsample = stage_idx > 1
        stride_first = 2 if downsample else 1
        project_first = downsample or (i == 0)

        x = depth_bottleneck_residual_block(
            x, output_filters, bottleneck_filters,
            groups=shuffle_groups, project_shortcut=project_first,
            stride=stride_first, block_prefix=f"s{stage_idx}_b1")

        for block_idx in range(2, num_blocks + 1):
            x = depth_bottleneck_residual_block(
                x, output_filters, bottleneck_filters,
                groups=shuffle_groups, project_shortcut=False,
                stride=1, block_prefix=f"s{stage_idx}_b{block_idx}")

        x = se_block(x, reduction_ratio=se_reduction, name_prefix=f"s{stage_idx}")

    # Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    if final_dropout_rate > 0:
        x = layers.Dropout(final_dropout_rate, name="final_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    return keras.Model(img_input, outputs, name="Improved_XRay_Classifier_v2")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Huấn luyện
# ─────────────────────────────────────────────────────────────────────────────
def main():
    from google.colab import drive
    drive.mount("/content/gdrive")

    train_set = load_split("train")
    val_set   = load_split("val")

    model = build_model_v2()
    model.summary(line_length=120)

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Bản gốc dùng thêm callbacks=[PlotLossesKeras()] (pip install livelossplot)
    history = model.fit(train_set,
                        epochs=EPOCHS,
                        validation_data=val_set)

    model.save(SAVE_PATH)
    print(f"✓ Đã lưu model: {SAVE_PATH}")
    # Khi load lại: keras.models.load_model(path, custom_objects={"ChannelShuffle": ChannelShuffle})
    return model, history


if __name__ == "__main__":
    main()
