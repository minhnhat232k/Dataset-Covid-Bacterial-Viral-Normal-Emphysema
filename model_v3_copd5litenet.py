# -*- coding: utf-8 -*-
"""
model_v3_copd5litenet.py
========================
Phiên bản V3 — COPD-5LiteNet v2

Nguồn: Model_COPDLiteNet_V2_Improve.ipynb (CELL 0 → CELL 7)

Thay đổi so với V2:
  Kiến trúc
    - Conv 1×1 bottleneck/expand chuyển sang grouped conv → ChannelShuffle có tác dụng thật
    - Thứ tự khối: PW1 (grouped) → ChannelShuffle → DW3×3 → PW2 (grouped)
    - Stride chuyển từ Conv 1×1 sang Depthwise 3×3
    - Activation: Hard-Swish (tuỳ chọn ReLU)
    - Thêm LayerScale (γ khởi tạo 1e-5) trên nhánh residual
    - Thêm Stochastic Depth, tỉ lệ tăng tuyến tính 0 → 0.10 theo độ sâu
    - Attention tuỳ chọn: SE | ECA | None
    - Head ép kiểu float32 trước softmax (cho mixed precision)
  Dữ liệu
    - Tiền xử lý CLAHE (kênh L, không gian LAB), chuẩn hoá [0, 1]
    - Augmentation: lật ngang, brightness/contrast ±10%, zoom ±10%
  Huấn luyện
    - Focal loss (γ = 2) với α theo nghịch đảo tần suất lớp
    - AdamW (weight decay 1e-4) + warm-up 5 epoch → cosine decay (1e-3 → 1e-6)
    - Mixed precision FP16
    - Callbacks: ModelCheckpoint, EarlyStopping (patience 15), CSVLogger, TensorBoard

Thư viện cần: tensorflow, opencv-python-headless, numpy
"""

import os
import math
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision

# ─────────────────────────────────────────────────────────────────────────────
# 1. Cấu hình
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42

DATA_ROOT  = "/content/drive/MyDrive/Dataset_5_Layer_Full_Crop_Balance"
OUTPUT_DIR = "/content/drive/MyDrive/COPD_Outputs"

CLASS_NAMES = ["Covid-19", "Emphysema", "Normal",
               "Pneumonia-Bacterial", "Pneumonia-Viral"]
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE    = 224
CHANNELS    = 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, CHANNELS)

# Huấn luyện
BATCH_SIZE   = 32          # giảm xuống 16 nếu OOM
EPOCHS       = 100
WARMUP_EP    = 5
LR_INIT      = 1e-3
LR_MIN       = 1e-6
WEIGHT_DECAY = 1e-4
FOCAL_GAMMA  = 2.0

# Kiến trúc
BLOCKS_CONFIG    = [2, 3, 3, 2]
INITIAL_FILTERS  = 48
BOTTLENECK_RATIO = 4
SHUFFLE_GROUPS   = 2
ATTENTION        = "se"      # "se" | "eca" | None
SE_REDUCTION     = 4
ACTIVATION       = "hswish"  # "hswish" | "relu"
MAX_DROP_PATH    = 0.10
DROPOUT_RATE     = 0.4

# Augmentation
AUG_ZOOM       = 0.10
AUG_BRIGHTNESS = 0.10
AUG_CONTRAST   = 0.10


def setup_runtime():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    if tf.config.list_physical_devices("GPU"):
        mixed_precision.set_global_policy("mixed_float16")
        print("✓ Mixed precision FP16 đã bật")
    else:
        print("⚠ Không thấy GPU — chạy FP32 trên CPU")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dữ liệu: CLAHE + augmentation + tf.data
# ─────────────────────────────────────────────────────────────────────────────
def apply_clahe(img_path: str) -> np.ndarray:
    """Đọc ảnh → resize → CLAHE trên kênh L (LAB) → float32 [0, 1]."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img.astype(np.float32) / 255.0


def build_file_lists(root: str):
    """Trả về (paths, labels, counts) cho một split."""
    paths, labels = [], []
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for idx, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(root, cls)
        if not os.path.isdir(folder):
            print(f"  ⚠ Không tìm thấy thư mục lớp: {folder}")
            continue
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        paths.extend(files)
        labels.extend([idx] * len(files))
        counts[idx] = len(files)
    return paths, labels, counts


def _random_zoom(img, zoom_range):
    """Crop ngẫu nhiên rồi resize lại kích thước gốc."""
    crop_frac = tf.random.uniform([], 0.0, zoom_range)
    crop_size = tf.cast(tf.cast(IMG_SIZE, tf.float32) * (1.0 - crop_frac), tf.int32)
    img = tf.image.random_crop(img, size=[crop_size, crop_size, CHANNELS])
    return tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")


def augment(img, label):
    """Augmentation bằng tf.image thuần (chỉ cho tập train)."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=AUG_BRIGHTNESS)
    img = tf.image.random_contrast(img, 1.0 - AUG_CONTRAST, 1.0 + AUG_CONTRAST)
    img = _random_zoom(img, AUG_ZOOM)
    return tf.clip_by_value(img, 0.0, 1.0), label


def load_and_preprocess(path, label):
    img = tf.numpy_function(lambda p: apply_clahe(p.decode("utf-8")), [path], tf.float32)
    img.set_shape([IMG_SIZE, IMG_SIZE, CHANNELS])
    return img, tf.one_hot(tf.cast(label, tf.int32), NUM_CLASSES)


def make_dataset(paths, labels, training=False, shuffle=False):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Custom layers
# ─────────────────────────────────────────────────────────────────────────────
class HardSwish(layers.Layer):
    """f(x) = x · relu6(x + 3) / 6   (MobileNetV3)."""

    def call(self, x):
        return x * tf.nn.relu6(x + 3.0) / 6.0


class LayerScale(layers.Layer):
    """Scale học được theo kênh trên nhánh residual (CaiT/ConvNeXt), γ ≈ 0 lúc đầu."""

    def __init__(self, init_value=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        # shape truyền theo vị trí để tương thích Keras 3
        self.gamma = self.add_weight(
            (input_shape[-1],), name="gamma",
            initializer=keras.initializers.Constant(self.init_value),
            trainable=True)
        super().build(input_shape)

    def call(self, x):
        return tf.cast(self.gamma, x.dtype) * x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"init_value": self.init_value})
        return cfg


class StochasticDepth(layers.Layer):
    """DropPath: bỏ ngẫu nhiên cả nhánh residual khi training."""

    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if not training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = [tf.shape(x)[0]] + [1] * (len(x.shape) - 1)
        mask = tf.floor(keep + tf.random.uniform(shape, dtype=x.dtype))
        return (x / keep) * mask

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"drop_prob": self.drop_prob})
        return cfg


class ChannelShuffle(layers.Layer):
    """Hoán vị kênh sau grouped conv (ShuffleNetV2)."""

    def __init__(self, groups, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups

    def call(self, x):
        s = tf.shape(x)
        b, h, w, c = s[0], s[1], s[2], s[3]
        x = tf.reshape(x, [b, h, w, self.groups, c // self.groups])
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        return tf.reshape(x, [b, h, w, c])

    def compute_output_shape(self, s):
        return s

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"groups": self.groups})
        return cfg


CUSTOM_OBJECTS = {"HardSwish": HardSwish, "LayerScale": LayerScale,
                  "StochasticDepth": StochasticDepth, "ChannelShuffle": ChannelShuffle}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Attention blocks
# ─────────────────────────────────────────────────────────────────────────────
def se_block(x, r=4, prefix=""):
    c = keras.backend.int_shape(x)[-1]
    s = layers.GlobalAveragePooling2D(name=f"{prefix}_se_gap")(x)
    s = layers.Dense(max(c // r, 1), activation="relu", name=f"{prefix}_se_fc1")(s)
    s = layers.Dense(c, activation="sigmoid", name=f"{prefix}_se_fc2")(s)
    s = layers.Reshape((1, 1, c), name=f"{prefix}_se_reshape")(s)
    return layers.multiply([x, s], name=f"{prefix}_se_out")


def eca_block(x, gamma=2, b=1, prefix=""):
    """Efficient Channel Attention (Wang et al., CVPR 2020)."""
    c = keras.backend.int_shape(x)[-1]
    t = int(abs((math.log2(c) + b) / gamma))
    k = t if t % 2 else t + 1
    y = layers.GlobalAveragePooling2D(name=f"{prefix}_eca_gap")(x)
    y = layers.Reshape((c, 1), name=f"{prefix}_eca_r1")(y)
    y = layers.Conv1D(1, k, padding="same", use_bias=False, name=f"{prefix}_eca_conv")(y)
    y = layers.Activation("sigmoid", name=f"{prefix}_eca_sig")(y)
    y = layers.Reshape((1, 1, c), name=f"{prefix}_eca_r2")(y)
    return layers.multiply([x, y], name=f"{prefix}_eca_out")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Residual block & kiến trúc
# ─────────────────────────────────────────────────────────────────────────────
def _safe_groups(in_c, out_c, g):
    """Chỉ dùng grouped conv khi in_c và out_c đều chia hết cho g."""
    return g if (g > 1 and in_c % g == 0 and out_c % g == 0) else 1


def residual_block(x_in, out_f, bot_f, groups=2, project=False, stride=1,
                   drop_path=0.0, act="hswish", prefix=""):
    """
        x ─┬─ PW1 1×1 grouped → BN → act
           │  ChannelShuffle
           │  DW 3×3 (stride) → BN → act
           │  PW2 1×1 grouped → BN
           │  LayerScale → StochasticDepth
           └─ shortcut (Conv1×1 + BN nếu project) ──► add → act
    """
    def activate(t, name):
        return HardSwish(name=name)(t) if act == "hswish" else layers.ReLU(name=name)(t)

    in_c = keras.backend.int_shape(x_in)[-1]
    sc = x_in

    g1 = _safe_groups(in_c, bot_f, groups)
    x = layers.Conv2D(bot_f, 1, padding="same", groups=g1, use_bias=False,
                      name=f"{prefix}_pw1")(x_in)
    x = layers.BatchNormalization(name=f"{prefix}_bn1")(x)
    x = activate(x, f"{prefix}_act1")

    if g1 > 1:
        x = ChannelShuffle(g1, name=f"{prefix}_shuffle")(x)

    x = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False,
                               name=f"{prefix}_dw")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn2")(x)
    x = activate(x, f"{prefix}_act2")

    g2 = _safe_groups(bot_f, out_f, groups)
    x = layers.Conv2D(out_f, 1, padding="same", groups=g2, use_bias=False,
                      name=f"{prefix}_pw2")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn3")(x)

    x = LayerScale(name=f"{prefix}_ls")(x)
    if drop_path > 0.0 and not project:
        x = StochasticDepth(drop_path, name=f"{prefix}_dp")(x)

    if project:
        sc = layers.Conv2D(out_f, 1, strides=stride, padding="same", use_bias=False,
                           name=f"{prefix}_sc_conv")(sc)
        sc = layers.BatchNormalization(name=f"{prefix}_sc_bn")(sc)

    x = layers.add([x, sc], name=f"{prefix}_add")
    return activate(x, f"{prefix}_act_out")


def build_model_v3(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                   blocks=BLOCKS_CONFIG, init_f=INITIAL_FILTERS,
                   bot_r=BOTTLENECK_RATIO, groups=SHUFFLE_GROUPS,
                   attn=ATTENTION, se_r=SE_REDUCTION, act=ACTIVATION,
                   max_dp=MAX_DROP_PATH, dropout=DROPOUT_RATE):
    """
    Stem (7×7/2 → MaxPool/2)        224 → 56
    Stage 1: 2 block,  96 filters   stride 1
    Stage 2: 3 block, 192 filters   56 → 28
    Stage 3: 3 block, 384 filters   28 → 14
    Stage 4: 2 block, 768 filters   14 → 7
    Attention cuối mỗi stage → GAP → Dropout → Dense(softmax, float32)
    """
    img = layers.Input(input_shape, name="input_image")

    x = layers.Conv2D(init_f, 7, strides=2, padding="same", use_bias=False,
                      name="stem_conv")(img)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = HardSwish(name="stem_act")(x) if act == "hswish" else layers.ReLU(name="stem_act")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="stem_pool")(x)

    # Stochastic depth tăng tuyến tính 0 → max_dp
    total_b = sum(blocks)
    dp_list = [max_dp * i / max(total_b - 1, 1) for i in range(total_b)]
    bi = 0

    for si, nb in enumerate(blocks):
        stage = si + 1
        out_f = init_f * (2 ** stage)
        bot_f = out_f // bot_r
        first_stage = si == 0
        stride0 = 1 if first_stage else 2

        x = residual_block(x, out_f, bot_f, groups=groups, project=True,
                           stride=stride0, drop_path=dp_list[bi], act=act,
                           prefix=f"s{stage}_b1")
        bi += 1
        for bj in range(2, nb + 1):
            x = residual_block(x, out_f, bot_f, groups=groups, project=False,
                               stride=1, drop_path=dp_list[bi], act=act,
                               prefix=f"s{stage}_b{bj}")
            bi += 1

        if attn == "se":
            x = se_block(x, r=se_r, prefix=f"s{stage}")
        elif attn == "eca":
            x = eca_block(x, prefix=f"s{stage}")

    x = layers.GlobalAveragePooling2D(name="head_gap")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    x = layers.Activation("linear", dtype="float32", name="cast_fp32")(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32",
                       name="classifier")(x)
    return keras.Model(img, out, name="COPD_5LiteNet_v2")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Loss & LR schedule
# ─────────────────────────────────────────────────────────────────────────────
def make_focal_loss(alpha, gamma=2.0):
    """L_FL = -Σ α_c · (1 - p_c)^γ · y_c · log(p_c)"""
    alpha_t = tf.constant(alpha, dtype=tf.float32)

    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        eps = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha_t * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(weight * ce, axis=-1)

    focal_loss.__name__ = "focal_loss"
    return focal_loss


class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Warm-up tuyến tính theo step, sau đó cosine decay về lr_min."""

    def __init__(self, lr_init, lr_min, warmup_steps, total_steps):
        super().__init__()
        self.lr_init = lr_init
        self.lr_min = lr_min
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        wu = tf.cast(self.warmup_steps, tf.float32)
        tot = tf.cast(self.total_steps, tf.float32)
        lr_max = tf.cast(self.lr_init, tf.float32)
        lr_min = tf.cast(self.lr_min, tf.float32)
        warmup_lr = lr_max * (step / tf.maximum(wu, 1.0))
        cosine_lr = lr_min + 0.5 * (lr_max - lr_min) * (
            1.0 + tf.cos(math.pi * (step - wu) / tf.maximum(tot - wu, 1.0)))
        return tf.where(step < wu, warmup_lr, cosine_lr)

    def get_config(self):
        return dict(lr_init=self.lr_init, lr_min=self.lr_min,
                    warmup_steps=self.warmup_steps, total_steps=self.total_steps)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Huấn luyện
# ─────────────────────────────────────────────────────────────────────────────
def main():
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)

    setup_runtime()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Dữ liệu
    train_paths, train_labels, train_counts = build_file_lists(os.path.join(DATA_ROOT, "train"))
    val_paths, val_labels, _ = build_file_lists(os.path.join(DATA_ROOT, "val"))
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)}")
    print(f"Ảnh/lớp (train): {dict(zip(CLASS_NAMES, train_counts.tolist()))}")

    train_ds = make_dataset(train_paths, train_labels, training=True, shuffle=True)
    val_ds = make_dataset(val_paths, val_labels, training=False, shuffle=False)

    # α cho focal loss: nghịch đảo tần suất, chuẩn hoá tổng = 1
    inv_freq = 1.0 / (train_counts.astype(np.float32) + 1e-8)
    class_alpha = (inv_freq / inv_freq.sum()).astype(np.float32)

    # Model
    model = build_model_v3()
    model.summary(line_length=120)
    print(f"Params: {model.count_params():,}")

    steps_per_epoch = len(train_ds)
    lr_schedule = WarmupCosineDecay(LR_INIT, LR_MIN,
                                    warmup_steps=steps_per_epoch * WARMUP_EP,
                                    total_steps=steps_per_epoch * EPOCHS)

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule,
                                         weight_decay=WEIGHT_DECAY),
        loss=make_focal_loss(class_alpha, FOCAL_GAMMA),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy"),
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")])

    ckpt_path = os.path.join(OUTPUT_DIR, "best_model.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy",
                                        save_best_only=True, mode="max", verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                      restore_best_weights=True, mode="max", verbose=1),
        keras.callbacks.CSVLogger(os.path.join(OUTPUT_DIR, "training_log.csv")),
        keras.callbacks.TensorBoard(log_dir=os.path.join(OUTPUT_DIR, "tb_logs"),
                                    histogram_freq=0, write_graph=False),
        # ReduceLROnPlateau đã bỏ: không dùng được cùng LearningRateSchedule
    ]

    t0 = time.time()
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=EPOCHS,
                        callbacks=callbacks,
                        verbose=1)
    h, rem = divmod(int(time.time() - t0), 3600)
    m, s = divmod(rem, 60)
    print(f"✓ Huấn luyện xong sau {h}h {m}m {s}s")
    print(f"✓ Best val_accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"✓ Checkpoint: {ckpt_path}")
    return model, history


if __name__ == "__main__":
    main()
