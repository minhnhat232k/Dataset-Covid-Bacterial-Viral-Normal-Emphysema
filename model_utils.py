# -*- coding: utf-8 -*-
"""
model_utils.py
==============
Các tiện ích dùng chung sau huấn luyện cho V1 / V2 / V3:

    1. load_trained_model()      — nạp model .h5 / .keras (kèm custom layers V2, V3)
    2. get_flops()               — đếm FLOPs / GFLOPs
    3. measure_inference_speed() — đo latency (ms/ảnh) và FPS
       load_nasnet_from_weights() — dựng lại NASNetMobile baseline rồi nạp trọng số
    4. grad_cam_tf(), overlay_cam(), plot_gradcam_grid() — Grad-CAM
    5. export_tflite()           — xuất model sang .tflite

Nguồn: gom từ Customize_TOPSIS_Model_5_Class_Take2_V100_epoch100.ipynb
       (FLOPs, tốc độ suy luận, NASNetMobile, TFLite) và
       Model_COPDLiteNet_V2_Improve.ipynb (CELL 9g mở rộng — Grad-CAM).

Đặt file này cùng thư mục với model_v1/v2/v3 để import custom layers.
"""

import os
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

CLASS_NAMES = ["Covid-19", "Emphysema", "Normal",
               "Pneumonia-Bacterial", "Pneumonia-Viral"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224


# ─────────────────────────────────────────────────────────────────────────────
# 1. Nạp model đã huấn luyện
# ─────────────────────────────────────────────────────────────────────────────
def load_trained_model(path):
    """
    Nạp model V1/V2/V3. compile=False → không cần optimizer, LR schedule hay
    focal loss; trọng số vẫn đầy đủ để suy luận, đo FLOPs, Grad-CAM, TFLite.
    """
    from model_v3_copd5litenet import CUSTOM_OBJECTS  # gồm cả ChannelShuffle của V2
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy model: {path}")
    model = keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)
    print(f"✓ Đã nạp {model.name} | params: {model.count_params():,} "
          f"| fp32 ≈ {model.count_params() * 4 / 1024 ** 2:.2f} MB")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. FLOPs
# ─────────────────────────────────────────────────────────────────────────────
def get_flops(model):
    """Đếm tổng float ops cho 1 ảnh (batch = 1) bằng TF profiler trên frozen graph."""
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2)

    concrete = tf.function(lambda x: model(x)).get_concrete_function(
        tf.TensorSpec([1] + list(model.input_shape[1:]), model.inputs[0].dtype))
    frozen = convert_variables_to_constants_v2(concrete)

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(frozen.graph.as_graph_def(), name="")
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        opts["output"] = "none"
        flops = tf.compat.v1.profiler.profile(
            graph=graph, run_meta=tf.compat.v1.RunMetadata(), cmd="op", options=opts)
    return flops.total_float_ops


def print_flops(model):
    flops = get_flops(model)
    print("-" * 30)
    print(f"Tổng số FLOPs : {flops:,}")
    print(f"GFLOPs        : {flops / 1e9:.4f}")
    print("-" * 30)
    return flops


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tốc độ suy luận
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_rescale(img_path, size=IMG_SIZE):
    """Đọc ảnh → resize → chia 255 (như đoạn đo tốc độ gốc của V1)."""
    img = keras.utils.load_img(img_path, target_size=(size, size))
    return np.expand_dims(keras.utils.img_to_array(img), 0) / 255.0


def preprocess_clahe(img_path, size=IMG_SIZE):
    """Tiền xử lý CLAHE giống pipeline huấn luyện V3."""
    from model_v3_copd5litenet import apply_clahe
    return np.expand_dims(apply_clahe(img_path), 0)


def preprocess_nasnet(img_path, size=IMG_SIZE):
    from tensorflow.keras.applications.nasnet import preprocess_input
    img = keras.utils.load_img(img_path, target_size=(size, size))
    return preprocess_input(np.expand_dims(keras.utils.img_to_array(img), 0))


def measure_inference_speed(model, img_path, preprocess_fn=preprocess_rescale,
                            warmup=10, iterations=100, label=None):
    """
    Đo latency trung bình (ms/ảnh), độ lệch chuẩn và FPS cho batch = 1.
    Chạy warm-up trước để GPU/CPU khởi tạo kernel.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")
    x = preprocess_fn(img_path)

    for _ in range(warmup):
        model.predict(x, verbose=0)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model.predict(x, verbose=0)
        times.append(time.perf_counter() - t0)

    avg_s = float(np.mean(times))
    result = {"latency_ms": avg_s * 1000,
              "std_ms": float(np.std(times)) * 1000,
              "fps": 1.0 / avg_s,
              "device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU"}

    print("\n" + "=" * 50)
    print(f"KẾT QUẢ ĐO TỐC ĐỘ — {label or model.name} (input {x.shape[1]}×{x.shape[2]})")
    print("=" * 50)
    print(f"Thời gian trung bình : {result['latency_ms']:.2f} ms / ảnh")
    print(f"Độ lệch chuẩn        : ±{result['std_ms']:.2f} ms")
    print(f"Tốc độ xử lý         : {result['fps']:.2f} FPS")
    print(f"Thiết bị             : {result['device']}")
    return result


def load_nasnet_from_weights(weights_path, input_size=224, num_classes=NUM_CLASSES):
    """
    Dựng lại khung NASNetMobile baseline (Flatten → Dense128 → Dense5) rồi nạp
    trọng số — dùng khi load_model() lỗi do khác phiên bản Keras.
    input_size phải khớp lúc huấn luyện (bản gốc dựng 224 nhưng đo bằng ảnh 128).
    Trên TF ≥ 2.16 có thể cần: os.environ["TF_USE_LEGACY_KERAS"] = "1" + pip install tf_keras
    """
    from tensorflow.keras.applications.nasnet import NASNetMobile
    base = NASNetMobile(input_shape=(input_size, input_size, 3),
                        include_top=False, weights=None)
    x = keras.layers.Flatten()(base.output)
    x = keras.layers.Dense(128, activation="relu")(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(base.input, out, name="NASNetMobile_baseline")
    model.load_weights(weights_path)
    print("✓ Đã nạp trọng số NASNetMobile")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────
def find_target_layer(model):
    """Layer Conv2D/DepthwiseConv2D cuối cùng có kernel > 1×1."""
    for layer in reversed(model.layers):
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
            ks = layer.get_config().get("kernel_size", (1, 1))
            if isinstance(ks, (list, tuple)) and ks[0] > 1:
                return layer.name
    raise ValueError("Không tìm thấy conv layer phù hợp cho Grad-CAM.")


def grad_cam_tf(model, img_array, class_idx, last_conv_name=None, out_size=IMG_SIZE):
    """Tính Grad-CAM cho 1 ảnh [H, W, 3], trả về float32 [H, W] trong [0, 1]."""
    name = last_conv_name or find_target_layer(model)
    target = model.get_layer(name)
    grad_model = keras.Model(model.inputs, [target.output, model.output])

    img_t = tf.cast(img_array[np.newaxis], tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_t)
        conv_out, preds = grad_model(img_t, training=False)
        conv_out = tf.cast(conv_out, tf.float32)   # an toàn với mixed precision
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads[0], axis=(0, 1))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1).numpy().astype(np.float32)
    cam = np.maximum(cam, 0)
    vmin, vmax = cam.min(), cam.max()
    cam = (cam - vmin) / (vmax - vmin) if vmax - vmin > 1e-8 else np.zeros_like(cam)
    return cv2.resize(cam, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def overlay_cam(img, cam, alpha=0.45):
    """Chồng heatmap JET lên ảnh gốc (ảnh trong [0, 1])."""
    hm = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(alpha * hm + (1 - alpha) * np.clip(img, 0, 1), 0, 1)


def collect_gradcam_samples(model, dataset, n_samples=2):
    """
    Chạy model trên dataset, lấy n_samples ảnh dự đoán đúng và n_samples ảnh
    dự đoán sai cho mỗi lớp. Trả về (correct, wrong) dạng dict theo chỉ số lớp.
    """
    correct = {c: [] for c in range(NUM_CLASSES)}
    wrong = {c: [] for c in range(NUM_CLASSES)}
    for xb, yb in dataset:
        preds = model(xb, training=False).numpy().argmax(1)
        trues = yb.numpy().argmax(1)
        for img, t, p in zip(xb.numpy(), trues, preds):
            t, p = int(t), int(p)
            if t == p and len(correct[t]) < n_samples:
                correct[t].append((img, t))
            elif t != p and len(wrong[t]) < n_samples:
                wrong[t].append((img, t, p))
        if all(len(correct[c]) >= n_samples and len(wrong[c]) >= n_samples
               for c in range(NUM_CLASSES)):
            break

    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:22s}: {len(correct[c])} đúng, {len(wrong[c])} sai")
    return correct, wrong


def plot_gradcam_grid(model, dataset, save_path, n_samples=2, target_layer=None):
    """
    Lưới Grad-CAM: mỗi hàng 1 lớp; mỗi cặp cột [ảnh | overlay CAM].
    n_samples ảnh đúng (trái, CAM theo lớp thật) + n_samples ảnh sai (phải, CAM theo lớp dự đoán).
    Lưu ý: ảnh trong dataset phải ở thang [0, 1] (như pipeline V3).
    """
    import matplotlib.pyplot as plt

    target_layer = target_layer or find_target_layer(model)
    print(f"Grad-CAM target layer: {target_layer}")
    correct, wrong = collect_gradcam_samples(model, dataset, n_samples)

    n_cols = n_samples * 4
    fig, axes = plt.subplots(NUM_CLASSES, n_cols,
                             figsize=(n_cols * 3.2, NUM_CLASSES * 3.5), squeeze=False)
    fig.suptitle("Grad-CAM Visualisation\n"
                 "Trái: dự đoán đúng  |  Phải: dự đoán sai",
                 fontsize=13, fontweight="bold", y=1.01)

    def draw(ax_img, ax_cam, img, cls, title="", color="black"):
        cam = grad_cam_tf(model, img, cls, target_layer)
        ax_img.imshow(np.clip(img, 0, 1))
        ax_img.set_title(title, fontsize=7, color=color)
        ax_cam.imshow(overlay_cam(img, cam))

    def empty(*axs):
        for ax in axs:
            ax.text(0.5, 0.5, "—", ha="center", va="center", transform=ax.transAxes)

    col_labels = [l for k in range(n_samples) for l in (f"Correct #{k + 1}", "CAM")] + \
                 [l for k in range(n_samples) for l in (f"Wrong #{k + 1}", "CAM")]

    for ri, cname in enumerate(CLASS_NAMES):
        axes[ri, 0].set_ylabel(cname, fontsize=9, fontweight="bold")
        col = 0
        for k in range(n_samples):
            a, b = axes[ri, col], axes[ri, col + 1]
            if k < len(correct[ri]):
                img, t = correct[ri][k]
                draw(a, b, img, t)
            else:
                empty(a, b)
            col += 2
        for k in range(n_samples):
            a, b = axes[ri, col], axes[ri, col + 1]
            if k < len(wrong[ri]):
                img, t, p = wrong[ri][k]
                draw(a, b, img, p, f"T:{CLASS_NAMES[t][:4]}→P:{CLASS_NAMES[p][:4]}", "red")
            else:
                empty(a, b)
            col += 2

    for ci, lbl in enumerate(col_labels):
        cur = axes[0, ci].get_title()
        axes[0, ci].set_title(f"{lbl}\n{cur}" if cur else lbl, fontsize=8,
                              fontweight="bold", color="green" if ci < n_samples * 2 else "red")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Grad-CAM grid đã lưu: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Xuất TFLite
# ─────────────────────────────────────────────────────────────────────────────
def export_tflite(model, out_path="lung_model.tflite", quantize=True):
    """Chuyển Keras model sang TFLite; quantize=True dùng dynamic-range quantization."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"✓ Đã xuất TFLite: {out_path} ({len(tflite_model) / 1024 ** 2:.2f} MB)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Ví dụ sử dụng trên Colab
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from google.colab import drive
    drive.mount("/content/drive")

    MODELS_DIR = "/content/drive/MyDrive/Models/5Layers"
    IMG_PATH = f"{MODELS_DIR}/Emphysema_24.jpg"

    # V1 — FLOPs, tốc độ, TFLite
    v1 = load_trained_model(
        f"{MODELS_DIR}/model_Customize_TOPSIS_Model_5_Class_Take4_V100_Epoch_100_elu.h5")
    print_flops(v1)
    measure_inference_speed(v1, IMG_PATH, preprocess_rescale, label="V1")
    export_tflite(v1, "lung_model_v1.tflite")

    # V3 — FLOPs, tốc độ, Grad-CAM
    v3 = load_trained_model("/content/drive/MyDrive/COPD_Outputs/best_model.keras")
    print_flops(v3)
    measure_inference_speed(v3, IMG_PATH, preprocess_clahe, label="V3")

    import model_v3_copd5litenet as m3
    test_paths, test_labels, _ = m3.build_file_lists(os.path.join(m3.DATA_ROOT, "test"))
    test_ds = m3.make_dataset(test_paths, test_labels)
    plot_gradcam_grid(v3, test_ds,
                      os.path.join(m3.OUTPUT_DIR, "gradcam_extended.png"))
