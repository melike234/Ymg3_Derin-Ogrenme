"""
Derin Öğrenme ve Evrişimsel Sinir Ağlarının (CNN) Temel Yapısının İncelenmesi
Öğrenci: Melike Yetmen | Öğrenci No: 02230224059 | Yazılım Mühendisliği 4. Sınıf

Bu script ödevin 3, 4, 5, 6, 7 ve 8. bölümlerindeki tüm TensorFlow/Keras
uygulamalarını çalıştırır, görselleri 'outputs/' klasörüne kaydeder ve sayısal
çıktıları 'outputs/results.json' dosyasına aktarır.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES_TR = [
    "Tişört/Üst",
    "Pantolon",
    "Kazak",
    "Elbise",
    "Kaban",
    "Sandalet",
    "Gömlek",
    "Spor Ayakkabı",
    "Çanta",
    "Bot",
]


def setup_plot_style() -> None:
    """Tüm grafikler için okunaklı ve sunuma uygun Matplotlib stili ayarlar."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def run_section_3_convolution() -> dict:
    """Bölüm 3: Conv2D katmanı oluşturur, ağırlıkları inceler ve görselleştirir."""
    print("\n" + "=" * 70)
    print("BÖLÜM 3: KONVOLÜSYON İŞLEMİNİN KODLANMASI")
    print("=" * 70)

    tf.keras.utils.set_random_seed(42)

    conv = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        activation="relu",
        kernel_initializer="he_normal",
    )

    dummy_input = tf.random.normal((1, 128, 128, 1))
    output = conv(dummy_input)
    weights, biases = conv.get_weights()

    print("Giriş boyutu:", dummy_input.shape)
    print("Çıkış boyutu:", output.shape)
    print("Filtre ağırlıklarının boyutu:", weights.shape)
    print("Bias boyutu:", biases.shape)
    print("İlk filtre (weights[:, :, 0, 0]):")
    print(weights[:, :, 0, 0])

    # 16 filtrenin başlangıç ağırlıklarını 4x4 grid olarak görselleştir
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    vmax = float(np.max(np.abs(weights)))
    for idx, ax in enumerate(axes.flat):
        kernel = weights[:, :, 0, idx]
        im = ax.imshow(kernel, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Filtre #{idx + 1}", fontsize=10)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.grid(False)
        for r in range(3):
            for c in range(3):
                val = kernel[r, c]
                txt_color = "white" if abs(val) > vmax * 0.55 else "black"
                ax.text(
                    c, r, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=7.5, color=txt_color, fontweight="bold",
                )

    fig.suptitle(
        "Bölüm 3: Conv2D (16 Adet 3×3 Filtre) Başlangıç Ağırlıkları (he_normal)",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(right=0.88, top=0.92, hspace=0.38, wspace=0.30)
    cbar_ax = fig.add_axes([0.90, 0.12, 0.025, 0.76])
    fig.colorbar(im, cax=cbar_ax, label="Ağırlık Değeri")
    fig.savefig(OUTPUT_DIR / "01_baslangic_filtreleri.png", bbox_inches="tight")
    plt.close(fig)

    # Örnek çıkış öznitelik haritalarından ilk 8 tanesini görselleştir
    fig_fm, axes_fm = plt.subplots(2, 4, figsize=(11, 5.5))
    out_np = output.numpy()[0]
    for idx, ax in enumerate(axes_fm.flat):
        ax.imshow(out_np[:, :, idx], cmap="viridis")
        ax.set_title(f"Özellik Haritası #{idx + 1}\n(126×126)", fontsize=9.5)
        ax.axis("off")
    fig_fm.suptitle(
        "Bölüm 3 (Ek): Rastgele Girdi (128×128) Üzerinde İlk 8 Konvolüsyon Çıkışı (ReLU Sonrası)",
        fontsize=12, fontweight="bold",
    )
    fig_fm.tight_layout()
    fig_fm.savefig(OUTPUT_DIR / "02_konvolusyon_ozellik_haritalari.png", bbox_inches="tight")
    plt.close(fig_fm)

    return {
        "input_shape": list(dummy_input.shape),
        "output_shape": list(output.shape),
        "weights_shape": list(weights.shape),
        "biases_shape": list(biases.shape),
        "first_filter": np.round(weights[:, :, 0, 0], 6).tolist(),
        "biases": np.round(biases, 6).tolist(),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "he_normal_theoretical_std": round(math.sqrt(2.0 / 9.0), 6),
        "he_normal_sample_std": round(float(np.std(weights)), 6),
    }


def run_section_4_pooling() -> dict:
    """Bölüm 4: 4x4 matris üzerinde MaxPooling2D(2,2) işlemini uygular ve görselleştirir."""
    print("\n" + "=" * 70)
    print("BÖLÜM 4: POOLING İŞLEMİNİN İNCELENMESİ")
    print("=" * 70)

    input_data = tf.constant([
        [[1., 3., 2., 4.],
         [5., 6., 1., 2.],
         [7., 2., 8., 3.],
         [4., 1., 5., 9.]]
    ])
    input_data = tf.reshape(input_data, (1, 4, 4, 1))

    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
    output = pool(input_data)

    in_mat = input_data.numpy()[0, :, :, 0]
    out_mat = output.numpy()[0, :, :, 0]

    print("Giriş boyutu:", input_data.shape)
    print("Giriş matrisi:\n", in_mat)
    print("Çıkış boyutu:", output.shape)
    print("Pooling sonucu:\n", out_mat)

    # 2x2 Max Pooling görselleştirmesi
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    region_colors = [
        ["#DBEAFE", "#DBEAFE", "#DCFCE7", "#DCFCE7"],
        ["#DBEAFE", "#DBEAFE", "#DCFCE7", "#DCFCE7"],
        ["#FEF3C7", "#FEF3C7", "#FCE7F3", "#FCE7F3"],
        ["#FEF3C7", "#FEF3C7", "#FCE7F3", "#FCE7F3"],
    ]
    max_coords = {(1, 1), (0, 3), (2, 0), (3, 3)}

    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 4)
    ax1.invert_yaxis()
    ax1.set_aspect("equal")
    ax1.set_title("Giriş Matrisi (4×4)\n2×2 Bölgeler")
    ax1.set_xticks(range(5))
    ax1.set_yticks(range(5))
    ax1.grid(False)

    for r in range(4):
        for c in range(4):
            rect = plt.Rectangle((c, r), 1, 1, facecolor=region_colors[r][c], edgecolor="#475569", linewidth=1.2)
            ax1.add_patch(rect)
            is_max = (r, c) in max_coords
            ax1.text(
                c + 0.5, r + 0.5, f"{int(in_mat[r, c])}",
                ha="center", va="center",
                fontsize=14 if is_max else 12,
                fontweight="bold" if is_max else "normal",
                color="#B91C1C" if is_max else "#0F172A",
            )
    ax1.plot([2, 2], [0, 4], color="#0F172A", linewidth=2.5)
    ax1.plot([0, 4], [2, 2], color="#0F172A", linewidth=2.5)

    out_colors = [["#DBEAFE", "#DCFCE7"], ["#FEF3C7", "#FCE7F3"]]
    ax2.set_xlim(0, 2)
    ax2.set_ylim(0, 2)
    ax2.invert_yaxis()
    ax2.set_aspect("equal")
    ax2.set_title("Max Pooling Çıkışı (2×2)\npool_size=(2,2), strides=2")
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.grid(False)

    for r in range(2):
        for c in range(2):
            rect = plt.Rectangle((c, r), 1, 1, facecolor=out_colors[r][c], edgecolor="#0F172A", linewidth=2)
            ax2.add_patch(rect)
            ax2.text(
                c + 0.5, r + 0.5, f"{int(out_mat[r, c])}",
                ha="center", va="center", fontsize=16, fontweight="bold", color="#B91C1C",
            )

    fig.suptitle("Bölüm 4: 2×2 Max Pooling İşleminin Adım Adım Görselleştirilmesi", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_max_pooling_giris_cikis.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "input_shape": list(input_data.shape),
        "output_shape": list(output.shape),
        "input_matrix": in_mat.tolist(),
        "output_matrix": out_mat.tolist(),
    }


def run_section_5_activations() -> dict:
    """Bölüm 5: ReLU ve Softmax aktivasyon fonksiyonlarını test eder ve görselleştirir."""
    print("\n" + "=" * 70)
    print("BÖLÜM 5: ReLU VE SOFTMAX FONKSİYONLARININ İNCELENMESİ")
    print("=" * 70)

    # A. ReLU
    x = tf.constant([-5., -2., -1., 0., 2., 4., 6.])
    relu = tf.keras.layers.ReLU()
    y = relu(x)
    print("ReLU Giriş:", x.numpy())
    print("ReLU Çıkış:", y.numpy())

    # B. Softmax
    logits = tf.constant([[2.0, 1.0, 0.1]])
    softmax = tf.keras.layers.Softmax()
    probabilities = softmax(logits)
    prob_sum = float(tf.reduce_sum(probabilities).numpy())
    print("Softmax Logitleri:", logits.numpy())
    print("Softmax Sonuçları:", probabilities.numpy())
    print("Olasılıkların Toplamı:", prob_sum)

    # Görselleştirme
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    xs_curve = np.linspace(-6, 7, 200)
    ys_curve = np.maximum(0, xs_curve)
    ax1.plot(xs_curve, ys_curve, color="#2563EB", linewidth=2.2, label="ReLU(x) = max(0, x)")
    ax1.scatter(x.numpy(), y.numpy(), color="#DC2626", zorder=5, s=55, label="Örnek Girdi Noktaları")
    for xv, yv in zip(x.numpy(), y.numpy()):
        offset_y = -15 if int(xv) == -1 else 8
        ax1.annotate(f"({int(xv)}, {int(yv)})", (xv, yv), textcoords="offset points", xytext=(0, offset_y), ha="center", fontsize=8)
    ax1.axhline(0, color="#64748B", linewidth=0.8)
    ax1.axvline(0, color="#64748B", linewidth=0.8)
    ax1.set_title("A. ReLU Aktivasyon Fonksiyonu")
    ax1.set_xlabel("Giriş (x)")
    ax1.set_ylabel("Çıkış f(x)")
    ax1.set_ylim(-1.1, 7.2)
    ax1.legend()

    probs_np = probabilities.numpy()[0]
    logits_np = logits.numpy()[0]
    labels = [f"Sınıf {i}\n(logit={logits_np[i]:.1f})" for i in range(len(probs_np))]
    colors = ["#16A34A" if i == int(np.argmax(probs_np)) else "#64748B" for i in range(len(probs_np))]
    bars = ax2.bar(labels, probs_np, color=colors, width=0.55)
    for bar, p in zip(bars, probs_np):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, p + 0.02,
            f"{p:.4f}\n(%{p * 100:.2f})",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax2.set_title(f"B. Softmax Olasılık Dağılımı (Toplam = {prob_sum:.1f})")
    ax2.set_ylabel("Olasılık")
    ax2.set_ylim(0, 0.85)

    fig.suptitle("Bölüm 5: ReLU ve Softmax Aktivasyon Fonksiyonları", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_relu_softmax_grafikleri.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "relu_input": x.numpy().tolist(),
        "relu_output": y.numpy().tolist(),
        "softmax_logits": logits_np.tolist(),
        "softmax_probabilities": [round(float(v), 6) for v in probs_np],
        "softmax_sum": prob_sum,
        "softmax_predicted_class": int(np.argmax(probs_np)),
    }


def load_fashion_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fashion-MNIST veri setini yükler, normalize eder ve kanal boyutunu ekler."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    return x_train, y_train, x_test, y_test


def save_fashion_mnist_samples(x_train: np.ndarray, y_train: np.ndarray) -> None:
    """Fashion-MNIST'in her 10 sınıfından 1'er örnek görüntüyü kaydeder."""
    fig, axes = plt.subplots(2, 5, figsize=(11, 5.6))
    for cls_idx, ax in enumerate(axes.flat):
        sample_idx = int(np.where(y_train == cls_idx)[0][0])
        ax.imshow(x_train[sample_idx, :, :, 0], cmap="gray")
        ax.set_title(f"{cls_idx}: {CLASS_NAMES_TR[cls_idx]}", fontsize=9.5, pad=6)
        ax.axis("off")
    fig.suptitle("Bölüm 6A: Fashion-MNIST Veri Setinden Sınıf Örnekleri (28×28 Gri Seviye)", fontsize=12, fontweight="bold", y=0.97)
    fig.subplots_adjust(hspace=0.28, wspace=0.12, top=0.86)
    fig.savefig(OUTPUT_DIR / "05_fashion_mnist_ornekleri.png", bbox_inches="tight")
    plt.close(fig)


def build_cnn_model(first_conv_filters: int = 8) -> tf.keras.Sequential:
    """Ödevin 6B ve 8. bölümleri için parametrik CNN modeli oluşturur."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(first_conv_filters, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def capture_model_summary(model: tf.keras.Model) -> str:
    """model.summary() çıktısını metin olarak yakalar."""
    buf = io.StringIO()
    model.summary(print_fn=lambda line: buf.write(line + "\n"))
    return buf.getvalue()


def run_experiment(
    first_conv_filters: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[tf.keras.Model, dict, dict]:
    """Verilen ilk Conv2D filtre sayısı ile modeli eğitir ve test eder."""
    tf.keras.utils.set_random_seed(42)
    model = build_cnn_model(first_conv_filters=first_conv_filters)
    summary_text = capture_model_summary(model)

    print(f"\n--- Konfigürasyon: İlk Conv2D Filtre Sayısı = {first_conv_filters} ---")
    print(summary_text)

    start_time = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=2,
    )
    elapsed_sec = time.perf_counter() - start_time

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Eğitim Süresi: {elapsed_sec:.2f} sn")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

    metrics = {
        "first_conv_filters": first_conv_filters,
        "total_params": int(model.count_params()),
        "train_accuracy_final": float(history.history["accuracy"][-1]),
        "val_accuracy_final": float(history.history["val_accuracy"][-1]),
        "train_loss_final": float(history.history["loss"][-1]),
        "val_loss_final": float(history.history["val_loss"][-1]),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "training_time_sec": round(elapsed_sec, 2),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "summary": summary_text,
    }
    return model, history.history, metrics


def plot_baseline_outputs(
    model: tf.keras.Model,
    history_dict: dict,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Bölüm 6D: Temel modelin (8 filtre) eğitim grafikleri, tahmin örnekleri ve karışıklık matrisi."""
    epochs = range(1, len(history_dict["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax1.plot(epochs, history_dict["accuracy"], marker="o", label="Training")
    ax1.plot(epochs, history_dict["val_accuracy"], marker="s", label="Validation")
    ax1.set_title("Accuracy (Doğruluk)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(list(epochs))
    ax1.legend()

    ax2.plot(epochs, history_dict["loss"], marker="o", label="Training")
    ax2.plot(epochs, history_dict["val_loss"], marker="s", label="Validation")
    ax2.set_title("Loss (Kayıp)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_xticks(list(epochs))
    ax2.legend()

    fig.suptitle("Bölüm 6D: Temel CNN Modeli (8 Filtre) Eğitim ve Doğrulama Grafikleri", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_egitim_dogrulama_grafikleri.png", bbox_inches="tight")
    plt.close(fig)

    # Test seti tahminleri ve karışıklık matrisi
    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    fig_pred, axes_pred = plt.subplots(2, 5, figsize=(12.2, 6.4))
    for i, ax in enumerate(axes_pred.flat):
        ax.imshow(x_test[i, :, :, 0], cmap="gray")
        pred_cls = int(y_pred[i])
        true_cls = int(y_test[i])
        conf = float(y_prob[i, pred_cls]) * 100
        color = "#15803D" if pred_cls == true_cls else "#B91C1C"
        ax.set_title(
            f"Tahmin: {CLASS_NAMES_TR[pred_cls]} (%{conf:.1f})\nGerçek: {CLASS_NAMES_TR[true_cls]}",
            fontsize=8.3, color=color, fontweight="bold", pad=6,
        )
        ax.axis("off")
    fig_pred.suptitle("Bölüm 6 (Ek): Test Kümesinden İlk 10 Görüntü İçin Model Tahminleri", fontsize=12, fontweight="bold", y=0.97)
    fig_pred.subplots_adjust(hspace=0.38, wspace=0.14, top=0.85)
    fig_pred.savefig(OUTPUT_DIR / "08_ornek_test_tahminleri.png", bbox_inches="tight")
    plt.close(fig_pred)

    # Karışıklık matrisi (pure NumPy)
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_test, y_pred):
        cm[int(t), int(p)] += 1

    fig_cm, ax_cm = plt.subplots(figsize=(8.8, 7.5))
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks(range(10))
    ax_cm.set_yticks(range(10))
    ax_cm.set_xticklabels(CLASS_NAMES_TR, rotation=35, ha="right", fontsize=8.5)
    ax_cm.set_yticklabels(CLASS_NAMES_TR, fontsize=8.5)
    ax_cm.set_xlabel("Tahmin Edilen Sınıf")
    ax_cm.set_ylabel("Gerçek Sınıf")
    ax_cm.set_title("Bölüm 6 (Ek): Fashion-MNIST Test Seti Karışıklık Matrisi (10.000 Örnek)")
    ax_cm.grid(False)
    thresh = cm.max() / 2.0
    for r in range(10):
        for c in range(10):
            ax_cm.text(
                c, r, str(cm[r, c]),
                ha="center", va="center", fontsize=8,
                color="white" if cm[r, c] > thresh else "black",
            )
    fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    fig_cm.tight_layout()
    fig_cm.savefig(OUTPUT_DIR / "09_karisiklik_matrisi.png", bbox_inches="tight")
    plt.close(fig_cm)

    per_class_acc = {
        CLASS_NAMES_TR[i]: round(float(cm[i, i] / cm[i].sum()), 4)
        for i in range(10)
    }

    # Eğitim sonrası ilk Conv2D katmanının öğrendiği 8 filtrenin görselleştirilmesi
    w_trained, _ = model.layers[0].get_weights()
    fig_tf, axes_tf = plt.subplots(2, 4, figsize=(9.5, 4.8))
    vmax_t = float(np.max(np.abs(w_trained)))
    for idx, ax in enumerate(axes_tf.flat):
        k = w_trained[:, :, 0, idx]
        im_t = ax.imshow(k, cmap="coolwarm", vmin=-vmax_t, vmax=vmax_t)
        ax.set_title(f"Öğrenilen Filtre #{idx + 1}", fontsize=9.5)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.grid(False)
        for r in range(3):
            for c in range(3):
                val = k[r, c]
                txt_c = "white" if abs(val) > vmax_t * 0.55 else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=7.5, color=txt_c, fontweight="bold")
    fig_tf.suptitle("Bölüm 6 (Ek): Eğitim Sonrası İlk Conv2D Katmanının 8 Öğrenilmiş Filtresi (3×3)", fontsize=11.5, fontweight="bold", y=0.98)
    fig_tf.subplots_adjust(right=0.88, top=0.86, hspace=0.38, wspace=0.28)
    cbar_ax_t = fig_tf.add_axes([0.90, 0.14, 0.025, 0.70])
    fig_tf.colorbar(im_t, cax=cbar_ax_t, label="Ağırlık Değeri")
    fig_tf.savefig(OUTPUT_DIR / "07_egitilmis_filtreler.png", bbox_inches="tight")
    plt.close(fig_tf)

    return {
        "per_class_accuracy": per_class_acc,
        "correct_predictions": int(np.sum(y_pred == y_test)),
        "total_test_samples": int(len(y_test)),
        "first_sample_probs": [round(float(v), 4) for v in y_prob[0]],
        "first_sample_pred": int(y_pred[0]),
        "first_sample_true": int(y_test[0]),
    }


def plot_experiment_comparison(exp1: dict, exp2: dict) -> None:
    """Bölüm 8: Deney 1 (8 filtre) ve Deney 2 (16 filtre) karşılaştırma grafiğini çizer."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.3))

    # 1. Doğruluk karşılaştırması
    labels = ["Eğitim\nDoğruluğu", "Doğrulama\nDoğruluğu", "Test\nDoğruluğu"]
    vals_8 = [exp1["train_accuracy_final"] * 100, exp1["val_accuracy_final"] * 100, exp1["test_accuracy"] * 100]
    vals_16 = [exp2["train_accuracy_final"] * 100, exp2["val_accuracy_final"] * 100, exp2["test_accuracy"] * 100]
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax1.bar(x - w / 2, vals_8, width=w, label="Deney 1 (8 Filtre)", color="#3B82F6")
    b2 = ax1.bar(x + w / 2, vals_16, width=w, label="Deney 2 (16 Filtre)", color="#F59E0B")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.18, f"{h:.2f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(82, 91.5)
    ax1.set_ylabel("Doğruluk (%)")
    ax1.set_title("Doğruluk Karşılaştırması")
    ax1.legend(loc="upper right", fontsize=8.5)

    # 2. Epoch bazında test/val doğruluk eğrisi
    ep = range(1, 6)
    ax2.plot(ep, [v * 100 for v in exp1["history"]["val_accuracy"]], marker="o", label="Deney 1 Val (8 Filtre)")
    ax2.plot(ep, [v * 100 for v in exp2["history"]["val_accuracy"]], marker="s", label="Deney 2 Val (16 Filtre)")
    ax2.set_title("Epoch Bazında Doğrulama Doğruluğu")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Doğruluk (%)")
    ax2.set_xticks(list(ep))
    ax2.legend(fontsize=8.5)

    # 3. Eğitim süresi karşılaştırması
    times = [exp1["training_time_sec"], exp2["training_time_sec"]]
    tb = ax3.bar(["Deney 1\n(8 Filtre)", "Deney 2\n(16 Filtre)"], times, color=["#3B82F6", "#F59E0B"], width=0.5)
    for bar, t in zip(tb, times):
        ax3.text(bar.get_x() + bar.get_width() / 2, t + max(times) * 0.02, f"{t:.2f} sn", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax3.set_ylim(0, max(times) * 1.2)
    ax3.set_ylabel("Süre (Saniye)")
    ax3.set_title("Eğitim Süresi Karşılaştırması (5 Epoch)")

    fig.suptitle("Bölüm 8: İlk Conv2D Katmanı Filtre Sayısı Deneyi (8 vs 16 Filtre)", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "10_deney_8_vs_16_karsilastirma.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_plot_style()
    results = {"tf_version": tf.__version__}

    results["section_3"] = run_section_3_convolution()
    results["section_4"] = run_section_4_pooling()
    results["section_5"] = run_section_5_activations()

    print("\n" + "=" * 70)
    print("BÖLÜM 6 & 8: FASHION-MNIST CNN EĞİTİMİ VE DENEY KARŞILAŞTIRMASI")
    print("=" * 70)
    x_train, y_train, x_test, y_test = load_fashion_mnist()
    print("Eğitim verisi:", x_train.shape)
    print("Test verisi:", x_test.shape)
    save_fashion_mnist_samples(x_train, y_train)

    results["dataset"] = {
        "x_train_shape": list(x_train.shape),
        "x_test_shape": list(x_test.shape),
        "y_train_shape": list(y_train.shape),
        "y_test_shape": list(y_test.shape),
    }

    model_8, hist_8, exp1 = run_experiment(8, x_train, y_train, x_test, y_test)
    baseline_details = plot_baseline_outputs(model_8, hist_8, x_test, y_test)

    _, _, exp2 = run_experiment(16, x_train, y_train, x_test, y_test)
    plot_experiment_comparison(exp1, exp2)

    results["section_6_baseline_details"] = baseline_details
    results["experiment_1_8_filters"] = exp1
    results["experiment_2_16_filters"] = exp2

    print("\n" + "=" * 70)
    print("BÖLÜM 8 ÖZET KARŞILAŞTIRMA TABLOSU")
    print("=" * 70)
    print(f"{'Ölçüt':<22} | {'Deney 1 (8 Filtre)':<20} | {'Deney 2 (16 Filtre)':<20}")
    print("-" * 68)
    print(f"{'Filtre Sayısı':<22} | {exp1['first_conv_filters']:<20} | {exp2['first_conv_filters']:<20}")
    print(f"{'Toplam Parametre':<22} | {exp1['total_params']:<20} | {exp2['total_params']:<20}")
    print(f"{'Eğitim Doğruluğu':<22} | {exp1['train_accuracy_final']:<20.4f} | {exp2['train_accuracy_final']:<20.4f}")
    print(f"{'Test Doğruluğu':<22} | {exp1['test_accuracy']:<20.4f} | {exp2['test_accuracy']:<20.4f}")
    print(f"{'Eğitim Süresi (sn)':<22} | {exp1['training_time_sec']:<20.2f} | {exp2['training_time_sec']:<20.2f}")

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    csv_path = OUTPUT_DIR / "deney_karsilastirma_tablosu.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["Ölçüt", "Deney 1 (8 Filtre)", "Deney 2 (16 Filtre)"])
        writer.writerow(["Filtre sayısı", exp1["first_conv_filters"], exp2["first_conv_filters"]])
        writer.writerow(["Eğitim doğruluğu", f"{exp1['train_accuracy_final']:.4f}", f"{exp2['train_accuracy_final']:.4f}"])
        writer.writerow(["Doğrulama doğruluğu", f"{exp1['val_accuracy_final']:.4f}", f"{exp2['val_accuracy_final']:.4f}"])
        writer.writerow(["Test doğruluğu", f"{exp1['test_accuracy']:.4f}", f"{exp2['test_accuracy']:.4f}"])
        writer.writerow(["Test kaybı", f"{exp1['test_loss']:.4f}", f"{exp2['test_loss']:.4f}"])
        writer.writerow(["Toplam parametre", exp1["total_params"], exp2["total_params"]])
        writer.writerow(["Eğitim süresi (sn)", f"{exp1['training_time_sec']:.2f}", f"{exp2['training_time_sec']:.2f}"])

    print(f"\nTüm çıktılar ve grafikler kaydedildi: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
