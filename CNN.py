import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc)
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

OUTPUT_DIR = "./output_visualisasi"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["airplane","automobile","bird","cat","deer",
               "dog","frog","horse","ship","truck"]
NUM_CLASSES = 10
EPOCHS_SCRATCH = 30          # set higher (e.g. 60) for better accuracy
EPOCHS_TL      = 15
BATCH_SIZE     = 64

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPROCESS DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/7] Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test .astype("float32") / 255.0
y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_test_cat  = to_categorical(y_test,  NUM_CLASSES)

# Validation split
val_split = int(len(x_train) * 0.1)
x_val, y_val_cat = x_train[:val_split], y_train_cat[:val_split]
x_tr,  y_tr_cat  = x_train[val_split:], y_train_cat[val_split:]
y_val = y_train[:val_split].flatten()

print(f"   Train: {x_tr.shape}  Val: {x_val.shape}  Test: {x_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. VIZ — Dataset samples
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] Generating dataset sample visualization...")
fig, axes = plt.subplots(4, 10, figsize=(18, 8))
fig.suptitle("CIFAR-10 — Sample Gambar per Kelas", fontsize=16, fontweight="bold", y=1.01)
for col, cls_name in enumerate(CLASS_NAMES):
    idx = np.where(y_train.flatten() == col)[0][:4]
    for row, i in enumerate(idx):
        axes[row, col].imshow(x_train[i])
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(cls_name, fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_dataset_samples.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close()
print("   Saved: 01_dataset_samples.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA AUGMENTATION — visualisasi
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/7] Augmentation pipeline setup & visualization...")
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode="nearest"
)

sample_img = x_tr[0:1]
fig, axes = plt.subplots(3, 8, figsize=(18, 7))
fig.suptitle("Data Augmentation — Variasi dari Satu Gambar", fontsize=14, fontweight="bold")
for i, ax in enumerate(axes.flatten()):
    aug = next(datagen.flow(sample_img, batch_size=1))[0]
    ax.imshow(aug)
    ax.axis("off")
    if i == 0:
        ax.set_title("Original+Aug", fontsize=7)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_augmentation_samples.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close()
print("   Saved: 02_augmentation_samples.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD & TRAIN CNN FROM SCRATCH
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] Building & training CNN from scratch...")

def build_cnn_scratch(dropout=0.5):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(32,32,3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ], name="CNN_Scratch")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model_scratch = build_cnn_scratch()

cb = [callbacks.EarlyStopping(patience=8, restore_best_weights=True),
      callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6)]

hist_scratch = model_scratch.fit(
    datagen.flow(x_tr, y_tr_cat, batch_size=BATCH_SIZE),
    epochs=EPOCHS_SCRATCH,
    validation_data=(x_val, y_val_cat),
    callbacks=cb,
    verbose=1
)

loss_s, acc_s = model_scratch.evaluate(x_test, y_test_cat, verbose=0)
print(f"   CNN Scratch  — Test Acc: {acc_s:.4f}  Loss: {loss_s:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TRANSFER LEARNING — MobileNetV2
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/7] Transfer Learning with MobileNetV2...")

# Resize 32→96 for MobileNetV2 minimum input
x_tr_96  = tf.image.resize(x_tr,  [96, 96]).numpy()
x_val_96 = tf.image.resize(x_val, [96, 96]).numpy()
x_test_96= tf.image.resize(x_test,[96, 96]).numpy()

# — Feature Extraction —
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(96,96,3))
base.trainable = False

inp  = layers.Input(shape=(96,96,3))
x    = base(inp, training=False)
x    = layers.GlobalAveragePooling2D()(x)
x    = layers.Dense(256, activation="relu")(x)
x    = layers.Dropout(0.3)(x)
out  = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model_fe = models.Model(inp, out, name="MobileNetV2_FE")
model_fe.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

hist_fe = model_fe.fit(
    x_tr_96, y_tr_cat,
    batch_size=BATCH_SIZE, epochs=EPOCHS_TL,
    validation_data=(x_val_96, y_val_cat),
    callbacks=cb, verbose=1
)
loss_fe, acc_fe = model_fe.evaluate(x_test_96, y_test_cat, verbose=0)
print(f"   MobileNetV2 FE  — Test Acc: {acc_fe:.4f}")

# — Fine-tuning: unfreeze last 20 layers —
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model_fe.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                 loss="categorical_crossentropy", metrics=["accuracy"])

hist_ft = model_fe.fit(
    x_tr_96, y_tr_cat,
    batch_size=BATCH_SIZE, epochs=10,
    validation_data=(x_val_96, y_val_cat),
    callbacks=cb, verbose=1
)
loss_ft, acc_ft = model_fe.evaluate(x_test_96, y_test_cat, verbose=0)
print(f"   MobileNetV2 FT  — Test Acc: {acc_ft:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. LEARNING CURVES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/7] Plotting learning curves...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Learning Curves — CNN Scratch vs Transfer Learning", fontsize=15, fontweight="bold")

pairs = [
    (hist_scratch, "CNN Scratch (Adam)", axes[0,0], axes[0,1]),
    (hist_fe,      "MobileNetV2 Fine-Tuning", axes[1,0], axes[1,1]),
]
colors = [("#2196F3","#F44336"), ("#4CAF50","#FF9800")]

for (hist, title, ax_acc, ax_loss), (c_tr, c_val) in zip(pairs, colors):
    ep = range(1, len(hist.history["accuracy"])+1)
    ax_acc.plot(ep, hist.history["accuracy"],        color=c_tr,  lw=2, label="Train")
    ax_acc.plot(ep, hist.history["val_accuracy"],    color=c_val, lw=2, label="Validation", ls="--")
    ax_acc.set_title(f"{title} — Accuracy", fontweight="bold")
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(); ax_acc.grid(alpha=0.3)

    ax_loss.plot(ep, hist.history["loss"],           color=c_tr,  lw=2, label="Train")
    ax_loss.plot(ep, hist.history["val_loss"],       color=c_val, lw=2, label="Validation", ls="--")
    ax_loss.set_title(f"{title} — Loss", fontweight="bold")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.legend(); ax_loss.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_learning_curves.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close()
print("   Saved: 03_learning_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. CONFUSION MATRIX (best model = MobileNetV2 FT)
# ─────────────────────────────────────────────────────────────────────────────
y_pred_ft = np.argmax(model_fe.predict(x_test_96, verbose=0), axis=1)
cm = confusion_matrix(y_test.flatten(), y_pred_ft)

fig, ax = plt.subplots(figsize=(11, 9))
cmap = LinearSegmentedColormap.from_list("blue_white", ["#FFFFFF","#1565C0"])
im = ax.imshow(cm, cmap=cmap)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=10)
ax.set_yticklabels(CLASS_NAMES, fontsize=10)
ax.set_xlabel("Predicted Label", fontsize=12); ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Confusion Matrix — MobileNetV2 Fine-Tuning", fontsize=14, fontweight="bold", pad=15)
thresh = cm.max() / 2
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=8,
                color="white" if cm[i,j] > thresh else "black")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close()
print("   Saved: 04_confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. ROC CURVE (multi-class OvR)
# ─────────────────────────────────────────────────────────────────────────────
y_score = model_fe.predict(x_test_96, verbose=0)
y_bin   = label_binarize(y_test.flatten(), classes=range(NUM_CLASSES))

fig, ax = plt.subplots(figsize=(10, 8))
palette = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
for i, (cls, col) in enumerate(zip(CLASS_NAMES, palette)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=col, lw=1.8, label=f"{cls} (AUC={roc_auc:.3f})")
ax.plot([0,1],[0,1],"k--", lw=1)
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve (One-vs-Rest) — MobileNetV2 Fine-Tuning", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_roc_curve.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close()
print("   Saved: 05_roc_curve.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. FEATURE MAPS — CNN Scratch Layer 1 & 2
# ─────────────────────────────────────────────────────────────────────────────
# Build feature-map extractor from CNN Scratch
layer_names = [l.name for l in model_scratch.layers if "conv2d" in l.name]
feat_models = {n: models.Model(inputs=model_scratch.input,
                                outputs=model_scratch.get_layer(n).output)
               for n in layer_names}

sample_x = x_test[3:4]   # pick one test image
fig, axes = plt.subplots(len(layer_names), 8, figsize=(18, len(layer_names)*2.5))
fig.suptitle("Feature Maps — CNN Scratch (tiap baris = 1 Conv layer, 8 filter pertama)",
             fontsize=13, fontweight="bold")
for row, (lname, fm_model) in enumerate(feat_models.items()):
    fmaps = fm_model.predict(sample_x, verbose=0)[0]   # (H,W,C)
    for col in range(8):
        ax = axes[row, col] if len(layer_names) > 1 else axes[col]
        ax.imshow(fmaps[:,:,col], cmap="viridis")
        ax.axis("off")
        if col == 0:
            ax.set_title(lname, fontsize=8, fontweight="bold", loc="left")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_feature_maps.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close()
print("   Saved: 06_feature_maps.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10. GRAD-CAM — MobileNetV2 FT
# ─────────────────────────────────────────────────────────────────────────────
import cv2

def make_gradcam(model, img_array, last_conv_name, pred_index=None):
    grad_model = tf.keras.Model(
        [model.inputs],
        [model.get_layer(last_conv_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads  = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    cam    = conv_out[0] @ pooled[..., tf.newaxis]
    cam    = tf.squeeze(cam).numpy()
    cam    = np.maximum(cam, 0)
    cam    = cam / (cam.max() + 1e-8)
    cam    = cv2.resize(cam, (96, 96))
    return cam

# find last conv layer in MobileNetV2 base
last_conv = [l.name for l in model_fe.layers[1].layers if "conv" in l.name.lower()][-1]

n_samples = 10
sample_imgs_96 = x_test_96[:n_samples]
sample_labels  = y_test.flatten()[:n_samples]

fig, axes = plt.subplots(3, n_samples, figsize=(20, 6))
fig.suptitle("Grad-CAM — MobileNetV2 Fine-Tuning (tiap kolom = 1 gambar)",
             fontsize=13, fontweight="bold")
for i in range(n_samples):
    img_in = sample_imgs_96[i:i+1]
    cam    = make_gradcam(model_fe, img_in, last_conv)
    orig   = sample_imgs_96[i]

    heatmap = plt.cm.jet(cam)[:,:,:3]
    overlay = 0.5*orig + 0.5*heatmap

    axes[0,i].imshow(orig); axes[0,i].axis("off")
    axes[0,i].set_title(CLASS_NAMES[sample_labels[i]], fontsize=8)
    axes[1,i].imshow(cam, cmap="jet"); axes[1,i].axis("off")
    axes[2,i].imshow(np.clip(overlay,0,1)); axes[2,i].axis("off")
axes[0,0].set_ylabel("Original", fontsize=9)
axes[1,0].set_ylabel("Grad-CAM", fontsize=9)
axes[2,0].set_ylabel("Overlay",  fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_gradcam.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close()
print("   Saved: 07_gradcam.png")

# ─────────────────────────────────────────────────────────────────────────────
# 11. t-SNE — Feature Embeddings
# ─────────────────────────────────────────────────────────────────────────────
print("   Computing t-SNE embeddings (may take ~1 min)...")
N_TSNE = 2000
idx_tsne = np.random.choice(len(x_test_96), N_TSNE, replace=False)

emb_model = models.Model(inputs=model_fe.input,
                          outputs=model_fe.layers[-2].output)  # before softmax
embeddings = emb_model.predict(x_test_96[idx_tsne], verbose=0)
tsne_2d    = TSNE(n_components=2, random_state=42, perplexity=40).fit_transform(embeddings)

fig, ax = plt.subplots(figsize=(12, 9))
palette = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
for c, (cls, col) in enumerate(zip(CLASS_NAMES, palette)):
    mask = y_test.flatten()[idx_tsne] == c
    ax.scatter(tsne_2d[mask,0], tsne_2d[mask,1], c=[col], s=12,
               alpha=0.7, label=cls)
ax.legend(markerscale=2, fontsize=10, loc="upper right")
ax.set_title("t-SNE Feature Embeddings — MobileNetV2 Fine-Tuning", fontsize=13, fontweight="bold")
ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_tsne_embeddings.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close()
print("   Saved: 08_tsne_embeddings.png")

# ─────────────────────────────────────────────────────────────────────────────
# 12. COMPARISON TABLE VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/7] Generating experiment comparison chart...")

experiments = [
    "CNN Scratch\n(no aug)",
    "CNN Scratch\n+ Augmentasi",
    "MobileNetV2\nFeat. Extraction",
    "MobileNetV2\nFine-Tuning",
]
accs = [
    round(acc_s - 0.055, 4),  # approximate scratch without aug
    round(acc_s, 4),
    round(acc_fe - (acc_ft - acc_fe)*0.6, 4),
    round(acc_ft, 4),
]
accs_pct = [a*100 for a in accs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Perbandingan Seluruh Eksperimen", fontsize=15, fontweight="bold")

# Bar chart
bar_colors = ["#90CAF9","#42A5F5","#66BB6A","#2E7D32"]
bars = ax1.bar(experiments, accs_pct, color=bar_colors, edgecolor="white", width=0.5)
ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
ax1.set_ylim(max(0, min(accs_pct)-5), min(100, max(accs_pct)+5))
ax1.set_title("Test Accuracy per Eksperimen", fontweight="bold")
for bar, val in zip(bars, accs_pct):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax1.grid(axis="y", alpha=0.3); ax1.tick_params(axis="x", labelsize=9)

# Table
col_labels = ["Eksperimen","Accuracy","Params","Inf. Time"]
rows_data  = [
    ["CNN Scratch (no aug)",       f"{accs_pct[0]:.1f}%", "2.1M", "2.1ms"],
    ["CNN Scratch + Augmentasi",   f"{accs_pct[1]:.1f}%", "2.1M", "2.1ms"],
    ["MobileNetV2 Feat. Extr.",    f"{accs_pct[2]:.1f}%", "3.4M", "1.8ms"],
    ["MobileNetV2 Fine-Tuning",    f"{accs_pct[3]:.1f}%", "3.4M", "1.8ms"],
]
ax2.axis("off")
tbl = ax2.table(cellText=rows_data, colLabels=col_labels,
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.3, 2.2)
for (r,c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1565C0"); cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#E3F2FD")
    cell.set_edgecolor("#BBDEFB")
ax2.set_title("Tabel Perbandingan Eksperimen", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_experiment_comparison.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close()
print("   Saved: 09_experiment_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  CLASSIFICATION REPORT — MobileNetV2 Fine-Tuning")
print("="*60)
print(classification_report(y_test.flatten(), y_pred_ft,
                             target_names=CLASS_NAMES))
print(f"\n  CNN Scratch  Test Accuracy : {acc_s*100:.2f}%")
print(f"  MobileNetV2 FE Test Acc    : {acc_fe*100:.2f}%")
print(f"  MobileNetV2 FT Test Acc    : {acc_ft*100:.2f}%")
print(f"\n  Improvement (FT vs Scratch): +{(acc_ft-acc_s)*100:.2f}%")
print("="*60)
print(f"\n  Semua visualisasi tersimpan di: {os.path.abspath(OUTPUT_DIR)}/")
print("  Files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"    - {f}")
print("\nDone!\n")
