"""
Proof of Resizing: Before vs After for 7 images.

Produces (in the SAME folder as fruit_cnn_vscode.py):
    outputs_fruit_cnn/resize_proof_before_after.png
    outputs_fruit_cnn/resize_proof_table.csv
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import kagglehub

SEED = 42
IMG_HEIGHT = 100
IMG_WIDTH = 100
NUM_IMAGES = 7

# Save output next to this file, inside outputs_fruit_cnn/
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs_fruit_cnn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)


# ---------- FIND DATASET ----------
dataset_root = Path(
    kagglehub.dataset_download(
        "kritikseth/fruit-and-vegetable-image-recognition"
    )
)
print("Dataset root:", dataset_root)


# ---------- LOCATE TRAIN DIRECTORY ----------
# FIX: use os.walk(), which yields (root, dirs, files),
# instead of Path.rglob(), which yields a single Path.
import os

train_dir = None
for root, dirs, files in os.walk(dataset_root):
    if os.path.basename(root).lower() in {"train", "training"}:
        train_dir = Path(root)
        break

if train_dir is None:
    raise FileNotFoundError("Could not find train directory.")

print("Train dir:", train_dir)


# ---------- COLLECT IMAGE PATHS ----------
# FIX: rglob on a Path returns Path objects (not tuples).
all_images = []
for p in train_dir.rglob("*"):
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        all_images.append(p)

print(f"Total training images found: {len(all_images)}")

if len(all_images) < NUM_IMAGES:
    raise RuntimeError(
        f"Only {len(all_images)} images found; need at least {NUM_IMAGES}."
    )

random.shuffle(all_images)
sample_paths = all_images[:NUM_IMAGES]

print("\nSelected 7 images:")
for p in sample_paths:
    print("   ", p)


# ---------- BUILD COMPARISON FIGURE ----------
fig, axes = plt.subplots(
    nrows=NUM_IMAGES,
    ncols=2,
    figsize=(8, 3.5 * NUM_IMAGES),
)

rows_for_csv = []

for row, img_path in enumerate(sample_paths):
    # BEFORE: original file from disk (no resize)
    original = Image.open(img_path)
    orig_w, orig_h = original.size

    # AFTER: resized exactly like in training
    resized = original.resize((IMG_WIDTH, IMG_HEIGHT))

    # Left column: original
    ax_left = axes[row][0]
    ax_left.imshow(original)
    ax_left.set_title(
        f"BEFORE (original)\n{orig_w} x {orig_h} px",
        fontsize=10,
        color="darkred",
    )
    ax_left.axis("off")

    # Right column: resized
    ax_right = axes[row][1]
    ax_right.imshow(resized)
    ax_right.set_title(
        f"AFTER (resized)\n{IMG_WIDTH} x {IMG_HEIGHT} px",
        fontsize=10,
        color="darkgreen",
    )
    ax_right.axis("off")

    # Class name on far left
    class_name = img_path.parent.name
    ax_left.text(
        -0.35, 0.5,
        class_name,
        transform=ax_left.transAxes,
        fontsize=9,
        va="center",
        ha="right",
        rotation=90,
        color="navy",
    )

    rows_for_csv.append({
        "Image": img_path.name,
        "Class": class_name,
        "Before Width": orig_w,
        "Before Height": orig_h,
        "After Width": IMG_WIDTH,
        "After Height": IMG_HEIGHT,
    })

# Column headers (top row)
axes[0][0].set_title(
    f"BEFORE (original size on disk)\n"
    f"{rows_for_csv[0]['Before Width']} x {rows_for_csv[0]['Before Height']} px",
    fontsize=12, color="darkred",
)
axes[0][1].set_title(
    f"AFTER (resized to {IMG_WIDTH} x {IMG_HEIGHT})\n"
    f"all images become {IMG_WIDTH} x {IMG_HEIGHT}",
    fontsize=12, color="darkgreen",
)

fig.suptitle(
    "Proof of Resizing: Before vs After (7 sample images)",
    fontsize=15, fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.98])

out_file = OUTPUT_DIR / "resize_proof_before_after.png"
plt.savefig(out_file, dpi=200, bbox_inches="tight")
plt.close()
print(f"\nSaved image: {out_file}")


# ---------- SAVE CSV TABLE ----------
csv_file = OUTPUT_DIR / "resize_proof_table.csv"
pd.DataFrame(rows_for_csv).to_csv(csv_file, index=False)
print(f"Saved table: {csv_file}")


# ---------- PRINT TABLE TO TERMINAL ----------
print("\nBefore vs After dimensions:")
print(pd.DataFrame(rows_for_csv).to_string(index=False))