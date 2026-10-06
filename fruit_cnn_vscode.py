"""
Fruit and Vegetable Image Classification using CNN

Dataset:
https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition

Install required packages:
pip install tensorflow kagglehub scikit-learn pandas matplotlib seaborn pillow

Run:
python fruit_cnn_vscode.py

All output files will be saved in:
outputs_fruit_cnn/
"""

import os
import sys
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

# Image preprocessing size
IMG_HEIGHT = 100
IMG_WIDTH = 100

# Use 8 if you get memory errors on your laptop.
BATCH_SIZE = 16

# Main CNN training epochs
MAIN_MODEL_EPOCHS = 15

# Epoch count for each experiment
EXPERIMENT_EPOCHS = 6

# First run: keep this False.
# After main CNN works successfully, set it to True.
RUN_EXPERIMENTS = False

# How many classes to display in readable confusion matrix
DISPLAY_CONFUSION_CLASSES = 25

# ------------------------------------------------------------
# FIX FOR OUTPUT DIRECTORY ERROR
# This makes the output folder beside this Python file.
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs_fruit_cnn"

# Create output directory safely.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def print_heading(text):
    """Print a formatted terminal heading."""
    print("\n" + "=" * 75)
    print(text)
    print("=" * 75)


def find_dataset_split_directories(dataset_root):
    """
    Automatically find:
    - train directory
    - validation / valid / val directory
    - test directory
    """

    train_dir = None
    val_dir = None
    test_dir = None

    train_names = {"train", "training"}
    validation_names = {"validation", "valid", "val"}
    test_names = {"test", "testing"}

    for root, dirs, files in os.walk(dataset_root):
        folder_name = os.path.basename(root).lower()

        if folder_name in train_names:
            train_dir = root

        elif folder_name in validation_names:
            val_dir = root

        elif folder_name in test_names:
            test_dir = root

    if train_dir is None:
        raise FileNotFoundError(
            f"Could not find train directory inside:\n{dataset_root}"
        )

    if val_dir is None:
        raise FileNotFoundError(
            f"Could not find validation directory inside:\n{dataset_root}"
        )

    if test_dir is None:
        raise FileNotFoundError(
            f"Could not find test directory inside:\n{dataset_root}"
        )

    return train_dir, val_dir, test_dir


def print_dataset_structure(dataset_root):
    """Print limited dataset structure for debugging."""

    print("\nDATASET FOLDER STRUCTURE:")

    folder_count = 0

    for root, dirs, files in os.walk(dataset_root):
        level = root.replace(str(dataset_root), "").count(os.sep)
        indentation = " " * 4 * level

        print(f"{indentation}{os.path.basename(root)}/")

        folder_count += 1

        if folder_count >= 60:
            print("    ... folder display stopped ...")
            break


def save_sample_images(dataset, class_names):
    """Save 16 sample training images."""

    images, labels = next(iter(dataset))

    plt.figure(figsize=(12, 10))

    for i in range(min(16, len(images))):
        plt.subplot(4, 4, i + 1)

        plt.imshow(images[i].numpy())

        label_index = int(labels[i].numpy())

        plt.title(
            class_names[label_index],
            fontsize=8
        )

        plt.axis("off")

    plt.suptitle("Sample Training Images", fontsize=16)
    plt.tight_layout()

    output_file = OUTPUT_DIR / "sample_training_images.png"

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved sample image grid: {output_file}")


def build_cnn(
    num_classes,
    filters=(32, 64, 128),
    dense_units=256,
    learning_rate=0.001,
    dropout_rate=0.30,
):
    """
    CNN architecture containing:
    - Convolutional layers
    - ReLU activation
    - Max-pooling layers
    - Flatten layer
    - Dense / Fully-connected layers
    - Softmax output layer
    """

    model_layers = [
        layers.Input(
            shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        ),

        # Image augmentation happens only during training.
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
    ]

    # Dynamic convolutional blocks
    for filter_count in filters:
        model_layers.extend([
            layers.Conv2D(
                filters=filter_count,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            ),

            layers.MaxPooling2D(
                pool_size=(2, 2)
            ),
        ])

    # Required classification layers
    model_layers.extend([
        layers.Flatten(),

        layers.Dense(
            dense_units,
            activation="relu"
        ),

        layers.Dropout(dropout_rate),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.20),

        layers.Dense(
            num_classes,
            activation="softmax"
        ),
    ])

    model = models.Sequential(model_layers)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"],
    )

    return model


def plot_training_history(history):
    """Save training/validation accuracy and loss graphs."""

    plt.figure(figsize=(14, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy",
        linewidth=2
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy",
        linewidth=2
    )

    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss plot
    plt.subplot(1, 2, 2)

    plt.plot(
        history.history["loss"],
        label="Training Loss",
        linewidth=2
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
        linewidth=2
    )

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = OUTPUT_DIR / "main_model_training_history.png"

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved training history graph: {output_file}")


def collect_predictions(model, test_ds):
    """
    Predict all test images.

    Stores all labels and predictions, but only keeps
    up to 16 correct and 16 incorrect images in RAM.
    """

    y_true = []
    y_pred = []

    correct_examples = []
    incorrect_examples = []

    for images, labels in test_ds:
        probabilities = model.predict(images, verbose=0)

        predictions = np.argmax(
            probabilities,
            axis=1
        )

        labels_numpy = labels.numpy()

        y_true.extend(labels_numpy)
        y_pred.extend(predictions)

        for image, actual, predicted, probability in zip(
            images.numpy(),
            labels_numpy,
            predictions,
            probabilities,
        ):
            confidence = float(np.max(probability))

            # Save correct examples
            if actual == predicted and len(correct_examples) < 16:
                correct_examples.append(
                    (
                        image,
                        int(actual),
                        int(predicted),
                        confidence,
                    )
                )

            # Save incorrect examples
            if actual != predicted and len(incorrect_examples) < 16:
                incorrect_examples.append(
                    (
                        image,
                        int(actual),
                        int(predicted),
                        confidence,
                    )
                )

    return (
        np.array(y_true),
        np.array(y_pred),
        correct_examples,
        incorrect_examples,
    )


def save_prediction_images(
    examples,
    class_names,
    title,
    output_name,
    is_correct,
):
    """Save a 4x4 grid of correct or incorrect predictions."""

    if len(examples) == 0:
        print(f"No examples available for: {title}")
        return

    plt.figure(figsize=(16, 10))

    for i, (image, actual, predicted, confidence) in enumerate(examples):
        plt.subplot(4, 4, i + 1)

        plt.imshow(image)

        text_color = "green" if is_correct else "red"

        plt.title(
            f"Actual: {class_names[actual]}\n"
            f"Predicted: {class_names[predicted]}\n"
            f"Confidence: {confidence * 100:.1f}%",
            fontsize=8,
            color=text_color,
        )

        plt.axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    output_file = OUTPUT_DIR / output_name

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved prediction image grid: {output_file}")


def save_confusion_matrices(y_true, y_pred, class_names):
    """Save full, readable, and normalized confusion matrices."""

    cm = confusion_matrix(y_true, y_pred)

    # Full confusion matrix
    plt.figure(figsize=(18, 16))

    sns.heatmap(
        cm,
        cmap="Blues",
        cbar=True,
    )

    plt.title("Full Confusion Matrix")
    plt.xlabel("Predicted Class Index")
    plt.ylabel("Actual Class Index")
    plt.tight_layout()

    full_cm_file = OUTPUT_DIR / "confusion_matrix_full.png"

    plt.savefig(
        full_cm_file,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved full confusion matrix: {full_cm_file}")

    # Readable first 25 classes
    display_count = min(
        DISPLAY_CONFUSION_CLASSES,
        len(class_names)
    )

    cm_small = cm[:display_count, :display_count]

    plt.figure(figsize=(18, 15))

    sns.heatmap(
        cm_small,
        cmap="Blues",
        annot=True,
        fmt="d",
        xticklabels=class_names[:display_count],
        yticklabels=class_names[:display_count],
    )

    plt.title(
        f"Confusion Matrix: First {display_count} Classes"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()

    small_cm_file = (
        OUTPUT_DIR / "confusion_matrix_first_25_classes.png"
    )

    plt.savefig(
        small_cm_file,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved readable confusion matrix: {small_cm_file}")

    # Normalized confusion matrix
    row_sums = cm.sum(axis=1, keepdims=True)

    cm_normalized = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    plt.figure(figsize=(18, 15))

    sns.heatmap(
        cm_normalized[:display_count, :display_count],
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        xticklabels=class_names[:display_count],
        yticklabels=class_names[:display_count],
    )

    plt.title(
        f"Normalized Confusion Matrix: First {display_count} Classes"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()

    normalized_cm_file = (
        OUTPUT_DIR / "normalized_confusion_matrix_first_25_classes.png"
    )

    plt.savefig(
        normalized_cm_file,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved normalized confusion matrix: "
        f"{normalized_cm_file}"
    )


def evaluate_model(model, test_ds, class_names):
    """
    Evaluate CNN using:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Classification report
    - Confusion matrix
    - Correct / incorrect images
    """

    print_heading("EVALUATING CNN MODEL")

    test_loss, test_accuracy = model.evaluate(
        test_ds,
        verbose=1
    )

    (
        y_true,
        y_pred,
        correct_examples,
        incorrect_examples,
    ) = collect_predictions(
        model,
        test_ds
    )

    accuracy = accuracy_score(y_true, y_pred)

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    print("\nFINAL TEST METRICS")
    print("-" * 55)

    print(f"Test Loss:                 {test_loss:.4f}")
    print(f"Test Accuracy:             {test_accuracy * 100:.2f}%")
    print(f"Calculated Accuracy:       {accuracy * 100:.2f}%")
    print(f"Weighted Precision:        {weighted_precision:.4f}")
    print(f"Weighted Recall:           {weighted_recall:.4f}")
    print(f"Weighted F1-score:         {weighted_f1:.4f}")
    print(f"Macro Precision:           {macro_precision:.4f}")
    print(f"Macro Recall:              {macro_recall:.4f}")
    print(f"Macro F1-score:            {macro_f1:.4f}")
    print(f"Correct Predictions:       {np.sum(y_true == y_pred)}")
    print(f"Incorrect Predictions:     {np.sum(y_true != y_pred)}")

    # Per-class report
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    report_file = OUTPUT_DIR / "classification_report.csv"

    report_df.to_csv(report_file)

    print(f"\nSaved classification report: {report_file}")

    # Overall metrics CSV
    metrics = {
        "Test Loss": test_loss,
        "Test Accuracy": test_accuracy,
        "Calculated Accuracy": accuracy,
        "Weighted Precision": weighted_precision,
        "Weighted Recall": weighted_recall,
        "Weighted F1 Score": weighted_f1,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1 Score": macro_f1,
        "Total Test Images": len(y_true),
        "Correct Predictions": int(np.sum(y_true == y_pred)),
        "Incorrect Predictions": int(np.sum(y_true != y_pred)),
    }

    metrics_df = pd.DataFrame([metrics])

    metrics_file = OUTPUT_DIR / "main_model_metrics.csv"

    metrics_df.to_csv(
        metrics_file,
        index=False
    )

    print(f"Saved final metrics: {metrics_file}")

    # Correct prediction images
    save_prediction_images(
        examples=correct_examples,
        class_names=class_names,
        title="Correctly Classified Images",
        output_name="correct_predictions.png",
        is_correct=True,
    )

    # Incorrect prediction images
    save_prediction_images(
        examples=incorrect_examples,
        class_names=class_names,
        title="Incorrectly Classified Images",
        output_name="incorrect_predictions.png",
        is_correct=False,
    )

    # Confusion matrices
    save_confusion_matrices(
        y_true,
        y_pred,
        class_names,
    )

    return metrics


def run_experiment(
    experiment_name,
    train_ds,
    val_ds,
    test_ds,
    num_classes,
    filters,
    learning_rate,
    epochs,
):
    """Train and test one hyperparameter experiment."""

    print_heading(f"EXPERIMENT: {experiment_name}")

    experiment_model = build_cnn(
        num_classes=num_classes,
        filters=filters,
        dense_units=256,
        learning_rate=learning_rate,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    history = experiment_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = experiment_model.evaluate(
        test_ds,
        verbose=0,
    )

    result = {
        "Experiment": experiment_name,
        "Convolution Layers": len(filters),
        "Filters": str(filters),
        "Learning Rate": learning_rate,
        "Maximum Epochs": epochs,
        "Actual Epochs Run": len(history.history["loss"]),
        "Best Training Accuracy": max(history.history["accuracy"]),
        "Best Validation Accuracy": max(history.history["val_accuracy"]),
        "Test Accuracy": test_accuracy,
        "Test Loss": test_loss,
    }

    print("\nExperiment Result")

    for key, value in result.items():
        print(f"{key}: {value}")

    del experiment_model
    tf.keras.backend.clear_session()

    return result


# ============================================================
# 4. MAIN FUNCTION
# ============================================================

def main():
    """
    Main program execution.
    """

    # Extra safety: ensure output folder always exists.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_heading(
        "FRUIT AND VEGETABLE IMAGE CLASSIFICATION USING CNN"
    )

    print("TensorFlow Version:", tf.__version__)
    print("Available GPU Devices:", tf.config.list_physical_devices("GPU"))
    print("Output Folder:", OUTPUT_DIR.resolve())

    # ========================================================
    # 5. DOWNLOAD DATASET
    # ========================================================

    print_heading("DOWNLOADING / LOCATING DATASET")

    try:
        dataset_root = kagglehub.dataset_download(
            "kritikseth/fruit-and-vegetable-image-recognition"
        )

        print("Dataset downloaded/found at:")
        print(dataset_root)

    except Exception as error:
        print("\nKaggle dataset download failed.")
        print("Reason:", error)

        print(
            "\nCheck the following:\n"
            "1. Internet is connected.\n"
            "2. Kaggle authentication is completed.\n"
            "3. kagglehub package is installed.\n"
        )

        sys.exit(1)

    # Display folders for debugging
    print_dataset_structure(dataset_root)

    try:
        train_dir, val_dir, test_dir = (
            find_dataset_split_directories(dataset_root)
        )

    except FileNotFoundError as error:
        print(error)
        sys.exit(1)

    print("\nDetected Dataset Directories")
    print("-" * 55)
    print("Training directory:  ", train_dir)
    print("Validation directory:", val_dir)
    print("Test directory:      ", test_dir)

    # ========================================================
    # 6. LOAD AND PREPROCESS DATA
    # ========================================================

    print_heading("LOADING AND PREPROCESSING DATA")

    print(
        f"Image size: {IMG_HEIGHT} x {IMG_WIDTH}\n"
        "Normalization: Pixels scaled from [0, 255] to [0, 1]\n"
        "Dataset split: Existing train / validation / test folders\n"
        f"Batch size: {BATCH_SIZE}"
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=SEED,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    class_names = train_ds.class_names
    validation_class_names = val_ds.class_names
    test_class_names = test_ds.class_names

    num_classes = len(class_names)

    if class_names != validation_class_names:
        print("\nWARNING: Training and validation class order differs.")

    if class_names != test_class_names:
        print("\nWARNING: Training and test class order differs.")

    print(f"\nNumber of Classes: {num_classes}")

    print("\nClass Names:")

    for index, class_name in enumerate(class_names):
        print(f"{index}: {class_name}")

    # Save all class names
    class_file = OUTPUT_DIR / "class_names.txt"

    with open(class_file, "w", encoding="utf-8") as file:
        for index, class_name in enumerate(class_names):
            file.write(f"{index}: {class_name}\n")

    print(f"\nSaved class names: {class_file}")

    # Normalize pixel values from [0, 255] to [0, 1]
    normalization_layer = layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(
        lambda images, labels: (
            normalization_layer(images),
            labels,
        ),
        num_parallel_calls=AUTOTUNE,
    )

    val_ds = val_ds.map(
        lambda images, labels: (
            normalization_layer(images),
            labels,
        ),
        num_parallel_calls=AUTOTUNE,
    )

    test_ds = test_ds.map(
        lambda images, labels: (
            normalization_layer(images),
            labels,
        ),
        num_parallel_calls=AUTOTUNE,
    )

    # Improve data loading performance
    train_ds = train_ds.shuffle(
        buffer_size=1000,
        seed=SEED,
    ).prefetch(AUTOTUNE)

    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    save_sample_images(
        train_ds,
        class_names,
    )

    # ========================================================
    # 7. BUILD CNN
    # ========================================================

    print_heading("BUILDING CNN MODEL")

    model = build_cnn(
        num_classes=num_classes,
        filters=(32, 64, 128),
        dense_units=256,
        learning_rate=0.001,
        dropout_rate=0.30,
    )

    model.summary()

    # Save CNN architecture summary
    model_summary_file = OUTPUT_DIR / "model_summary.txt"

    with open(model_summary_file, "w", encoding="utf-8") as file:
        model.summary(
            print_fn=lambda line: file.write(line + "\n")
        )

    print(f"\nSaved model summary: {model_summary_file}")

    # ========================================================
    # 8. TRAIN MAIN MODEL
    # ========================================================

    print_heading("TRAINING MAIN CNN MODEL")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAIN_MODEL_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    plot_training_history(history)

    # Save trained CNN model
    model_file = OUTPUT_DIR / "fruit_cnn_model.keras"

    model.save(model_file)

    print(f"Saved trained model: {model_file}")

    # ========================================================
    # 9. EVALUATE MAIN MODEL
    # ========================================================

    final_metrics = evaluate_model(
        model=model,
        test_ds=test_ds,
        class_names=class_names,
    )

    # ========================================================
    # 10. HYPERPARAMETER EXPERIMENTS
    # ========================================================

    if RUN_EXPERIMENTS:
        print_heading("HYPERPARAMETER EXPERIMENTS")

        print(
            "Each experiment changes one parameter.\n"
            "This can take time on a CPU-only laptop."
        )

        experiment_settings = [
            {
                "experiment_name": "Baseline - 3 Convolution Layers",
                "filters": (32, 64, 128),
                "learning_rate": 0.001,
                "epochs": EXPERIMENT_EPOCHS,
            },

            {
                "experiment_name": "Fewer Layers - 2 Convolution Layers",
                "filters": (32, 64),
                "learning_rate": 0.001,
                "epochs": EXPERIMENT_EPOCHS,
            },

            {
                "experiment_name": "More Layers - 4 Convolution Layers",
                "filters": (32, 64, 128, 256),
                "learning_rate": 0.001,
                "epochs": EXPERIMENT_EPOCHS,
            },

            {
                "experiment_name": "More Filters - 64, 128, 256",
                "filters": (64, 128, 256),
                "learning_rate": 0.001,
                "epochs": EXPERIMENT_EPOCHS,
            },

            {
                "experiment_name": "Lower Learning Rate - 0.0001",
                "filters": (32, 64, 128),
                "learning_rate": 0.0001,
                "epochs": EXPERIMENT_EPOCHS,
            },

            {
                "experiment_name": "More Epochs",
                "filters": (32, 64, 128),
                "learning_rate": 0.001,
                "epochs": EXPERIMENT_EPOCHS * 2,
            },
        ]

        experiment_results = []

        for setting in experiment_settings:
            result = run_experiment(
                experiment_name=setting["experiment_name"],
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                num_classes=num_classes,
                filters=setting["filters"],
                learning_rate=setting["learning_rate"],
                epochs=setting["epochs"],
            )

            experiment_results.append(result)

        results_df = pd.DataFrame(experiment_results)

        results_df = results_df.sort_values(
            by="Test Accuracy",
            ascending=False,
        ).reset_index(drop=True)

        results_file = (
            OUTPUT_DIR / "hyperparameter_experiment_results.csv"
        )

        results_df.to_csv(
            results_file,
            index=False,
        )

        print_heading("EXPERIMENT RESULTS")

        print(results_df.to_string(index=False))

        print(f"\nSaved experiment results: {results_file}")

        # Create experiment comparison graph
        plt.figure(figsize=(13, 7))

        sns.barplot(
            data=results_df,
            x="Test Accuracy",
            y="Experiment",
            palette="viridis",
        )

        plt.title("CNN Hyperparameter Experiments: Test Accuracy")
        plt.xlabel("Test Accuracy")
        plt.ylabel("Experiment")
        plt.xlim(0, 1)
        plt.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        experiment_plot_file = (
            OUTPUT_DIR / "experiment_test_accuracy_comparison.png"
        )

        plt.savefig(
            experiment_plot_file,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

        print(
            "Saved experiment comparison graph: "
            f"{experiment_plot_file}"
        )

    # ========================================================
    # 11. PROJECT FINISHED
    # ========================================================

    print_heading("PROJECT COMPLETED SUCCESSFULLY")

    print("All generated files are saved in:")
    print(OUTPUT_DIR.resolve())

    print("\nImportant files:")
    print("- class_names.txt")
    print("- sample_training_images.png")
    print("- model_summary.txt")
    print("- fruit_cnn_model.keras")
    print("- main_model_training_history.png")
    print("- main_model_metrics.csv")
    print("- classification_report.csv")
    print("- confusion_matrix_full.png")
    print("- confusion_matrix_first_25_classes.png")
    print("- normalized_confusion_matrix_first_25_classes.png")
    print("- correct_predictions.png")
    print("- incorrect_predictions.png")

    if RUN_EXPERIMENTS:
        print("- hyperparameter_experiment_results.csv")
        print("- experiment_test_accuracy_comparison.png")

    print("\nFinal Main Model Metrics:")

    for metric_name, metric_value in final_metrics.items():
        print(f"{metric_name}: {metric_value}")


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()