"""Train the diabetic retinopathy classifier (MobileNetV2 transfer learning).

Usage:
    python src/train.py --data-dir data/ratinopathy
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (160, 160)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/ratinopathy")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--model-out", default="models/retinopathy_model.keras")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    images_dir = os.path.join(args.data_dir, "images")
    df = pd.read_csv(os.path.join(args.data_dir, "data_all.csv"))
    df["cat_str"] = df["cat"].astype(str)

    # Stratified split (the CSV is sorted by class, so a plain slice would be invalid)
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["cat"], random_state=42
    )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255, horizontal_flip=True, rotation_range=15,
        zoom_range=0.1, brightness_range=[0.9, 1.1],
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, directory=images_dir, x_col="file", y_col="cat_str",
        target_size=IMG_SIZE, batch_size=args.batch_size,
        class_mode="categorical", shuffle=True, seed=42,
    )
    val_gen = val_datagen.flow_from_dataframe(
        val_df, directory=images_dir, x_col="file", y_col="cat_str",
        target_size=IMG_SIZE, batch_size=args.batch_size,
        class_mode="categorical", shuffle=False,
    )

    # Class weights from the training split only
    classes = np.array(sorted(train_df["cat"].unique()))
    weights = compute_class_weight("balanced", classes=classes, y=train_df["cat"])
    class_weights = {train_gen.class_indices[str(c)]: w for c, w in zip(classes, weights)}

    # Frozen MobileNetV2 backbone + small classification head
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(64, activation="relu")(x)
    out = Dense(len(classes), activation="softmax")(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(
        train_gen, validation_data=val_gen, epochs=args.epochs,
        class_weight=class_weights,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    )

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)

    # Training curves
    plt.figure(figsize=(12, 4))
    for i, (k, title) in enumerate([("accuracy", "Accuracy"), ("loss", "Loss")], 1):
        plt.subplot(1, 2, i)
        plt.plot(history.history[k], label=f"Train {title}")
        plt.plot(history.history[f"val_{k}"], label=f"Val {title}")
        plt.title(f"{title} over Epochs"); plt.xlabel("Epoch"); plt.ylabel(title); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "training_curves.png"), dpi=150)
    plt.close()

    # Evaluation on the validation split
    preds = np.argmax(model.predict(val_gen), axis=1)
    true = val_gen.classes
    labels = [f"Category {c}" for c in classes]

    report = classification_report(true, preds, target_names=labels)
    print(report)
    with open(os.path.join(args.results_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(true, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion Matrix - Diabetic Retinopathy Category")
    plt.savefig(os.path.join(args.results_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    model.save(args.model_out)
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
