"""
Two-phase VGG16 transfer learning for OSCC vs normal histopathology.

Phase 1 trains a small head on frozen ImageNet features. Phase 2 unfreezes the
last convolutional block and fine-tunes at a low learning rate. Class weights
correct for the dataset's imbalance. Best checkpoint lands in models/best.keras.
"""
import collections
import pathlib

import numpy as np
import tensorflow as tf

from data import load_split, augmentation, IMG_SIZE

HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 10


def build_model() -> tf.keras.Model:
    base = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = augmentation()(inputs)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)


def class_weights(ds: tf.data.Dataset) -> dict:
    """Weight each class inversely to its frequency in the training split."""
    counts = collections.Counter()
    for _, labels in ds:
        values, freq = np.unique(labels.numpy().astype(int), return_counts=True)
        counts.update(dict(zip(values.tolist(), freq.tolist())))
    total = sum(counts.values())
    return {cls: total / (len(counts) * n) for cls, n in counts.items()}


def main() -> None:
    train_ds = load_split("train", shuffle=True)
    val_ds = load_split("val")
    weights = class_weights(train_ds)
    print("class weights:", weights)

    model = build_model()
    pathlib.Path("models").mkdir(exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "models/best.keras", monitor="val_accuracy", save_best_only=True)
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True)

    # Phase 1: head only.
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS,
              class_weight=weights, callbacks=[checkpoint, early])

    # Phase 2: unfreeze the last conv block (block5) and fine-tune gently.
    base = next(l for l in model.layers if l.name == "vgg16")
    base.trainable = True
    for layer in base.layers:
        layer.trainable = layer.name.startswith("block5")

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
              class_weight=weights, callbacks=[checkpoint, early])

    print("best checkpoint: models/best.keras")


if __name__ == "__main__":
    main()
