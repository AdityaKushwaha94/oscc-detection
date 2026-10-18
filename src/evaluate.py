"""Held-out test metrics for the saved checkpoint: the numbers that count."""
import numpy as np
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)

from data import load_split


def main() -> None:
    model = tf.keras.models.load_model("models/best.keras")
    test_ds = load_split("test")

    y_true, y_prob = [], []
    for images, labels in test_ds:
        y_true.extend(labels.numpy().ravel().astype(int))
        y_prob.extend(model.predict(images, verbose=0).ravel())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    print(classification_report(y_true, y_pred, target_names=["normal", "oscc"]))
    print("confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
    print(f"accuracy: {(y_pred == y_true).mean():.4f}")


if __name__ == "__main__":
    main()
