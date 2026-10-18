"""Dataset loaders and augmentation. Binary layout: data/<split>/{normal,oscc}/*.jpg"""
import tensorflow as tf

IMG_SIZE = (224, 224)   # VGG16's native input size
BATCH = 32
SEED = 42


def load_split(split: str, shuffle: bool = False) -> tf.data.Dataset:
    ds = tf.keras.utils.image_dataset_from_directory(
        f"data/{split}",
        labels="inferred",
        label_mode="binary",       # 0 = normal, 1 = oscc (alphabetical)
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=shuffle,
        seed=SEED,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def augmentation() -> tf.keras.Sequential:
    """Light augmentation only: histology slides lose meaning under heavy distortion."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="augmentation")
