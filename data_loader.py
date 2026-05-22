# data_loader.py
import tensorflow as tf
import os

IMG_SIZE = 224
BATCH_SIZE = 32
CLASS_NAMES = ['moderate', 'short', 'tall']

def load_data(data_dir='dataset'):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir, shuffle=True, seed=42,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode='categorical'
    )
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir, shuffle=False, seed=42,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode='categorical'
    )

    rescale = tf.keras.layers.Rescaling(1.0 / 255)
    train_dataset = train_dataset.map(lambda x, y: (rescale(x), y))
    val_dataset = val_dataset.map(lambda x, y: (rescale(x), y))

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
    ])
    train_dataset = train_dataset.map(lambda x, y: (augmentation(x, training=True), y))

    AUTOTUNE = tf.data.AUTOTUNE
    return train_dataset.prefetch(buffer_size=AUTOTUNE), val_dataset.prefetch(buffer_size=AUTOTUNE)
