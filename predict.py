# predict.py
import tensorflow as tf
import numpy as np
import os

CLASS_NAMES = ['Moderate', 'Short', 'Tall']
MODEL_PATH = 'best_model.keras'
IMG_SIZE = 224

def predict_single_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target frame sequence missing: {image_path}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Target workspace model architecture file best_model.keras missing.")
        
    model = tf.keras.models.load_model(MODEL_PATH)
    
    img_raw = tf.io.read_file(image_path)
    img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    img_batch = tf.expand_dims(img, axis=0)

    probs = model.predict(img_batch, verbose=0)[0]
    best_idx = np.argmax(probs)
    return CLASS_NAMES[best_idx], float(probs[best_idx] * 100)
