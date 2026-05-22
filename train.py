# train.py
import tensorflow as tf
import matplotlib.pyplot as plt
from model import build_model
from data_loader import load_data

EPOCHS = 15
SAVE_PATH = 'best_model.keras'

def train():
    print("🧪🧪" * 15)
    print("  HEIGHT CLASSIFIER — TRAINING PIPELINE")
    print("🧪🧪" * 15)
    
    print("\nLoading datasets...")
    train_ds, val_ds = load_data(data_dir='dataset')

    print("\nBuilding core network model...")
    model, base_model = build_model(num_classes=3)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    print("\nExecuting Pipeline — Phase 1 (Frozen Base Data Layer)...")
    history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks, verbose=1)

    if base_model is not model:
        print("\nExecuting Pipeline — Phase 2 Fine-Tuning Stage...")
        base_model.trainable = True
        for layer in base_model.layers[:-12]:
            layer.trainable = False
            
        model.compile(
            optimizer=tf.keras.optimizers.Nadam(learning_rate=0.002 * 0.1),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        history2 = model.fit(train_ds, epochs=8, validation_data=val_ds, callbacks=callbacks, verbose=1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    try:
        plt.plot(history2.history['accuracy'], label='Train')
        plt.plot(history2.history['val_accuracy'], label='Validation')
    except:
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('EfficientNetB0_B — Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    try:
        plt.plot(history2.history['loss'], label='Train')
        plt.plot(history2.history['val_loss'], label='Validation')
    except:
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
    plt.title('EfficientNetB0_B — Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)

    print("\nTraining pipeline executed successfully. Model saved to:", SAVE_PATH)

if __name__ == '__main__':
    train()
