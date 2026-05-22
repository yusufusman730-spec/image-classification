# evaluate.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_data

CLASS_NAMES = ['Moderate', 'Short', 'Tall']
MODEL_PATH = 'best_model.keras'

def evaluate():
    print("=" * 45)
    print("  HEIGHT CLASSIFIER — METRIC EVALUATION")
    print("=" * 45)

    if not os.path.exists(MODEL_PATH):
        print("Precompiled weights file missing. Run train.py first.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    _, val_ds = load_data(data_dir='dataset')

    all_preds, all_labels = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        all_preds.extend(np.argmax(preds, axis=1))
        all_labels.extend(np.argmax(labels.numpy(), axis=1))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\nClassification Analytics Performance Output:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    ticks = range(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES)
    plt.yticks(ticks, CLASS_NAMES)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)

if __name__ == '__main__':
    evaluate()
