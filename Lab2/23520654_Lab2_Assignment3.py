import numpy as np
import matplotlib.pyplot as plt
import struct
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay)

def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    return images

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

X_train = load_images("/content/drive/MyDrive/DS102/Lab2/data/train-images.idx3-ubyte")
y_train = load_labels("/content/drive/MyDrive/DS102/Lab2/data/train-labels.idx1-ubyte")

X_test = load_images("/content/drive/MyDrive/DS102/Lab2/data/t10k-images.idx3-ubyte")
y_test = load_labels("/content/drive/MyDrive/DS102/Lab2/data/t10k-labels.idx1-ubyte")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

print("Classes:", np.unique(y_train))

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

X_train = X_train.astype(np.float64) / 255.0
X_test = X_test.astype(np.float64) / 255.0

train_mask = (y_train == 0) | (y_train == 1)
test_mask = (y_test == 0) | (y_test == 1)

X_train_binary = X_train[train_mask]
y_train_binary = y_train[train_mask]

X_test_binary = X_test[test_mask]
y_test_binary = y_test[test_mask]

print("Binary classification:")
print(X_train_binary.shape)
print(X_test_binary.shape)

print("\nMulticlass classification:")
print(X_train.shape)
print(X_test.shape)

logistic_model = LogisticRegression(solver="lbfgs", max_iter=500, random_state=42)
logistic_model.fit(X_train_binary, y_train_binary)
y_pred_binary = logistic_model.predict(X_test_binary)

print("Logistic Regression Evaluation\n")
print("Accuracy :", accuracy_score(y_test_binary, y_pred_binary))
print("Precision:", precision_score(y_test_binary, y_pred_binary))
print("Recall   :", recall_score(y_test_binary, y_pred_binary))
print("F1-score :", f1_score(y_test_binary, y_pred_binary))
print("\nClassification Report:")
print(classification_report(y_test_binary, y_pred_binary, digits=4, zero_division=0))

softmax_model = LogisticRegression(solver="lbfgs", max_iter=500, random_state=42)
softmax_model.fit(X_train, y_train)

y_pred_softmax = softmax_model.predict(X_test)
print("Softmax Regression Evaluation\n")
print("Accuracy:", accuracy_score(y_test, y_pred_softmax))
print("Macro Precision:", precision_score(y_test, y_pred_softmax, average="macro", zero_division=0))
print("Macro Recall:", recall_score(y_test, y_pred_softmax, average="macro", zero_division=0))
print("Macro F1-score:", f1_score(y_test, y_pred_softmax, average="macro", zero_division=0))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_softmax, digits=4, zero_division=0))


fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ConfusionMatrixDisplay.from_predictions(y_test_binary, y_pred_binary, ax=axes[0], cmap="Blues", values_format="d")
axes[0].set_title("Logistic Regression (0, 1)")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_softmax, ax=axes[1], cmap="Blues", values_format="d")
axes[1].set_title("Softmax Regression (0-9)")

plt.tight_layout()
plt.show()