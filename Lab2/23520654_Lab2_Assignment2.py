import numpy as np
import matplotlib.pyplot as plt
import struct

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

X_train = load_images("data/train-images.idx3-ubyte")
y_train = load_labels("data/train-labels.idx1-ubyte")

X_test = load_images("data/t10k-images.idx3-ubyte")
y_test = load_labels("data/t10k-labels.idx1-ubyte")

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

print("Train classes:", np.unique(y_train))
print("Test classes :", np.unique(y_test))

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

X_train = X_train.astype(np.float64) / 255.0
X_test = X_test.astype(np.float64) / 255.0

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("Min:", X_train.min())
print("Max:", X_train.max())

def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]
Y_train = one_hot_encode(y_train, num_classes=10)

print("Original labels:")
print(y_train[:5])
print("\nOne-hot labels:")
print(Y_train[:5])
print("\nShape:", Y_train.shape)

class SoftmaxRegression:

    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = None
        self.loss_history = []

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        probabilities = (exp_z / np.sum(exp_z, axis=1, keepdims=True))
        return probabilities

    def compute_loss(self, Y, probabilities):
        eps = 1e-15
        probabilities = np.clip(probabilities, eps, 1 - eps)
        loss = -np.mean(np.sum(Y * np.log(probabilities), axis=1))
        return loss

    def fit(self, X, Y):
        self.loss_history = []
        n_samples, n_features = X.shape
        n_classes = Y.shape[1]
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)
        for epoch in range(self.epochs):
            scores = X @ self.W + self.b
            probabilities = self.softmax(scores)
            loss = self.compute_loss(Y, probabilities)
            self.loss_history.append(loss)

            error = probabilities - Y
            dW = (X.T @ error) / n_samples
            db = np.mean(error, axis=0)

            self.W -= (self.learning_rate * dW)
            self.b -= (self.learning_rate * db)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:3d} | "
                    f"Loss = {loss:.6f}"
                )

    def predict_proba(self, X):
        scores = X @ self.W + self.b
        return self.softmax(scores)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

model = SoftmaxRegression(learning_rate=0.1, epochs=200)
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
print("Actual:")
print(y_test[:20])
print("\nPredicted:")
print(y_pred[:20])

def evaluate_multiclass(y_true, y_pred, num_classes=10):
    accuracy = np.mean(y_true == y_pred)
    precisions = []
    recalls = []
    f1_scores = []

    print(
        f"{'Class':<8}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1-score':<12}"
    )

    print("-" * 44)

    for c in range(num_classes):
        TP = np.sum((y_true == c) & (y_pred == c))
        FP = np.sum((y_true != c) & (y_pred == c))
        FN = np.sum((y_true == c) & (y_pred != c))
        precision = (TP / (TP + FP + 1e-15))
        recall = (TP / (TP + FN + 1e-15))
        f1 = (2 * precision * recall / (precision + recall + 1e-15))
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        print(
            f"{c:<8}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
        )

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)
    print("-" * 44)
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Macro Precision : {macro_precision:.4f}")
    print(f"Macro Recall    : {macro_recall:.4f}")
    print(f"Macro F1-score  : {macro_f1:.4f}")

y_pred = model.predict(X_test)

print("Actual    :", y_test[:20])
print("Predicted :", y_pred[:20])
print("\nEvaluation:")
evaluate_multiclass(y_test, y_pred)

plt.figure(figsize=(8, 5))
plt.plot(model.loss_history)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Softmax Regression Training Loss")
plt.grid()
plt.show()