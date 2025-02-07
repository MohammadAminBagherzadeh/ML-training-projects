import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self, learning_rate=0.01, n_epochs=1000, tol=1e-3):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.tol = tol
        self.weights = None
        self.bias = None
        self.errors_ = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.n_epochs):
            errors = 0
            for idx in range(n_samples):
                linear_output = np.dot(X[idx], self.weights) + self.bias
                y_pred = self._unit_step(linear_output)
                if y[idx] != y_pred:
                    update = self.learning_rate * y[idx]
                    self.weights += update * X[idx]
                    self.bias += update
                    errors += 1
            self.errors_.append(errors)

            if errors <= self.tol:
                print(f"Converged after {epoch + 1} epochs")
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._unit_step(linear_output)

    def _unit_step(self, x):
        return np.where(x >= 0, 1, -1)


def load_and_preprocess_data():
    iris = load_iris()
    data = iris.data
    target = iris.target

    threshold = np.median(target)
    target_binary = np.where(target > threshold, 1, -1)

    selected_features = ['petal length (cm)', 'petal width (cm)']
    df = pd.DataFrame(data, columns=iris.feature_names)
    df['target'] = target_binary
    X_selected = df[selected_features].values
    y_selected = target_binary

    return train_test_split(X_selected, y_selected, test_size=0.2, stratify=y_selected)


def plot_learning_progress(perceptron):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(perceptron.errors_) + 1, 20), perceptron.errors_[::20], marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of Misclassifications')
    plt.title('Perceptron Learning Progress')
    plt.grid(True)
    plt.show()


def plot_decision_boundary(perceptron, X_train, X_test, y_test):
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = perceptron.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))

    if perceptron.weights[1] != 0:
        x_vals = np.array([x_min, x_max])
        y_vals = -(perceptron.weights[0] * x_vals + perceptron.bias) / perceptron.weights[1]
        plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
    else:
        print("Warning: Cannot plot decision boundary because weights[1] is zero!")

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X_test[y_test == -1, 0], X_test[y_test == -1, 1], color='red', marker='o', edgecolor='k', label='Setosa')
    plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='blue', marker='s', edgecolor='k', label='Versicolor')

    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.title('Perceptron Decision Boundary and Decision Regions (Iris Dataset)')
    plt.legend()
    plt.grid(True)
    plt.ylim(y_min, y_max)
    plt.show()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    perceptron = Perceptron(learning_rate=0.02, n_epochs=1000, tol=1e-3)
    perceptron.fit(X_train, y_train)

    plot_learning_progress(perceptron)
    plot_decision_boundary(perceptron, X_train, X_test, y_test)