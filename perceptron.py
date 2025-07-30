import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.weight_history = []  # To track weights over time

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

            # Store current weights and bias
            self.weight_history.append((self.weights.copy(), self.bias))

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.activation_function(x) for x in linear_output])

    def plot_weight_updates(self):
        weights = np.array([w for w, _ in self.weight_history])
        bias = [b for _, b in self.weight_history]

        plt.plot(weights[:, 0], label='Weight 1 (IQ)')
        plt.plot(weights[:, 1], label='Weight 2 (Study Hours)')
        plt.plot(bias, label='Bias')
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.title("Perceptron Weights & Bias Over Time")
        plt.legend()
        plt.grid()
        plt.show()

import numpy as np

if __name__ == "__main__":
    print("Starting Perceptron training...")

    # Sample dataset: [IQ, Study Hours]
    X = np.array([[95, 3], [110, 4], [100, 5], [120, 6]])
    y = np.array([0, 1, 1, 1])  # 0 = Fail, 1 = Pass

    # Create and train the model
    perceptron = Perceptron(learning_rate=0.1, n_iters=20)
    perceptron.fit(X, y)

    print("Model trained successfully.")

    # Predict on new data
    new_data = np.array([[105, 4]])
    prediction = perceptron.predict(new_data)
    print(f"Prediction for [105, 4]: {prediction[0]}")

    # Plot weight updates
    perceptron.plot_weight_updates()

