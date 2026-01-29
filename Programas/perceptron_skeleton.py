import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def activation_function(self, x):
        """
        Apply the step activation function.
        Returns 1 if x >= 0, else 0.
        (See Slide 17 for reference)
        """
        # TODO: Implement the step function
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        """
        Calculate the output prediction.
        Formula: y = f(dot_product(X, W) + b)
        (See Slide 15)
        """
        # TODO: Calculate the linear output (Z = W*X + b)
        linear_output = np.dot(X, self.weights) + self.bias

        # TODO: Apply the activation function
        y_predicted = self.activation_function(linear_output)

        return y_predicted

    def train(self, X, y):
        """
        Train the perceptron using the update rules.
        (See Slide 16 for the algorithm)
        """
        n_samples, n_features = X.shape

        # 1. Initialize parameters (Slide 16: "Initialize to zeros")
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 2. Training Loop
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # TODO: Get the prediction for the current sample
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                # TODO: Calculate the error (Difference between actual and predicted)
                error = y[idx] - y_predicted

                # TODO: Update weights and bias (Slide 16: w = w + error * x)
                self.weights += error * x_i
                self.bias += error

# --- Test Your Code ---
if __name__ == "__main__":
    # OR Gate Data (from Slide 18)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])

    p = Perceptron(learning_rate=0.1, n_iters=10)
    p.train(X, y)

    print("Predicted:", p.predict(X))
    print("Expected: [0 1 1 1]")