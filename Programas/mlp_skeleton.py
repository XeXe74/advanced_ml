import numpy as np


class MLP:
    def __init__(self, learning_rate=0.1, n_iters=5000):
        self.lr = learning_rate
        self.n_iters = n_iters

        # Initialize Weights and Biases randomly
        # Architecture: 2 Inputs -> 2 Hidden Neurons -> 1 Output Neuron
        self.w_hidden = np.random.randn(2, 2)
        self.b_hidden = np.zeros((1, 2))
        self.w_output = np.random.randn(2, 1)
        self.b_output = np.zeros((1, 1))

    def sigmoid(self, x):
        # Activation function (Slide 29)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivative for Backpropagation: f'(x) = f(x) * (1 - f(x))
        # (See Slide 31)
        return x * (1 - x)

    def forward(self, X):
        """
        Move data from Input -> Hidden -> Output
        (See Slide 33)
        """
        # TODO 1: Calculate Hidden Layer Output
        # Hint: Dot product of X and w_hidden, plus bias
        self.hidden_input = ______________________________
        self.hidden_output = self.sigmoid(self.hidden_input)

        # TODO 2: Calculate Final Output
        # Hint: Dot product of hidden_output and w_output, plus bias
        self.final_input = _______________________________
        self.final_output = self.sigmoid(self.final_input)

        return self.final_output

    def backward(self, X, y, output):
        """
        Backpropagate the error and update weights
        (See Slide 37)
        """
        # 1. Calculate Error at Output
        error = y - output

        # 2. Calculate Gradient for Output Layer
        # Gradient = Error * Derivative of Activation
        d_output = error * self.sigmoid_derivative(output)

        # 3. Calculate Error contribution of Hidden Layer
        # How much did the hidden layer contribute to the output error?
        # Hint: Dot product of d_output and w_output.T (Transpose)
        error_hidden = ___________________________________

        # 4. Calculate Gradient for Hidden Layer
        d_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)

        # TODO 3: Update Weights and Biases (Gradient Descent)
        # Hint: Weight += Learning Rate * Dot Product(Layer Input.T, Layer Gradient)

        # Update Output Weights (Input is self.hidden_output)
        self.w_output += _________________________________
        self.b_output += np.sum(d_output, axis=0, keepdims=True) * self.lr

        # Update Hidden Weights (Input is X)
        self.w_hidden += _________________________________
        self.b_hidden += np.sum(d_hidden, axis=0, keepdims=True) * self.lr

    def train(self, X, y):
        for _ in range(self.n_iters):
            output = self.forward(X)
            self.backward(X, y, output)


# --- Test on XOR Problem (Slide 20) ---
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR Logic

    mlp = MLP()
    mlp.train(X, y)

    print("Predictions:")
    print(mlp.forward(X))