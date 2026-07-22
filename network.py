import numpy as np
import pickle


class DataScaler:
    """Scales data to have a mean of 0 and standard deviation of 1."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Add a small epsilon to prevent division by zero
        self.std[self.std == 0] = 1e-8
        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        Initializes the network.
        layer_sizes: list of integers representing nodes per layer (e.g., [5, 16, 16, 1]).
        """
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        # He Initialization for weights (optimized for ReLU)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2. / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        """Passes the input vector forward through the network."""
        self.activations = [X]
        self.z_values = []
        current_activation = X

        # Hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            current_activation = self._relu(z)
            self.activations.append(current_activation)

        # Output layer (Linear activation for regression)
        z_out = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_out)
        self.activations.append(z_out)

        return z_out

    def backward(self, y, learning_rate):
        """Calculates gradients and updates weights using backpropagation."""
        m = y.shape[0]

        # Derivative of MSE Loss w.r.t linear output
        dz = self.activations[-1] - y

        for i in reversed(range(len(self.weights))):
            dw = np.dot(self.activations[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m

            # Calculate error for the previous layer (if not the first layer)
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * self._relu_derivative(self.z_values[i - 1])

            # Update weights and biases
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

    def train(self, X, y, epochs, learning_rate=0.01):
        """Trains the network over a given number of epochs."""
        for epoch in range(epochs):
            self.forward(X)
            self.backward(y, learning_rate)

            if epoch % 1000 == 0:
                loss = np.mean((self.activations[-1] - y) ** 2)
                print(f"Epoch {epoch} | Mean Squared Error: {loss:.6f}")

    def predict(self, X):
        """Makes a prediction using trained weights."""
        return self.forward(X)

    def save_model(self, filename, input_scaler, output_scaler):
        """Saves the model weights, biases, and scalers to a file."""
        model_data = {
            'layer_sizes': self.layer_sizes,
            'weights': self.weights,
            'biases': self.biases,
            'input_scaler': input_scaler,
            'output_scaler': output_scaler
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model successfully saved to {filename}")

    def save_to_txt(self, weights_filename="weights.txt", biases_filename="biases.txt"):
        """Exports the weights and biases to readable text files."""
        # Export Weights
        with open(weights_filename, 'w') as f_w:
            for i, w in enumerate(self.weights):
                f_w.write(f"--- Weights: Layer {i} to Layer {i + 1} ---\n")
                # Save matrix with 8 decimal precision
                np.savetxt(f_w, w, fmt='%.8f')
                f_w.write("\n")

        # Export Biases
        with open(biases_filename, 'w') as f_b:
            for i, b in enumerate(self.biases):
                f_b.write(f"--- Biases: Layer {i + 1} ---\n")
                np.savetxt(f_b, b, fmt='%.8f')
                f_b.write("\n")

        print(f"Matrices successfully exported to {weights_filename} and {biases_filename}")

    @classmethod
    def load_model(cls, filename):
        """Loads a model and its scalers from a file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        nn = cls(data['layer_sizes'])
        nn.weights = data['weights']
        nn.biases = data['biases']
        return nn, data['input_scaler'], data['output_scaler']
