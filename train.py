import numpy as np
from network import NeuralNetwork, DataScaler

def calculate_resonance_frequency(L, w, t, E, rho):
    """Calculates the analytical first resonance frequency (Euler-Bernoulli Free-Free)."""
    beta_1 = 4.73004
    A = w * t
    I = (w * (t ** 3)) / 12
    frequency = ((beta_1 ** 2) / (2 * np.pi * (L ** 2))) * np.sqrt((E * I) / (rho * A))
    return frequency

if __name__ == "__main__":
    # 1. Generate Synthetic Training Data
    print("Generating dataset...")
    num_samples = 1000
    np.random.seed(42)

    # Inputs: Length (m), Width (m), Thickness (m), Young's Modulus (Pa), Density (kg/m^3)
    L_data = np.random.uniform(1.0, 5.0, num_samples)
    w_data = np.random.uniform(0.05, 0.2, num_samples)
    t_data = np.random.uniform(0.01, 0.05, num_samples)
    E_data = np.random.uniform(69e9, 300e9, num_samples) # Aluminum to Steel range
    rho_data = np.random.uniform(2700, 10000, num_samples)

    X_train = np.column_stack((L_data, w_data, t_data, E_data, rho_data))
    y_train = np.zeros((num_samples, 1))

    for i in range(num_samples):
        y_train[i, 0] = calculate_resonance_frequency(L_data[i], w_data[i], t_data[i], E_data[i], rho_data[i])

    # 2. Normalize the Data
    print("Scaling data...")
    input_scaler = DataScaler()
    output_scaler = DataScaler()

    X_train_scaled = input_scaler.fit_transform(X_train)
    y_train_scaled = output_scaler.fit_transform(y_train)

    # 3. Initialize and Train the Neural Network
    layer_architecture = [5, 10, 10, 1]
    nn = NeuralNetwork(layer_architecture)

    print("Training the Neural Network...")
    nn.train(X_train_scaled, y_train_scaled, epochs=5000, learning_rate=0.05)

    # 4. Save the trained model to disk
    nn.save_model('outputs/beam_model.pkl', input_scaler, output_scaler)

    # 5. Export the weights and biases to text files
    nn.save_to_txt('outputs/weights.txt', 'outputs/biases.txt')

