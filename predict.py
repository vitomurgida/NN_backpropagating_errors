import numpy as np
from network import NeuralNetwork


def predict_beam_frequency(L, w, t, E, rho, model_path='outputs/beam_model.pkl'):
    # 1. Load the model and scalers
    nn, input_scaler, output_scaler = NeuralNetwork.load_model(model_path)

    # 2. Prepare the input vector
    input_vector = np.array([[L, w, t, E, rho]])

    # 3. Scale the input using the loaded input scaler
    scaled_input = input_scaler.transform(input_vector)

    # 4. Make the prediction (returns scaled output)
    scaled_prediction = nn.predict(scaled_input)

    # 5. Inverse transform to get actual physical frequency in Hz
    actual_frequency = output_scaler.inverse_transform(scaled_prediction)

    return actual_frequency[0][0]


if __name__ == "__main__":
    # Example Beam Parameters:
    # A 2.5m Steel beam, 0.1m wide, 0.02m thick.
    # E = 200 GPa, Density = 7850 kg/m^3

    test_L = 2.5
    test_w = 0.1
    test_t = 0.02
    test_E = 200e9
    test_rho = 7850

    predicted_freq = predict_beam_frequency(test_L, test_w, test_t, test_E, test_rho)

    # Calculate analytical for comparison
    from train import calculate_resonance_frequency

    true_freq = calculate_resonance_frequency(test_L, test_w, test_t, test_E, test_rho)

    print("\n--- Model Prediction Results ---")
    print(f"Predicted Frequency: {predicted_freq:.2f} Hz")
    print(f"Analytical Frequency: {true_freq:.2f} Hz")

    error_margin = abs(predicted_freq - true_freq) / true_freq * 100
    print(f"Error Margin: {error_margin:.3f}%")
