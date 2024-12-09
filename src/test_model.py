import torch
import numpy as np
import matplotlib.pyplot as plt
from forest_ca import ForestFire, generate_dataset
from neural_net import BurnPredictor


def load_model(model_path, input_size):
    # Load the trained model
    model = BurnPredictor(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def test_predictions(num_tests=100):
    """Generate test cases and get predictions"""
    # Load the trained model
    input_size = 400  # 20x20 grid
    model = load_model("forest_fire_model.pth", input_size)

    # Generate test cases
    print("Generating test cases...")
    states, actuals = generate_dataset(num_samples=num_tests)
    states_tensor = torch.FloatTensor(np.array([state.flatten() for state in states]))

    # Get predictions
    with torch.no_grad():
        predictions = model(states_tensor).numpy()
    # Plot results
    plt.figure(figsize=(15, 5))

    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(actuals, predictions, alpha=0.6)
    plt.plot([0, max(actuals)], [0, max(actuals)], "r--", label="Perfect Prediction")
    plt.xlabel("Actual Burn Fraction")
    plt.ylabel("Predicted Burn Fraction")
    plt.title("Test Predictions vs Actuals")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Error histogram
    plt.subplot(1, 2, 2)
    errors = predictions - actuals
    plt.hist(errors, bins=20, alpha=0.7)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.title("Error Distribution")
    plt.grid(True, alpha=0.3)

    mae = np.mean(np.abs(errors))
    plt.text(
        0.05,
        0.95,
        f"MAE: {mae:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()

    # Print stats
    print("\nDetailed Statistics:")
    print(f"MAE: {mae:.4f}")

    # Error analysis by range
    ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    print("\nMAE by range:")
    for low, high in ranges:
        mask = (actuals >= low) & (actuals < high)
        if np.sum(mask) > 0:
            range_mae = np.mean(np.abs(predictions[mask] - actuals[mask]))
            print(f"{low:.1f}-{high:.1f}: {range_mae:.4f} (n={np.sum(mask)})")


if __name__ == "__main__":
    test_predictions()
