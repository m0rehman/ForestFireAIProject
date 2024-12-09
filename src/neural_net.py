import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from forest_ca import generate_dataset


class FireDataset(Dataset):
    def __init__(self, grid_states, burn_fractions):
        flat_grids = np.array([grid.flatten() for grid in grid_states])
        self.grid_states = torch.FloatTensor(flat_grids)
        self.burn_fractions = torch.FloatTensor(burn_fractions)

    def __len__(self):
        return len(self.grid_states)

    def __getitem__(self, idx):
        return self.grid_states[idx], self.burn_fractions[idx]


class BurnPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x).squeeze()


def weighted_mse_loss(pred, target):
    weights = 1 + 5 * target
    return torch.mean(weights * (pred - target) ** 2)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0


def prepare_data(num_samples=2000, train_ratio=0.8, batch_size=32):
    print("Generating forest fire data...")
    grid_states, burn_fractions = generate_dataset(num_samples=num_samples)

    print("\nDataset statistics:")
    percentiles = np.percentile(burn_fractions, [25, 50, 75])
    print(f"25th percentile: {percentiles[0]:.3f}")
    print(f"Median: {percentiles[1]:.3f}")
    print(f"75th percentile: {percentiles[2]:.3f}")

    split_idx = int(len(grid_states) * train_ratio)
    train_states = grid_states[:split_idx]
    train_fractions = burn_fractions[:split_idx]
    test_states = grid_states[split_idx:]
    test_fractions = burn_fractions[split_idx:]

    train_dataset = FireDataset(train_states, train_fractions)
    test_dataset = FireDataset(test_states, test_fractions)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print(f"\nDataset sizes:")
    print(f"Training samples: {len(train_states)}")
    print(f"Testing samples: {len(test_states)}")

    return train_loader, test_loader, train_dataset.grid_states.shape[1]


def train_model(train_loader, test_loader, input_size, epochs=100):
    model = BurnPredictor(input_size)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(patience=10)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for grids, fractions in train_loader:
            optimizer.zero_grad()
            predictions = model(grids)
            loss = weighted_mse_loss(predictions, fractions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_actual = []

        with torch.no_grad():
            for grids, fractions in test_loader:
                predictions = model(grids)
                val_loss += weighted_mse_loss(predictions, fractions).item()
                val_preds.extend(predictions.numpy())
                val_actual.extend(fractions.numpy())

        val_loss = val_loss / len(test_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            mae = np.mean(np.abs(np.array(val_preds) - np.array(val_actual)))
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Training loss: {train_loss:.4f}")
            print(f"Validation loss: {val_loss:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}\n")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            model.load_state_dict(early_stopping.best_model)
            # Save the best model
            torch.save(model.state_dict(), "forest_fire_model.pth")
            break

    return model, train_losses, val_losses, val_preds, val_actual


def plot_results(train_losses, val_losses, predictions, actuals):
    plt.figure(figsize=(15, 5))

    # Training history
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Predictions vs Actuals
    plt.subplot(1, 2, 2)
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([0, max(actuals)], [0, max(actuals)], "r--")
    plt.xlabel("Actual Burn Fraction")
    plt.ylabel("Predicted Burn Fraction")
    plt.title("Predictions vs Actuals")
    plt.grid(True, alpha=0.3)

    mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
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


if __name__ == "__main__":
    # Prepare data
    train_loader, test_loader, input_dim = prepare_data(num_samples=2000)

    # Train model
    print("\nTraining model...")
    model, train_losses, val_losses, predictions, actuals = train_model(
        train_loader, test_loader, input_dim, epochs=100
    )

    # Plot results
    plot_results(train_losses, val_losses, predictions, actuals)
