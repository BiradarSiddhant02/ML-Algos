import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters for the dataset
n_samples = 100
n_features = 2  # Use only 2 features for simplicity
centers = 2
noise = 100


# Generate synthetic circular data
def generate_circular_data(n_samples, centers, noise):
    np.random.seed(42)
    angles = np.linspace(0, 2 * np.pi, n_samples // centers, endpoint=False)
    radii = np.random.uniform(0.5, 1.0, size=(n_samples // centers, centers))
    X = np.concatenate(
        [
            np.vstack([r * np.cos(angles), r * np.sin(angles)]).T
            + np.random.normal(scale=noise, size=(n_samples // centers, 2))
            for r in radii.T
        ],
        axis=0,
    )
    y = np.concatenate(
        [np.full(n_samples // centers, i) for i in range(centers)], axis=0
    )
    return X, y


# Generate the data
X, y = generate_circular_data(n_samples, centers, noise)

# Combine features and labels
data = np.hstack((X, y.reshape(-1, 1)))

# Create a DataFrame
df = pd.DataFrame(
    data, columns=[f"feature_{i+1}" for i in range(n_features)] + ["label"]
)

# Save DataFrame to CSV
csv_file_path = "nn_data.csv"
df.to_csv(csv_file_path, index=False)

print(f"Data saved to {csv_file_path}")

# Plot the data
plt.figure(figsize=(8, 6))
for i in range(centers):
    plt.scatter(
        X[y == i, 0], X[y == i, 1], label=f"Class {i}", alpha=0.6, edgecolors="w"
    )
plt.title("Circular Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
