import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

# Parameters for the dataset
n_samples = 100
n_features = 2
centers = 2

# Generate synthetic data
X, y = make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=centers,
    cluster_std=5,
    random_state=42,
)

# Combine features and labels
data = np.hstack((X, y.reshape(-1, 1)))

# Create a DataFrame
df = pd.DataFrame(
    data, columns=[f"feature_{i+1}" for i in range(n_features)] + ["label"]
)

# Save DataFrame to CSV
csv_file_path = "svm_data.csv"
df.to_csv(csv_file_path, index=False)

print(f"Data saved to {csv_file_path}")
