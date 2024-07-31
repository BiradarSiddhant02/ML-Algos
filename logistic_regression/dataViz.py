import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

df = pd.read_csv("data.csv")

plt.figure(figsize=(8, 6))
plt.hist(df['Purchased'], bins=[-0.5, 0.5, 1.5], edgecolor='black', alpha=0.7)
plt.xticks([0, 1])
plt.xlabel('Purchased')
plt.ylabel('Frequency')
plt.title('Histogram of Purchased Column')
plt.grid(True)
plt.show()
