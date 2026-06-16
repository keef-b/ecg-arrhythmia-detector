import numpy as np

y = np.load("../data/y.npy")

unique, counts = np.unique(y, return_counts=True)

print("Class Distribution")
print("-" * 30)

for label, count in zip(unique, counts):
    print(f"{label}: {count}")
