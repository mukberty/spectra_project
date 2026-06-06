import numpy as np

train_idx = np.load("data/splits/train_indices.npy")
test_idx = np.load("data/splits/test_indices.npy")

print(len(train_idx))
print(len(test_idx))
print(train_idx[:5])
print(test_idx[:5])
