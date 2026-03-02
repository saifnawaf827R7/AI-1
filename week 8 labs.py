import numpy as np

# --- Day 36: Vectors ---
# Tasks: Create vectors, compute addition/scaling, and norms [cite: 3, 5, 7]
feature = np.array([30.0, 50.0, 10.0]) [cite: 11, 15]
weights = np.array([0.05, 0.8, -0.1]) [cite: 14, 15]

addition = feature + weights
scaled_feature = feature * 0.5
feature_norm = np.linalg.norm(feature)

print("--- Day 36: Vectors ---")
print(f"Feature Shape: {feature.shape}") [cite: 18]
print(f"Addition: {addition}")
print(f"L2 Norm of Feature: {feature_norm:.4f}\n")

# --- Day 37: Dot Product ---
# Tasks: Compute dot product and cosine similarity [cite: 22, 23]
a = np.array([1.0, 2.0, 3.0]) [cite: 28, 29, 30]
b = np.array([0.5, 1.0, 1.5]) [cite: 31, 32, 33]

dot_product = np.dot(a, b)
cos_sim = dot_product / (np.linalg.norm(a) * np.linalg.norm(b))

print("--- Day 37: Dot Product ---")
print(f"Dot Product: {dot_product}")
print(f"Cosine Similarity: {cos_sim:.4f}\n")

# --- Day 38: Matrices ---
# Tasks: Compute X @ W and inspect shapes [cite: 38, 39, 40]
X_38 = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
]) [cite: 43, 44, 45]

W_38 = np.array([
    [0.1, -0.2],
    [0.4, 0.0],
    [-0.3, 0.5]
]) [cite: 47, 48, 50, 52]

Y_38 = X_38 @ W_38 [cite: 56]

print("--- Day 38: Matrices ---")
print(f"Matrix Multiplication Result:\n{Y_38}")
print(f"Shape of Y: {Y_38.shape}\n") [cite: 57]

# --- Day 39: Broadcasting ---
# Tasks: Normalize matrix by row so each row has unit norm [cite: 60, 61]
X_39 = np.array([
    [3.0, 4.0],
    [1.0, 2.0],
    [0.0, 5.0]
]) [cite: 65, 66, 67, 68]

# Compute L2 norm for each row (axis=1)
row_norms = np.linalg.norm(X_39, axis=1, keepdims=True) [cite: 70]
X_normalized = X_39 / row_norms [cite: 70]

print("--- Day 39: Broadcasting ---")
print(f"Row Norms:\n{row_norms}") [cite: 71]
print(f"Normalized Matrix:\n{X_normalized}")
# Verification: Norm of each row in X_normalized should be 1.0
print(f"Verification (New Norms): {np.linalg.norm(X_normalized, axis=1)}\n")

# --- Day 40: Matrix Operations ---
# Tasks: Linear transformation Y = X @ W + b [cite: 75, 76, 77]
X_40 = np.array([
    [1.0, 0.5],
    [2.0, -1.0],
    [0.0, 3.0]
]) [cite: 81, 82, 84, 86]

W_40 = np.array([
    [0.2, 0.1, 0.5],
    [0.7, 0.3, -0.2]
]) [cite: 89, 90, 92]

b_40 = np.array([0.1, 0.0, -0.3]) [cite: 95]

Y_40 = (X_40 @ W_40) + b_40 [cite: 96]

print("--- Day 40: Matrix Operations ---")
print(f"Linear Transformation Result (Y):\n{Y_40}")
print(f"Final Output Shape: {Y_40.shape}") [cite: 97]
