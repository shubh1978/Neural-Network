# -------------------------------
# Experiment 1: NumPy Operations
# -------------------------------

import numpy as np

print("===== NUMPY LAB EXPERIMENT =====")

# -------------------------------
# 1. Array Creation
# -------------------------------

print("\n--- Array Creation ---")

arr1 = np.array([10, 20, 30, 40, 50])
arr2 = np.array([5, 4, 3, 2, 1])

print("Array 1:", arr1)
print("Array 2:", arr2)

# Creating different types of arrays
zeros_arr = np.zeros(5)
ones_arr = np.ones(5)
range_arr = np.arange(1, 11)

print("Zeros Array:", zeros_arr)
print("Ones Array:", ones_arr)
print("Range Array:", range_arr)

# -------------------------------
# 2. Basic Operations
# -------------------------------

print("\n--- Basic Operations ---")

add = arr1 + arr2
sub = arr1 - arr2
mul = arr1 * arr2
div = arr1 / arr2

print("Addition:", add)
print("Subtraction:", sub)
print("Multiplication:", mul)
print("Division:", div)

# -------------------------------
# 3. Statistical Functions
# -------------------------------

print("\n--- Statistical Operations ---")

print("Mean:", np.mean(arr1))
print("Median:", np.median(arr1))
print("Standard Deviation:", np.std(arr1))
print("Variance:", np.var(arr1))
print("Minimum:", np.min(arr1))
print("Maximum:", np.max(arr1))

# -------------------------------
# 4. 2D Arrays and Matrix Ops
# -------------------------------

print("\n--- Matrix Operations ---")

matrix1 = np.array([[1,2,3],[4,5,6]])
matrix2 = np.array([[7,8,9],[1,2,3]])

print("Matrix 1:\n", matrix1)
print("Matrix 2:\n", matrix2)

print("Matrix Addition:\n", matrix1 + matrix2)
print("Matrix Subtraction:\n", matrix1 - matrix2)

# Transpose
print("Transpose of Matrix1:\n", matrix1.T)

# Dot product
dot_product = np.dot(matrix1, matrix2.T)
print("Dot Product:\n", dot_product)

# -------------------------------
# 5. Indexing and Slicing
# -------------------------------

print("\n--- Indexing and Slicing ---")

arr = np.array([10,20,30,40,50,60])

print("Original Array:", arr)
print("First Element:", arr[0])
print("Last Element:", arr[-1])
print("Slice (1 to 4):", arr[1:5])

# 2D slicing
mat = np.array([[1,2,3],[4,5,6],[7,8,9]])

print("Matrix:\n", mat)
print("First Row:", mat[0])
print("Second Column:", mat[:,1])
print("Submatrix:\n", mat[0:2, 1:3])

# -------------------------------
# 6. Reshaping and Flattening
# -------------------------------

print("\n--- Reshaping ---")

arr = np.arange(1,13)

reshaped = arr.reshape(3,4)
print("Reshaped (3x4):\n", reshaped)

flattened = reshaped.flatten()
print("Flattened Array:", flattened)

# -------------------------------
# 7. Random Functions
# -------------------------------

print("\n--- Random Numbers ---")

rand_arr = np.random.rand(5)
rand_int = np.random.randint(1, 100, 5)

print("Random Float Array:", rand_arr)
print("Random Integer Array:", rand_int)

# -------------------------------
# 8. Broadcasting
# -------------------------------

print("\n--- Broadcasting ---")

arr = np.array([1,2,3])
scalar = 5

print("Array:", arr)
print("Scalar:", scalar)
print("Broadcast Addition:", arr + scalar)

# -------------------------------
# 9. Comparison Operations
# -------------------------------

print("\n--- Comparison ---")

arr = np.array([10,20,30,40,50])
condition = arr > 25

print("Condition (arr > 25):", condition)
print("Filtered Values:", arr[arr > 25])

# -------------------------------
# 10. Conclusion
# -------------------------------

print("\n===== EXPERIMENT COMPLETED SUCCESSFULLY =====")