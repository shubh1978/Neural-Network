import numpy as np
import matplotlib.pyplot as plt

print("===== ACTIVATION FUNCTION EXPERIMENT =====")

# Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)

# Input range
x = np.linspace(-10, 10, 200)

# Compute outputs
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)
y_leaky = leaky_relu(x)

# Compute derivatives
dy_sigmoid = sigmoid_derivative(x)
dy_relu = relu_derivative(x)
dy_tanh = tanh_derivative(x)

# Plot functions
plt.figure(figsize=(10,6))

plt.plot(x, y_sigmoid, label="Sigmoid")
plt.plot(x, y_relu, label="ReLU")
plt.plot(x, y_tanh, label="Tanh")
plt.plot(x, y_leaky, label="Leaky ReLU")

plt.title("Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid()
plt.show()

# Plot derivatives
plt.figure(figsize=(10,6))

plt.plot(x, dy_sigmoid, label="Sigmoid Derivative")
plt.plot(x, dy_relu, label="ReLU Derivative")
plt.plot(x, dy_tanh, label="Tanh Derivative")

plt.title("Derivatives of Activation Functions")
plt.xlabel("Input")
plt.ylabel("Gradient")
plt.legend()
plt.grid()
plt.show()

# Comparison values
test_values = np.array([-2, -1, 0, 1, 2])

print("\nTest Values:", test_values)
print("Sigmoid:", sigmoid(test_values))
print("ReLU:", relu(test_values))
print("Tanh:", tanh(test_values))
print("Leaky ReLU:", leaky_relu(test_values))

# Small neural example
weights = np.array([0.5, -0.3])
bias = 0.1

sample = np.array([1, 2])

z = np.dot(sample, weights) + bias

print("\nWeighted Input:", z)
print("Sigmoid Output:", sigmoid(z))
print("ReLU Output:", relu(z))
print("Tanh Output:", tanh(z))

print("\n===== EXPERIMENT COMPLETED SUCCESSFULLY =====")