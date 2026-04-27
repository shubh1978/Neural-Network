import numpy as np
import matplotlib.pyplot as plt

print("===== XOR NEURAL NETWORK EXPERIMENT =====")

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative
def sigmoid_derivative(x):
    return x * (1 - x)

# Dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Seed
np.random.seed(42)

# Architecture
input_neurons = 2
hidden_neurons = 4
output_neurons = 1

# Weights
W1 = np.random.rand(input_neurons, hidden_neurons)
b1 = np.random.rand(1, hidden_neurons)

W2 = np.random.rand(hidden_neurons, output_neurons)
b2 = np.random.rand(1, output_neurons)

lr = 0.5
epochs = 5000

losses = []

# Training
for epoch in range(epochs):

    # Forward
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    # Loss
    loss = np.mean((y - final_output)**2)
    losses.append(loss)

    # Backprop
    error = y - final_output
    d_output = error * sigmoid_derivative(final_output)

    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update
    W2 += hidden_output.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr

    W1 += X.T.dot(d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss = {loss}")

# Final Output
print("\nFinal Predictions:")
print(final_output.round())

# Accuracy
pred = final_output.round()
accuracy = np.mean(pred == y)
print("Accuracy:", accuracy)

# Loss Graph
plt.figure(figsize=(6,4))
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()

# Decision Boundary
def plot_decision():
    plt.figure(figsize=(5,5))

    for i in range(len(X)):
        if y[i] == 0:
            plt.scatter(X[i][0], X[i][1], color='red')
        else:
            plt.scatter(X[i][0], X[i][1], color='green')

    x_vals = np.linspace(-0.5, 1.5, 100)
    y_vals = np.linspace(-0.5, 1.5, 100)

    for x1 in x_vals:
        for x2 in y_vals:
            inp = np.array([[x1, x2]])

            h = sigmoid(np.dot(inp, W1) + b1)
            out = sigmoid(np.dot(h, W2) + b2)

            if out > 0.5:
                plt.scatter(x1, x2, color='lightgreen', alpha=0.1)
            else:
                plt.scatter(x1, x2, color='lightcoral', alpha=0.1)

    plt.title("XOR Decision Boundary")
    plt.show()

plot_decision()

# Test
test = np.array([[0,1],[1,1],[1,0]])
print("\nTest Predictions:")
print(np.round(sigmoid(np.dot(sigmoid(np.dot(test, W1)+b1), W2)+b2)))

print("\n===== EXPERIMENT COMPLETED SUCCESSFULLY =====")