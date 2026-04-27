import numpy as np
import matplotlib.pyplot as plt

print("===== MLP EXPERIMENT =====")

# Activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Dataset (simple classification)
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Architecture
input_layer = 2
hidden_layer1 = 5
hidden_layer2 = 4
output_layer = 1

np.random.seed(1)

# Weights initialization
W1 = np.random.randn(input_layer, hidden_layer1)
b1 = np.zeros((1, hidden_layer1))

W2 = np.random.randn(hidden_layer1, hidden_layer2)
b2 = np.zeros((1, hidden_layer2))

W3 = np.random.randn(hidden_layer2, output_layer)
b3 = np.zeros((1, output_layer))

lr = 0.5
epochs = 6000

losses = []

# Training
for epoch in range(epochs):

    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(a2, W3) + b3
    output = sigmoid(z3)

    # Loss
    loss = np.mean((y - output)**2)
    losses.append(loss)

    # Backpropagation
    error = y - output
    d3 = error * sigmoid_derivative(output)

    error2 = d3.dot(W3.T)
    d2 = error2 * sigmoid_derivative(a2)

    error1 = d2.dot(W2.T)
    d1 = error1 * sigmoid_derivative(a1)

    # Update weights
    W3 += a2.T.dot(d3) * lr
    b3 += np.sum(d3, axis=0, keepdims=True) * lr

    W2 += a1.T.dot(d2) * lr
    b2 += np.sum(d2, axis=0, keepdims=True) * lr

    W1 += X.T.dot(d1) * lr
    b1 += np.sum(d1, axis=0, keepdims=True) * lr

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss = {loss}")

# Final output
print("\nFinal Predictions:")
print(output.round())

# Accuracy
pred = output.round()
accuracy = np.mean(pred == y)
print("Accuracy:", accuracy)

# Loss graph
plt.figure(figsize=(6,4))
plt.plot(losses)
plt.title("Training Loss (MLP)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()

# Decision Boundary
def plot_boundary():
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

            a1 = sigmoid(np.dot(inp, W1) + b1)
            a2 = sigmoid(np.dot(a1, W2) + b2)
            out = sigmoid(np.dot(a2, W3) + b3)

            if out > 0.5:
                plt.scatter(x1, x2, color='lightgreen', alpha=0.1)
            else:
                plt.scatter(x1, x2, color='lightcoral', alpha=0.1)

    plt.title("MLP Decision Boundary")
    plt.show()

plot_boundary()

# Test new inputs
test = np.array([[0,1],[1,1],[1,0]])

a1 = sigmoid(np.dot(test, W1) + b1)
a2 = sigmoid(np.dot(a1, W2) + b2)
test_output = sigmoid(np.dot(a2, W3) + b3)

print("\nTest Predictions:")
print(test_output.round())

print("\n===== EXPERIMENT COMPLETED SUCCESSFULLY =====")