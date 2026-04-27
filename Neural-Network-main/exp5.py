import numpy as np
import matplotlib.pyplot as plt

print("===== PERCEPTRON LAB EXPERIMENT =====")

# Activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron class
class Perceptron:
    def __init__(self, lr=0.1, epochs=20):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            total_error = 0

            for i in range(len(X)):
                linear = np.dot(X[i], self.weights) + self.bias
                y_pred = step_function(linear)

                error = y[i] - y_pred
                total_error += abs(error)

                self.weights += self.lr * error * X[i]
                self.bias += self.lr * error

            print(f"Epoch {epoch+1}, Error = {total_error}")

    def predict(self, X):
        results = []
        for x in X:
            linear = np.dot(x, self.weights) + self.bias
            results.append(step_function(linear))
        return np.array(results)


# Dataset (AND Gate)
X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0,0,0,1])

# Dataset (OR Gate)
X_or = np.array([[0,0],[0,1],[1,0],[1,1]])
y_or = np.array([0,1,1,1])

# Train AND
print("\n--- AND Gate Training ---")
model_and = Perceptron(lr=0.1, epochs=10)
model_and.fit(X_and, y_and)

pred_and = model_and.predict(X_and)
print("Predictions:", pred_and)

accuracy_and = np.mean(pred_and == y_and)
print("Accuracy:", accuracy_and)

# Train OR
print("\n--- OR Gate Training ---")
model_or = Perceptron(lr=0.1, epochs=10)
model_or.fit(X_or, y_or)

pred_or = model_or.predict(X_or)
print("Predictions:", pred_or)

accuracy_or = np.mean(pred_or == y_or)
print("Accuracy:", accuracy_or)


# Visualization function
def plot_decision_boundary(X, y, model, title):
    plt.figure(figsize=(5,5))

    for i in range(len(X)):
        if y[i] == 0:
            plt.scatter(X[i][0], X[i][1], color='red')
        else:
            plt.scatter(X[i][0], X[i][1], color='green')

    x_vals = np.linspace(-1, 2, 100)
    if model.weights[1] != 0:
        y_vals = -(model.weights[0] * x_vals + model.bias) / model.weights[1]
        plt.plot(x_vals, y_vals)

    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid()
    plt.show()


# Plot decision boundaries
plot_decision_boundary(X_and, y_and, model_and, "AND Gate Decision Boundary")
plot_decision_boundary(X_or, y_or, model_or, "OR Gate Decision Boundary")


# Testing new inputs
test_data = np.array([[0,1],[1,1],[0,0]])

print("\n--- Testing New Data (AND) ---")
print(model_and.predict(test_data))

print("\n--- Testing New Data (OR) ---")
print(model_or.predict(test_data))


print("\n===== EXPERIMENT COMPLETED SUCCESSFULLY =====")