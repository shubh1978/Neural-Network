import numpy as np
import matplotlib.pyplot as plt

print("===== OPTIMIZER EXPERIMENT =====")

# Function (simple quadratic)
def f(x):
    return x**2 + 4*x + 4

def grad(x):
    return 2*x + 4

# Gradient Descent
x_gd = 10
lr = 0.1
epochs = 50

gd_history = []

for i in range(epochs):
    x_gd = x_gd - lr * grad(x_gd)
    gd_history.append(f(x_gd))

print("Final x (GD):", x_gd)

# Adam Optimizer
x_adam = 10
lr = 0.1
m = 0
v = 0
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

adam_history = []

for t in range(1, epochs+1):
    g = grad(x_adam)

    m = beta1*m + (1-beta1)*g
    v = beta2*v + (1-beta2)*(g**2)

    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    x_adam = x_adam - lr * m_hat / (np.sqrt(v_hat) + eps)

    adam_history.append(f(x_adam))

print("Final x (Adam):", x_adam)

# Plot cost comparison
plt.figure(figsize=(6,4))
plt.plot(gd_history, label="Gradient Descent")
plt.plot(adam_history, label="Adam")
plt.title("Optimizer Comparison")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend()
plt.grid()
plt.show()

# Path visualization
x_vals = np.linspace(-10, 10, 100)
y_vals = f(x_vals)

plt.figure(figsize=(6,4))
plt.plot(x_vals, y_vals, label="Function")

# Plot GD path
x_temp = 10
for i in range(10):
    plt.scatter(x_temp, f(x_temp), color='red')
    x_temp = x_temp - lr * grad(x_temp)

# Plot Adam path
x_temp = 10
m = 0
v = 0

for t in range(1, 10):
    g = grad(x_temp)
    m = beta1*m + (1-beta1)*g
    v = beta2*v + (1-beta2)*(g**2)

    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    plt.scatter(x_temp, f(x_temp), color='green')
    x_temp = x_temp - lr * m_hat / (np.sqrt(v_hat) + eps)

plt.title("Optimization Path")
plt.legend()
plt.grid()
plt.show()

# Multiple learning rates
lrs = [0.01, 0.1, 0.5]

plt.figure(figsize=(6,4))

for lr in lrs:
    x = 10
    history = []
    for i in range(30):
        x = x - lr * grad(x)
        history.append(f(x))
    plt.plot(history, label=f"lr={lr}")

plt.title("Effect of Learning Rate")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend()
plt.grid()
plt.show()

print("\n===== EXPERIMENT COMPLETED SUCCESSFULLY =====")