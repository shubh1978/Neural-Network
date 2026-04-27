import numpy as np
import matplotlib.pyplot as plt

print("===== HOPFIELD NETWORK EXPERIMENT =====")

# -------------------------------
# 1. DEFINE PATTERNS
# -------------------------------

patterns = np.array([
    [1, -1, 1, -1],
    [-1, 1, -1, 1],
    [1, 1, -1, -1]
])

print("\nStored Patterns:\n", patterns)

# -------------------------------
# 2. TRAIN (WEIGHT MATRIX)
# -------------------------------

n = patterns.shape[1]
W = np.zeros((n, n))

for p in patterns:
    p = p.reshape(n, 1)
    W += np.dot(p, p.T)

np.fill_diagonal(W, 0)

print("\nWeight Matrix:\n", W)

# -------------------------------
# 3. ENERGY FUNCTION
# -------------------------------

def energy(state, W):
    return -0.5 * np.dot(state.T, np.dot(W, state))

# -------------------------------
# 4. UPDATE FUNCTION
# -------------------------------

def update(state, W):
    new_state = state.copy()
    for i in range(len(state)):
        raw = np.dot(W[i], state)
        new_state[i] = 1 if raw >= 0 else -1
    return new_state

# -------------------------------
# 5. TEST WITH NOISE
# -------------------------------

test_pattern = np.array([1, -1, -1, -1])  # noisy
print("\nNoisy Input:", test_pattern)

states = [test_pattern.copy()]
energies = [energy(test_pattern, W)]

# Iterative update
current = test_pattern.copy()

for i in range(10):
    current = update(current, W)
    states.append(current.copy())
    energies.append(energy(current, W))

print("\nRecovered Pattern:", current)

# -------------------------------
# 6. ENERGY GRAPH
# -------------------------------

plt.figure(figsize=(6,4))
plt.plot(energies, marker='o')
plt.title("Energy Minimization")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.grid()
plt.show()

# -------------------------------
# 7. VISUALIZE STATES
# -------------------------------

plt.figure(figsize=(10,2))

for i, state in enumerate(states[:5]):
    plt.subplot(1,5,i+1)
    plt.imshow(state.reshape(2,2), cmap='gray')
    plt.title(f"Step {i}")
    plt.axis('off')

plt.show()

# -------------------------------
# 8. MULTIPLE TESTS
# -------------------------------

test_cases = [
    np.array([1,-1,-1,-1]),
    np.array([-1,1,-1,1]),
    np.array([1,1,1,-1])
]

for idx, test in enumerate(test_cases):
    current = test.copy()
    for _ in range(5):
        current = update(current, W)
    print(f"\nTest {idx+1} Input:", test)
    print("Recovered:", current)

# -------------------------------
# 9. CAPACITY CHECK
# -------------------------------

random_patterns = np.random.choice([-1,1], size=(5,4))

W_test = np.zeros((4,4))
for p in random_patterns:
    p = p.reshape(4,1)
    W_test += np.dot(p,p.T)

np.fill_diagonal(W_test, 0)

print("\nRandom Patterns Stored:\n", random_patterns)
print("New Weight Matrix:\n", W_test)

# -------------------------------
# 10. FINAL OUTPUT
# -------------------------------

print("\n===== EXPERIMENT COMPLETED SUCCESSFULLY =====")