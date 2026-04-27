from scipy import integrate, optimize, linalg
import numpy as np

print("===== SCIPY LAB EXPERIMENT =====")

# Integration
f = lambda x: x**2 + 3*x + 2
res, err = integrate.quad(f, 0, 5)
print("\nIntegration Result:", res)

# Double integration
f2 = lambda y, x: x*y
res2, err2 = integrate.dblquad(f2, 0, 2, lambda x: 0, lambda x: 1)
print("Double Integration:", res2)

# Solve equation
func = lambda x: x**3 - 2*x - 5
root = optimize.fsolve(func, 2)
print("\nRoot:", root)

# Minimize function
func2 = lambda x: (x-3)**2 + 4
min_res = optimize.minimize(func2, x0=0)
print("Minimum Value:", min_res.fun)
print("At x =", min_res.x)

# Linear algebra
A = np.array([[3,2],[1,2]])
B = np.array([5,5])

solution = linalg.solve(A, B)
print("\nSolution of Ax = B:", solution)

# Determinant and inverse
det = linalg.det(A)
inv = linalg.inv(A)

print("Determinant:", det)
print("Inverse:\n", inv)

# Eigenvalues and eigenvectors
eig_val, eig_vec = linalg.eig(A)

print("\nEigenvalues:", eig_val)
print("Eigenvectors:\n", eig_vec)

# Interpolation
from scipy.interpolate import interp1d

x = np.array([0,1,2,3,4])
y = np.array([0,1,4,9,16])

interp_func = interp1d(x, y, kind='linear')
print("\nInterpolation at 2.5:", interp_func(2.5))

# Statistics
from scipy import stats

data = np.array([10,20,30,40,50])

mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data)

print("\nMean:", mean)
print("Median:", median)
print("Mode:", mode.mode)

# Random distribution
rand = stats.norm.rvs(size=10)

print("\nRandom Normal Values:", rand)

print("\n===== EXPERIMENT COMPLETED SUCCESSFULLY =====")