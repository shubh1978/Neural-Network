import matplotlib.pyplot as plt
import numpy as np

print("===== MATPLOTLIB LAB EXPERIMENT =====")

# Data
x = np.arange(1, 11)
y1 = x * 2
y2 = x ** 2
y3 = np.sin(x)

# Line Plot
plt.figure(figsize=(6,4))
plt.plot(x, y1, label="y = 2x", marker='o')
plt.plot(x, y2, label="y = x^2", marker='s')
plt.title("Line Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()
plt.show()

# Bar Chart
students = ['A', 'B', 'C', 'D', 'E']
marks = [85, 90, 78, 92, 88]

plt.figure(figsize=(6,4))
plt.bar(students, marks)
plt.title("Bar Chart - Marks")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.show()

# Horizontal Bar
plt.figure(figsize=(6,4))
plt.barh(students, marks, color='orange')
plt.title("Horizontal Bar Chart")
plt.show()

# Scatter Plot
np.random.seed(0)
x_scatter = np.random.rand(50)
y_scatter = np.random.rand(50)

plt.figure(figsize=(6,4))
plt.scatter(x_scatter, y_scatter)
plt.title("Scatter Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Histogram
data = np.random.randn(1000)

plt.figure(figsize=(6,4))
plt.hist(data, bins=30)
plt.title("Histogram")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()

# Pie Chart
labels = ['Python', 'Java', 'C++', 'JS']
sizes = [40, 25, 20, 15]

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title("Pie Chart")
plt.show()

# Subplots
plt.figure(figsize=(10,6))

plt.subplot(2,2,1)
plt.plot(x, y1)
plt.title("Linear")

plt.subplot(2,2,2)
plt.plot(x, y2)
plt.title("Quadratic")

plt.subplot(2,2,3)
plt.plot(x, y3)
plt.title("Sine")

plt.subplot(2,2,4)
plt.bar(students, marks)
plt.title("Marks")

plt.tight_layout()
plt.show()

# Box Plot
data_box = [np.random.normal(0, std, 100) for std in range(1,5)]

plt.figure(figsize=(6,4))
plt.boxplot(data_box)
plt.title("Box Plot")
plt.show()

# Area Plot
plt.figure(figsize=(6,4))
plt.stackplot(x, y1, y2, labels=['y1','y2'])
plt.legend()
plt.title("Area Plot")
plt.show()

# Multiple Scatter with colors
colors = np.random.rand(50)

plt.figure(figsize=(6,4))
plt.scatter(x_scatter, y_scatter, c=colors)
plt.title("Colored Scatter")
plt.show()

print("\n===== EXPERIMENT COMPLETED SUCCESSFULLY =====")