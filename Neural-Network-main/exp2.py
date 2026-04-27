# ----------------------------------------
# Experiment 2: Pandas Data Analysis
# ----------------------------------------

import pandas as pd
import numpy as np

print("===== PANDAS LAB EXPERIMENT =====")

# ----------------------------------------
# 1. Creating DataFrame
# ----------------------------------------

print("\n--- Creating DataFrame ---")

data = {
    'Name': ['Amit', 'Ravi', 'Sita', 'Neha', 'Karan'],
    'Age': [20, 21, 19, 22, 23],
    'Marks': [85, 90, 78, 92, 88],
    'City': ['Delhi', 'Mumbai', 'Pune', 'Delhi', 'Mumbai']
}

df = pd.DataFrame(data)

print("DataFrame:\n", df)

# ----------------------------------------
# 2. Basic Information
# ----------------------------------------

print("\n--- Data Info ---")

print("Shape:", df.shape)
print("Columns:", df.columns)
print("Data Types:\n", df.dtypes)

print("\nSummary Statistics:\n", df.describe())

# ----------------------------------------
# 3. Column Operations
# ----------------------------------------

print("\n--- Column Operations ---")

print("Marks Column:\n", df['Marks'])

print("Mean Marks:", df['Marks'].mean())
print("Max Marks:", df['Marks'].max())
print("Min Marks:", df['Marks'].min())

# Add new column
df['Pass'] = df['Marks'] >= 80

print("\nUpdated DataFrame:\n", df)

# ----------------------------------------
# 4. Filtering Data
# ----------------------------------------

print("\n--- Filtering ---")

high_scorers = df[df['Marks'] > 85]
print("Students with Marks > 85:\n", high_scorers)

delhi_students = df[df['City'] == 'Delhi']
print("Students from Delhi:\n", delhi_students)

# ----------------------------------------
# 5. Sorting Data
# ----------------------------------------

print("\n--- Sorting ---")

sorted_df = df.sort_values(by='Marks', ascending=False)
print("Sorted by Marks:\n", sorted_df)

# ----------------------------------------
# 6. Grouping Data
# ----------------------------------------

print("\n--- GroupBy ---")

grouped = df.groupby('City')['Marks'].mean()
print("Average Marks by City:\n", grouped)

# ----------------------------------------
# 7. Handling Missing Values
# ----------------------------------------

print("\n--- Missing Values ---")

df2 = df.copy()

# Introduce missing values
df2.loc[2, 'Marks'] = np.nan

print("Data with Missing Value:\n", df2)

print("Check Missing:\n", df2.isnull())

# Fill missing values
df2['Marks'].fillna(df2['Marks'].mean(), inplace=True)

print("After Filling Missing:\n", df2)

# ----------------------------------------
# 8. Indexing (loc and iloc)
# ----------------------------------------

print("\n--- Indexing ---")

print("Using loc:\n", df.loc[0])

print("Using iloc:\n", df.iloc[1])

print("Specific rows and columns:\n", df.loc[0:2, ['Name', 'Marks']])

# ----------------------------------------
# 9. Applying Functions
# ----------------------------------------

print("\n--- Apply Function ---")

def grade(marks):
    if marks >= 90:
        return 'A'
    elif marks >= 80:
        return 'B'
    else:
        return 'C'

df['Grade'] = df['Marks'].apply(grade)

print("With Grades:\n", df)

# ----------------------------------------
# 10. Merging DataFrames
# ----------------------------------------

print("\n--- Merging ---")

extra_data = pd.DataFrame({
    'Name': ['Amit', 'Ravi', 'Sita', 'Neha', 'Karan'],
    'Attendance': [90, 85, 88, 92, 87]
})

merged = pd.merge(df, extra_data, on='Name')

print("Merged DataFrame:\n", merged)

# ----------------------------------------
# 11. Saving and Loading Data
# ----------------------------------------

print("\n--- File Operations ---")

# Save to CSV
df.to_csv("students.csv", index=False)

# Read from CSV
loaded_df = pd.read_csv("students.csv")

print("Loaded Data:\n", loaded_df)

# ----------------------------------------
# 12. Conclusion
# ----------------------------------------

print("\n===== EXPERIMENT COMPLETED SUCCESSFULLY =====")