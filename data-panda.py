# Import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# -------------------------
# Task 1: Load and Explore the Dataset
# -------------------------
try:
    # Load the Iris dataset from sklearn
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    # Replace the default species names with cat family species
    cat_family = ['lion', 'tiger', 'cheetah']
    df['species'] = pd.Categorical.from_codes(iris.target, cat_family)


    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())

    # Clean: Drop or fill missing values (none expected in Iris, but just in case)
    df.dropna(inplace=True)

except FileNotFoundError:
    print("Error: Dataset file not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# -------------------------
# Task 2: Basic Data Analysis
# -------------------------
print("\nDescriptive statistics:")
print(df.describe())

print("\nAverage values grouped by species:")
print(df.groupby("species").mean())

# -------------------------
# Task 3: Data Visualization
# -------------------------
sns.set(style="whitegrid")

# 1. Line Chart – Simulate time series using index
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length")
plt.title("Line Chart of Sepal Length over Index (Simulated Time)")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar Chart – Average petal length by species
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="species", y="petal length (cm)", ci=None)
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# 3. Histogram – Distribution of sepal width
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="sepal width (cm)", kde=True, bins=20)
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.tight_layout()
plt.show()

# 4. Scatter Plot – Sepal length vs Petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="sepal length (cm)", y="petal length (cm)", hue="species")
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.tight_layout()
plt.show()
