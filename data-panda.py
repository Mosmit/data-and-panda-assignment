# Import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# -------------------------
# Task 1: Load and Explore the Dataset
# -------------------------
try:
    # Load the Iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    # Rename columns to fit cat anatomy
    df.rename(columns={
        'sepal length (cm)': 'fang length (cm)',
        'sepal width (cm)': 'fang width (cm)',
        'petal length (cm)': 'claw length (cm)',
        'petal width (cm)': 'claw width (cm)'
    }, inplace=True)

    # Rename species to big cat family
    cat_family = ['lion', 'tiger', 'cheetah']
    df['species'] = pd.Categorical.from_codes(iris.target, cat_family)

    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())

    # Clean missing values (if any)
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

print("\nAverage values grouped by cat species:")
print(df.groupby("species").mean())

# -------------------------
# Task 3: Data Visualization
# -------------------------
sns.set(style="whitegrid")

# 1. Line Chart – Fang length over index
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["fang length (cm)"], label="Fang Length", color="orange")
plt.title("Fang Length Trend Among Cat Species")
plt.xlabel("Observation Index")
plt.ylabel("Fang Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar Chart – Average claw length by cat species
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="species", y="claw length (cm)", palette="Set2", ci=None)
plt.title("Average Claw Length by Cat Species")
plt.xlabel("Cat Species")
plt.ylabel("Claw Length (cm)")
plt.tight_layout()
plt.show()

# 3. Histogram – Distribution of fang width
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="fang width (cm)", kde=True, bins=20, color="purple")
plt.title("Distribution of Fang Width Among Big Cats")
plt.xlabel("Fang Width (cm)")
plt.tight_layout()
plt.show()

# 4. Scatter Plot – Fang length vs Claw length
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="fang length (cm)", y="claw length (cm)", hue="species", palette="deep")
plt.title("Fang Length vs Claw Length by Cat Species")
plt.xlabel("Fang Length (cm)")
plt.ylabel("Claw Length (cm)")
plt.legend(title="Cat Species")
plt.tight_layout()
plt.show()
