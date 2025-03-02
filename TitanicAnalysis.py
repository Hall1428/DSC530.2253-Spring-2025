# Aaron Hall
# Titanic Analysis
# 2/28/2025

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Load the Titanic data
file_path = "C:\\Users\\aaron\\Desktop\\Aaron\\Bellevue Masters\\Bellevue Winter 2024\\Data Exploration and Analysis\\titanic3.csv"
df = pd.read_csv(file_path)

# Define male and female
df['sex'] = df['sex'].replace({'male': 0, 'female': 1})  

# Define Variables
print("Survival: 0 = did not survive, 1 = survived")
print("Class: 1 = 1st class, 2 = 2nd class, 3 = 3rd class")
print("Age: age in years")
print("Ticket Fare: price paid")
print("Sex: 0 = male, 1 = female")

# Variable Histograms
plt.figure(figsize=(15, 10))
for i, var in enumerate(['survived', 'pclass', 'age', 'fare', 'sex'], 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[var].dropna(), bins=20)
    plt.title(f'Histogram of {var}')
plt.tight_layout()
plt.show()

# Outliers explanation
print("No defined outliers. High fare over $300 could be luxery rooms")

# Descriptive Stats
print("\nDescriptive Stats:")
for var in ['survived', 'pclass', 'age', 'fare', 'sex']:
    mean = df[var].mean()
    mode = df[var].mode()[0]
    spread = df[var].max() - df[var].min()
    print(f"{var}:")
    print(f"  Mean = {mean:.2f}")
    print(f"  Mode = {mode}")
    print(f"  Spread = {spread}")
    print(f"  Tails: {'long right tail' if df[var].skew() > 0 else 'long left tail'}")

# PMF - 1st vs 3rd Class Survival Comparison
first_class = df[df['pclass'] == 1]['survived'].value_counts(normalize=True)
third_class = df[df['pclass'] == 3]['survived'].value_counts(normalize=True)
plt.figure(figsize=(6, 4))
plt.bar([0, 1], first_class, width=0.4, label='1st Class', alpha=0.7)
plt.bar([0.4, 1.4], third_class, width=0.4, label='3rd Class', alpha=0.7)
plt.xticks([0.2, 1.2], ['Did not Survive', 'Survived'])
plt.title('Survival Chance by Class')
plt.ylabel('Probability')
plt.legend()
plt.show()

# CDF of Age
ages = df['age'].dropna().sort_values()
cdf = np.arange(len(ages)) / len(ages)
plt.figure(figsize=(6, 4))
plt.plot(ages, cdf)
plt.title('CDF of Age')
plt.xlabel('Age')
plt.ylabel('Probability')
plt.grid()
plt.show()
print("\nMost passengers under 40")

# Normal Distribution for Age
plt.figure(figsize=(6, 4))
sns.histplot(df['age'].dropna(), bins=20, stat='density')
x = np.linspace(df['age'].min(), df['age'].max(), 100)
plt.plot(x, stats.norm.pdf(x, df['age'].mean(), df['age'].std()), 'r-')
plt.title('Age Distribution vs Normal Curve')
plt.show()
print("\nSomewhat normal with right tail")

# Scatter Plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(df['age'], df['fare'], alpha=0.5)
plt.title('Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')

plt.subplot(1, 2, 2)
plt.scatter(df['pclass'], df['survived'], alpha=0.5)
plt.title('Class vs Survival')
plt.xlabel('Class')
plt.ylabel('Survived')
plt.tight_layout()
plt.show()
print("No clear pattern for age vs fare")
print("1st class was more likely to survive")

# Hypothesis Test
table = pd.crosstab(df['pclass'], df['survived'])
chi2, p_value, _, _ = stats.chi2_contingency(table)
print("\nTest: Class affects Survival:")
print(f"Chi-square value: {chi2:.2f}")
print(f"P-value: {p_value:.4f}")
print("Small p-value means class greatly affects survival")

# Regression Model
X = df[['pclass']].dropna()
y = df['survived'].dropna()
X = np.c_[np.ones(len(X)), X]  # Add constant
slope, intercept = np.linalg.lstsq(X, y, rcond=None)[0]
print("\nRegression Results:")
print(f"Survival decreases by {abs(slope):.2f} per class")
print(f"Base survival chance (intercept): {intercept:.2f}")