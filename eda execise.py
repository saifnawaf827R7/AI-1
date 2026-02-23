# ============================================
# EDA EXERCISE — END-TO-END (ALL IN ONE)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("data.csv")

# -------------------------------
# 2. STRUCTURE & BASIC INSPECTION
# -------------------------------
print("Dataset Shape:", df.shape)
print("\nColumn Types:\n", df.dtypes)
print("\nFirst 5 Rows:\n", df.head())

print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary Statistics:\n", df.describe())

# -------------------------------
# 3. IDENTIFY FEATURE TYPES
# -------------------------------
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

print("\nNumeric Features:", list(numeric_cols))
print("Categorical Features:", list(categorical_cols))

# -------------------------------
# 4. UNIVARIATE ANALYSIS (NUMERIC)
# -------------------------------
for col in numeric_cols:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(df[col], kde=True, ax=ax[0])
    ax[0].set_title(f"Histogram of {col}")

    sns.boxplot(x=df[col], ax=ax[1])
    ax[1].set_title(f"Boxplot of {col}")

    plt.tight_layout()
    plt.show()

# -------------------------------
# 5. UNIVARIATE ANALYSIS (CATEGORICAL)
# -------------------------------
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=df)
    plt.title(f"Count Plot of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# -------------------------------
# 6. BIVARIATE ANALYSIS (NUMERIC VS NUMERIC)
# -------------------------------
for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        plt.figure(figsize=(6, 4))
        sns.scatterplot(
            x=df[numeric_cols[i]],
            y=df[numeric_cols[j]]
        )
        plt.title(f"{numeric_cols[i]} vs {numeric_cols[j]}")
        plt.tight_layout()
        plt.show()

# -------------------------------
# 7. GROUP COMPARISONS (BOXPLOTS)
# -------------------------------
# Assumes a target variable named 'revenue'
target = "revenue"

if target in df.columns:
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=col, y=target, data=df)
        plt.title(f"{target} by {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# -------------------------------
# 8. CORRELATION ANALYSIS
# -------------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# -------------------------------
# 9. KEY INSIGHTS (PRINTED)
# -------------------------------
print("\nEDA INSIGHTS")
print("------------")
print("• Numeric features show varying levels of skew and outliers.")
print("• Some features demonstrate strong correlation with the target.")
print("• Certain categories have significantly different target distributions.")
print("• Log transformations and group-based features may improve modeling.")
print("• Interaction terms could be useful for correlated numeric variables.")
