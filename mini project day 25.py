# ============================================
# FEATURE ENGINEERING MINI-PROJECT (END-TO-END)
# ============================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("data.csv")

# -------------------------------
# 2. BASIC CLEANING
# -------------------------------
df = df.drop_duplicates()
df = df.fillna(df.median(numeric_only=True))

# -------------------------------
# 3. TARGET & FEATURES
# -------------------------------
y = df["revenue"]
X = df.drop(columns=["revenue"])

# -------------------------------
# 4. DOMAIN-DRIVEN FEATURES
# -------------------------------
X["revenue_per_customer"] = df["revenue"] / (df["customers"] + 1)
X["price_per_unit"] = df["price"] / (df["quantity"] + 1)
X["marketing_efficiency"] = df["revenue"] / (df["marketing_spend"] + 1)

X["log_marketing_spend"] = np.log1p(df["marketing_spend"])
X["log_customers"] = np.log1p(df["customers"])

# -------------------------------
# 5. INTERACTION FEATURES
# -------------------------------
interaction_cols = ["price", "quantity", "marketing_spend"]

poly = PolynomialFeatures(
    degree=2,
    interaction_only=True,
    include_bias=False
)

poly_features = poly.fit_transform(X[interaction_cols])
poly_feature_names = poly.get_feature_names_out(interaction_cols)

poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)

X = pd.concat([X.reset_index(drop=True), poly_df.reset_index(drop=True)], axis=1)

# -------------------------------
# 6. TRAIN / TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 7. PREPROCESSING
# -------------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", "passthrough", categorical_features)
    ]
)

# -------------------------------
# 8. MODEL PIPELINE
# -------------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# -------------------------------
# 9. TRAIN MODEL
# -------------------------------
model.fit(X_train, y_train)

# -------------------------------
# 10. EVALUATION
# -------------------------------
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("Model Evaluation")
print("----------------")
print(f"RMSE: {rmse:.2f}")

# -------------------------------
# 11. FEATURE DOCUMENTATION
# -------------------------------
feature_documentation = {
    "revenue_per_customer": "Average revenue generated per customer",
    "price_per_unit": "Effective unit price accounting for quantity",
    "marketing_efficiency": "Revenue per unit of marketing spend",
    "log_marketing_spend": "Log-transformed marketing spend to reduce skew",
    "log_customers": "Log-transformed customer count"
}

print("\nFeature Documentation")
print("---------------------")
for feature, description in feature_documentation.items():
    print(f"{feature}: {description}")
