import pandas as pd
import numpy as np

df_input = pd.DataFrame({
    "user_age": [18, 25, None, 37, 45, None],
    "salary": [4000, 7500, 6200, None, 180000, 9000],
    "location": ["Riyadh", " riyadh ", "Jeddah", "Dammam", "Riyadh ", None],
    "created_at": [
        "2024-01-01 10:30",
        "2024/01/02 14:00",
        None,
        "2024-01-04 18:45",
        "2024-01-05 09:00",
        "2024-01-06"
    ]
})

def clean_dataset(df_input):
    df = df_input.copy()
    
    df["user_age"] = pd.to_numeric(df["user_age"], errors="coerce")
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    
    df["user_age_missing"] = df["user_age"].isna().astype(int)
    df["salary_missing"] = df["salary"].isna().astype(int)
    
    df["user_age"] = df["user_age"].fillna(df["user_age"].median())
    df["salary"] = df["salary"].fillna(df["salary"].median())
    
    salary_cap = df["salary"].quantile(0.99)
    df["salary"] = df["salary"].clip(upper=salary_cap)
    
    df["location"] = df["location"].str.lower().str.strip()
    
    df["created_at"] = df["created_at"].dt.tz_localize("UTC")
    
    return df

print(df_input.head())
print(df_input.info())

df_cleaned = clean_dataset(df_input)

print(df_cleaned.info())
print(df_cleaned[["user_age", "salary"]].describe())
print(df_cleaned["location"].value_counts())
print(df_cleaned["created_at"].dt.tz)
