import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

# Load data
df = pd.read_excel("Dataset.xlsx")

# -------------------------------
# Task 1: Data Quality Assessment
# -------------------------------
print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate customerID:", df["customerID"].duplicated().sum())

# Convert TotalCharges to numeric (blanks -> NaN)
blank_totalcharges = df["TotalCharges"].astype(str).str.strip().eq("").sum()
print("\nBlank strings in TotalCharges:", blank_totalcharges)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Missing values after conversion:\n", df.isna().sum()[df.isna().sum() > 0])

# ------------------------------------------
# Task 2: Missing Value Strategy (Domain-Aware)
# ------------------------------------------
# In this dataset, missing TotalCharges rows correspond to new customers (tenure == 0),
# so TotalCharges should logically be 0.
mask = df["TotalCharges"].isna() & (df["tenure"] == 0)
df.loc[mask, "TotalCharges"] = 0.0

# Defensive fallback (should not be needed)
if df["TotalCharges"].isna().sum() > 0:
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

print("\nRemaining missing values in TotalCharges:", df["TotalCharges"].isna().sum())

# ------------------------------------
# Task 3: Detect & Handle Outliers (IQR)
# ------------------------------------
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
df_iqr = df.copy()
summary = []

for col in num_cols:
    q1 = df_iqr[col].quantile(0.25)
    q3 = df_iqr[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_mask = (df_iqr[col] < lower) | (df_iqr[col] > upper)
    outlier_count = int(outlier_mask.sum())

    before = df_iqr[col].copy()
    df_iqr[col] = df_iqr[col].clip(lower, upper)   # IQR capping (winsorization)
    capped_count = int((before != df_iqr[col]).sum())

    summary.append([col, q1, q3, iqr, lower, upper, outlier_count, capped_count])

iqr_summary = pd.DataFrame(summary, columns=[
    "feature", "Q1", "Q3", "IQR", "lower_bound", "upper_bound", "outlier_count", "capped_count"
])
print("\nIQR Outlier Summary:\n", iqr_summary)

# -------------------------------------------
# Task 4: Normalize Numerical Features (2 ways)
# -------------------------------------------
# Min-Max
mm = MinMaxScaler()
df_minmax = df_iqr.copy()
df_minmax[[f"{c}_MinMax" for c in num_cols]] = mm.fit_transform(df_iqr[num_cols])

# Z-score
scaler = StandardScaler()
df_zscore = df_iqr.copy()
df_zscore[[f"{c}_Z" for c in num_cols]] = scaler.fit_transform(df_iqr[num_cols])

print("\nMin-Max min/max check:\n", df_minmax[[f"{c}_MinMax" for c in num_cols]].agg(["min", "max"]))
print("\nZ-score mean/std check:\n", df_zscore[[f"{c}_Z" for c in num_cols]].agg(["mean", "std"]))

# -------------------------------
# Task 5: PCA + Explained Variance
# -------------------------------
X_std = scaler.fit_transform(df_iqr[num_cols])
pca = PCA()
pcs = pca.fit_transform(X_std)

explained = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(X_std.shape[1])],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_)
})
loadings = pd.DataFrame(pca.components_.T, index=num_cols, columns=explained["PC"])

print("\nExplained Variance:\n", explained)
print("\nPCA Loadings:\n", loadings)

# Save outputs (optional)
df_iqr.to_csv("telco_after_iqr_handling.csv", index=False)
df_minmax.to_csv("telco_minmax_normalized.csv", index=False)
df_zscore.to_csv("telco_zscore_normalized.csv", index=False)
explained.to_csv("pca_explained_variance.csv", index=False)
loadings.to_csv("pca_loadings.csv")
