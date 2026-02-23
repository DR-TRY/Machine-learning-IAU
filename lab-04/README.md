# Data Quality Assessment & Preprocessing (Telco Customer Churn)

حل كامل لمتطلبات الواجب على ملف **Dataset.xlsx** (Telco Customer Churn).

## Tasks Covered
1. **Identify data quality issues**
2. **Apply one missing value strategy + explanation**
3. **Detect and handle outliers using IQR**
4. **Normalize numerical features using Min-Max and Z-score**
5. **Apply PCA and interpret explained variance**

## Main Findings (Summary)
- Dataset shape: **7043 rows × 21 columns**
- No duplicate rows and no duplicate `customerID`
- Main data quality issue: `TotalCharges` stored as **object** because it contains blank strings
- Blank / missing values in `TotalCharges`: **11**
- All missing `TotalCharges` rows correspond to **`tenure = 0`** (new customers), so they were filled with **0.0** (domain-aware imputation)
- IQR outlier check on (`tenure`, `MonthlyCharges`, `TotalCharges`) found **no outliers** under the standard **1.5×IQR** rule, so no clipping changes were needed
- PCA on standardized numerical features:
  - **PC1 explains 72.69%**
  - **PC1 + PC2 explain 98.01%** (almost all variance)

## Files Included
- `telco_preprocessing_solution.py` → Python solution script
- `Assignment_Solution_Telco_Preprocessing.ipynb` → Jupyter notebook solution
- `telco_preprocessing_results/` → exported cleaned/normalized/PCA result files
  - `01_telco_cleaned_missing_handled.csv`
  - `02_iqr_outlier_summary.csv`
  - `03_telco_after_iqr_handling.csv`
  - `04_telco_minmax_normalized.csv`
  - `05_telco_zscore_normalized.csv`
  - `06_pca_explained_variance.csv`
  - `07_pca_loadings.csv`
  - `08_pca_components_dataset.csv`

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
python telco_preprocessing_solution.py
```

## Notes
- The solution uses **domain knowledge** for missing values in `TotalCharges` (tenure = 0 → total charges should be 0).
- PCA is applied on **standardized** continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`), which is the recommended approach.
