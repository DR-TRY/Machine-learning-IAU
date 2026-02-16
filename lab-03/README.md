# ARTI308 — Lab 3 (EDA) Solution using Dataset.xlsx (Telco Customer Churn)

**Exploratory Data Analysis (EDA)** solution for ARTI308 Lab 3 using **Dataset.xlsx**.

## Files
- `CCSIT_ARTI308_Lab3_Solution_Telco_EN.ipynb` — The EDA solution notebook
- `Dataset.xlsx` — The dataset (Telco Customer Churn)

## Dataset Overview
- Rows: 7043
- Columns: 21
- Target: `Churn` (Yes/No)

## What the notebook does (Lab 3 requirements)
1. Load the dataset and preview it
2. Check missing values
3. Check duplicate rows
4. Show number of rows and columns
5. Show data types of columns
6. Clean/convert types (fix `TotalCharges` to numeric)
7. Descriptive statistics (numeric + categorical)
8. Univariate analysis (histograms + churn distribution)
9. Bivariate analysis (churn rate by Contract / InternetService / PaymentMethod + boxplot)
10. Correlation matrix for numeric columns
11. `Time-based analysis` using **tenure (months)** as a proxy (churn trend by tenure)

## How to run
1. Put the notebook and `Dataset.xlsx` in the same folder.
2. Open the notebook in Jupyter / Colab / VS Code.
3. Run all cells from top to bottom.

## Notes
- This dataset does not include calendar dates, so the notebook uses **tenure (months)** as a time proxy.
- `TotalCharges` may contain blanks for customers with `tenure = 0`.  
  The notebook converts it to numeric and fills missing values using:
  `TotalCharges = MonthlyCharges * tenure`.

1) Load the dataset and preview it

How: Read Dataset.xlsx using pandas and preview rows with `head()` and `info()`.
Why: To confirm the dataset is loaded correctly and understand its structure quickly.

2) Check missing values

How: Use  `isna().sum()` to count missing values per column.
Why: Missing values can cause errors and affect results, so we must detect them early.

3) Check duplicate rows

How: Use `duplicated().sum()` and remove duplicates if necessary.
Why: Duplicates can bias statistics and churn rates.

4) Show number of rows and columns

How: Use df.shape.
Why: To document dataset size and confirm it matches expected dimensions.

5) Show data types of columns

How: Use `df.dtypes` or `df.info().`
Why: Correct data types are required for proper calculations and visualizations.

6) Clean/convert types (fix TotalCharges to numeric)

How: Convert TotalCharges to numeric and handle blanks as missing values.
Why: This column is needed for numeric analysis, but it may contain text/blank values.

7) Descriptive statistics (numeric + categorical)

How: Use describe() for numeric columns and value_counts() for categorical columns.
Why: To summarize the dataset and understand distributions and common categories.

8) Univariate analysis (histograms + churn distribution)

How: Plot histograms for numeric features and a count plot for Churn.
Why: To understand each variable on its own and see churn balance (Yes vs No).

9) Bivariate analysis (relationships with churn)

How: Compute churn rates by Contract, InternetService, and PaymentMethod, and draw a boxplot for numeric comparison.
Why: To identify which customer groups are more likely to churn and how features differ by churn status.

10) Correlation matrix for numeric columns

How: Select numeric columns and compute a correlation matrix (optionally visualize it).
Why: To find relationships between numeric variables and detect strong associations.

11) Time-based analysis using tenure as a proxy

How: Analyze churn trends across different tenure (months) ranges.
Why: The dataset has no dates, so tenure is used as a time-like indicator.

## 1) 1- Data Visualization - Matplotlib.ipynb

A beginner-friendly notebook that introduces Matplotlib (the core Python plotting library).

What it covers:

Installing and importing Matplotlib

Creating basic plots (e.g., line plots)

Adding labels, titles, and axis names

Creating multiple plots on the same canvas (subplots)

Using the Object-Oriented (OO) approach (figure and axes) for better control

Adjusting figure size, DPI, and layout

Saving figures to files

Styling plots (colors, line width, line styles, markers)

Controlling axis appearance and plot ranges

Why it’s useful:

Matplotlib gives you full control over how your plots look and is the foundation for many visualization tools in Python.
