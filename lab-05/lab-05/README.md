# ARTI 308 – Lab 5: Feature Engineering (Classification)

**Student:** Mohammed Hussain Alkahmis  
**Student ID:** 2240005057

This folder is ready to upload to GitHub as **lab-05**.

## Lab idea
This lab predicts **Order_Status** using the uploaded **Talabat-style orders dataset**.  
The work focuses on:
- feature engineering
- testing different feature ideas
- comparing model performance
- checking whether feature selection helps

## Files
- `ARTI308_Lab5_Solution.ipynb` → the full lab notebook with the solved tasks
- `talabat_enhanced_orders.csv` → the dataset used in the notebook
- `requirements.txt` → Python packages needed to run the notebook

## Solved tasks
The notebook includes completed answers for:

1. **Task 1:** Added a new engineered feature called `driver_close_to_restaurant`
2. **Task 2:** Tested a wider `is_peak_hour` rule
3. **Task 3:** Compared `top_k` values of `Item_Name_reduced` using 10, 30, and 50
4. **Task 4:** Compared the baseline model with feature selection

## Result summary
In my local run on your uploaded CSV, the comparison was:

- **Baseline accuracy:** about **0.8499**
- **Task 1 (`driver_close_to_restaurant`) accuracy:** about **0.8501**
- **Task 2 (expanded peak-hour rule) accuracy:** about **0.8507**
- **Task 3 (`top_k = 10, 30, 50`) accuracy:** almost unchanged at about **0.8499**
- **Task 4 (feature selection) accuracy:** about **0.8497**

## Main conclusion
- The best small improvement came from using a **broader peak-hour definition**
- The new driver proximity feature was still a good business-driven feature, but the gain was very small
- Changing `top_k` for `Item_Name_reduced` did not noticeably affect the model
- Feature selection was **not** beneficial for this dataset

## How to run
1. Open the folder
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Run `ARTI308_Lab5_Solution.ipynb`

## Suggested GitHub structure
```text
lab-05/
├── ARTI308_Lab5_Solution.ipynb
├── talabat_enhanced_orders.csv
├── requirements.txt
└── README.md
```