### **IBM Employee Attrition: Early Warning System**

## 📌 Project Overview

This project is an end-to-end Machine Learning pipeline designed to predict employee churn. By analyzing demographic, performance, and role-based metrics from the IBM HR Analytics dataset, this pipeline deploys a Logistic Regression model to act as an "Early Warning System." The goal is to identify flight-risk employees so HR can intervene with retention strategies before the employee resigns.

## 🛠️ Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Logistic Regression, StandardScaler, Metrics)
* **Statistics:** SciPy (Chi-Square Test)

## 🚀 The Journey & Overcoming Bottlenecks

Building this pipeline wasn't a straight line. Here is how the model evolved from encountering fatal math errors to achieving a production-ready Recall score.

### 1. The Zero-Variance Trap (`LinAlgError`)

* **The Issue:** Initial attempts to plot Kernel Density Estimate (KDE) curves crashed.
* **The Cause:** The dataset contained columns with zero standard deviation (e.g., `EmployeeCount` = 1 for everyone). These constant variables broke the mathematical division required for density curves.
* **The Fix:** Wrote a programmatic loop to identify and drop any column where `std() == 0` before visualizing.

### 2. The Multicollinearity Discovery

* **The Issue:** Feeding redundant data to a linear model causes unstable weights.
* **The Cause:** A Correlation Heatmap revealed a `0.95` correlation between `JobLevel` and `MonthlyIncome`. In the corporate world, rank strictly dictates salary.
* **The Fix:** Dropped `JobLevel` to eliminate multicollinearity, keeping `MonthlyIncome` because its continuous nature provides more granular mathematical data points for the gradient descent algorithm.

### 3. The Dummy Variable Target Bug

* **The Issue:** Target leakage and duplication.
* **The Cause:** Running `pd.get_dummies()` on the entire dataset after mapping the `Attrition` target to 1/0 caused the function to encode the target variable a second time.
* **The Fix:** Separated the target variable manually and strictly applied `get_dummies` only to the predictor variables.

### 4. The Imbalance Trap (19% Recall)

* **The Issue:** The initial model achieved an AUC-ROC of 0.75, but a dismal **19.1% Recall**. It was missing 81% of the employees who actually quit.
* **The Cause:** Class Imbalance. Only ~16% of the dataset represented employees leaving. The model optimized for overall accuracy by playing it safe and simply predicting that almost everyone would stay.
* **The Fix:** Introduced the `class_weight='balanced'` hyperparameter. This mathematically penalized the algorithm heavily for False Negatives, forcing it to pay attention to the minority class.
* **The Result:** Recall skyrocketed from **19.1% to 61.7%**, successfully flagging the majority of flight-risk employees for HR intervention without breaking the overall ROC curve.

## 📊 Feature Selection Methodology

Instead of throwing all 35 columns into the model, rigorous feature selection was applied:

1. **Numerical Features:** Analyzed 25 overlapping KDE plots to find "daylight" (shifts in distribution between people who left vs. stayed). Selected strong predictors like `StockOptionLevel` and `YearsAtCompany`.
2. **Categorical Features:** Ran a **Chi-Square Test of Independence** on text columns. Filtered out noise (like `Gender` with a p-value of 0.29) and kept mathematically significant drivers (like `OverTime` with a p-value of 8.15e-21).

