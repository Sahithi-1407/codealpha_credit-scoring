# FinGuard: Credit Scoring & Default Risk Assessment Model 🏦

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-red.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20German%20Credit-green.svg)](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
[![License](https://img.shields.io/badge/License-Academic%20%2F%20MIT-lightgrey.svg)]()

A complete, beginner-friendly machine learning project designed for retail banking credit risk analysis and automated loan underwriting. Built with **Python**, **Scikit-Learn**, and **Streamlit** using the authentic **Statlog (German Credit Data)** dataset from the UCI Machine Learning Repository.

---

## 📌 1. Project Objective

In retail credit scoring, commercial financial institutions evaluate applicants seeking credit facilities to predict their creditworthiness. The classification objective is binary:
* **Good Credit Risk (`1`)**: Borrower possesses high repayment probability; eligible for credit approval.
* **Bad Credit Risk (`0`)**: Borrower exhibits high probability of delinquency or default; requires credit decline or conditional mitigation.

### The Asymmetric Cost Problem in Credit Risk
In traditional machine learning benchmarks, models are frequently judged solely by overall **Accuracy**. In commercial lending, however, accuracy is a misleading metric due to severe class imbalance (70% Good vs. 30% Bad) and **asymmetric financial costs**:

$$\text{Total Cost} = 5 \times \text{False Positives (Defaults Approved)} + 1 \times \text{False Negatives (Solvent Borrowers Declined)}$$

* **Type II Error (Approving a Defaulter)**: Results in write-offs of the loan principal and costly legal collections. Standard banking penalty = **5 units**.
* **Type I Error (Declining a Solvent Borrower)**: Results only in lost net interest margin. Standard banking penalty = **1 unit**.

This project explicitly prioritizes **Recall on Bad Credit Risk** and **Total Financial Cost Reduction** rather than selecting models on naive accuracy alone.

---

## 📁 2. Project Structure

```
Credit_Scoring_Project/
│
├── data/
│   └── german.data                   # Authentic UCI German Credit dataset (1,000 records)
│
├── models/
│   ├── best_credit_model.joblib      # Champion pipeline (Decision Tree - Lowest Cost)
│   ├── decision_tree.joblib          # Fitted Decision Tree pipeline
│   ├── random_forest.joblib          # Fitted Random Forest pipeline
│   ├── logistic_regression.joblib    # Fitted Logistic Regression pipeline
│   ├── metrics.json                  # Complete cross-validation and test set metrics
│   ├── model_comparison.json         # Benchmarking summary and selection justification
│   ├── roc_curve_comparison.png      # High-resolution comparative ROC curves
│   ├── confusion_matrices.png        # Comparative confusion matrices
│   └── feature_importance.png        # Top feature importance and coefficient signals
│
├── eda_plots/
│   ├── target_distribution.png       # Good vs. Bad target distribution chart
│   ├── age_distribution.png          # Applicant age KDE distribution by credit risk
│   ├── credit_amount_distribution.png# Credit amount histogram and boxplots
│   ├── job_distribution.png          # Risk breakdown by employment job category
│   ├── housing_distribution.png      # Housing tenure (Rent/Own/Free) vs default
│   ├── credit_duration_distribution.png # Loan duration horizon distribution
│   └── correlation_heatmap.png       # Numerical feature Pearson correlation matrix
│
├── src/
│   ├── __init__.py                   # Package initializer
│   ├── data_preprocessing.py         # Ingestion, category mapping, feature engineering, pipeline
│   ├── train.py                      # 5-fold CV, model training, evaluation, cost matrix, plotting
│   └── predict.py                    # Inference engine, FICO-scale scoring, factor analysis
│
├── app.py                            # Multi-page interactive Streamlit banking web application
├── requirements.txt                  # Pinned Python package dependencies
└── README.md                         # Project documentation and academic report
```

---

## 📊 3. Dataset Information

* **Source**: UCI Machine Learning Repository — [Statlog (German Credit Data)](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data)
* **Sample Size**: 1,000 credit agreements
* **Class Imbalance**: 700 Good Credit Risk (70%) / 300 Bad Credit Risk (30%)
* **Format**: Whitespace-separated text without header rows
* **Features**: 20 attributes (7 numerical, 13 categorical)

### Attribute Schema
| Feature | Type | Description / Categories |
| :--- | :---: | :--- |
| `checking_status` | Categorical | Current account balance (`< 0 DM`, `0-200 DM`, `>= 200 DM`, `No account`) |
| `duration` | Numerical | Credit agreement term in months (4 to 72) |
| `credit_history` | Categorical | Prior credit record (`No credits`, `All paid duly`, `Delays in past`, `Critical account`) |
| `purpose` | Categorical | Loan purpose (`Car`, `Furniture`, `Radio/TV`, `Appliances`, `Education`, `Business`) |
| `credit_amount` | Numerical | Credit principal in Deutsche Marks (250 to 18,424 DM) |
| `savings_status` | Categorical | Liquid savings & bond balance (`< 100 DM` to `>= 1,000 DM`, `Unknown`) |
| `employment` | Categorical | Current employment tenure (`Unemployed` to `>= 7 years`) |
| `installment_rate`| Numerical | Monthly installment as percentage of disposable income (1% to 4%) |
| `personal_status` | Categorical | Marital status and sex (`Male single`, `Male married`, `Female`, etc.) |
| `other_debtors` | Categorical | Guarantors or co-applicants (`None`, `Co-applicant`, `Guarantor`) |
| `residence_since` | Numerical | Years at present residence (1 to 4) |
| `property` | Categorical | Most valuable collateral asset (`Real estate`, `Life insurance`, `Car`, `None`) |
| `age` | Numerical | Chronological age in years (19 to 75) |
| `other_installment_plans` | Categorical | Installment credit with other providers (`Bank`, `Stores`, `None`) |
| `housing` | Categorical | Housing ownership status (`Rent`, `Own`, `For free`) |
| `existing_credits`| Numerical | Number of credit accounts at this bank (1 to 4) |
| `job` | Categorical | Employment qualification level (`Unskilled`, `Skilled`, `Management`) |
| `num_dependents` | Numerical | Number of maintenance dependents (1 or 2) |
| `telephone` | Categorical | Telephone registered under customer name (`Yes` / `None`) |
| `foreign_worker` | Categorical | Foreign worker status (`Yes` / `No`) |
| `credit_risk` | Target | **1 = Good Credit Risk (Solvent)**, **0 = Bad Credit Risk (Default)** |

---

## 🛠️ 4. Feature Engineering

Four domain-specific financial features are dynamically synthesized inside a scikit-learn custom transformer (`CreditFeatureEngineer`), preventing training-serving skew:

1. **`monthly_installment`** ($\text{Credit Amount} / \text{Duration}$):
   * *Financial Rationale*: Measures the monthly cash burden required to service the loan. A short loan with high principal creates severe liquidity pressure, significantly escalating default hazard.
2. **`age_group`** (`Young <25`, `Adult 25-35`, `Middle-Aged 35-50`, `Senior 50+`):
   * *Financial Rationale*: Age strongly tracks financial life stages. Younger borrowers typically have shorter credit histories and lower liquidity buffers.
3. **`duration_group`** (`Short <=12m`, `Medium 13-24m`, `Long 25-36m`, `Very Long >36m`):
   * *Financial Rationale*: Credit risk compounds over time. Multi-year commitments remain exposed to macroeconomic volatility and personal disruptions.
4. **`credit_amount_group`** (`Low <=2000 DM`, `Medium 2001-5000 DM`, `High 5001-10000 DM`, `Very High >10000 DM`):
   * *Financial Rationale*: Segregates small retail consumer borrowing from capital-intensive exposures that carry severe default loss severity.

---

## 🔬 5. Model Training & Evaluation Results

All candidate models were trained with `class_weight='balanced'` on an **80/20 stratified train-test split** (800 training records, 200 held-out test records). Models were first cross-validated across **5 stratified folds**, then tested on the unseen 200-sample test set.

### Comprehensive Test Set Benchmarking (200 Applicants: 140 Good, 60 Bad)

| Model Architecture | Test Accuracy | ROC-AUC | Bad-Risk Recall (Catch Defaults) | Good-Risk Recall | Defaults Incurred (Type II) | Total Financial Loss Cost | Champion Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | 66.00% | 0.7151 | **80.00%** | 60.00% | **12** | **116 units** | 🏆 **CHAMPION** |
| **Random Forest** | **69.50%** | **0.7880** | 71.67% | **68.57%** | 17 | 129 units | Runner-up |
| **Logistic Regression** | 69.00% | 0.7536 | 71.67% | 67.86% | 17 | 130 units | Baseline |

### Why Decision Tree Won Over Random Forest
* **Financial Bottom Line**: The Decision Tree saved the bank **13 units of financial cost** over Random Forest ($116 \text{ vs } 129$) by catching **80.00% of defaulting borrowers** (letting only 12 defaults through, versus 17 for Random Forest).
* **Compliance & Adverse Action**: Banking consumer regulations require lenders to provide specific adverse action reasons when denying credit. Decision tree rule splits are auditable and transparent to regulatory examiners.

---

## 🚀 6. Installation & Execution Guide

### Prerequisites
* Python 3.10 or 3.11
* Pip package manager

### 1. Clone or Navigate to the Project
```bash
cd d:/ML_Projects/Credit_Scoring_Project
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv credit_env
# On Windows:
.\credit_env\Scripts\activate
# On Linux / macOS:
source credit_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Models
To execute the complete 5-fold cross-validation, train all three models, compute the financial cost matrix, and generate performance plots:
```bash
python src/train.py
```

### 5. Run the Interactive Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ 7. Streamlit Application Walkthrough

The web application features 5 interconnected modules designed with a banking-grade interface:

1. **🏠 Home Dashboard**:
   * Portfolio KPI metrics (1,000 applications, 30% default rate, 80% bad risk recall).
   * Visual asymmetric cost matrix ($5\times \text{ vs } 1\times$).
   * End-to-end machine learning lifecycle overview.
2. **📁 Dataset Overview**:
   * Interactive data explorer with Good/Bad risk filtering.
   * Full 20-attribute data dictionary and summary statistics.
3. **📊 Exploratory Data Analysis**:
   * Tabbed visualization suite covering class imbalance, age, credit amount, employment, housing, duration, and correlation heatmaps with banking takeaways.
4. **📈 Model Performance & Comparison**:
   * Comparative metrics table across all three algorithms.
   * Receiver Operating Characteristic (ROC) curve overlays.
   * Side-by-side confusion matrices and feature importance analysis.
5. **🎯 Credit Risk Prediction (Underwriting Interface)**:
   * Quick-load realistic applicant profiles (*Prime Professional*, *Subprime Overdrawn*, *Borderline*).
   * Complete 20-input underwriting disclosure form.
   * Automated decision card (`APPROVED` vs `DECLINED`).
   * FICO-style credit score (300 to 850 scale) and default probability gauges.
   * Positive solvency drivers and negative credit deterioration factors.

---

## ⚖️ 8. Limitations, Regulatory Compliance & Ethical AI

1. **Historical Dataset Context**:
   * The German Credit dataset was compiled in 1994 in Deutsche Marks (DM). Economic baselines, inflation, and retail credit products have evolved significantly since that era.
2. **Protected Demographic Attributes**:
   * The dataset contains attributes such as `personal_status` (encoding sex and marital status) and `foreign_worker`.
   * Under modern statutory fair lending legislation (e.g., the U.S. **Equal Credit Opportunity Act (ECOA)** and the **EU AI Act / GDPR Article 22**), using gender, marital status, or nationality in automated credit underwriting is strictly prohibited.
3. **Algorithmic Fairness & Disparate Impact**:
   * Production credit scoring models must undergo formal disparate impact and demographic parity audits to verify that minority or protected classes are not subject to systemic underwriting bias.
4. **Human-in-the-Loop Underwriting**:
   * This platform is designed as an **underwriter assistant tool**, not a replacement for human credit officers. High-stakes financial approvals must maintain a human-in-the-loop escalation pathway.

