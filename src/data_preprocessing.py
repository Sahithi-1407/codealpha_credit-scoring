# ============================================================
# Credit_Scoring_Project/src/data_preprocessing.py
# Data preprocessing and feature engineering
# Six-feature Credit Scoring Model
# ============================================================

import os
import sys
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "matplotlib_credit")
)

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ============================================================
# 1. DATASET COLUMN NAMES
# ============================================================

RAW_COLUMN_NAMES = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_rate",
    "personal_status",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "credit_risk"
]


# ============================================================
# 2. CATEGORY MAPPINGS
# ============================================================

VALUE_MAPPINGS = {

    "checking_status": {
        "A11": "< 0 DM",
        "A12": "0 to < 200 DM",
        "A13": ">= 200 DM / Salary assignments >= 1 yr",
        "A14": "No checking account"
    },

    "credit_history": {
        "A30": "No credits taken / All paid back duly",
        "A31": "All credits at this bank paid back duly",
        "A32": "Existing credits paid back duly till now",
        "A33": "Delay in paying off in past",
        "A34": "Critical account / Other credits existing"
    },

    "purpose": {
        "A40": "Car (new)",
        "A41": "Car (used)",
        "A42": "Furniture / Equipment",
        "A43": "Radio / Television",
        "A44": "Domestic appliances",
        "A45": "Repairs",
        "A46": "Education",
        "A47": "Vacation",
        "A48": "Retraining",
        "A49": "Business",
        "A410": "Others"
    },

    "savings_status": {
        "A61": "< 100 DM",
        "A62": "100 to < 500 DM",
        "A63": "500 to < 1000 DM",
        "A64": ">= 1000 DM",
        "A65": "Unknown / No savings account"
    },

    "employment": {
        "A71": "Unemployed",
        "A72": "< 1 year",
        "A73": "1 to < 4 years",
        "A74": "4 to < 7 years",
        "A75": ">= 7 years"
    },

    "personal_status": {
        "A91": "Male : Divorced / Separated",
        "A92": "Female : Divorced / Separated / Married",
        "A93": "Male : Single",
        "A94": "Male : Married / Widowed",
        "A95": "Female : Single"
    },

    "other_debtors": {
        "A101": "None",
        "A102": "Co-applicant",
        "A103": "Guarantor"
    },

    "property": {
        "A121": "Real estate",
        "A122": "Building society savings / Life insurance",
        "A123": "Car or other property",
        "A124": "Unknown / No property"
    },

    "other_installment_plans": {
        "A141": "Bank",
        "A142": "Stores",
        "A143": "None"
    },

    "housing": {
        "A151": "Rent",
        "A152": "Own",
        "A153": "For free"
    },

    "job": {
        "A171": "Unemployed / Unskilled non-resident",
        "A172": "Unskilled resident",
        "A173": "Skilled employee / Official",
        "A174": "Management / Self-employed / Highly qualified"
    },

    "telephone": {
        "A191": "None",
        "A192": "Yes (registered under customer name)"
    },

    "foreign_worker": {
        "A201": "Yes",
        "A202": "No"
    }
}


# ============================================================
# 3. SELECTED SIX FEATURES
# ============================================================

NUMERICAL_COLS_RAW = [
    "duration",
    "credit_amount",
    "age"
]

CATEGORICAL_COLS_RAW = [
    "checking_status",
    "credit_history",
    "savings_status"
]

ENGINEERED_NUMERICAL_COLS = [
    "duration",
    "credit_amount",
    "age"
]

ENGINEERED_CATEGORICAL_COLS = [
    "checking_status",
    "credit_history",
    "savings_status"
]

SELECTED_FEATURES = [
    "age",
    "checking_status",
    "credit_history",
    "credit_amount",
    "duration",
    "savings_status"
]


# ============================================================
# 4. DATA LOADING
# ============================================================

def load_data(data_path=None, map_categories=True):

    if data_path is None:

        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        data_path = os.path.join(
            project_root,
            "data",
            "german.data"
        )

    if not os.path.exists(data_path):

        raise FileNotFoundError(
            f"Dataset not found at {data_path}"
        )

    print(f"Loading dataset: {data_path}")

    df = pd.read_csv(
        data_path,
        sep=r"\s+",
        header=None,
        names=RAW_COLUMN_NAMES
    )

    # Original target: 1 = Good, 2 = Bad
    # New target: 1 = Good, 0 = Bad

    df["credit_risk"] = (
        df["credit_risk"] == 1
    ).astype(int)

    if map_categories:

        for col, mapping in VALUE_MAPPINGS.items():

            if col in df.columns:

                df[col] = (
                    df[col]
                    .map(mapping)
                    .fillna(df[col])
                )

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())
    print(
        "Target distribution:",
        df["credit_risk"].value_counts().to_dict()
    )

    return df


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

class CreditFeatureEngineer(
    BaseEstimator,
    TransformerMixin
):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if isinstance(X, pd.DataFrame):

            df = X.copy()

        else:

            df = pd.DataFrame(
                X,
                columns=SELECTED_FEATURES
            )

        # Map original category codes if necessary

        for col, mapping in VALUE_MAPPINGS.items():

            if col in df.columns:

                if len(df[col].dropna()) > 0:

                    sample_val = df[col].dropna().iloc[0]

                    if sample_val in mapping:

                        df[col] = (
                            df[col]
                            .map(mapping)
                            .fillna(df[col])
                        )

        # Convert numerical columns

        for col in NUMERICAL_COLS_RAW:

            if col in df.columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        # Keep only the six selected features

        df = df[SELECTED_FEATURES].copy()

        return df


# ============================================================
# 6. PREPROCESSING PIPELINE
# ============================================================

def build_preprocessor():

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                ENGINEERED_NUMERICAL_COLS
            ),
            (
                "cat",
                categorical_transformer,
                ENGINEERED_CATEGORICAL_COLS
            )
        ],
        remainder="drop"
    )

    return preprocessor


# ============================================================
# 7. COMPLETE MACHINE LEARNING PIPELINE
# ============================================================

def build_full_pipeline(classifier):

    pipeline = Pipeline(
        steps=[
            (
                "feature_engineer",
                CreditFeatureEngineer()
            ),
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "classifier",
                classifier
            )
        ]
    )

    return pipeline


# ============================================================
# 8. EDA VISUALIZATION GENERATOR
# ============================================================

def generate_eda_plots(df, save_dir=None):

    if save_dir is None:

        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        save_dir = os.path.join(
            project_root,
            "eda_plots"
        )

    os.makedirs(save_dir, exist_ok=True)

    generated_plots = []

    plt.style.use("default")

    green = "#059669"
    red = "#dc2626"

    palette = {
        1: green,
        0: red
    }

    plot_df = df.copy()

    plot_df["Risk_Label"] = (
        plot_df["credit_risk"]
        .map({
            1: "Good Credit Risk",
            0: "Bad Credit Risk"
        })
    )

    # --------------------------------------------------------
    # 1. Credit Risk Distribution
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))

    counts = plot_df["Risk_Label"].value_counts()

    bars = ax.bar(
        counts.index,
        counts.values,
        color=[green, red]
    )

    for bar in bars:

        height = bar.get_height()

        ax.annotate(
            str(int(height)),
            (
                bar.get_x() + bar.get_width() / 2,
                height
            ),
            ha="center",
            xytext=(0, 5),
            textcoords="offset points"
        )

    ax.set_title("Credit Risk Distribution")
    ax.set_ylabel("Number of Applicants")

    plt.tight_layout()

    path = os.path.join(
        save_dir,
        "target_distribution.png"
    )

    plt.savefig(path, dpi=150)
    plt.close()

    generated_plots.append(path)

    # --------------------------------------------------------
    # 2. Age Distribution
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        data=plot_df,
        x="age",
        hue="Risk_Label",
        kde=True,
        common_norm=False,
        ax=ax
    )

    ax.set_title("Applicant Age Distribution")

    plt.tight_layout()

    path = os.path.join(
        save_dir,
        "age_distribution.png"
    )

    plt.savefig(path, dpi=150)
    plt.close()

    generated_plots.append(path)

    # --------------------------------------------------------
    # 3. Credit Amount Distribution
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        data=plot_df,
        x="credit_amount",
        hue="Risk_Label",
        kde=True,
        common_norm=False,
        ax=ax
    )

    ax.set_title("Credit Amount Distribution")

    plt.tight_layout()

    path = os.path.join(
        save_dir,
        "credit_amount_distribution.png"
    )

    plt.savefig(path, dpi=150)
    plt.close()

    generated_plots.append(path)

    # --------------------------------------------------------
    # 4. Credit History
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 6))

    sns.countplot(
        data=plot_df,
        y="credit_history",
        hue="Risk_Label",
        ax=ax
    )

    ax.set_title("Credit History by Risk")

    plt.tight_layout()

    path = os.path.join(
        save_dir,
        "credit_history_distribution.png"
    )

    plt.savefig(path, dpi=150)
    plt.close()

    generated_plots.append(path)

    # --------------------------------------------------------
    # 5. Savings Distribution
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.countplot(
        data=plot_df,
        x="savings_status",
        hue="Risk_Label",
        ax=ax
    )

    ax.set_title("Savings Status by Credit Risk")
    ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()

    path = os.path.join(
        save_dir,
        "savings_distribution.png"
    )

    plt.savefig(path, dpi=150)
    plt.close()

    generated_plots.append(path)

    # --------------------------------------------------------
    # 6. Loan Duration Distribution
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        data=plot_df,
        x="duration",
        hue="Risk_Label",
        kde=True,
        common_norm=False,
        ax=ax
    )

    ax.set_title("Loan Duration by Credit Risk")

    plt.tight_layout()

    path = os.path.join(
        save_dir,
        "credit_duration_distribution.png"
    )

    plt.savefig(path, dpi=150)
    plt.close()

    generated_plots.append(path)

    # --------------------------------------------------------
    # 7. Correlation Heatmap
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    numeric_df = df[
        NUMERICAL_COLS_RAW + ["credit_risk"]
    ].copy()

    correlation = numeric_df.corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Selected Numerical Features Correlation")

    plt.tight_layout()

    path = os.path.join(
        save_dir,
        "correlation_heatmap.png"
    )

    plt.savefig(path, dpi=150)
    plt.close()

    generated_plots.append(path)

    print(
        f"Generated {len(generated_plots)} EDA plots."
    )

    return generated_plots


# ============================================================
# 9. SELF TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING SIX-FEATURE DATA PREPROCESSING")
    print("=" * 60)

    df = load_data()

    X = df[SELECTED_FEATURES]
    y = df["credit_risk"]

    print("\nSelected features:")
    print(X.columns.tolist())

    print("\nFeature shape:", X.shape)

    print("\nTarget shape:", y.shape)

    engineered = CreditFeatureEngineer().fit_transform(X)

    print("\nEngineered feature shape:", engineered.shape)

    print("\nGenerating EDA plots...")

    plots = generate_eda_plots(df)

    print("Generated plots:", len(plots))

    print("\nPreprocessing test completed successfully.")