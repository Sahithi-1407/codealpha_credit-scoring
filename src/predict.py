
# ============================================================
# Credit_Scoring_Project/src/predict.py
# Six-feature Credit Risk Prediction Module
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import CreditFeatureEngineer

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "best_credit_model.joblib")

# These must match SELECTED_FEATURES in data_preprocessing.py
FEATURE_COLUMNS = [
    "age",
    "checking_status",
    "credit_history",
    "credit_amount",
    "duration",
    "savings_status"
]


class CreditRiskPredictor:
    """Loads a trained model and predicts credit risk from six inputs."""

    def __init__(self, model_name_or_path=None):

        if model_name_or_path is None:
            model_path = DEFAULT_MODEL_PATH

        elif os.path.isabs(str(model_name_or_path)) or os.path.exists(
            str(model_name_or_path)
        ):
            model_path = model_name_or_path

        else:
            name_map = {
                "decision tree": "decision_tree.joblib",
                "random forest": "random_forest.joblib",
                "logistic regression": "logistic_regression.joblib",
                "best": "best_credit_model.joblib"
            }

            cleaned = str(model_name_or_path).strip().lower()
            filename = name_map.get(cleaned, str(model_name_or_path))

            if not filename.endswith(".joblib"):
                filename += ".joblib"

            model_path = os.path.join(MODELS_DIR, filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                "Please run python src/train.py first."
            )

        self.model_path = model_path
        self.model_name = (
            os.path.basename(model_path)
            .replace(".joblib", "")
            .replace("_", " ")
            .title()
        )

        self.pipeline = joblib.load(model_path)
        self.feature_columns = FEATURE_COLUMNS

    def _prepare_dataframe(self, data):

        if isinstance(data, dict):
            df = pd.DataFrame([data])

        elif isinstance(data, list):
            df = pd.DataFrame(data)

        elif isinstance(data, pd.DataFrame):
            df = data.copy()

        else:
            raise TypeError(
                "Input must be a dictionary, list of dictionaries, "
                "or pandas DataFrame."
            )

        missing = [
            col for col in self.feature_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                "Missing required applicant information: "
                + ", ".join(missing)
            )

        df = df[self.feature_columns].copy()

        numeric_columns = ["age", "credit_amount", "duration"]

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="raise")

        return df

    def predict(self, data):
        """1 = Good Credit Risk, 0 = Bad Credit Risk."""

        df = self._prepare_dataframe(data)
        return self.pipeline.predict(df)

    def predict_proba(self, data):
        """Returns probabilities in the model's class order."""

        df = self._prepare_dataframe(data)
        return self.pipeline.predict_proba(df)

    def assess_applicant(self, applicant_data):

        df = self._prepare_dataframe(applicant_data)

        probabilities = self.pipeline.predict_proba(df)[0]
        classes = list(self.pipeline.classes_)

        prob_good = float(probabilities[classes.index(1)])
        prob_bad = float(probabilities[classes.index(0)])

        prediction = int(self.pipeline.predict(df)[0])

        # Illustrative score derived from model probability.
        # This is NOT an official FICO or bureau credit score.
        credit_score = int(round(300 + prob_good * 550))
        credit_score = max(300, min(850, credit_score))

        if prediction == 1 and prob_good >= 0.70:
            risk_category = "Good Credit Risk"
            decision = "APPROVED (Low Risk)"
            badge_color = "green"
            risk_tier = "Prime / Tier-1"

        elif prediction == 1 and prob_good >= 0.50:
            risk_category = "Good Credit Risk"
            decision = "CONDITIONAL APPROVAL (Moderate Risk)"
            badge_color = "amber"
            risk_tier = "Near-Prime / Tier-2"

        else:
            risk_category = "Bad Credit Risk"
            decision = "DECLINED (High Risk of Default)"
            badge_color = "red"
            risk_tier = "Subprime / High Risk"

        positive_factors = []
        risk_factors = []

        checking = str(applicant_data["checking_status"])
        history = str(applicant_data["credit_history"])
        savings = str(applicant_data["savings_status"])

        age = float(applicant_data["age"])
        duration = float(applicant_data["duration"])
        amount = float(applicant_data["credit_amount"])

        # Checking account analysis
        if "< 0" in checking or "A11" in checking:
            risk_factors.append(
                "Negative checking balance indicates possible cash-flow stress."
            )

        elif "No checking" in checking or "A14" in checking:
            positive_factors.append(
                "No active checking-account overdraft is indicated."
            )

        elif ">= 200" in checking or "A13" in checking:
            positive_factors.append(
                "Higher checking-account balance indicates stronger liquidity."
            )

        # Credit history analysis
        if "Critical" in history or "A34" in history:
            risk_factors.append(
                "Critical credit history indicates previous repayment concerns."
            )

        elif (
            "paid back duly" in history
            or "A30" in history
            or "A31" in history
            or "A32" in history
        ):
            positive_factors.append(
                "Previous credit history indicates timely repayments."
            )

        # Savings analysis
        if "< 100" in savings or "A61" in savings:
            risk_factors.append(
                "Low savings may provide limited protection against financial shocks."
            )

        elif ">= 1000" in savings or "A64" in savings:
            positive_factors.append(
                "Higher savings provide a stronger financial buffer."
            )

        # Loan duration analysis
        if duration > 36:
            risk_factors.append(
                f"Long repayment period of {duration:.0f} months."
            )

        elif 0 < duration <= 12:
            positive_factors.append(
                f"Short repayment period of {duration:.0f} months."
            )

        # Loan amount analysis
        if amount > 8000:
            risk_factors.append(
                f"High requested loan amount: {amount:,.0f} DM."
            )

        elif 0 < amount <= 2000:
            positive_factors.append(
                f"Relatively modest loan amount: {amount:,.0f} DM."
            )

        # Age context
        if age < 25:
            risk_factors.append(
                "Younger applicant; repayment capacity should be assessed carefully."
            )

        elif age >= 40:
            positive_factors.append(
                "Applicant is in a more established age group."
            )

        monthly_installment = round(
            amount / max(duration, 1), 2
        )

        return {
            "prediction": prediction,
            "risk_category": risk_category,
            "decision": decision,
            "badge_color": badge_color,
            "risk_tier": risk_tier,
            "prob_good_pct": round(prob_good * 100, 2),
            "prob_bad_pct": round(prob_bad * 100, 2),
            "credit_score": credit_score,
            "monthly_installment_dm": monthly_installment,
            "positive_factors": positive_factors or [
                "No additional positive signals identified."
            ],
            "risk_factors": risk_factors or [
                "No additional risk signals identified."
            ],
            "model_path": self.model_path
        }


def get_sample_applicants():

    return {
        "Prime Professional (Low Risk)": {
            "age": 45,
            "checking_status": ">= 200 DM / Salary assignments >= 1 yr",
            "credit_history": "All credits at this bank paid back duly",
            "credit_amount": 1500,
            "duration": 12,
            "savings_status": ">= 1000 DM"
        },

        "Subprime Borrower (High Risk)": {
            "age": 23,
            "checking_status": "< 0 DM",
            "credit_history": "Critical account / Other credits existing",
            "credit_amount": 9500,
            "duration": 48,
            "savings_status": "< 100 DM"
        },

        "Borderline Applicant (Moderate Risk)": {
            "age": 31,
            "checking_status": "0 to < 200 DM",
            "credit_history": "Existing credits paid back duly till now",
            "credit_amount": 3800,
            "duration": 24,
            "savings_status": "100 to < 500 DM"
        }
    }


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("CREDIT RISK PREDICTOR - SIX FEATURE TEST")
    print("=" * 60)

    predictor = CreditRiskPredictor()
    samples = get_sample_applicants()

    for title, applicant in samples.items():

        report = predictor.assess_applicant(applicant)

        print(f"\nApplicant: {title}")
        print("-" * 45)
        print("Decision:", report["decision"])
        print("Risk:", report["risk_category"])
        print("Illustrative Score:", report["credit_score"], "/ 850")
        print("Good Credit Probability:", report["prob_good_pct"], "%")
        print("Bad Credit Probability:", report["prob_bad_pct"], "%")

        print("Positive Factors:")
        for factor in report["positive_factors"]:
            print(" +", factor)

        print("Risk Factors:")
        for factor in report["risk_factors"]:
            print(" -", factor)

    print("\nPredictor test completed.")
