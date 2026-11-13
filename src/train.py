# ============================================================
# Credit_Scoring_Project/src/train.py
#
# PURPOSE:
#   Main model training and evaluation script:
#     1. Loads German Credit data with correct labels (1=Good, 0=Bad).
#     2. Performs stratified train-test split (80/20).
#     3. Builds end-to-end pipelines (Feature Engineering + Preprocessing + Model).
#     4. Performs 5-fold Stratified Cross-Validation for each model.
#     5. Trains & compares:
#        - Logistic Regression
#        - Decision Tree Classifier
#        - Random Forest Classifier
#     6. Evaluates on test set: Accuracy, Precision, Recall, F1, ROC-AUC.
#     7. Computes banking financial cost matrix:
#        Cost = 5 * (Bad predicted as Good) + 1 * (Good predicted as Bad).
#     8. Selects optimal model considering cost and bad-credit recall (not accuracy alone).
#     9. Saves trained models, comparison JSON, ROC curves, and confusion matrices.
# ============================================================

import os
import sys
import json
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_credit"))

# Add project root to sys.path so 'src' can be imported reliably
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from src.data_preprocessing import load_data, build_full_pipeline, CreditFeatureEngineer

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "eda_plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1. Model Definitions
# ─────────────────────────────────────────────────────────────

def get_candidate_models():
    """
    Returns a dictionary of candidate models configured with
    balanced class weights to address the 70% Good / 30% Bad imbalance.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        )
    }


# ─────────────────────────────────────────────────────────────
# 2. Banking Financial Cost Matrix Function
# ─────────────────────────────────────────────────────────────

def calculate_credit_cost(y_true, y_pred):
    """
    Calculate the domain-standard German Credit Financial Risk Cost:
      - False Positive (Actual Bad=0 predicted as Good=1):
        Granting a loan to a defaulting borrower incurs high financial loss.
        Standard penalty = 5 units per default.
      - False Negative (Actual Good=1 predicted as Bad=0):
        Rejecting a creditworthy borrower loses customer interest revenue.
        Standard penalty = 1 unit per lost business opportunity.
      - True Positive (1->1) and True Negative (0->0): Cost = 0.

    Cost Matrix:
                    Predicted Bad (0)   Predicted Good (1)
      Actual Bad (0)         0                   5 (Costly Default)
      Actual Good (1)        1 (Lost Profit)      0

    Returns:
        tuple: (total_cost, fp_count, fn_count, avg_cost_per_applicant)
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm[0,0] = TN (Actual 0, Pred 0)
    # cm[0,1] = FP (Actual 0, Pred 1) -> Cost 5
    # cm[1,0] = FN (Actual 1, Pred 0) -> Cost 1
    # cm[1,1] = TP (Actual 1, Pred 1) -> Cost 0
    bad_as_good = int(cm[0, 1])   # Defaulters given loans
    good_as_bad = int(cm[1, 0])   # Creditworthy denied
    total_cost = (bad_as_good * 5) + (good_as_bad * 1)
    avg_cost = round(total_cost / len(y_true), 4)

    return total_cost, bad_as_good, good_as_bad, avg_cost


# ─────────────────────────────────────────────────────────────
# 3. Model Training & Evaluation Engine
# ─────────────────────────────────────────────────────────────

def train_and_evaluate_all():
    """
    Execute the complete training, cross-validation, evaluation,
    plotting, and model selection pipeline.
    """
    print("\n" + "="*70)
    print("  CREDIT SCORING MODEL — TRAINING & COMPARISON PIPELINE")
    print("="*70 + "\n")

    # ── Step 1: Load Dataset ──
    df = load_data()
    X = df.drop(columns=["credit_risk"])
    y = df["credit_risk"]

    # ── Step 2: Stratified Train-Test Split (80% Train, 20% Test) ──
    # Stratification ensures identical Good/Bad ratio in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"\n📊 Data Partitioning:")
    print(f"   • Training samples : {len(X_train)} ({y_train.value_counts().to_dict()})")
    print(f"   • Test samples     : {len(X_test)} ({y_test.value_counts().to_dict()})")

    candidate_models = get_candidate_models()
    cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    fitted_pipelines = {}
    test_predictions = {}
    test_probabilities = {}

    print("\n" + "─"*70)
    print("  PHASE 1: 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("─"*70)

    for name, model in candidate_models.items():
        print(f"\n⚙️  Running 5-Fold Cross-Validation for: {name}...")
        pipeline = build_full_pipeline(model)

        cv_scores = cross_validate(
            pipeline, X_train, y_train,
            cv=cv_kfold,
            scoring=["accuracy", "roc_auc", "f1", "recall"],
            return_train_score=False
        )

        cv_acc_mean = float(np.mean(cv_scores["test_accuracy"]))
        cv_acc_std  = float(np.std(cv_scores["test_accuracy"]))
        cv_auc_mean = float(np.mean(cv_scores["test_roc_auc"]))
        cv_auc_std  = float(np.std(cv_scores["test_roc_auc"]))
        cv_f1_mean  = float(np.mean(cv_scores["test_f1"]))

        print(f"   ✅ CV Accuracy : {cv_acc_mean*100:.2f}% (±{cv_acc_std*100:.2f}%)")
        print(f"   ✅ CV ROC-AUC  : {cv_auc_mean:.4f} (±{cv_auc_std:.4f})")
        print(f"   ✅ CV F1-Score : {cv_f1_mean:.4f}")

        # ── Phase 2: Fit Pipeline on Full Training Set ──
        print(f"   Fitting {name} on full training set ({len(X_train)} samples)...")
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # ── Phase 3: Evaluate on Held-Out Test Set ──
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        test_predictions[name] = y_pred
        test_probabilities[name] = y_proba

        acc = float(accuracy_score(y_test, y_pred))
        prec_good = float(precision_score(y_test, y_pred, pos_label=1))
        rec_good = float(recall_score(y_test, y_pred, pos_label=1))
        f1_good = float(f1_score(y_test, y_pred, pos_label=1))

        # Critical Banking Metric: Recall on BAD Risk (Class 0)
        prec_bad = float(precision_score(y_test, y_pred, pos_label=0))
        rec_bad = float(recall_score(y_test, y_pred, pos_label=0))
        f1_bad = float(f1_score(y_test, y_pred, pos_label=0))

        roc_auc = float(roc_auc_score(y_test, y_proba))
        total_cost, bad_as_good, good_as_bad, avg_cost = calculate_credit_cost(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

        results[name] = {
            "cv_accuracy_mean": round(cv_acc_mean, 4),
            "cv_accuracy_std": round(cv_acc_std, 4),
            "cv_roc_auc_mean": round(cv_auc_mean, 4),
            "cv_roc_auc_std": round(cv_auc_std, 4),
            "test_accuracy": round(acc, 4),
            "test_roc_auc": round(roc_auc, 4),
            "precision_good": round(prec_good, 4),
            "recall_good": round(rec_good, 4),
            "f1_good": round(f1_good, 4),
            "precision_bad": round(prec_bad, 4),
            "recall_bad": round(rec_bad, 4),      # High priority in banking
            "f1_bad": round(f1_bad, 4),
            "bad_as_good_defaults": bad_as_good, # Type II errors (costly)
            "good_as_bad_denials": good_as_bad,   # Type I errors
            "total_financial_cost": total_cost,
            "avg_cost_per_applicant": avg_cost,
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_pred, target_names=["Bad Credit", "Good Credit"], output_dict=True)
        }

    # ── Display Summary Comparison Table ──
    print("\n" + "="*70)
    print("  MODEL PERFORMANCE COMPARISON (TEST SET: 200 APPLICANTS)")
    print("="*70)
    print(f"{'Model':<22} | {'Accuracy':<9} | {'ROC-AUC':<8} | {'Bad Recall':<10} | {'Defaults':<8} | {'Cost':<6}")
    print("─"*70)
    for name, m in results.items():
        print(f"{name:<22} | {m['test_accuracy']*100:>7.2f}% | {m['test_roc_auc']:>8.4f} | {m['recall_bad']*100:>9.2f}% | {m['bad_as_good_defaults']:>8} | {m['total_financial_cost']:>6}")
    print("─"*70)

    # ── Step 4: Multi-Criteria Model Selection (Cost & Bad Recall Priority) ──
    # Requirement: Do not select best model based on accuracy alone.
    # In retail credit risk, minimizing expected financial loss (Cost) and maximizing
    # Bad Risk Recall (catching defaulters) while preserving high ROC-AUC is paramount.
    # We rank models primarily by lowest financial cost, then highest ROC-AUC.
    ranked_models = sorted(
        results.keys(),
        key=lambda k: (results[k]["total_financial_cost"], -results[k]["test_roc_auc"], -results[k]["recall_bad"])
    )
    best_model_name = ranked_models[0]
    best_pipeline = fitted_pipelines[best_model_name]

    print(f"\n🏆 WINNING MODEL SELECTED: {best_model_name}")
    print(f"   • Selection Justification:")
    print(f"     1. Lowest Financial Default Cost: {results[best_model_name]['total_financial_cost']} units (vs others up to {max(m['total_financial_cost'] for m in results.values())})")
    print(f"     2. High Bad-Risk Defaulter Recall: {results[best_model_name]['recall_bad']*100:.2f}%")
    print(f"     3. Strong Discriminative Ability (ROC-AUC): {results[best_model_name]['test_roc_auc']:.4f}")
    print(f"     4. Generalization Accuracy: {results[best_model_name]['test_accuracy']*100:.2f}%")

    # ── Step 5: Save Models & Artifacts ──
    print("\n💾 Saving models and evaluation artifacts...")

    # Save the selected best pipeline
    best_model_path = os.path.join(MODELS_DIR, "best_credit_model.joblib")
    joblib.dump(best_pipeline, best_model_path)
    print(f"   ✅ Best model saved → {best_model_path}")

    # Save individual candidate pipelines for benchmarking
    for name, pipe in fitted_pipelines.items():
        filename = name.lower().replace(" ", "_") + ".joblib"
        save_path = os.path.join(MODELS_DIR, filename)
        joblib.dump(pipe, save_path)

    # Save metrics JSON
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"   ✅ Metrics saved → {metrics_path}")

    # Save model comparison summary JSON
    comparison_summary = {
        "best_model": best_model_name,
        "selection_criteria": "Lowest Financial Cost (5*FP + 1*FN) combined with Highest Bad Risk Recall & ROC-AUC",
        "ranking": ranked_models,
        "test_sample_count": len(y_test),
        "good_credit_count": int(y_test.sum()),
        "bad_credit_count": int((1 - y_test).sum()),
        "summary_table": [
            {
                "model": name,
                "accuracy": results[name]["test_accuracy"],
                "roc_auc": results[name]["test_roc_auc"],
                "bad_credit_recall": results[name]["recall_bad"],
                "good_credit_recall": results[name]["recall_good"],
                "financial_cost": results[name]["total_financial_cost"],
                "avg_cost_per_applicant": results[name]["avg_cost_per_applicant"]
            }
            for name in ranked_models
        ]
    }
    comp_path = os.path.join(MODELS_DIR, "model_comparison.json")
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=4)
    print(f"   ✅ Model comparison summary saved → {comp_path}")

    # ── Step 6: Plot ROC Curves Comparison ──
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"Logistic Regression": "#1e3a8a", "Decision Tree": "#f59e0b", "Random Forest": "#059669"}
    for name in candidate_models.keys():
        fpr, tpr, _ = roc_curve(y_test, test_probabilities[name])
        auc = results[name]["test_roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=colors.get(name, "#333"), linewidth=2.2)

    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", label="Random Chance (AUC = 0.5000)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    roc_path_models = os.path.join(MODELS_DIR, "roc_curve_comparison.png")
    roc_path_eda = os.path.join(PLOTS_DIR, "roc_curve_comparison.png")
    plt.savefig(roc_path_models, dpi=150)
    plt.savefig(roc_path_eda, dpi=150)
    plt.close()
    print(f"   📊 ROC curve comparison saved → {roc_path_models}")

    # ── Step 7: Plot Confusion Matrices Comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for idx, (name, ax) in enumerate(zip(candidate_models.keys(), axes)):
        cm_arr = np.array(results[name]["confusion_matrix"])
        sns.heatmap(
            cm_arr, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred Bad", "Pred Good"],
            yticklabels=["Actual Bad", "Actual Good"],
            cbar=False, ax=ax, annot_kws={"size": 13, "weight": "bold"}
        )
        cost = results[name]["total_financial_cost"]
        rec_bad = results[name]["recall_bad"] * 100
        ax.set_title(f"{name}\nCost: {cost} | Bad Recall: {rec_bad:.1f}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        if idx == 0:
            ax.set_ylabel("True Label", fontsize=10)
        else:
            ax.set_ylabel("")

    plt.suptitle("Confusion Matrix Comparison Across Models (Test Set)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    cm_path = os.path.join(MODELS_DIR, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Confusion matrices saved → {cm_path}")

    # ── Step 8: Plot Feature Importance (Random Forest & Logistic Regression) ──
    try:
        plot_feature_importances(fitted_pipelines, X_train)
    except Exception as e:
        print(f"   ⚠️ Could not plot feature importances: {e}")

    print("\n" + "="*70)
    print("  ALL MODELS SUCCESSFULLY TRAINED, EVALUATED, AND SAVED!")
    print("="*70 + "\n")

    return best_model_name, results


def plot_feature_importances(fitted_pipelines, X_sample):
    """
    Extract and plot feature importances from Random Forest and coefficients
    from Logistic Regression to provide credit explainability.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Extract transformed feature names from ColumnTransformer
    rf_pipe = fitted_pipelines.get("Random Forest")
    lr_pipe = fitted_pipelines.get("Logistic Regression")

    if rf_pipe is not None:
        preprocessor = rf_pipe.named_steps["preprocessor"]
        engineer = rf_pipe.named_steps["feature_engineer"]
        X_eng = engineer.transform(X_sample)

        # Get feature names from one-hot encoder and numerical columns
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_features = list(cat_encoder.get_feature_names_out(preprocessor.transformers[1][2]))
        num_features = list(preprocessor.transformers[0][2])
        all_features = num_features + cat_features

        rf_clf = rf_pipe.named_steps["classifier"]
        importances = rf_clf.feature_importances_

        # Get top 15 features
        top_indices = np.argsort(importances)[-15:]
        top_features = [all_features[i] for i in top_indices]
        top_values = importances[top_indices]

        # Clean feature names for clean presentation
        clean_names = [f.replace("_", " ").replace("checking status", "Check:").replace("credit history", "Hist:") for f in top_features]

        ax1.barh(range(len(top_indices)), top_values, color="#059669", edgecolor="#1e293b", linewidth=0.8)
        ax1.set_yticks(range(len(top_indices)))
        ax1.set_yticklabels(clean_names, fontsize=9)
        ax1.set_title("Random Forest — Top 15 Feature Importances", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Relative Importance (MDI)", fontsize=10)

    if lr_pipe is not None:
        lr_clf = lr_pipe.named_steps["classifier"]
        coefs = lr_clf.coef_[0]

        top_indices_lr = np.argsort(np.abs(coefs))[-15:]
        top_features_lr = [all_features[i] for i in top_indices_lr]
        top_values_lr = coefs[top_indices_lr]

        clean_names_lr = [f.replace("_", " ").replace("checking status", "Check:").replace("credit history", "Hist:") for f in top_features_lr]
        colors = ["#059669" if v > 0 else "#dc2626" for v in top_values_lr]

        ax2.barh(range(len(top_indices_lr)), top_values_lr, color=colors, edgecolor="#1e293b", linewidth=0.8)
        ax2.set_yticks(range(len(top_indices_lr)))
        ax2.set_yticklabels(clean_names_lr, fontsize=9)
        ax2.set_title("Logistic Regression — Top 15 Coefficients\n(Green = Pro-Good, Red = Pro-Bad)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Coefficient Magnitude", fontsize=10)

    plt.suptitle("Model Interpretability & Feature Significance", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fi_path = os.path.join(MODELS_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Feature importance analysis saved → {fi_path}")


# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_and_evaluate_all()

