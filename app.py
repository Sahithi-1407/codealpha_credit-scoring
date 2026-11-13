
import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# FIN GUARD - CREDIT SCORING APPLICATION
# ============================================================

ROOT = os.path.dirname(os.path.abspath(__file__))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_preprocessing import load_data, VALUE_MAPPINGS, NUMERICAL_COLS_RAW
from src.predict import CreditRiskPredictor, get_sample_applicants


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FinGuard | Credit Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM FINTECH DESIGN
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #f4f7fb;
    color: #172b4d;
}

.main .block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    display: none;
}

h1, h2, h3 {
    color: #10233f !important;
}

.brand {
    font-size: 26px;
    font-weight: 900;
    color: #10233f;
    padding: 8px 0;
}

.brand span {
    color: #0f9b8e;
}

.hero {
    background: linear-gradient(120deg, #10233f, #164b66, #0f9b8e);
    padding: 35px;
    border-radius: 20px;
    color: white;
    margin: 15px 0 25px;
    animation: fadein .6s ease;
}

.hero h1 {
    color: white !important;
    font-size: 32px;
}

.hero p {
    color: #e0f2f1;
    font-size: 15px;
}

.card {
    background: white;
    border: 1px solid #e3eaf1;
    border-top: 4px solid #0f9b8e;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 5px 18px rgba(16,35,63,.06);
    transition: .25s;
    animation: fadein .5s ease;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 25px rgba(16,35,63,.12);
}

.card-label {
    font-size: 12px;
    color: #64748b;
    font-weight: 700;
    text-transform: uppercase;
}

.card-value {
    font-size: 28px;
    color: #10233f;
    font-weight: 850;
    margin-top: 8px;
}

.card-note {
    color: #0f9b8e;
    font-size: 12px;
}

.decision-good {
    background: linear-gradient(120deg,#087f5b,#16a085);
    color: white;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
}

.decision-bad {
    background: linear-gradient(120deg,#a83242,#dc5b5b);
    color: white;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(110deg,#10233f,#0f9b8e);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 45px;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    color: white;
    transform: translateY(-2px);
}

div[data-testid="stMetric"] {
    background: white;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #e3eaf1;
}

.disclaimer {
    background: #fff8e7;
    border-left: 5px solid #e3a52f;
    padding: 15px;
    border-radius: 8px;
    color: #76551a;
    margin-top: 25px;
}

div[data-testid="stRadio"] > div {
    gap: 8px;
}

@keyframes fadein {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media(prefers-reduced-motion:reduce) {
    * {
        animation: none !important;
        transition: none !important;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def get_dataset():
    return load_data()


@st.cache_data
def get_metrics():

    metrics_path = os.path.join(
        ROOT, "models", "metrics.json"
    )

    comparison_path = os.path.join(
        ROOT, "models", "model_comparison.json"
    )

    metrics = None
    comparison = None

    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as file:
            metrics = json.load(file)

    if os.path.exists(comparison_path):
        with open(comparison_path, encoding="utf-8") as file:
            comparison = json.load(file)

    return metrics, comparison


# ============================================================
# TOP NAVIGATION
# ============================================================

left, middle, right = st.columns([1, 4, 1.5])

with left:
    st.markdown(
        '<div class="brand">Fin<span>Guard</span> ✦</div>',
        unsafe_allow_html=True
    )

with middle:
    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📁 Dataset",
            "📊 Analytics",
            "📈 Models",
            "🎯 Prediction"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

with right:
    selected_model = st.selectbox(
        "Model",
        [
            "Random Forest",
            "Decision Tree",
            "Logistic Regression"
        ],
        label_visibility="collapsed"
    )


model_files = {
    "Random Forest": "random_forest.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Logistic Regression": "logistic_regression.joblib"
}


st.markdown("""
<div style="
    background:white;
    border:1px solid #e3eaf1;
    border-radius:12px;
    padding:12px 18px;
    margin-bottom:20px;
">
    <span style="color:#64748b;font-size:12px;">
        SMARTER CREDIT INSIGHTS
    </span>
    <span style="float:right;color:#0f9b8e;font-size:12px;">
        EDUCATIONAL DEMO
    </span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# HOME DASHBOARD
# ============================================================

if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <h1>🏦 Welcome to FinGuard</h1>
        <p>Intelligent Credit Risk Analysis & Financial Decision Support</p>
    </div>
    """, unsafe_allow_html=True)

    a, b, c, d = st.columns(4)

    cards = [
        ("TOTAL APPLICATIONS", "1,000", "German Credit Dataset"),
        ("GOOD CREDIT", "70%", "700 applicants"),
        ("BAD CREDIT", "30%", "300 applicants"),
        ("MODEL ROC-AUC", "0.788", "Previously observed score")
    ]

    for col, item in zip([a, b, c, d], cards):

        with col:
            st.markdown(f"""
            <div class="card">
                <div class="card-label">{item[0]}</div>
                <div class="card-value">{item[1]}</div>
                <div class="card-note">{item[2]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### About FinGuard")

    st.write("""
    FinGuard is a machine learning-based credit scoring application.
    It analyzes applicant financial information and estimates the
    likelihood of good or bad credit outcomes.

    The application uses three machine learning algorithms and
    provides an easy-to-understand credit risk assessment.
    """)

    st.markdown("### Machine Learning Algorithms")

    x, y, z = st.columns(3)

    x.info("**Logistic Regression**\n\nInterpretable classification baseline.")

    y.info("**Decision Tree**\n\nRule-based credit classification.")

    z.info("**Random Forest**\n\nEnsemble-based credit risk prediction.")

    st.markdown("""
    <div class="disclaimer">
        <b>Educational Disclaimer:</b> This is an academic prototype,
        not a real-world lending or financial decision system.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# DATASET OVERVIEW
# ============================================================

elif page == "📁 Dataset":

    st.markdown("""
    <div class="hero">
        <h1>📁 Dataset Overview</h1>
        <p>Explore the German Credit dataset used for model training.</p>
    </div>
    """, unsafe_allow_html=True)

    try:

        df = get_dataset()

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Records", len(df))
        c2.metric("Total Columns", len(df.columns))
        c3.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        st.markdown("### Dataset Preview")

        if "credit_risk" in df.columns:

            choice = st.selectbox(
                "Filter Credit Risk",
                ["All", "Good Credit", "Bad Credit"]
            )

            if choice == "Good Credit":
                df = df[df["credit_risk"] == 1]

            elif choice == "Bad Credit":
                df = df[df["credit_risk"] == 0]

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

        st.markdown("### Numerical Summary")

        st.dataframe(
            df.describe().T,
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Unable to load dataset: {e}")


# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

elif page == "📊 Analytics":

    st.markdown("""
    <div class="hero">
        <h1>📊 Exploratory Data Analysis</h1>
        <p>Understand applicant characteristics and credit risk patterns.</p>
    </div>
    """, unsafe_allow_html=True)

    try:

        df = get_dataset()

        st.markdown("### Credit Risk Distribution")

        if "credit_risk" in df.columns:

            counts = df["credit_risk"].value_counts().sort_index()

            st.bar_chart(counts)

            st.write(
                "1 = Good Credit Risk | 0 = Bad Credit Risk"
            )

        st.markdown("### Numerical Feature Distributions")

        numerical_cols = [
            col for col in NUMERICAL_COLS_RAW
            if col in df.columns
        ]

        if numerical_cols:

            selected_feature = st.selectbox(
                "Choose a numerical feature",
                numerical_cols
            )

            st.bar_chart(
                df[selected_feature].value_counts().sort_index()
            )

        st.markdown("### Correlation Matrix")

        numeric_df = df.select_dtypes(include=np.number)

        if not numeric_df.empty:

            st.dataframe(
                numeric_df.corr().round(2),
                use_container_width=True
            )

        st.markdown("### Existing EDA Visualizations")

        eda_folder = os.path.join(ROOT, "eda_plots")

        if os.path.isdir(eda_folder):

            images = [
                filename
                for filename in os.listdir(eda_folder)
                if filename.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                )
            ]

            if images:

                for image_name in images:

                    st.image(
                        os.path.join(eda_folder, image_name),
                        caption=image_name,
                        use_container_width=True
                    )

            else:
                st.info("No saved EDA images found.")

        else:
            st.info("The eda_plots folder was not found.")

    except Exception as e:
        st.error(f"Unable to load analytics: {e}")


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📈 Models":

    st.markdown("""
    <div class="hero">
        <h1>📈 Model Performance</h1>
        <p>Compare machine learning algorithms and their evaluation metrics.</p>
    </div>
    """, unsafe_allow_html=True)

    metrics, comparison = get_metrics()

    if metrics:

        rows = []

        for name, result in metrics.items():

            rows.append({
                "Model": name,

                "Accuracy": round(
                    result.get("test_accuracy", 0) * 100, 2
                ),

                "ROC-AUC": round(
                    result.get("test_roc_auc", 0), 4
                ),

                "Bad Recall": round(
                    result.get("recall_bad", 0) * 100, 2
                ),

                "Good Recall": round(
                    result.get("recall_good", 0) * 100, 2
                ),

                "Financial Cost": result.get(
                    "total_financial_cost", "N/A"
                )
            })

        result_df = pd.DataFrame(rows)

        st.markdown("### Model Comparison Table")

        st.dataframe(
            result_df,
            use_container_width=True
        )

        st.markdown("### Accuracy Comparison")

        st.bar_chart(
            result_df.set_index("Model")["Accuracy"]
        )

        st.markdown("### ROC-AUC Comparison")

        st.bar_chart(
            result_df.set_index("Model")["ROC-AUC"]
        )

    else:

        st.warning(
            "Model metrics not found. Run python src/train.py first."
        )

    st.markdown("### Saved Model Visualizations")

    for filename in [
        "roc_curve_comparison.png",
        "confusion_matrices.png",
        "feature_importance.png"
    ]:

        path = os.path.join(ROOT, "models", filename)

        if os.path.exists(path):

            st.image(
                path,
                caption=filename,
                use_container_width=True
            )


# ============================================================
# CREDIT RISK PREDICTION - SIX INPUT FEATURES
# ============================================================

elif page == "🎯 Prediction":

    st.markdown("""
    <div class="hero">
        <h1>🎯 Credit Risk Prediction</h1>
        <p>Enter six simple financial details to estimate creditworthiness.</p>
    </div>
    """, unsafe_allow_html=True)

    try:

        predictor = CreditRiskPredictor(
            model_files[selected_model]
        )

        sample_profiles = get_sample_applicants()

        preset_names = list(sample_profiles.keys())

        st.markdown("### Choose an Applicant Profile")

        preset = st.selectbox(
            "Select a sample profile or enter your own details",
            ["Custom Applicant"] + preset_names
        )

        # Initialize default applicant
        if "applicant_values" not in st.session_state:

            st.session_state.applicant_values = (
                sample_profiles[preset_names[0]].copy()
            )

        # Load selected preset
        if preset in sample_profiles:

            if st.session_state.get("last_preset") != preset:

                st.session_state.applicant_values = (
                    sample_profiles[preset].copy()
                )

                st.session_state.last_preset = preset

        else:

            st.session_state.last_preset = "Custom Applicant"

        vals = st.session_state.applicant_values

        st.markdown("### Applicant Financial Information")

        st.info(
            "Enter the six details below. "
            "These are the only features used by the trained model."
        )

        with st.form("credit_prediction_form"):

            payload = {}

            col1, col2 = st.columns(2)

            with col1:

                payload["age"] = st.number_input(
                    "Applicant Age (years)",
                    min_value=18,
                    max_value=90,
                    value=int(vals.get("age", 30)),
                    step=1
                )

                checking_options = list(
                    VALUE_MAPPINGS["checking_status"].values()
                )

                checking_default = vals.get(
                    "checking_status",
                    checking_options[0]
                )

                if checking_default not in checking_options:
                    checking_default = checking_options[0]

                payload["checking_status"] = st.selectbox(
                    "Bank Account Balance",
                    checking_options,
                    index=checking_options.index(checking_default)
                )

                history_options = list(
                    VALUE_MAPPINGS["credit_history"].values()
                )

                history_default = vals.get(
                    "credit_history",
                    history_options[0]
                )

                if history_default not in history_options:
                    history_default = history_options[0]

                payload["credit_history"] = st.selectbox(
                    "Previous Loan Repayment History",
                    history_options,
                    index=history_options.index(history_default)
                )

            with col2:

                payload["credit_amount"] = st.number_input(
                    "Loan Amount (DM)",
                    min_value=250,
                    max_value=25000,
                    value=int(vals.get("credit_amount", 1500)),
                    step=100
                )

                payload["duration"] = st.slider(
                    "Repayment Period (Months)",
                    min_value=4,
                    max_value=72,
                    value=int(vals.get("duration", 12))
                )

                savings_options = list(
                    VALUE_MAPPINGS["savings_status"].values()
                )

                savings_default = vals.get(
                    "savings_status",
                    savings_options[0]
                )

                if savings_default not in savings_options:
                    savings_default = savings_options[0]

                payload["savings_status"] = st.selectbox(
                    "Savings Level",
                    savings_options,
                    index=savings_options.index(savings_default)
                )

            submitted = st.form_submit_button(
                "⚡ Calculate Credit Risk",
                use_container_width=True
            )

        # Run prediction
        if submitted:

            assessment = predictor.assess_applicant(payload)

            st.session_state.last_assessment = assessment

            st.session_state.last_assessment_model = selected_model

            st.session_state.applicant_values = payload.copy()

            st.session_state.last_preset = "Custom Applicant"

        # Display latest assessment
        assessment = st.session_state.get("last_assessment")

        if (
            assessment is not None
            and st.session_state.get("last_assessment_model")
            == selected_model
        ):

            st.markdown("---")

            st.markdown("### 📊 Credit Assessment Result")

            if assessment["prediction"] == 1:

                st.markdown(
                    f"""
                    <div class="decision-good">
                        <h2 style="color:white!important;">
                            ✅ {assessment['decision']}
                        </h2>
                        Applicant classified as
                        {assessment['risk_category']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="decision-bad">
                        <h2 style="color:white!important;">
                            ❌ {assessment['decision']}
                        </h2>
                        Applicant classified as
                        {assessment['risk_category']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            a, b, c, d = st.columns(4)

            a.metric(
                "Illustrative Credit Score",
                f"{assessment['credit_score']} / 850"
            )

            b.metric(
                "Good Credit Probability",
                f"{assessment['prob_good_pct']}%"
            )

            c.metric(
                "Default Probability",
                f"{assessment['prob_bad_pct']}%"
            )

            d.metric(
                "Monthly Repayment",
                f"{assessment['monthly_installment_dm']:,.2f} DM"
            )

            st.caption(
                "The 300–850 score is an illustrative model-derived "
                "score, not an official FICO or credit-bureau score."
            )

            st.markdown("### Probability Analysis")

            col1, col2 = st.columns(2)

            with col1:

                st.write("Good Credit Probability")

                st.progress(
                    float(assessment["prob_good_pct"]) / 100
                )

            with col2:

                st.write("Default Probability")

                st.progress(
                    float(assessment["prob_bad_pct"]) / 100
                )

            f1, f2 = st.columns(2)

            with f1:

                st.success("Positive Credit Factors")

                for factor in assessment.get(
                    "positive_factors", []
                ):

                    st.write("✓", factor)

            with f2:

                st.error("Risk Factors")

                for factor in assessment.get(
                    "risk_factors", []
                ):

                    st.write("•", factor)

    except Exception as e:

        st.error(f"Prediction error: {e}")

        st.info(
            "Check that the six-feature trained model files "
            "exist inside the models folder."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#64748b;padding:10px;">
        <b style="color:#10233f;">FinGuard</b> |
        Machine Learning Credit Risk Assessment |
        Academic Project
        <br><br>
        <small>
        This application is an educational prototype and
        must not be used for actual lending decisions.
        </small>
    </div>
    """,
    unsafe_allow_html=True
)
