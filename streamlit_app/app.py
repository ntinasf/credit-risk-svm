"""
Credit Risk Classification - Streamlit Demo

An interactive demo for the credit risk ensemble classifier.
Users can test with random samples or input custom values.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from scorer import EnsembleScorer

# Page configuration
st.set_page_config(
    page_title="Credit Risk Classifier",
    page_icon="💳",
    layout="wide"
)

# ============================================================
# FEATURE DEFINITIONS
# ============================================================

CATEGORICAL_FEATURES = {
    "checking_account_status": [
        "no checking account",
        "< 0 DM",
        "0 <= ... < 200 DM",
        ">= 200 DM / salary assign."
    ],
    "credit_history": [
        "no credits/all paid duly",
        "all credits here paid duly",
        "existing credits paid duly",
        "delay in paying off in past",
        "critical/other credits exist"
    ],
    "purpose": [
        "car (new)",
        "car (used)",
        "furniture/equipment",
        "radio/television",
        "domestic appliances",
        "repairs",
        "education",
        "retraining",
        "business",
        "others"
    ],
    "savings_account_bonds": [
        "unknown/no savings account",
        "< 100 DM",
        "100 <= ... < 500 DM",
        "500 <= ... < 1000 DM",
        ">= 1000 DM"
    ],
    "present_employment_since": [
        "unemployed",
        "< 1 year",
        "1 <= ... < 4 years",
        "4 <= ... < 7 years",
        ">= 7 years"
    ],
    "personal_status_sex": [
        "male: divorced/separated",
        "female: div/sep/married",
        "male: single",
        "male: married/widowed"
    ],
    "other_debtors_guarantors": [
        "none",
        "co-applicant",
        "guarantor"
    ],
    "property": [
        "real estate",
        "bldg society/life ins.",
        "car or other",
        "unknown/no property"
    ],
    "other_installment_plans": [
        "bank",
        "stores",
        "none"
    ],
    "housing": [
        "rent",
        "own",
        "for free"
    ],
    "job": [
        "unemployed/unskilled non-res.",
        "unskilled resident",
        "skilled employee/official",
        "management/self-employed/etc"
    ],
    "telephone": [
        "none",
        "yes, registered"
    ],
    "foreign_worker": [
        "yes",
        "no"
    ]
}

NUMERIC_FEATURES = {
    "duration_months": {"min": 4, "max": 72, "default": 24, "step": 1},
    "credit_amount": {"min": 250, "max": 20000, "default": 3000, "step": 100},
    "installment_rate_pct_of_disp_income": {"min": 1, "max": 4, "default": 3, "step": 1},
    "present_residence_since": {"min": 1, "max": 4, "default": 2, "step": 1},
    "age_years": {"min": 18, "max": 80, "default": 35, "step": 1},
    "existing_credits_count": {"min": 1, "max": 4, "default": 1, "step": 1},
    "people_liable_for_maintenance": {"min": 1, "max": 2, "default": 1, "step": 1}
}

# Feature display names for better UX
FEATURE_LABELS = {
    "checking_account_status": "Checking Account Status",
    "duration_months": "Loan Duration (months)",
    "credit_history": "Credit History",
    "purpose": "Loan Purpose",
    "credit_amount": "Credit Amount (DM)",
    "savings_account_bonds": "Savings Account / Bonds",
    "present_employment_since": "Present Employment Since",
    "installment_rate_pct_of_disp_income": "Installment Rate (% of Income)",
    "present_residence_since": "Present Residence Since (years)",
    "personal_status_sex": "Personal Status & Sex",
    "other_debtors_guarantors": "Other Debtors / Guarantors",
    "property": "Property",
    "age_years": "Age (years)",
    "other_installment_plans": "Other Installment Plans",
    "housing": "Housing",
    "existing_credits_count": "Existing Credits at Bank",
    "job": "Job",
    "people_liable_for_maintenance": "People Liable for Maintenance",
    "telephone": "Telephone",
    "foreign_worker": "Foreign Worker"
}

# Feature descriptions for hover tooltips
FEATURE_HELP = {
    "checking_account_status": "Status of the applicant's checking account at the bank",
    "duration_months": "Duration of the loan in months",
    "credit_history": "Past credit behavior and repayment history",
    "purpose": "Purpose for which the loan is being requested",
    "credit_amount": "Total amount of credit requested in Deutsche Marks",
    "savings_account_bonds": "Amount in savings accounts or bonds",
    "present_employment_since": "Duration of current employment",
    "installment_rate_pct_of_disp_income": "Loan installment as percentage of disposable income (1-4)",
    "present_residence_since": "Years at current residence (1-4 scale)",
    "personal_status_sex": "Gender and marital status of applicant",
    "other_debtors_guarantors": "Whether there are co-applicants or guarantors",
    "property": "Most valuable property owned by applicant",
    "age_years": "Age of the applicant in years",
    "other_installment_plans": "Other ongoing installment plans (bank/stores)",
    "housing": "Current housing situation of applicant",
    "existing_credits_count": "Number of existing credits at this bank (1-4)",
    "job": "Employment type and skill level",
    "people_liable_for_maintenance": "Number of dependents (1-2)",
    "telephone": "Whether applicant has a registered telephone",
    "foreign_worker": "Whether applicant is a foreign worker"
}


# ============================================================
# LOAD MODELS AND DATA
# ============================================================

@st.cache_resource
def load_scorer():
    """Load the ensemble scorer (cached)."""
    scorer = EnsembleScorer(weights=[2.5, 1.5, 3], threshold=0.63)
    scorer.load_models(models_dir=Path(__file__).parent / "models")
    return scorer


@st.cache_data
def load_sample_data():
    """Load sample data for demo (cached)."""
    data_path = Path(__file__).parent / "data" / "sample_data.csv"
    return pd.read_csv(data_path)


# ============================================================
# UI COMPONENTS
# ============================================================

def display_prediction_results(results, input_data):
    """Display prediction results with visual breakdown."""
    
    ensemble_pred = results['ensemble_pred'][0]
    ensemble_proba = results['ensemble_proba'][0]
    
    # Main prediction
    st.markdown("---")
    st.subheader("Prediction Result")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if ensemble_pred == 1:
            st.error("⚠️ **HIGH RISK**")
        else:
            st.success("✅ **LOW RISK**")
        
        risk_pct = ensemble_proba * 100
        st.metric("Risk Probability", f"{risk_pct:.1f}%")
    
    with col2:
        st.markdown("**Model Breakdown:**")
        
        # Individual model results
        models = [
            ("Logistic Regression", results['lrc_proba'][0]),
            ("Random Forest", results['rfc_proba'][0]),
            ("SVC", results['svc_proba'][0]),
        ]
        
        for name, proba in models:
            risk_label = "High Risk" if proba >= 0.5 else "Low Risk"
            st.progress(proba, text=f"{name}: {proba*100:.1f}% ({risk_label})")
        
        st.markdown("---")
        st.progress(ensemble_proba, text=f"**Ensemble: {ensemble_proba*100:.1f}%**")
    
    # Show input data summary in column format
    with st.expander("View Input Data"):
        # Transpose to show features as rows for better readability
        display_df = input_data.T.reset_index()
        display_df.columns = ['Feature', 'Value']
        display_df['Feature'] = display_df['Feature'].map(lambda x: FEATURE_LABELS.get(x, x))
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def random_sample_section(sample_data, scorer):
    """Section for random sample testing."""
    
    st.subheader("🎲 Test with Random Samples")
    st.write("Click a button to load a random sample from the test dataset. Click again for a different sample!")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🟢 Sample Good Risk", use_container_width=True, help="Randomly sample an applicant labeled as good risk"):
            good_samples = sample_data[sample_data["class"] == 0]
            sample = good_samples.sample(n=1)
            st.session_state.current_sample = sample.drop(columns=["class"])
            st.session_state.true_label = 0
            st.session_state.sample_type = "Good Risk"
            
    with col2:
        if st.button("🔴 Sample Bad Risk", use_container_width=True, help="Randomly sample an applicant labeled as bad risk"):
            bad_samples = sample_data[sample_data["class"] == 1]
            sample = bad_samples.sample(n=1)
            st.session_state.current_sample = sample.drop(columns=["class"])
            st.session_state.true_label = 1
            st.session_state.sample_type = "Bad Risk"
    
    with col3:
        if st.button("🎯 Random Sample", use_container_width=True, help="Randomly sample any applicant"):
            sample = sample_data.sample(n=1)
            st.session_state.current_sample = sample.drop(columns=["class"])
            st.session_state.true_label = sample["class"].values[0]
            st.session_state.sample_type = "Random"
    
    # Display results if sample exists
    if "current_sample" in st.session_state:
        input_data = st.session_state.current_sample
        
        st.markdown("---")
        
        # Show sample info
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"📋 **Sample Type:** {st.session_state.sample_type}")
        with col_info2:
            true_label = st.session_state.true_label
            label_text = "Bad Risk (1)" if true_label == 1 else "Good Risk (0)"
            if true_label == 1:
                st.warning(f"🏷️ **True Label:** {label_text}")
            else:
                st.success(f"🏷️ **True Label:** {label_text}")
        
        # Get predictions
        results = scorer.predict_with_details(input_data)
        display_prediction_results(results, input_data)


def manual_input_section(scorer):
    """Section for manual data input."""
    
    st.subheader("✏️ Manual Input")
    st.write("Enter applicant details to get a credit risk prediction.")
    
    with st.expander("Open Input Form", expanded=False):
        input_data = {}
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        # Categorical features
        feature_list = list(CATEGORICAL_FEATURES.keys())
        half = len(feature_list) // 2
        
        with col1:
            st.markdown("**Categorical Features**")
            for feature in feature_list[:half]:
                label = FEATURE_LABELS.get(feature, feature)
                options = CATEGORICAL_FEATURES[feature]
                help_text = FEATURE_HELP.get(feature, None)
                input_data[feature] = st.selectbox(label, options, key=f"cat_{feature}", help=help_text)
        
        with col2:
            st.markdown("** **")  # Spacer
            for feature in feature_list[half:]:
                label = FEATURE_LABELS.get(feature, feature)
                options = CATEGORICAL_FEATURES[feature]
                help_text = FEATURE_HELP.get(feature, None)
                input_data[feature] = st.selectbox(label, options, key=f"cat_{feature}", help=help_text)
        
        st.markdown("---")
        
        # Numeric features
        col3, col4 = st.columns(2)
        numeric_list = list(NUMERIC_FEATURES.keys())
        half_num = len(numeric_list) // 2
        
        with col3:
            st.markdown("**Numeric Features**")
            for feature in numeric_list[:half_num]:
                label = FEATURE_LABELS.get(feature, feature)
                config = NUMERIC_FEATURES[feature]
                help_text = FEATURE_HELP.get(feature, None)
                input_data[feature] = st.number_input(
                    label,
                    min_value=config["min"],
                    max_value=config["max"],
                    value=config["default"],
                    step=config["step"],
                    key=f"num_{feature}",
                    help=help_text
                )
        
        with col4:
            st.markdown("** **")  # Spacer
            for feature in numeric_list[half_num:]:
                label = FEATURE_LABELS.get(feature, feature)
                config = NUMERIC_FEATURES[feature]
                help_text = FEATURE_HELP.get(feature, None)
                input_data[feature] = st.number_input(
                    label,
                    min_value=config["min"],
                    max_value=config["max"],
                    value=config["default"],
                    step=config["step"],
                    key=f"num_{feature}",
                    help=help_text
                )
        
        # Predict button
        if st.button("🔮 Predict Risk", type="primary", use_container_width=True):
            input_df = pd.DataFrame([input_data])
            results = scorer.predict_with_details(input_df)
            display_prediction_results(results, input_df)


# ============================================================
# MAIN APP
# ============================================================

def main():
    # Header
    st.title("💳 Credit Risk Classifier")
    st.markdown("""
    An ensemble machine learning model that predicts credit risk for loan applicants.
    The model combines **Logistic Regression**, **Random Forest**, and **SVC** using soft voting.
    """)
    
    # Cost-sensitive optimization context
    with st.expander("ℹ️ About the Model's Cost-Sensitive Approach", expanded=False):
        st.markdown("""
        This model is optimized for **minimizing business cost**, not just accuracy.
        
        According to the [German Credit Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) documentation:
        
        | Prediction | Actual | Cost |
        |------------|--------|------|
        | Good Risk | Bad Risk (FP) | **5** |
        | Bad Risk | Good Risk (FN) | **1** |
        
        **What this means:** It's 5× more costly to approve a loan for someone who will default 
        than to reject a creditworthy applicant. Therefore, the model is intentionally **conservative** 
        — it may classify some good-risk applicants as bad-risk to avoid the larger cost of defaults.
        
        This cost-aware approach reflects real-world lending where loan defaults typically cause 
        greater financial harm than missed business opportunities.
        """)
    
    # Load resources
    try:
        scorer = load_scorer()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()
        return  # Explicit return to satisfy Python's flow analysis
    
    try:
        sample_data = load_sample_data()
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        st.stop()
        return  # Explicit return to satisfy Python's flow analysis
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["🎲 Random Samples", "✏️ Manual Input"])
    
    with tab1:
        random_sample_section(sample_data, scorer)
    
    with tab2:
        manual_input_section(scorer)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.9em;">
        Built with Streamlit | 
        <a href="https://github.com/ntinasf/credit-risk-svm">View Source Code</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
