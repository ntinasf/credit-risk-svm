import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd().parent))
#import mlflow
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder, WOEEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import OneHotEncoder, WOEEncoder
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


RANDOM_STATE = 8

home = Path.cwd()#.parent
data_dir = home / "data"
notebook_dir = home / "notebooks"
df = pd.read_csv(data_dir / "processed" / "german_credit.csv")
sklearn.set_config(transform_output="pandas")


X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["class"]),
    df["class"],
    test_size=0.3,
    random_state=RANDOM_STATE
)

print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')


# Custom transformer for feature engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for creating new features and transforming existing ones."""
    
    def __init__(self):
        self.cols_to_drop = [
            "other_debtors_guarantors",
            "telephone",
            "foreign_worker",
            "present_residence_since",
            "existing_credits_count",
            "people_liable_for_maintenance",
            "installment_rate_pct_of_disp_income",
            "personal_status_sex"
        ]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Drop unnecessary columns (only if they exist)
        cols_to_drop_existing = [c for c in self.cols_to_drop if c in X.columns]
        X = X.drop(columns=cols_to_drop_existing)
        
        # Create new features
        X['duration_squared'] = X['duration_months'] ** 2
        X['monthly_burden'] = X['credit_amount'] / X['duration_months']
        
        # Merge purpose categories
        X['purpose'] = X['purpose'].replace(
            ['education', 'retraining'], 'personal_development'
        )
        X['purpose'] = X['purpose'].replace(
            ['domestic appliances', 'repairs', 'others'], 'home_improvement'
        )
        
        # Bin credit amount
        X['credit_amount_bins'] = pd.cut(
            X['credit_amount'],
            bins=[0, 2000, 4000, 7000, 10000, 50000],
            labels=['a', 'b', 'c', 'd', 'e']
        )
        
        # Merge savings categories
        X['savings_account_bonds'] = X['savings_account_bonds'].replace(
            ['< 100 DM', '100 <= ... < 500 DM'], '< 500 DM'
        )
        X['savings_account_bonds'] = X['savings_account_bonds'].replace(
            ['500 <= ... < 1000 DM', '>= 1000 DM'], '>= 500 DM'
        )
        
        # Create age groups
        X['age_group'] = pd.cut(
            X['age_years'],
            bins=[0, 25, 35, 50, 65, 100],
            labels=['Young', 'Early_Career', 'Prime', 'Mature', 'Senior']
        )
        
        # Drop original columns that were transformed
        X = X.drop(columns=['duration_months', 'credit_amount', 'age_years'])
        
        return X


# Define column groups for encoding
one_hot_cols = [
    'credit_history',
    'purpose',
    'credit_amount_bins',
    'property',
    'housing',
]

woe_cols = [
    'checking_account_status',
    'savings_account_bonds',
    'present_employment_since',
    'age_group',
    'other_installment_plans',
    'job'
]

numeric_cols = [
    'duration_squared',
    'monthly_burden',
]


# Create the encoding pipeline
encoding_pipeline = ColumnTransformer(
    transformers=[
        ('one_hot', OneHotEncoder(cols=one_hot_cols, use_cat_names=True), one_hot_cols),
        #('woe', WOEEncoder(cols=woe_cols), woe_cols),
        ('scaler', StandardScaler(), numeric_cols)
    ],
    remainder='drop'  # or 'passthrough' if you want to keep other columns
)


# Full pipeline combining feature engineering + encoding
full_pipeline = Pipeline([
    ('feature_engineer', FeatureEngineer()),
    ('encoder', encoding_pipeline)
])

train_df = full_pipeline.fit_transform(X_train, y_train)
print(f'Transformed train_df shape: {train_df.shape}')
test_df = full_pipeline.transform(X_test)
print(f'Transformed test_df shape: {test_df.shape}')

# ============================================================================
# PHASE 1: BASE MODELS EVALUATION (Pre-Tuning)
# ============================================================================

svm = SVC(probability=True, random_state=RANDOM_STATE)
log_reg = LogisticRegression(random_state=RANDOM_STATE)
rfc = RandomForestClassifier(random_state=RANDOM_STATE)

# Store pre-tuning results
pre_tuning_results = []

print("\n" + "=" * 70)
print("📊 PHASE 1: BASE MODELS EVALUATION (Pre-Tuning)")
print("=" * 70)

print("\n🔄 Cross-Validation Results (5-fold):")
print("-" * 50)

for model in [svm, log_reg, rfc]:
    model_name = model.__class__.__name__
    scores = cross_val_score(model, train_df, y_train, cv=5, scoring='roc_auc')
    cv_mean = np.mean(scores)
    cv_std = np.std(scores)
    print(f"   {model_name:<25} CV AUC: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Fit and evaluate on test set
    model.fit(train_df, y_train)
    y_pred_proba = model.predict_proba(test_df)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    pre_tuning_results.append({
        'Model': model_name,
        'CV AUC Mean': cv_mean,
        'CV AUC Std': cv_std,
        'Test AUC': test_auc
    })

print("\n🎯 Test Set Results:")
print("-" * 50)
for result in pre_tuning_results:
    print(f"   {result['Model']:<25} Test AUC: {result['Test AUC']:.4f}")

# ============================================================================
# PHASE 2: HYPERPARAMETER TUNING
# ============================================================================

print("\n" + "=" * 70)
print("🔧 PHASE 2: HYPERPARAMETER TUNING (Bayesian Optimization)")
print("=" * 70)

# SVC search space
svc_space = {
    'C': Real(0.001, 100, prior='uniform'),
    'gamma': Real(1e-4, 1, prior='uniform'),
    'kernel': Categorical(['rbf','poly', 'linear']),
    'tol': Real(1e-4, 1e-1, prior='log-uniform') 
}


log_reg_space = {
    'penalty': Categorical(['l2']),
    'C': Real(1e-3, 10, prior='uniform'),
    'solver': Categorical(['liblinear']),
    'max_iter': Integer(100, 4000),
}


rfc_space = {
    'n_estimators': Integer(50, 1000),
    'max_depth': Integer(3, 12),
    'min_samples_leaf': Integer(20, 50),
    'class_weight': Categorical(['balanced', None]),
    'min_samples_split': Integer(2, 10)
}

# Store post-tuning results
post_tuning_results = []

print("\n🔄 Training tuned models...")
print("-" * 50)

svc_tuned = BayesSearchCV(
    estimator=SVC(probability=True, random_state=RANDOM_STATE),
    search_spaces=svc_space,
    n_iter=20,
    scoring='roc_auc',
    cv=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0
)
log_reg_tuned = BayesSearchCV(
    estimator=LogisticRegression(random_state=RANDOM_STATE),
    search_spaces=log_reg_space,
    n_iter=20,
    scoring='roc_auc',
    cv=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0
)

rfc_tuned = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE),
    search_spaces=rfc_space,
    n_iter=20,
    scoring='roc_auc',
    cv=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0
)

for model in [svc_tuned, log_reg_tuned, rfc_tuned]:
    model_name = model.estimator.__class__.__name__
    print(f"\n   ⏳ Tuning {model_name}...")
    model.fit(train_df, y_train)
    
    cv_best_score = model.best_score_
    y_pred_proba = model.best_estimator_.predict_proba(test_df)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"   ✅ {model_name} tuning complete")
    print(f"      Best CV AUC: {cv_best_score:.4f}")
    print(f"      Test AUC: {test_auc:.4f}")
    print(f"      Best params: {model.best_params_}")
    
    post_tuning_results.append({
        'Model': model_name,
        'Best CV AUC': cv_best_score,
        'Test AUC': test_auc,
        'Best Params': model.best_params_
    })

# ============================================================================
# FINAL RESULTS SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("📋 FINAL RESULTS SUMMARY")
print("=" * 70)

# Pre-tuning results table
print("\n📊 PRE-TUNING RESULTS (Base Models):")
print("-" * 70)
pre_tuning_df = pd.DataFrame(pre_tuning_results)
pre_tuning_df['CV AUC'] = pre_tuning_df.apply(
    lambda x: f"{x['CV AUC Mean']:.4f} ± {x['CV AUC Std']:.4f}", axis=1
)
pre_tuning_df['Test AUC'] = pre_tuning_df['Test AUC'].apply(lambda x: f"{x:.4f}")
pre_tuning_display = pre_tuning_df[['Model', 'CV AUC', 'Test AUC']].copy()
pre_tuning_display.index = range(1, len(pre_tuning_display) + 1)
pre_tuning_display.index.name = 'Rank'
print(pre_tuning_display.to_string())

# Post-tuning results table
print("\n📊 POST-TUNING RESULTS (Tuned Models):")
print("-" * 70)
post_tuning_df = pd.DataFrame(post_tuning_results)
post_tuning_df['Best CV AUC'] = post_tuning_df['Best CV AUC'].apply(lambda x: f"{x:.4f}")
post_tuning_df['Test AUC'] = post_tuning_df['Test AUC'].apply(lambda x: f"{x:.4f}")
post_tuning_display = post_tuning_df[['Model', 'Best CV AUC', 'Test AUC']].copy()
post_tuning_display.index = range(1, len(post_tuning_display) + 1)
post_tuning_display.index.name = 'Rank'
print(post_tuning_display.to_string())

# Comparison table
print("\n📊 IMPROVEMENT COMPARISON (Pre vs Post Tuning):")
print("-" * 70)
comparison_data = []
for pre, post in zip(pre_tuning_results, post_tuning_results):
    improvement = post['Test AUC'] - pre['Test AUC']
    comparison_data.append({
        'Model': pre['Model'],
        'Pre-Tuning Test AUC': f"{pre['Test AUC']:.4f}",
        'Post-Tuning Test AUC': f"{post['Test AUC']:.4f}",
        'Improvement': f"{improvement:+.4f}"
    })
comparison_df = pd.DataFrame(comparison_data)
comparison_df.index = range(1, len(comparison_df) + 1)
print(comparison_df.to_string())

# Best model overall
best_post = max(post_tuning_results, key=lambda x: x['Test AUC'])
print("\n" + "=" * 70)
print(f"🏆 BEST MODEL: {best_post['Model']}")
print(f"   Test AUC: {best_post['Test AUC']:.4f}")
print(f"   Best Parameters: {best_post['Best Params']}")
print("=" * 70)

