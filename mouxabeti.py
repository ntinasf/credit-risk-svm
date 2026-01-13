cols_to_drop = [
    "other_debtors_guarantors",
    "telephone",
    "foreign_worker",
    "present_residence_since",
    "existing_credits_count",
    "people_liable_for_maintenance",
    "installment_rate_pct_of_disp_income",
    "personal_status_sex"
]
train_df.drop()

train_df['duration_squared'] = train_df['duration_months'] ** 2
train_df['monthly_burden'] = train_df['credit_amount'] / train_df['duration_months']
train_df['purpose'] = train_df['purpose'].replace(['education', 'retraining'], 'personal_development')
train_df['purpose'] = train_df['purpose'].replace(['domestic appliances', 'repairs', 'others'], 'home_improvement')
train_df['credit_amount_bins'] = pd.cut(train_df['credit_amount'],
                                        bins=[0, 2000, 4000, 7000, 10000, 50000],
                                        labels=['a', 'b', 'c', 'd', 'e'])
train_df['savings_account_bonds'] = train_df['savings_account_bonds'].replace(['< 100 DM', '100 <= ... < 500 DM'], '< 500 DM')
train_df['savings_account_bonds'] = train_df['savings_account_bonds'].replace(['500 <= ... < 1000 DM', '>= 1000 DM'], '>= 500 DM')
train_df['age_group'] = pd.cut(train_df['age_years'],
                               bins=[0, 25, 35, 50, 65, 100],
                               labels=['Young', 'Early_Career', 'Prime', 'Mature', 'Senior'])

cols_to_remove = [
    'duration_months',
    'credit_amount',
    'age_years'
]

train_df_dropped = train_df.drop(columns=cols_to_remove_2)

one_hot_cols_2 = [
    'credit_history',
    'purpose',
    'credit_amount_bins',
    'property',
    'housing',
    'personal_status_sex'

]

woe_cols_2 = [
    'checking_account_status',
    'savings_account_bonds',
    'present_employment_since',
    'age_group',
    'other_installment_plans',
    'job'
]

numeric_cols_2 = [
    'duration_squared',
    'monthly_burden',
    #'duration_months',
    #'credit_amount',
    #'age_years'
]

from sklearn.compose import ColumnTransformer

preprocessing_pipeline_final = ColumnTransformer(
    transformers=[
        ('one_hot', OneHotEncoder(cols=one_hot_cols_2), one_hot_cols_2),
        ('woe', WOEEncoder(cols=woe_cols_2), woe_cols_2),
        ('scaler', StandardScaler(), numeric_cols_2)
    ],
    remainder='passthrough'  # keeps any other columns unchanged
)

final_train_df = preprocessing_pipeline_final.fit_transform(train_df_dropped, y_train)
final_train_df.head(10)



===============================================================

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import OneHotEncoder, WOEEncoder
import pandas as pd
import numpy as np


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
        ('woe', WOEEncoder(cols=woe_cols), woe_cols),
        ('scaler', StandardScaler(), numeric_cols)
    ],
    remainder='drop'  # or 'passthrough' if you want to keep other columns
)


# Full pipeline combining feature engineering + encoding
full_pipeline = Pipeline([
    ('feature_engineer', FeatureEngineer()),
    ('encoder', encoding_pipeline)
])


# Usage:
# X_train_transformed = full_pipeline.fit_transform(X_train, y_train)
# X_test_transformed = full_pipeline.transform(X_test)