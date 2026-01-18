"""
Custom Transformers for Credit Risk Model

This module contains custom sklearn transformers that are embedded in the 
model pipelines. These must be importable when loading the pickled models.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for creating new features and transforming existing ones.
    
    This transformer is embedded in all model pipelines and performs:
    - Feature creation (monthly_burden, duration_to_age_ratio, etc.)
    - Log transformations
    - Binning continuous variables
    - Merging sparse categories
    - Dropping low-information columns
    """

    def __init__(self, duplicate_checking=False, duplicate_amount=False):
        self.cols_to_drop = [
            "other_debtors_guarantors",
            "telephone",
            "people_liable_for_maintenance",
        ]
        self.duplicate_checking = duplicate_checking
        self.duplicate_amount = duplicate_amount
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Drop unnecessary columns first (only if they exist)
        cols_to_drop_existing = [c for c in self.cols_to_drop if c in X.columns]
        X = X.drop(columns=cols_to_drop_existing)
        
        if self.duplicate_checking:
            X['checking_2'] = X['checking_account_status'].copy()
            X['personal_status_2'] = X['personal_status_sex'].copy()
            
        X['no_checking'] = (X['checking_account_status'] == 'no checking account').astype(int)

        X['credit_amount_squared'] = X['credit_amount'] ** 2
        X['duration_squared'] = X['duration_months'] ** 2

        # Create new features 
        X['monthly_burden'] = X['credit_amount'] / X['duration_months']
        X['duration_to_age_ratio'] = X['duration_months'] / X['age_years']

        # Apply transformations to new features
        X['duration_to_age_ratio_sqrt'] = np.sqrt(X['duration_to_age_ratio'])
        X['credit_log'] = np.log(X['credit_amount'] + 1)
        X['duration_log'] = np.log(X['duration_months'] + 1)
        X['monthly_burden_log'] = np.log(X['monthly_burden'])

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
        
        # Create age groups BEFORE dropping age_years
        X['age_group'] = pd.cut(
            X['age_years'],
            bins=[0, 25, 35, 55, 100],
            labels=['Young', 'Early_Career', 'Prime', 'Mature']
        )
        
        # Merge housing categories
        X['housing'] = X['housing'].replace(['for free', 'rent'], 'not_own')

        # Merge credit history categories
        X['credit_history'] = X['credit_history'].replace(
            ['all credits here paid duly', 'no credits/all paid duly'], 'all credits paid'
        )

        return X


class BaselineEngineer(BaseEstimator, TransformerMixin):
    """
    Baseline transformer that applies simple preprocessing:
    - One-hot encoding to all categorical columns
    - StandardScaler to all numerical columns
    """

    def __init__(self, categorical_cols=None, numerical_cols=None):
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.preprocessor_ = None
        self._categorical_cols = None
        self._numerical_cols = None
        
    def fit(self, X, y=None):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler
        from category_encoders import OneHotEncoder
        
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Auto-detect column types if not provided
        if self.categorical_cols is None:
            self._categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            self._categorical_cols = [c for c in self.categorical_cols if c in X.columns]
            
        if self.numerical_cols is None:
            self._numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
        else:
            self._numerical_cols = [c for c in self.numerical_cols if c in X.columns]
        
        # Build the preprocessing pipeline
        transformers = []
        
        if self._categorical_cols:
            transformers.append(
                ('one_hot', OneHotEncoder(cols=self._categorical_cols, use_cat_names=True), 
                 self._categorical_cols)
            )
        
        if self._numerical_cols:
            transformers.append(
                ('scaler', StandardScaler(), self._numerical_cols)
            )
        
        self.preprocessor_ = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        self.preprocessor_.fit(X, y)
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self.preprocessor_.transform(X)
    
    def get_feature_names_out(self, input_features=None):
        if self.preprocessor_ is not None:
            return self.preprocessor_.get_feature_names_out(input_features)
        return None
