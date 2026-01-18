"""
Simplified Ensemble Scorer for Streamlit Deployment

Loads pre-trained models from pickle files (exported from MLflow).
Each model has its own preprocessing pipeline embedded.

IMPORTANT: custom_transformers must be imported BEFORE loading models
so that joblib can find the FeatureEngineer class definition.
"""

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# CRITICAL: Import custom transformers before loading pickled models
# This registers the FeatureEngineer class so joblib can deserialize it
import custom_transformers
from custom_transformers import FeatureEngineer, BaselineEngineer


# ============================================================
# COST FUNCTIONS (needed for unpickling TunedThresholdClassifierCV)
# ============================================================

def calculate_cost(y_true, y_pred, cost_fp=5, cost_fn=1, print_results=False):
    """
    Calculate the cost of predictions based on false positives and false negatives.
    """
    from sklearn.metrics import confusion_matrix
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    n_samples = len(y_true)
    avg_cost = total_cost / n_samples
    
    return total_cost, avg_cost


def cost_scorer_fn(y_true, y_pred):
    """
    Scorer-compatible function that returns only total cost.
    Use with: make_scorer(cost_scorer_fn, greater_is_better=False)
    """
    total_cost, _ = calculate_cost(y_true, y_pred)
    return total_cost


# ============================================================
# FAKE SCRIPTS MODULE (for unpickling models)
# ============================================================

# Create a fake 'scripts' module so pickled models can find classes/functions
# The models were trained with scripts.functions.FeatureEngineer
class FakeScriptsModule:
    pass

class FakeFunctionsModule:
    FeatureEngineer = FeatureEngineer
    BaselineEngineer = BaselineEngineer
    calculate_cost = staticmethod(calculate_cost)
    cost_scorer_fn = staticmethod(cost_scorer_fn)

fake_scripts = FakeScriptsModule()
fake_scripts.functions = FakeFunctionsModule()
sys.modules['scripts'] = fake_scripts
sys.modules['scripts.functions'] = fake_scripts.functions


class EnsembleScorer:
    """
    Ensemble scorer that combines predictions from multiple models.
    
    Loads models from pickle files for portable deployment.
    """
    
    def __init__(self, weights=None, threshold=0.5):
        """
        Initialize the ensemble scorer.
        
        Args:
            weights: List of weights for [lrc, rfc, svc]. Default is [2.5, 1.5, 3].
            threshold: Decision threshold for converting probabilities to classes.
        """
        self.lrc_model = None
        self.rfc_model = None
        self.svc_model = None
        self.weights = weights if weights is not None else [2.5, 1.5, 3]
        self.threshold = threshold
        self.models_loaded = False
        
    def load_models(self, models_dir="models"):
        """
        Load models from pickle files.
        
        Args:
            models_dir: Directory containing the model pickle files.
        """
        models_path = Path(models_dir)
        
        self.lrc_model = joblib.load(models_path / "lrc_pipeline.pkl")
        self.rfc_model = joblib.load(models_path / "rfc_pipeline.pkl")
        self.svc_model = joblib.load(models_path / "svc_pipeline.pkl")
        
        self.models_loaded = True
        
    def predict_proba(self, X):
        """
        Get probability predictions from all models and combine them.
        
        Args:
            X: Raw input data (pandas DataFrame)
            
        Returns:
            numpy array of shape (n_samples, 2) with combined probabilities
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Get probabilities from each model
        proba_lrc = self.lrc_model.predict_proba(X)
        proba_rfc = self.rfc_model.predict_proba(X)
        proba_svc = self.svc_model.predict_proba(X)
        
        # Weighted average of probabilities (soft voting)
        total_weight = sum(self.weights)
        weighted_proba = (
            self.weights[0] * proba_lrc +
            self.weights[1] * proba_rfc +
            self.weights[2] * proba_svc
        ) / total_weight
        
        return weighted_proba
    
    def predict(self, X):
        """
        Get class predictions using soft voting ensemble.
        
        Args:
            X: Raw input data (pandas DataFrame)
            
        Returns:
            numpy array of class predictions (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)
    
    def predict_with_details(self, X):
        """
        Get predictions with detailed breakdown from each model.
        
        Args:
            X: Raw input data (pandas DataFrame)
            
        Returns:
            Dictionary containing individual and ensemble predictions
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Get predictions from each model
        proba_lrc = self.lrc_model.predict_proba(X)[:, 1]
        proba_rfc = self.rfc_model.predict_proba(X)[:, 1]
        proba_svc = self.svc_model.predict_proba(X)[:, 1]
        
        # Ensemble predictions
        ensemble_proba = self.predict_proba(X)[:, 1]
        ensemble_pred = self.predict(X)
        
        return {
            'lrc_proba': proba_lrc,
            'rfc_proba': proba_rfc,
            'svc_proba': proba_svc,
            'ensemble_proba': ensemble_proba,
            'ensemble_pred': ensemble_pred
        }
