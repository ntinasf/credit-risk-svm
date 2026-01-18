"""
Ensemble Scorer for Credit Risk Model

This module provides an EnsembleScorer class that loads three pre-trained models
(Logistic Regression, Random Forest, SVC) from MLflow registry and combines their
predictions using soft voting.

Each model has its own preprocessing pipeline embedded, so this scorer accepts
raw data directly.

Usage:
    # Score with latest model versions
    python score_model.py
    
    # Or use programmatically:
    from score_model import EnsembleScorer
    scorer = EnsembleScorer()
    scorer.load_models()
    predictions = scorer.predict(raw_test_data)
"""

import sys
from pathlib import Path

# Add parent directory to path to enable imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
import sklearn
import warnings

from scripts.functions import evaluate_model, calculate_cost

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 8


class EnsembleScorer:
    """
    Ensemble scorer that combines predictions from multiple models.
    
    Each model (LRC, RFC, SVC) has its own preprocessing pipeline embedded,
    so this scorer accepts raw data directly and handles preprocessing
    internally for each model.
    
    Attributes:
        lrc_model: Logistic Regression model with preprocessing pipeline
        rfc_model: Random Forest model with preprocessing pipeline
        svc_model: SVC model with preprocessing pipeline
        weights: Weights for each model in soft voting (default: equal weights)
        threshold: Decision threshold for final prediction (default: 0.5)
    """
    
    def __init__(self, weights=None, threshold=0.5):
        """
        Initialize the ensemble scorer.
        
        Args:
            weights: List of weights for [lrc, rfc, svc]. Default is equal weights.
            threshold: Decision threshold for converting probabilities to classes.
        """
        self.lrc_model = None
        self.rfc_model = None
        self.svc_model = None
        self.weights = weights if weights is not None else [1, 1, 1]
        self.threshold = threshold
        self.models_loaded = False
        
    def load_models(self, lrc_version="latest", rfc_version="latest", svc_version="latest"):
        """
        Load models from MLflow model registry.
        
        Args:
            lrc_version: Version of LRC model to load (default: "latest")
            rfc_version: Version of RFC model to load (default: "latest")
            svc_version: Version of SVC model to load (default: "latest")
        """
        print("Loading models from MLflow registry...")
        
        # Construct model URIs
        lrc_uri = f"models:/credit-risk-lrc/{lrc_version}"
        rfc_uri = f"models:/credit-risk-rfc/{rfc_version}"
        svc_uri = f"models:/credit-risk-svc/{svc_version}"
        
        print(f"  Loading LRC from: {lrc_uri}")
        self.lrc_model = mlflow.sklearn.load_model(lrc_uri)
        
        print(f"  Loading RFC from: {rfc_uri}")
        self.rfc_model = mlflow.sklearn.load_model(rfc_uri)
        
        print(f"  Loading SVC from: {svc_uri}")
        self.svc_model = mlflow.sklearn.load_model(svc_uri)
        
        self.models_loaded = True
        print("All models loaded successfully!\n")
        
    def load_models_from_run(self, lrc_run_id, rfc_run_id, svc_run_id):
        """
        Load models directly from MLflow run IDs (alternative to registry).
        
        Args:
            lrc_run_id: MLflow run ID for LRC model
            rfc_run_id: MLflow run ID for RFC model
            svc_run_id: MLflow run ID for SVC model
        """
        print("Loading models from MLflow runs...")
        
        print(f"  Loading LRC from run: {lrc_run_id}")
        self.lrc_model = mlflow.sklearn.load_model(f"runs:/{lrc_run_id}/model")
        
        print(f"  Loading RFC from run: {rfc_run_id}")
        self.rfc_model = mlflow.sklearn.load_model(f"runs:/{rfc_run_id}/model")
        
        print(f"  Loading SVC from run: {svc_run_id}")
        self.svc_model = mlflow.sklearn.load_model(f"runs:/{svc_run_id}/model")
        
        self.models_loaded = True
        print("All models loaded successfully!\n")
        
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
        
        # Get probabilities from each model (each applies its own preprocessing)
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
        pred_lrc = self.lrc_model.predict(X)
        pred_rfc = self.rfc_model.predict(X)
        pred_svc = self.svc_model.predict(X)
        
        proba_lrc = self.lrc_model.predict_proba(X)[:, 1]
        proba_rfc = self.rfc_model.predict_proba(X)[:, 1]
        proba_svc = self.svc_model.predict_proba(X)[:, 1]
        
        # Ensemble predictions
        ensemble_proba = self.predict_proba(X)[:, 1]
        ensemble_pred = self.predict(X)
        
        return {
            'lrc_pred': pred_lrc,
            'lrc_proba': proba_lrc,
            'rfc_pred': pred_rfc,
            'rfc_proba': proba_rfc,
            'svc_pred': pred_svc,
            'svc_proba': proba_svc,
            'ensemble_pred': ensemble_pred,
            'ensemble_proba': ensemble_proba
        }
    
    def evaluate(self, X, y_true, show_details=True):
        """
        Evaluate the ensemble on test data.
        
        Args:
            X: Raw input data (pandas DataFrame)
            y_true: True labels
            show_details: Whether to show individual model performance
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        print("=" * 60)
        print("ENSEMBLE MODEL EVALUATION")
        print("=" * 60)
        
        if show_details:
            print("\n--- Individual Model Performance ---\n")
            
            # Evaluate LRC
            print("Logistic Regression:")
            lrc_pred = self.lrc_model.predict(X)
            lrc_proba = self.lrc_model.predict_proba(X)[:, 1]
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            print(f"  Accuracy: {accuracy_score(y_true, lrc_pred):.4f}")
            print(f"  F1 Score: {f1_score(y_true, lrc_pred):.4f}")
            print(f"  ROC AUC:  {roc_auc_score(y_true, lrc_proba):.4f}")
            print(f"  Cost:     {calculate_cost(y_true, lrc_pred)}")
            
            # Evaluate RFC
            print("\nRandom Forest:")
            rfc_pred = self.rfc_model.predict(X)
            rfc_proba = self.rfc_model.predict_proba(X)[:, 1]
            print(f"  Accuracy: {accuracy_score(y_true, rfc_pred):.4f}")
            print(f"  F1 Score: {f1_score(y_true, rfc_pred):.4f}")
            print(f"  ROC AUC:  {roc_auc_score(y_true, rfc_proba):.4f}")
            print(f"  Cost:     {calculate_cost(y_true, rfc_pred)}")
            
            # Evaluate SVC
            print("\nSVC:")
            svc_pred = self.svc_model.predict(X)
            svc_proba = self.svc_model.predict_proba(X)[:, 1]
            print(f"  Accuracy: {accuracy_score(y_true, svc_pred):.4f}")
            print(f"  F1 Score: {f1_score(y_true, svc_pred):.4f}")
            print(f"  ROC AUC:  {roc_auc_score(y_true, svc_proba):.4f}")
            print(f"  Cost:     {calculate_cost(y_true, svc_pred)}")
        
        print("\n--- Ensemble Performance (Soft Voting) ---\n")
        
        # Ensemble evaluation
        ensemble_pred = self.predict(X)
        ensemble_proba = self.predict_proba(X)[:, 1]
        
        from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
        
        accuracy = accuracy_score(y_true, ensemble_pred)
        f1 = f1_score(y_true, ensemble_pred)
        precision = precision_score(y_true, ensemble_pred)
        roc_auc = roc_auc_score(y_true, ensemble_proba)
        cost = calculate_cost(y_true, ensemble_pred)
        
        print(f"  Weights:    LRC={self.weights[0]}, RFC={self.weights[1]}, SVC={self.weights[2]}")
        print(f"  Threshold:  {self.threshold}")
        print(f"  Accuracy:   {accuracy:.4f}")
        print(f"  F1 Score:   {f1:.4f}")
        print(f"  Precision:  {precision:.4f}")
        print(f"  ROC AUC:    {roc_auc:.4f}")
        print(f"  Total Cost: {cost}")
        print("=" * 60)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'roc_auc': roc_auc,
            'cost': cost
        }


def main():
    """Main function to demonstrate ensemble scoring on test data."""
    
    # Load data
    home = Path.cwd()
    data_dir = home / "data"
    df = pd.read_csv(data_dir / "processed" / "test_data.csv")
    sklearn.set_config(transform_output="pandas")
    
    print(f"Test set size: {len(df)} samples\n")
    
    # Create and load ensemble scorer
    scorer = EnsembleScorer(weights=[2.5, 1.5, 3], threshold=0.63)
    
    try:
        scorer.load_models()
    except Exception as e:
        print(f"Error loading models from registry: {e}")
        print("\nMake sure you have trained and registered the models first!")
        print("Run: python scripts/final_model.py")
        return
    
    # Evaluate ensemble on test data
    metrics = scorer.evaluate(df.drop(columns=["class"]), df["class"], show_details=True)
    
    return scorer, metrics


if __name__ == "__main__":
    main()
