"""Score and log ensemble model performance to MLflow.

This script loads the three pre-trained models (LRC, RFC, SVC) from MLflow registry,
creates an ensemble using soft voting, and logs the ensemble's performance metrics
to MLflow for comparison with individual models.

Usage:
    python scripts/score_ensemble.py --weights 2.5 1.5 3.0 --threshold 0.63
    python scripts/score_ensemble.py --help
"""

import sys
import warnings
from pathlib import Path

# Add parent directory to path to enable imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

import json
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from scripts.functions import calculate_cost, plot_confusion_matrix, plot_precision_recall_curve

# Suppress warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
RANDOM_STATE = 8


def load_models_from_registry(lrc_version="latest", rfc_version="latest", svc_version="latest"):
    """Load models from MLflow model registry.
    
    Args:
        lrc_version: Version of LRC model (default: "latest")
        rfc_version: Version of RFC model (default: "latest")
        svc_version: Version of SVC model (default: "latest")
        
    Returns:
        Tuple of (lrc_model, rfc_model, svc_model)
    """
    print("Loading models from MLflow registry...")
    
    lrc_uri = f"models:/credit-risk-lrc/{lrc_version}"
    rfc_uri = f"models:/credit-risk-rfc/{rfc_version}"
    svc_uri = f"models:/credit-risk-svc/{svc_version}"
    
    print(f"  Loading LRC from: {lrc_uri}")
    lrc_model = mlflow.sklearn.load_model(lrc_uri)
    
    print(f"  Loading RFC from: {rfc_uri}")
    rfc_model = mlflow.sklearn.load_model(rfc_uri)
    
    print(f"  Loading SVC from: {svc_uri}")
    svc_model = mlflow.sklearn.load_model(svc_uri)
    
    print("All models loaded successfully!\n")
    
    return lrc_model, rfc_model, svc_model


def ensemble_predict_proba(X, lrc_model, rfc_model, svc_model, weights):
    """Get ensemble probability predictions using soft voting.
    
    Args:
        X: Raw input data (pandas DataFrame)
        lrc_model: Logistic Regression model with preprocessing
        rfc_model: Random Forest model with preprocessing
        svc_model: SVC model with preprocessing
        weights: List of weights [lrc_weight, rfc_weight, svc_weight]
        
    Returns:
        numpy array of ensemble probabilities for class 1
    """
    # Get probabilities from each model
    proba_lrc = lrc_model.predict_proba(X)[:, 1]
    proba_rfc = rfc_model.predict_proba(X)[:, 1]
    proba_svc = svc_model.predict_proba(X)[:, 1]
    
    # Weighted average
    total_weight = sum(weights)
    ensemble_proba = (
        weights[0] * proba_lrc +
        weights[1] * proba_rfc +
        weights[2] * proba_svc
    ) / total_weight
    
    return ensemble_proba, proba_lrc, proba_rfc, proba_svc


def score_ensemble(X, y, lrc_model, rfc_model, svc_model, 
                   weights, threshold, log_to_mlflow=True):
    """Score ensemble and optionally log to MLflow.
    
    Args:
        X: Raw input features
        y: True labels
        lrc_model, rfc_model, svc_model: Pre-trained models
        weights: List of weights [lrc, rfc, svc]
        threshold: Decision threshold
        log_to_mlflow: Whether to log metrics to MLflow
        
    Returns:
        Dictionary of metrics
    """
    # Get ensemble predictions
    ensemble_proba, proba_lrc, proba_rfc, proba_svc = ensemble_predict_proba(
        X, lrc_model, rfc_model, svc_model, weights
    )
    ensemble_pred = (ensemble_proba >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y, ensemble_pred)
    f1 = f1_score(y, ensemble_pred)
    precision = precision_score(y, ensemble_pred)
    recall = recall_score(y, ensemble_pred)
    roc_auc = roc_auc_score(y, ensemble_proba)
    total_cost, avg_cost = calculate_cost(y, ensemble_pred)
    
    metrics = {
        "val_accuracy": accuracy,
        "val_f1": f1,
        "val_precision": precision,
        "val_recall": recall,
        "val_roc_auc": roc_auc,
        "val_cost": total_cost,
        "val_avg_cost": avg_cost,
    }
    
    # Print results
    print("=" * 60)
    print("ENSEMBLE MODEL EVALUATION")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Weights:   LRC={weights[0]}, RFC={weights[1]}, SVC={weights[2]}")
    print(f"  Threshold: {threshold}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    print(f"  Cost:      {total_cost} (avg: {avg_cost:.4f})")
    print("=" * 60)
    
    if log_to_mlflow:
        # Generate run name
        run_name = f"Ensemble_w{weights[0]}-{weights[1]}-{weights[2]}_t{threshold}"
        
        mlflow.set_experiment("credit-risk-ensemble")
        
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params({
                "model_type": "Ensemble",
                "voting_type": "soft",
                "weight_lrc": weights[0],
                "weight_rfc": weights[1],
                "weight_svc": weights[2],
                "threshold": threshold,
                "n_val_samples": len(y),
            })
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log confusion matrix
            import matplotlib.pyplot as plt
            cm_fig = plot_confusion_matrix(y, ensemble_pred, model_name="Ensemble", show_plot=False)
            mlflow.log_figure(cm_fig, "confusion_matrix.png")
            plt.close(cm_fig)
            
            # Log precision-recall curve
            pr_fig, pr_metrics = plot_precision_recall_curve(y, ensemble_proba, model_name="Ensemble", show_plot=False)
            mlflow.log_figure(pr_fig, "precision_recall_curve.png")
            mlflow.log_metric("val_average_precision", pr_metrics['average_precision'])
            plt.close(pr_fig)
            
            # Log ensemble config as artifact
            config = {
                "weights": weights,
                "threshold": threshold,
                "component_models": [
                    "credit-risk-lrc",
                    "credit-risk-rfc",
                    "credit-risk-svc"
                ]
            }
            config_path = Path("ensemble_config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            mlflow.log_artifact(config_path)
            config_path.unlink()  # Clean up temp file
            
            # Set tags
            mlflow.set_tags({
                "model_family": "Ensemble",
                "voting_method": "Soft Voting",
                "n_estimators": 3,
            })
            
            print(f"\n✅ Logged to MLflow experiment 'credit-risk-ensemble'")
    
    return metrics


def parse_args():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Score ensemble model and log to MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--weights", nargs=3, type=float, default=[2.5, 1.5, 3.0],
                        metavar=("LRC", "RFC", "SVC"),
                        help="Weights for each model [LRC, RFC, SVC]")
    parser.add_argument("--threshold", type=float, default=0.63,
                        help="Decision threshold for classification")
    parser.add_argument("--no-log", dest="log_to_mlflow", action="store_false",
                        help="Don't log to MLflow (just print metrics)")
    parser.add_argument("--val-size", type=int, default=150,
                        help="Validation set size")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE,
                        help="Random seed for reproducibility")
    parser.add_argument("--lrc-version", type=str, default="latest",
                        help="LRC model version in registry")
    parser.add_argument("--rfc-version", type=str, default="latest",
                        help="RFC model version in registry")
    parser.add_argument("--svc-version", type=str, default="latest",
                        help="SVC model version in registry")
    return parser.parse_args()


def main():
    """Main function to score ensemble on validation data."""
    args = parse_args()
    
    # Load data (same split as individual models)
    home = Path.cwd()
    data_dir = home / "data"
    df = pd.read_csv(data_dir / "processed" / "train_data.csv")
    sklearn.set_config(transform_output="pandas")
    
    # Create same validation split as train_*.py scripts
    X_train, X_val, y_train, y_val = train_test_split(
        df.drop(columns=["class"]),
        df["class"],
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=df["class"]
    )
    
    print(f"Validation set size: {len(X_val)} samples\n")
    
    # Load models from registry
    try:
        lrc_model, rfc_model, svc_model = load_models_from_registry(
            lrc_version=args.lrc_version,
            rfc_version=args.rfc_version,
            svc_version=args.svc_version
        )
    except Exception as e:
        print(f"Error loading models: {e}")
        print("\nMake sure you have trained and registered the models first!")
        print("Run the training scripts with --log-model flag:")
        print("  python scripts/train_lrc.py --tune --smote --tune-threshold --log-model")
        print("  python scripts/train_rf.py --tune --tune-threshold --log-model")
        print("  python scripts/train_svc.py --tune --smote --tune-threshold --log-model")
        return
    
    # Score ensemble
    metrics = score_ensemble(
        X_val, y_val,
        lrc_model, rfc_model, svc_model,
        weights=args.weights,
        threshold=args.threshold,
        log_to_mlflow=args.log_to_mlflow
    )
    
    return metrics


if __name__ == "__main__":
    main()
