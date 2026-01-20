"""Train Logistic Regression model for credit risk classification.

This script trains a Logistic Regression classifier with optional:
- Bayesian hyperparameter tuning (BayesSearchCV)
- SMOTE for class imbalance handling
- Cost-sensitive threshold tuning
- MLflow experiment tracking and model registry

Usage:
    python scripts/train_lrc.py --tune --smote --tune-threshold --log-model
    python scripts/train_lrc.py --help
"""

import sys
import warnings
from pathlib import Path

# Add parent directory to path to enable imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import sklearn
from category_encoders import CountEncoder, OneHotEncoder, WOEEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import TunedThresholdClassifierCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

from scripts.functions import (
    FeatureEngineer,
    cost_scorer_fn,
    evaluate_model,
    plot_confusion_matrix,
    plot_learning_curve,
    plot_precision_recall_curve,
)

# Suppress warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
RANDOM_STATE = 8
CV = 5

def lrc_preprocess(X, y):
    """Preprocess the data for LRC model."""

    one_hot_cols_lrc = [
        'credit_history', 
        'savings_account_bonds',
        'present_residence_since',
        'personal_status_sex',
        'age_group',
        'job',
        'foreign_worker',
    ]

    woe_cols_lrc = [
        'checking_account_status',
        'purpose',
        'present_employment_since',      
        'property',
        'housing',                     
    ]

    count_cols_lrc = [
        "installment_rate_pct_of_disp_income",
        'other_installment_plans',  
    ]

    numeric_cols = [
        'duration_log',
        'credit_log',
        #'age_years',
        "existing_credits_count",
        #'duration_to_age_ratio_sqrt',
        'monthly_burden_log',
    ]

    encoding_pipeline_lrc = ColumnTransformer(
        transformers=[
            ('one_hot', OneHotEncoder(cols=one_hot_cols_lrc, use_cat_names=True), one_hot_cols_lrc),
            ('woe', WOEEncoder(cols=woe_cols_lrc), woe_cols_lrc),
            ('scaler', StandardScaler(), numeric_cols),
            ('count', CountEncoder(cols=count_cols_lrc, normalize=True), count_cols_lrc),
            ('passthrough', 'passthrough', ['no_checking'])
        ], 
        remainder='drop'
    )

    full_pipeline_lrc = Pipeline(steps=[
        ('feature_engineering', FeatureEngineer()),        
        ('encoder', encoding_pipeline_lrc)
    ])

    X_processed = full_pipeline_lrc.fit_transform(X, y)

    return X_processed, full_pipeline_lrc


def train_lrc(X_train, y_train, X_val, y_val, preprocessing_pipeline, cv=CV, random_state=RANDOM_STATE,
              tune=False, use_smote=False, evaluate=True, tune_threshold=False, log_model=True):
    
    # Generate distinctive run name based on parameters
    smote_tag = "SMOTE" if use_smote else "NoSMOTE"
    tune_tag = "Tuned" if tune else "Baseline"
    run_name = f"LRC_{tune_tag}_{smote_tag}_cv{cv}_rs{random_state}"
    
    # Set up MLflow experiment
    mlflow.set_experiment("credit-risk-logistic-regression")
    
    with mlflow.start_run(run_name=run_name):
        # Log initial parameters
        mlflow.log_params({
            "model_type": "LRC",
            "cv_folds": cv,
            "random_state": random_state,
            "tune_hyperparameters": tune,
            "use_smote": use_smote,
            "n_train_samples": len(X_train),
            "n_val_samples": len(X_val),
            "n_features_raw": X_train.shape[1],
        })

        lrc = LogisticRegression(random_state=random_state, max_iter=1000)

        X_train_processed = preprocessing_pipeline.fit_transform(X_train, y_train)
        X_val_processed = preprocessing_pipeline.transform(X_val)

        if tune:

            if use_smote:

                # Define the SMOTE sampler
                smote_sampler = SMOTE(random_state=random_state)

                lrc_space = {
                    'smote__k_neighbors': Integer(3, 10),
                    'smote__sampling_strategy': Real(0.5, 1.0, prior='uniform'),
                    'lrc__C': Real(0.1, 20, prior='log-uniform'),
                    'lrc__penalty': Categorical(['l2']),
                    'lrc__solver': Categorical(['liblinear']),
                    'lrc__max_iter': Integer(1000, 8000),
                    'lrc__class_weight': Categorical(['balanced', None])
                }
                # Create a pipeline that first applies SMOTE then fits the model
                # Use imblearn's Pipeline to support SMOTE
                model = ImbPipeline([
                    ('smote', smote_sampler),
                    ('lrc', lrc)
                ])

            else:

                lrc_space = {
                    'C': Real(0.1, 20, prior='log-uniform'),
                    'penalty': Categorical(['l2']),
                    'solver': Categorical(['liblinear']),
                    'max_iter': Integer(1000, 8000),
                    'class_weight': Categorical(['balanced', None])
                }

                model = lrc
            # Set up Bayesian Optimization with Cross-Validation
            lrc_tuned = BayesSearchCV(
                estimator=model,
                search_spaces=lrc_space,
                n_iter=25,
                scoring='roc_auc',
                cv=cv,
                random_state=random_state,
                n_jobs=-1,
                verbose=0,
                n_points=5
            )
            # Fit the model
            lrc_tuned.fit(X_train_processed, y_train)
            lrc = lrc_tuned.best_estimator_
            
            # Log tuning results
            mlflow.log_param("bayes_n_iter", 20)
            mlflow.log_metric("best_cv_score", lrc_tuned.best_score_)
            
            # Log best hyperparameters
            best_params = lrc_tuned.best_params_
            for param_name, param_value in best_params.items():
                # Handle None values and convert to string for MLflow
                if param_value is None:
                    mlflow.log_param(f"best_{param_name}", "None")
                else:
                    mlflow.log_param(f"best_{param_name}", param_value)

        else:

            lrc.fit(X_train_processed, y_train)
            
            # Log default Logistic Regression parameters
            mlflow.log_params({
                "lrc_C": lrc.C,
                "lrc_penalty": lrc.penalty,
                "lrc_solver": lrc.solver,
                "lrc_max_iter": lrc.max_iter,
                "lrc_class_weight": str(lrc.class_weight),
            })

        # Log number of features after preprocessing
        mlflow.log_param("n_features_processed", X_train_processed.shape[1])

        print(f"\n{'─' * 40}\n")
        print('Logistic Regression\n')
        
        # Create full pipeline for learning curve (preprocessing + model)
        # This avoids data leakage by fitting preprocessing inside each CV fold
        lc_pipeline = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('model', clone(lrc) if not hasattr(lrc, 'steps') else clone(lrc))
        ])
        
        # Get learning curve figure and metrics - use RAW X_train to avoid leakage
        lc_fig, lc_metrics = plot_learning_curve(lc_pipeline, X_train, y_train, cv=cv, random_state=random_state, show_plot=True)
        
        # Log learning curve metrics
        mlflow.log_metrics(lc_metrics)
        
        # Log learning curve figure
        mlflow.log_figure(lc_fig, "learning_curve.png")
        plt.close(lc_fig)

        print("\n")        

        if tune_threshold:
            cost_scorer = make_scorer(cost_scorer_fn, greater_is_better=False)
            tuned_cost_model = TunedThresholdClassifierCV(lrc, scoring=cost_scorer, cv=cv)
            tuned_cost_model.fit(X_train_processed, y_train)
            mlflow.log_param("tuned_decision_threshold", tuned_cost_model.best_threshold_)
            lrc = tuned_cost_model



        if evaluate:

            metrics_dict = evaluate_model(X_val_processed, y_val, lrc, model_name="Logistic Regression")
            
            # Log evaluation metrics
            mlflow.log_metrics({
                "val_roc_auc": metrics_dict['roc_auc'],
                "val_accuracy": metrics_dict['accuracy'],
                "val_f1": metrics_dict['f1'],
                "val_precision": metrics_dict['precision'],
                "val_cost": metrics_dict['cost'],
                "val_avg_cost": metrics_dict['avg_cost'],
            })
            
            # Generate predictions for plots
            y_val_pred = lrc.predict(X_val_processed)
            y_val_proba = lrc.predict_proba(X_val_processed)[:, 1]
            
            # Plot and log confusion matrix
            cm_fig = plot_confusion_matrix(y_val, y_val_pred, model_name="Logistic Regression", show_plot=False)
            mlflow.log_figure(cm_fig, "confusion_matrix.png")
            plt.close(cm_fig)
            
            # Plot and log precision-recall curve
            pr_fig, pr_metrics = plot_precision_recall_curve(y_val, y_val_proba, model_name="Logistic Regression", show_plot=False)
            mlflow.log_figure(pr_fig, "precision_recall_curve.png")
            mlflow.log_metric("val_average_precision", pr_metrics['average_precision'])
            plt.close(pr_fig)
            
            # Set tags for easy filtering
            mlflow.set_tags({
                "model_family": "Logistic Regression",
                "preprocessing": "FeatureEngineer+Encoders",
                "tuning_method": "BayesSearchCV" if tune else "None",
                "imbalance_handling": "SMOTE" if use_smote else "None",
            })

        # Create full pipeline (preprocessing + model) and log to MLflow
        full_pipeline = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('model', lrc)
        ])

        if log_model:
            mlflow.sklearn.log_model(
                full_pipeline,
                artifact_path="model",
                registered_model_name="credit-risk-lrc"
            )

    return lrc, preprocessing_pipeline
        

def parse_args():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression model for credit risk classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--tune", action="store_true", default=False,
                        help="Enable Bayesian hyperparameter tuning")
    parser.add_argument("--no-tune", dest="tune", action="store_false",
                        help="Disable hyperparameter tuning (use defaults)")
    parser.add_argument("--smote", action="store_true", default=False,
                        help="Use SMOTE for class imbalance")
    parser.add_argument("--no-smote", dest="smote", action="store_false",
                        help="Disable SMOTE")
    parser.add_argument("--tune-threshold", action="store_true", default=False,
                        help="Tune decision threshold using cost-sensitive optimization")
    parser.add_argument("--no-tune-threshold", dest="tune_threshold", action="store_false",
                        help="Use default threshold (0.5)")
    parser.add_argument("--evaluate", action="store_true", default=True,
                        help="Evaluate model on validation set")
    parser.add_argument("--no-evaluate", dest="evaluate", action="store_false",
                        help="Skip evaluation")
    parser.add_argument("--log-model", action="store_true", default=False,
                        help="Log model to MLflow registry")
    parser.add_argument("--no-log-model", dest="log_model", action="store_false",
                        help="Don't log model to registry")
    parser.add_argument("--cv", type=int, default=CV,
                        help="Number of cross-validation folds")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE,
                        help="Random seed for reproducibility")
    parser.add_argument("--val-size", type=int, default=150,
                        help="Validation set size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    home = Path.cwd()
    data_dir = home / "data"
    notebook_dir = home / "notebooks"
    df = pd.read_csv(data_dir / "processed" / "train_data.csv")
    sklearn.set_config(transform_output="pandas")

    X_train, X_val, y_train, y_val = train_test_split(
        df.drop(columns=["class"]),
        df["class"],
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=df["class"]
    )

    _, preprocessing_pipeline = lrc_preprocess(X_train, y_train)

    train_lrc(X_train, y_train, X_val, y_val, preprocessing_pipeline=preprocessing_pipeline,
              cv=args.cv, random_state=args.random_state,
              tune=args.tune, use_smote=args.smote, evaluate=args.evaluate,
              tune_threshold=args.tune_threshold, log_model=args.log_model)

