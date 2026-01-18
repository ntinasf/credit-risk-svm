import sys
from pathlib import Path
import sklearn
from sklearn.model_selection import train_test_split

# Add parent directory to path to enable imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from category_encoders import OneHotEncoder, CountEncoder, TargetEncoder, OrdinalEncoder
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import warnings
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.metrics import make_scorer

# Import from functions module (now works with relative path)
from scripts.functions import (plot_learning_curve, FeatureEngineer, evaluate_model, 
                               cost_scorer_fn, plot_confusion_matrix, plot_precision_recall_curve)


warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 8
CV = 5

def rf_preprocess(X, y):
    """Preprocess the data for Random Forest model."""

    one_hot_rfc = [
        'credit_history', 
        'purpose',
        'savings_account_bonds',
        'present_employment_since',
        'personal_status_sex',
        'other_installment_plans',
        'housing',
        
        'foreign_worker',

        #'checking_2', 
    ]

    ordinal_cols_rfc = [
        "installment_rate_pct_of_disp_income",
        'job',
        'personal_status_2',
        'checking_account_status',
        'age_group',
    ]

    target_cols_rfc = [
        'property',
    ]

    numeric_cols = [
        #'duration_months',
        'credit_log',
        'duration_to_age_ratio',
    ]
    
    passthrough_cols = [
        'no_checking',
        'present_residence_since',
        "existing_credits_count",
        'monthly_burden',
    ]

    encoding_pipeline_rfc = ColumnTransformer(
        transformers=[
            ('one_hot', OneHotEncoder(cols=one_hot_rfc, use_cat_names=True), one_hot_rfc),
            ('ordinal', OrdinalEncoder(cols=ordinal_cols_rfc), ordinal_cols_rfc),
            ('target', TargetEncoder(cols=target_cols_rfc, smoothing=5), target_cols_rfc),
            ('scaler', StandardScaler(), numeric_cols),
            ('passthrough', 'passthrough', passthrough_cols)
        ],
        remainder='drop'  # or 'passthrough' if you want to keep other columns
    )

    # Full pipeline combining feature engineering + encoding
    full_pipeline_rfc = Pipeline([
        ('feature_engineer', FeatureEngineer(duplicate_checking=True)),
        ('encoder', encoding_pipeline_rfc)
    ])

    X_processed = full_pipeline_rfc.fit_transform(X, y)

    return X_processed, full_pipeline_rfc


def train_rfc(X_train, y_train, X_val, y_val, preprocessing_pipeline, cv=CV, random_state=RANDOM_STATE,
              tune=False, evaluate=True, tune_threshold=False, log_model=False):
    
    # Generate distinctive run name based on parameters
    smote_tag = "NoSMOTE"
    tune_tag = "Tuned" if tune else "Baseline"
    run_name = f"RFC_{tune_tag}_{smote_tag}_cv{cv}_rs{random_state}"
    
    # Set up MLflow experiment
    mlflow.set_experiment("credit-risk-rfc")
    
    with mlflow.start_run(run_name=run_name):
        # Log initial parameters
        mlflow.log_params({
            "model_type": "RFC",
            "cv_folds": cv,
            "random_state": random_state,
            "tune_hyperparameters": tune,
            "use_smote": False,
            "n_train_samples": len(X_train),
            "n_val_samples": len(X_val),
            "n_features_raw": X_train.shape[1],
        })

        rfc = RandomForestClassifier(random_state=random_state, n_jobs=-1,
                                     class_weight={0: 1, 1: 5})

        X_train_processed = preprocessing_pipeline.fit_transform(X_train, y_train)
        X_val_processed = preprocessing_pipeline.transform(X_val)

        if tune:

            ran_forest_space = {
                    # More trees = more stable, less overfitting
                    'n_estimators': Integer(400, 800),
                    # SHALLOW trees are key to reducing overfitting
                    'max_depth': Integer(3, 10),
                    # Higher = more regularization
                    'min_samples_split': Integer(20, 80),
                    'min_samples_leaf': Integer(10, 40),
                    # Fewer features per split = less overfitting
                    'max_features': Categorical(['sqrt', 'log2', 0.3, 0.5]),
                    # Bootstrap sample size - great regularizer
                    'max_samples': Real(0.5, 0.9, prior='uniform'),
                    # Cost-complexity pruning
                    'ccp_alpha': Real(0.0, 0.02, prior='uniform'),
                }

            # Set up Bayesian Optimization with Cross-Validation
            rfc_tuned = BayesSearchCV(
                estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                search_spaces=ran_forest_space,
                n_iter=40,  # More iterations for larger search space
                scoring='roc_auc',
                cv=8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=0,
                n_points=5
            )

            # Fit the model
            rfc_tuned.fit(X_train_processed, y_train)
            rfc = rfc_tuned.best_estimator_
            
            # Log tuning results
            mlflow.log_param("bayes_n_iter", 40)
            mlflow.log_metric("best_cv_score", rfc_tuned.best_score_)
            
            # Log best hyperparameters
            best_params = rfc_tuned.best_params_
            for param_name, param_value in best_params.items():
                # Handle None values and convert to string for MLflow
                if param_value is None:
                    mlflow.log_param(f"best_{param_name}", "None")
                else:
                    mlflow.log_param(f"best_{param_name}", param_value)

        else:

            rfc.fit(X_train_processed, y_train)
            
            # Log default RFC parameters
            mlflow.log_params({
                "rfc_n_estimators": rfc.n_estimators,
                "rfc_max_depth": rfc.max_depth,
                "rfc_min_samples_split": rfc.min_samples_split,
                "rfc_min_samples_leaf": rfc.min_samples_leaf,
                "rfc_max_features": rfc.max_features,
                "rfc_class_weight": str(rfc.class_weight),
            })

        # Log number of features after preprocessing
        mlflow.log_param("n_features_processed", X_train_processed.shape[1])

        print(f"\n{'─' * 40}\n")
        print('Random Forest Classifier\n')
        
        # Create full pipeline for learning curve (preprocessing + model)
        # This avoids data leakage by fitting preprocessing inside each CV fold
        from sklearn.base import clone
        lc_pipeline = Pipeline([
            ('preprocessing', clone(preprocessing_pipeline)),
            ('model', clone(rfc) if not hasattr(rfc, 'steps') else clone(rfc))
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
            tuned_cost_model = TunedThresholdClassifierCV(rfc, scoring=cost_scorer, cv=cv)
            tuned_cost_model.fit(X_train_processed, y_train)
            mlflow.log_param("tuned_decision_threshold", tuned_cost_model.best_threshold_)
            rfc = tuned_cost_model



        if evaluate:

            metrics_dict = evaluate_model(X_val_processed, y_val, rfc, model_name="Random Forest Classifier")
            
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
            y_val_pred = rfc.predict(X_val_processed)
            y_val_proba = rfc.predict_proba(X_val_processed)[:, 1]
            
            # Plot and log confusion matrix
            cm_fig = plot_confusion_matrix(y_val, y_val_pred, model_name="Random Forest", show_plot=False)
            mlflow.log_figure(cm_fig, "confusion_matrix.png")
            plt.close(cm_fig)
            
            # Plot and log precision-recall curve
            pr_fig, pr_metrics = plot_precision_recall_curve(y_val, y_val_proba, model_name="Random Forest", show_plot=False)
            mlflow.log_figure(pr_fig, "precision_recall_curve.png")
            mlflow.log_metric("val_average_precision", pr_metrics['average_precision'])
            plt.close(pr_fig)
            
            # Set tags for easy filtering
            mlflow.set_tags({
                "model_family": "Random Forest",
                "preprocessing": "FeatureEngineer+Encoders",
                "tuning_method": "BayesSearchCV" if tune else "None",
                "imbalance_handling": "Cost Sensitive",
            })

        # Create full pipeline (preprocessing + model) and log to MLflow
        full_pipeline = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('model', rfc)
        ])

        if log_model:
            mlflow.sklearn.log_model(
                full_pipeline,
                artifact_path="model",
                registered_model_name="credit-risk-rfc"
            )

    return rfc, preprocessing_pipeline
        

if __name__ == "__main__":

    home = Path.cwd()
    data_dir = home / "data"
    notebook_dir = home / "notebooks"
    df = pd.read_csv(data_dir / "processed" / "train_data.csv")
    sklearn.set_config(transform_output="pandas")

    X_train, X_val, y_train, y_val = train_test_split(
        df.drop(columns=["class"]),
        df["class"],
        test_size=150,
        random_state=RANDOM_STATE,
        stratify=df["class"]
    )

    _, preprocessing_pipeline = rf_preprocess(X_train, y_train)

    train_rfc(X_train, y_train, X_val, y_val, preprocessing_pipeline=preprocessing_pipeline, 
              tune=True, evaluate=True, tune_threshold=True, log_model=True)

