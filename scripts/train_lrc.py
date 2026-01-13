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
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from category_encoders import OneHotEncoder, WOEEncoder, CountEncoder
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from imblearn.over_sampling import SMOTE
import warnings
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.metrics import make_scorer

# Import from functions module (now works with relative path)
from scripts.functions import plot_learning_curve, FeatureEngineer, evaluate_model, calculate_cost


warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 8
CV = 10

def lrc_preprocess(X, y):
    """Preprocess the data for LRC model."""

    one_hot_cols_lrc = [
        'purpose',
        'job'
    ]

    woe_cols_lrc = [
        'checking_account_status',
        'credit_history',
        'savings_account_bonds',
        'present_employment_since',
        'housing',
        'duration_bins'
    ]

    count_cols_lrc = [
        'property',
        'other_installment_plans',
        'credit_amount_bins',
        'age_group'
    ]

    numeric_cols = [
        'monthly_burden_log',
    ]

    encoding_pipeline_lrc = ColumnTransformer(transformers=[
        ('one_hot', OneHotEncoder(cols=one_hot_cols_lrc, use_cat_names=True), one_hot_cols_lrc),
        ('woe', WOEEncoder(cols=woe_cols_lrc), woe_cols_lrc),
        ('scaler', StandardScaler(), numeric_cols),
        ('count', CountEncoder(cols=count_cols_lrc), count_cols_lrc)
    ], 
    remainder='passthrough'
    )

    full_pipeline_lrc = Pipeline(steps=[
        ('feature_engineering', FeatureEngineer()),        
        ('encoder', encoding_pipeline_lrc)
    ])

    X_processed = full_pipeline_lrc.fit_transform(X, y)

    return X_processed, full_pipeline_lrc


def train_lrc(X_train, y_train, X_val, y_val, preprocessing_pipeline, cv=CV, random_state=RANDOM_STATE,
              tune=False, use_smote=False, evaluate=True, tune_threshold=False):
    
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
                n_iter=20,
                scoring='roc_auc',
                cv=cv,
                random_state=random_state,
                n_jobs=-1,
                verbose=0
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
        
        # Get learning curve figure and metrics
        lc_fig, lc_metrics = plot_learning_curve(lrc, X_train_processed, y_train, cv=cv, random_state=random_state, show_plot=True)
        
        # Log learning curve metrics
        mlflow.log_metrics(lc_metrics)
        
        # Log learning curve figure
        mlflow.log_figure(lc_fig, "learning_curve.png")
        plt.close(lc_fig)

        print("\n")        

        if tune_threshold:
            cost_scorer = make_scorer(calculate_cost, greater_is_better=False)
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
            })
            
            # Set tags for easy filtering
            mlflow.set_tags({
                "model_family": "SVM",
                "preprocessing": "FeatureEngineer+Encoders",
                "tuning_method": "BayesSearchCV" if tune else "None",
                "imbalance_handling": "SVMSMOTE" if use_smote else "None",
            })

    return lrc
        

if __name__ == "__main__":

    home = Path.cwd()
    data_dir = home / "data"
    notebook_dir = home / "notebooks"
    df = pd.read_csv(data_dir / "processed" / "german_credit.csv")
    sklearn.set_config(transform_output="pandas")

    X_temp, X_test, y_temp, y_test = train_test_split(
        df.drop(columns=["class"]),
        df["class"],
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=df["class"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=len(X_test),
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    _, preprocessing_pipeline = lrc_preprocess(X_train, y_train)

    train_lrc(X_train, y_train, X_val, y_val, preprocessing_pipeline=preprocessing_pipeline, 
              tune=True, use_smote=False, evaluate=True, tune_threshold=False)

