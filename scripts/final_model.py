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
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from category_encoders import OneHotEncoder, WOEEncoder, CountEncoder, TargetEncoder
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from imblearn.over_sampling import SVMSMOTE
import warnings
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.metrics import make_scorer

# Import from functions module (now works with relative path)
from scripts.functions import plot_learning_curve, FeatureEngineer, evaluate_model, calculate_cost
from scripts.train_lrc import lrc_preprocess, train_lrc
from scripts.train_rf import rf_preprocess, train_rfc
from scripts.train_svc import svc_preprocess, train_svc

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=FutureWarning)

def train_final_model(X_train, y_train, X_val, y_val, 
                      RANDOM_STATE=8, CV=10, use_smote=True, evaluate=True):
    """Train all individual models and register them to MLflow.
    
    Each model is trained with its own preprocessing pipeline and registered
    to MLflow model registry. The ensemble scoring is handled separately
    by the EnsembleScorer class in score_model.py.
    """

    print("=" * 60)
    print("Training Logistic Regression Model...")
    print("=" * 60)
    
    # Train LRC with its preprocessing
    _, lrc_pipeline = lrc_preprocess(X_train, y_train)
    lrc_model, _ = train_lrc(X_train, y_train, X_val=X_val, y_val=y_val, 
                              preprocessing_pipeline=lrc_pipeline,
                              cv=CV, random_state=RANDOM_STATE, tune=True,
                              use_smote=use_smote, evaluate=evaluate, tune_threshold=True)

    print("\n" + "=" * 60)
    print("Training Random Forest Model...")
    print("=" * 60)
    
    # Train RFC with its preprocessing
    _, rfc_pipeline = rf_preprocess(X_train, y_train)
    rfc_model, _ = train_rfc(X_train, y_train, X_val=X_val, y_val=y_val, 
                              preprocessing_pipeline=rfc_pipeline,
                              cv=CV, random_state=RANDOM_STATE, tune=True,
                              evaluate=evaluate, tune_threshold=True)

    print("\n" + "=" * 60)
    print("Training SVC Model...")
    print("=" * 60)
    
    # Train SVC with its preprocessing
    _, svc_pipeline = svc_preprocess(X_train, y_train)
    svc_model, _ = train_svc(X_train, y_train, X_val=X_val, y_val=y_val, 
                              preprocessing_pipeline=svc_pipeline, 
                              cv=CV, random_state=RANDOM_STATE, tune=True,
                              evaluate=evaluate, tune_threshold=True,
                              use_smote=use_smote)

    print("\n" + "=" * 60)
    print("All models trained and registered to MLflow!")
    print("=" * 60)
    print("\nRegistered models:")
    print("  - credit-risk-lrc")
    print("  - credit-risk-rfc") 
    print("  - credit-risk-svc")
    print("\nUse score_model.py to create ensemble predictions on new data.")

    return {
        'lrc': lrc_model,
        'rfc': rfc_model,
        'svc': svc_model
    }

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
        random_state=8,
        stratify=df["class"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=len(X_test),
        random_state=8,
        stratify=y_temp
    )

    train_final_model(X_train, y_train, X_val, y_val, use_smote=True, evaluate=True)
