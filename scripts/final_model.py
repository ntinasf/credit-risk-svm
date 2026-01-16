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
    """Train a final ensemble model using VotingClassifier."""

    # Get preprocessing pipelines (fitted on training data)
    _, lrc_pipeline = lrc_preprocess(X_train, y_train)
    lrc_model = train_lrc(X_train, y_train, X_val=X_val, y_val=y_val, preprocessing_pipeline=lrc_pipeline,
                          cv=CV, random_state=RANDOM_STATE, tune=True,
                        use_smote=use_smote, evaluate=False, tune_threshold=True)

    _, rfc_pipeline = rf_preprocess(X_train, y_train)
    rfc_model = train_rfc(X_train, y_train, X_val=X_val, y_val=y_val, preprocessing_pipeline=rfc_pipeline,
                          cv=CV, random_state=RANDOM_STATE, tune=True,
                        evaluate=False, tune_threshold=True)

    _, svc_pipeline = svc_preprocess(X_train, y_train)
    svc_model = train_svc(X_train, y_train, X_val=X_val, y_val=y_val, preprocessing_pipeline=svc_pipeline, 
                          cv=CV, random_state=RANDOM_STATE, tune=True,
                        evaluate=False, tune_threshold=True,
                        use_smote=use_smote)

    # Create VotingClassifier
    voting_clf = VotingClassifier(
        estimators=[
            ('lrc', lrc_model),
            ('rfc', rfc_model),
            ('svc', svc_model)
        ],
        voting='soft',
        weights=[1, 1, 1],
        n_jobs=3
    )

    # Fit the ensemble model
    voting_clf.fit(X_train, y_train)

    if evaluate:
        print("Evaluating Final Ensemble Model on Validation Set:")
        evaluate_model(X_val, y_val, voting_clf, model_name="VotingClassifier Ensemble")

    return voting_clf

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
