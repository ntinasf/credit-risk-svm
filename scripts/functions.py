import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, 
    precision_score, roc_auc_score, precision_recall_curve, average_precision_score,
    PrecisionRecallDisplay
)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import OneHotEncoder, WOEEncoder, CountEncoder, TargetEncoder
from sklearn.model_selection import learning_curve  
from sklearn.svm import SVC

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for creating new features and transforming existing ones."""

    def __init__(self, duplicate_checking=False, duplicate_amount=False):
        self.cols_to_drop = [
            "other_debtors_guarantors",
            "telephone",
            #"foreign_worker",
            #"existing_credits_count",
            "people_liable_for_maintenance",
            #"installment_rate_pct_of_disp_income",
            #"personal_status_sex",
        ]
        self.duplicate_checking = duplicate_checking
        self.duplicate_amount = duplicate_amount
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        import pandas as pd
        import numpy as np
        
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
    
    This serves as a baseline for comparison against more sophisticated
    feature engineering approaches.
    """

    def __init__(self, categorical_cols=None, numerical_cols=None):
        """
        Parameters
        ----------
        categorical_cols : list, optional
            List of categorical column names. If None, auto-detects object/category dtypes.
        numerical_cols : list, optional
            List of numerical column names. If None, auto-detects numeric dtypes.
        """
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.preprocessor_ = None
        self._categorical_cols = None
        self._numerical_cols = None
        
    def fit(self, X, y=None):
        import pandas as pd
        
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
        import pandas as pd
        
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return self.preprocessor_.transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names after transformation."""
        if self.preprocessor_ is not None:
            return self.preprocessor_.get_feature_names_out(input_features)
        return None


def plot_learning_curve(estimator, X, y, cv=5, random_state=8, show_plot=True):
    """
    Diagnose if you need more data, better features, or tuning.
    Returns figure and metrics dict for MLflow logging.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        cv=cv,
        scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
        random_state=random_state
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    print(f"ROC AUC: {train_mean[-1]:.4f} ± {train_std[-1]:.4f} (train), {val_mean[-1]:.4f} ± {val_std[-1]:.4f} (validation)")
    
    # Diagnosis (print before showing plot to avoid blocking)
    gap = train_mean[-1] - val_mean[-1]
    
    print("\n🔍 DIAGNOSIS:")
    if val_mean[-1] < 0.70:
        print("❌ Low validation score - Need better features or different model")
    if gap > 0.10:
        print("⚠️  High variance (overfitting) - Need regularization or more data")
    if gap < 0.05 and val_mean[-1] < 0.75:
        print("⚠️  High bias (underfitting) - Need more complex model or better features")
    if val_mean[-1] > 0.75 and gap < 0.10:
        print("✅ Good bias-variance tradeoff - Ready for tuning")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, label='Training score')
    ax.plot(train_sizes, val_mean, label='Validation score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('AUC Score')
    ax.set_title('Learning Curve')
    ax.legend()
    ax.grid(True)
    
    if show_plot:
        plt.show()
    
    # Return figure and metrics for MLflow logging
    learning_curve_metrics = {
        'lc_train_auc_final': train_mean[-1],
        'lc_train_auc_std': train_std[-1],
        'lc_val_auc_final': val_mean[-1],
        'lc_val_auc_std': val_std[-1],
        'lc_bias_variance_gap': gap
    }
    
    return fig, learning_curve_metrics


def calculate_cost(y_true, y_pred, show_matrix=False, print_results=False):
    """
    Calculate total and average cost based on confusion matrix.
    
    Returns
    -------
    tuple : (total_cost, avg_cost)
    """
    # Define costs
    cost_fp = 5  # Cost of false positive
    cost_fn = 1  # Cost of false negative

    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    if show_matrix:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix :')
        plt.grid(False)
        plt.show()

    # Calculate total and average cost
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    n_samples = len(y_true)
    avg_cost = total_cost / n_samples
    
    if print_results:
        print(f"Total Cost: {total_cost} | Avg Cost: {avg_cost:.4f}")
    return total_cost, avg_cost


def cost_scorer_fn(y_true, y_pred):
    """
    Scorer-compatible function that returns only total cost.
    Use with: make_scorer(cost_scorer_fn, greater_is_better=False)
    """
    total_cost, _ = calculate_cost(y_true, y_pred)
    return total_cost


def plot_confusion_matrix(y_true, y_pred, model_name="Model", show_plot=True):
    """
    Plot confusion matrix for model evaluation.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model for the title
    show_plot : bool
        Whether to display the plot
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object for MLflow logging
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bad (0)', 'Good (1)'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.grid(False)
    
    # Add text annotations with counts and percentages
    total = cm.sum()
    tn, fp, fn, tp = cm.ravel()
    
    # Add summary text below the matrix
    summary_text = (
        f"TN: {tn} ({tn/total:.1%}) | FP: {fp} ({fp/total:.1%})\n"
        f"FN: {fn} ({fn/total:.1%}) | TP: {tp} ({tp/total:.1%})"
    )
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if show_plot:
        plt.show()
    
    return fig


def plot_precision_recall_curve(y_true, y_pred_proba, model_name="Model", show_plot=True):
    """
    Plot precision-recall curve for model evaluation.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities for the positive class
    model_name : str
        Name of the model for the title
    show_plot : bool
        Whether to display the plot
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object for MLflow logging
    metrics : dict
        Dictionary containing average precision score
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the PR curve
    ax.plot(recall, precision, 'b-', linewidth=2, 
            label=f'{model_name} (AP = {avg_precision:.3f})')
    
    # Add baseline (random classifier)
    baseline = y_true.sum() / len(y_true)
    ax.axhline(y=baseline, color='r', linestyle='--', 
               label=f'Baseline (No Skill) = {baseline:.3f}')
    
    # Fill area under curve
    ax.fill_between(recall, precision, alpha=0.2)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    metrics = {
        'average_precision': avg_precision
    }
    
    return fig, metrics


def evaluate_model(X_val, y_val, fitted_model_pipeline, model_name):

    y_pred = fitted_model_pipeline.predict(X_val)
    y_pred_proba = fitted_model_pipeline.predict_proba(X_val)[:, 1]
    
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    cost, avg_cost = calculate_cost(y_val, y_pred, show_matrix=False)
    
    print(f"\n{'─' * 40}")
    print(f'Evaluation Metrics for {model_name}:')
    print(f"   ROC AUC:  {roc_auc:.4f}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1:       {f1:.4f}")
    print(f"   Precision:{precision:.4f}\n")
    print(f"   Total Cost: {cost} | Avg Cost: {avg_cost:.4f}\n")
    
    return {
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'cost': cost,
        'avg_cost': avg_cost
    }

def test_model(X_train, y_train, X_test, y_test, model_name="Model", tune_hyperparameters=False):
    """
    Test a model on the test set using various random seeds.
    Note: This function uses lazy imports to avoid circular dependencies.
    """
    print(f"\n{'=' * 40}\n")
    print(f'Testing {model_name} on Test Set Using Various Seeds:\n')

    aucs = []
    accuracies = []
    f1s = []
    precisions = []
    costs = []
    avg_costs = []
    
    if model_name == "SVC":
        # Lazy import to avoid circular dependency
        from scripts.train_svc import train_svc
        
        for seed in range(5):
            print(f"--- Seed {seed} ---")
            svc_model, pipeline_svc = train_svc(X_train, y_train, X_test, y_test, evaluate=True, tune=tune_hyperparameters, random_state=seed)
            print("\n")
            X_test_processed = pipeline_svc.transform(X_test)

            scores = evaluate_model(X_test_processed, y_test, svc_model, model_name=model_name)
            accuracies.append(scores['accuracy'])
            f1s.append(scores['f1'])
            precisions.append(scores['precision'])
            aucs.append(scores['roc_auc'])
            costs.append(scores['cost'])
            avg_costs.append(scores['avg_cost'])
            print("\n")

    if model_name == 'logistic_regression':
        pass  # Placeholder for logistic regression testing

    print(f"\n{'=' * 40}\n")
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)
    mean_precision = np.mean(precisions)
    std_precision = np.std(precisions)
    mean_cost = np.mean(costs)
    std_cost = np.std(costs)
    mean_avg_cost = np.mean(avg_costs)
    std_avg_cost = np.std(avg_costs)

    print(f'Final Test Set Performance for {model_name} over various seeds:')
    print(f"   ROC AUC:  {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"   Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"   F1:       {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"   Precision:{mean_precision:.4f} ± {std_precision:.4f}")
    print(f"   Total Cost: {mean_cost:.2f} ± {std_cost:.2f} | Avg Cost: {mean_avg_cost:.4f} ± {std_avg_cost:.4f}\n")
    return {
        'roc_auc': (mean_auc, std_auc),
        'accuracy': (mean_accuracy, std_accuracy),
        'f1': (mean_f1, std_f1),
        'precision': (mean_precision, std_precision),
        'cost': (mean_cost, std_cost),
        'avg_cost': (mean_avg_cost, std_avg_cost)
    }

