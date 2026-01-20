# Credit Risk Classification with Machine Learning

This is a personal portfolio project that builds an ensemble credit risk classifier using the [German Credit Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data). The project demonstrates end-to-end machine learning workflow including data preprocessing, feature engineering, model training, hyperparameter tuning and experiment tracking with MLflow. The final product is a soft voting ensemble of three classifiers optimized for a business-relevant cost function. You can try out the model via a Streamlit web application.

 **[Try the Live Demo](https://your-streamlit-app.streamlit.app)**

---

## Project Overview

This project predicts credit risk (good/bad) for loan applicants using three models:
- **Support Vector Classifier (SVC)** with RBF kernel
- **Logistic Regression** with L2 regularization  
- **Random Forest Classifier**

Each model has its own preprocessing pipeline optimized for its characteristics and the final prediction uses soft voting across all three models.

### Key Features

- **Custom Feature Engineering**: Domain-specific transformations, numerical features transformations and categorizations, rare label consolidations
- **Multiple Encoding Strategies**: WOE, Target, Count and One-Hot encoding tailored per model
- **Class Imbalance Handling**: SMOTE/SVMSMOTE and cost-sensitive learning
- **Cost-Sensitive Optimization**: Custom cost function (FP=5, FN=1) reflecting real-world business impact per the dataset documentation
- **Threshold Tuning**: For individual models using `TunedThresholdClassifierCV` or fixed class balance, manual optimization for the ensemble
- **Experiment Tracking**: Full MLflow integration with metrics, parameters, artifacts and model versioning

---

## Project Structure
```
credit-risk-svm/
├── data/
│   ├── raw/                    # Original German Credit dataset
│   └── processed/              # Cleaned and split data
│       ├── german_credit.csv
│       ├── train_data.csv
│       └── test_data.csv
├── notebooks/
│   ├── eda.ipynb              # Exploratory Data Analysis
│   └── feature_engineering.ipynb # Engineered features versus simple approach
│   └── hyp_tuning.ipynb       # Hyperparameter tuning using Bayesian Optimization
│   └── voting_clf.ipynb       # Final performance evaluation
├── scripts/
│   ├── process_data.py        # Transform raw data to processed format
│   ├── functions.py           # Shared utilities & FeatureEngineer transformer
│   ├── eda_toolkit.py         # EDA helper functions
│   ├── eda_toolkit_index.py   # Documentation for eda_toolkit.py
│   ├── split_data.py          # Stratified hash-based train/test split
│   ├── train_svc.py           # SVC training script
│   ├── train_lrc.py           # Logistic Regression training script
│   ├── train_rf.py            # Random Forest training script
│   ├── final_model.py         # Ensemble training orchestration
│   └── score_model.py         # Production scoring with EnsembleScorer
├── mlruns/                    # MLflow experiment artifacts
├── mlflow.db                  # MLflow tracking database
├── streamlit_app/             # Streamlit web application
├── requirements.txt
├── export_models.py          # Export trained models for deployment
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ntinasf/credit-risk-svm.git
   cd credit-risk-svm
   ```

2. **Create virtual environment**
   ```bash
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate
   
   # Or using pip
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   # Using uv
   uv pip install -r requirements.txt
   
   # Or using pip
   pip install -r requirements.txt
   ```

---

## Reproducing the Results

### Step 1: Process Raw Data

Process the raw German Credit dataset:

```bash
python scripts/process_data.py
```

### Step 2: Split the Data

Create train/test split using stratified hash-based splitting:

```bash
python scripts/split_data.py --test-size 0.15
```

This creates `train_data.csv` and `test_data.csv` in `data/processed/`.

### Step 3: Train Individual Models

Train each model separately with MLflow tracking:

```bash
# Train SVC with hyperparameter tuning and SMOTE
python scripts/train_svc.py

# Train Logistic Regression
python scripts/train_lrc.py

# Train Random Forest
python scripts/train_rf.py
```

### Step 4: Train the Ensemble

Train all models and register the ensemble:

```bash
python scripts/final_model.py
```

### Step 5: View Experiments in MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://localhost:5000 to view experiments, compare runs, and inspect artifacts.

### Step 6: Score New Data

```bash
python scripts/score_model.py
```

---

## Model Performance after Threshold Tuning

| Model | ROC AUC | Avg Cost | 
|-------|---------|----------|-----------|
| SVC (SMOTE) | ~0.81 | ~0.43 | 
| Logistic Regression | ~0.80 | ~0.49 |
| Random Forest | ~0.82 | ~0.53 | 
| **Ensemble** | **~0.79** | **~0.43** | Soft Voting |

*Note: Results may vary slightly between runs.*

---

## Cost Function

The model optimizes for a business-realistic cost function as mentioned in the German Credit dataset documentation:

- **False Negative (FN)**: Rejecting a good customer = **1 cost unit** (lost business opportunity)
- **False Positive (FP)**: Accepting a bad customer = **5 cost units** (potential default loss)

---

## Technical Details

### Feature Engineering

The `FeatureEngineer` transformer applies:
- **Monthly Burden**: `credit_amount / duration_months` (log-transformed)
- **Duration Bins**: Quintile-based categorization
- **Age Groups**: Young, Early Career, Prime, Mature
- **Category Consolidation**: Merging sparse categories for better generalization

### Preprocessing Pipelines

Each model uses a tailored encoding strategy:

| Model | Encoding Strategy |
|-------|-------------------|
| SVC | WOE + Target + Count + One-Hot |
| LRC | WOE + Count + One-Hot |
| RFC | One-Hot + Count + Target |

### MLflow Artifacts

Each training run logs:
- `learning_curve.png` - Bias-variance diagnostic
- `confusion_matrix.png` - Classification results
- `precision_recall_curve.png` - PR curve with average precision
- Full pipeline model (preprocessing + classifier)

*All the above can be found in the MLflow UI under each run's artifacts.*

---

## 🌐 Streamlit App

A Streamlit web application is provided for interactive model testing. Users are be able to:

- Sample data from the unseen test set and get model prediction along with confidence scores
- Input custom applicant information according to the feature schema

---

## Dataset Information

- **Samples**: 1,000 loan applicants
- **Features**: 20 attributes (7 numerical, 13 categorical)
- **Target**: Binary (Good=0, Bad=1)
- **Class Distribution**: ~70% Good, ~30% Bad

---

## 📝 License

This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the German Credit Dataset
