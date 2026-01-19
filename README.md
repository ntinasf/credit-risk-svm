# Credit Risk Classification with Machine Learning

A personal portfolio project that builds an ensemble credit risk classifier using the German Credit Dataset. The project demonstrates end-to-end machine learning workflow including feature engineering, model training, hyperparameter tuning, and experiment tracking with MLflow.

🚀 **[Try the Live Demo](https://your-streamlit-app.streamlit.app)**

**[Project page]**(https://ntinasf.github.io/projects/credit-risk-classifier)
---

## 📋 Project Overview

This project predicts credit risk (good/bad) for loan applicants using an ensemble of three models:
- **Support Vector Classifier (SVC)** with RBF kernel
- **Logistic Regression** with L2 regularization  
- **Random Forest Classifier**

Each model has its own preprocessing pipeline optimized for its characteristics, and the final prediction uses soft voting across all three models.

### Key Features

- 🔧 **Custom Feature Engineering**: Domain-specific transformations including monthly burden ratios, age groups, and category consolidation
- 📊 **Multiple Encoding Strategies**: WOE, Target, Count, and One-Hot encoding tailored per model
- ⚖️ **Class Imbalance Handling**: SMOTE/SVMSMOTE and cost-sensitive learning
- 🎯 **Cost-Sensitive Optimization**: Custom cost function (FP=5, FN=1) reflecting real-world business impact
- 📈 **Threshold Tuning**: `TunedThresholdClassifierCV` for optimal decision boundaries
- 🔬 **Experiment Tracking**: Full MLflow integration with metrics, parameters, and artifacts

---

## 📁 Project Structure

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
│   └── feature_engineering.ipynb
├── scripts/
│   ├── functions.py           # Shared utilities & FeatureEngineer transformer
│   ├── split_data.py          # Stratified hash-based train/test split
│   ├── train_svc.py           # SVC training with MLflow
│   ├── train_lrc.py           # Logistic Regression training
│   ├── train_rf.py            # Random Forest training
│   ├── final_model.py         # Ensemble training orchestration
│   └── score_model.py         # Production scoring with EnsembleScorer
├── mlruns/                    # MLflow experiment artifacts
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

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

## 🔄 Reproducing the Results

### Step 1: Split the Data

Create train/test split using stratified hash-based splitting:

```bash
python scripts/split_data.py --test-size 0.15
```

This creates `train_data.csv` and `test_data.csv` in `data/processed/`.

### Step 2: Train Individual Models

Train each model separately with MLflow tracking:

```bash
# Train SVC with hyperparameter tuning and SMOTE
python scripts/train_svc.py

# Train Logistic Regression
python scripts/train_lrc.py

# Train Random Forest
python scripts/train_rf.py
```

### Step 3: Train the Ensemble

Train all models and register the ensemble:

```bash
python scripts/final_model.py
```

### Step 4: View Experiments in MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://localhost:5000 to view experiments, compare runs, and inspect artifacts.

### Step 5: Score New Data

```bash
python scripts/score_model.py
```

---

## 📊 Model Performance

| Model | ROC AUC | Avg Cost | Threshold |
|-------|---------|----------|-----------|
| SVC (SMOTE) | ~0.78 | ~0.45 | Tuned |
| Logistic Regression | ~0.77 | ~0.48 | Tuned |
| Random Forest | ~0.76 | ~0.46 | Tuned |
| **Ensemble** | **~0.79** | **~0.43** | Soft Voting |

*Note: Results may vary slightly between runs.*

---

## 🎯 Cost Function

The model optimizes for a business-realistic cost function:

- **False Negative (FN)**: Rejecting a good customer = **1 units** (lost business opportunity)
- **False Positive (FP)**: Accepting a bad customer = **5 unit** (potential default loss)

---

## 🛠️ Technical Details

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

---

## 🌐 Streamlit App

A Streamlit web application is being developed for interactive credit risk prediction. Users will be able to:

- Input applicant information through a user-friendly form
- Get instant credit risk predictions with confidence scores
- View feature importance and model explanations
- Compare predictions across ensemble models

**Status**: *Coming Soon*

---

## 📚 Dataset

**German Credit Dataset** by Prof. Dr. Hans Hofmann, University of Hamburg

- **Samples**: 1,000 loan applicants
- **Features**: 20 attributes (7 numerical, 13 categorical)
- **Target**: Binary (Good=0, Bad=1)
- **Class Distribution**: ~70% Good, ~30% Bad

[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

---

## 📝 License

This project is for educational and portfolio purposes.

---

## 👤 Author

**Fotis N.**

- GitHub: [@ntinasf](https://github.com/ntinasf)

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the German Credit Dataset
- scikit-learn, MLflow, and the open-source ML community
