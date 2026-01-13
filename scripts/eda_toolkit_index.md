# EDA Toolkit Index

Quick reference for `eda_toolkit.py` - functions for initial exploratory data analysis.

## Functions Overview

| Function | Input | Purpose |
|----------|-------|---------|
| `analyze_numerical()` | 1 numerical Series | Univariate analysis for continuous variables |
| `analyze_categorical()` | 1 categorical Series | Univariate analysis for categorical variables |
| `analyze_numerical_numerical()` | 2 numerical Series | Correlation analysis between continuous variables |
| `analyze_categorical_numerical()` | 1 categorical + 1 numerical Series | Group comparisons (e.g., ANOVA) |
| `analyze_categorical_categorical()` | 2 categorical Series | Chi-square test of independence |
| `quick_eda()` | 1 Series (any type) | Auto-detect type and run appropriate analysis |

---

## When to Use Parametric vs Non-Parametric Tests

The `parametric` argument is available in bivariate functions involving numerical data.

### Use **Parametric** (`parametric=True`) when:

- Data is approximately **normally distributed** (check Q-Q plot, Shapiro-Wilk test)
- Sample sizes are **large** (n > 30 per group, Central Limit Theorem applies)
- Variances are **roughly equal** across groups (Levene's test p > 0.05)
- The relationship is **linear** (for correlation)

**Parametric tests used:**
- `analyze_numerical_numerical`: Pearson correlation
- `analyze_categorical_numerical`: Independent t-test (2 groups) or One-way ANOVA (3+ groups)

### Use **Non-Parametric** (`parametric=False`) when:

- Data is **non-normal** or heavily skewed
- **Outliers** are present and influential
- Sample sizes are **small** (n < 30)
- Data is **ordinal** (ranked categories like ratings 1-5)
- Variances are **unequal** across groups
- The relationship is **monotonic but not linear**

**Non-parametric tests used:**
- `analyze_numerical_numerical`: Spearman correlation
- `analyze_categorical_numerical`: Mann-Whitney U (2 groups) or Kruskal-Wallis H (3+ groups)

---

## Quick Decision Flowchart

```
Is your numerical data normally distributed?
│
├── YES → Use parametric=True
│         (More statistical power if assumptions are met)
│
└── NO or UNSURE → Use parametric=False
                   (Safer, more robust to violations)
```

---

## Usage Examples

```python
import pandas as pd
from eda_toolkit import (
    analyze_numerical,
    analyze_categorical,
    analyze_numerical_numerical,
    analyze_categorical_numerical,
    analyze_categorical_categorical
)

# Load your data
df = pd.read_csv('your_data.csv')

# Univariate analysis
analyze_numerical(df['age'])
analyze_categorical(df['gender'])

# Bivariate analysis
analyze_numerical_numerical(df['height'], df['weight'], parametric=True)
analyze_categorical_numerical(df['gender'], df['salary'], parametric=False)
analyze_categorical_categorical(df['education'], df['employment_status'])
```

---

## Output

Each function:
1. **Prints** formatted statistics to the console
2. **Displays** appropriate visualizations
3. **Returns** a dictionary with all computed values for programmatic use

Set `show_plot=False` to suppress visualizations.
