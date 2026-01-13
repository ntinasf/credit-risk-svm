"""
EDA Toolkit - Univariate and Bivariate Exploratory Data Analysis
Author: Data Science Toolkit
Description: Functions for initial EDA of categorical and numerical variables.
             Designed to work with pandas Series (not necessarily from the same DataFrame).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (
    ttest_ind, mannwhitneyu, f_oneway, kruskal,
    chi2_contingency, pearsonr, spearmanr, 
    shapiro, normaltest, levene
)
from typing import Optional, Tuple, Dict, Union
import warnings

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

# Accessible, colorblind-friendly palette
PALETTE_CATEGORICAL = ['#0077BB', '#EE7733', '#009988', '#CC3311', '#33BBEE', '#EE3377', '#BBBBBB']
PALETTE_SEQUENTIAL = 'viridis'
PALETTE_DIVERGING = 'RdBu_r'

# Set consistent style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(PALETTE_CATEGORICAL)
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def _print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _print_subheader(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def _get_series_name(series: pd.Series, default: str = "Variable") -> str:
    """Get a display name for a series."""
    return series.name if series.name is not None else default


# =============================================================================
# UNIVARIATE ANALYSIS FUNCTIONS
# =============================================================================

def analyze_numerical(
    series: pd.Series,
    figsize: Tuple[int, int] = (12, 8),
    bins: int = 30,
    show_plot: bool = True
) -> Dict:
    """
    Comprehensive univariate analysis for a numerical variable.
    
    Parameters
    ----------
    series : pd.Series
        Numerical pandas Series to analyze.
    figsize : tuple, optional
        Figure size (width, height). Default is (12, 8).
    bins : int, optional
        Number of bins for histogram. Default is 30.
    show_plot : bool, optional
        Whether to display plots. Default is True.
    
    Returns
    -------
    dict
        Dictionary containing all computed statistics.
    
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name='values')
    >>> results = analyze_numerical(data)
    """
    var_name = _get_series_name(series, "Numerical Variable")
    data = series.dropna()
    
    if len(data) == 0:
        print(f"⚠️ Warning: '{var_name}' has no non-null values.")
        return {}
    
    _print_header(f"UNIVARIATE ANALYSIS: {var_name}")
    
    # Basic statistics
    results = {
        'variable': var_name,
        'count': len(data),
        'missing': series.isna().sum(),
        'missing_pct': (series.isna().sum() / len(series)) * 100,
        'unique': data.nunique(),
        'mean': data.mean(),
        'median': data.median(),
        'mode': data.mode().iloc[0] if len(data.mode()) > 0 else np.nan,
        'std': data.std(),
        'variance': data.var(),
        'min': data.min(),
        'max': data.max(),
        'range': data.max() - data.min(),
        'iqr': data.quantile(0.75) - data.quantile(0.25),
        'skewness': data.skew(),
        'kurtosis': data.kurtosis(),
        'cv': (data.std() / data.mean() * 100) if data.mean() != 0 else np.nan,
    }
    
    # Percentiles
    results['percentiles'] = {
        '1%': data.quantile(0.01),
        '5%': data.quantile(0.05),
        '25%': data.quantile(0.25),
        '50%': data.quantile(0.50),
        '75%': data.quantile(0.75),
        '95%': data.quantile(0.95),
        '99%': data.quantile(0.99),
    }
    
    # Outliers (IQR method)
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    results['outliers_count'] = len(outliers)
    results['outliers_pct'] = (len(outliers) / len(data)) * 100
    results['outlier_bounds'] = (lower_bound, upper_bound)
    
    # Normality tests
    if len(data) >= 8:
        if len(data) <= 5000:
            stat_shapiro, p_shapiro = shapiro(data)
            results['shapiro_stat'] = stat_shapiro
            results['shapiro_pvalue'] = p_shapiro
        if len(data) >= 20:
            stat_dagostino, p_dagostino = normaltest(data)
            results['dagostino_stat'] = stat_dagostino
            results['dagostino_pvalue'] = p_dagostino
    
    # Print results
    _print_subheader("Descriptive Statistics")
    print(f"  Count:        {results['count']:,}")
    print(f"  Missing:      {results['missing']:,} ({results['missing_pct']:.1f}%)")
    print(f"  Unique:       {results['unique']:,}")
    print(f"  Mean:         {results['mean']:.4f}")
    print(f"  Median:       {results['median']:.4f}")
    print(f"  Std Dev:      {results['std']:.4f}")
    print(f"  CV:           {results['cv']:.2f}%" if not np.isnan(results['cv']) else "  CV:           N/A")
    
    _print_subheader("Distribution Shape")
    print(f"  Min:          {results['min']:.4f}")
    print(f"  Max:          {results['max']:.4f}")
    print(f"  Range:        {results['range']:.4f}")
    print(f"  IQR:          {results['iqr']:.4f}")
    print(f"  Skewness:     {results['skewness']:.4f}", end="")
    if abs(results['skewness']) < 0.5:
        print(" (approximately symmetric)")
    elif results['skewness'] > 0:
        print(" (right-skewed)")
    else:
        print(" (left-skewed)")
    print(f"  Kurtosis:     {results['kurtosis']:.4f}", end="")
    if results['kurtosis'] > 1:
        print(" (heavy tails / peaked)")
    elif results['kurtosis'] < -1:
        print(" (light tails / flat)")
    else:
        print(" (approximately normal tails)")
    
    _print_subheader("Outliers (IQR Method)")
    print(f"  Lower bound:  {lower_bound:.4f}")
    print(f"  Upper bound:  {upper_bound:.4f}")
    print(f"  Outliers:     {results['outliers_count']} ({results['outliers_pct']:.1f}%)")
    
    _print_subheader("Normality Tests")
    if 'shapiro_pvalue' in results:
        print(f"  Shapiro-Wilk: W={results['shapiro_stat']:.4f}, p={results['shapiro_pvalue']:.4f}", end="")
        print(" ✓ Normal" if results['shapiro_pvalue'] > 0.05 else " ✗ Non-normal")
    if 'dagostino_pvalue' in results:
        print(f"  D'Agostino:   K²={results['dagostino_stat']:.4f}, p={results['dagostino_pvalue']:.4f}", end="")
        print(" ✓ Normal" if results['dagostino_pvalue'] > 0.05 else " ✗ Non-normal")
    
    # Plotting
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Univariate Analysis: {var_name}', fontweight='bold')
        
        # Histogram with KDE
        axes[0, 0].hist(data, bins=bins, density=True, alpha=0.7, 
                        color=PALETTE_CATEGORICAL[0], edgecolor='white')
        data.plot.kde(ax=axes[0, 0], color=PALETTE_CATEGORICAL[1], linewidth=2)
        axes[0, 0].axvline(data.mean(), color=PALETTE_CATEGORICAL[2], linestyle='--', 
                          linewidth=2, label=f'Mean: {data.mean():.2f}')
        axes[0, 0].axvline(data.median(), color=PALETTE_CATEGORICAL[3], linestyle='--', 
                          linewidth=2, label=f'Median: {data.median():.2f}')
        axes[0, 0].set_title('Distribution with KDE')
        axes[0, 0].set_xlabel(var_name)
        axes[0, 0].legend()
        
        # Box plot
        bp = axes[0, 1].boxplot(data, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor(PALETTE_CATEGORICAL[0])
        bp['boxes'][0].set_alpha(0.7)
        bp['medians'][0].set_color(PALETTE_CATEGORICAL[3])
        axes[0, 1].set_title('Box Plot')
        axes[0, 1].set_ylabel(var_name)
        axes[0, 1].set_xticklabels([var_name])
        
        # Q-Q plot
        stats.probplot(data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal)')
        axes[1, 0].get_lines()[0].set_color(PALETTE_CATEGORICAL[0])
        axes[1, 0].get_lines()[1].set_color(PALETTE_CATEGORICAL[1])
        
        # Violin plot
        parts = axes[1, 1].violinplot([data], positions=[1], showmeans=True, 
                                      showextrema=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(PALETTE_CATEGORICAL[0])
            pc.set_alpha(0.7)
        parts['cmeans'].set_color(PALETTE_CATEGORICAL[2])
        parts['cmedians'].set_color(PALETTE_CATEGORICAL[3])
        axes[1, 1].set_title('Violin Plot')
        axes[1, 1].set_xticks([1])
        axes[1, 1].set_xticklabels([var_name])
        
        plt.tight_layout()
        plt.show()
    
    return results


def analyze_categorical(
    series: pd.Series,
    figsize: Tuple[int, int] = (12, 5),
    max_categories: int = 20,
    show_plot: bool = True
) -> Dict:
    """
    Comprehensive univariate analysis for a categorical variable.
    
    Parameters
    ----------
    series : pd.Series
        Categorical pandas Series to analyze.
    figsize : tuple, optional
        Figure size (width, height). Default is (12, 5).
    max_categories : int, optional
        Maximum categories to display in plots. Default is 20.
    show_plot : bool, optional
        Whether to display plots. Default is True.
    
    Returns
    -------
    dict
        Dictionary containing all computed statistics.
    
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series(['A', 'B', 'A', 'C', 'B', 'A'], name='category')
    >>> results = analyze_categorical(data)
    """
    var_name = _get_series_name(series, "Categorical Variable")
    data = series.dropna()
    
    if len(data) == 0:
        print(f"⚠️ Warning: '{var_name}' has no non-null values.")
        return {}
    
    _print_header(f"UNIVARIATE ANALYSIS: {var_name}")
    
    value_counts = data.value_counts()
    proportions = data.value_counts(normalize=True)
    
    # Calculate entropy and Gini impurity
    probs = proportions.values
    entropy = -np.sum(probs * np.log2(probs + 1e-15))
    gini = 1 - np.sum(probs ** 2)
    
    results = {
        'variable': var_name,
        'count': len(data),
        'missing': series.isna().sum(),
        'missing_pct': (series.isna().sum() / len(series)) * 100,
        'unique': data.nunique(),
        'mode': value_counts.index[0],
        'mode_count': value_counts.iloc[0],
        'mode_pct': proportions.iloc[0] * 100,
        'entropy': entropy,
        'gini_impurity': gini,
        'value_counts': value_counts.to_dict(),
        'proportions': proportions.to_dict(),
    }
    
    # Print results
    _print_subheader("Basic Statistics")
    print(f"  Count:        {results['count']:,}")
    print(f"  Missing:      {results['missing']:,} ({results['missing_pct']:.1f}%)")
    print(f"  Unique:       {results['unique']}")
    print(f"  Mode:         {results['mode']} ({results['mode_count']:,} occurrences, {results['mode_pct']:.1f}%)")
    
    _print_subheader("Distribution Metrics")
    print(f"  Entropy:      {entropy:.4f} (max possible: {np.log2(results['unique']):.4f})")
    print(f"  Gini Index:   {gini:.4f}")
    
    _print_subheader("Value Counts")
    display_counts = value_counts.head(max_categories)
    for idx, (cat, count) in enumerate(display_counts.items()):
        pct = (count / len(data)) * 100
        bar = "█" * int(pct / 2)
        print(f"  {cat:20s} {count:8,} ({pct:5.1f}%) {bar}")
    if len(value_counts) > max_categories:
        print(f"  ... and {len(value_counts) - max_categories} more categories")
    
    # Plotting
    if show_plot:
        n_categories = len(value_counts)
        use_pie = n_categories <= 6
        
        fig, axes = plt.subplots(1, 2 if use_pie else 1, figsize=figsize)
        fig.suptitle(f'Univariate Analysis: {var_name}', fontweight='bold')
        
        if use_pie:
            ax_bar, ax_pie = axes
        else:
            ax_bar = axes if not use_pie else axes[0]
        
        # Bar chart
        plot_data = value_counts.head(max_categories)
        colors = [PALETTE_CATEGORICAL[i % len(PALETTE_CATEGORICAL)] for i in range(len(plot_data))]
        plot_data.plot(kind='bar', ax=ax_bar, color=colors, alpha=0.8, edgecolor='white')
        ax_bar.set_title('Frequency Distribution')
        ax_bar.set_xlabel(var_name)
        ax_bar.set_ylabel('Count')
        ax_bar.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, (idx, val) in enumerate(plot_data.items()):
            ax_bar.annotate(f'{val:,}', xy=(i, val), ha='center', va='bottom', fontsize=9)
        
        # Pie chart (only for few categories)
        if use_pie:
            ax_pie.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%',
                      colors=colors, startangle=90)
            ax_pie.set_title('Proportion Distribution')
        
        plt.tight_layout()
        plt.show()
    
    return results


# =============================================================================
# BIVARIATE ANALYSIS FUNCTIONS
# =============================================================================

def analyze_numerical_numerical(
    series1: pd.Series,
    series2: pd.Series,
    parametric: bool = True,
    figsize: Tuple[int, int] = (12, 5),
    show_plot: bool = True,
    alpha: float = 0.05
) -> Dict:
    """
    Bivariate analysis for two numerical variables.
    
    Parameters
    ----------
    series1 : pd.Series
        First numerical pandas Series.
    series2 : pd.Series
        Second numerical pandas Series.
    parametric : bool, optional
        If True, use Pearson correlation; if False, use Spearman.
        Default is True.
    figsize : tuple, optional
        Figure size (width, height). Default is (12, 5).
    show_plot : bool, optional
        Whether to display plots. Default is True.
    alpha : float, optional
        Significance level. Default is 0.05.
    
    Returns
    -------
    dict
        Dictionary containing correlation coefficients and test results.
    
    Notes
    -----
    - Use parametric=True (Pearson) when both variables are approximately 
      normally distributed and the relationship is linear.
    - Use parametric=False (Spearman) when data is non-normal, ordinal, 
      or the relationship may be monotonic but not linear.
    
    Examples
    --------
    >>> x = pd.Series([1, 2, 3, 4, 5], name='x')
    >>> y = pd.Series([2, 4, 5, 4, 5], name='y')
    >>> results = analyze_numerical_numerical(x, y, parametric=True)
    """
    name1 = _get_series_name(series1, "Variable 1")
    name2 = _get_series_name(series2, "Variable 2")
    
    # Align series and drop missing values
    df_temp = pd.DataFrame({name1: series1, name2: series2}).dropna()
    x, y = df_temp[name1], df_temp[name2]
    
    if len(x) < 3:
        print("⚠️ Warning: Not enough paired observations for analysis.")
        return {}
    
    _print_header(f"BIVARIATE ANALYSIS: {name1} vs {name2}")
    
    # Correlation tests
    r_pearson, p_pearson = pearsonr(x, y)
    r_spearman, p_spearman = spearmanr(x, y)
    
    # Choose primary correlation based on parametric flag
    if parametric:
        r_primary, p_primary = r_pearson, p_pearson
        test_name = "Pearson"
    else:
        r_primary, p_primary = r_spearman, p_spearman
        test_name = "Spearman"
    
    results = {
        'variable1': name1,
        'variable2': name2,
        'n_pairs': len(x),
        'parametric': parametric,
        'primary_test': test_name,
        'correlation': r_primary,
        'p_value': p_primary,
        'significant': p_primary < alpha,
        'pearson_r': r_pearson,
        'pearson_p': p_pearson,
        'spearman_r': r_spearman,
        'spearman_p': p_spearman,
        'r_squared': r_primary ** 2,
    }
    
    # Interpret correlation strength
    abs_r = abs(r_primary)
    if abs_r < 0.1:
        strength = "negligible"
    elif abs_r < 0.3:
        strength = "weak"
    elif abs_r < 0.5:
        strength = "moderate"
    elif abs_r < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    results['correlation_strength'] = strength
    
    direction = "positive" if r_primary > 0 else "negative"
    results['correlation_direction'] = direction
    
    # Print results
    _print_subheader("Sample Information")
    print(f"  Paired observations: {len(x):,}")
    print(f"  Test type:           {'Parametric' if parametric else 'Non-parametric'}")
    
    _print_subheader(f"Correlation Analysis ({test_name})")
    print(f"  Correlation (r):     {r_primary:.4f}")
    print(f"  R-squared:           {results['r_squared']:.4f}")
    print(f"  p-value:             {p_primary:.4e}")
    print(f"  Significant (α={alpha}):", "Yes ✓" if results['significant'] else "No ✗")
    print(f"  Interpretation:      {strength} {direction} correlation")
    
    _print_subheader("Both Correlation Methods (for comparison)")
    print(f"  Pearson:   r={r_pearson:.4f}, p={p_pearson:.4e}")
    print(f"  Spearman:  r={r_spearman:.4f}, p={p_spearman:.4e}")
    
    # Plotting
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'Bivariate Analysis: {name1} vs {name2}', fontweight='bold')
        
        # Scatter plot with regression line
        axes[0].scatter(x, y, alpha=0.6, color=PALETTE_CATEGORICAL[0], edgecolor='white')
        
        # Add regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        axes[0].plot(x_line, p(x_line), color=PALETTE_CATEGORICAL[1], linewidth=2,
                    label=f'Trend line (r={r_primary:.3f})')
        
        axes[0].set_xlabel(name1)
        axes[0].set_ylabel(name2)
        axes[0].set_title('Scatter Plot with Trend Line')
        axes[0].legend()
        
        # Hexbin for density
        hb = axes[1].hexbin(x, y, gridsize=20, cmap='Blues', mincnt=1)
        axes[1].set_xlabel(name1)
        axes[1].set_ylabel(name2)
        axes[1].set_title('Density Plot')
        plt.colorbar(hb, ax=axes[1], label='Count')
        
        plt.tight_layout()
        plt.show()
    
    return results


def analyze_categorical_numerical(
    categorical_series: pd.Series,
    numerical_series: pd.Series,
    parametric: bool = True,
    figsize: Tuple[int, int] = (12, 5),
    show_plot: bool = True,
    alpha: float = 0.05
) -> Dict:
    """
    Bivariate analysis for a categorical and a numerical variable.
    
    Parameters
    ----------
    categorical_series : pd.Series
        Categorical pandas Series (grouping variable).
    numerical_series : pd.Series
        Numerical pandas Series (measurement variable).
    parametric : bool, optional
        If True, use t-test/ANOVA; if False, use Mann-Whitney/Kruskal-Wallis.
        Default is True.
    figsize : tuple, optional
        Figure size (width, height). Default is (12, 5).
    show_plot : bool, optional
        Whether to display plots. Default is True.
    alpha : float, optional
        Significance level. Default is 0.05.
    
    Returns
    -------
    dict
        Dictionary containing group statistics and test results.
    
    Notes
    -----
    - Use parametric=True when the numerical variable is approximately 
      normally distributed within each group and variances are roughly equal.
    - Use parametric=False when data is non-normal, has unequal variances,
      or when dealing with ordinal data.
    
    Examples
    --------
    >>> groups = pd.Series(['A', 'A', 'B', 'B', 'C', 'C'], name='group')
    >>> values = pd.Series([1.2, 1.5, 2.1, 2.3, 3.0, 3.2], name='value')
    >>> results = analyze_categorical_numerical(groups, values, parametric=True)
    """
    cat_name = _get_series_name(categorical_series, "Category")
    num_name = _get_series_name(numerical_series, "Value")
    
    # Align and drop missing
    df_temp = pd.DataFrame({cat_name: categorical_series, num_name: numerical_series}).dropna()
    categories = df_temp[cat_name]
    values = df_temp[num_name]
    
    unique_cats = categories.unique()
    n_groups = len(unique_cats)
    
    if n_groups < 2:
        print("⚠️ Warning: Need at least 2 groups for comparison.")
        return {}
    
    _print_header(f"BIVARIATE ANALYSIS: {num_name} by {cat_name}")
    
    # Group statistics
    group_stats = df_temp.groupby(cat_name)[num_name].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    groups_data = [values[categories == cat].values for cat in unique_cats]
    
    results = {
        'categorical_var': cat_name,
        'numerical_var': num_name,
        'n_groups': n_groups,
        'total_n': len(df_temp),
        'parametric': parametric,
        'group_stats': group_stats.to_dict('index'),
    }
    
    # Statistical tests
    if n_groups == 2:
        # Two-group comparison
        group1, group2 = groups_data[0], groups_data[1]
        
        if parametric:
            stat, p_value = ttest_ind(group1, group2)
            test_name = "Independent t-test"
        else:
            stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            test_name = "Mann-Whitney U"
        
        # Effect size: Cohen's d
        pooled_std = np.sqrt(((len(group1)-1)*np.std(group1, ddof=1)**2 + 
                              (len(group2)-1)*np.std(group2, ddof=1)**2) / 
                             (len(group1) + len(group2) - 2))
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
        results['effect_size'] = cohens_d
        results['effect_size_name'] = "Cohen's d"
        
    else:
        # Multi-group comparison
        if parametric:
            stat, p_value = f_oneway(*groups_data)
            test_name = "One-way ANOVA"
        else:
            stat, p_value = kruskal(*groups_data)
            test_name = "Kruskal-Wallis H"
        
        # Effect size: Eta-squared (for ANOVA)
        grand_mean = values.mean()
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups_data)
        ss_total = sum((values - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        results['effect_size'] = eta_squared
        results['effect_size_name'] = "Eta-squared"
    
    results.update({
        'test_name': test_name,
        'statistic': stat,
        'p_value': p_value,
        'significant': p_value < alpha,
    })
    
    # Levene's test for homogeneity of variances
    if len(groups_data) >= 2 and all(len(g) >= 2 for g in groups_data):
        levene_stat, levene_p = levene(*groups_data)
        results['levene_stat'] = levene_stat
        results['levene_p'] = levene_p
        results['equal_variances'] = levene_p > alpha
    
    # Print results
    _print_subheader("Group Statistics")
    print(f"  Number of groups: {n_groups}")
    print(f"  Total observations: {len(df_temp):,}")
    print()
    print(group_stats.to_string())
    
    _print_subheader(f"Statistical Test ({test_name})")
    print(f"  Test type:           {'Parametric' if parametric else 'Non-parametric'}")
    print(f"  Statistic:           {stat:.4f}")
    print(f"  p-value:             {p_value:.4e}")
    print(f"  Significant (α={alpha}):", "Yes ✓" if results['significant'] else "No ✗")
    
    if 'effect_size' in results:
        effect = results['effect_size']
        if results['effect_size_name'] == "Cohen's d":
            if abs(effect) < 0.2:
                effect_interp = "small"
            elif abs(effect) < 0.8:
                effect_interp = "medium"
            else:
                effect_interp = "large"
        else:  # Eta-squared
            if effect < 0.01:
                effect_interp = "small"
            elif effect < 0.06:
                effect_interp = "medium"
            else:
                effect_interp = "large"
        print(f"  {results['effect_size_name']}:       {effect:.4f} ({effect_interp})")
    
    if 'levene_p' in results:
        _print_subheader("Assumption Check: Homogeneity of Variances (Levene's Test)")
        print(f"  Statistic:           {results['levene_stat']:.4f}")
        print(f"  p-value:             {results['levene_p']:.4e}")
        print(f"  Equal variances:    ", "Yes ✓" if results['equal_variances'] else "No ✗ (consider non-parametric)")
    
    # Plotting
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'Bivariate Analysis: {num_name} by {cat_name}', fontweight='bold')
        
        # Box plot
        box_colors = [PALETTE_CATEGORICAL[i % len(PALETTE_CATEGORICAL)] for i in range(n_groups)]
        bp = axes[0].boxplot([values[categories == cat] for cat in unique_cats],
                            labels=unique_cats, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0].set_xlabel(cat_name)
        axes[0].set_ylabel(num_name)
        axes[0].set_title('Box Plot by Group')
        axes[0].tick_params(axis='x', rotation=45 if n_groups > 5 else 0)
        
        # Violin plot
        parts = axes[1].violinplot([values[categories == cat] for cat in unique_cats],
                                   positions=range(1, n_groups + 1),
                                   showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(box_colors[i])
            pc.set_alpha(0.7)
        axes[1].set_xticks(range(1, n_groups + 1))
        axes[1].set_xticklabels(unique_cats)
        axes[1].set_xlabel(cat_name)
        axes[1].set_ylabel(num_name)
        axes[1].set_title('Violin Plot by Group')
        axes[1].tick_params(axis='x', rotation=45 if n_groups > 5 else 0)
        
        plt.tight_layout()
        plt.show()
    
    return results


def analyze_categorical_categorical(
    series1: pd.Series,
    series2: pd.Series,
    figsize: Tuple[int, int] = (12, 5),
    show_plot: bool = True,
    alpha: float = 0.05
) -> Dict:
    """
    Bivariate analysis for two categorical variables.
    
    Parameters
    ----------
    series1 : pd.Series
        First categorical pandas Series.
    series2 : pd.Series
        Second categorical pandas Series.
    figsize : tuple, optional
        Figure size (width, height). Default is (12, 5).
    show_plot : bool, optional
        Whether to display plots. Default is True.
    alpha : float, optional
        Significance level. Default is 0.05.
    
    Returns
    -------
    dict
        Dictionary containing contingency table and chi-square test results.
    
    Notes
    -----
    Chi-square test assumptions:
    - Expected frequencies should be at least 5 in each cell.
    - If expected frequencies are too small, consider Fisher's exact test
      (not implemented here) or combining categories.
    
    Examples
    --------
    >>> cat1 = pd.Series(['A', 'A', 'B', 'B', 'A', 'B'], name='var1')
    >>> cat2 = pd.Series(['X', 'Y', 'X', 'Y', 'Y', 'X'], name='var2')
    >>> results = analyze_categorical_categorical(cat1, cat2)
    """
    name1 = _get_series_name(series1, "Variable 1")
    name2 = _get_series_name(series2, "Variable 2")
    
    # Align and drop missing
    df_temp = pd.DataFrame({name1: series1, name2: series2}).dropna()
    var1, var2 = df_temp[name1], df_temp[name2]
    
    _print_header(f"BIVARIATE ANALYSIS: {name1} vs {name2}")
    
    # Contingency table
    contingency = pd.crosstab(var1, var2)
    contingency_pct = pd.crosstab(var1, var2, normalize='all') * 100
    row_pct = pd.crosstab(var1, var2, normalize='index') * 100
    col_pct = pd.crosstab(var1, var2, normalize='columns') * 100
    
    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    # Cramér's V (effect size)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0
    
    results = {
        'variable1': name1,
        'variable2': name2,
        'n_obs': len(df_temp),
        'contingency_table': contingency.to_dict(),
        'chi2_statistic': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'expected_frequencies': pd.DataFrame(expected, index=contingency.index, 
                                             columns=contingency.columns).to_dict(),
        'cramers_v': cramers_v,
        'significant': p_value < alpha,
    }
    
    # Check assumption: expected frequencies >= 5
    min_expected = expected.min()
    pct_low_expected = (expected < 5).sum() / expected.size * 100
    results['min_expected'] = min_expected
    results['pct_cells_low_expected'] = pct_low_expected
    results['assumption_met'] = min_expected >= 5
    
    # Interpret Cramér's V
    if cramers_v < 0.1:
        effect_interp = "negligible"
    elif cramers_v < 0.3:
        effect_interp = "small"
    elif cramers_v < 0.5:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    results['effect_interpretation'] = effect_interp
    
    # Print results
    _print_subheader("Contingency Table (Counts)")
    print(contingency.to_string())
    
    _print_subheader("Contingency Table (Row %)")
    print(row_pct.round(1).to_string())
    
    _print_subheader("Chi-Square Test of Independence")
    print(f"  Chi-square statistic: {chi2:.4f}")
    print(f"  Degrees of freedom:   {dof}")
    print(f"  p-value:              {p_value:.4e}")
    print(f"  Significant (α={alpha}):", "Yes ✓" if results['significant'] else "No ✗")
    print(f"  Cramér's V:           {cramers_v:.4f} ({effect_interp} association)")
    
    _print_subheader("Assumption Check")
    print(f"  Min expected frequency: {min_expected:.2f}")
    print(f"  Cells with expected < 5: {pct_low_expected:.1f}%")
    if not results['assumption_met']:
        print("  ⚠️ Warning: Some expected frequencies are < 5. Results may be unreliable.")
        print("     Consider combining categories or using Fisher's exact test.")
    else:
        print("  ✓ Assumption met (all expected frequencies >= 5)")
    
    # Plotting
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'Bivariate Analysis: {name1} vs {name2}', fontweight='bold')
        
        # Heatmap of counts
        sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   cbar_kws={'label': 'Count'})
        axes[0].set_title('Contingency Table (Counts)')
        axes[0].set_xlabel(name2)
        axes[0].set_ylabel(name1)
        
        # Stacked bar chart (row percentages)
        row_pct.plot(kind='bar', stacked=True, ax=axes[1], 
                    color=PALETTE_CATEGORICAL[:len(row_pct.columns)], alpha=0.8)
        axes[1].set_title('Stacked Bar Chart (Row %)')
        axes[1].set_xlabel(name1)
        axes[1].set_ylabel('Percentage')
        axes[1].legend(title=name2, bbox_to_anchor=(1.02, 1), loc='upper left')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    return results


# =============================================================================
# CONVENIENCE / QUICK ANALYSIS
# =============================================================================

def quick_eda(series: pd.Series, **kwargs) -> Dict:
    """
    Automatically detect variable type and run appropriate univariate analysis.
    
    Parameters
    ----------
    series : pd.Series
        Pandas Series to analyze.
    **kwargs
        Additional arguments passed to the underlying function.
    
    Returns
    -------
    dict
        Results from either analyze_numerical or analyze_categorical.
    """
    if pd.api.types.is_numeric_dtype(series):
        # Check if it's really categorical (low cardinality)
        if series.nunique() <= 10:
            print("ℹ️ Numeric variable with ≤10 unique values detected.")
            print("   Treating as categorical. Use analyze_numerical() to force numeric analysis.")
            return analyze_categorical(series, **kwargs)
        return analyze_numerical(series, **kwargs)
    else:
        return analyze_categorical(series, **kwargs)


if __name__ == "__main__":
    # Demo with sample data
    print("EDA Toolkit Demo")
    print("================")
    
    # Create sample data
    np.random.seed(42)
    n = 200
    
    demo_numerical = pd.Series(np.random.normal(100, 15, n), name='test_scores')
    demo_categorical = pd.Series(np.random.choice(['A', 'B', 'C', 'D'], n, p=[0.4, 0.3, 0.2, 0.1]), name='grade')
    demo_numerical2 = pd.Series(demo_numerical + np.random.normal(0, 5, n) * 0.5, name='final_scores')
    demo_categorical2 = pd.Series(np.random.choice(['Male', 'Female'], n), name='gender')
    
    print("\n\n" + "="*60)
    print("DEMO: analyze_numerical()")
    print("="*60)
    analyze_numerical(demo_numerical)
