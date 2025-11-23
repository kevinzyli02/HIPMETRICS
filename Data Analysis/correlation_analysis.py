import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import scipy.stats as stats
from scipy.stats import kruskal, f_oneway, spearmanr, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from visualization import create_summary_figure

# Get the root directory of your project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Set the output file directly in the current "Data Analysis" folder
PATIENT_DB_PATH= os.path.join(PROJECT_ROOT, "patient_database.xlsx")
# Configuration
MEASUREMENTS_PATH = r'C:\Users\SR207348\PycharmProjects\HIPMETRICS\Data Analysis\measurements_summary.xlsx'
OUTPUT_DIR = PROJECT_ROOT



# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # Load datasets
    patient_db = pd.read_excel(PATIENT_DB_PATH)
    measurements = pd.read_excel(MEASUREMENTS_PATH)

    # Clean and merge data
    merged = preprocess_and_merge(patient_db, measurements)

    if merged.empty:
        print("No valid data to analyze after merging and cleaning.")
        return

    # 1. Descriptive Group Comparisons
    perform_group_comparisons(merged)

    # 2. Ordinal Correlation
    perform_ordinal_correlation(merged)

    # 3. Predictive Modeling
    perform_ordinal_regression(merged)

    # 4. Threshold Analysis
    results = perform_threshold_analysis(merged)

    print(f"Analysis complete. Results saved to: {OUTPUT_DIR}")

    # After analyses, create summary figure
    di_roc = results['deformity_index']  # From threshold_analysis results

    # Get optimal threshold for DI
    di_threshold = di_roc['optimal_threshold']
    sensitivity = di_roc['sensitivity']
    specificity = di_roc['specificity']

    # Load ordinal regression coefficients
    coef_path = os.path.join(OUTPUT_DIR, 'ordinal_regression_coefficients.xlsx')
    coef_df = pd.read_excel(coef_path)

    # Create the figure
    create_summary_figure(merged, coef_df, di_threshold, sensitivity, specificity, OUTPUT_DIR)

    print(f"Saved clinical summary figure to {OUTPUT_DIR}")

def preprocess_and_merge(patients, measurements):
    """Clean and merge patient data with measurements"""
    patients_clean = patients.copy()

    # Convert to string first to handle both numeric and Roman numeral formats
    stulberg_str = patients['stulberg_classification'].astype(str)

    # Map Stulberg classes to ordinal scale (1=1, 2=2, 3=3, 4=4, 5=5, I=1, II=2, III=3, IV=4, V=5)
    stulberg_map = {
        '1': 1, 'I': 1,
        '2': 2, 'II': 2,
        '3': 3, 'III': 3,
        '4': 4, 'IV': 4,
        '5': 5, 'V': 5
    }

    # Extract first character set (works for both numbers and Roman numerals)
    extracted = stulberg_str.str.extract(r'(\d+|I{1,3}|IV|V)', expand=False)

    # Map to ordinal scale
    patients_clean['stulberg_ordinal'] = extracted.map(stulberg_map)

    # Create Stulberg groups for ANOVA - FIXED TYPE ISSUE
    conditions = [
        patients_clean['stulberg_ordinal'].isin([1, 2]),
        patients_clean['stulberg_ordinal'] == 3,
        patients_clean['stulberg_ordinal'].isin([4, 5])
    ]

    choices = ['I/II', 'III', 'IV/V']

    # Initialize with None (object type)
    patients_clean['stulberg_group'] = None

    # Assign values based on conditions
    for cond, choice in zip(conditions, choices):
        patients_clean.loc[cond, 'stulberg_group'] = choice

    # Create binary outcome for ROC analysis
    patients_clean['favorable_outcome'] = patients_clean['stulberg_ordinal'].isin([1, 2]).astype(int)

    # Merge datasets
    merged = pd.merge(
        patients_clean,
        measurements,
        on='patient_id',
        how='inner'
    )

    # Drop rows with missing group or ordinal
    return merged.dropna(subset=['stulberg_ordinal', 'stulberg_group'])


def perform_group_comparisons(data):
    """ANOVA/Kruskal-Wallis with post-hoc tests for group comparisons"""
    results = {}

    # Print group counts for debugging
    print("\nStulberg Group Counts:")
    print(data['stulberg_group'].value_counts())

    for measurement in ['lateral_ratio', 'eq_ratio', 'deformity_index']:
        print(f"\n===== Group Comparisons: {measurement} =====")

        # Extract measurement values by group
        groups = [data[data['stulberg_group'] == group][measurement].dropna()
                  for group in ['I/II', 'III', 'IV/V']]

        # Filter out empty groups
        valid_groups = [g for g in groups if len(g) >= 3]
        group_names = ['I/II', 'III', 'IV/V'][:len(valid_groups)]

        if len(valid_groups) < 2:
            print(f"Not enough groups with sufficient data (need at least 2 groups with ≥3 samples)")
            results[measurement] = {
                'test': 'Skipped',
                'reason': f"Only {len(valid_groups)} valid groups",
                'p_value': np.nan
            }
            continue

        # Normality test (Shapiro-Wilk) - only if sample size ≤5000
        if len(data[measurement]) <= 5000:
            norm_test = stats.shapiro(data[measurement])
            print(f"Normality test (p={norm_test.pvalue:.4f}): {'Normal' if norm_test.pvalue > 0.05 else 'Non-normal'}")
            normal_dist = norm_test.pvalue > 0.05
        else:
            print("Sample size too large for Shapiro-Wilk, assuming non-normal")
            normal_dist = False

        # Equal variance test (Levene) only if ≥3 groups with ≥3 samples
        if len(valid_groups) >= 3:
            try:
                var_test = stats.levene(*valid_groups)
                print(
                    f"Equal variance test (p={var_test.pvalue:.4f}): {'Equal' if var_test.pvalue > 0.05 else 'Unequal'}")
                equal_var = var_test.pvalue > 0.05
            except Exception as e:
                print(f"Levene test failed: {e}")
                equal_var = False
        else:
            print("Skipping Levene test (need ≥3 groups)")
            equal_var = False  # Conservative assumption

        # Choose appropriate test
        if normal_dist and equal_var:
            # Parametric ANOVA
            anova_result = f_oneway(*valid_groups)
            print(f"ANOVA: F={anova_result.statistic:.3f}, p={anova_result.pvalue:.4f}")
            test_type = "ANOVA"

            # Tukey HSD post-hoc only if ≥3 groups
            if anova_result.pvalue < 0.05 and len(valid_groups) >= 3:
                try:
                    tukey = pairwise_tukeyhsd(
                        endog=data[measurement],
                        groups=data['stulberg_group'],
                        alpha=0.05
                    )
                    print(tukey)
                    results[measurement] = {
                        'test': 'ANOVA',
                        'p_value': anova_result.pvalue,
                        'post_hoc': str(tukey)
                    }
                except Exception as e:
                    print(f"Tukey HSD failed: {e}")
                    results[measurement] = {
                        'test': 'ANOVA',
                        'p_value': anova_result.pvalue,
                        'post_hoc': f"Error: {e}"
                    }
            else:
                results[measurement] = {
                    'test': 'ANOVA',
                    'p_value': anova_result.pvalue,
                    'post_hoc': 'Not performed (p≥0.05 or <3 groups)'
                }
        else:
            # Non-parametric Kruskal-Wallis
            try:
                kruskal_result = kruskal(*valid_groups)
                print(f"Kruskal-Wallis: H={kruskal_result.statistic:.3f}, p={kruskal_result.pvalue:.4f}")
                test_type = "Kruskal-Wallis"
                results[measurement] = {
                    'test': 'Kruskal-Wallis',
                    'p_value': kruskal_result.pvalue,
                    'post_hoc': 'N/A'  # Dunn's test could be added later
                }
            except Exception as e:
                print(f"Kruskal-Wallis failed: {e}")
                test_type = "Failed"
                results[measurement] = {
                    'test': 'Error',
                    'p_value': np.nan,
                    'post_hoc': f"Error: {e}"
                }

        # Visualization only if we have valid groups
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='stulberg_group', y=measurement, data=data,
                    order=['I/II', 'III', 'IV/V'])
        plt.title(f'{measurement} by Stulberg Group\n{test_type}')
        plt.savefig(os.path.join(OUTPUT_DIR, f'group_comparison_{measurement}.png'), dpi=300)
        plt.close()

    # Save results
    pd.DataFrame(results).T.to_excel(os.path.join(OUTPUT_DIR, 'group_comparisons.xlsx'))

def perform_ordinal_correlation(data):
    """Spearman's correlation between measurements and Stulberg ordinal"""
    results = {}

    for measurement in ['lateral_ratio', 'eq_ratio', 'deformity_index']:
        corr, pval = spearmanr(
            data[measurement],
            data['stulberg_ordinal'],
            nan_policy='omit'
        )
        print(f"\n{measurement} vs Stulberg:")
        print(f"Spearman's ρ = {corr:.3f}, p = {pval:.4f}")

        results[measurement] = {
            'rho': corr,
            'p_value': pval
        }

        # Visualization
        plt.figure(figsize=(10, 6))
        sns.regplot(x=measurement, y='stulberg_ordinal', data=data,
                    scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
        plt.title(f'{measurement} vs Stulberg Classification\nρ = {corr:.3f}, p = {pval:.4f}')
        plt.ylabel('Stulberg Class (I=1, V=5)')
        plt.savefig(os.path.join(OUTPUT_DIR, f'correlation_{measurement}.png'), dpi=300)
        plt.close()

    # Save results
    pd.DataFrame(results).T.to_excel(os.path.join(OUTPUT_DIR, 'ordinal_correlations.xlsx'))


def perform_ordinal_regression(data):
    """Ordinal logistic regression modeling Stulberg class"""
    # Prepare data - drop missing values
    model_data = data[['stulberg_ordinal', 'lateral_ratio', 'eq_ratio', 'deformity_index']].dropna()

    # Convert Stulberg to ordered categorical
    model_data['stulberg_cat'] = pd.Categorical(
        model_data['stulberg_ordinal'],
        categories=[1, 2, 3, 4, 5],
        ordered=True
    )

    # Define predictors and outcome
    X = model_data[['lateral_ratio', 'eq_ratio', 'deformity_index']]
    y = model_data['stulberg_cat']

    # Fit ordinal logistic regression
    model = OrderedModel(y, X)
    result = model.fit(method='bfgs', disp=False)

    # Print and save results
    print("\n===== Ordinal Logistic Regression =====")
    print(result.summary())

    # Save model summary
    with open(os.path.join(OUTPUT_DIR, 'ordinal_regression.txt'), 'w') as f:
        f.write(result.summary().as_text())

    # Get all parameter names (thresholds + predictors)
    param_names = result.model.exog_names

    # Save coefficients
    coef_df = pd.DataFrame({
        'Parameter': param_names,
        'Coefficient': result.params,
        'SE': result.bse,
        'p_value': result.pvalues
    })

    coef_df.to_excel(os.path.join(OUTPUT_DIR, 'ordinal_regression_coefficients.xlsx'), index=False)

    return result
def perform_threshold_analysis(data):
    """ROC analysis for favorable vs unfavorable outcomes"""
    results = {}

    for measurement in ['lateral_ratio', 'eq_ratio', 'deformity_index']:
        # Remove missing values
        clean_data = data[['favorable_outcome', measurement]].dropna()
        y_true = clean_data['favorable_outcome']
        y_score = clean_data[measurement]

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]

        print(f"\n{measurement} ROC Analysis:")
        print(f"AUC = {roc_auc:.3f}")
        print(f"Optimal Threshold = {optimal_threshold:.3f}")
        print(f"Sensitivity = {tpr[optimal_idx]:.3f}")
        print(f"Specificity = {1 - fpr[optimal_idx]:.3f}")

        results[measurement] = {
            'auc': roc_auc,
            'optimal_threshold': optimal_threshold,
            'sensitivity': tpr[optimal_idx],
            'specificity': 1 - fpr[optimal_idx]
        }

        # Visualization
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red',
                    label=f'Optimal threshold: {optimal_threshold:.2f}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'ROC Curve: {measurement} vs Favorable Outcome')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(OUTPUT_DIR, f'roc_{measurement}.png'), dpi=300)
        plt.close()

    # Save results
    pd.DataFrame(results).T.to_excel(os.path.join(OUTPUT_DIR, 'threshold_analysis.xlsx'))


if __name__ == '__main__':
    main()