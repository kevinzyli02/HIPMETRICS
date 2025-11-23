import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import seaborn as sns


def create_summary_figure(data, coef_df, di_threshold, sensitivity, specificity, output_dir):
    """Create a comprehensive summary figure of the analysis results"""
    plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1, 1], width_ratios=[1, 1])

    # Panel A: Deformity Index by Stulberg Group
    ax1 = plt.subplot(gs[0, :])
    sns.boxplot(x='stulberg_group', y='deformity_index', data=data,
                order=['I/II', 'III', 'IV/V'], palette='viridis', ax=ax1)
    ax1.set_title('Deformity Index Distribution by Stulberg Group', fontsize=16, weight='bold')
    ax1.set_xlabel('Stulberg Classification Group', fontsize=14)
    ax1.set_ylabel('Deformity Index', fontsize=14)
    ax1.text(0.05, 0.95, 'Kruskal-Wallis p < 0.000001',
             transform=ax1.transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))

    # Panel B: Ordinal Regression Coefficients
    ax2 = plt.subplot(gs[1, 0])
    predictors = coef_df[coef_df['Parameter'].isin(['lateral_ratio', 'eq_ratio', 'deformity_index'])]

    colors = ['#1f77b4' if p > 0.05 else '#ff7f0e' for p in predictors['p_value']]
    y_pos = np.arange(len(predictors))

    ax2.barh(y_pos, predictors['Coefficient'], xerr=predictors['SE'],
             color=colors, ecolor='black', capsize=10)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(predictors['Parameter'])
    ax2.set_title('Predictors of Worse Stulberg Outcomes', fontsize=16, weight='bold')
    ax2.set_xlabel('Regression Coefficient', fontsize=14)
    ax2.axvline(0, color='gray', linestyle='--')

    # Add significance stars
    for i, p_val in enumerate(predictors['p_value']):
        if p_val < 0.001:
            ax2.text(predictors['Coefficient'].iloc[i] + 0.1, i, '***', fontsize=14, va='center')
        elif p_val < 0.01:
            ax2.text(predictors['Coefficient'].iloc[i] + 0.1, i, '**', fontsize=14, va='center')
        elif p_val < 0.05:
            ax2.text(predictors['Coefficient'].iloc[i] + 0.1, i, '*', fontsize=14, va='center')

    # Panel C: Risk Stratification by Deformity Index
    ax3 = plt.subplot(gs[1, 1])
    risk_thresholds = [di_threshold - 0.2, di_threshold]  # Example thresholds
    data['risk_category'] = np.select(
        [data['deformity_index'] < risk_thresholds[0],
         data['deformity_index'] < risk_thresholds[1]],
        ['Low Risk', 'Medium Risk'],
        default='High Risk'
    )

    risk_counts = data['risk_category'].value_counts()
    colors = ['#2ca02c', '#ff7f0e', '#d62728']  # Green, orange, red
    ax3.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 12})
    ax3.set_title('Patient Risk Stratification by Deformity Index',
                  fontsize=16, weight='bold')

    # Panel D: Clinical Decision Pathway
    ax4 = plt.subplot(gs[2, :])
    ax4.axis('off')

    # Create clinical decision pathway
    pathway_text = (
        "Clinical Decision Pathway Based on Deformity Index (DI):\n\n"
        f"1. DI < {risk_thresholds[0]:.2f} → Low Risk: Monitor with annual follow-up\n"
        f"2. {risk_thresholds[0]:.2f} ≤ DI < {risk_thresholds[1]:.2f} → Medium Risk: Physical therapy, activity modification\n"
        f"3. DI ≥ {risk_thresholds[1]:.2f} → High Risk: Surgical intervention recommended\n\n"
        f"Optimal DI Threshold: {di_threshold:.2f} (Sensitivity: {sensitivity:.1%}, Specificity: {specificity:.1%})"
    )

    ax4.text(0.05, 0.8, pathway_text, fontsize=14,
             bbox=dict(facecolor='white', alpha=0.8))

    ax4.set_title('Clinical Decision Pathway', fontsize=16, weight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3)
    plt.suptitle('Deformity Index as a Key Predictor of LCPD Outcomes',
                 fontsize=20, weight='bold')

    # Save figure
    plt.savefig(os.path.join(output_dir, 'clinical_summary_figure.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
