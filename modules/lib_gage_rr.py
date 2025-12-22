#!/usr/bin/env python3
"""
Gage R&R Analysis Utilities
Shared functions for statistical analysis and visualization

Author: Khang Tran
Date: December 2025
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib.ticker import FormatStrFormatter

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class GageRRAnalyzer:
    """Gage R&R statistical analysis functionality"""

    def __init__(self, spec_limits=None):
        """
        Initialize the analyzer

        Args:
            spec_limits (dict): Specification limits for each metric
        """
        self.spec_limits = spec_limits or {
            "RMSE": (0, 0.08),
            "SAM": (0, 3.2),
            "Uniformity": (0.4, 1.0)
        }

    def calculate_anova(self, df, ctq):
        """
        Calculate ANOVA for Gage R&R analysis

        Args:
            df (pd.DataFrame): Input data with columns: Operator, Sample, metric values
            ctq (str): Critical-to-Quality metric name

        Returns:
            dict: ANOVA results and variance components
        """
        try:
            # Prepare data for ANOVA
            df_clean = df.dropna(subset=[ctq])

            if len(df_clean) < 3:
                print(f"WARNING: Insufficient data for {ctq} ANOVA")
                return None

            # Fit ANOVA model
            formula = f"{ctq} ~ C(Operator) + C(Sample) + C(Operator):C(Sample)"
            model = ols(formula, data=df_clean).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            # Calculate variance components
            variance_components = self._calculate_variance_components(anova_table, df_clean, ctq)

            return {
                'anova_table': anova_table,
                'variance_components': variance_components,
                'model': model
            }

        except Exception as e:
            print(f"ERROR: ANOVA calculation failed for {ctq}: {str(e)}")
            return None

    def _calculate_variance_components(self, anova_table, df, ctq):
        """Calculate Gage R&R variance components"""
        try:
            # Extract mean squares from ANOVA
            ms_operator = anova_table.loc['C(Operator)', 'sum_sq'] / anova_table.loc['C(Operator)', 'df']
            ms_sample = anova_table.loc['C(Sample)', 'sum_sq'] / anova_table.loc['C(Sample)', 'df']
            ms_interaction = anova_table.loc['C(Operator):C(Sample)', 'sum_sq'] / anova_table.loc['C(Operator):C(Sample)', 'df']
            ms_error = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']

            # Calculate variance components
            n_operators = df['Operator'].nunique()
            n_samples = df['Sample'].nunique()
            n_reps = len(df) / (n_operators * n_samples)

            var_operator = max(0, (ms_operator - ms_interaction) / (n_samples * n_reps))
            var_sample = max(0, (ms_sample - ms_interaction) / (n_operators * n_reps))
            var_interaction = max(0, (ms_interaction - ms_error) / n_reps)
            var_error = ms_error

            var_reproducibility = var_operator
            var_repeatability = var_error + var_interaction
            var_gage_rr = var_reproducibility + var_repeatability
            var_total = var_gage_rr + var_sample

            # Calculate percentages
            pct_reproducibility = (var_reproducibility / var_total) * 100 if var_total > 0 else 0
            pct_repeatability = (var_repeatability / var_total) * 100 if var_total > 0 else 0
            pct_gage_rr = (var_gage_rr / var_total) * 100 if var_total > 0 else 0
            pct_part_to_part = (var_sample / var_total) * 100 if var_total > 0 else 0

            # Calculate standard deviations
            std_reproducibility = np.sqrt(var_reproducibility)
            std_repeatability = np.sqrt(var_repeatability)
            std_gage_rr = np.sqrt(var_gage_rr)
            std_part_to_part = np.sqrt(var_sample)
            std_total = np.sqrt(var_total)

            # Calculate study variation (6 sigma)
            sv_reproducibility = 6 * std_reproducibility
            sv_repeatability = 6 * std_repeatability
            sv_gage_rr = 6 * std_gage_rr
            sv_part_to_part = 6 * std_part_to_part
            sv_total = 6 * std_total

            # Calculate tolerance percentages
            tolerance = self.spec_limits[ctq][1] - self.spec_limits[ctq][0] if ctq in self.spec_limits else 1.0
            pct_tolerance_gage_rr = (sv_gage_rr / tolerance) * 100 if tolerance > 0 else 0

            # Calculate Number of Distinct Categories (NDC)
            ndc = max(1, int(1.41 * (std_part_to_part / std_gage_rr))) if std_gage_rr > 0 else 1

            return {
                'variance': {
                    'reproducibility': var_reproducibility,
                    'repeatability': var_repeatability,
                    'gage_rr': var_gage_rr,
                    'part_to_part': var_sample,
                    'total': var_total
                },
                'std_dev': {
                    'reproducibility': std_reproducibility,
                    'repeatability': std_repeatability,
                    'gage_rr': std_gage_rr,
                    'part_to_part': std_part_to_part,
                    'total': std_total
                },
                'study_variation': {
                    'reproducibility': sv_reproducibility,
                    'repeatability': sv_repeatability,
                    'gage_rr': sv_gage_rr,
                    'part_to_part': sv_part_to_part,
                    'total': sv_total
                },
                'percentages': {
                    'reproducibility': pct_reproducibility,
                    'repeatability': pct_repeatability,
                    'gage_rr': pct_gage_rr,
                    'part_to_part': pct_part_to_part
                },
                'tolerance': {
                    'gage_rr_pct': pct_tolerance_gage_rr
                },
                'ndc': ndc,
                'metrics': {
                    'operators': n_operators,
                    'samples': n_samples,
                    'replications': n_reps
                }
            }

        except Exception as e:
            print(f"ERROR: Variance component calculation failed: {str(e)}")
            return None


class GageRRVisualizer:
    """Gage R&R visualization functionality"""

    def __init__(self, spec_limits=None):
        """Initialize the visualizer"""
        self.spec_limits = spec_limits or {
            "RMSE": (0, 0.08),
            "SAM": (0, 3.2),
            "Uniformity": (0.4, 1.0)
        }

    def plot_summary_dashboard(self, variance_components, ctq, output_folder):
        """
        Create summary dashboard for Gage R&R results

        Args:
            variance_components (dict): Results from variance component analysis
            ctq (str): Critical-to-Quality metric name
            output_folder (str): Output directory for plots
        """
        try:
            fig = plt.figure(figsize=(14, 8))
            gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.2, height_ratios=[0.8, 0.2, 0.75, 0.75])

            # Extract key metrics
            metrics = variance_components['metrics']
            percentages = variance_components['percentages']
            tolerance = variance_components['tolerance']
            ndc = variance_components['ndc']

            # Summary table (top-left)
            ax_summary = fig.add_subplot(gs[0, 0])
            ax_summary.axis("off")

            summary_data = [
                ["Metric", "Value"],
                ["Parts", f"{metrics['samples']:.0f}"],
                ["Operators", f"{metrics['operators']:.0f}"],
                ["Replications", f"{metrics['replications']:.1f}"],
                ["GRR % of Tolerance", f"{tolerance['gage_rr_pct']:.1f}%"],
                ["Number of Distinct Categories", f"{ndc}"]
            ]

            table = ax_summary.table(cellText=summary_data[1:], colLabels=summary_data[0],
                                   cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.8)
            ax_summary.set_title(f"{ctq} - Gage R&R Summary", fontsize=12, fontweight='bold')

            # Variance components pie chart (top-right)
            ax_pie = fig.add_subplot(gs[0, 1])

            pie_labels = ['Gage R&R', 'Part-to-Part']
            pie_values = [percentages['gage_rr'], percentages['part_to_part']]
            colors = ['lightcoral', 'lightblue']

            ax_pie.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', colors=colors)
            ax_pie.set_title('Variance Components', fontweight='bold')

            # Variance breakdown table (bottom-left)
            ax_breakdown = fig.add_subplot(gs[2:, 0])
            ax_breakdown.axis("off")

            breakdown_data = [
                ["Source", "% Contribution"],
                ["Repeatability", f"{percentages['repeatability']:.1f}%"],
                ["Reproducibility", f"{percentages['reproducibility']:.1f}%"],
                ["Gage R&R", f"{percentages['gage_rr']:.1f}%"],
                ["Part-to-Part", f"{percentages['part_to_part']:.1f}%"]
            ]

            table2 = ax_breakdown.table(cellText=breakdown_data[1:], colLabels=breakdown_data[0],
                                      cellLoc='center', loc='center')
            table2.auto_set_font_size(False)
            table2.set_fontsize(10)
            table2.scale(1.2, 2.0)
            ax_breakdown.set_title("Variance Breakdown", fontsize=11, fontweight='bold')

            # Variance components bar chart (bottom-right)
            ax_bar = fig.add_subplot(gs[2:, 1])

            components = ['Repeatability', 'Reproducibility', 'Part-to-Part']
            values = [percentages['repeatability'], percentages['reproducibility'], percentages['part_to_part']]

            bars = ax_bar.bar(components, values, color=['orange', 'red', 'blue'])
            ax_bar.set_ylabel('% of Total Variation')
            ax_bar.set_title('Variance Components')
            ax_bar.set_ylim(0, 100)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value:.1f}%', ha='center', va='bottom')

            plt.tight_layout()

            # Save plot
            output_path = os.path.join(output_folder, f"{ctq}_gage_rr_summary_dashboard.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Summary dashboard saved: {os.path.basename(output_path)}")

        except Exception as e:
            print(f"ERROR: Failed to create summary dashboard for {ctq}: {str(e)}")


def extract_numeric_value(value_str):
    """
    Extract numeric value from string (handles units like degrees)

    Args:
        value_str (str): String containing numeric value with possible units

    Returns:
        float: Extracted numeric value
    """
    if pd.isna(value_str):
        return np.nan

    if isinstance(value_str, (int, float)):
        return float(value_str)

    try:
        # Remove common units and extract number
        import re
        # Remove degree symbol and other units
        cleaned = re.sub(r'[°%]', '', str(value_str))
        # Extract first number found
        numbers = re.findall(r'-?\d+\.?\d*', cleaned)
        if numbers:
            return float(numbers[0])
        else:
            return np.nan
    except:
        return np.nan


def load_operator_data(operator, base_path="results"):
    """
    Load analysis data for a specific operator

    Args:
        operator (str): Operator identifier
        base_path (str): Base path for results

    Returns:
        pd.DataFrame: Loaded data or None if failed
    """
    # Try multiple possible file locations
    possible_paths = [
        f"{base_path}/{operator}/scatter_plots/batch_analysis_summary.csv",
        f"{base_path}/{operator}/batch_analysis_summary.csv"
    ]

    for file_path in possible_paths:
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Add operator information
                df['Operator'] = operator
                return df
        except Exception as e:
            print(f"ERROR: Failed to load {file_path}: {str(e)}")
            continue

    print(f"WARNING: No data file found for {operator} in {base_path}")
    return None


def categorize_sample_performance(df, thresholds):
    """
    Categorize samples based on CTQ thresholds

    Args:
        df (pd.DataFrame): Input data
        thresholds (dict): Threshold values for each metric

    Returns:
        pd.DataFrame: Data with performance categories
    """
    try:
        df_copy = df.copy()

        # Initialize pass/fail columns
        for metric in thresholds.keys():
            if metric in df_copy.columns:
                if metric == "Uniformity_Score":
                    # Higher is better
                    df_copy[f"{metric}_Pass"] = df_copy[metric] >= thresholds[metric]
                else:
                    # Lower is better
                    df_copy[f"{metric}_Pass"] = df_copy[metric] <= thresholds[metric]

        # Overall pass (all metrics must pass)
        pass_columns = [col for col in df_copy.columns if col.endswith('_Pass')]
        if pass_columns:
            df_copy['Overall_Pass'] = df_copy[pass_columns].all(axis=1)

        return df_copy

    except Exception as e:
        print(f"ERROR: Sample categorization failed: {str(e)}")
        return df


def generate_operator_info(operators):
    """
    Generate operator information from operator names

    Args:
        operators (list): List of operator names (e.g., ['KhangT1', 'KhangT2'])

    Returns:
        dict: Operator information with person and trial number
    """
    operator_info = {}
    for op in operators:
        # Extract person name and trial number
        if 'T' in op:
            parts = op.split('T')
            person = parts[0] if len(parts) > 1 else op
            try:
                trial = int(parts[1]) if len(parts) > 1 else 1
            except ValueError:
                trial = 1
        else:
            person = op
            trial = 1

        operator_info[op] = {"person": person, "trial": trial}

    return operator_info


def load_thresholds_from_config(config):
    """
    Load quality thresholds dynamically from config dictionary

    Args:
        config (dict): Configuration dictionary from config.json

    Returns:
        tuple: (main_thresholds, all_thresholds) dictionaries
    """
    thresholds = config.get('quality_thresholds', {})

    # Main CTQ thresholds for combined analysis
    main_thresholds = {
        "RMSE Per Pixel P95": thresholds.get('rmse_per_pixel_p95_max', 0.08),
        "SAM Mean": thresholds.get('sam_mean_max', 2.6),
        "Uniformity Score": thresholds.get('uniformity_score_min', 0.6)
    }

    # Comprehensive CTQ thresholds for all metrics
    all_thresholds = {
        "RMSE_Overall": thresholds.get('rmse_overall_max', 0.08),
        "RMSE_Per_Pixel_Mean": thresholds.get('rmse_per_pixel_mean_max', 0.06),
        "RMSE_Per_Pixel_Median": thresholds.get('rmse_per_pixel_median_max', 0.09),
        "RMSE_Per_Pixel_P95": thresholds.get('rmse_per_pixel_p95_max', 0.08),
        "SAM_Mean": thresholds.get('sam_mean_max', 2.6),
        "SAM_Median": thresholds.get('sam_median_max', 3.0),
        "SAM_P95": thresholds.get('sam_p95_max', 5.2),
        "Uniformity_Score": thresholds.get('uniformity_score_min', 0.6)
    }

    return main_thresholds, all_thresholds


def calculate_gage_rr_statistics(combined_df):
    """
    Calculate measurement system statistics for defect detection validation
    Uses single-measurement design (no replications)

    Args:
        combined_df (pd.DataFrame): Combined data from all operators
        main_thresholds (dict): Main CTQ thresholds

    Returns:
        dict: Statistics for each CTQ metric
    """
    # Map old column names to new ones
    ctq_mapping = {
        "RMSE Per Pixel P95": "RMSE_Per_Pixel_P95",
        "SAM Mean": "SAM_Mean",
        "Uniformity Score": "Uniformity_Score"
    }

    ctq_metrics = ["RMSE Per Pixel P95", "SAM Mean", "Uniformity Score"]
    results = {}

    # Expected sample categories
    expected_good_samples = [4, 20]

    for ctq in ctq_metrics:
        actual_column = ctq_mapping.get(ctq, ctq)

        if actual_column not in combined_df.columns:
            print(f"WARNING: {actual_column} not found in data")
            continue

        data = combined_df.dropna(subset=[actual_column])
        if len(data) == 0:
            continue

        # Basic statistics
        total_var = data[actual_column].var()
        mean_value = data[actual_column].mean()
        std_value = data[actual_column].std()

        # Repeatability: consistency within each operator
        operator_within_std = []
        for operator in data['Operator'].unique():
            op_data = data[data['Operator'] == operator]
            sample_stds = []
            for sample in op_data['Sample'].unique():
                sample_data = op_data[op_data['Sample'] == sample][actual_column]
                if len(sample_data) > 1:
                    sample_stds.append(sample_data.std())

            if sample_stds:
                operator_within_std.append(np.mean(sample_stds))

        avg_repeatability_std = np.mean(operator_within_std) if operator_within_std else 0

        # Reproducibility: consistency between operators
        sample_between_op_std = []
        for sample in data['Sample'].unique():
            sample_data = data[data['Sample'] == sample]
            if len(sample_data['Operator'].unique()) > 1:
                op_means = sample_data.groupby('Operator')[actual_column].mean()
                sample_between_op_std.append(op_means.std())

        avg_reproducibility_std = np.mean(sample_between_op_std) if sample_between_op_std else 0

        # Discrimination capability
        good_sample_data = data[data['Sample'].str.replace('Sample', '').str.replace('sample', '').astype(int).isin(expected_good_samples)]
        defective_sample_data = data[~data['Sample'].str.replace('Sample', '').str.replace('sample', '').astype(int).isin(expected_good_samples)]

        if len(good_sample_data) > 0 and len(defective_sample_data) > 0:
            good_mean = good_sample_data[actual_column].mean()
            defective_mean = defective_sample_data[actual_column].mean()

            signal = abs(good_mean - defective_mean)
            noise = max(float(avg_repeatability_std), float(avg_reproducibility_std), 0.001)

            discrimination_ratio = signal / noise

            pooled_std = np.sqrt(((len(good_sample_data) - 1) * good_sample_data[actual_column].var() +
                                 (len(defective_sample_data) - 1) * defective_sample_data[actual_column].var()) /
                                (len(good_sample_data) + len(defective_sample_data) - 2))

            separation_std = signal / pooled_std if pooled_std > 0 else 0
        else:
            discrimination_ratio = 0
            separation_std = 0
            good_mean = defective_mean = mean_value

        # Measurement system capability metrics
        total_measurement_variation = avg_repeatability_std + avg_reproducibility_std

        if std_value > 0:
            repeatability_pct = min((avg_repeatability_std / std_value) * 100, 100)
            reproducibility_pct = min((avg_reproducibility_std / std_value) * 100, 100)
            total_ms_pct = min((total_measurement_variation / std_value) * 100, 100)
        else:
            repeatability_pct = reproducibility_pct = total_ms_pct = 0

        # Assessment
        if total_ms_pct < 10:
            ms_assessment = "EXCELLENT"
        elif total_ms_pct < 30:
            ms_assessment = "ACCEPTABLE"
        else:
            ms_assessment = "NEEDS IMPROVEMENT"

        if discrimination_ratio >= 10:
            discrimination_assessment = "EXCELLENT"
        elif discrimination_ratio >= 5:
            discrimination_assessment = "GOOD"
        else:
            discrimination_assessment = "POOR"

        results[ctq] = {
            'mean_value': mean_value,
            'std_value': std_value,
            'cv_percent': (std_value / abs(mean_value)) * 100 if mean_value != 0 else 0,
            'repeatability_std': avg_repeatability_std,
            'reproducibility_std': avg_reproducibility_std,
            'total_measurement_std': total_measurement_variation,
            'repeatability_percent': repeatability_pct,
            'reproducibility_percent': reproducibility_pct,
            'measurement_system_percent': total_ms_pct,
            'good_sample_mean': good_mean,
            'defective_sample_mean': defective_mean,
            'signal_to_noise_ratio': discrimination_ratio,
            'separation_std_devs': separation_std,
            'measurement_system_assessment': ms_assessment,
            'discrimination_assessment': discrimination_assessment,
            'good_sample_count': len(good_sample_data),
            'defective_sample_count': len(defective_sample_data),
            'total_samples': len(data)
        }

    return results


def plot_gage_rr_summary(combined_df, summary_metrics, ctq, output_folder, main_thresholds, anova_results=None):
    """
    Create multi-panel professional Gage R&R summary dashboard

    Args:
        combined_df: Combined operator data
        summary_metrics: Dictionary with measurement system metrics
        ctq: CTQ metric name
        output_folder: Output directory for plots
        main_thresholds: Main threshold values
        anova_results: Optional ANOVA results
    """
    from matplotlib.ticker import FormatStrFormatter

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.2, height_ratios=[0.8, 0.2, 0.75, 0.75])

    # Map CTQ names to column names
    ctq_mapping = {
        "RMSE Per Pixel P95": "RMSE_Per_Pixel_P95",
        "SAM Mean": "SAM_Mean",
        "Uniformity Score": "Uniformity_Score"
    }

    actual_column = ctq_mapping.get(ctq, ctq)

    if actual_column not in combined_df.columns:
        print(f"Warning: {actual_column} not found in data")
        return

    # Get metrics
    n_parts = combined_df['Sample'].nunique()
    n_ops = combined_df['Operator'].nunique()
    n_rep = combined_df.groupby(['Sample', 'Operator']).size().mean()

    # Get threshold from main_thresholds
    threshold = main_thresholds.get(ctq, 0)

    # Set spec limits based on CTQ type
    if ctq == "Uniformity Score":
        lsl = main_thresholds.get("Uniformity Score", 0.6)
        usl = 1.0
    elif ctq == "RMSE Per Pixel P95":
        lsl = 0
        usl = main_thresholds.get("RMSE Per Pixel P95", 0.08)
    else:  # SAM Mean
        lsl = 0
        usl = main_thresholds.get("SAM Mean", 2.6)

    grr_pct = summary_metrics.get('measurement_system_percent', 0)
    ndc = summary_metrics.get('ndc', 0)

    # Summary table (top-left)
    ax_summary = fig.add_subplot(gs[0, 0])
    ax_summary.axis("off")
    summary_info = [
        ["Input Summary", ""],
        ["Data Type", "Continuous"],
        ["Analysis Type", "Crossed"],
        ["Samples", str(n_parts)],
        ["Operators", str(n_ops)],
        ["Avg Trials", f"{n_rep:.1f}"],
        ["Spec. Limits", f"[{lsl}, {usl}]"]
    ]
    table_summary = ax_summary.table(
        cellText=summary_info, cellLoc="left", loc="center",
        colWidths=[0.55, 0.45], bbox=[0, 0, 1, 1]
    )
    table_summary.auto_set_font_size(False)
    table_summary.set_fontsize(9)
    table_summary.scale(1, 1.85)
    for j in range(2):
        cell = table_summary[(0, j)]
        cell.set_facecolor("#C0C0C0")
        cell.set_text_props(weight="bold", fontsize=9)

    # Checks table (middle-left)
    ax_checks = fig.add_subplot(gs[1, 0])
    ax_checks.axis("off")
    checks_info = [
        ["Assumption Checks", ""],
        ["Number of samples sufficient", "for this analysis"]
    ]
    table_checks = ax_checks.table(
        cellText=checks_info, cellLoc="left", loc="center",
        colWidths=[0.55, 0.45], bbox=[0, 0, 1, 1]
    )
    table_checks.auto_set_font_size(False)
    table_checks.set_fontsize(9)
    table_checks.scale(1, 2.0)
    for j in range(2):
        cell = table_checks[(0, j)]
        cell.set_facecolor("#C0C0C0")
        cell.set_text_props(weight="bold", fontsize=9)

    # ANOVA table (lower-left) - Optional
    ax_anova = fig.add_subplot(gs[2, 0])
    ax_anova.axis("off")
    if anova_results is not None:
        anova_data = [
            ["ANOVA Table", "", "", ""],
            ["Source", "Variation", "Percent", "Assessment"]
        ]

        for key in ['repeatability_percent', 'reproducibility_percent', 'part_to_part_percent']:
            if key in summary_metrics:
                source = key.replace('_percent', '').replace('_', ' ').title()
                pct = summary_metrics[key]
                assessment = "Good" if pct < 10 else "Acceptable" if pct < 30 else "Poor"
                anova_data.append([source, f"{pct:.1f}%", "", assessment])

        table_anova = ax_anova.table(
            cellText=anova_data, cellLoc="center", loc="center", bbox=[0, 0, 1, 1]
        )
        table_anova.auto_set_font_size(False)
        table_anova.set_fontsize(8.5)
        table_anova.scale(1, 1.8)
        for j in range(4):
            cell = table_anova[(0, j)]
            cell.set_facecolor("#C0C0C0")
            cell.set_text_props(weight="bold", fontsize=8.5)
            cell = table_anova[(1, j)]
            cell.set_facecolor("#C0C0C0")
            cell.set_text_props(weight="bold", fontsize=8.5)

    # Results table (bottom-left)
    ax_results = fig.add_subplot(gs[3, 0])
    ax_results.axis("off")

    rep_pct = summary_metrics.get('repeatability_percent', 0)
    repro_pct = summary_metrics.get('reproducibility_percent', 0)
    part_pct = summary_metrics.get('part_to_part_percent', 100 - grr_pct)

    results_info = [
        ["Analysis Results", "Percent", "Assessment"],
        ["Total Gage R&R", f"{grr_pct:.2f}%", "Good" if grr_pct < 10 else "Acceptable" if grr_pct < 30 else "Poor"],
        ["  Repeatability", f"{rep_pct:.2f}%", ""],
        ["  Reproducibility", f"{repro_pct:.2f}%", ""],
        ["Part-to-Part", f"{part_pct:.2f}%", ""],
        ["Total Variation", "100.00%", ""],
    ]

    table_results = ax_results.table(
        cellText=results_info, cellLoc="center", loc="center",
        colWidths=[0.45, 0.30, 0.25], bbox=[0, 0, 1, 1]
    )
    table_results.auto_set_font_size(False)
    table_results.set_fontsize(8.5)
    table_results.scale(1, 1.95)

    for j in range(3):
        cell = table_results[(0, j)]
        cell.set_facecolor("#C0C0C0")
        cell.set_text_props(weight="bold", fontsize=8.5)

    last_idx = len(results_info) - 1
    for j in range(3):
        cell = table_results[(last_idx, j)]
        cell.set_facecolor("#D3D3D3")
        cell.set_text_props(weight="bold", fontsize=8.5)

    conclusion_text = (
        "MSA is excellent" if grr_pct < 10 else
        "MSA is acceptable" if grr_pct < 30 else
        "MSA needs improvement"
    )
    ax_results.text(
        0.02, -0.18, f"Number of distinct categories: {int(ndc) if ndc and not np.isnan(ndc) else 'N/A'}",
        fontsize=9, fontweight="bold", transform=ax_results.transAxes
    )
    ax_results.text(
        0.02, -0.35, f"Conclusion: {conclusion_text}",
        fontsize=9, fontweight="bold", transform=ax_results.transAxes
    )

    # Bar chart (top-right)
    ax_bars = fig.add_subplot(gs[0:1, 1])
    categories = ["GR&R Study", "Part-to-Part"]
    values = [grr_pct, part_pct]

    y_max = max(values) * 1.2 if max(values) > 0 else 100
    ax_bars.axhspan(0, 10, facecolor="#D4EDDA", alpha=0.6, zorder=0)
    ax_bars.axhspan(10, 30, facecolor="#FFF3CD", alpha=0.6, zorder=0)
    ax_bars.axhspan(30, y_max, facecolor="#F8D7DA", alpha=0.6, zorder=0)

    bars = ax_bars.bar(
        categories, values, color=["#4472C4", "#FF69B4"],
        edgecolor="black", linewidth=1.5, width=0.4, zorder=3
    )

    ax_bars.set_ylabel("% of Total Variation", fontsize=11, fontweight="bold")
    ax_bars.set_ylim(0, y_max)
    ax_bars.set_title(
        f"Gage R&R Summary\nGR&R: {grr_pct:.2f}%, Part: {part_pct:.2f}%",
        fontsize=11, fontweight="bold", pad=12
    )
    ax_bars.grid(True, alpha=0.25, axis="y", linewidth=0.7, zorder=2)
    ax_bars.tick_params(labelsize=9)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax_bars.text(
            bar.get_x() + bar.get_width() / 2.0, height + (max(values) * 0.03),
            f"{val:.1f}%", ha="center", va="bottom",
            fontsize=11, fontweight="bold", zorder=4
        )

    # Interaction plot (bottom-right)
    ax_inter = fig.add_subplot(gs[2:, 1])
    plot_data = combined_df[["Sample", "Operator", actual_column]].dropna(subset=[actual_column]).copy()
    samples = sorted(plot_data["Sample"].unique())
    operators = sorted(plot_data["Operator"].unique())

    color_palette = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F"]
    colors_op = {op: color_palette[i % len(color_palette)] for i, op in enumerate(operators)}

    for op in operators:
        op_data = plot_data[plot_data["Operator"] == op]
        means = [op_data[op_data["Sample"] == s][actual_column].mean() for s in samples]
        ax_inter.plot(
            range(1, len(samples) + 1), means, marker="o", label=str(op),
            color=colors_op[op], linewidth=2.2, markersize=7
        )

    ax_inter.set_xlabel("Samples", fontsize=11, fontweight="bold")
    ax_inter.set_ylabel("Average", fontsize=11, fontweight="bold")
    ax_inter.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    min_val = plot_data[actual_column].min()
    max_val = plot_data[actual_column].max()
    ax_inter.set_title(
        f"Interaction Plot (Sample x Operator)\nN: {len(samples)}, YMin: {min_val:.2f}, YMax: {max_val:.2f}",
        fontsize=11, fontweight="bold", pad=12
    )
    ax_inter.legend(loc="best", fontsize=9, framealpha=0.95, edgecolor="black", fancybox=True)
    ax_inter.grid(True, alpha=0.3, linewidth=0.7)
    ax_inter.tick_params(labelsize=9)
    ax_inter.set_xticks(range(1, len(samples) + 1))

    numeric_samples = [s.replace("Sample", "").replace("sample", "") for s in samples]
    ax_inter.set_xticklabels(numeric_samples, rotation=45, fontsize=8)

    plt.suptitle("")
    out_path = os.path.join(output_folder, "gage_rr_dashboards", f"{ctq.replace(' ', '_')}_gage_rr_summary_dashboard.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"  SUCCESS: Summary dashboard saved: {os.path.basename(out_path)}")


def plot_gage_rr_diagnostics(combined_df, ctq, output_folder, main_thresholds):
    """
    Create three-panel diagnostic plots

    Args:
        combined_df: Combined operator data
        ctq: CTQ metric name
        output_folder: Output directory
        main_thresholds: Main threshold values
    """
    from matplotlib.ticker import FormatStrFormatter

    ctq_mapping = {
        "RMSE Per Pixel P95": "RMSE_Per_Pixel_P95",
        "SAM Mean": "SAM_Mean",
        "Uniformity Score": "Uniformity_Score"
    }

    actual_column = ctq_mapping.get(ctq, ctq)

    if actual_column not in combined_df.columns:
        print(f"Warning: {actual_column} not found in data")
        return

    plot_data = combined_df[["Sample", "Operator", actual_column]].dropna(subset=[actual_column]).copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: boxplots with jittered points
    ax1 = axes[0]
    operators = sorted(plot_data["Operator"].unique())
    bp_data = [plot_data[plot_data["Operator"] == op][actual_column].values for op in operators]
    bp = ax1.boxplot(bp_data, labels=operators, patch_artist=True, widths=0.6)

    for patch in bp["boxes"]:
        patch.set_facecolor("#B0E0E6")
        patch.set_edgecolor("black")
        patch.set_linewidth(1)

    for i, op in enumerate(operators):
        vals = plot_data[plot_data["Operator"] == op][actual_column].values
        x_pos = np.random.normal(i + 1, 0.03, size=len(vals))
        ax1.scatter(x_pos, vals, color="red", marker="s", s=40, alpha=0.6, zorder=3)

    n_total = len(plot_data)
    n_groups = len(operators)
    n_items = plot_data["Sample"].nunique()

    ax1.set_xlabel("Operators", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Value", fontsize=9, fontweight="bold")
    ax1.set_title(
        f"Measurements by Operator\nN: {n_total}, Groups: {n_groups}, Min: {plot_data[actual_column].min():.2f}, Max: {plot_data[actual_column].max():.2f}",
        fontsize=9, fontweight="bold"
    )
    ax1.grid(True, alpha=0.2, linewidth=0.5)
    ax1.tick_params(labelsize=8, axis='x', rotation=45)

    # Panel 2: sample means
    ax2 = axes[1]
    samples = sorted(plot_data["Sample"].unique())
    sample_means = [plot_data[plot_data["Sample"] == s][actual_column].mean() for s in samples]

    ax2.plot(
        range(1, len(samples) + 1), sample_means, marker="o",
        color="black", linewidth=1.8, markersize=5,
        markerfacecolor="white", markeredgewidth=1.5
    )
    ax2.set_xlabel("Samples", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Values", fontsize=9, fontweight="bold")
    ax2.set_xticks(range(1, len(samples) + 1))

    numeric_samples = [s.replace("Sample", "").replace("sample", "") for s in samples]
    ax2.set_xticklabels(numeric_samples, rotation=45, fontsize=8)

    ax2.set_title(
        f"Measurements by Sample\nN: {len(samples)}, Groups: {n_items}, Min: {plot_data[actual_column].min():.2f}, Max: {plot_data[actual_column].max():.2f}",
        fontsize=9, fontweight="bold"
    )
    ax2.grid(True, alpha=0.2, linewidth=0.5)
    ax2.tick_params(labelsize=8)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Panel 3: run chart
    ax3 = axes[2]
    overall_avg = plot_data[actual_column].mean()

    color_palette = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F"]
    colors_op = {op: color_palette[i % len(color_palette)] for i, op in enumerate(operators)}

    for op in operators:
        op_data = plot_data[plot_data["Operator"] == op]
        op_means = [op_data[op_data["Sample"] == s][actual_column].mean() for s in samples]
        ax3.plot(
            range(1, len(samples) + 1), op_means, marker="o",
            label=str(op), color=colors_op[op], linewidth=1.8, markersize=5
        )

    ax3.axhline(y=overall_avg, color="black", linestyle="--", linewidth=1.5, label="Average", zorder=1)

    sd = plot_data[actual_column].std()
    ax3.axhline(y=overall_avg + 2 * sd, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax3.axhline(y=overall_avg - 2 * sd, color="gray", linestyle=":", linewidth=1, alpha=0.6)

    ax3.set_xlabel("Samples", fontsize=9, fontweight="bold")
    ax3.set_ylabel("Measured Values", fontsize=9, fontweight="bold")
    ax3.set_xticks(range(1, len(samples) + 1))
    ax3.set_xticklabels(numeric_samples, rotation=45, fontsize=8)

    ax3.set_title(
        f"Gage Run Chart by Sample, Operator\nSamples: {len(samples)}, Operators: {len(operators)}, Avg: {overall_avg:.2f}",
        fontsize=9, fontweight="bold"
    )
    ax3.legend(loc="best", fontsize=8, framealpha=0.9)
    ax3.grid(True, alpha=0.2, linewidth=0.5)
    ax3.tick_params(labelsize=8)

    plt.suptitle("")
    plt.tight_layout()

    plot_path = os.path.join(output_folder, "diagnostics", f"{ctq.replace(' ', '_')}_gage_rr_analysis.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"  ✅ Diagnostic plot saved: {os.path.basename(plot_path)}")


def plot_distinct_categories_analysis(combined_df, ctq, metrics, output_folder, main_thresholds):
    """
    Create distinct categories analysis plot

    Args:
        combined_df: Combined operator data
        ctq: CTQ metric name
        metrics: Metrics dictionary
        output_folder: Output directory
        main_thresholds: Main threshold values
    """
    ctq_mapping = {
        "RMSE Per Pixel P95": "RMSE_Per_Pixel_P95",
        "SAM Mean": "SAM_Mean",
        "Uniformity Score": "Uniformity_Score"
    }

    actual_column = ctq_mapping.get(ctq, ctq)

    if actual_column not in combined_df.columns:
        print(f"Warning: {actual_column} not found in data")
        return

    df_ctq = combined_df[["Sample", actual_column]].dropna().copy()

    # Calculate mean per sample
    sample_stats = df_ctq.groupby("Sample")[actual_column].mean().reset_index()
    sample_stats.rename(columns={actual_column: "Mean_Value"}, inplace=True)
    sample_stats = sample_stats.sort_values("Mean_Value").reset_index(drop=True)

    # Get GRR std
    avg_repeatability_std = metrics.get('avg_repeatability_std', 0)
    avg_reproducibility_std = metrics.get('avg_reproducibility_std', 0)
    sd_grr = np.sqrt(avg_repeatability_std**2 + avg_reproducibility_std**2)

    # Assign distinct categories
    category = 1
    sample_stats["Distinct_Category"] = 1
    reference_mean = sample_stats.loc[0, "Mean_Value"]

    for i in range(1, len(sample_stats)):
        current_mean = sample_stats.loc[i, "Mean_Value"]
        if abs(current_mean - reference_mean) >= sd_grr:
            category += 1
            reference_mean = current_mean
        sample_stats.loc[i, "Distinct_Category"] = category

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    n_categories = sample_stats['Distinct_Category'].nunique()
    palette = sns.color_palette("viridis", n_categories)
    category_colors = {
        cat: palette[i] for i, cat in enumerate(sorted(sample_stats['Distinct_Category'].unique()))
    }

    # Plot each sample
    for i, (_, row) in enumerate(sample_stats.iterrows()):
        category = row['Distinct_Category']
        color = category_colors[category]

        ax.scatter(
            i, category, color=color, s=200,
            edgecolor='black', linewidth=2, zorder=3,
            label=f'Category {category}' if i == sample_stats[sample_stats['Distinct_Category'] == category].index[0] else ""
        )

        sample_numeric = row['Sample'].replace('Sample', '').replace('sample', '')
        ax.text(
            i, category + 0.15, sample_numeric,
            ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=45
        )

        ax.text(
            i, category - 0.15, f"{row['Mean_Value']:.3f}",
            ha='center', va='top', fontsize=8, alpha=0.7
        )

    ax.set_xlabel('Samples (Ordered by Mean Value)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distinct Category', fontsize=12, fontweight='bold')

    y_ticks = sorted(sample_stats['Distinct_Category'].unique())
    ax.set_yticks(y_ticks)
    ax.set_ylim(0.5, max(y_ticks) + 0.5)

    ax.set_xticks(range(len(sample_stats)))
    ax.set_xticklabels([], rotation=45)

    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    ndc = metrics.get('ndc', 0)
    part_to_part_std = metrics.get('part_to_part_std', 0)

    title = f"{ctq} - Distinct Categories Analysis\n"
    title += f"NDC: {ndc:.1f} | GRR σ: {sd_grr:.4f} | Part σ: {part_to_part_std:.4f}"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Categories")

    interpretation = (
        "Excellent discrimination" if ndc >= 5 else
        "Adequate discrimination" if ndc >= 3 else
        "Poor discrimination"
    )
    color = "green" if ndc >= 5 else "orange" if ndc >= 3 else "red"

    ax.text(
        0.02, 0.98, f'Interpretation: {interpretation}',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor=color, alpha=0.2)
    )

    for i in range(1, max(y_ticks)):
        ax.axhline(y=i + 0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()

    out_path = os.path.join(output_folder, "distinct_categories", f"{ctq.replace(' ', '_')}_distinct_categories_analysis.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"  ✅ Distinct categories plot saved: {os.path.basename(out_path)}")

