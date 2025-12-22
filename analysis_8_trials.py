#!/usr/bin/env python3
"""
Comprehensive Gage R&R Analysis for Hyperspectral Defect Detection
Analyzes Critical-to-Quality (CTQ) metrics across multiple operators and trials

This tool provides:
1. Statistical Gage R&R analysis with ANOVA
2. Defect detection performance validation
3. Comprehensive visualization and reporting
4. Cross-platform compatibility

Author: Khang Tran
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
from modules.lib_gage_rr import (
    GageRRAnalyzer, GageRRVisualizer, extract_numeric_value, load_operator_data,
    categorize_sample_performance, generate_operator_info, load_thresholds_from_config,
    calculate_gage_rr_statistics, plot_gage_rr_summary, plot_gage_rr_diagnostics,
    plot_distinct_categories_analysis
)

# Configure matplotlib for cross-platform compatibility
plt.style.use('default')
sns.set_palette("husl")

# Configure stdout encoding for Windows compatibility
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Output directory structure for organized results
OUTPUT_DIRS = {
    'root': 'analysis_results',
    'plots': 'analysis_results/plots',
    'gage_rr': 'analysis_results/plots/gage_rr_dashboards',
    'diagnostics': 'analysis_results/plots/diagnostics',
    'categories': 'analysis_results/plots/distinct_categories',
    'sample_analysis': 'analysis_results/plots/sample_analysis',
    'data': 'analysis_results/data',
    'reports': 'analysis_results/reports'
}

def create_output_directories():
    """Create organized output directory structure"""
    for dir_name, dir_path in OUTPUT_DIRS.items():
        os.makedirs(dir_path, exist_ok=True)

# Load configuration
def load_analysis_config(config_path="config.json"):
    """Load analysis configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"WARNING: Config file {config_path} not found, using defaults")
        return {}
    except Exception as e:
        print(f"ERROR: Failed to load config: {str(e)}")
        return {}

# Load configuration or use defaults
CONFIG = load_analysis_config()

# Define operators - can be overridden by config file
DEFAULT_OPERATORS = [
    "KhangT1", "KhangT2",      # Khang Trial 1, Trial 2
    "KiduT1", "KiduT2",        # Kidu Trial 1, Trial 2
    "LuelT1", "LuelT2",        # Luel Trial 1, Trial 2
    "AnirbanT1", "AnirbanT2"   # Anirban Trial 1, Trial 2
]

OPERATORS = CONFIG.get('analysis_operators', DEFAULT_OPERATORS)

# Generate operator info using function from lib_gage_rr
OPERATOR_INFO = generate_operator_info(OPERATORS)

# Load quality thresholds from config.json using function from lib_gage_rr
MAIN_THRESHOLDS, ALL_THRESHOLDS = load_thresholds_from_config(CONFIG)

# Define expected results based on test design
EXPECTED_GOOD_SAMPLES = [4, 20]  # Only these should pass
EXPECTED_DEFECTIVE_SAMPLES = [1, 2, 6, 10, 12, 13, 14, 15, 17, 22]  # These should fail

def extract_numeric_sam(sam_str):
    """Extract numeric value from SAM string - wrapper for module function"""
    return extract_numeric_value(sam_str)

# load_operator_data is imported from modules.lib_gage_rr
# No need to redefine it here

# calculate_gage_rr_statistics is imported from modules.lib_gage_rr
# Using the module version which has the correct signature

def calculate_gage_rr_statistics_legacy(combined_df):
    """
    Legacy local implementation - kept for reference but not used
    Use calculate_gage_rr_statistics from lib_gage_rr instead
    """
    # Map old column names to new ones from batch_analysis_summary.csv
    ctq_mapping = {
        "RMSE Per Pixel P95": "RMSE_Per_Pixel_P95",
        "SAM Mean": "SAM_Mean",
        "Uniformity Score": "Uniformity_Score"
    }

    ctq_metrics = ["RMSE Per Pixel P95", "SAM Mean", "Uniformity Score"]
    results = {}

    for ctq in ctq_metrics:
        # Use mapped column name
        actual_column = ctq_mapping.get(ctq, ctq)

        if actual_column not in combined_df.columns:
            print(f"WARNING:  Warning: {actual_column} not found in data")
            continue

        # Remove any non-numeric values
        data = combined_df.dropna(subset=[actual_column])

        if len(data) == 0:
            continue

        # Get basic statistics
        total_var = data[actual_column].var()
        mean_value = data[actual_column].mean()
        std_value = data[actual_column].std()

        # For defect detection validation, we focus on:
        # 1. Measurement consistency within operators (repeatability)
        # 2. Consistency between operators (reproducibility)
        # 3. Ability to distinguish between good and defective samples (discrimination)

        # Calculate measurement system consistency metrics

        # Repeatability: consistency within each operator
        operator_within_std = []
        for operator in data['Operator'].unique():
            op_data = data[data['Operator'] == operator]
            # Group by sample within operator
            sample_stds = []
            for sample in op_data['Sample'].unique():
                sample_data = op_data[op_data['Sample'] == sample][actual_column]
                if len(sample_data) > 1:
                    sample_stds.append(sample_data.std())

            if sample_stds:
                operator_within_std.append(np.mean(sample_stds))

        avg_repeatability_std = np.mean(operator_within_std) if operator_within_std else 0

        # Reproducibility: consistency between operators for same samples
        sample_between_op_std = []
        for sample in data['Sample'].unique():
            sample_data = data[data['Sample'] == sample]
            if len(sample_data['Operator'].unique()) > 1:
                op_means = sample_data.groupby('Operator')[actual_column].mean()
                sample_between_op_std.append(op_means.std())

        avg_reproducibility_std = np.mean(sample_between_op_std) if sample_between_op_std else 0

        # Calculate discrimination capability
        # Can the system distinguish between good samples (4, 20) and defective samples?
        good_sample_data = data[data['Sample'].isin(['Sample04', 'sample04', 'Sample20', 'sample20'])]
        defective_sample_data = data[~data['Sample'].isin(['Sample04', 'sample04', 'Sample20', 'sample20'])]

        if len(good_sample_data) > 0 and len(defective_sample_data) > 0:
            good_mean = good_sample_data[actual_column].mean()
            defective_mean = defective_sample_data[actual_column].mean()

            # Signal-to-noise ratio: difference between groups vs measurement noise
            signal = abs(good_mean - defective_mean)
            noise = max(float(avg_repeatability_std), float(avg_reproducibility_std), 0.001)  # Avoid division by zero

            discrimination_ratio = signal / noise

            # Calculate separation between good and defective samples in standard deviations
            pooled_std = np.sqrt(((len(good_sample_data) - 1) * good_sample_data[actual_column].var() +
                                 (len(defective_sample_data) - 1) * defective_sample_data[actual_column].var()) /
                                (len(good_sample_data) + len(defective_sample_data) - 2))

            if pooled_std > 0:
                separation_std = signal / pooled_std
            else:
                separation_std = 0
        else:
            discrimination_ratio = 0
            separation_std = 0
            good_mean = defective_mean = mean_value

        # Calculate measurement system capability metrics
        # Convert to percentages for easier interpretation
        total_measurement_variation = avg_repeatability_std + avg_reproducibility_std

        if std_value > 0:
            # Percentage of total variation due to measurement system
            repeatability_pct = (avg_repeatability_std / std_value) * 100
            reproducibility_pct = (avg_reproducibility_std / std_value) * 100
            total_ms_pct = (total_measurement_variation / std_value) * 100

            # Ensure percentages don't exceed 100%
            repeatability_pct = min(repeatability_pct, 100)
            reproducibility_pct = min(reproducibility_pct, 100)
            total_ms_pct = min(total_ms_pct, 100)
        else:
            repeatability_pct = reproducibility_pct = total_ms_pct = 0

        # Assessment criteria for defect detection systems
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

        # Store results
        results[ctq] = {
            # Basic statistics
            'mean_value': mean_value,
            'std_value': std_value,
            'cv_percent': (std_value / abs(mean_value)) * 100 if mean_value != 0 else 0,

            # Measurement system consistency
            'repeatability_std': avg_repeatability_std,
            'reproducibility_std': avg_reproducibility_std,
            'total_measurement_std': total_measurement_variation,

            # Measurement system percentages (relative to total variation)
            'repeatability_percent': repeatability_pct,
            'reproducibility_percent': reproducibility_pct,
            'measurement_system_percent': total_ms_pct,

            # Discrimination capability
            'good_sample_mean': good_mean,
            'defective_sample_mean': defective_mean,
            'signal_to_noise_ratio': discrimination_ratio,
            'separation_std_devs': separation_std,

            # Assessments
            'measurement_system_assessment': ms_assessment,
            'discrimination_assessment': discrimination_assessment,

            # Additional metrics for reporting
            'good_sample_count': len(good_sample_data),
            'defective_sample_count': len(defective_sample_data),
            'total_samples': len(data)
        }

    return results

def create_ctq_summary_plots(combined_df, gage_rr_stats):
    """Create comprehensive CTQ analysis plots"""
    # Map old column names to new ones from batch_analysis_summary.csv
    ctq_mapping = {
        "RMSE Per Pixel P95": "RMSE_Per_Pixel_P95",
        "SAM Mean": "SAM_Mean",
        "Uniformity Score": "Uniformity_Score"
    }

    # Main 3 CTQs for box plots
    main_ctq_metrics = ["RMSE Per Pixel P95", "SAM Mean", "Uniformity Score"]

    # All CTQs available in batch_analysis_summary.csv for sample performance plots
    all_ctq_metrics = [
        ("RMSE_Overall", "RMSE Overall", "Max", ALL_THRESHOLDS["RMSE_Overall"]),
        ("RMSE_Per_Pixel_Mean", "RMSE Per Pixel Mean", "Max", ALL_THRESHOLDS["RMSE_Per_Pixel_Mean"]),
        ("RMSE_Per_Pixel_Median", "RMSE Per Pixel Median", "Max", ALL_THRESHOLDS["RMSE_Per_Pixel_Median"]),
        ("RMSE_Per_Pixel_P95", "RMSE Per Pixel P95", "Max", ALL_THRESHOLDS["RMSE_Per_Pixel_P95"]),
        ("SAM_Mean", "SAM Mean", "Max", ALL_THRESHOLDS["SAM_Mean"]),
        ("SAM_Median", "SAM Median", "Max", ALL_THRESHOLDS["SAM_Median"]),
        ("SAM_P95", "SAM P95", "Max", ALL_THRESHOLDS["SAM_P95"]),
        ("Uniformity_Score", "Uniformity Score", "Min", ALL_THRESHOLDS["Uniformity_Score"])
    ]

    # Create output directory
    os.makedirs("analysis_results", exist_ok=True)

    # 1. Box plots by operator (main 3 CTQs only)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Gage R&R CTQ Analysis - Box Plots by Operator', fontsize=16, fontweight='bold')

    for i, ctq in enumerate(main_ctq_metrics):
        actual_column = ctq_mapping.get(ctq, ctq)
        if actual_column not in combined_df.columns:
            continue

        if i < 2:
            ax = axes[0, i]
        else:
            ax = axes[1, 0]

        # Create box plot
        box_data = [combined_df[combined_df['Operator'] == op][actual_column].dropna()
                   for op in OPERATORS if op in combined_df['Operator'].unique()]
        box_labels = [op for op in OPERATORS if op in combined_df['Operator'].unique()]

        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)

        # Color boxes
        colors = plt.get_cmap('Set3')(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add threshold line
        threshold = MAIN_THRESHOLDS[ctq]
        if ctq == "Uniformity Score":
            ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7,
                      label=f'Min Threshold: {threshold}')
        else:
            ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7,
                      label=f'Max Threshold: {threshold}')

        ax.set_title(f'{ctq}')
        ax.set_xlabel('Operator')
        ax.set_ylabel(ctq)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Remove unused subplot
    axes[1, 1].remove()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIRS['sample_analysis'], 'ctq_boxplots_by_operator.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Sample-level analysis across operators - Create plots for ALL CTQs
    for column_name, display_name, threshold_type, threshold in all_ctq_metrics:
        if column_name not in combined_df.columns:
            print(f"WARNING:  Skipping {display_name} - column {column_name} not found in data")
            continue

        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.suptitle(f'{display_name} - Sample Performance Across All Operators', fontsize=16, fontweight='bold')

        # Get all unique sample numbers and sort them
        all_sample_nums = []
        for operator in OPERATORS:
            if operator in combined_df['Operator'].unique():
                op_data = combined_df[combined_df['Operator'] == operator]
                sample_nums = op_data['Sample'].str.replace('Sample', '').str.replace('sample', '').astype(int)
                all_sample_nums.extend(sample_nums.tolist())

        unique_sample_nums = sorted(list(set(all_sample_nums)))

        # Create scatter plot showing samples across operators
        for j, operator in enumerate(OPERATORS):
            if operator in combined_df['Operator'].unique():
                op_data = combined_df[combined_df['Operator'] == operator]
                sample_nums = op_data['Sample'].str.replace('Sample', '').str.replace('sample', '').astype(int)
                values = op_data[column_name].dropna()

                # Map sample numbers to evenly spaced x positions
                x_positions = [unique_sample_nums.index(num) for num in sample_nums]

                # Use different colors for each operator
                color = plt.get_cmap('tab10')(j % 10)
                ax.scatter(x_positions, values, alpha=0.7, label=operator, s=60, color=color)

        # Add threshold line
        if threshold_type == "Min":
            ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.8, linewidth=2,
                      label=f'Min Threshold: {threshold}')
        else:
            ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.8, linewidth=2,
                      label=f'Max Threshold: {threshold}')

        # Set x-axis to show sample numbers but with even spacing
        ax.set_xticks(range(len(unique_sample_nums)))
        ax.set_xticklabels([f'{num:02d}' for num in unique_sample_nums])

        ax.set_title(f'{display_name} by Sample Number')
        ax.set_xlabel('Sample Number')
        ax.set_ylabel(display_name)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # Set reasonable y-axis limits based on data
        all_values = combined_df[column_name].dropna()
        if len(all_values) > 0:
            y_min, y_max = all_values.min(), all_values.max()
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)

        plt.tight_layout()

        # Save each CTQ plot separately
        ctq_filename = display_name.replace(' ', '_').replace('/', '_').lower()
        plt.savefig(os.path.join(OUTPUT_DIRS['sample_analysis'], f'{ctq_filename}_by_samples.png'), dpi=300, bbox_inches='tight')
        plt.show()

        print(f"SUCCESS: Generated sample performance plot for {display_name}")

def analyze_pass_fail_rates(combined_df):
    """Comprehensive pass/fail analysis across all CTQs and operators"""
    # Map old column names to new ones from batch_analysis_summary.csv
    ctq_mapping = {
        "RMSE Per Pixel P95": "RMSE_Per_Pixel_P95",
        "SAM Mean": "SAM_Mean",
        "Uniformity Score": "Uniformity_Score"
    }

    ctq_metrics = ["RMSE Per Pixel P95", "SAM Mean", "Uniformity Score"]

    # Create pass/fail status for each CTQ (main 3 CTQs only)
    for ctq in ctq_metrics:
        actual_column = ctq_mapping.get(ctq, ctq)
        if actual_column not in combined_df.columns:
            continue

        threshold = MAIN_THRESHOLDS[ctq]
        if ctq == "Uniformity Score":
            # Higher is better - pass if >= threshold
            combined_df[f'{ctq}_Pass'] = combined_df[actual_column] >= threshold
        else:
            # Lower is better - pass if <= threshold
            combined_df[f'{ctq}_Pass'] = combined_df[actual_column] <= threshold

    # Overall pass status (must pass ALL main CTQs)
    pass_columns = [f'{ctq}_Pass' for ctq in ctq_metrics if f'{ctq}_Pass' in combined_df.columns]
    if pass_columns:
        combined_df['Overall_Pass'] = combined_df[pass_columns].all(axis=1)

    # Create pass/fail status for ALL CTQs
    for ctq_col, threshold in ALL_THRESHOLDS.items():
        if ctq_col in combined_df.columns:
            if ctq_col == "Uniformity_Score":
                # Higher is better - pass if >= threshold
                combined_df[f'{ctq_col}_Pass'] = combined_df[ctq_col] >= threshold
            else:
                # Lower is better - pass if <= threshold
                combined_df[f'{ctq_col}_Pass'] = combined_df[ctq_col] <= threshold

    return combined_df

def create_pass_fail_visualizations(combined_df):
    """Create comprehensive pass/fail analysis visualizations"""
    ctq_metrics = ["RMSE Per Pixel P95", "SAM Mean", "Uniformity Score"]

    # 1. Pass Rate Summary by Operator and CTQ
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pass/Fail Analysis - Pass Rates by Operator and CTQ', fontsize=16, fontweight='bold')

    # Calculate pass rates by operator for each CTQ
    pass_rate_data = []
    for operator in OPERATORS:
        if operator not in combined_df['Operator'].unique():
            continue

        op_data = combined_df[combined_df['Operator'] == operator]
        row = {'Operator': operator, 'Person': OPERATOR_INFO[operator]['person'],
               'Trial': OPERATOR_INFO[operator]['trial']}

        for ctq in ctq_metrics:
            if f'{ctq}_Pass' in combined_df.columns:
                pass_count = op_data[f'{ctq}_Pass'].sum()
                total_count = len(op_data[f'{ctq}_Pass'].dropna())
                pass_rate = (pass_count / total_count * 100) if total_count > 0 else 0
                row[f'{ctq}_PassRate'] = pass_rate

        if 'Overall_Pass' in combined_df.columns:
            pass_count = op_data['Overall_Pass'].sum()
            total_count = len(op_data['Overall_Pass'].dropna())
            row['Overall_PassRate'] = (pass_count / total_count * 100) if total_count > 0 else 0

        pass_rate_data.append(row)

    pass_rate_df = pd.DataFrame(pass_rate_data)

    # Plot pass rates for each CTQ
    plot_idx = 0
    for i, ctq in enumerate(ctq_metrics):
        if f'{ctq}_PassRate' not in pass_rate_df.columns:
            continue

        ax = axes[plot_idx // 2, plot_idx % 2]

        # Create bar plot
        x_pos = np.arange(len(pass_rate_df))
        bars = ax.bar(x_pos, pass_rate_df[f'{ctq}_PassRate'], alpha=0.7, color=plt.get_cmap('Set3')(i))

        # Color bars based on pass rate (green > 90%, yellow 70-90%, red < 70%)
        for j, bar in enumerate(bars):
            pass_rate = pass_rate_df.iloc[j][f'{ctq}_PassRate']
            if pass_rate >= 90:
                bar.set_color('green')
            elif pass_rate >= 70:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        # Add pass rate labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        ax.set_title(f'{ctq} - Pass Rate by Operator')
        ax.set_xlabel('Operator')
        ax.set_ylabel('Pass Rate (%)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pass_rate_df['Operator'], rotation=45)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        # Add horizontal lines for performance levels
        ax.axhline(y=90, color='green', linestyle=':', alpha=0.5, label='Excellent (90%+)')
        ax.axhline(y=70, color='orange', linestyle=':', alpha=0.5, label='Good (70%+)')
        ax.legend()

        plot_idx += 1

    # Overall pass rate plot
    if 'Overall_PassRate' in pass_rate_df.columns:
        ax = axes[1, 1]

        x_pos = np.arange(len(pass_rate_df))
        bars = ax.bar(x_pos, pass_rate_df['Overall_PassRate'], alpha=0.7)

        # Color bars based on overall pass rate
        for j, bar in enumerate(bars):
            pass_rate = pass_rate_df.iloc[j]['Overall_PassRate']
            if pass_rate >= 90:
                bar.set_color('green')
            elif pass_rate >= 70:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        # Add pass rate labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        ax.set_title('Overall Pass Rate (All CTQs) by Operator')
        ax.set_xlabel('Operator')
        ax.set_ylabel('Pass Rate (%)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pass_rate_df['Operator'], rotation=45)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        # Add horizontal lines for performance levels
        ax.axhline(y=90, color='green', linestyle=':', alpha=0.5, label='Excellent (90%+)')
        ax.axhline(y=70, color='orange', linestyle=':', alpha=0.5, label='Good (70%+)')
        ax.legend()

    plt.tight_layout()
    plt.savefig('analysis_results/pass_fail_rates_by_operator.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Sample Pass/Fail Analysis Across All Operators
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    fig.suptitle('Sample Pass/Fail Analysis - Overall Performance Across All Operators', fontsize=16, fontweight='bold')

    # Get unique sample numbers
    all_sample_nums = []
    for operator in OPERATORS:
        if operator in combined_df['Operator'].unique():
            op_data = combined_df[combined_df['Operator'] == operator]
            sample_nums = op_data['Sample'].str.replace('Sample', '').str.replace('sample', '').astype(int)
            all_sample_nums.extend(sample_nums.tolist())

    unique_sample_nums = sorted(list(set(all_sample_nums)))

    # Calculate pass rates for each sample across all operators
    sample_pass_rates = []
    for sample_num in unique_sample_nums:
        sample_data = combined_df[
            combined_df['Sample'].str.replace('Sample', '').str.replace('sample', '').astype(int) == sample_num
        ]

        if len(sample_data) > 0 and 'Overall_Pass' in combined_df.columns:
            pass_count = sample_data['Overall_Pass'].sum()
            total_count = len(sample_data['Overall_Pass'].dropna())
            pass_rate = (pass_count / total_count * 100) if total_count > 0 else 0
            sample_pass_rates.append(pass_rate)
        else:
            sample_pass_rates.append(0)

    # Create bar plot for sample pass rates
    x_pos = np.arange(len(unique_sample_nums))
    bars = ax.bar(x_pos, sample_pass_rates, alpha=0.7)

    # Color bars based on pass rate
    for i, (bar, pass_rate) in enumerate(zip(bars, sample_pass_rates)):
        if pass_rate >= 90:
            bar.set_color('green')
        elif pass_rate >= 70:
            bar.set_color('orange')
        elif pass_rate > 0:
            bar.set_color('red')
        else:
            bar.set_color('darkred')

        # Add pass rate labels on bars
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
               f'{pass_rate:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_title('Overall Pass Rate by Sample (Across All Operators)')
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('Pass Rate (%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{num:02d}' for num in unique_sample_nums])
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Add horizontal lines for performance levels
    ax.axhline(y=90, color='green', linestyle=':', alpha=0.7, label='Excellent (90%+)')
    ax.axhline(y=70, color='orange', linestyle=':', alpha=0.7, label='Good (70%+)')
    ax.axhline(y=50, color='red', linestyle=':', alpha=0.7, label='Poor (50%+)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('analysis_results/sample_pass_fail_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Person vs Trial Analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pass/Fail Analysis - Person vs Trial Comparison', fontsize=16, fontweight='bold')

    # Calculate pass rates by person and trial
    person_trial_data = []
    for person in ['Khang', 'Kidu', 'Luel', 'Anirban']:
        for trial in [1, 2]:
            operator = f"{person}T{trial}"
            if operator in combined_df['Operator'].unique():
                op_data = combined_df[combined_df['Operator'] == operator]

                row = {'Person': person, 'Trial': trial, 'Operator': operator}

                for ctq in ctq_metrics:
                    if f'{ctq}_Pass' in combined_df.columns:
                        pass_count = op_data[f'{ctq}_Pass'].sum()
                        total_count = len(op_data[f'{ctq}_Pass'].dropna())
                        pass_rate = (pass_count / total_count * 100) if total_count > 0 else 0
                        row[f'{ctq}_PassRate'] = pass_rate

                if 'Overall_Pass' in combined_df.columns:
                    pass_count = op_data['Overall_Pass'].sum()
                    total_count = len(op_data['Overall_Pass'].dropna())
                    row['Overall_PassRate'] = (pass_count / total_count * 100) if total_count > 0 else 0

                person_trial_data.append(row)

    person_trial_df = pd.DataFrame(person_trial_data)

    # Plot comparison for each CTQ + Overall
    all_ctqs = ctq_metrics + ['Overall']
    for idx, ctq in enumerate(all_ctqs):
        if idx >= 4:  # Only 4 subplots available
            break

        ax = axes[idx // 2, idx % 2]

        rate_col = f'{ctq}_PassRate' if ctq != 'Overall' else 'Overall_PassRate'
        if rate_col not in person_trial_df.columns:
            continue

        # Create grouped bar plot
        persons = ['Khang', 'Kidu', 'Luel', 'Anirban']
        trial1_rates = []
        trial2_rates = []

        for person in persons:
            t1_data = person_trial_df[(person_trial_df['Person'] == person) & (person_trial_df['Trial'] == 1)]
            t2_data = person_trial_df[(person_trial_df['Person'] == person) & (person_trial_df['Trial'] == 2)]

            trial1_rates.append(t1_data[rate_col].iloc[0] if len(t1_data) > 0 else 0)
            trial2_rates.append(t2_data[rate_col].iloc[0] if len(t2_data) > 0 else 0)

        x_pos = np.arange(len(persons))
        width = 0.35

        bars1 = ax.bar(x_pos - width/2, trial1_rates, width, label='Trial 1', alpha=0.8, color='lightblue')
        bars2 = ax.bar(x_pos + width/2, trial2_rates, width, label='Trial 2', alpha=0.8, color='lightcoral')

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

        ax.set_title(f'{ctq} Pass Rate - Trial Comparison')
        ax.set_xlabel('Person')
        ax.set_ylabel('Pass Rate (%)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(persons)
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('analysis_results/person_trial_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return pass_rate_df, person_trial_df

def generate_ctq_report(combined_df, gage_rr_stats, pass_rate_df=None, person_trial_df=None):
    """Generate comprehensive CTQ analysis report"""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("GAGE R&R CTQ ANALYSIS REPORT")
    report_lines.append("=" * 80)

    # Overview
    report_lines.append(f"Analysis Date: November 30, 2025")
    report_lines.append(f"Total Operators: {len(OPERATORS)} (4 persons × 2 trials)")
    report_lines.append(f"Total Samples: {len(combined_df)}")
    report_lines.append(f"CTQs Analyzed: RMSE Per Pixel P95, SAM Mean, Uniformity Score")
    report_lines.append("")

    # Operator Summary
    report_lines.append("OPERATOR SUMMARY:")
    report_lines.append("-" * 40)
    for operator in OPERATORS:
        if operator in combined_df['Operator'].unique():
            count = len(combined_df[combined_df['Operator'] == operator])
            person = OPERATOR_INFO[operator]['person']
            trial = OPERATOR_INFO[operator]['trial']
            report_lines.append(f"  {operator}: {person} Trial {trial} ({count} samples)")
    report_lines.append("")

    # CTQ Analysis
    ctq_metrics = ["RMSE Per Pixel P95", "SAM Mean", "Uniformity Score"]

    for ctq in ctq_metrics:
        if ctq not in gage_rr_stats:
            continue

        stats = gage_rr_stats[ctq]
        threshold = MAIN_THRESHOLDS[ctq]

        report_lines.append(f"CTQ: {ctq}")
        report_lines.append("=" * len(f"CTQ: {ctq}"))

        # Basic statistics
        report_lines.append(f"Mean Value: {stats['mean_value']:.4f}")
        report_lines.append(f"Standard Deviation: {stats['std_value']:.4f}")
        report_lines.append(f"Coefficient of Variation: {stats['cv_percent']:.1f}%")
        report_lines.append(f"Threshold: {threshold} ({'min' if ctq == 'Uniformity Score' else 'max'})")
        report_lines.append("")

        # Gage R&R Components
        report_lines.append("Gage R&R Variance Components:")
        report_lines.append(f"  Repeatability: {stats['repeatability_percent']:.1f}%")
        report_lines.append(f"  Reproducibility: {stats['reproducibility_percent']:.1f}%")
        report_lines.append(f"  Total R&R: {stats['measurement_system_percent']:.1f}%")
        # Part-to-Part variation can be calculated as the complement of measurement system variation
        part_to_part_pct = 100 - stats['measurement_system_percent']
        report_lines.append(f"  Part-to-Part: {part_to_part_pct:.1f}%")
        report_lines.append("")

        # R&R Acceptance Criteria (typical industry standards)
        rr_pct = stats['measurement_system_percent']
        if rr_pct < 10:
            acceptance = "EXCELLENT"
        elif rr_pct < 30:
            acceptance = "ACCEPTABLE"
        else:
            acceptance = "NEEDS IMPROVEMENT"

        report_lines.append(f"R&R Assessment: {acceptance} ({rr_pct:.1f}%)")
        report_lines.append("")

        # Pass rate analysis
        ctq_mapping = {
            "RMSE Per Pixel P95": "RMSE_Per_Pixel_P95",
            "SAM Mean": "SAM_Mean",
            "Uniformity Score": "Uniformity_Score"
        }
        actual_column = ctq_mapping.get(ctq, ctq)

        if actual_column in combined_df.columns:
            data = combined_df.dropna(subset=[actual_column])
            if ctq == "Uniformity Score":
                pass_count = len(data[data[actual_column] >= threshold])
            else:
                pass_count = len(data[data[actual_column] <= threshold])

            pass_rate = (pass_count / len(data)) * 100
            report_lines.append(f"Pass Rate: {pass_count}/{len(data)} ({pass_rate:.1f}%)")
        else:
            report_lines.append(f"Pass Rate: N/A (column {actual_column} not found)")

        report_lines.append("")
        report_lines.append("-" * 60)
        report_lines.append("")

    # PASS/FAIL ANALYSIS SECTION
    overall_pass_rate = 0  # Initialize to avoid undefined reference
    failing_samples = []  # Initialize to avoid undefined reference
    if pass_rate_df is not None and 'Overall_Pass' in combined_df.columns:
        report_lines.append("PASS/FAIL ANALYSIS:")
        report_lines.append("=" * 20)

        # Overall pass rate statistics
        overall_pass_count = combined_df['Overall_Pass'].sum()
        overall_total_count = len(combined_df['Overall_Pass'].dropna())
        overall_pass_rate = (overall_pass_count / overall_total_count * 100) if overall_total_count > 0 else 0

        report_lines.append(f"Overall System Pass Rate: {overall_pass_count}/{overall_total_count} ({overall_pass_rate:.1f}%)")
        report_lines.append("")

        # Pass rates by operator
        report_lines.append("PASS RATES BY OPERATOR:")
        report_lines.append("-" * 30)
        for _, row in pass_rate_df.iterrows():
            operator = row['Operator']
            person = row['Person']
            trial = row['Trial']

            report_lines.append(f"{operator} ({person} T{trial}):")

            for ctq in ctq_metrics:
                rate_col = f'{ctq}_PassRate'
                if rate_col in pass_rate_df.columns:
                    pass_rate = row[rate_col]
                    status = "SUCCESS:" if pass_rate >= 90 else "WARNING:" if pass_rate >= 70 else "ERROR:"
                    report_lines.append(f"  {ctq}: {pass_rate:.1f}% {status}")

            if 'Overall_PassRate' in pass_rate_df.columns:
                overall_rate = row['Overall_PassRate']
                status = "SUCCESS:" if overall_rate >= 90 else "WARNING:" if overall_rate >= 70 else "ERROR:"
                report_lines.append(f"  Overall: {overall_rate:.1f}% {status}")

            report_lines.append("")

        # Pass rate performance classification
        report_lines.append("OPERATOR PERFORMANCE CLASSIFICATION:")
        report_lines.append("-" * 40)

        if 'Overall_PassRate' in pass_rate_df.columns:
            excellent_ops = pass_rate_df[pass_rate_df['Overall_PassRate'] >= 90]['Operator'].tolist()
            good_ops = pass_rate_df[(pass_rate_df['Overall_PassRate'] >= 70) & (pass_rate_df['Overall_PassRate'] < 90)]['Operator'].tolist()
            poor_ops = pass_rate_df[pass_rate_df['Overall_PassRate'] < 70]['Operator'].tolist()

            report_lines.append(f"Excellent (≥90%): {', '.join(excellent_ops) if excellent_ops else 'None'}")
            report_lines.append(f"Good (70-89%): {', '.join(good_ops) if good_ops else 'None'}")
            report_lines.append(f"Needs Improvement (<70%): {', '.join(poor_ops) if poor_ops else 'None'}")
            report_lines.append("")

        # Sample analysis
        unique_samples = combined_df['Sample'].str.replace('Sample', '').str.replace('sample', '').astype(int).unique()
        unique_samples = sorted(unique_samples)

        # Find consistently failing samples
        for sample_num in unique_samples:
            sample_data = combined_df[
                combined_df['Sample'].str.replace('Sample', '').str.replace('sample', '').astype(int) == sample_num
            ]

            if len(sample_data) > 0:
                pass_count = sample_data['Overall_Pass'].sum()
                total_count = len(sample_data['Overall_Pass'].dropna())
                pass_rate = (pass_count / total_count * 100) if total_count > 0 else 0

                if pass_rate < 50:  # Less than 50% pass rate across operators
                    failing_samples.append(f"Sample{sample_num:02d} ({pass_rate:.0f}%)")

        report_lines.append("CONSISTENTLY FAILING SAMPLES (<50% pass rate):")
        report_lines.append("-" * 50)
        if failing_samples:
            for sample in failing_samples:
                report_lines.append(f"  • {sample}")
        else:
            report_lines.append("  None - All samples have ≥50% pass rate")
        report_lines.append("")

        # Trial consistency analysis
        if person_trial_df is not None:
            report_lines.append("TRIAL CONSISTENCY ANALYSIS:")
            report_lines.append("-" * 30)

            for person in ['Khang', 'Kidu', 'Luel', 'Anirban']:
                t1_data = person_trial_df[(person_trial_df['Person'] == person) & (person_trial_df['Trial'] == 1)]
                t2_data = person_trial_df[(person_trial_df['Person'] == person) & (person_trial_df['Trial'] == 2)]

                if len(t1_data) > 0 and len(t2_data) > 0 and 'Overall_PassRate' in person_trial_df.columns:
                    t1_rate = t1_data['Overall_PassRate'].iloc[0]
                    t2_rate = t2_data['Overall_PassRate'].iloc[0]
                    diff = abs(t1_rate - t2_rate)

                    consistency = "Excellent" if diff < 5 else "Good" if diff < 15 else "Poor"
                    report_lines.append(f"{person}: T1={t1_rate:.1f}%, T2={t2_rate:.1f}%, Diff={diff:.1f}% ({consistency})")

            report_lines.append("")

        report_lines.append("-" * 60)
        report_lines.append("")

    # Overall Assessment
    report_lines.append("OVERALL ASSESSMENT:")
    report_lines.append("=" * 20)

    avg_rr = np.mean([gage_rr_stats[ctq]['measurement_system_percent']
                     for ctq in ctq_metrics if ctq in gage_rr_stats])

    if avg_rr < 10:
        overall = "EXCELLENT"
    elif avg_rr < 30:
        overall = "ACCEPTABLE"
    else:
        overall = "NEEDS IMPROVEMENT"

    report_lines.append(f"Average R&R: {avg_rr:.1f}%")
    report_lines.append(f"Overall System: {overall}")

    if 'Overall_Pass' in combined_df.columns:
        report_lines.append(f"Overall Pass Rate: {overall_pass_rate:.1f}%")

    report_lines.append("")

    # Recommendations
    report_lines.append("RECOMMENDATIONS:")
    report_lines.append("-" * 15)

    for ctq in ctq_metrics:
        if ctq not in gage_rr_stats:
            continue
        stats = gage_rr_stats[ctq]

        if stats['repeatability_percent'] > 15:
            report_lines.append(f"• {ctq}: High repeatability variation - check measurement consistency")

        if stats['reproducibility_percent'] > 15:
            report_lines.append(f"• {ctq}: High reproducibility variation - standardize operator procedures")

        if stats['measurement_system_percent'] > 30:
            report_lines.append(f"• {ctq}: Poor measurement system - requires significant improvement")

    if avg_rr < 10:
        report_lines.append("• Measurement system performs excellently for all CTQs")

    # Pass/fail specific recommendations
    if pass_rate_df is not None and 'Overall_PassRate' in pass_rate_df.columns:
        poor_operators = pass_rate_df[pass_rate_df['Overall_PassRate'] < 70]
        if len(poor_operators) > 0:
            report_lines.append(f"• {len(poor_operators)} operators have <70% pass rate - require additional training")

        if len(failing_samples) > 0:
            report_lines.append(f"• {len(failing_samples)} samples consistently fail - investigate root causes")

    report_lines.append("")
    report_lines.append("=" * 80)

    # Save report
    report_text = "\n".join(report_lines)

    os.makedirs("analysis_results", exist_ok=True)
    with open("analysis_results/gage_rr_ctq_report.txt", "w") as f:
        f.write(report_text)

    print("\n" + report_text)

    return report_text

def analyze_defect_detection_performance(combined_df):
    """Analyze how well the measurement system detects defects vs acceptable samples"""

    detection_results = []

    # Get unique sample numbers
    unique_samples = combined_df['Sample'].str.replace('Sample', '').str.replace('sample', '').astype(int).unique()
    unique_samples = sorted(unique_samples)

    for sample_num in unique_samples:
        sample_data = combined_df[
            combined_df['Sample'].str.replace('Sample', '').str.replace('sample', '').astype(int) == sample_num
        ]

        if len(sample_data) > 0 and 'Overall_Pass' in combined_df.columns:
            pass_count = sample_data['Overall_Pass'].sum()
            total_count = len(sample_data['Overall_Pass'].dropna())
            pass_rate = (pass_count / total_count * 100) if total_count > 0 else 0

            # Determine expected result
            expected_result = "PASS" if sample_num in EXPECTED_GOOD_SAMPLES else "FAIL"

            # Determine actual result (using majority vote across operators)
            actual_result = "PASS" if pass_rate >= 50 else "FAIL"

            # Determine detection accuracy
            correct_detection = (expected_result == actual_result)

            detection_results.append({
                'Sample': sample_num,
                'Expected': expected_result,
                'Actual': actual_result,
                'Pass_Rate': pass_rate,
                'Correct_Detection': correct_detection,
                'Sample_Type': 'Good' if sample_num in EXPECTED_GOOD_SAMPLES else 'Defective'
            })

    detection_df = pd.DataFrame(detection_results)

    # Calculate detection performance metrics
    total_samples = len(detection_df)
    correct_detections = detection_df['Correct_Detection'].sum()
    overall_accuracy = (correct_detections / total_samples * 100) if total_samples > 0 else 0

    # Calculate specific metrics
    good_samples = detection_df[detection_df['Sample_Type'] == 'Good']
    defective_samples = detection_df[detection_df['Sample_Type'] == 'Defective']

    # Sensitivity (True Positive Rate) - correctly identifying defective samples as defective
    true_negatives = defective_samples[defective_samples['Correct_Detection'] == True]
    sensitivity = (len(true_negatives) / len(defective_samples) * 100) if len(defective_samples) > 0 else 0

    # Specificity (True Negative Rate) - correctly identifying good samples as good
    true_positives = good_samples[good_samples['Correct_Detection'] == True]
    specificity = (len(true_positives) / len(good_samples) * 100) if len(good_samples) > 0 else 0

    return detection_df, {
        'overall_accuracy': overall_accuracy,
        'sensitivity': sensitivity,  # Ability to detect defects
        'specificity': specificity,  # Ability to identify good parts
        'total_samples': total_samples,
        'correct_detections': correct_detections,
        'good_samples_tested': len(good_samples),
        'defective_samples_tested': len(defective_samples)
    }

def create_defect_detection_visualizations(detection_df, detection_metrics):
    """Create visualizations for defect detection performance"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('IC Grating Defect Detection Performance Analysis', fontsize=16, fontweight='bold')

    # 1. Detection Accuracy by Sample
    ax1 = axes[0, 0]

    colors = ['green' if correct else 'red' for correct in detection_df['Correct_Detection']]
    bars = ax1.bar(range(len(detection_df)), detection_df['Pass_Rate'], color=colors, alpha=0.7)

    # Add sample numbers and expected vs actual labels
    for i, (_, row) in enumerate(detection_df.iterrows()):
        sample_num = row['Sample']
        expected = row['Expected']
        actual = row['Actual']

        # Add sample number at the bottom
        ax1.text(i, -5, f'{sample_num:02d}', ha='center', va='top', fontweight='bold')

        # Add expected vs actual at the top
        status = "✓" if row['Correct_Detection'] else "✗"
        ax1.text(i, row['Pass_Rate'] + 2, f'{expected}\nvs\n{actual}\n{status}',
                ha='center', va='bottom', fontsize=8)

    ax1.set_title('Sample Detection Performance\n(Expected vs Actual Results)')
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('Pass Rate (%)')
    ax1.set_ylim(-10, 110)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Decision Threshold (50%)')
    ax1.legend()

    # 2. Detection Performance Summary
    ax2 = axes[0, 1]

    metrics = ['Overall\nAccuracy', 'Sensitivity\n(Defect Detection)', 'Specificity\n(Good Part ID)']
    values = [detection_metrics['overall_accuracy'], detection_metrics['sensitivity'], detection_metrics['specificity']]

    bars = ax2.bar(metrics, values, color=['blue', 'red', 'green'], alpha=0.7)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax2.set_title('Detection Performance Metrics')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3)

    # Add performance level lines
    ax2.axhline(y=90, color='green', linestyle=':', alpha=0.5, label='Excellent (90%+)')
    ax2.axhline(y=70, color='orange', linestyle=':', alpha=0.5, label='Good (70%+)')
    ax2.legend()

    # 3. Sample Type Analysis
    ax3 = axes[1, 0]

    sample_types = ['Good Samples\n(Should Pass)', 'Defective Samples\n(Should Fail)']
    correct_counts = [
        len(detection_df[(detection_df['Sample_Type'] == 'Good') & (detection_df['Correct_Detection'] == True)]),
        len(detection_df[(detection_df['Sample_Type'] == 'Defective') & (detection_df['Correct_Detection'] == True)])
    ]
    total_counts = [
        len(detection_df[detection_df['Sample_Type'] == 'Good']),
        len(detection_df[detection_df['Sample_Type'] == 'Defective'])
    ]

    x_pos = np.arange(len(sample_types))
    bars1 = ax3.bar(x_pos - 0.2, correct_counts, 0.4, label='Correctly Identified', color='green', alpha=0.7)
    bars2 = ax3.bar(x_pos + 0.2, [total - correct for total, correct in zip(total_counts, correct_counts)],
                   0.4, label='Incorrectly Identified', color='red', alpha=0.7)

    # Add value labels
    for bars, values in [(bars1, correct_counts), (bars2, [total - correct for total, correct in zip(total_counts, correct_counts)])]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{value}', ha='center', va='bottom', fontweight='bold')

    ax3.set_title('Detection Performance by Sample Type')
    ax3.set_xlabel('Sample Type')
    ax3.set_ylabel('Number of Samples')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(sample_types)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Confusion Matrix
    ax4 = axes[1, 1]

    # Calculate confusion matrix values
    tp = len(detection_df[(detection_df['Sample_Type'] == 'Good') & (detection_df['Actual'] == 'PASS')])  # True Positive
    fn = len(detection_df[(detection_df['Sample_Type'] == 'Good') & (detection_df['Actual'] == 'FAIL')])  # False Negative
    tn = len(detection_df[(detection_df['Sample_Type'] == 'Defective') & (detection_df['Actual'] == 'FAIL')])  # True Negative
    fp = len(detection_df[(detection_df['Sample_Type'] == 'Defective') & (detection_df['Actual'] == 'PASS')])  # False Positive

    confusion_matrix = np.array([[tp, fn], [fp, tn]])

    im = ax4.imshow(confusion_matrix, cmap='Blues', alpha=0.7)

    # Add text annotations
    labels = [['True Positive\n(Good→Pass)', 'False Negative\n(Good→Fail)'],
             ['False Positive\n(Defect→Pass)', 'True Negative\n(Defect→Fail)']]

    for i in range(2):
        for j in range(2):
            text = ax4.text(j, i, f'{labels[i][j]}\n{confusion_matrix[i, j]}',
                          ha="center", va="center", fontweight='bold')

    ax4.set_title('Detection Confusion Matrix')
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Pass', 'Fail'])
    ax4.set_yticklabels(['Good', 'Defective'])

    plt.tight_layout()
    plt.savefig('analysis_results/defect_detection_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

    return detection_df

# plot_gage_rr_summary is imported from modules.lib_gage_rr
# Using the module version which has the correct signature

def plot_gage_rr_summary_legacy(combined_df, summary_metrics, ctq, output_folder, anova_results=None):
    """
    Legacy local implementation - kept for reference but not used
    Use plot_gage_rr_summary from lib_gage_rr instead
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
    n_ops = len(OPERATORS)
    n_rep = combined_df.groupby(['Sample', 'Operator']).size().mean()

    # Get threshold (spec limits) from MAIN_THRESHOLDS
    threshold = MAIN_THRESHOLDS.get(ctq, 0)

    # Set spec limits based on CTQ type
    if ctq == "Uniformity Score":
        lsl = MAIN_THRESHOLDS.get("Uniformity Score", 0.6)
        usl = 1.0  # Maximum possible uniformity score
    elif ctq == "RMSE Per Pixel P95":
        lsl = 0
        usl = MAIN_THRESHOLDS.get("RMSE Per Pixel P95", 0.08)
    else:  # SAM Mean
        lsl = 0
        usl = MAIN_THRESHOLDS.get("SAM Mean", 2.6)

    # Calculate GRR percentage
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

        # Add data rows
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
    part_pct = summary_metrics.get('part_to_part_percent', 0)

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


def plot_gage_rr_diagnostics(combined_df, ctq, output_folder):
    """
    Three-panel diagnostic plots: box-by-operator, sample means, run chart
    Adapted from original Gage R&R analysis
    """
    from matplotlib.ticker import FormatStrFormatter

    # Map CTQ names
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

    # Panel 1: boxplots with jittered points per operator
    ax1 = axes[0]
    operators = sorted(plot_data["Operator"].unique())
    bp_data = [plot_data[plot_data["Operator"] == op][actual_column].values for op in operators]
    bp = ax1.boxplot(bp_data, labels=operators, patch_artist=True, widths=0.6)

    for patch in bp["boxes"]:
        patch.set_facecolor("#B0E0E6")
        patch.set_edgecolor("black")
        patch.set_linewidth(1)

    # Add jittered points
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

    # Panel 3: run chart (operator-by-sample means)
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
    print(f"  SUCCESS: Diagnostic plot saved: {os.path.basename(plot_path)}")


def plot_distinct_categories_analysis(combined_df, ctq, metrics, output_folder):
    """
    Create distinct categories analysis plot showing sample discrimination
    Samples are ordered by mean value and assigned to categories
    """
    # Map CTQ names
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

    # Get GRR standard deviation from metrics
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

    # Color palette for categories
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

        # Add sample name
        sample_numeric = row['Sample'].replace('Sample', '').replace('sample', '')
        ax.text(
            i, category + 0.15, sample_numeric,
            ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=45
        )

        # Add mean value
        ax.text(
            i, category - 0.15, f"{row['Mean_Value']:.3f}",
            ha='center', va='top', fontsize=8, alpha=0.7
        )

    # Customize plot
    ax.set_xlabel('Samples (Ordered by Mean Value)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distinct Category', fontsize=12, fontweight='bold')

    y_ticks = sorted(sample_stats['Distinct_Category'].unique())
    ax.set_yticks(y_ticks)
    ax.set_ylim(0.5, max(y_ticks) + 0.5)

    ax.set_xticks(range(len(sample_stats)))
    ax.set_xticklabels([], rotation=45)

    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Add title with NDC information
    ndc = metrics.get('ndc', 0)
    part_to_part_std = metrics.get('part_to_part_std', 0)

    title = f"{ctq} - Distinct Categories Analysis\n"
    title += f"NDC: {ndc:.1f} | GRR σ: {sd_grr:.4f} | Part σ: {part_to_part_std:.4f}"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Categories")

    # Add interpretation
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

    # Add horizontal lines separating categories
    for i in range(1, max(y_ticks)):
        ax.axhline(y=i + 0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()

    out_path = os.path.join(output_folder, "distinct_categories", f"{ctq.replace(' ', '_')}_distinct_categories_analysis.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"  SUCCESS: Distinct categories plot saved: {os.path.basename(out_path)}")


def main():
    """Main analysis function using lib_gage_rr module"""
    print("Comprehensive Gage R&R Analysis for Hyperspectral Defect Detection")
    print("=" * 70)
    print(f"Operators configured: {len(OPERATORS)}")
    print(f"Expected good samples: {EXPECTED_GOOD_SAMPLES}")
    print(f"Expected defective samples: {EXPECTED_DEFECTIVE_SAMPLES}")
    print("\nQuality Thresholds (from config.json):")
    print(f"  RMSE Per Pixel P95: ≤ {MAIN_THRESHOLDS['RMSE Per Pixel P95']}")
    print(f"  SAM Mean: ≤ {MAIN_THRESHOLDS['SAM Mean']}")
    print(f"  Uniformity Score: ≥ {MAIN_THRESHOLDS['Uniformity Score']}")
    print("=" * 70)

    # Create organized output directory structure
    print("\nInitializing output directories...")
    create_output_directories()
    print(f"  Output root: {OUTPUT_DIRS['root']}")
    print(f"  Plots: {OUTPUT_DIRS['plots']}")
    print(f"  Data: {OUTPUT_DIRS['data']}")
    print(f"  Reports: {OUTPUT_DIRS['reports']}")

    # Step 1: Load data from all operators using module function
    print("\nStep 1: Loading operator data...")
    all_data = []

    for operator in OPERATORS:
        # Use module function with the correct path where data exists
        df = load_operator_data(operator, "results/450-950")

        if df is not None:
            # Add operator info that module function doesn't include
            df['Person'] = OPERATOR_INFO[operator]["person"]
            df['Trial'] = OPERATOR_INFO[operator]["trial"]
            all_data.append(df)
            print(f"  SUCCESS: Loaded {len(df)} samples from {operator}")
        else:
            print(f"  WARNING: No data found for {operator}")

    if not all_data:
        print("ERROR: No data loaded. Please check that results exist.")
        return 1

    # Step 2: Combine and prepare data
    print(f"\nStep 2: Data preparation...")
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"  Combined dataset: {len(combined_df)} total samples from {len(all_data)} operators")

    # Extract numeric values from SAM columns using module function
    sam_columns = [col for col in combined_df.columns if 'SAM' in col]
    for col in sam_columns:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].apply(extract_numeric_value)

    # Step 3: Statistical Gage R&R analysis using simplified statistics
    # Note: This is a single-measurement design (no replications within operator-sample)
    # Traditional ANOVA-based Gage R&R requires replications, so we use an alternative approach
    print(f"\nStep 3: Statistical Gage R&R analysis...")
    print("  Using measurement system statistics (single-measurement design)")

    # Use plots directory for outputs
    output_dir = OUTPUT_DIRS['plots']

    # Calculate Gage R&R statistics using the function from lib_gage_rr
    gage_rr_results = calculate_gage_rr_statistics(combined_df)

    if gage_rr_results:
        print(f"  SUCCESS: Analyzed {len(gage_rr_results)} CTQ metrics")
        for ctq, stats in gage_rr_results.items():
            print(f"    {ctq}:")
            print(f"      Measurement System: {stats['measurement_system_percent']:.1f}% - {stats['measurement_system_assessment']}")
            print(f"      Discrimination: S/N={stats['signal_to_noise_ratio']:.1f} - {stats['discrimination_assessment']}")
    else:
        print("  WARNING: No Gage R&R results calculated")

    # Step 4: Categorize sample performance using module function
    print(f"\nStep 4: Sample performance categorization...")
    combined_df = categorize_sample_performance(combined_df, ALL_THRESHOLDS)

    # Step 4.5: Create additional Gage R&R visualizations from original analysis
    print(f"\nStep 4.5: Creating enhanced Gage R&R visualizations...")
    ctq_metrics_for_plots = ["RMSE Per Pixel P95", "SAM Mean", "Uniformity Score"]

    for ctq in ctq_metrics_for_plots:
        print(f"  Creating plots for {ctq}...")

        # Get the metrics for this CTQ from gage_rr_results
        if ctq in gage_rr_results:
            stats = gage_rr_results[ctq]

            # Convert statistics to the format expected by the plotting functions
            total_var = stats['std_value'] ** 2
            part_var = total_var * (1 - stats['measurement_system_percent']/100)

            # Calculate NDC (Number of Distinct Categories)
            # NDC = 1.41 * (part std / measurement system std)
            part_std = np.sqrt(max(0, part_var))
            ms_std = stats['total_measurement_std']
            ndc = int(1.41 * (part_std / ms_std)) if ms_std > 0 else 1

            summary_metrics = {
                'measurement_system_percent': stats['measurement_system_percent'],
                'repeatability_percent': stats['repeatability_percent'],
                'reproducibility_percent': stats['reproducibility_percent'],
                'part_to_part_percent': 100 - stats['measurement_system_percent'],
                'ndc': ndc,
                'avg_repeatability_std': stats['repeatability_std'],
                'avg_reproducibility_std': stats['reproducibility_std'],
                'part_to_part_std': part_std,
            }

            try:
                # Create summary dashboard
                plot_gage_rr_summary(combined_df, summary_metrics, ctq, output_dir, MAIN_THRESHOLDS, anova_results=None)

                # Create diagnostic plots
                plot_gage_rr_diagnostics(combined_df, ctq, output_dir)

                # Create distinct categories analysis
                plot_distinct_categories_analysis(combined_df, ctq, summary_metrics, output_dir)

            except Exception as e:
                print(f"    WARNING: Plot creation failed for {ctq}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"    WARNING: No Gage R&R results available for {ctq}")

    # Step 4.6: Create sample analysis plots
    print(f"\nStep 4.6: Creating sample analysis plots...")
    create_ctq_summary_plots(combined_df, gage_rr_results)

    # Step 5: Defect detection analysis
    print(f"\nStep 5: Defect detection performance analysis...")
    detection_results = analyze_defect_detection_performance(combined_df)

    # Step 6: Save results
    print(f"\nStep 6: Saving results...")

    # Save combined data to data directory
    combined_output_path = os.path.join(OUTPUT_DIRS['data'], "all_ctq_data_with_thresholds.csv")
    combined_df.to_csv(combined_output_path, index=False)
    print(f"  Combined data saved: {combined_output_path}")

    # Create summary report in reports directory
    create_analysis_summary_report(gage_rr_results, detection_results, OUTPUT_DIRS['reports'])

    print(f"\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"Results saved to organized directories:")
    print(f"  Root: {OUTPUT_DIRS['root']}/")
    print(f"  Plots: {OUTPUT_DIRS['plots']}/")
    print(f"    - Gage R&R Dashboards: {OUTPUT_DIRS['gage_rr']}/")
    print(f"    - Diagnostics: {OUTPUT_DIRS['diagnostics']}/")
    print(f"    - Distinct Categories: {OUTPUT_DIRS['categories']}/")
    print(f"    - Sample Analysis: {OUTPUT_DIRS['sample_analysis']}/")
    print(f"  Data: {OUTPUT_DIRS['data']}/")
    print(f"  Reports: {OUTPUT_DIRS['reports']}/")
    print(f"\nSummary:")
    print(f"  Total measurements analyzed: {len(combined_df)}")
    print(f"  Operators analyzed: {len(all_data)}")
    print(f"  CTQ metrics analyzed: {len(gage_rr_results)}")
    print("="*70)

    return 0

def create_analysis_summary_report(gage_rr_results, detection_results, output_dir):
    """Create a comprehensive summary report using analysis results"""
    try:
        summary_path = os.path.join(output_dir, "comprehensive_analysis_summary.txt")

        with open(summary_path, 'w') as f:
            f.write("Comprehensive Gage R&R Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
            f.write(f"Operators: {', '.join(OPERATORS)}\n")
            f.write(f"Expected Good Samples: {EXPECTED_GOOD_SAMPLES}\n")
            f.write(f"Expected Defective Samples: {EXPECTED_DEFECTIVE_SAMPLES}\n\n")

            # Gage R&R Statistical Results
            f.write("GAGE R&R STATISTICAL ANALYSIS\n")
            f.write("-" * 30 + "\n\n")

            for metric, results in gage_rr_results.items():
                if results and 'variance_components' in results:
                    vc = results['variance_components']
                    f.write(f"{metric}:\n")
                    f.write(f"  Gage R&R %: {vc['percentages']['gage_rr']:.1f}%\n")
                    f.write(f"  Repeatability %: {vc['percentages']['repeatability']:.1f}%\n")
                    f.write(f"  Reproducibility %: {vc['percentages']['reproducibility']:.1f}%\n")
                    f.write(f"  Part-to-Part %: {vc['percentages']['part_to_part']:.1f}%\n")
                    f.write(f"  NDC: {vc['ndc']}\n")

                    # Assessment
                    grr_pct = vc['percentages']['gage_rr']
                    if grr_pct < 10:
                        assessment = "EXCELLENT"
                    elif grr_pct < 30:
                        assessment = "ACCEPTABLE"
                    else:
                        assessment = "NEEDS IMPROVEMENT"

                    f.write(f"  Assessment: {assessment}\n\n")

            # Detection Performance
            if detection_results and isinstance(detection_results, pd.DataFrame):
                f.write("DEFECT DETECTION PERFORMANCE\n")
                f.write("-" * 30 + "\n\n")

                # Calculate overall statistics
                total_measurements = len(detection_results)
                correct_classifications = len(detection_results[detection_results['Detection_Correct'] == True])
                accuracy = (correct_classifications / total_measurements * 100) if total_measurements > 0 else 0

                f.write(f"Total Measurements: {total_measurements}\n")
                f.write(f"Correct Classifications: {correct_classifications}\n")
                f.write(f"Overall Accuracy: {accuracy:.1f}%\n\n")

            f.write(f"Analysis completed successfully!\n")

        print(f"  Comprehensive summary saved: {summary_path}")

    except Exception as e:
        print(f"  ERROR: Failed to create summary report: {str(e)}")



if __name__ == "__main__":
    try:
        result = main()
        sys.exit(result if result is not None else 0)
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
