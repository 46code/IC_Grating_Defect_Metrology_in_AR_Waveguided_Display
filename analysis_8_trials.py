#!/usr/bin/env python3
"""
Gage R&R CTQ Analysis for IC Grating Defect Detection
Analyzes Critical-to-Quality (CTQ) metrics across 8 operators (4 operators × 2 trials each)

This is a validation test of the measurement system's ability to detect defective IC gratings.
Expected results:
- Sample04 and Sample20: PASS (acceptable gratings)
- All other samples: FAIL (intentionally defective gratings)

CTQs analyzed:
1. RMSE Per Pixel P95 (lower is better)
2. SAM Mean (lower is better)
3. Uniformity Score (higher is better)

Author: Khang Tran
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import re

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Define operators (4 operators × 2 trials each = 8 total)
OPERATORS = [
    "KhangT1", "KhangT2",      # Khang Trial 1, Trial 2
    "KiduT1", "KiduT2",        # Kidu Trial 1, Trial 2
    "LuelT1", "LuelT2",        # Luel Trial 1, Trial 2
    "AnirbanT1", "AnirbanT2"   # Anirban Trial 1, Trial 2
]

# Map operators to person and trial
OPERATOR_INFO = {
    "KhangT1": {"person": "Khang", "trial": 1},
    "KhangT2": {"person": "Khang", "trial": 2},
    "KiduT1": {"person": "Kidu", "trial": 1},
    "KiduT2": {"person": "Kidu", "trial": 2},
    "LuelT1": {"person": "Luel", "trial": 1},
    "LuelT2": {"person": "Luel", "trial": 2},
    "AnirbanT1": {"person": "Anirban", "trial": 1},
    "AnirbanT2": {"person": "Anirban", "trial": 2}
}

# Quality thresholds (from config.json)
# Main CTQ thresholds for combined analysis
MAIN_THRESHOLDS = {
    "RMSE Per Pixel P95": 0.08,      # max threshold (lower is better)
    "SAM Mean": 3.2,                 # max threshold (lower is better)
    "Uniformity Score": 0.4          # min threshold (higher is better)
}

# Comprehensive CTQ thresholds for all metrics
ALL_THRESHOLDS = {
    "RMSE_Overall": 0.08,            # max threshold (lower is better)
    "RMSE_Per_Pixel_Mean": 0.08,     # max threshold (lower is better)
    "RMSE_Per_Pixel_Median": 0.08,   # max threshold (lower is better)
    "RMSE_Per_Pixel_P95": 0.08,      # max threshold (lower is better)
    "SAM_Mean": 3.2,                 # max threshold (lower is better)
    "SAM_Median": 4,               # max threshold (lower is better)
    "SAM_P95": 4,                  # max threshold (lower is better)
    "Uniformity_Score": 0.4          # min threshold (higher is better)
}

# Define expected results based on test design
EXPECTED_GOOD_SAMPLES = [4, 20]  # Only these should pass
EXPECTED_DEFECTIVE_SAMPLES = [1, 2, 6, 10, 12, 13, 14, 15, 17, 22]  # These should fail

def extract_numeric_sam(sam_str):
    """Extract numeric value from SAM string (e.g., '1.72°' -> 1.72)"""
    if isinstance(sam_str, str):
        match = re.search(r'(\d+\.?\d*)', sam_str)
        return float(match.group(1)) if match else np.nan
    return sam_str

def load_operator_data(operator):
    """Load batch_analysis_summary.csv for a specific operator"""
    csv_path = f"results/{operator}/scatter_plots/batch_analysis_summary.csv"

    if not os.path.exists(csv_path):
        print(f"⚠️  Warning: {csv_path} not found")
        return None

    try:
        df = pd.read_csv(csv_path)

        # Add operator info
        df['Operator'] = operator
        df['Person'] = OPERATOR_INFO[operator]["person"]
        df['Trial'] = OPERATOR_INFO[operator]["trial"]

        print(f"✅ Loaded {len(df)} samples from {operator}")
        return df

    except Exception as e:
        print(f"❌ Error loading {csv_path}: {str(e)}")
        return None

def calculate_gage_rr_statistics(combined_df):
    """Calculate measurement system statistics appropriate for defect detection validation"""
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
            print(f"⚠️  Warning: {actual_column} not found in data")
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
            noise = max(avg_repeatability_std, avg_reproducibility_std, 0.001)  # Avoid division by zero

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
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
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
    plt.savefig('analysis_results/ctq_boxplots_by_operator.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Sample-level analysis across operators - Create plots for ALL CTQs
    for column_name, display_name, threshold_type, threshold in all_ctq_metrics:
        if column_name not in combined_df.columns:
            print(f"⚠️  Skipping {display_name} - column {column_name} not found in data")
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
                color = plt.cm.tab10(j % 10)
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
        plt.savefig(f'analysis_results/{ctq_filename}_by_samples.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ Generated sample performance plot for {display_name}")

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
        bars = ax.bar(x_pos, pass_rate_df[f'{ctq}_PassRate'], alpha=0.7, color=plt.cm.Set3(i))

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
                    status = "✅" if pass_rate >= 90 else "⚠️" if pass_rate >= 70 else "❌"
                    report_lines.append(f"  {ctq}: {pass_rate:.1f}% {status}")

            if 'Overall_PassRate' in pass_rate_df.columns:
                overall_rate = row['Overall_PassRate']
                status = "✅" if overall_rate >= 90 else "⚠️" if overall_rate >= 70 else "❌"
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
        failing_samples = []
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

def main():
    """Main analysis function"""
    print("🔬 IC Grating Defect Detection Validation - Gage R&R Analysis")
    print("=" * 60)
    print("Expected Results:")
    print("✅ Sample04 & Sample20: PASS (Good IC gratings)")
    print("❌ All other samples: FAIL (Defective IC gratings)")
    print("=" * 60)

    # Load data from all operators
    all_data = []

    for operator in OPERATORS:
        df = load_operator_data(operator)
        if df is not None:
            all_data.append(df)

    if not all_data:
        print("❌ No data loaded. Please check that results exist.")
        return

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ Combined dataset: {len(combined_df)} total samples from {len(all_data)} operators")

    # Perform pass/fail analysis
    print("\n🎯 Performing pass/fail analysis...")
    combined_df = analyze_pass_fail_rates(combined_df)

    # Analyze defect detection performance
    print("\n🔍 Analyzing defect detection performance...")
    detection_df, detection_metrics = analyze_defect_detection_performance(combined_df)

    # Calculate Gage R&R statistics
    print("\n📊 Calculating Gage R&R statistics...")
    gage_rr_stats = calculate_gage_rr_statistics(combined_df)

    # Create summary plots
    print("\n📈 Creating CTQ analysis plots...")
    create_ctq_summary_plots(combined_df, gage_rr_stats)

    # Generate comprehensive report
    print("\n📋 Generating comprehensive analysis report...")
    # Create pass rate dataframes for reporting without creating visualizations
    pass_rate_data = []
    for operator in OPERATORS:
        if operator not in combined_df['Operator'].unique():
            continue

        op_data = combined_df[combined_df['Operator'] == operator]
        row = {'Operator': operator, 'Person': OPERATOR_INFO[operator]['person'],
               'Trial': OPERATOR_INFO[operator]['trial']}

        ctq_metrics = ["RMSE Per Pixel P95", "SAM Mean", "Uniformity Score"]
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

    # Create person trial data for reporting
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

    # Save datasets as requested
    print("\n💾 Saving analysis datasets...")
    os.makedirs("analysis_results", exist_ok=True)

    # 1. Main 3 CTQs dataset (no change from original)
    main_ctq_columns = ['Sample', 'Operator', 'Person', 'Trial', 'RMSE_Per_Pixel_P95', 'SAM_Mean', 'Uniformity_Score']
    main_ctq_columns += [col for col in combined_df.columns if '_Pass' in col and any(ctq.replace(' ', '_').replace('Per_Pixel_P95', 'Per_Pixel_P95') in col for ctq in ["RMSE Per Pixel P95", "SAM Mean", "Uniformity Score"])]
    main_ctq_columns += ['Overall_Pass'] if 'Overall_Pass' in combined_df.columns else []

    # Filter to include only columns that actually exist
    main_ctq_columns = [col for col in main_ctq_columns if col in combined_df.columns]
    combined_main_ctq_df = combined_df[main_ctq_columns].copy()

    # Rename columns to match original format for backward compatibility
    rename_mapping = {
        'RMSE_Per_Pixel_P95': 'RMSE Per Pixel P95',
        'SAM_Mean': 'SAM Mean',
        'Uniformity_Score': 'Uniformity Score'
    }
    combined_main_ctq_df = combined_main_ctq_df.rename(columns=rename_mapping)
    combined_main_ctq_df.to_csv("analysis_results/main_ctq_data.csv", index=False)

    # 2. All CTQs dataset with individual thresholds (no Overall_Pass)
    all_ctq_columns = ['Sample', 'Operator', 'Person', 'Trial']
    all_ctq_columns += list(ALL_THRESHOLDS.keys())  # Add all CTQ metrics
    all_ctq_columns += [f"{ctq}_Pass" for ctq in ALL_THRESHOLDS.keys()]  # Add pass/fail for each CTQ

    # Filter to include only columns that actually exist
    all_ctq_columns = [col for col in all_ctq_columns if col in combined_df.columns]
    combined_all_ctq_df = combined_df[all_ctq_columns].copy()
    combined_all_ctq_df.to_csv("analysis_results/all_ctq_data_with_thresholds.csv", index=False)

    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved in: analysis_results/")
    print(f"   - gage_rr_defect_detection_report.txt (comprehensive report)")
    print(f"   - ctq_boxplots_by_operator.png")
    print(f"   - rmse_per_pixel_p95_by_samples.png")
    print(f"   - sam_mean_by_samples.png")
    print(f"   - uniformity_score_by_samples.png")
    print(f"   - defect_detection_performance.png")
    print(f"   - main_ctq_data.csv (main 3 CTQs with Overall_Pass)")
    print(f"   - all_ctq_data_with_thresholds.csv (all CTQs with individual thresholds, no Overall_Pass)")
    print(f"\n📊 Dataset summary:")
    print(f"   Main CTQ dataset: {len(combined_main_ctq_df)} rows, {len(combined_main_ctq_df.columns)} columns")
    print(f"   All CTQ dataset: {len(combined_all_ctq_df)} rows, {len(combined_all_ctq_df.columns)} columns")

if __name__ == "__main__":
    main()
