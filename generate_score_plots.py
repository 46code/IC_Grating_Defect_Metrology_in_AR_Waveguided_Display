"""
Generate Hyperspectral Analysis Score Plots
Reads config.json to determine which samples to process and creates scatter plots for:
- RMSE Overall
- SAM Mean
- Uniformity Score

Author: Khang Tran
Date: November 2025
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_config():
    """Load configuration from config.json"""
    config_path = "config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Configuration loaded from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        print("   Please create config.json with required parameters.")
        return None

def extract_dataset_identifier(config):
    """Extract dataset identifier from sample_path (e.g., 'KhangT1' from '${base_path}/KhangT1')"""
    try:
        sample_path = config['data_paths']['sample_path']
        
        # Handle variable substitution
        if '${base_path}' in sample_path:
            base_path = config['data_paths']['base_path']
            full_path = sample_path.replace('${base_path}', base_path)
        else:
            full_path = sample_path
            
        # Extract the last directory name
        # Remove trailing slashes and split by path separator
        clean_path = full_path.rstrip('/')
        path_parts = clean_path.split('/')
        
        # Get the last non-empty part
        for part in reversed(path_parts):
            if part.strip():
                return part
                
        return None
    except (KeyError, IndexError):
        return None

def collect_analysis_data(samples, results_dir_prefix):
    """Collect analysis data from processed samples"""
    batch_data = []

    print(f"📊 Collecting data from {len(samples)} samples...")
    
    # Auto-detect naming convention (Defect vs Sample) - case insensitive and typo tolerant
    sample_type = "Sample"
    for sample in samples:
        sample_lower = sample.lower()
        # Check for defect patterns first
        if any(pattern in sample_lower for pattern in ['defect', 'def']):
            sample_type = "Defect"
            break
        # Check for sample patterns (including typos like 'smple')
        elif any(pattern in sample_lower for pattern in ['sample', 'smple', 'samp']):
            sample_type = "Sample"

    print(f"   Detected naming convention: {sample_type}")

    for sample_name in samples:
        results_dir = os.path.join(results_dir_prefix, f"analysis_{sample_name}")
        summary_file = os.path.join(results_dir, 'analysis_summary.csv')

        if os.path.exists(summary_file):
            try:
                summary_df = pd.read_csv(summary_file)

                # Extract RMSE metrics
                rmse_overall = float(summary_df[summary_df['Metric'] == 'RMSE Overall']['Value'].iloc[0])
                rmse_per_pixel_mean = float(summary_df[summary_df['Metric'] == 'RMSE Per-Pixel Mean']['Value'].iloc[0])
                rmse_per_pixel_median = float(summary_df[summary_df['Metric'] == 'RMSE Per-Pixel Median']['Value'].iloc[0])
                rmse_per_pixel_p95 = float(summary_df[summary_df['Metric'] == 'RMSE Per-Pixel P95']['Value'].iloc[0])

                # Extract SAM metrics (already in degrees from main.py)
                sam_mean_str = summary_df[summary_df['Metric'] == 'SAM Mean']['Value'].iloc[0]
                sam_mean = float(sam_mean_str.split('°')[0])  # Extract numeric value before '°'
                sam_median_str = summary_df[summary_df['Metric'] == 'SAM Median']['Value'].iloc[0]
                sam_median = float(sam_median_str.split('°')[0])
                sam_p95_str = summary_df[summary_df['Metric'] == 'SAM P95']['Value'].iloc[0]
                sam_p95 = float(sam_p95_str.split('°')[0])

                # Extract other metrics
                uniformity_score = float(summary_df[summary_df['Metric'] == 'Uniformity Score']['Value'].iloc[0])

                # Robust sample number extraction - handles all variations and typos
                import re
                sample_num = extract_sample_number(sample_name, sample_type)

                batch_data.append({
                    'Sample': sample_name,
                    'Sample_Num': sample_num,
                    'Sample_Type': sample_type,
                    'RMSE_Overall': rmse_overall,
                    'RMSE_Per_Pixel_Mean': rmse_per_pixel_mean,
                    'RMSE_Per_Pixel_Median': rmse_per_pixel_median,
                    'RMSE_Per_Pixel_P95': rmse_per_pixel_p95,
                    'SAM_Mean': sam_mean,
                    'SAM_Median': sam_median,
                    'SAM_P95': sam_p95,
                    'Uniformity_Score': uniformity_score
                })

                print(f"✅ Loaded {sample_name}: RMSE_Overall={rmse_overall:.6f}, SAM_P95={sam_p95:.2f}°, Uniformity={uniformity_score:.3f}")

            except Exception as e:
                print(f"⚠️  Error reading {sample_name}: {str(e)}")
        else:
            print(f"⚠️  Analysis summary not found for {sample_name}: {summary_file}")

    return batch_data

def extract_sample_number(sample_name, sample_type):
    """
    Extract sample number from sample name, handling all variations and typos
    Examples: sample01, Sample01, smple14, defect1, Defect10, etc.
    """
    import re

    sample_lower = sample_name.lower()

    if sample_type == "Sample":
        # Handle various sample patterns (including typos)
        sample_patterns = [
            r'sample(\d+)',     # sample01, sample1
            r'smple(\d+)',      # smple14 (typo)
            r'samp(\d+)',       # samp01 (abbreviation)
            r'sam(\d+)',        # sam01 (short form)
            r's(\d+)',          # s01 (very short)
        ]

        for pattern in sample_patterns:
            match = re.search(pattern, sample_lower)
            if match:
                return int(match.group(1))

    else:  # Defect type
        # Handle various defect patterns
        defect_patterns = [
            r'defect(\d+)',     # defect1, defect10
            r'def(\d+)',        # def1 (abbreviation)
            r'd(\d+)',          # d1 (very short)
        ]

        for pattern in defect_patterns:
            match = re.search(pattern, sample_lower)
            if match:
                return int(match.group(1))

    # Fallback: extract any number from the string
    numbers = re.findall(r'\d+', sample_name)
    if numbers:
        return int(numbers[0])

    # Ultimate fallback: use hash of the name for consistent ordering
    return abs(hash(sample_name)) % 10000

def generate_scatter_plots(batch_data, output_dir, config):
    """Generate 3 scatter plots: RMSE, SAM_Mean, and Uniformity Score for all defects"""
    if len(batch_data) < 1:
        print("⚠️  No data available to generate scatter plots")
        return

    print(f"\n📊 GENERATING SCATTER PLOTS FOR {len(batch_data)} DEFECTS")
    print("="*80)

    # Convert to DataFrame
    df_batch = pd.DataFrame(batch_data)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get thresholds from config
    thresholds = config['quality_thresholds']

    # Plot configurations for the 3 requested plots with thresholds from config
    plot_configs = [
        {
            'metric': 'RMSE_Overall',
            'title': 'RMSE Overall Analysis Across All Defects',
            'ylabel': 'RMSE Overall',
            'color': '#e74c3c',
            'filename': 'RMSE_Overall_Scatter',
            'threshold': thresholds['rmse_overall_max'],
            'threshold_type': 'below'  # Pass if value is below threshold
        },
        {
            'metric': 'SAM_P95',
            'title': 'SAM P95 Analysis Across All Defects',
            'ylabel': 'SAM P95 (degrees)',
            'color': '#3498db',
            'filename': 'SAM_P95_Scatter',
            'threshold': thresholds['sam_p95_max'],
            'threshold_type': 'below'  # Pass if value is below threshold
        },
        {
            'metric': 'Uniformity_Score',
            'title': 'Uniformity Score Analysis Across All Defects',
            'ylabel': 'Uniformity Score',
            'color': '#2ecc71',
            'filename': 'Uniformity_Score_Scatter',
            'threshold': thresholds['uniformity_score_min'],
            'threshold_type': 'above'  # Pass if value is above threshold
        }
    ]

    # Generate individual scatter plots
    for config in plot_configs:
        plt.figure(figsize=(12, 8))

        metric = config['metric']
        values = df_batch[metric].values
        sample_nums = df_batch['Sample_Num'].values
        threshold = config['threshold']
        threshold_type = config['threshold_type']

        # Calculate statistics
        min_val = values.min()
        max_val = values.max()
        mean_val = values.mean()
        std_val = values.std()

        # Determine which samples pass/fail the threshold
        if threshold_type == 'below':
            pass_threshold = values < threshold
            threshold_label = f'Threshold < {threshold}'
        else:  # above
            pass_threshold = values > threshold
            threshold_label = f'Threshold > {threshold}'

        # Scatter plot
        plt.scatter(sample_nums, values, s=200, c=config['color'], alpha=0.7,
                   edgecolors='black', linewidth=2, zorder=3)

        # Add X markers for samples that don't pass threshold
        failed_samples = ~pass_threshold
        if np.any(failed_samples):
            failed_x = sample_nums[failed_samples]
            failed_y = values[failed_samples]
            plt.scatter(failed_x, failed_y, s=300, marker='x', c='red', linewidth=4, zorder=4,
                       label=f'Failed ({np.sum(failed_samples)} samples)')

        # Add sample labels
        for x, y, sample in zip(sample_nums, values, df_batch['Sample']):
            plt.annotate(sample, (x, y), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')

        # Add statistical lines
        plt.axhline(y=min_val, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Min: {min_val:.4f}')
        plt.axhline(y=max_val, color='green', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Max: {max_val:.4f}')
        plt.axhline(y=mean_val, color='blue', linestyle=':', linewidth=2,
                   alpha=0.6, label=f'Mean: {mean_val:.4f}')

        # Add threshold line
        plt.axhline(y=threshold, color='orange', linestyle='-', linewidth=3,
                   alpha=0.8, label=threshold_label)

        # Formatting
        plt.xlabel('Defect Number', fontsize=12, fontweight='bold')
        plt.ylabel(config['ylabel'], fontsize=12, fontweight='bold')
        plt.title(config['title'], fontsize=14, fontweight='bold', pad=20)
        plt.xlim(0, max(sample_nums) + 1)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='best', fontsize=10, framealpha=0.9)

        # Add statistics box with pass/fail info
        pass_count = np.sum(pass_threshold)
        fail_count = np.sum(failed_samples)
        stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nRange: {max_val-min_val:.4f}\n✅ Passed: {pass_count}\n❌ Failed: {fail_count}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=10, va='top', ha='left', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))

        # Save plot
        plt.tight_layout()
        png_path = os.path.join(output_dir, f"{config['filename']}.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Generated: {config['filename']} (✅ {pass_count} passed, ❌ {fail_count} failed)")

    # Generate combined plot with all 3 metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Hyperspectral Analysis Summary - All Defects', fontsize=16, fontweight='bold')

    for i, config in enumerate(plot_configs):
        ax = axes[i]
        metric = config['metric']
        values = df_batch[metric].values
        sample_nums = df_batch['Sample_Num'].values
        threshold = config['threshold']
        threshold_type = config['threshold_type']

        # Determine which samples pass/fail the threshold
        if threshold_type == 'below':
            pass_threshold = values < threshold
        else:  # above
            pass_threshold = values > threshold

        # Scatter plot
        ax.scatter(sample_nums, values, s=120, c=config['color'], alpha=0.7,
                  edgecolors='black', linewidth=1.5, zorder=3)

        # Add X markers for samples that don't pass threshold
        failed_samples = ~pass_threshold
        if np.any(failed_samples):
            failed_x = sample_nums[failed_samples]
            failed_y = values[failed_samples]
            ax.scatter(failed_x, failed_y, s=150, marker='x', c='red', linewidth=3, zorder=4)

        # Add sample labels (smaller for combined plot)
        for x, y, sample in zip(sample_nums, values, df_batch['Sample']):
            ax.annotate(sample.replace('Sample', 'S'), (x, y), textcoords="offset points",
                       xytext=(0, 8), ha='center', fontsize=8, fontweight='bold')

        # Statistical lines
        mean_val = values.mean()
        ax.axhline(y=mean_val, color='blue', linestyle=':', linewidth=1.5,
                  alpha=0.6, label=f'Mean: {mean_val:.4f}')

        # Add threshold line
        ax.axhline(y=threshold, color='orange', linestyle='-', linewidth=2,
                  alpha=0.8, label=f'Threshold: {threshold}')

        # Formatting
        ax.set_xlabel('Defect Number', fontsize=10, fontweight='bold')
        ax.set_ylabel(config['ylabel'], fontsize=10, fontweight='bold')
        ax.set_title(config['title'].replace(' Analysis Across All Defects', ''),
                    fontsize=11, fontweight='bold')
        ax.set_xlim(0, max(sample_nums) + 1)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    combined_png = os.path.join(output_dir, "Combined_Analysis_Scatter.png")
    plt.savefig(combined_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Generated: Combined_Analysis_Scatter")

    # Save batch analysis summary
    batch_summary_df = pd.DataFrame(batch_data)
    batch_summary_path = os.path.join(output_dir, "batch_analysis_summary.csv")
    batch_summary_df.to_csv(batch_summary_path, index=False)

    # Analyze samples that pass ALL thresholds
    print(f"\n{'='*80}")
    print("🎯 THRESHOLD ANALYSIS SUMMARY")
    print(f"{'='*80}")

    # Use thresholds from config
    threshold_config = {
        'RMSE_Overall': {'value': thresholds['rmse_overall_max'], 'type': 'below'},
        'SAM_P95': {'value': thresholds['sam_p95_max'], 'type': 'below'},
        'Uniformity_Score': {'value': thresholds['uniformity_score_min'], 'type': 'above'}
    }

    # Check which samples pass each threshold
    all_pass_samples = []

    for _, row in df_batch.iterrows():
        sample_name = row['Sample']

        # Check RMSE threshold (pass if below threshold)
        rmse_pass = row['RMSE_Overall'] < threshold_config['RMSE_Overall']['value']

        # Check SAM threshold (pass if below threshold)
        sam_pass = row['SAM_P95'] < threshold_config['SAM_P95']['value']

        # Check Uniformity threshold (pass if above threshold)
        uniformity_pass = row['Uniformity_Score'] > threshold_config['Uniformity_Score']['value']

        # Sample passes all thresholds if all three conditions are met
        if rmse_pass and sam_pass and uniformity_pass:
            all_pass_samples.append({
                'Sample': sample_name,
                'RMSE': row['RMSE_Overall'],
                'SAM_P95': row['SAM_P95'],
                'Uniformity': row['Uniformity_Score']
            })

    # Print results
    print(f"📊 THRESHOLD CRITERIA:")
    print(f"   • RMSE Overall < {threshold_config['RMSE_Overall']['value']}")
    print(f"   • SAM P95 < {threshold_config['SAM_P95']['value']} rad")
    print(f"   • Uniformity Score > {threshold_config['Uniformity_Score']['value']}")

    if all_pass_samples:
        print(f"\n✅ SAMPLES PASSING ALL THRESHOLDS ({len(all_pass_samples)}/{len(df_batch)} samples):")
        print("-" * 80)
        print(f"{'Sample':<10} {'RMSE':<12} {'SAM P95 (rad)':<15} {'Uniformity':<12}")
        print("-" * 80)
        for sample in all_pass_samples:
            print(f"{sample['Sample']:<10} {sample['RMSE']:<12.6f} {sample['SAM_P95']:<15.6f} {sample['Uniformity']:<12.3f}")

        # Save list of high-quality samples
        all_pass_df = pd.DataFrame(all_pass_samples)
        high_quality_path = os.path.join(output_dir, "high_quality_samples.csv")
        all_pass_df.to_csv(high_quality_path, index=False)
        print(f"\n💾 High-quality samples list saved to: {high_quality_path}")
    else:
        print(f"\n❌ NO SAMPLES PASS ALL THRESHOLDS")
        print("   All samples fail at least one quality criterion.")

    print(f"\n📊 SCATTER PLOT SUMMARY:")
    print(f"   Generated 4 plots saved to: {output_dir}")
    print(f"   - RMSE_Overall_Scatter: RMSE analysis across all defects")
    print(f"   - SAM_P95_Scatter: SAM P95 analysis across all defects")
    print(f"   - Uniformity_Score_Scatter: Uniformity analysis across all defects")
    print(f"   - Combined_Analysis_Scatter: All three metrics in one view")
    print(f"   Batch data summary saved to: {batch_summary_path}")

    return output_dir

def get_available_metrics():
    """Get list of available metrics for plotting"""
    return {
        'RMSE_Overall': 'RMSE Overall',
        'RMSE_Per_Pixel_Mean': 'RMSE Per-Pixel Mean',
        'RMSE_Per_Pixel_Median': 'RMSE Per-Pixel Median',
        'RMSE_Per_Pixel_P95': 'RMSE Per-Pixel P95',
        'SAM_Mean': 'SAM Mean (degrees)',
        'SAM_Median': 'SAM Median (degrees)',
        'SAM_P95': 'SAM P95 (degrees)',
        'Uniformity_Score': 'Uniformity Score'
    }

def get_selected_metrics_from_config(config):
    """Get selected metrics from config with their display properties and thresholds"""
    selected_configs = []

    plotting_config = config['plotting_configuration']
    selected_metrics = plotting_config.get('selected_metrics', {})
    quality_thresholds = config['quality_thresholds']

    # Default colors for different CTQ types
    colors = {
        'CTQ1_RMSE': '#e74c3c',     # Red for RMSE
        'CTQ2_SAM': '#3498db',      # Blue for SAM
        'CTQ3_Uniformity': '#2ecc71' # Green for Uniformity
    }

    # Metric display names and units
    metric_info = {
        'RMSE_Overall': {'name': 'RMSE Overall', 'unit': ''},
        'RMSE_Per_Pixel_Mean': {'name': 'RMSE Per-Pixel Mean', 'unit': ''},
        'RMSE_Per_Pixel_Median': {'name': 'RMSE Per-Pixel Median', 'unit': ''},
        'RMSE_Per_Pixel_P95': {'name': 'RMSE Per-Pixel P95', 'unit': ''},
        'SAM_Mean': {'name': 'SAM Mean', 'unit': ' (degrees)'},
        'SAM_Median': {'name': 'SAM Median', 'unit': ' (degrees)'},
        'SAM_P95': {'name': 'SAM P95', 'unit': ' (degrees)'},
        'Uniformity_Score': {'name': 'Uniformity Score', 'unit': ''}
    }

    # Process each selected metric
    for ctq_type, metric_key in selected_metrics.items():
        if metric_key not in metric_info:
            print(f"⚠️  Unknown metric '{metric_key}' for {ctq_type}. Skipping.")
            continue

        # Get threshold key mapping
        threshold_key = get_threshold_key_for_metric(metric_key)

        if not threshold_key or threshold_key not in quality_thresholds:
            print(f"⚠️  No threshold found for '{metric_key}' (expected: {threshold_key}). Skipping.")
            continue

        # Get metric info
        info = metric_info[metric_key]
        display_name = info['name'] + info['unit']

        # Determine threshold type
        threshold_type = 'above' if 'uniformity' in threshold_key.lower() else 'below'

        selected_configs.append({
            'ctq_type': ctq_type,
            'metric': metric_key,
            'title': f'{info["name"]} Analysis Across All Defects',
            'ylabel': display_name,
            'color': colors.get(ctq_type, '#95a5a6'),  # Default gray if CTQ type not found
            'filename': f'{ctq_type}_{metric_key}_Scatter',
            'threshold': quality_thresholds[threshold_key],
            'threshold_type': threshold_type
        })

    return selected_configs

def get_threshold_key_for_metric(metric_key):
    """Map metric keys to their corresponding threshold keys in config"""
    mapping = {
        'RMSE_Overall': 'rmse_overall_max',
        'RMSE_Per_Pixel_Mean': 'rmse_per_pixel_mean_max',
        'RMSE_Per_Pixel_Median': 'rmse_per_pixel_median_max',
        'RMSE_Per_Pixel_P95': 'rmse_per_pixel_p95_max',
        'SAM_Mean': 'sam_mean_max',
        'SAM_Median': 'sam_median_max',
        'SAM_P95': 'sam_p95_max',
        'Uniformity_Score': 'uniformity_score_min'
    }
    return mapping.get(metric_key)

def generate_flexible_scatter_plots(batch_data, output_dir, config, selected_configs):
    """Generate scatter plots for selected metrics with flexible configuration"""
    if len(batch_data) < 1:
        print("⚠️  No data available to generate scatter plots")
        return

    # Auto-detect sample type from data
    sample_type = batch_data[0]['Sample_Type']
    sample_type_plural = f"{sample_type}s"
    sample_short = sample_type[0]  # 'S' for Sample, 'D' for Defect

    # Extract dataset identifier from config
    dataset_identifier = extract_dataset_identifier(config)
    dataset_suffix = f" - {dataset_identifier}" if dataset_identifier else ""

    print(f"\n📊 GENERATING SCATTER PLOTS FOR {len(batch_data)} {sample_type_plural.upper()}")
    if dataset_identifier:
        print(f"   Dataset: {dataset_identifier}")
    print("="*80)

    # Convert to DataFrame
    df_batch = pd.DataFrame(batch_data)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate individual scatter plots for each selected metric
    for plot_config in selected_configs:
        plt.figure(figsize=(12, 8))

        metric = plot_config['metric']
        values = df_batch[metric].values
        sample_nums = df_batch['Sample_Num'].values
        threshold = plot_config['threshold']
        threshold_type = plot_config['threshold_type']

        # Calculate statistics
        min_val = values.min()
        max_val = values.max()
        mean_val = values.mean()
        std_val = values.std()

        # Determine which samples pass/fail the threshold
        if threshold_type == 'below':
            pass_threshold = values <= threshold
            threshold_label = f'Threshold <= {threshold}'
        else:  # above
            pass_threshold = values >= threshold
            threshold_label = f'Threshold >= {threshold}'

        # Scatter plot
        plt.scatter(sample_nums, values, s=200, c=plot_config['color'], alpha=0.7,
                   edgecolors='black', linewidth=2, zorder=3)

        # Add X markers for samples that don't pass threshold
        failed_samples = ~pass_threshold
        if np.any(failed_samples):
            failed_x = sample_nums[failed_samples]
            failed_y = values[failed_samples]
            plt.scatter(failed_x, failed_y, s=300, marker='x', c='red', linewidth=4, zorder=4,
                       label=f'Failed ({np.sum(failed_samples)} {sample_type_plural.lower()})')

        # Add sample labels
        for x, y, sample in zip(sample_nums, values, df_batch['Sample']):
            plt.annotate(sample, (x, y), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')

        # Add statistical lines
        plt.axhline(y=min_val, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Min: {min_val:.4f}')
        plt.axhline(y=max_val, color='green', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Max: {max_val:.4f}')
        plt.axhline(y=mean_val, color='blue', linestyle=':', linewidth=2,
                   alpha=0.6, label=f'Mean: {mean_val:.4f}')

        # Add threshold line
        plt.axhline(y=threshold, color='orange', linestyle='-', linewidth=3,
                   alpha=0.8, label=threshold_label)

        # Dynamic formatting based on sample type
        plt.xlabel(f'{sample_type} Number', fontsize=12, fontweight='bold')
        plt.ylabel(plot_config['ylabel'], fontsize=12, fontweight='bold')

        # Update title to use dynamic sample type and dataset identifier
        dynamic_title = plot_config['title'].replace('Defects', sample_type_plural).replace('Defect', sample_type)
        dynamic_title += dataset_suffix
        plt.title(dynamic_title, fontsize=14, fontweight='bold', pad=20)

        plt.xlim(0, max(sample_nums) + 1)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='best', fontsize=10, framealpha=0.9)

        # Add statistics box with pass/fail info
        pass_count = np.sum(pass_threshold)
        fail_count = np.sum(failed_samples)
        stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nRange: {max_val-min_val:.4f}\n✅ Passed: {pass_count}\n❌ Failed: {fail_count}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=10, va='top', ha='left', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))

        # Save plot
        plt.tight_layout()
        png_path = os.path.join(output_dir, f"{plot_config['filename']}.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Generated: {plot_config['filename']} (✅ {pass_count} passed, ❌ {fail_count} failed)")

    # Generate combined plot if we have 3 or fewer metrics
    if len(selected_configs) <= 3:
        fig, axes = plt.subplots(1, len(selected_configs), figsize=(6*len(selected_configs), 6))
        if len(selected_configs) == 1:
            axes = [axes]

        # Dynamic title for combined plot with dataset identifier
        combined_title = f'Hyperspectral Analysis Summary - All {sample_type_plural}{dataset_suffix}'
        fig.suptitle(combined_title, fontsize=16, fontweight='bold')

        for i, plot_config in enumerate(selected_configs):
            ax = axes[i]
            metric = plot_config['metric']
            values = df_batch[metric].values
            sample_nums = df_batch['Sample_Num'].values
            threshold = plot_config['threshold']
            threshold_type = plot_config['threshold_type']

            # Determine which samples pass/fail the threshold
            if threshold_type == 'below':
                pass_threshold = values <= threshold
            else:  # above
                pass_threshold = values >= threshold

            # Scatter plot
            ax.scatter(sample_nums, values, s=120, c=plot_config['color'], alpha=0.7,
                      edgecolors='black', linewidth=1.5, zorder=3)

            # Add X markers for samples that don't pass threshold
            failed_samples = ~pass_threshold
            if np.any(failed_samples):
                failed_x = sample_nums[failed_samples]
                failed_y = values[failed_samples]
                ax.scatter(failed_x, failed_y, s=150, marker='x', c='red', linewidth=3, zorder=4)

            # Add sample labels (shorter for combined plot)
            for x, y, sample in zip(sample_nums, values, df_batch['Sample']):
                ax.annotate(sample.replace(sample_type, sample_short), (x, y), textcoords="offset points",
                           xytext=(0, 8), ha='center', fontsize=8, fontweight='bold')

            # Statistical lines
            mean_val = values.mean()
            ax.axhline(y=mean_val, color='blue', linestyle=':', linewidth=1.5,
                      alpha=0.6, label=f'Mean: {mean_val:.4f}')

            # Add threshold line
            ax.axhline(y=threshold, color='orange', linestyle='-', linewidth=2,
                      alpha=0.8, label=f'Threshold: {threshold}')

            # Dynamic formatting
            ax.set_xlabel(f'{sample_type} Number', fontsize=10, fontweight='bold')
            ax.set_ylabel(plot_config['ylabel'], fontsize=10, fontweight='bold')

            # Update subplot title
            subplot_title = plot_config['title'].replace(f' Analysis Across All Defects', '').replace('Defects', sample_type_plural).replace('Defect', sample_type)
            ax.set_title(subplot_title, fontsize=11, fontweight='bold')

            ax.set_xlim(0, max(sample_nums) + 1)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=8)

        plt.tight_layout()
        combined_png = os.path.join(output_dir, "Combined_Analysis_Scatter.png")
        plt.savefig(combined_png, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Generated: Combined_Analysis_Scatter")

    # Save batch analysis summary
    batch_summary_df = pd.DataFrame(batch_data)
    batch_summary_path = os.path.join(output_dir, "batch_analysis_summary.csv")
    batch_summary_df.to_csv(batch_summary_path, index=False)

    print(f"\n📊 SCATTER PLOT SUMMARY:")
    print(f"   Generated {len(selected_configs)} individual plots + 1 combined plot saved to: {output_dir}")
    for plot_config in selected_configs:
        dynamic_description = plot_config['ylabel'] + f" analysis across {sample_type_plural.lower()}"
        print(f"   - {plot_config['filename']}: {dynamic_description}")
    if len(selected_configs) <= 3:
        print(f"   - Combined_Analysis_Scatter: All metrics in one view")
    print(f"   Batch data summary saved to: {batch_summary_path}")

    return output_dir

def validate_plotting_configuration(config):
    """Validate that the plotting configuration is valid"""
    if 'plotting_configuration' not in config:
        return True, "No plotting_configuration found - will use defaults"

    plotting_config = config['plotting_configuration']
    selected_metrics = plotting_config.get('selected_metrics', {})
    available_options = plotting_config.get('available_options', {})
    quality_thresholds = config.get('quality_thresholds', {})

    errors = []
    warnings = []

    # Validate each selected metric
    for ctq_type, metric_key in selected_metrics.items():
        # Check if the CTQ type has available options
        if ctq_type not in available_options:
            warnings.append(f"No available options defined for {ctq_type}")
            continue

        # Check if the selected metric is in the available options
        if metric_key not in available_options[ctq_type]:
            errors.append(f"Selected metric '{metric_key}' for {ctq_type} is not in available options: {available_options[ctq_type]}")

        # Check if there's a corresponding threshold
        threshold_key = get_threshold_key_for_metric(metric_key)
        if not threshold_key or threshold_key not in quality_thresholds:
            errors.append(f"No threshold found for '{metric_key}' (expected: {threshold_key})")

    if errors:
        return False, f"Configuration errors: {'; '.join(errors)}"
    elif warnings:
        return True, f"Configuration warnings: {'; '.join(warnings)}"
    else:
        return True, "Configuration is valid"

def evaluate_ctq_criteria(batch_data, config):
    """
    Evaluate CTQ (Critical to Quality) pass/fail criteria for all samples

    Args:
        batch_data: List of dictionaries containing sample analysis results
        config: Configuration dictionary containing thresholds

    Returns:
        dict: CTQ evaluation results with pass/fail status for each sample
    """
    quality_thresholds = config['quality_thresholds']
    selected_metrics = config['plotting_configuration']['selected_metrics']

    ctq_results = {}

    for sample_data in batch_data:
        sample_name = sample_data['Sample']

        # Extract the selected metrics for CTQ evaluation
        ctq_evaluations = {}

        for ctq_type, metric_key in selected_metrics.items():
            if metric_key not in sample_data:
                continue

            threshold_key = get_threshold_key_for_metric(metric_key)
            if not threshold_key or threshold_key not in quality_thresholds:
                continue

            value = sample_data[metric_key]
            threshold = quality_thresholds[threshold_key]

            # Determine pass/fail based on metric type
            if 'uniformity' in threshold_key.lower():
                # Uniformity: pass if above threshold
                passes = value >= threshold
                comparison = f">= {threshold}"
            else:
                # RMSE/SAM: pass if below threshold
                passes = value <= threshold
                comparison = f"<= {threshold}"

            ctq_evaluations[ctq_type] = {
                'metric_key': metric_key,
                'value': value,
                'threshold': threshold,
                'comparison': comparison,
                'pass': passes
            }

        # Overall pass if all CTQs pass
        overall_pass = all(eval_data['pass'] for eval_data in ctq_evaluations.values())

        ctq_results[sample_name] = {
            'evaluations': ctq_evaluations,
            'overall_pass': overall_pass
        }
    return ctq_results

def generate_final_summary_table(batch_data, ctq_results, config, output_dir):
    """Generate and display final summary table with sample names as rows"""
    selected_metrics = config['plotting_configuration']['selected_metrics']

    print("🎯 FINAL SUMMARY TABLE")
    print(f"{'='*120}")

    # Create table data structure with samples as rows
    sample_names = [data['Sample'] for data in batch_data]

    # Create column headers: metrics + CTQ pass/fail + total pass
    metric_columns = []
    for ctq_type, metric_key in selected_metrics.items():
        metric_columns.append(metric_key.replace('_', ' '))

    ctq_columns = [f"{ctq_type}" for ctq_type in selected_metrics.keys()]

    columns = ['Sample'] + metric_columns + ctq_columns + ['Total_Pass']

    # Create rows - one per sample
    rows = []

    for sample_data in batch_data:
        sample_name = sample_data['Sample']
        row = [sample_name]

        # Add metric values
        for ctq_type, metric_key in selected_metrics.items():
            if metric_key in sample_data:
                value = sample_data[metric_key]
                if 'SAM' in metric_key:
                    # Format SAM values with degrees symbol
                    row.append(f"{value:.2f}°")
                elif 'Uniformity' in metric_key:
                    # Format uniformity as 3 decimal places
                    row.append(f"{value:.3f}")
                else:
                    # Format RMSE values with 4 decimal places
                    row.append(f"{value:.4f}")
            else:
                row.append("N/A")

        # Add CTQ pass/fail status
        for ctq_type in selected_metrics.keys():
            if sample_name in ctq_results and ctq_type in ctq_results[sample_name]['evaluations']:
                passes = ctq_results[sample_name]['evaluations'][ctq_type]['pass']
                row.append("✅" if passes else "❌")
            else:
                row.append("N/A")

        # Add total pass status
        if sample_name in ctq_results:
            overall_pass = ctq_results[sample_name]['overall_pass']
            row.append("Pass" if overall_pass else "Fail")
        else:
            row.append("N/A")

        rows.append(row)

    # Create and display DataFrame
    summary_df = pd.DataFrame(rows, columns=columns)

    print(summary_df.to_string(index=False))

    # Save the summary table to CSV
    summary_table_path = os.path.join(output_dir, "final_summary_table.csv")
    summary_df.to_csv(summary_table_path, index=False)

    # Calculate and display statistics
    total_samples = len(sample_names)
    passed_samples = sum(1 for sample_name in sample_names
                        if sample_name in ctq_results and ctq_results[sample_name]['overall_pass'])
    failed_samples = total_samples - passed_samples

    print(f"\n{'='*120}")
    print("📊 SUMMARY STATISTICS:")
    print(f"   Total Samples: {total_samples}")
    print(f"   ✅ Passed All CTQs: {passed_samples} ({passed_samples/total_samples*100:.1f}%)")
    print(f"   ❌ Failed At Least One CTQ: {failed_samples} ({failed_samples/total_samples*100:.1f}%)")

    # Show thresholds used
    print(f"\n📋 CTQ THRESHOLDS USED:")
    quality_thresholds = config['quality_thresholds']
    for ctq_type, metric_key in selected_metrics.items():
        threshold_key = get_threshold_key_for_metric(metric_key)
        if threshold_key and threshold_key in quality_thresholds:
            threshold = quality_thresholds[threshold_key]
            comparison = ">=" if 'uniformity' in threshold_key.lower() else "<="
            unit = "°" if 'SAM' in metric_key else ""
            print(f"   {ctq_type} ({metric_key}): {comparison} {threshold}{unit}")

    print(f"\n💾 Final summary table saved to: {summary_table_path}")
    print(f"{'='*120}")

    return summary_df

def main():
    """Main function to generate scatter plots and final summary"""
    print("📊 Hyperspectral Analysis Score Plot Generator")
    print("="*80)

    # Load configuration
    config = load_config()
    if not config:
        return

    # Validate configuration
    is_valid, message = validate_plotting_configuration(config)
    if not is_valid:
        print(f"❌ Configuration validation failed: {message}")
        return
    elif message != "Configuration is valid":
        print(f"⚠️  {message}")

    # Extract parameters from config
    samples = config['data_paths']['samples']
    results_dir_prefix = config['data_paths']['results_dir_prefix']

    # Extract dataset identifier for output directory
    dataset_identifier = extract_dataset_identifier(config)

    if dataset_identifier:
        output_dir = os.path.join(results_dir_prefix, "scatter_plots")
        print(f"📁 Dataset identified: {dataset_identifier}")
    else:
        output_dir = os.path.join(results_dir_prefix, "scatter_plots")
        print(f"📁 No dataset identifier found, using default output directory")

    print(f"📁 Output directory: {output_dir}")

    # Collect analysis data from processed samples
    batch_data = collect_analysis_data(samples, results_dir_prefix)

    if not batch_data:
        print("❌ No analysis data found. Please run the main analysis first.")
        return

    # Get selected metrics from config
    selected_configs = get_selected_metrics_from_config(config)

    if not selected_configs:
        print("❌ No valid metrics selected for plotting.")
        return

    print(f"📊 Selected {len(selected_configs)} metrics for plotting:")
    for plot_config in selected_configs:
        print(f"   - {plot_config['ctq_type']}: {plot_config['ylabel']}")

    # Generate scatter plots
    output_path = generate_flexible_scatter_plots(batch_data, output_dir, config, selected_configs)

    # Evaluate CTQ criteria for all samples
    ctq_results = evaluate_ctq_criteria(batch_data, config)

    # Generate and display final summary table
    summary_df = generate_final_summary_table(batch_data, ctq_results, config, output_dir)

    print(f"\n✅ Analysis complete! All outputs saved to: {output_path}")

if __name__ == "__main__":
    main()
