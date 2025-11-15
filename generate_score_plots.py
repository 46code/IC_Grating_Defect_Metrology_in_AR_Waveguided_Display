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

def collect_analysis_data(samples, results_dir_prefix):
    """Collect analysis data from processed samples"""
    batch_data = []

    print(f"📊 Collecting data from {len(samples)} samples...")

    for sample_name in samples:
        results_dir = results_dir_prefix + sample_name
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

                batch_data.append({
                    'Sample': sample_name,
                    'Sample_Num': int(sample_name.replace('Defect', '')),
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
            ax.annotate(sample.replace('Defect', 'D'), (x, y), textcoords="offset points",
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

def select_metrics_for_plotting(available_metrics, config_thresholds):
    """Select which metrics to plot based on available thresholds in config"""
    selected_configs = []

    # Default colors for different metric types
    colors = {
        'RMSE': ['#e74c3c', '#c0392b', '#a93226', '#922b21'],
        'SAM': ['#3498db', '#2980b9', '#1f618d', '#154360'],
        'Other': ['#2ecc71', '#27ae60', '#229954', '#1e8449']
    }

    rmse_color_idx = 0
    sam_color_idx = 0
    other_color_idx = 0

    for metric_key, metric_name in available_metrics.items():
        # Check if we have a threshold for this metric in config
        threshold_key = None
        if metric_key == 'RMSE_Overall':
            threshold_key = 'rmse_overall_max'
        elif metric_key == 'RMSE_Per_Pixel_Mean':
            threshold_key = 'rmse_per_pixel_mean_max'
        elif metric_key == 'RMSE_Per_Pixel_Median':
            threshold_key = 'rmse_per_pixel_median_max'
        elif metric_key == 'RMSE_Per_Pixel_P95':
            threshold_key = 'rmse_per_pixel_p95_max'
        elif metric_key == 'SAM_Mean':
            threshold_key = 'sam_mean_max'
        elif metric_key == 'SAM_Median':
            threshold_key = 'sam_median_max'
        elif metric_key == 'SAM_P95':
            threshold_key = 'sam_p95_max'
        elif metric_key == 'Uniformity_Score':
            threshold_key = 'uniformity_score_min'

        if threshold_key and threshold_key in config_thresholds:
            # Determine color and threshold type
            if 'RMSE' in metric_key:
                color = colors['RMSE'][rmse_color_idx % len(colors['RMSE'])]
                rmse_color_idx += 1
                threshold_type = 'below'
            elif 'SAM' in metric_key:
                color = colors['SAM'][sam_color_idx % len(colors['SAM'])]
                sam_color_idx += 1
                threshold_type = 'below'
            else:
                color = colors['Other'][other_color_idx % len(colors['Other'])]
                other_color_idx += 1
                threshold_type = 'above' if 'Uniformity' in metric_key else 'below'

            selected_configs.append({
                'metric': metric_key,
                'title': f'{metric_name} Analysis Across All Defects',
                'ylabel': metric_name,
                'color': color,
                'filename': f'{metric_key}_Scatter',
                'threshold': config_thresholds[threshold_key],
                'threshold_type': threshold_type
            })

    return selected_configs

def generate_flexible_scatter_plots(batch_data, output_dir, config, selected_configs):
    """Generate scatter plots for selected metrics with flexible configuration"""
    if len(batch_data) < 1:
        print("⚠️  No data available to generate scatter plots")
        return

    print(f"\n📊 GENERATING SCATTER PLOTS FOR {len(batch_data)} DEFECTS")
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
            pass_threshold = values < threshold
            threshold_label = f'Threshold < {threshold}'
        else:  # above
            pass_threshold = values > threshold
            threshold_label = f'Threshold > {threshold}'

        # Scatter plot
        plt.scatter(sample_nums, values, s=200, c=plot_config['color'], alpha=0.7,
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
        plt.ylabel(plot_config['ylabel'], fontsize=12, fontweight='bold')
        plt.title(plot_config['title'], fontsize=14, fontweight='bold', pad=20)
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
        fig.suptitle('Hyperspectral Analysis Summary - All Defects', fontsize=16, fontweight='bold')

        for i, plot_config in enumerate(selected_configs):
            ax = axes[i]
            metric = plot_config['metric']
            values = df_batch[metric].values
            sample_nums = df_batch['Sample_Num'].values
            threshold = plot_config['threshold']
            threshold_type = plot_config['threshold_type']

            # Determine which samples pass/fail the threshold
            if threshold_type == 'below':
                pass_threshold = values < threshold
            else:  # above
                pass_threshold = values > threshold

            # Scatter plot
            ax.scatter(sample_nums, values, s=120, c=plot_config['color'], alpha=0.7,
                      edgecolors='black', linewidth=1.5, zorder=3)

            # Add X markers for samples that don't pass threshold
            failed_samples = ~pass_threshold
            if np.any(failed_samples):
                failed_x = sample_nums[failed_samples]
                failed_y = values[failed_samples]
                ax.scatter(failed_x, failed_y, s=150, marker='x', c='red', linewidth=3, zorder=4)

            # Add sample labels (smaller for combined plot)
            for x, y, sample in zip(sample_nums, values, df_batch['Sample']):
                ax.annotate(sample.replace('Defect', 'D'), (x, y), textcoords="offset points",
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
            ax.set_ylabel(plot_config['ylabel'], fontsize=10, fontweight='bold')
            ax.set_title(plot_config['title'].replace(' Analysis Across All Defects', ''),
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

    # Fix SAM units display in threshold summary
    print(f"\n{'='*80}")
    print("🎯 THRESHOLD ANALYSIS SUMMARY")
    print(f"{'='*80}")

    # Print threshold criteria with correct units
    print(f"📊 THRESHOLD CRITERIA:")
    for plot_config in selected_configs:
        threshold_op = "<" if plot_config['threshold_type'] == 'below' else ">"
        unit_str = ""
        if "degrees" in plot_config['ylabel'].lower():
            unit_str = "°"
        print(f"   • {plot_config['ylabel']} {threshold_op} {plot_config['threshold']}{unit_str}")

    print(f"\n📊 SCATTER PLOT SUMMARY:")
    print(f"   Generated {len(selected_configs)} individual plots + 1 combined plot saved to: {output_dir}")
    for plot_config in selected_configs:
        print(f"   - {plot_config['filename']}: {plot_config['ylabel']} analysis")
    if len(selected_configs) <= 3:
        print(f"   - Combined_Analysis_Scatter: All metrics in one view")
    print(f"   Batch data summary saved to: {batch_summary_path}")

    return output_dir

def get_selected_metrics_from_config(config):
    """Get selected metrics from config with their display properties and thresholds"""
    selected_configs = []

    # Check if plotting_configuration exists in config
    if 'plotting_configuration' not in config:
        print("⚠️  No plotting_configuration found in config. Using default metrics.")
        return select_metrics_for_plotting(get_available_metrics(), config['quality_thresholds'])

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

def validate_plotting_configuration(config):
    """Validate that the plotting configuration is valid"""
    if 'plotting_configuration' not in config:
        return True, "No plotting_configuration found - will use defaults"

    plotting_config = config['plotting_configuration']
    selected_metrics = plotting_config.get('selected_metrics', {})
    available_options = plotting_config.get('available_options', {})

    errors = []
    warnings = []

    # Check that selected metrics are in available options
    for ctq_type, selected_metric in selected_metrics.items():
        if ctq_type not in available_options:
            errors.append(f"CTQ type '{ctq_type}' not found in available_options")
            continue

        if selected_metric not in available_options[ctq_type]:
            errors.append(f"Selected metric '{selected_metric}' not available for {ctq_type}")
            errors.append(f"Available options: {available_options[ctq_type]}")
            continue

        # Check if threshold exists for selected metric
        threshold_key = get_threshold_key_for_metric(selected_metric)
        if threshold_key and threshold_key not in config.get('quality_thresholds', {}):
            warnings.append(f"No threshold found for '{selected_metric}' (expected: {threshold_key})")

    if errors:
        return False, "Configuration errors: " + "; ".join(errors)
    elif warnings:
        return True, "Configuration warnings: " + "; ".join(warnings)
    else:
        return True, "Configuration is valid"

def main():
    """Main function to generate scatter plots based on config.json"""
    print("="*80)
    print("🔬 Hyperspectral Analysis Score Plot Generator")
    print("="*80)

    # Load configuration
    config = load_config()
    if config is None:
        return

    # Validate plotting configuration
    is_valid, message = validate_plotting_configuration(config)
    if not is_valid:
        print(f"❌ {message}")
        return
    elif "warnings" in message.lower():
        print(f"⚠️  {message}")

    # Extract configuration parameters
    samples = config['data_paths']['samples']
    results_dir_prefix = config['data_paths']['results_dir_prefix']

    print(f"📋 Configuration:")
    print(f"   Samples to analyze: {len(samples)} - {samples}")
    print(f"   Results directory prefix: {results_dir_prefix}")

    # Display selected metrics from config
    if 'plotting_configuration' in config:
        plotting_config = config['plotting_configuration']
        selected_metrics = plotting_config.get('selected_metrics', {})
        print(f"   Selected metrics from config:")
        for ctq_type, metric_key in selected_metrics.items():
            print(f"     • {ctq_type}: {metric_key}")

    # Get selected metrics based on config
    selected_configs = get_selected_metrics_from_config(config)

    if not selected_configs:
        print("❌ No valid metrics found for plotting. Please check your configuration.")
        return

    print(f"   Plotting {len(selected_configs)} metrics:")
    for config_item in selected_configs:
        threshold_op = "<" if config_item['threshold_type'] == 'below' else ">"
        unit_str = "°" if "degrees" in config_item['ylabel'].lower() else ""
        print(f"     • {config_item['ylabel']} {threshold_op} {config_item['threshold']}{unit_str}")

    # Collect analysis data
    batch_data = collect_analysis_data(samples, results_dir_prefix)

    if not batch_data:
        print("❌ No analysis data found. Please run main.py first to generate analysis results.")
        return

    # Generate scatter plots with config-driven metric selection
    output_dir = "assets/scatter_plots"
    generate_flexible_scatter_plots(batch_data, output_dir, config, selected_configs)

    print(f"\n{'='*80}")
    print(f"✅ Scatter plot generation complete!")
    print(f"   All plots saved to: {output_dir}")
    print(f"{'='*80}")

    # Display how to change metric selections
    if 'plotting_configuration' in config:
        print(f"\n💡 To change which metrics are plotted:")
        print(f"   Edit the 'selected_metrics' section in config.json")
        available_options = config['plotting_configuration'].get('available_options', {})
        for ctq_type, options in available_options.items():
            current_selection = config['plotting_configuration']['selected_metrics'].get(ctq_type, 'None')
            print(f"   {ctq_type} (current: {current_selection}): {', '.join(options)}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

