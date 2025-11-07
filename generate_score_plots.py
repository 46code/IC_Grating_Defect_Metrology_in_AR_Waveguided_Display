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

                # Extract metrics from the summary
                rmse_overall = float(summary_df[summary_df['Metric'] == 'RMSE Overall']['Value'].iloc[0])
                sam_mean_str = summary_df[summary_df['Metric'] == 'SAM Mean']['Value'].iloc[0]
                sam_mean = float(sam_mean_str.split()[0])  # Extract numeric value before 'rad'
                uniformity_score = float(summary_df[summary_df['Metric'] == 'Uniformity Score']['Value'].iloc[0])

                batch_data.append({
                    'Sample': sample_name,
                    'Sample_Num': int(sample_name.replace('Defect', '')),
                    'RMSE_Overall': rmse_overall,
                    'SAM_Mean': sam_mean,
                    'Uniformity_Score': uniformity_score
                })

                print(f"✅ Loaded {sample_name}: RMSE={rmse_overall:.6f}, SAM={sam_mean:.6f}, Uniformity={uniformity_score:.3f}")

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
            'metric': 'SAM_Mean',
            'title': 'SAM Mean Analysis Across All Defects',
            'ylabel': 'SAM Mean (radians)',
            'color': '#3498db',
            'filename': 'SAM_Mean_Scatter',
            'threshold': thresholds['sam_mean_max'],
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
        'SAM_Mean': {'value': thresholds['sam_mean_max'], 'type': 'below'},
        'Uniformity_Score': {'value': thresholds['uniformity_score_min'], 'type': 'above'}
    }

    # Check which samples pass each threshold
    all_pass_samples = []

    for _, row in df_batch.iterrows():
        sample_name = row['Sample']

        # Check RMSE threshold (pass if below threshold)
        rmse_pass = row['RMSE_Overall'] < threshold_config['RMSE_Overall']['value']

        # Check SAM threshold (pass if below threshold)
        sam_pass = row['SAM_Mean'] < threshold_config['SAM_Mean']['value']

        # Check Uniformity threshold (pass if above threshold)
        uniformity_pass = row['Uniformity_Score'] > threshold_config['Uniformity_Score']['value']

        # Sample passes all thresholds if all three conditions are met
        if rmse_pass and sam_pass and uniformity_pass:
            all_pass_samples.append({
                'Sample': sample_name,
                'RMSE': row['RMSE_Overall'],
                'SAM': row['SAM_Mean'],
                'Uniformity': row['Uniformity_Score']
            })

    # Print results
    print(f"📊 THRESHOLD CRITERIA:")
    print(f"   • RMSE Overall < {threshold_config['RMSE_Overall']['value']}")
    print(f"   • SAM Mean < {threshold_config['SAM_Mean']['value']} rad")
    print(f"   • Uniformity Score > {threshold_config['Uniformity_Score']['value']}")

    if all_pass_samples:
        print(f"\n✅ SAMPLES PASSING ALL THRESHOLDS ({len(all_pass_samples)}/{len(df_batch)} samples):")
        print("-" * 80)
        print(f"{'Sample':<10} {'RMSE':<12} {'SAM (rad)':<12} {'Uniformity':<12}")
        print("-" * 80)
        for sample in all_pass_samples:
            print(f"{sample['Sample']:<10} {sample['RMSE']:<12.6f} {sample['SAM']:<12.6f} {sample['Uniformity']:<12.3f}")

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
    print(f"   - SAM_Mean_Scatter: SAM analysis across all defects")
    print(f"   - Uniformity_Score_Scatter: Uniformity analysis across all defects")
    print(f"   - Combined_Analysis_Scatter: All three metrics in one view")
    print(f"   Batch data summary saved to: {batch_summary_path}")

    return output_dir

def main():
    """Main function to generate scatter plots based on config.json"""
    print("="*80)
    print("🔬 Hyperspectral Analysis Score Plot Generator")
    print("="*80)

    # Load configuration
    config = load_config()
    if config is None:
        return

    # Extract configuration parameters
    samples = config['data_paths']['samples']
    results_dir_prefix = config['data_paths']['results_dir_prefix']

    # Display threshold settings from config
    thresholds = config['quality_thresholds']

    print(f"📋 Configuration:")
    print(f"   Samples to analyze: {len(samples)} - {samples}")
    print(f"   Results directory prefix: {results_dir_prefix}")
    print(f"   Quality thresholds:")
    print(f"     • RMSE Overall < {thresholds['rmse_overall_max']}")
    print(f"     • SAM Mean < {thresholds['sam_mean_max']} rad")
    print(f"     • Uniformity Score > {thresholds['uniformity_score_min']}")

    # Collect analysis data
    batch_data = collect_analysis_data(samples, results_dir_prefix)

    if not batch_data:
        print("❌ No analysis data found. Please run main.py first to generate analysis results.")
        return

    # Generate scatter plots with config
    output_dir = "assets/scatter_plots"
    generate_scatter_plots(batch_data, output_dir, config)

    print(f"\n{'='*80}")
    print(f"✅ Scatter plot generation complete!")
    print(f"   All plots saved to: {output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
