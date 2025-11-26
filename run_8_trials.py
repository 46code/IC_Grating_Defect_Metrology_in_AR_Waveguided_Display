#!/usr/bin/env python3
"""
Simple Gage R&R Trial Runner
Runs 6 trials through main.py with different percentile_threshold values

Author: Khang Tran
Date: November 2025
"""

import json
import subprocess
import os

# List of operators to process (6 trials)
OPERATORS = [
    "KhangT1",
    "KhangT2",
    "KiduT1",
    "KiduT2",
    "LuelT1",
    "LuelT2",
    "AnirbanT1",
    "AnirbanT2"
]

# Different percentile_threshold values for each trial
PERCENTILE_THRESHOLDS = [3.4, 3.8, 4, 3.3, 3.3, 4, 4, 3.9]
# 450 - 800: 3.5, 3.8, 4.2, 3.5, 3.4, 3.93, 4.5, 3.9
# 450 - 950: 3.4, 3.8, 4, 3.3, 3.3, 4, 4, 3.9

def update_config_for_trial(operator, percentile_threshold):
    """Update config.json for the current trial"""

    # Read current config
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Update paths for the operator
    config["data_paths"]["results_dir_prefix"] = f"results/{operator}"
    config["data_paths"]["sample_path"] = f"${{base_path}}/{operator}"
    config["data_paths"]["sample_white_path"] = f"${{base_path}}/{operator}/White"
    config["data_paths"]["sample_dark_path"] = f"${{base_path}}/{operator}/Dark"

    # Update percentile threshold
    config["analysis_parameters"]["percentile_threshold"] = percentile_threshold

    # Write updated config
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Updated config for {operator} (threshold: {percentile_threshold})")

def run_trial(trial_num, operator, percentile_threshold):
    """Run a single trial"""
    print(f"\n{'='*60}")
    print(f"🚀 TRIAL {trial_num}/8: {operator}")
    print(f"   Percentile Threshold: {percentile_threshold}")
    print(f"{'='*60}")

    try:
        # Update config for this trial
        update_config_for_trial(operator, percentile_threshold)

        # Run main.py
        print("   Running main.py...")
        result = subprocess.run(['python3', 'main.py'],
                              capture_output=True, text=True, timeout=1800)

        if result.returncode == 0:
            print(f"✅ Main analysis completed successfully for Trial {trial_num}")

            # Run generate_score_plots.py after successful main analysis
            print("   Running generate_score_plots.py...")
            plot_result = subprocess.run(['python3', 'generate_score_plots.py'],
                                       capture_output=True, text=True, timeout=300)  # 5 min timeout for plots

            if plot_result.returncode == 0:
                print(f"✅ Score plots generated successfully for Trial {trial_num}")
                return True
            else:
                print(f"⚠️  Score plots failed for Trial {trial_num}, but main analysis succeeded")
                print(f"   Plot error: {plot_result.stderr[-300:] if plot_result.stderr else 'No error details'}")
                # Still consider this a success since main analysis worked
                return True

        else:
            print(f"❌ Trial {trial_num} failed!")
            print(f"   Error: {result.stderr[-500:]}")  # Last 500 chars of error
            return False

    except subprocess.TimeoutExpired:
        print(f"❌ Trial {trial_num} timed out (30 minutes)")
        return False
    except Exception as e:
        print(f"❌ Trial {trial_num} error: {str(e)}")
        return False

def main():
    """Run all 8 trials"""
    print("🔬 Gage R&R - 8 Trial Runner")
    print("=" * 60)
    print("Operators:")
    for i, op in enumerate(OPERATORS, 1):
        print(f"   {i}. {op} (threshold: {PERCENTILE_THRESHOLDS[i-1]})")
    print("Reference: KhangT2 (fixed)")
    print("=" * 60)

    successful_trials = 0
    failed_trials = []

    # Run each trial
    for i, (operator, threshold) in enumerate(zip(OPERATORS, PERCENTILE_THRESHOLDS), 1):
        success = run_trial(i, operator, threshold)

        if success:
            successful_trials += 1
        else:
            failed_trials.append(f"Trial {i} ({operator})")

    # Final summary
    print(f"\n{'='*60}")
    print("🎯 FINAL RESULTS")
    print(f"{'='*60}")
    print(f"✅ Successful trials: {successful_trials}/8")

    if failed_trials:
        print(f"❌ Failed trials: {len(failed_trials)}")
        for trial in failed_trials:
            print(f"   - {trial}")

    if successful_trials == 8:
        print("\n🎉 All trials completed successfully!")
    else:
        print(f"\n⚠️  {6-successful_trials} trials failed")

    print(f"📊 Results saved in results/ directories")
    print("=" * 60)

if __name__ == "__main__":
    main()
