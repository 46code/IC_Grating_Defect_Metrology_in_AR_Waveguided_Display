#!/usr/bin/env python3
"""
Gage R&R Trial Runner
Runs trials through main.py with different operators and percentile_threshold values

Author: Khang Tran
Date: December 2025
"""

import json
import subprocess
import sys

# Configure stdout encoding for Windows compatibility
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Operators to process
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
PERCENTILE_THRESHOLDS = [3.4, 3.8, 4, 3.3, 3.45, 4, 4, 3.9]
# 450 - 800: 3.5, 3.8, 4.2, 3.5, 3.4, 3.93, 4.5, 3.9
# 450 - 950: 3.4, 3.8, 4, 3.3, 3.45, 4, 4, 3.9

def extract_failure_summary(stdout_text):
    """Extract detailed failure information from main.py output"""
    if not stdout_text:
        return []

    failure_info = []
    lines = stdout_text.split('\n')
    failed_samples = set()

    for line in lines:
        line = line.strip()

        # Capture specific failure reasons with sample names
        if "FAILURE_REASON:" in line:
            reason = line.split("FAILURE_REASON:", 1)[1].strip()
            failure_info.append(f"REASON: {reason}")

            # Extract sample name from failure reason
            if " for " in reason:
                parts = reason.split(" for ")
                if len(parts) > 1:
                    sample_part = parts[1].split(" ")[0]  # Get first word after "for"
                    if sample_part.startswith("Sample"):
                        failed_samples.add(sample_part)

        # Capture sample processing failures
        elif "FAILURE_DETECTED:" in line:
            sample_info = line.replace("FAILURE_DETECTED:", "").strip()
            # Extract sample name more carefully
            words = sample_info.split()
            for word in words:
                if word.startswith("Sample") and len(word) > 6:  # Sample + number
                    failed_samples.add(word)

        # Capture error exceptions
        elif "ERROR_DETECTED:" in line:
            sample_info = line.replace("ERROR_DETECTED:", "").strip()
            # Extract sample name more carefully
            words = sample_info.split()
            for word in words:
                if word.startswith("Sample") and len(word) > 6:  # Sample + number
                    failed_samples.add(word)

    # Add summary of failed samples count if any specific samples were detected
    if failed_samples:
        failure_info.append(f"Failed samples: {', '.join(sorted(failed_samples))}")

    return failure_info

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

    print(f"Config updated: {operator} (threshold: {percentile_threshold})")

def run_trial(trial_num, operator, percentile_threshold):
    """Run a single trial with the specified operator and threshold"""
    total_trials = len(OPERATORS)
    print(f"\nTrial {trial_num}/{total_trials}: {operator} (threshold: {percentile_threshold})")
    print("-" * 50)

    try:
        # Update config for this trial
        update_config_for_trial(operator, percentile_threshold)

        # Run main.py
        print("Running analysis...")
        result = subprocess.run([sys.executable, 'main.py'],
                              capture_output=True, text=True, timeout=1800)


        # Extract detailed failure information
        failure_summary = extract_failure_summary(result.stdout)
        failed_samples_detected = len(failure_summary) > 0

        if failed_samples_detected:
            print("Sample failures detected:")
            for failure in failure_summary:
                print(f"  {failure}")

        if result.returncode == 0 and not failed_samples_detected:
            print("Analysis completed successfully")

            # Run generate_score_plots.py after successful main analysis
            print("Running generate_score_plots.py...")
            plot_result = subprocess.run([sys.executable, 'generate_score_plots.py'],
                                       capture_output=True, text=True, timeout=300)  # 5 min timeout for plots

            if plot_result.returncode == 0:
                print("Score plots generated successfully")
                return True
            else:
                print("Score plots failed, but main analysis succeeded")
                print(f"Plot error: {plot_result.stderr[-300:] if plot_result.stderr else 'No error details'}")
                # Still consider this a success since main analysis worked
                return True

        else:
            if failed_samples_detected:
                print("Trial failed due to sample failures")
            else:
                print("Trial failed")

            if result.stderr:
                print(f"Error: {result.stderr[-500:]}")  # Last 500 chars of error
            return False

    except subprocess.TimeoutExpired:
        print("Trial timed out (30 minutes)")
        return False
    except Exception as e:
        print(f"Trial error: {str(e)}")
        return False

def main():
    """Run all trials for the configured operators"""
    total_trials = len(OPERATORS)

    print("Gage R&R Trial Runner")
    print("-" * 30)
    print("Operators:")
    for i, op in enumerate(OPERATORS, 1):
        print(f"  {i}. {op} (threshold: {PERCENTILE_THRESHOLDS[i-1]})")
    print("Reference: KhangT2 (fixed)")
    print("-" * 30)

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
    print(f"\nRESULTS")
    print("-" * 30)
    print(f"Successful trials: {successful_trials}/{total_trials}")

    if failed_trials:
        print(f"Failed trials: {len(failed_trials)}")
        for trial in failed_trials:
            print(f"  - {trial}")

    if successful_trials == total_trials:
        print("\nAll trials completed successfully!")
    else:
        failed_count = total_trials - successful_trials
        print(f"\n{failed_count} trial(s) failed")

    print(f"Results saved in results/ directories")
    print("-" * 30)

if __name__ == "__main__":
    main()
