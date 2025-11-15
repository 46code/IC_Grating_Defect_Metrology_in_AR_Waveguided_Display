"""
Hyperspectral Analysis Pipeline
Complete workflow for fiducial detection, circle detection, homography, and reflectance analysis

Author: Khang Tran
Date: October 2025
"""

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

import numpy as np
import pandas as pd
import os
import json

# Import our custom libraries
from modules import SpectralDataLoader, FeatureDetector, ImageRegistration, ReflectanceAnalyzer, HyperspectralAnalyzer, HyperspectralPlotter

# =============================================================================
# LOAD CONFIGURATION
# =============================================================================

print("🔬 Hyperspectral Defect Analysis Pipeline")
print("=" * 80)

# Load configuration from JSON file
config_path = "config.json"
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"✅ Configuration loaded from: {config_path}")
except FileNotFoundError:
    print(f"❌ Configuration file not found: {config_path}")
    print("   Please create config.json with required parameters.")
    exit(1)

# Initialize plotter (this handles matplotlib configuration)
plotter = HyperspectralPlotter(config)

# =============================================================================
# EXTRACT CONFIGURATION PARAMETERS
# =============================================================================

# Data paths
BASE_PATH = config['data_paths']['base_path']
SAMPLES = config['data_paths']['samples']
RESULTS_DIR_PREFIX = config['data_paths']['results_dir_prefix']

# Analysis parameters
ANALYSIS_WAVELENGTH = config['analysis_parameters']['analysis_wavelength']
WAVELENGTH_RANGE = config['analysis_parameters']['wavelength_range']
CIRCLE_CROP_REGION = tuple(config['analysis_parameters']['circle_crop_region'])
NUM_SECTORS = config['analysis_parameters']['num_sectors']
NUM_FIDUCIALS = config['analysis_parameters']['num_fiducials']

# Visualization parameters
GENERATE_PLOTS = config['visualization_parameters']['generate_plots']

# Validate wavelength is within acceptable range
if not (WAVELENGTH_RANGE['min'] <= ANALYSIS_WAVELENGTH <= WAVELENGTH_RANGE['max']):
    print(f"❌ Error: Analysis wavelength {ANALYSIS_WAVELENGTH}nm is outside valid range")
    print(f"   Valid range: {WAVELENGTH_RANGE['min']}-{WAVELENGTH_RANGE['max']}nm")
    exit(1)

# Validate samples list
if not SAMPLES or len(SAMPLES) == 0:
    print("❌ Error: No samples specified in configuration")
    print("   Please add sample names to the 'samples' array in config.json")
    exit(1)

print(f"📋 Configuration:")
print(f"   Samples to process: {len(SAMPLES)} - {SAMPLES}")
print(f"   Analysis wavelength: {ANALYSIS_WAVELENGTH}nm (range: {WAVELENGTH_RANGE['min']}-{WAVELENGTH_RANGE['max']}nm)")
print(f"   Results directory prefix: {RESULTS_DIR_PREFIX}")
print(f"   Uniformity sectors: {NUM_SECTORS}")
print(f"   Generate plots: {'Yes' if GENERATE_PLOTS else 'No'}")

def process_sample(sample_name):
    """Process a single sample through the complete analysis pipeline"""
    print(f"\n{'='*60}")
    print(f"🔬 PROCESSING SAMPLE: {sample_name}")
    print(f"{'='*60}")

    # Create assets directory for this sample
    RESULTS_DIR = RESULTS_DIR_PREFIX + sample_name
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # =============================================================================
    # STEP 1: DATA LOADING
    # =============================================================================

    print(f"\n{'='*20} STEP 1: DATA LOADING {'='*20}")

    loader = SpectralDataLoader(BASE_PATH)
    datasets = loader.load_all_datasets(sample_name)

    if datasets is None:
        print(f"❌ Failed to load datasets for {sample_name}. Skipping.")
        return None

    # Extract individual cubes for convenience
    reference_cube = datasets['reference']
    sample_cube = datasets[sample_name]
    white_cube = datasets['white']
    dark_cube = datasets['Darkreference']

    print(f"✅ Loaded cubes: Ref{reference_cube.shape}, Sample{sample_cube.shape}")

    # Filter spectral cubes to WAVELENGTH_RANGE
    print(f"🔧 Filtering spectral data to wavelength range: {WAVELENGTH_RANGE['min']}-{WAVELENGTH_RANGE['max']}nm")

    # Get wavelength indices within the specified range
    wavelengths = np.arange(450, 951, 10)  # Standard wavelength array for this dataset
    valid_indices = np.where(
        (wavelengths >= WAVELENGTH_RANGE['min']) &
        (wavelengths <= WAVELENGTH_RANGE['max'])
    )[0]

    if len(valid_indices) == 0:
        print(f"❌ Error: No wavelengths found in range {WAVELENGTH_RANGE['min']}-{WAVELENGTH_RANGE['max']}nm")
        return None

    # Crop all cubes to the valid wavelength range
    reference_cube = reference_cube[:, :, valid_indices]
    sample_cube = sample_cube[:, :, valid_indices]
    white_cube = white_cube[:, :, valid_indices]
    dark_cube = dark_cube[:, :, valid_indices]

    # Update wavelength array
    filtered_wavelengths = wavelengths[valid_indices]

    print(f"✅ Filtered cubes: Ref{reference_cube.shape}, Sample{sample_cube.shape}")
    print(f"   Wavelength bands: {len(filtered_wavelengths)} ({filtered_wavelengths[0]}nm - {filtered_wavelengths[-1]}nm)")

    # Get analysis wavelength band from filtered data
    # Find the band index for the analysis wavelength in the filtered data
    analysis_band_idx = None
    for i, wl in enumerate(filtered_wavelengths):
        if wl == ANALYSIS_WAVELENGTH:
            analysis_band_idx = i
            break

    if analysis_band_idx is None:
        print(f"❌ Error: Analysis wavelength {ANALYSIS_WAVELENGTH}nm not found in filtered range")
        return None

    ref_band = reference_cube[:, :, analysis_band_idx]
    sample_band = sample_cube[:, :, analysis_band_idx]
    band_idx = analysis_band_idx

    print(f"   Using band {band_idx} for {ANALYSIS_WAVELENGTH}nm analysis")

    # =============================================================================
    # STEP 2: DETECT FIDUCIAL POINTS
    # =============================================================================

    print(f"\n{'='*20} STEP 2: FIDUCIAL DETECTION {'='*20}")

    detector = FeatureDetector()

    # Detect fiducials in reference and sample images
    reference_fiducials = detector.detect_fiducials(ref_band, num_fiducials=NUM_FIDUCIALS)
    sample_fiducials = detector.detect_fiducials(sample_band, num_fiducials=NUM_FIDUCIALS)

    print(f"✅ Fiducial Detection Results:")
    print(f"   Reference fiducials: {len(reference_fiducials)} points")
    print(f"   Sample fiducials: {len(sample_fiducials)} points")

    # Visualize fiducial detection using plotter
    if GENERATE_PLOTS:
        plotter.plot_fiducials(ref_band, sample_band, reference_fiducials, sample_fiducials,
                              sample_name, ANALYSIS_WAVELENGTH, RESULTS_DIR)

    # =============================================================================
    # STEP 3: DETECT IC CIRCLE IN REFERENCE
    # =============================================================================

    print(f"\n{'='*20} STEP 3: IC CIRCLE DETECTION {'='*20}")

    ic_circle = detector.detect_circle(ref_band, crop_region=CIRCLE_CROP_REGION)

    center = ic_circle['center']
    radius = ic_circle['radius']

    print(f"✅ IC Circle detected:")
    print(f"   Center: ({center[0]:.1f}, {center[1]:.1f})")
    print(f"   Radius: {radius:.1f} pixels")

    # Visualize circle detection using plotter
    if GENERATE_PLOTS:
        plotter.plot_circle_detection(ref_band, reference_fiducials, center, radius,
                                     ANALYSIS_WAVELENGTH, RESULTS_DIR)

    # =============================================================================
    # STEP 4: COMPUTE HOMOGRAPHY AND REGISTER SAMPLE
    # =============================================================================

    print(f"\n{'='*20} STEP 4: IMAGE REGISTRATION {'='*20}")

    registrator = ImageRegistration()

    # Compute homography matrix
    homography_matrix = registrator.compute_homography(sample_fiducials, reference_fiducials)

    # Get registration quality
    quality_metrics = registrator.get_registration_quality()
    print(f"✅ Registration Quality: {quality_metrics['quality']}")
    if 'reprojection_error' in quality_metrics:
        print(f"   Reprojection error: {quality_metrics['reprojection_error']:.4f} pixels")

    # Register the sample cube
    print("   Registering sample cube...")
    registered_sample_cube = registrator.register_cube(sample_cube, homography_matrix)

    # Visualize registration assets using plotter
    if GENERATE_PLOTS:
        registered_band = registered_sample_cube[:, :, band_idx]
        plotter.plot_registration(ref_band, registered_band, reference_fiducials, RESULTS_DIR)

    # =============================================================================
    # STEP 5: CREATE IC ROI MASK AND COMPUTE REFLECTANCE
    # =============================================================================

    print(f"\n{'='*20} STEP 5: REFLECTANCE COMPUTATION {'='*20}")

    # Create ROI mask from IC circle
    print("   Creating IC ROI mask...")
    roi_mask = registrator.create_roi_mask(ref_band.shape, ic_circle)
    print(f"   ROI contains {np.sum(roi_mask):,} pixels")

    # Initialize reflectance analyzer
    analyzer = ReflectanceAnalyzer()

    # Compute reflectance for reference and sample
    print("   Computing reference reflectance...")
    reference_reflectance = analyzer.compute_reflectance(reference_cube, white_cube, dark_cube)

    print("   Computing sample reflectance...")
    sample_reflectance = analyzer.compute_reflectance(registered_sample_cube, white_cube, dark_cube)

    # Visualize ROI and reflectance using plotter
    if GENERATE_PLOTS:
        registered_band = registered_sample_cube[:, :, band_idx]
        plotter.plot_reflectance(ref_band, registered_band, roi_mask,
                                reference_reflectance, sample_reflectance, band_idx,
                                sample_name, ANALYSIS_WAVELENGTH, RESULTS_DIR)

    # =============================================================================
    # STEP 6: HYPERSPECTRAL ANALYSIS
    # =============================================================================

    print(f"\n{'='*20} STEP 6: HYPERSPECTRAL ANALYSIS {'='*20}")

    # Initialize hyperspectral analyzer
    analyzer = HyperspectralAnalyzer()

    # Perform complete analysis
    results = analyzer.analyze_sample(
        sample_reflectance,
        reference_reflectance,
        center,
        roi_mask,
        num_sectors=NUM_SECTORS
    )

    if results is None:
        print("❌ Analysis failed. Exiting.")
        exit(1)

    # Extract assets
    rmse_results = results['rmse']
    sam_results = results['sam']
    ring_results = results['ring']
    uniformity_results = results['uniformity']

    print(f"\n✅ Analysis Results for {sample_name}:")
    print(f"   RMSE Overall: {rmse_results['overall_rmse']:.6f}")
    print(f"   RMSE Per-Pixel Mean: {rmse_results['rmse_per_pixel_mean']:.6f}")
    print(f"   RMSE Per-Pixel Median: {rmse_results['rmse_per_pixel_median']:.6f}")
    print(f"   RMSE Per-Pixel P95: {rmse_results['rmse_per_pixel_p95']:.6f}")
    print(f"   SAM Mean: {np.degrees(sam_results['sam_mean']):.2f}°")
    print(f"   SAM Median: {np.degrees(sam_results['sam_median']):.2f}°")
    print(f"   SAM P95: {np.degrees(sam_results['sam_p95']):.2f}°")
    print(f"   Ring Delta: {np.degrees(ring_results['delta_ring']):.2f}°")
    print(f"   Uniformity Score: {uniformity_results['U']:.3f}")

    # =============================================================================
    # STEP 7: VISUALIZATION (RMSE & SAM Maps)
    # =============================================================================

    print(f"\n{'='*20} STEP 7: VISUALIZATION (RMSE & SAM Maps) {'='*20}")

    # Visualize analysis maps using plotter and get crop parameters for uniformity plot
    if GENERATE_PLOTS:
        crop_params = plotter.plot_analysis_maps(rmse_results['rmse_map'], sam_results['sam_map'],
                                                 roi_mask, center, rmse_results, ring_results,
                                                 sample_name, RESULTS_DIR)
    else:
        # Need crop parameters for uniformity analysis even when not plotting
        roi_coords = np.where(roi_mask)
        roi_distances = np.sqrt((roi_coords[1] - center[0])**2 + (roi_coords[0] - center[1])**2)
        max_radius = np.max(roi_distances)
        padded_radius = int(max_radius * plotter.config['visualization_parameters']['padding_factor'])
        x_min = max(0, int(center[0] - padded_radius))
        y_min = max(0, int(center[1] - padded_radius))
        crop_params = (x_min, y_min, max_radius)

    # =============================================================================
    # STEP 8: UNIFORMITY ANALYSIS VISUALIZATION
    # =============================================================================

    print(f"\n{'='*20} STEP 8: UNIFORMITY ANALYSIS VISUALIZATION {'='*20}")

    print(f"   Uniformity Score: {uniformity_results['U']:.3f}")
    print(f"   Ring Delta: {ring_results['delta_ring']:.4f} rad ({np.degrees(ring_results['delta_ring']):.2f}°)")

    # Visualize uniformity analysis using plotter
    if GENERATE_PLOTS:
        plotter.plot_uniformity_analysis(sam_results['sam_map'], roi_mask, center,
                                         uniformity_results, ring_results,
                                         sample_name, RESULTS_DIR, crop_params)

    print(f"   Uniformity analysis complete")

    # =============================================================================
    # STEP 9: FINAL SUMMARY
    # =============================================================================

    print(f"\n{'='*20} STEP 9: FINAL SUMMARY {'='*20}")

    line = "=" * 80

    # Create summary table
    summary_data = [
        ["RMSE Overall", f"{rmse_results['overall_rmse']:.6f}"],
        ["RMSE Per-Pixel Mean", f"{rmse_results['rmse_per_pixel_mean']:.6f}"],
        ["RMSE Per-Pixel Median", f"{rmse_results['rmse_per_pixel_median']:.6f}"],
        ["RMSE Per-Pixel P95", f"{rmse_results['rmse_per_pixel_p95']:.6f}"],
        ["SAM Mean", f"{np.degrees(sam_results['sam_mean']):.2f}°"],
        ["SAM Median", f"{np.degrees(sam_results['sam_median']):.2f}°"],
        ["SAM P95", f"{np.degrees(sam_results['sam_p95']):.2f}°"],
        ["Ring Delta", f"{np.degrees(ring_results['delta_ring']):.2f}°"],
        ["Ring Inner Mean", f"{ring_results['inner_mean']:.6f} rad"],
        ["Ring Outer Mean", f"{ring_results['outer_mean']:.6f} rad"],
        ["Uniformity Score", f"{uniformity_results['U']:.3f}"],
        ["Number of Sectors", f"{len(uniformity_results['sector_meds'])}"],
        ["ROI Pixels", f"{int(np.sum(roi_mask)):,}"]
    ]

    summary_df = pd.DataFrame(summary_data, columns=["Metric", "Value"])

    # Display assets
    print(f"\n{line}")
    print("📊 HYPERSPECTRAL ANALYSIS SUMMARY")
    print(f"{line}")
    print(summary_df.to_string(index=False))

    print(f"\n{line}")
    print("📋 METRIC BREAKDOWN:")
    print(f"   RMSE: Measures spectral accuracy between sample and reference")
    print(f"   SAM: Spectral Angle Mapper - measures spectral similarity")
    print(f"   Ring Delta: Difference between inner and outer ring regions")
    print(f"   Uniformity: Sector-based uniformity score (0-1, higher is better)")
    print(f"{line}")

    # Save assets to CSV
    summary_df.to_csv(f'{RESULTS_DIR}/analysis_summary.csv', index=False)

    print(f"✅ Hyperspectral Analysis complete! Results saved to: {RESULTS_DIR}")
    print(f"{line}")

    return {
        'sample_name': sample_name,
        'results_dir': RESULTS_DIR
    }

# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

all_results = []
failed_samples = []

for sample_name in SAMPLES:
    try:
        result = process_sample(sample_name)
        if result:
            all_results.append(result)
        else:
            failed_samples.append(sample_name)
    except Exception as e:
        print(f"❌ Error processing {sample_name}: {str(e)}")
        failed_samples.append(sample_name)

# =============================================================================
# FINAL SUMMARY FOR ALL SAMPLES
# =============================================================================

print(f"\n{'='*80}")
print("🎯 BATCH PROCESSING SUMMARY")
print(f"{'='*80}")
print(f"✅ Successfully processed: {len(all_results)} samples")
if all_results:
    for result in all_results:
        print(f"   - {result['sample_name']}: Results saved to {result['results_dir']}")

if failed_samples:
    print(f"❌ Failed to process: {len(failed_samples)} samples")
    for sample in failed_samples:
        print(f"   - {sample}")

print(f"✅ Batch processing complete!")
print(f"\nℹ️  To generate scatter plots, run: python3 generate_score_plots.py")
print(f"{'='*80}")
