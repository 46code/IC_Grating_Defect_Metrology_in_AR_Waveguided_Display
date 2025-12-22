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

print("Hyperspectral Defect Analysis Pipeline")
print("=" * 80)

# Load configuration from JSON file
config_path = "config.json"
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Configuration loaded from: {config_path}")
except FileNotFoundError:
    print(f"ERROR: Configuration file not found: {config_path}")
    print("Please create config.json with required parameters.")
    exit(1)
except json.JSONDecodeError as e:
    print(f"ERROR: Invalid JSON in configuration file: {e}")
    exit(1)

# Initialize plotter (this handles matplotlib configuration)
plotter = HyperspectralPlotter(config)

# =============================================================================
# EXTRACT CONFIGURATION PARAMETERS
# =============================================================================

# Data paths
DATA_PATHS = config['data_paths']
SAMPLES = config['data_paths']['samples']
RESULTS_DIR_PREFIX = config['data_paths']['results_dir_prefix']

# Analysis parameters
PERCENTILE_THRESHOLD = config['analysis_parameters']['percentile_threshold']
WAVELENGTH_RANGE = config['analysis_parameters']['wavelength_range']
CIRCLE_CROP_REGION = tuple(config['analysis_parameters']['circle_crop_region'])
NUM_SECTORS = config['analysis_parameters']['num_sectors']
NUM_FIDUCIALS = config['analysis_parameters']['num_fiducials']

# Visualization parameters
GENERATE_PLOTS = config['visualization_parameters']['generate_plots']

# Validate samples list
if not SAMPLES or len(SAMPLES) == 0:
    print("ERROR: No samples specified in configuration")
    print("Please add sample names to the 'samples' array in config.json")
    exit(1)

# Validate required configuration sections
required_sections = ['data_paths', 'analysis_parameters', 'visualization_parameters']
for section in required_sections:
    if section not in config:
        print(f"ERROR: Missing required configuration section: {section}")
        exit(1)

# Validate critical parameters
try:
    percentile = config['analysis_parameters']['percentile_threshold']
    if not isinstance(percentile, (int, float)) or percentile <= 0:
        print("ERROR: percentile_threshold must be a positive number")
        exit(1)
except KeyError:
    print("ERROR: Missing required parameter: analysis_parameters.percentile_threshold")
    exit(1)

print(f"Configuration Summary:")
print(f"   Samples to process: {len(SAMPLES)} - {SAMPLES}")
print(f"   Percentile threshold: {PERCENTILE_THRESHOLD}")
print(f"   Results directory prefix: {RESULTS_DIR_PREFIX}")
print(f"   Uniformity sectors: {NUM_SECTORS}")
print(f"   Generate plots: {'Yes' if GENERATE_PLOTS else 'No'}")

def process_sample(sample_name):
    """Process a single sample through the complete analysis pipeline"""
    print(f"\nProcessing Sample: {sample_name}")
    print("=" * 60)

    # Create assets directory for this sample
    RESULTS_DIR = os.path.join(RESULTS_DIR_PREFIX, f"analysis_{sample_name}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # =============================================================================
    # STEP 1: DATA LOADING AND REFLECTANCE COMPUTATION
    # =============================================================================

    print(f"\nStep 1: Data Loading and Reflectance Computation")
    print("-" * 50)

    loader = SpectralDataLoader(DATA_PATHS)
    datasets = loader.load_all_datasets(sample_name)

    if datasets is None:
        print(f"FAILURE_REASON: Failed to load datasets for {sample_name} - missing data files")
        return None

    # Extract individual cubes for convenience with validation
    try:
        reference_cube = datasets['reference']
        sample_cube = datasets[sample_name]
        white_cube = datasets['white']
        dark_cube = datasets['dark']

        # Validate cube shapes are consistent
        if not all(cube.shape[:2] == reference_cube.shape[:2] for cube in [sample_cube, white_cube, dark_cube]):
            print(f"FAILURE_REASON: Inconsistent cube dimensions for {sample_name}")
            return None

        # Validate cube data is not empty or invalid
        for name, cube in [('reference', reference_cube), ('sample', sample_cube),
                          ('white', white_cube), ('dark', dark_cube)]:
            if cube.size == 0:
                print(f"FAILURE_REASON: Empty {name} cube for {sample_name}")
                return None
            if np.isnan(cube).all():
                print(f"FAILURE_REASON: {name} cube contains only NaN values for {sample_name}")
                return None

    except KeyError as e:
        print(f"FAILURE_REASON: Missing dataset key {e} for {sample_name}")
        return None
    except Exception as e:
        print(f"FAILURE_REASON: Dataset extraction failed for {sample_name}: {str(e)}")
        return None

    print(f"Loaded cubes: Ref{reference_cube.shape}, Sample{sample_cube.shape}")

    # Filter spectral cubes to WAVELENGTH_RANGE
    print(f"Filtering spectral data to wavelength range: {WAVELENGTH_RANGE['min']}-{WAVELENGTH_RANGE['max']}nm")

    # Get wavelength indices within the specified range
    wavelengths = np.arange(450, 951, 10)  # Standard wavelength array for this dataset
    valid_indices = np.where(
        (wavelengths >= WAVELENGTH_RANGE['min']) &
        (wavelengths <= WAVELENGTH_RANGE['max'])
    )[0]

    if len(valid_indices) == 0:
        print(f"FAILURE_REASON: No wavelengths found in range {WAVELENGTH_RANGE['min']}-{WAVELENGTH_RANGE['max']}nm for {sample_name}")
        return None

    # Crop all cubes to the valid wavelength range
    reference_cube = reference_cube[:, :, valid_indices]
    sample_cube = sample_cube[:, :, valid_indices]
    white_cube = white_cube[:, :, valid_indices]
    dark_cube = dark_cube[:, :, valid_indices]

    # Update wavelength array
    filtered_wavelengths = wavelengths[valid_indices]

    print(f"Filtered cubes: Ref{reference_cube.shape}, Sample{sample_cube.shape}")
    print(f"   Wavelength bands: {len(filtered_wavelengths)} ({filtered_wavelengths[0]}nm - {filtered_wavelengths[-1]}nm)")

    # Load reference-specific white and Dark data
    print("   Loading reference calibration data...")
    reference_white_cube, reference_dark_cube = loader.load_reference_calibration_data()

    if reference_white_cube is None or reference_dark_cube is None:
        print(f"FAILURE_REASON: Failed to load reference calibration data for {sample_name}")
        return None

    # Filter reference calibration data to same wavelength range
    reference_white_cube = reference_white_cube[:, :, valid_indices]
    reference_dark_cube = reference_dark_cube[:, :, valid_indices]

    print(f"Filtered reference calibration cubes: White{reference_white_cube.shape}, Dark{reference_dark_cube.shape}")

    # Initialize reflectance analyzer
    analyzer = ReflectanceAnalyzer()

    # Compute reflectance for reference using its specific white/Dark data
    print("   Computing reference reflectance using reference-specific calibration data...")
    try:
        reference_reflectance = analyzer.compute_reflectance(reference_cube, reference_white_cube, reference_dark_cube)

        if reference_reflectance is None:
            print(f"FAILURE_REASON: Reference reflectance computation returned None for {sample_name}")
            return None

        if reference_reflectance.size == 0:
            print(f"FAILURE_REASON: Empty reference reflectance data for {sample_name}")
            return None

        if np.isnan(reference_reflectance).all():
            print(f"FAILURE_REASON: Reference reflectance contains only NaN values for {sample_name}")
            return None

    except Exception as e:
        print(f"FAILURE_REASON: Reference reflectance computation failed for {sample_name}: {str(e)}")
        return None

    # Compute reflectance for sample using sample white/Dark data
    print("   Computing sample reflectance using sample calibration data...")
    try:
        sample_reflectance = analyzer.compute_reflectance(sample_cube, white_cube, dark_cube)

        if sample_reflectance is None:
            print(f"FAILURE_REASON: Sample reflectance computation returned None for {sample_name}")
            return None

        if sample_reflectance.size == 0:
            print(f"FAILURE_REASON: Empty sample reflectance data for {sample_name}")
            return None

        if np.isnan(sample_reflectance).all():
            print(f"FAILURE_REASON: Sample reflectance contains only NaN values for {sample_name}")
            return None

        # Validate reflectance ranges are reasonable
        ref_min, ref_max = reference_reflectance.min(), reference_reflectance.max()
        sam_min, sam_max = sample_reflectance.min(), sample_reflectance.max()

        if ref_min < -0.1 or ref_max > 5.0:
            print(f"FAILURE_REASON: Reference reflectance out of range for {sample_name}: {ref_min:.3f} to {ref_max:.3f}")
            return None

        if sam_min < -0.1 or sam_max > 5.0:
            print(f"FAILURE_REASON: Sample reflectance out of range for {sample_name}: {sam_min:.3f} to {sam_max:.3f}")
            return None

        print(f"   Reference range: {ref_min:.3f} - {ref_max:.3f}")
        print(f"   Sample range: {sam_min:.3f} - {sam_max:.3f}")

    except Exception as e:
        print(f"FAILURE_REASON: Sample reflectance computation failed for {sample_name}: {str(e)}")
        return None

    # Create projections for detection
    print("   Creating projections for feature detection...")

    # Use the same wavelength range that was used for filtering
    proj_wl_range = (WAVELENGTH_RANGE['min'], WAVELENGTH_RANGE['max'])

    reference_projection = analyzer.create_projection(reference_reflectance, filtered_wavelengths, proj_wl_range)
    sample_projection = analyzer.create_projection(sample_reflectance, filtered_wavelengths, proj_wl_range)

    if reference_projection is None or sample_projection is None:
        print(f"FAILURE_REASON: Failed to create projections for {sample_name}")
        return None

    print(f"Projections created with range {proj_wl_range[0]}-{proj_wl_range[1]}nm")

    # =============================================================================
    # STEP 2: DETECT FIDUCIAL POINTS (using projections)
    # =============================================================================

    print(f"\n{'='*20} STEP 2: FIDUCIAL DETECTION (using projections) {'='*20}")

    detector = FeatureDetector()

    # Convert projections to uint8 for detection (they are normalized 0-1)
    ref_projection_uint8 = (reference_projection * 255).astype(np.uint8)
    sample_projection_uint8 = (sample_projection * 255).astype(np.uint8)

    # Detect fiducials in reference and sample projections with error handling
    try:
        reference_fiducials, ref_binary = detector.detect_fiducials(ref_projection_uint8, num_fiducials=NUM_FIDUCIALS,
                                                                   percentile=3)
        sample_fiducials, sample_binary = detector.detect_fiducials(sample_projection_uint8, num_fiducials=NUM_FIDUCIALS,
                                                                  percentile=PERCENTILE_THRESHOLD)

        # Validate fiducial detection results
        if reference_fiducials is None or sample_fiducials is None:
            print(f"FAILURE_REASON: Fiducial detection returned None for {sample_name}")
            return None

        # Convert to lists if needed and validate
        if not isinstance(reference_fiducials, list):
            reference_fiducials = list(reference_fiducials) if reference_fiducials is not None else []
        if not isinstance(sample_fiducials, list):
            sample_fiducials = list(sample_fiducials) if sample_fiducials is not None else []

    except Exception as e:
        print(f"FAILURE_REASON: Fiducial detection failed for {sample_name}: {str(e)}")
        return None

    print(f"Fiducial Detection Results:")
    print(f"   Reference fiducials: {len(reference_fiducials)} points")
    print(f"   Sample fiducials: {len(sample_fiducials)} points")

    # Check for insufficient fiducials
    if len(reference_fiducials) < NUM_FIDUCIALS:
        print(f"FAILURE_REASON: Insufficient reference fiducials for {sample_name} - found {len(reference_fiducials)}, need {NUM_FIDUCIALS}")
        return None

    if len(sample_fiducials) < NUM_FIDUCIALS:
        print(f"FAILURE_REASON: Insufficient sample fiducials for {sample_name} - found {len(sample_fiducials)}, need {NUM_FIDUCIALS}")
        return None

    # Visualize fiducial detection using plotter
    if GENERATE_PLOTS:
        plotter.plot_fiducials(ref_projection_uint8, sample_projection_uint8, reference_fiducials, sample_fiducials,
                              sample_name, f"Projection_{proj_wl_range[0]}-{proj_wl_range[1]}nm", RESULTS_DIR,
                              ref_binary=ref_binary, sample_binary=sample_binary)

    # =============================================================================
    # STEP 3: DETECT IC CIRCLE IN REFERENCE (using projection)
    # =============================================================================

    # Step 3: Circle Detection
    print(f"\nStep 3: Circle Detection")
    print("-" * 50)

    # Detect IC circle with comprehensive error handling
    try:
        ic_circle = detector.detect_circle(ref_projection_uint8, crop_region=CIRCLE_CROP_REGION)

        if ic_circle is None or 'center' not in ic_circle or 'radius' not in ic_circle:
            print(f"FAILURE_REASON: Failed to detect IC circle for {sample_name}")
            return None

        center = ic_circle['center']
        radius = ic_circle['radius']

        # Validate circle parameters
        if not isinstance(center, (tuple, list)) or len(center) != 2:
            print(f"FAILURE_REASON: Invalid circle center for {sample_name}: {center}")
            return None

        if not isinstance(radius, (int, float)) or radius <= 0:
            print(f"FAILURE_REASON: Invalid circle radius for {sample_name}: {radius}")
            return None

        # Check if circle is within image bounds
        height, width = ref_projection_uint8.shape
        if (center[0] < radius or center[0] > width - radius or
            center[1] < radius or center[1] > height - radius):
            print(f"FAILURE_REASON: Circle extends beyond image bounds for {sample_name}")
            return None

    except Exception as e:
        print(f"FAILURE_REASON: Circle detection failed for {sample_name}: {str(e)}")
        return None

    print(f"IC Circle detected:")
    print(f"   Center: ({center[0]:.1f}, {center[1]:.1f})")
    print(f"   Radius: {radius:.1f} pixels")

    # Visualize circle detection using plotter
    if GENERATE_PLOTS:
        plotter.plot_circle_detection(ref_projection_uint8, reference_fiducials, center, radius,
                                     f"Projection_{proj_wl_range[0]}-{proj_wl_range[1]}nm", RESULTS_DIR, CIRCLE_CROP_REGION)

    # Step 2: Feature Detection
    print(f"\nStep 2: Feature Detection")
    print("-" * 50)

    registrator = ImageRegistration()

    # Compute homography matrix with error handling
    try:
        homography_matrix = registrator.compute_homography(sample_fiducials, reference_fiducials)

        if homography_matrix is None:
            print(f"FAILURE_REASON: Failed to compute homography matrix for {sample_name}")
            return None

        # Validate homography matrix
        if np.isnan(homography_matrix).any() or np.isinf(homography_matrix).any():
            print(f"FAILURE_REASON: Invalid homography matrix for {sample_name} - contains NaN or Inf values")
            return None

    except Exception as e:
        print(f"FAILURE_REASON: Homography computation failed for {sample_name}: {str(e)}")
        return None

    # Get registration quality
    quality_metrics = registrator.get_registration_quality()
    print(f"Registration Quality: {quality_metrics['quality']}")
    if 'reprojection_error' in quality_metrics:
        print(f"   Reprojection error: {quality_metrics['reprojection_error']:.4f} pixels")

    # Register the sample reflectance cube with error handling
    print("   Registering sample reflectance cube...")
    try:
        registered_sample_reflectance = registrator.register_cube(sample_reflectance, homography_matrix)

        if registered_sample_reflectance is None:
            print(f"FAILURE_REASON: Failed to register sample reflectance cube for {sample_name}")
            return None

        # Validate registered cube
        if registered_sample_reflectance.size == 0:
            print(f"FAILURE_REASON: Empty registered cube for {sample_name}")
            return None

        if np.isnan(registered_sample_reflectance).all():
            print(f"FAILURE_REASON: Registered cube contains only NaN values for {sample_name}")
            return None

    except Exception as e:
        print(f"FAILURE_REASON: Cube registration failed for {sample_name}: {str(e)}")
        return None

    # Create registered projection for visualization
    registered_sample_projection = analyzer.create_projection(registered_sample_reflectance, filtered_wavelengths, proj_wl_range)
    registered_projection_uint8 = (registered_sample_projection * 255).astype(np.uint8)

    # Visualize registration assets using plotter
    if GENERATE_PLOTS:
        plotter.plot_registration(ref_projection_uint8, registered_projection_uint8, reference_fiducials, RESULTS_DIR)

    # Step 5: ROI Mask Creation
    print(f"\nStep 5: ROI Mask Creation")
    print("-" * 50)

    # Create ROI mask from IC circle
    print("   Creating IC ROI mask...")
    # Create ROI mask with error handling
    try:
        roi_mask = registrator.create_roi_mask(reference_projection.shape, ic_circle)

        if roi_mask is None:
            print(f"FAILURE_REASON: Failed to create ROI mask for {sample_name}")
            return None

        # Validate ROI mask
        roi_pixel_count = np.sum(roi_mask)
        if roi_pixel_count == 0:
            print(f"FAILURE_REASON: Empty ROI mask for {sample_name} - no pixels selected")
            return None

        if roi_pixel_count < 100:  # Minimum reasonable number of pixels
            print(f"FAILURE_REASON: ROI mask too small for {sample_name} - only {roi_pixel_count} pixels")
            return None

        print(f"   ROI contains {roi_pixel_count:,} pixels")

    except Exception as e:
        print(f"FAILURE_REASON: ROI mask creation failed for {sample_name}: {str(e)}")
        return None

    # Visualize ROI and reflectance using plotter (using projection for display)
    if GENERATE_PLOTS:
        # For visualization, use the middle band from reflectance data
        analysis_band_idx = len(filtered_wavelengths) // 2

        plotter.plot_reflectance(ref_projection_uint8, registered_projection_uint8, roi_mask,
                                reference_reflectance, registered_sample_reflectance, analysis_band_idx,
                                sample_name, f"Projection_{proj_wl_range[0]}-{proj_wl_range[1]}nm", RESULTS_DIR)

    # Step 6: Hyperspectral Analysis
    print(f"\nStep 6: Hyperspectral Analysis")
    print("-" * 50)

    # Initialize hyperspectral analyzer
    hyperspectral_analyzer = HyperspectralAnalyzer()

    # Perform complete analysis using registered sample reflectance with error handling
    try:
        results = hyperspectral_analyzer.analyze_sample(
            registered_sample_reflectance,
            reference_reflectance,
            center,
            roi_mask,
            num_sectors=NUM_SECTORS
        )

        if results is None:
            print(f"FAILURE_REASON: Hyperspectral analysis returned None for {sample_name}")
            return None

        # Validate required result components
        required_keys = ['rmse', 'sam', 'ring', 'uniformity']
        for key in required_keys:
            if key not in results:
                print(f"FAILURE_REASON: Missing analysis component '{key}' for {sample_name}")
                return None

        # Validate individual result components
        rmse_results = results['rmse']
        sam_results = results['sam']
        ring_results = results['ring']
        uniformity_results = results['uniformity']

        # Check for invalid numerical results
        if any(np.isnan(list(rmse_results.values())).any() for rmse_results in [rmse_results] if isinstance(rmse_results, dict)):
            print(f"FAILURE_REASON: RMSE results contain NaN values for {sample_name}")
            return None

        if any(np.isnan(list(sam_results.values())).any() for sam_results in [sam_results] if isinstance(sam_results, dict)):
            print(f"FAILURE_REASON: SAM results contain NaN values for {sample_name}")
            return None

    except Exception as e:
        print(f"FAILURE_REASON: Hyperspectral analysis failed for {sample_name}: {str(e)}")
        return None

    print(f"\nAnalysis Results for {sample_name}:")
    print(f"   RMSE Overall: {rmse_results['overall_rmse']:.6f}")
    print(f"   RMSE Per-Pixel Mean: {rmse_results['rmse_per_pixel_mean']:.6f}")
    print(f"   RMSE Per-Pixel Median: {rmse_results['rmse_per_pixel_median']:.6f}")
    print(f"   RMSE Per-Pixel P95: {rmse_results['rmse_per_pixel_p95']:.6f}")
    print(f"   SAM Mean: {sam_results['sam_mean']:.2f} degrees")
    print(f"   SAM Median: {sam_results['sam_median']:.2f} degrees")
    print(f"   SAM P95: {sam_results['sam_p95']:.2f} degrees")
    print(f"   Ring Delta: {np.degrees(ring_results['delta_ring']):.2f}°")
    print(f"   Uniformity Score: {uniformity_results['U']:.3f}")

    # Step 7: Visualization
    print(f"\nStep 7: Visualization")
    print("-" * 50)

    print(f"\n{'='*20} STEP 7: VISUALIZATION (RMSE & SAM Maps) {'='*20}")

    # Visualize analysis maps using plotter and get crop parameters for uniformity plot
    if GENERATE_PLOTS:
        crop_params = plotter.plot_analysis_maps(rmse_results['rmse_map'], sam_results['sam_map'],
                                                 roi_mask, center, rmse_results, sam_results, ring_results,
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

    # Step 4: Image Registration
    print(f"\nStep 4: Image Registration")
    print("-" * 50)

    print(f"\n{'='*20} STEP 8: UNIFORMITY ANALYSIS VISUALIZATION {'='*20}")

    print(f"   Uniformity Score: {uniformity_results['U']:.3f}")
    print(f"   Ring Delta: {ring_results['delta_ring']:.4f} rad ({np.degrees(ring_results['delta_ring']):.2f} degrees)")

    # Visualize uniformity analysis using plotter
    if GENERATE_PLOTS:
        plotter.plot_uniformity_analysis(sam_results['sam_map'], roi_mask, center,
                                         uniformity_results, ring_results,
                                         sample_name, RESULTS_DIR, crop_params)

    print(f"   Uniformity analysis complete")

    # Step 3: Circle Detection
    print(f"\nStep 3: Circle Detection")
    print("-" * 50)

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
    print("HYPERSPECTRAL ANALYSIS SUMMARY")
    print(f"{line}")
    print(summary_df.to_string(index=False))

    print(f"\n{line}")
    print("METRIC BREAKDOWN:")
    print(f"   RMSE: Measures spectral accuracy between sample and reference")
    print(f"   SAM: Spectral Angle Mapper - measures spectral similarity")
    print(f"   Ring Delta: Difference between inner and outer ring regions")
    print(f"   Uniformity: Sector-based uniformity score (0-1, higher is better)")
    print(f"{line}")

    # Save assets to CSV
    summary_df.to_csv(f'{RESULTS_DIR}/analysis_summary.csv', index=False)

    print(f"Hyperspectral Analysis complete! Results saved to: {RESULTS_DIR}")
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
            print(f"FAILURE_DETECTED: Sample {sample_name} failed during processing")
    except Exception as e:
        print(f"ERROR_DETECTED: Sample {sample_name} failed with exception: {str(e)}")
        failed_samples.append(sample_name)

# =============================================================================
# FINAL SUMMARY FOR ALL SAMPLES
# =============================================================================

print(f"\n{'='*80}")
print("BATCH_PROCESSING_SUMMARY")
print(f"{'='*80}")
print(f"SUCCESS: Successfully processed: {len(all_results)} samples")
if all_results:
    for result in all_results:
        print(f"   - {result['sample_name']}: Results saved to {result['results_dir']}")

if failed_samples:
    print(f"FAILED_TO_PROCESS: {len(failed_samples)} samples")
    for sample in failed_samples:
        print(f"   - {sample}")

print(f"Batch processing complete!")
print(f"\nINFO: To generate scatter plots, run: python3 generate_score_plots.py")
print(f"{'='*80}")
