#!/usr/bin/env python3
"""
Reflectance Comparison Script - Following Main Pipeline Flow
Compares reflectance data from four different datasets following the exact same workflow as main.py:
1. Load raw spectral cubes
2. Filter to wavelength range
3. Compute reflectance
4. Create projections from reflectance
5. Detect circles on projections
6. Apply circular masks for analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile
import sys
import os
import json
import cv2

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from lib_spectral_loader import SpectralDataLoader
from lib_reflectance_analysis import ReflectanceAnalyzer
from lib_feature_detection import FeatureDetector


class ReflectanceComparer:
    """Class for comparing reflectance data following main pipeline workflow"""

    def __init__(self, config_path="config.json"):
        # Load configuration exactly like main.py
        self.config = self.load_config(config_path)

        # Extract configuration parameters like main.py
        self.WAVELENGTH_RANGE = self.config.get('analysis_parameters', {}).get('wavelength_range', {'min': 450, 'max': 950})
        self.CIRCLE_CROP_REGION = tuple(self.config.get('analysis_parameters', {}).get('circle_crop_region', [0.075, 0.5, 0.0, 0.5]))
        self.PERCENTILE_THRESHOLD = self.config.get('analysis_parameters', {}).get('percentile_threshold', 3)

        # Standard wavelength array (same as main.py)
        self.wavelengths = np.arange(450, 951, 10)  # 450-950nm in 10nm steps

        # Initialize analyzers exactly like main.py
        self.reflectance_analyzer = ReflectanceAnalyzer()
        self.feature_detector = FeatureDetector()

        print(f"🎯 Configuration loaded:")
        print(f"   Wavelength range: {self.WAVELENGTH_RANGE['min']}-{self.WAVELENGTH_RANGE['max']}nm")
        print(f"   Circle crop region: {self.CIRCLE_CROP_REGION}")
        print(f"   Percentile threshold: {self.PERCENTILE_THRESHOLD}")

    def load_config(self, config_path):
        """Load configuration from JSON file (same as main.py)"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ Configuration loaded from: {config_path}")
            return config
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {config_path}")
            print("   Using default parameters.")
            return {
                'analysis_parameters': {
                    'wavelength_range': {'min': 450, 'max': 950},
                    'circle_crop_region': [0.075, 0.5, 0.0, 0.5],
                    'percentile_threshold': 3
                }
            }

    def load_and_filter_spectral_cubes(self, sample_path, white_path, dark_path, dataset_name):
        """
        Load and filter spectral cubes following main.py workflow
        """
        print(f"\n🔬 Processing {dataset_name} - Following Main Pipeline Flow")
        print("-" * 60)

        # STEP 1: Load raw spectral cubes (same as main.py)
        print(f"📁 Loading raw spectral cubes...")

        sample_cube = self.load_spectral_cube_raw(sample_path)
        white_cube = self.load_spectral_cube_raw(white_path)
        dark_cube = self.load_spectral_cube_raw(dark_path)

        if any(cube is None for cube in [sample_cube, white_cube, dark_cube]):
            print(f"❌ Failed to load cubes for {dataset_name}")
            return None

        print(f"✅ Loaded cubes: Sample{sample_cube.shape}, White{white_cube.shape}, Dark{dark_cube.shape}")

        # STEP 2: Filter spectral cubes to WAVELENGTH_RANGE (same as main.py)
        print(f"🔧 Filtering spectral data to wavelength range: {self.WAVELENGTH_RANGE['min']}-{self.WAVELENGTH_RANGE['max']}nm")

        # Get wavelength indices within the specified range (same logic as main.py)
        valid_indices = np.where(
            (self.wavelengths >= self.WAVELENGTH_RANGE['min']) &
            (self.wavelengths <= self.WAVELENGTH_RANGE['max'])
        )[0]

        if len(valid_indices) == 0:
            print(f"❌ Error: No wavelengths found in range {self.WAVELENGTH_RANGE['min']}-{self.WAVELENGTH_RANGE['max']}nm")
            return None

        # Crop all cubes to the valid wavelength range (same as main.py)
        sample_cube = sample_cube[:, :, valid_indices]
        white_cube = white_cube[:, :, valid_indices]
        dark_cube = dark_cube[:, :, valid_indices]

        # Update wavelength array (same as main.py)
        filtered_wavelengths = self.wavelengths[valid_indices]

        print(f"✅ Filtered cubes: Sample{sample_cube.shape}, White{white_cube.shape}, Dark{dark_cube.shape}")
        print(f"   Wavelength bands: {len(filtered_wavelengths)} ({filtered_wavelengths[0]}nm - {filtered_wavelengths[-1]}nm)")

        return {
            'sample_cube': sample_cube,
            'white_cube': white_cube,
            'dark_cube': dark_cube,
            'filtered_wavelengths': filtered_wavelengths
        }

    def load_spectral_cube_raw(self, path):
        """Load raw spectral cube without any processing (same as SpectralDataLoader)"""
        path = Path(path)
        if not path.exists():
            print(f"❌ Path not found: {path}")
            return None

        # Get sorted TIFF files
        tiff_files = sorted(list(path.glob("*.tif")))
        if not tiff_files:
            print(f"❌ No TIFF files found in {path}")
            return None

        # Load first image to get dimensions
        first_img = tifffile.imread(tiff_files[0])
        height, width = first_img.shape
        num_bands = len(tiff_files)

        # Initialize cube
        cube = np.zeros((height, width, num_bands), dtype=first_img.dtype)

        # Load all bands
        for i, tiff_file in enumerate(tiff_files):
            cube[:, :, i] = tifffile.imread(tiff_file)

        return cube

    def process_dataset_main_flow(self, sample_path, white_path, dark_path, dataset_name):
        """
        Process dataset following exact main.py workflow:
        1. Load and filter cubes
        2. Compute reflectance
        3. Create projection
        4. Detect circle on projection
        5. Apply circular mask
        """

        # STEP 1-2: Load and filter cubes (following main.py)
        cubes = self.load_and_filter_spectral_cubes(sample_path, white_path, dark_path, dataset_name)
        if cubes is None:
            return None

        sample_cube = cubes['sample_cube']
        white_cube = cubes['white_cube']
        dark_cube = cubes['dark_cube']
        filtered_wavelengths = cubes['filtered_wavelengths']

        # STEP 3: Compute reflectance (same as main.py)
        print("   Computing reflectance using calibration data...")
        reflectance_cube = self.reflectance_analyzer.compute_reflectance(sample_cube, white_cube, dark_cube)

        if reflectance_cube is None:
            print(f"❌ Failed to compute reflectance for {dataset_name}")
            return None

        # STEP 4: Create projection for detection (same as main.py)
        print("   Creating projection for feature detection...")
        proj_wl_range = (self.WAVELENGTH_RANGE['min'], self.WAVELENGTH_RANGE['max'])

        projection = self.reflectance_analyzer.create_projection(reflectance_cube, filtered_wavelengths, proj_wl_range)

        if projection is None:
            print("❌ Failed to create projection")
            return None

        print(f"✅ Projection created with range {proj_wl_range[0]}-{proj_wl_range[1]}nm")

        # STEP 5: Detect circle on projection (same as main.py)
        print(f"   Detecting IC circle on projection...")

        # Convert projection to uint8 for detection (same as main.py)
        projection_uint8 = (projection * 255).astype(np.uint8)

        # Get appropriate crop region for this dataset
        crop_region = self.get_crop_region_for_dataset(dataset_name)
        print(f"   Using crop region: {crop_region} for {dataset_name}")

        ic_circle = self.feature_detector.detect_circle(projection_uint8, crop_region=crop_region)

        if ic_circle is None:
            print("❌ No circle detected on projection")
            return None

        center = ic_circle['center']
        radius = ic_circle['radius']

        print(f"✅ IC Circle detected on projection:")
        print(f"   Center: ({center[0]:.1f}, {center[1]:.1f})")
        print(f"   Radius: {radius:.1f} pixels")

        # STEP 6: Apply circular mask to reflectance cube (following main.py logic)
        print("   Applying detected circular mask to reflectance data...")

        # Create circular mask based on detected circle
        height, width = reflectance_cube.shape[:2]
        y, x = np.ogrid[:height, :width]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2

        # Apply mask to reflectance cube
        masked_reflectance = reflectance_cube.copy()
        for band_idx in range(reflectance_cube.shape[2]):
            masked_reflectance[:, :, band_idx] = np.where(mask, reflectance_cube[:, :, band_idx], 0)

        print(f"   Applied circular mask: {np.sum(mask):,} valid pixels")

        return {
            'reflectance_cube': masked_reflectance,
            'projection': projection_uint8,
            'mask': mask,
            'circle': ic_circle,
            'filtered_wavelengths': filtered_wavelengths,
            'crop_region': crop_region  # Store the crop region used
        }

    def extract_circular_spectrum(self, reflectance_cube, mask, circle_result, wavelengths):
        """Extract spectrum from circular region (following main.py approach)"""
        if reflectance_cube is None or mask is None:
            return None

        print("📊 Extracting circular ROI spectrum (main pipeline style)...")

        # Get valid (masked) coordinates
        valid_coords = np.where(mask > 0)
        num_valid_pixels = len(valid_coords[0])

        if num_valid_pixels == 0:
            print("❌ No valid pixels in circular mask")
            return None

        num_bands = reflectance_cube.shape[2]

        # Extract reflectance for all valid pixels
        valid_reflectance = np.zeros((num_valid_pixels, num_bands), dtype=np.float32)
        for band_idx in range(num_bands):
            valid_reflectance[:, band_idx] = reflectance_cube[valid_coords[0], valid_coords[1], band_idx]

        # Compute statistics
        mean_spectrum = np.mean(valid_reflectance, axis=0)
        std_spectrum = np.std(valid_reflectance, axis=0)
        median_spectrum = np.median(valid_reflectance, axis=0)

        print(f"   Analyzed {num_valid_pixels:,} pixels in detected circle")
        print(f"   Circle: center=({circle_result['center'][0]:.1f}, {circle_result['center'][1]:.1f}), radius={circle_result['radius']:.1f}")
        print(f"   Detection score: {circle_result['score']:.3f}")

        return {
            'mean_spectrum': mean_spectrum,
            'std_spectrum': std_spectrum,
            'median_spectrum': median_spectrum,
            'valid_pixels': num_valid_pixels,
            'mask_coverage': num_valid_pixels / (mask.shape[0] * mask.shape[1]),
            'circle_center': circle_result['center'],
            'circle_radius': circle_result['radius'],
            'detection_score': circle_result['score'],
            'wavelengths': wavelengths
        }

    def compare_all_datasets(self):
        """Compare all datasets following main.py workflow"""
        print("🚀 Starting Reflectance Comparison - Main Pipeline Flow")
        print("=" * 70)

        base_path = Path("/")

        # Dataset definitions
        datasets = {
            'IDP Group C': {
                'sample': base_path / "IDP Group C" / "reference",
                'white': base_path / "IDP Group C" / "white",
                'dark': base_path / "IDP Group C" / "dark"
            },
            'KhangT2': {
                'sample': base_path / "Gage R&R" / "KhangT2" / "Reference",
                'white': base_path / "Gage R&R" / "KhangT2" / "White",
                'dark': base_path / "Gage R&R" / "KhangT2" / "Dark"
            },
            'AnirbanT1-sample01': {
                'sample': base_path / "Gage R&R" / "AnirbanT1" / "sample01",
                'white': base_path / "Gage R&R" / "AnirbanT1" / "whiteref",
                'dark': base_path / "Gage R&R" / "AnirbanT1" / "darkref"
            },
            'AnirbanT1-sample04': {
                'sample': base_path / "Gage R&R" / "AnirbanT1" / "sample04",
                'white': base_path / "Gage R&R" / "AnirbanT1" / "whiteref",
                'dark': base_path / "Gage R&R" / "AnirbanT1" / "darkref"
            }
        }

        # Process each dataset following main.py workflow
        results = {}
        for dataset_name, paths in datasets.items():
            result = self.process_dataset_main_flow(
                paths['sample'], paths['white'], paths['dark'], dataset_name
            )

            if result is not None:
                # Extract spectrum from circular ROI (following main.py style)
                spectrum_data = self.extract_circular_spectrum(result['reflectance_cube'], result['mask'],
                                                             result['circle'], result['filtered_wavelengths'])
                results[dataset_name] = {
                    'reflectance_cube': result['reflectance_cube'],
                    'projection': result['projection'],
                    'spectrum': spectrum_data,
                    'mask': result['mask'],
                    'circle': result['circle'],
                    'wavelengths': result['filtered_wavelengths'],
                    'crop_region': result['crop_region']
                }
            else:
                print(f"❌ Failed to process {dataset_name}")

        return results

    def plot_comparison_main_style(self, results):
        """Create comparison plots following main.py visualization style"""
        print("\n📊 Creating comparison plots (main pipeline style)...")

        # Setup figure - main pipeline style layout with more rows for individual spectra
        fig, axes = plt.subplots(4, 4, figsize=(28, 24))
        fig.suptitle('Reflectance Comparison Analysis - Main Pipeline Flow', fontsize=16, fontweight='bold')

        # Colors for different datasets
        colors = ['blue', 'red', 'green', 'orange']
        dataset_names = list(results.keys())

        # Plot 1: Projections used for circle detection (like main.py visualization)
        for i, (name, data) in enumerate(results.items()):
            if i < 4:  # Show all 4 projections
                ax = axes[0, i]
                if data.get('projection') is not None:
                    projection = data['projection']
                    circle = data['circle']

                    im = ax.imshow(projection, cmap='gray')

                    # Overlay detected circle (like main.py)
                    center = circle['center']
                    radius = circle['radius']
                    circle_outline = plt.Circle(center, radius, fill=False, color='red', linewidth=2)
                    ax.add_patch(circle_outline)
                    ax.plot(center[0], center[1], 'r+', markersize=10, markeredgewidth=2)

                    ax.set_title(f'{name}\nProjection + Circle')
                    ax.set_xlabel('X (pixels)')
                    ax.set_ylabel('Y (pixels)')
                    plt.colorbar(im, ax=ax)

        # Plot 2: Individual Mean reflectance spectra (split into separate subplots)
        for i, (name, data) in enumerate(results.items()):
            if i < 4:  # Show up to 4 individual spectra
                ax = axes[1, i]
                if data['spectrum'] is not None:
                    spectrum = data['spectrum']['mean_spectrum']
                    wavelengths = data['wavelengths']  # Use actual filtered wavelengths
                    ax.plot(wavelengths, spectrum, color=colors[i], linewidth=3, marker='o', markersize=4,
                           markerfacecolor='white', markeredgecolor=colors[i], markeredgewidth=2)

                    # Add statistics to the plot
                    mean_val = np.mean(spectrum)
                    max_val = np.max(spectrum)
                    peak_idx = np.argmax(spectrum)
                    peak_wl = wavelengths[peak_idx]

                    ax.axhline(y=mean_val, color=colors[i], linestyle='--', alpha=0.7, linewidth=1)
                    ax.text(0.05, 0.95, f'Mean: {mean_val:.3f}\nMax: {max_val:.3f}\nPeak: {peak_wl:.0f}nm',
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                ax.set_xlabel('Wavelength (nm)')
                ax.set_ylabel('Reflectance')
                ax.set_title(f'{name}\nMean Reflectance Spectrum')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, None)  # Start y-axis from 0

        # Plot 3: Standard deviation comparison
        ax2 = axes[2, 0]
        for i, (name, data) in enumerate(results.items()):
            if data['spectrum'] is not None:
                std_spectrum = data['spectrum']['std_spectrum']
                wavelengths = data['wavelengths']
                ax2.plot(wavelengths, std_spectrum, color=colors[i], linewidth=2, label=name, marker='s', markersize=3)

        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('Reflectance Std Dev')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 4: Circle detection quality metrics
        ax3 = axes[2, 1]
        detection_scores = []
        radii = []
        names = []

        for name, data in results.items():
            if data['spectrum'] is not None:
                detection_scores.append(data['spectrum']['detection_score'])
                radii.append(data['spectrum']['circle_radius'])
                names.append(name)

        x_pos = np.arange(len(names))
        width = 0.35

        ax3_twin = ax3.twinx()
        bars1 = ax3.bar(x_pos - width/2, detection_scores, width, label='Detection Score', alpha=0.7, color='skyblue')
        bars2 = ax3_twin.bar(x_pos + width/2, radii, width, label='Radius (px)', alpha=0.7, color='lightcoral')

        ax3.set_xlabel('Dataset')
        ax3.set_ylabel('Detection Score', color='blue')
        ax3_twin.set_ylabel('Radius (pixels)', color='red')
        ax3.set_title('Circle Detection Quality')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)

        # Plot 5: Wavelength range comparison
        ax4 = axes[2, 2]
        wl_ranges = []
        for name, data in results.items():
            if data['spectrum'] is not None:
                wavelengths = data['wavelengths']
                wl_ranges.append((wavelengths[0], wavelengths[-1]))

        for i, ((wl_min, wl_max), name) in enumerate(zip(wl_ranges, names)):
            ax4.barh(i, wl_max - wl_min, left=wl_min, alpha=0.7, color=colors[i], label=name)

        ax4.set_xlabel('Wavelength (nm)')
        ax4.set_ylabel('Dataset')
        ax4.set_title('Filtered Wavelength Ranges')
        ax4.set_yticks(range(len(names)))
        ax4.set_yticklabels(names)
        ax4.grid(True, alpha=0.3)

        # Plot 6: Combined comparison for reference
        ax_combined = axes[2, 3]
        for i, (name, data) in enumerate(results.items()):
            if data['spectrum'] is not None:
                spectrum = data['spectrum']['mean_spectrum']
                wavelengths = data['wavelengths']
                ax_combined.plot(wavelengths, spectrum, color=colors[i], linewidth=2, label=name,
                               marker='o', markersize=2, alpha=0.8)

        ax_combined.set_xlabel('Wavelength (nm)')
        ax_combined.set_ylabel('Reflectance')
        ax_combined.set_title('All Spectra Combined\n(for comparison)')
        ax_combined.legend(fontsize=8)
        ax_combined.grid(True, alpha=0.3)

        # Plot 7-10: Individual reflectance maps with circle overlays
        for i, (name, data) in enumerate(results.items()):
            if i < 4:
                ax = axes[3, i]
                if data['reflectance_cube'] is not None and data['mask'] is not None:
                    # Show mean reflectance across all bands
                    mean_reflectance = np.mean(data['reflectance_cube'], axis=2)
                    mask = data['mask']
                    circle = data['circle']

                    # Apply mask for visualization
                    masked_reflectance = np.where(mask, mean_reflectance, np.nan)

                    im = ax.imshow(masked_reflectance, cmap='viridis', vmin=0, vmax=1)

                    # Overlay circle outline
                    center = circle['center']
                    radius = circle['radius']
                    circle_outline = plt.Circle(center, radius, fill=False, color='white', linewidth=2)
                    ax.add_patch(circle_outline)
                    ax.plot(center[0], center[1], 'w+', markersize=8, markeredgewidth=2)

                    ax.set_title(f'{name}\nMean Reflectance')
                    plt.colorbar(im, ax=ax)

        plt.tight_layout()

        # Save plot
        output_path = "reflectance_comparison_main_pipeline_flow.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved as: {output_path}")

        plt.show()

    def generate_summary_report_main_style(self, results):
        """Generate summary report following main.py style"""
        print("\n📋 MAIN PIPELINE FLOW - REFLECTANCE COMPARISON SUMMARY")
        print("=" * 80)
        print(f"🎯 Configuration:")
        print(f"   Wavelength Range: {self.WAVELENGTH_RANGE['min']}-{self.WAVELENGTH_RANGE['max']}nm")
        print(f"   Circle Crop Region: {self.CIRCLE_CROP_REGION}")
        print(f"   Percentile Threshold: {self.PERCENTILE_THRESHOLD}")

        for name, data in results.items():
            print(f"\n🔬 Dataset: {name}")
            print("-" * 50)

            if data['spectrum'] is not None:
                spectrum = data['spectrum']['mean_spectrum']
                std_spectrum = data['spectrum']['std_spectrum']
                valid_pixels = data['spectrum']['valid_pixels']
                coverage = data['spectrum']['mask_coverage'] * 100
                center = data['spectrum']['circle_center']
                radius = data['spectrum']['circle_radius']
                score = data['spectrum']['detection_score']
                wavelengths = data['wavelengths']
                crop_region = data.get('crop_region', self.CIRCLE_CROP_REGION)

                # Calculate key statistics
                mean_reflectance = np.mean(spectrum)
                max_reflectance = np.max(spectrum)
                min_reflectance = np.min(spectrum)
                mean_std = np.mean(std_spectrum)

                # Find peak wavelength
                peak_idx = np.argmax(spectrum)
                peak_wavelength = wavelengths[peak_idx]

                print(f"Wavelength Processing:")
                print(f"  Filtered Range: {wavelengths[0]:.0f}-{wavelengths[-1]:.0f}nm ({len(wavelengths)} bands)")
                print(f"Circle Detection (on projection):")
                print(f"  Center: ({center[0]:.1f}, {center[1]:.1f})")
                print(f"  Radius: {radius:.1f} pixels")
                print(f"  Detection Score: {score:.3f}")
                print(f"Reflectance Analysis:")
                print(f"  Valid Pixels: {valid_pixels:,}")
                print(f"  Image Coverage: {coverage:.1f}%")
                print(f"  Mean Reflectance: {mean_reflectance:.4f}")
                print(f"  Max Reflectance: {max_reflectance:.4f}")
                print(f"  Min Reflectance: {min_reflectance:.4f}")
                print(f"  Mean Std Dev: {mean_std:.4f}")
                print(f"  Peak: {peak_wavelength:.0f}nm (R={spectrum[peak_idx]:.4f})")
                print(f"  SNR: {mean_reflectance/mean_std:.2f}")
                print(f"Crop Region Used: {crop_region}")
            else:
                print("❌ No analysis data available")

        # Cross-dataset comparison
        if len(results) > 1:
            print(f"\n🔄 CROSS-DATASET COMPARISON")
            print("-" * 50)

            valid_datasets = [(name, data) for name, data in results.items()
                            if data['spectrum'] is not None]

            if len(valid_datasets) >= 2:
                # Compare results
                means = [np.mean(data['spectrum']['mean_spectrum']) for name, data in valid_datasets]
                scores = [data['spectrum']['detection_score'] for name, data in valid_datasets]
                radii = [data['spectrum']['circle_radius'] for name, data in valid_datasets]

                max_idx = np.argmax(means)
                min_idx = np.argmin(means)

                print(f"Reflectance Performance:")
                print(f"  Highest: {valid_datasets[max_idx][0]} ({means[max_idx]:.4f})")
                print(f"  Lowest: {valid_datasets[min_idx][0]} ({means[min_idx]:.4f})")
                print(f"  Range: {max(means) - min(means):.4f}")

                print(f"Circle Detection Quality:")
                print(f"  Detection Scores: {min(scores):.3f} - {max(scores):.3f}")
                print(f"  Circle Radii: {min(radii):.1f} - {max(radii):.1f} pixels")
                print(f"  Consistency: {'High' if (max(scores) - min(scores)) < 0.1 else 'Moderate'}")

    def get_crop_region_for_dataset(self, dataset_name):
        """Get the appropriate crop region for a specific dataset"""
        # Special case for KhangT1 and KhangT2 - use different crop region
        if 'KhangT1' in dataset_name or 'KhangT2' in dataset_name:
            return (0.05, 0.5, 0.0, 0.5)
        else:
            # Use default crop region from config for all other datasets
            return self.CIRCLE_CROP_REGION


def main():
    """Main execution function"""
    print("🚀 Starting Reflectance Comparison Analysis")
    print("=" * 60)

    # Initialize comparer
    comparer = ReflectanceComparer()

    # Run comparison
    results = comparer.compare_all_datasets()

    if results:
        # Generate plots
        comparer.plot_comparison_main_style(results)

        # Generate summary report
        comparer.generate_summary_report_main_style(results)

        print(f"\n✅ Analysis complete! Check the generated plots and summary above.")
    else:
        print("❌ No valid results obtained")


if __name__ == "__main__":
    main()
