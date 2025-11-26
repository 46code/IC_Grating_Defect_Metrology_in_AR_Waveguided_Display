#!/usr/bin/env python3
"""
Quick test script for fiducial detection
Tests fiducial detection on Sample01 data with configurable parameters
"""

import numpy as np
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from modules.lib_feature_detection import FeatureDetector

def load_spectral_cube(sample_path):
    """Load complete spectral cube"""
    tiff_files = sorted(list(sample_path.glob("*.tif")))
    if not tiff_files:
        print(f"❌ No TIFF files found in {sample_path}")
        return None, None

    print(f"📁 Loading spectral cube from {sample_path}...")

    # Load first image to get dimensions
    first_img = tifffile.imread(tiff_files[0])
    height, width = first_img.shape
    num_bands = len(tiff_files)

    # Initialize cube
    cube = np.zeros((height, width, num_bands), dtype=first_img.dtype)

    # Load all bands
    for i, tiff_file in enumerate(tiff_files):
        cube[:, :, i] = tifffile.imread(tiff_file)

    # Generate wavelength array (450-950nm in 10nm steps)
    wavelengths = np.arange(450, 451 + num_bands * 10, 10)[:num_bands]

    print(f"✅ Loaded cube: {cube.shape}")
    print(f"   Wavelengths: {wavelengths[0]}nm - {wavelengths[-1]}nm")

    return cube, wavelengths

def test_fiducial_detection_with_params(cube, wavelengths, target_wavelength=650, percentile_threshold=3):
    """Test fiducial detection with configurable parameters"""

    # Find closest wavelength band
    wavelength_idx = np.argmin(np.abs(wavelengths - target_wavelength))
    actual_wavelength = wavelengths[wavelength_idx]

    print(f"\n🎯 Testing fiducial detection...")
    print(f"   Target wavelength: {target_wavelength}nm")
    print(f"   Actual wavelength: {actual_wavelength}nm (band {wavelength_idx})")
    print(f"   Percentile threshold: {percentile_threshold}%")

    # Extract the specific wavelength band
    test_image = cube[:, :, wavelength_idx]

    # Manual preprocessing to show intermediate steps
    print(f"\n🔧 Preprocessing steps:")

    # Step 1: Normalize
    image_norm = ((test_image - test_image.min()) / (test_image.max() - test_image.min()) * 255).astype(np.uint8)
    print(f"   ✅ Normalized image: {image_norm.min()}-{image_norm.max()}")

    # # Step 2: Blur
    blurred = cv2.GaussianBlur(image_norm, (3, 3), 1)
    print(f"   ✅ Gaussian blur applied")

    # Step 3: Thresholding
    threshold_value = np.percentile(blurred, percentile_threshold)
    binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)[1]
    print(f"   ✅ Threshold value: {threshold_value:.1f}")
    print(f"   ✅ Binary pixels: {np.sum(binary > 0):,} / {binary.size:,} ({100*np.sum(binary > 0)/binary.size:.2f}%)")

    # Initialize detector and run detection
    detector = FeatureDetector()
    fiducials = detector.detect_fiducials(test_image, num_fiducials=4, percentile=percentile_threshold)

    print(f"\n✅ Detection complete!")
    print(f"   Found {len(fiducials)} fiducials")
    for i, (x, y) in enumerate(fiducials):
        print(f"   Fiducial {i+1}: ({x:.1f}, {y:.1f})")

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original image
    axes[0, 0].imshow(test_image, cmap='gray')
    axes[0, 0].set_title(f'Original Image\n{actual_wavelength}nm (Band {wavelength_idx})')
    axes[0, 0].axis('off')

    # Normalized image
    axes[0, 1].imshow(image_norm, cmap='gray')
    axes[0, 1].set_title(f'Normalized Image\nRange: {image_norm.min()}-{image_norm.max()}')
    axes[0, 1].axis('off')

    # Blurred image
    axes[0, 2].imshow(blurred, cmap='gray')
    axes[0, 2].set_title(f'Gaussian Blurred\nKernel: 3x3, σ=1')
    axes[0, 2].axis('off')

    # Binary thresholded image
    axes[1, 0].imshow(binary, cmap='gray')
    axes[1, 0].set_title(f'Binary Thresholded\nP{percentile_threshold}% = {threshold_value:.1f}')
    axes[1, 0].axis('off')

    # Original with detected fiducials
    axes[1, 1].imshow(test_image, cmap='gray')
    for i, (x, y) in enumerate(fiducials):
        axes[1, 1].plot(x, y, 'ro', markersize=12, markeredgewidth=2, markeredgecolor='white')
        axes[1, 1].text(x+20, y-20, f'F{i+1}', color='red', fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    axes[1, 1].set_title(f'Detected Fiducials\nFound: {len(fiducials)} fiducials')
    axes[1, 1].axis('off')

    # Binary with detected fiducials overlay
    axes[1, 2].imshow(binary, cmap='gray')
    for i, (x, y) in enumerate(fiducials):
        axes[1, 2].plot(x, y, 'ro', markersize=12, markeredgewidth=2, markeredgecolor='yellow')
        axes[1, 2].text(x+20, y-20, f'F{i+1}', color='red', fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    axes[1, 2].set_title(f'Binary + Fiducials\nThreshold: P{percentile_threshold}%')
    axes[1, 2].axis('off')

    plt.suptitle(f'Fiducial Detection Analysis - Sample01\nWavelength: {actual_wavelength}nm, Percentile: {percentile_threshold}%',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save result
    output_filename = f'fiducial_test_{actual_wavelength}nm_P{percentile_threshold}.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\n💾 Result saved as: {output_filename}")

    plt.show()

    return fiducials, binary, threshold_value

def main():
    # Configuration - Change these values to test different settings
    TARGET_WAVELENGTH = 650  # Target wavelength in nm
    PERCENTILE_THRESHOLD = 7  # Percentile threshold (1-10 typically)

    # Load Sample01 data
    sample_path = Path("../Gage R&R/KhangT1/Sample22")

    if not sample_path.exists():
        print(f"❌ Sample path not found: {sample_path}")
        return

    # Load spectral cube
    cube, wavelengths = load_spectral_cube(sample_path)
    if cube is None:
        return

    # Test fiducial detection
    fiducials, binary_map, threshold_val = test_fiducial_detection_with_params(
        cube, wavelengths, TARGET_WAVELENGTH, PERCENTILE_THRESHOLD
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY:")
    print(f"   Wavelength used: {wavelengths[np.argmin(np.abs(wavelengths - TARGET_WAVELENGTH))]}nm")
    print(f"   Percentile threshold: {PERCENTILE_THRESHOLD}% = {threshold_val:.1f}")
    print(f"   Binary pixels: {np.sum(binary_map > 0):,} ({100*np.sum(binary_map > 0)/binary_map.size:.2f}%)")
    print(f"   Fiducials detected: {len(fiducials)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
