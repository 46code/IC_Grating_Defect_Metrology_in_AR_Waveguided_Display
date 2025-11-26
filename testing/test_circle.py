#!/usr/bin/env python3
"""
Quick test script for circle detection
Tests circle detection on reference data with configurable parameters
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

def test_circle_detection_with_params(cube, wavelengths, target_wavelength=700,
                                    crop_region=(0.075, 0.5, 0.0, 0.5),
                                    hough_param1=25, hough_param2=30,
                                    min_radius=10, max_radius=100):
    """Test circle detection with configurable parameters"""

    # Find closest wavelength band
    wavelength_idx = np.argmin(np.abs(wavelengths - target_wavelength))
    actual_wavelength = wavelengths[wavelength_idx]

    print(f"\n🎯 Testing circle detection...")
    print(f"   Target wavelength: {target_wavelength}nm")
    print(f"   Actual wavelength: {actual_wavelength}nm (band {wavelength_idx})")
    print(f"   Crop region: {crop_region}")
    print(f"   Hough params: param1={hough_param1}, param2={hough_param2}")
    print(f"   Radius range: {min_radius}-{max_radius} pixels")

    # Extract the specific wavelength band
    test_image = cube[:, :, wavelength_idx]
    height, width = test_image.shape

    # Apply cropping to show the region
    x_min, x_max, y_min, y_max = crop_region
    x_start = int(x_min * width)
    x_end = int(x_max * width)
    y_start = int(y_min * height)
    y_end = int(y_max * height)

    cropped_image = test_image[y_start:y_end, x_start:x_end]

    print(f"\n🔧 Preprocessing steps:")
    print(f"   ✅ Original image: {test_image.shape}")
    print(f"   ✅ Cropped region: {cropped_image.shape}")
    print(f"   ✅ Crop coordinates: [{x_start}:{x_end}, {y_start}:{y_end}]")

    # Manual preprocessing to show intermediate steps
    # Step 1: Normalize
    image_norm = ((cropped_image - cropped_image.min()) / (cropped_image.max() - cropped_image.min()) * 255).astype(np.uint8)
    print(f"   ✅ Normalized cropped image: {image_norm.min()}-{image_norm.max()}")

    # Step 2: Bilateral filter
    blurred = cv2.bilateralFilter(image_norm, 9, 75, 75)
    print(f"   ✅ Bilateral filter applied")

    # Step 3: Hough Circle Detection
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=hough_param1,
        param2=hough_param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(f"   ✅ Found {len(circles)} circle candidates")
    else:
        print(f"   ❌ No circles detected by HoughCircles")
        circles = []

    # Initialize detector with custom parameters and run detection
    detector = FeatureDetector()
    # Temporarily update detector parameters
    detector.circle_params.update({
        'min_radius': min_radius,
        'max_radius': max_radius,
        'hough_param1': hough_param1,
        'hough_param2': hough_param2
    })

    circle_result = detector.detect_circle(test_image, crop_region=crop_region)

    print(f"\n✅ Detection complete!")
    if circle_result:
        center = circle_result['center']
        radius = circle_result['radius']
        score = circle_result['score']
        print(f"   Circle found: center=({center[0]:.1f}, {center[1]:.1f}), radius={radius:.1f}, score={score:.3f}")
    else:
        print(f"   No valid circles found")

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original full image with crop region highlighted
    axes[0, 0].imshow(test_image, cmap='gray')
    # Draw crop region rectangle
    rect = plt.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start,
                        linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title(f'Original Image\n{actual_wavelength}nm (Band {wavelength_idx})')
    axes[0, 0].axis('off')

    # Cropped region
    axes[0, 1].imshow(cropped_image, cmap='gray')
    axes[0, 1].set_title(f'Cropped Region\n{cropped_image.shape[1]}×{cropped_image.shape[0]} px')
    axes[0, 1].axis('off')

    # Normalized cropped image
    axes[0, 2].imshow(image_norm, cmap='gray')
    axes[0, 2].set_title(f'Normalized\nRange: {image_norm.min()}-{image_norm.max()}')
    axes[0, 2].axis('off')

    # Bilateral filtered image
    axes[1, 0].imshow(blurred, cmap='gray')
    axes[1, 0].set_title(f'Bilateral Filtered\nKernel: 9×9')
    axes[1, 0].axis('off')

    # Cropped image with all circle candidates
    axes[1, 1].imshow(cropped_image, cmap='gray')
    if len(circles) > 0:
        for i, (x, y, r) in enumerate(circles):
            circle_patch = plt.Circle((x, y), r, fill=False, color='yellow', linewidth=2, alpha=0.7)
            axes[1, 1].add_patch(circle_patch)
            axes[1, 1].plot(x, y, '+', color='yellow', markersize=10, markeredgewidth=2)
            axes[1, 1].text(x+r+5, y, f'C{i+1}', color='yellow', fontweight='bold')
    axes[1, 1].set_title(f'All Circle Candidates\nFound: {len(circles)} circles')
    axes[1, 1].axis('off')

    # Final result on full image
    axes[1, 2].imshow(test_image, cmap='gray')
    # Draw crop region
    rect2 = plt.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start,
                         linewidth=2, edgecolor='red', facecolor='none', linestyle='--', alpha=0.6)
    axes[1, 2].add_patch(rect2)

    if circle_result:
        center = circle_result['center']
        radius = circle_result['radius']
        # Draw detected circle
        circle_patch = plt.Circle(center, radius, fill=False, color='lime', linewidth=3)
        axes[1, 2].add_patch(circle_patch)
        axes[1, 2].plot(center[0], center[1], '+', color='lime', markersize=15, markeredgewidth=3)
        axes[1, 2].text(center[0]+radius+10, center[1],
                       f'R={radius:.1f}\nS={score:.3f}',
                       color='lime', fontweight='bold', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        title_suffix = f'DETECTED'
    else:
        title_suffix = 'NOT DETECTED'

    axes[1, 2].set_title(f'Final Result\n{title_suffix}')
    axes[1, 2].axis('off')

    plt.suptitle(f'Circle Detection Analysis - Reference\nWavelength: {actual_wavelength}nm, Params: P1={hough_param1}, P2={hough_param2}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save result
    output_filename = f'circle_test_{actual_wavelength}nm_P1-{hough_param1}_P2-{hough_param2}.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\n💾 Result saved as: {output_filename}")

    plt.show()

    return circle_result, circles, blurred

def main():
    # Configuration - Change these values to test different settings
    TARGET_WAVELENGTH = 650  # Target wavelength in nm
    CROP_REGION = (0.05, 0.5, 0.0, 0.5)  # (x_min, x_max, y_min, y_max) as fractions
    HOUGH_PARAM1 = 25  # Edge detector threshold
    HOUGH_PARAM2 = 30  # Accumulator threshold (lower = more circles detected)
    MIN_RADIUS = 10    # Minimum circle radius
    MAX_RADIUS = 100   # Maximum circle radius

    # Load Reference data (where we expect to find circles)
    reference_path = Path("../Gage R&R/KhangT2/Reference")

    if not reference_path.exists():
        print(f"❌ Reference path not found: {reference_path}")
        return

    # Load spectral cube
    cube, wavelengths = load_spectral_cube(reference_path)
    if cube is None:
        return

    # Test circle detection
    circle_result, all_circles, filtered_image = test_circle_detection_with_params(
        cube, wavelengths,
        target_wavelength=TARGET_WAVELENGTH,
        crop_region=CROP_REGION,
        hough_param1=HOUGH_PARAM1,
        hough_param2=HOUGH_PARAM2,
        min_radius=MIN_RADIUS,
        max_radius=MAX_RADIUS
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY:")
    print(f"   Wavelength used: {wavelengths[np.argmin(np.abs(wavelengths - TARGET_WAVELENGTH))]}nm")
    print(f"   Crop region: {CROP_REGION}")
    print(f"   Hough parameters: P1={HOUGH_PARAM1}, P2={HOUGH_PARAM2}")
    print(f"   Radius range: {MIN_RADIUS}-{MAX_RADIUS} pixels")
    print(f"   Circle candidates found: {len(all_circles)}")

    if circle_result:
        print(f"   ✅ CIRCLE DETECTED:")
        print(f"      Center: ({circle_result['center'][0]:.1f}, {circle_result['center'][1]:.1f})")
        print(f"      Radius: {circle_result['radius']:.1f} pixels")
        print(f"      Score: {circle_result['score']:.3f}")
    else:
        print(f"   ❌ NO VALID CIRCLE DETECTED")
        print(f"      Try adjusting: wavelength, crop region, or Hough parameters")

    print(f"{'='*60}")

    # Suggestions for parameter tuning
    print(f"\n💡 PARAMETER TUNING TIPS:")
    print(f"   • Lower HOUGH_PARAM2 to detect more circles (current: {HOUGH_PARAM2})")
    print(f"   • Adjust MIN/MAX_RADIUS if circle size is known")
    print(f"   • Try different wavelengths (current: {TARGET_WAVELENGTH}nm)")
    print(f"   • Modify crop region if circle is elsewhere")

if __name__ == "__main__":
    main()
