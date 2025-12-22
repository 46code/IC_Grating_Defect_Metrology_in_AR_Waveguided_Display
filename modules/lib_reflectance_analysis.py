#!/usr/bin/env python3
"""
Reflectance Analysis Library
Handles reflectance computation and ROI analysis for hyperspectral data
"""

import numpy as np
import matplotlib.pyplot as plt

class ReflectanceAnalyzer:
    """Library for reflectance computation and analysis"""

    def __init__(self):
        """Initialize the reflectance analyzer"""
        self.wavelengths = list(range(450, 951, 10))  # 450-950nm in 10nm steps

    def compute_reflectance(self, sample_cube, white_cube, dark_cube):
        """
        Compute reflectance using standard formula: R = (Sample - Dark) / (White - Dark)

        Args:
            sample_cube (np.ndarray): Sample spectral cube
            white_cube (np.ndarray): White reference cube
            dark_cube (np.ndarray): Dark reference cube

        Returns:
            np.ndarray: Reflectance cube (same shape as input, float32)
        """
        print("🔬 Computing reflectance...")

        # Ensure same shape
        if not (sample_cube.shape == white_cube.shape == dark_cube.shape):
            print("ERROR: All cubes must have same shape")
            return None

        height, width, num_bands = sample_cube.shape
        reflectance_cube = np.zeros_like(sample_cube, dtype=np.float32)

        print(f"   Processing {num_bands} wavelength bands...")

        for band_idx in range(num_bands):
            if band_idx % 10 == 0:
                print(f"   Band {band_idx+1}/{num_bands}")

            # Get bands as float32
            sample_band = sample_cube[:, :, band_idx].astype(np.float32)
            white_band = white_cube[:, :, band_idx].astype(np.float32)
            dark_band = dark_cube[:, :, band_idx].astype(np.float32)

            # Compute reflectance with zero-division protection
            denominator = white_band - dark_band
            denominator = np.where(denominator == 0, 1e-10, denominator)

            reflectance = (sample_band - dark_band) / denominator
            reflectance = np.clip(reflectance, 0, 2)  # Reasonable range

            reflectance_cube[:, :, band_idx] = reflectance

        print("SUCCESS: Reflectance computation complete")
        return reflectance_cube

    def extract_roi_spectrum(self, reflectance_cube, roi_mask):
        """
        Extract reflectance spectrum from ROI

        Args:
            reflectance_cube (np.ndarray): Reflectance cube
            roi_mask (np.ndarray): Binary ROI mask

        Returns:
            dict: ROI analysis assets
        """
        print(" Extracting ROI reflectance spectrum...")

        # Get ROI coordinates
        roi_coords = np.where(roi_mask > 0)
        num_roi_pixels = len(roi_coords[0])

        if num_roi_pixels == 0:
            print("ERROR: No ROI pixels found")
            return None

        num_bands = reflectance_cube.shape[2]

        # Extract ROI reflectance for all pixels
        roi_reflectance = np.zeros((num_roi_pixels, num_bands), dtype=np.float32)
        for band_idx in range(num_bands):
            roi_reflectance[:, band_idx] = reflectance_cube[roi_coords[0], roi_coords[1], band_idx]

        # Compute statistics
        mean_spectrum = np.mean(roi_reflectance, axis=0)
        std_spectrum = np.std(roi_reflectance, axis=0)
        median_spectrum = np.median(roi_reflectance, axis=0)

        # Overall statistics
        overall_mean = np.mean(mean_spectrum)
        overall_std = np.mean(std_spectrum)
        snr = overall_mean / overall_std if overall_std > 0 else 0

        results = {
            'roi_pixels': num_roi_pixels,
            'roi_reflectance_data': roi_reflectance,
            'mean_spectrum': mean_spectrum,
            'std_spectrum': std_spectrum,
            'median_spectrum': median_spectrum,
            'wavelengths': self.wavelengths[:num_bands],
            'statistics': {
                'mean': overall_mean,
                'std': overall_std,
                'snr': snr,
                'min': np.min(mean_spectrum),
                'max': np.max(mean_spectrum)
            }
        }

        print(f"SUCCESS: ROI spectrum extracted:")
        print(f"   ROI pixels: {num_roi_pixels:,}")
        print(f"   Mean reflectance: {overall_mean:.4f}")
        print(f"   Std deviation: {overall_std:.4f}")
        print(f"   SNR: {snr:.1f}")

        return results

    def compare_roi_spectra(self, reference_results, sample_results):
        """
        Compare ROI spectra between reference and sample

        Args:
            reference_results (dict): Reference ROI analysis assets
            sample_results (dict): Sample ROI analysis assets

        Returns:
            dict: Comparison analysis
        """
        print(" Comparing reference vs sample ROI spectra...")

        ref_spectrum = reference_results['mean_spectrum']
        sample_spectrum = sample_results['mean_spectrum']

        if len(ref_spectrum) != len(sample_spectrum):
            print("ERROR: Spectra have different lengths")
            return None

        # Compute differences
        absolute_diff = sample_spectrum - ref_spectrum
        relative_diff = (absolute_diff / ref_spectrum) * 100
        relative_diff = np.where(np.isfinite(relative_diff), relative_diff, 0)

        # Statistics
        mean_abs_diff = np.mean(absolute_diff)
        mean_rel_diff = np.mean(relative_diff)
        max_abs_diff = np.max(np.abs(absolute_diff))

        # Wavelength of maximum difference
        max_diff_idx = np.argmax(np.abs(absolute_diff))
        max_diff_wavelength = reference_results['wavelengths'][max_diff_idx]

        comparison = {
            'absolute_difference': absolute_diff,
            'relative_difference_percent': relative_diff,
            'wavelengths': reference_results['wavelengths'],
            'statistics': {
                'mean_absolute_diff': mean_abs_diff,
                'mean_relative_diff_percent': mean_rel_diff,
                'max_absolute_diff': max_abs_diff,
                'max_diff_wavelength': max_diff_wavelength
            },
            'reference_stats': reference_results['statistics'],
            'sample_stats': sample_results['statistics']
        }

        print(f"SUCCESS: Spectral comparison complete:")
        print(f"   Mean difference: {mean_abs_diff:.4f}")
        print(f"   Relative change: {mean_rel_diff:+.2f}%")
        print(f"   Max difference at {max_diff_wavelength}nm: {max_abs_diff:.4f}")

        return comparison

    def plot_roi_spectrum(self, roi_results, title="ROI Reflectance Spectrum"):
        """
        Plot ROI reflectance spectrum

        Args:
            roi_results (dict): ROI analysis assets
            title (str): Plot title

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        wavelengths = roi_results['wavelengths']
        mean_spectrum = roi_results['mean_spectrum']
        std_spectrum = roi_results['std_spectrum']

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot mean spectrum with error bars
        ax.plot(wavelengths, mean_spectrum, 'b-', linewidth=2, label='Mean')
        ax.fill_between(wavelengths,
                       mean_spectrum - std_spectrum,
                       mean_spectrum + std_spectrum,
                       alpha=0.3, color='blue', label='±1 std')

        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add statistics text
        stats = roi_results['statistics']
        stats_text = f"Pixels: {roi_results['roi_pixels']:,}\n"
        stats_text += f"Mean: {stats['mean']:.4f}\n"
        stats_text += f"SNR: {stats['snr']:.1f}"

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))

        return fig

    def plot_comparison(self, comparison_results, title="Reference vs Sample Comparison"):
        """
        Plot comparison between reference and sample spectra

        Args:
            comparison_results (dict): Comparison analysis assets
            title (str): Plot title

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        wavelengths = comparison_results['wavelengths']
        abs_diff = comparison_results['absolute_difference']
        rel_diff = comparison_results['relative_difference_percent']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Absolute difference
        ax1.plot(wavelengths, abs_diff, 'r-', linewidth=2)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Absolute Difference')
        ax1.set_title('Absolute Difference (Sample - Reference)')
        ax1.grid(True, alpha=0.3)

        # Relative difference
        ax2.plot(wavelengths, rel_diff, 'g-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Relative Difference (%)')
        ax2.set_title('Relative Difference (%)')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        # Add statistics
        stats = comparison_results['statistics']
        stats_text = f"Mean diff: {stats['mean_absolute_diff']:.4f}\n"
        stats_text += f"Rel. change: {stats['mean_relative_diff_percent']:+.2f}%"

        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                verticalalignment='top', bbox=dict(boxstyle="round", facecolor='lightcoral', alpha=0.8))

        return fig

    def create_projection(self, reflectance_cube, wavelengths, wl_range=(450, 800)):
        """
        Create a normalized projection from reflectance cube by averaging over wavelength range

        Args:
            reflectance_cube (np.ndarray): Reflectance cube (H x W x bands)
            wavelengths (np.ndarray): Wavelength array corresponding to bands
            wl_range (tuple): Wavelength range for projection (min, max)

        Returns:
            np.ndarray: Normalized projection image (H x W)
        """
        print(f"Detecting Creating projection from wavelength range {wl_range[0]}-{wl_range[1]}nm...")

        # Create mask for wavelength range
        mask = (wavelengths >= wl_range[0]) & (wavelengths <= wl_range[1])

        if not np.any(mask):
            print(f"ERROR: No wavelengths found in range {wl_range}")
            return None

        # Average over selected wavelengths
        proj = np.mean(reflectance_cube[:, :, mask], axis=2)

        # Normalize to 0-1 range
        proj_norm = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)

        num_bands_used = np.sum(mask)
        wl_used = wavelengths[mask]

        print(f"SUCCESS: Projection created using {num_bands_used} bands ({wl_used[0]:.0f}-{wl_used[-1]:.0f}nm)")
        print(f"   Projection range: {proj.min():.4f} - {proj.max():.4f} (raw)")
        print(f"   Normalized range: {proj_norm.min():.4f} - {proj_norm.max():.4f}")

        return proj_norm
