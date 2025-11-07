#!/usr/bin/env python3
"""
Optimized Hyperspectral Analysis Library
Provides raw metrics for RMSE, SAM, uniformity, and ring analysis

Simplified version - no scoring, weighting, or normalization
Returns raw measurements only
"""

import numpy as np

class HyperspectralAnalyzer:
    """
    Simplified hyperspectral analysis for defect detection
    Returns raw metrics without scoring or weighting
    """

    def __init__(self):
        """Initialize the analyzer"""
        self.eps = 1e-8

    def calculate_rmse_metrics(self, sample_reflectance, reference_reflectance, roi_mask):
        """
        Calculate RMSE metrics between sample and reference reflectance

        Args:
            sample_reflectance: Sample reflectance cube
            reference_reflectance: Reference reflectance cube
            roi_mask: ROI mask (binary)

        Returns:
            dict: RMSE metrics (overall_rmse, rmse_per_pixel_mean, rmse_map)
        """
        if reference_reflectance is None or sample_reflectance is None:
            return None

        # Ensure same shape
        if reference_reflectance.shape != sample_reflectance.shape:
            print(f"   ⚠️  Shape mismatch: ref {reference_reflectance.shape} vs sample {sample_reflectance.shape}")
            return None

        # Use ROI mask (binary mask)
        roi_binary = roi_mask > 0

        # Extract ROI pixels
        ref_roi = reference_reflectance[roi_binary]  # Shape: (num_pixels, num_bands)
        sample_roi = sample_reflectance[roi_binary]

        # Calculate RMSE metrics
        rmse_per_pixel = np.sqrt(np.mean((ref_roi - sample_roi)**2, axis=1))
        overall_rmse = np.sqrt(np.mean((ref_roi - sample_roi)**2))
        rmse_per_pixel_mean = np.mean(rmse_per_pixel)

        # Create per-pixel RMSE map
        rmse_map = np.zeros(roi_mask.shape)
        rmse_map[roi_binary] = rmse_per_pixel

        return {
            'overall_rmse': float(overall_rmse),
            'rmse_per_pixel_mean': float(rmse_per_pixel_mean),
            'rmse_map': rmse_map
        }

    def calculate_sam_metrics(self, sample_reflectance, reference_reflectance, roi_mask):
        """
        Calculate SAM (Spectral Angle Mapper) metrics

        Args:
            sample_reflectance: Sample reflectance cube
            reference_reflectance: Reference reflectance cube
            roi_mask: ROI mask (binary)

        Returns:
            dict: SAM metrics (sam_map, sam_median, sam_mean)
        """
        if reference_reflectance is None or sample_reflectance is None:
            return None

        # Ensure same shape
        if reference_reflectance.shape != sample_reflectance.shape:
            return None

        roi_binary = roi_mask > 0
        sam_map = np.zeros(roi_mask.shape)

        # Calculate SAM for each pixel in ROI
        for i in range(roi_mask.shape[0]):
            for j in range(roi_mask.shape[1]):
                if roi_binary[i, j]:
                    ref_spectrum = reference_reflectance[i, j, :]
                    sample_spectrum = sample_reflectance[i, j, :]

                    # Calculate spectral angle
                    dot_product = np.dot(ref_spectrum, sample_spectrum)
                    norms = np.linalg.norm(ref_spectrum) * np.linalg.norm(sample_spectrum)

                    if norms > self.eps:
                        cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
                        sam_angle = np.arccos(cos_angle)
                        sam_map[i, j] = sam_angle

        # Calculate statistics for ROI pixels
        roi_sam_values = sam_map[roi_binary]
        sam_median = float(np.median(roi_sam_values))
        sam_mean = float(np.mean(roi_sam_values))

        return {
            'sam_map': sam_map,
            'sam_median': sam_median,
            'sam_mean': sam_mean
        }

    def calculate_ring_metrics(self, metric_map, center, roi_mask, inner_frac=0.2, outer_frac=0.8):
        """
        Calculate ring delta metrics (inner vs outer ring comparison)

        Args:
            metric_map: Metric map (e.g., SAM map)
            center: Circle center (x0, y0)
            roi_mask: ROI mask
            inner_frac: Inner ring fraction (default 0.7 = 70%)
            outer_frac: Outer ring fraction (default 0.85 = 85%)

        Returns:
            dict: Ring metrics (delta_ring, inner_mean, outer_mean)
        """
        roi_coords = np.where(roi_mask)
        if len(roi_coords[0]) == 0:
            return None

        roi_distances = np.sqrt((roi_coords[1] - center[0])**2 + (roi_coords[0] - center[1])**2)
        max_radius = np.max(roi_distances)

        inner_radius = max_radius * inner_frac
        outer_radius_start = max_radius * outer_frac

        # Extract ring values
        inner_mask = roi_distances <= inner_radius
        outer_mask = roi_distances >= outer_radius_start

        roi_values = metric_map[roi_coords]
        inner_values = roi_values[inner_mask]
        outer_values = roi_values[outer_mask]

        if len(inner_values) == 0 or len(outer_values) == 0:
            return None

        inner_mean = float(np.mean(inner_values))
        outer_mean = float(np.mean(outer_values))
        delta_ring = abs(outer_mean - inner_mean)

        return {
            'delta_ring': float(delta_ring),
            'inner_mean': inner_mean,
            'outer_mean': outer_mean
        }

    def calculate_uniformity_metrics(self, metric_map, center, roi_mask, num_sectors=8, min_pix=50):
        """
        Calculate sector uniformity metrics

        Args:
            metric_map: Metric map (e.g., SAM map)
            center: Circle center (x0, y0)
            roi_mask: ROI mask
            num_sectors: Number of sectors (default 8)
            min_pix: Minimum pixels per sector

        Returns:
            dict: Uniformity metrics (U, sector_meds)
        """
        roi_coords = np.where(roi_mask)
        if len(roi_coords[0]) == 0:
            return None

        ys, xs = roi_coords

        # Calculate angles in [0, 2π)
        ang = (np.arctan2(ys - center[1], xs - center[0]) + 2 * np.pi) % (2 * np.pi)
        bins = np.linspace(0.0, 2 * np.pi, num_sectors + 1)

        vals = metric_map[ys, xs]
        sector_meds = []

        for k in range(num_sectors):
            sel = (ang >= bins[k]) & (ang < bins[k + 1])
            if np.count_nonzero(sel) >= min_pix:
                sector_meds.append(np.median(vals[sel]))

        sector_meds = np.asarray(sector_meds, dtype=float)

        # If too few sectors survived, return neutral
        if sector_meds.size < max(3, num_sectors // 2):
            return {'U': 1.0, 'sector_meds': np.zeros(num_sectors)}

        med = np.median(sector_meds)
        mad = np.median(np.abs(sector_meds - med))

        # Robust coefficient of variation & range ratio
        rCV = mad / (med + self.eps)
        RR = (sector_meds.max() - sector_meds.min()) / (med + self.eps)

        # Normalize to [0,1] with thresholds
        T_rCV, T_RR = 0.30, 1.40
        rCVn = float(np.clip(rCV / T_rCV, 0.0, 1.0))
        RRn = float(np.clip(RR / T_RR, 0.0, 1.0))

        # Blend (α for rCV, (1-α) for RR)
        alpha = 0.6
        U = 1.0 - (alpha * rCVn + (1.0 - alpha) * RRn)
        U = float(np.clip(U, 0.0, 1.0))

        return {
            'U': U,
            'sector_meds': sector_meds
        }

    def analyze_sample(self, sample_reflectance, reference_reflectance, center, roi_mask, num_sectors=8):
        """
        Complete analysis returning raw metrics only

        Args:
            sample_reflectance: Sample reflectance cube
            reference_reflectance: Reference reflectance cube
            center: Circle center (x0, y0)
            roi_mask: ROI mask
            num_sectors: Number of sectors for uniformity analysis

        Returns:
            dict: Raw metrics organized by category
        """
        print("🔬 Starting hyperspectral analysis...")

        # Calculate RMSE metrics
        print("   Computing RMSE metrics...")
        rmse_results = self.calculate_rmse_metrics(sample_reflectance, reference_reflectance, roi_mask)

        # Calculate SAM metrics
        print("   Computing SAM metrics...")
        sam_results = self.calculate_sam_metrics(sample_reflectance, reference_reflectance, roi_mask)

        if sam_results is None:
            print("❌ SAM calculation failed")
            return None

        # Calculate ring metrics using SAM map
        print("   Computing ring delta metrics...")
        ring_results = self.calculate_ring_metrics(sam_results['sam_map'], center, roi_mask)

        # Calculate uniformity metrics using SAM map
        print("   Computing uniformity metrics...")
        uniformity_results = self.calculate_uniformity_metrics(sam_results['sam_map'], center, roi_mask, num_sectors)

        # Compile assets
        results = {
            'rmse': rmse_results,
            'sam': sam_results,
            'ring': ring_results,
            'uniformity': uniformity_results
        }

        print("✅ Hyperspectral analysis complete!")
        return results

