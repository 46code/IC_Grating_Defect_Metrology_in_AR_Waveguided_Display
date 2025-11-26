#!/usr/bin/env python3
"""
Plotting Library for Hyperspectral Analysis
Handles all visualization functions for the analysis pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
import os

class HyperspectralPlotter:
    """
    Plotting utilities for hyperspectral analysis pipeline
    """

    def __init__(self, config):
        """Initialize plotter with configuration"""
        self.config = config
        self.padding_factor = config['visualization_parameters']['padding_factor']
        self.inner_radius_fraction = config['visualization_parameters']['inner_radius_fraction']
        self.outer_radius_fraction = config['visualization_parameters']['outer_radius_fraction']

        # Configure matplotlib
        plt.style.use(config['plotting_settings']['style'])
        plt.rcParams['figure.dpi'] = config['plotting_settings']['figure_dpi']
        plt.rcParams['savefig.dpi'] = config['plotting_settings']['save_dpi']

    def plot_fiducials(self, ref_band, sample_band, ref_fiducials, sample_fiducials,
                      sample_name, wavelength, results_dir, ref_binary=None, sample_binary=None):
        """Plot fiducial detection assets including binary maps"""
        # Create subplot layout based on whether binary maps are provided
        if ref_binary is not None or sample_binary is not None:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            ax1, ax2, ax3, ax4 = axes.flatten()
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax3 = ax4 = None

        # Reference with fiducials
        ax1.imshow(ref_band, cmap='gray')
        for i, (x, y) in enumerate(ref_fiducials):
            ax1.plot(x, y, 'bo', markersize=10)
            ax1.text(x+15, y-15, f'R{i+1}', color='blue', fontsize=12, fontweight='bold')
        ax1.set_title(f'Reference Fiducials at {wavelength}nm', fontweight='bold')
        ax1.axis('off')

        # Sample with fiducials
        ax2.imshow(sample_band, cmap='gray')
        for i, (x, y) in enumerate(sample_fiducials):
            ax2.plot(x, y, 'ro', markersize=10)
            ax2.text(x+15, y-15, f'S{i+1}', color='red', fontsize=12, fontweight='bold')
        ax2.set_title(f'{sample_name} Fiducials at {wavelength}nm', fontweight='bold')
        ax2.axis('off')

        # Add binary maps if provided and axes exist
        if ax3 is not None and ax4 is not None:
            # Reference binary map
            if ref_binary is not None:
                ax3.imshow(ref_binary, cmap='gray')
                for i, (x, y) in enumerate(ref_fiducials):
                    ax3.plot(x, y, 'co', markersize=8)
                    ax3.text(x+10, y-10, f'R{i+1}', color='cyan', fontsize=10, fontweight='bold')
                ax3.set_title(f'Reference Binary Map at {wavelength}nm', fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'No Reference Binary Map', ha='center', va='center',
                        transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Reference Binary Map (Not Available)', fontweight='bold')
            ax3.axis('off')

            # Sample binary map
            if sample_binary is not None:
                ax4.imshow(sample_binary, cmap='gray')
                for i, (x, y) in enumerate(sample_fiducials):
                    ax4.plot(x, y, 'mo', markersize=8)
                    ax4.text(x+10, y-10, f'S{i+1}', color='magenta', fontsize=10, fontweight='bold')
                ax4.set_title(f'{sample_name} Binary Map at {wavelength}nm', fontweight='bold')
            else:
                ax4.text(0.5, 0.5, 'No Sample Binary Map', ha='center', va='center',
                        transform=ax4.transAxes, fontsize=12)
                ax4.set_title(f'{sample_name} Binary Map (Not Available)', fontweight='bold')
            ax4.axis('off')

        plt.tight_layout()
        plt.savefig(f'{results_dir}/01_fiducial_detection.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_circle_detection(self, ref_band, ref_fiducials, center, radius,
                            wavelength, results_dir, crop_region=None):
        """Plot circle detection results showing only the cropped region"""
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # Apply cropping if specified
        if crop_region:
            x_min, x_max, y_min, y_max = crop_region
            height, width = ref_band.shape
            x_start = int(x_min * width)
            x_end = int(x_max * width)
            y_start = int(y_min * height)
            y_end = int(y_max * height)
            
            # Crop the image
            cropped_image = ref_band[y_start:y_end, x_start:x_end]
            
            # Adjust coordinates to cropped image space
            adjusted_center = (center[0] - x_start, center[1] - y_start)
            
            # Filter fiducials to only show those within the crop region
            cropped_fiducials = []
            for i, (x, y) in enumerate(ref_fiducials):
                if x_start <= x <= x_end and y_start <= y <= y_end:
                    cropped_fiducials.append((x - x_start, y - y_start, i))
            
            display_image = cropped_image
            display_center = adjusted_center
            display_fiducials = cropped_fiducials
            title_suffix = f"(Cropped Region)"
        else:
            display_image = ref_band
            display_center = center
            display_fiducials = [(x, y, i) for i, (x, y) in enumerate(ref_fiducials)]
            title_suffix = ""

        ax.imshow(display_image, cmap='gray')

        # Draw fiducials that are within the displayed region
        for x, y, orig_idx in display_fiducials:
            ax.plot(x, y, 'bo', markersize=8)
            ax.text(x+15, y-15, f'R{orig_idx+1}', color='blue', fontsize=10, fontweight='bold')

        # Draw IC circle
        circle_patch = plt.Circle(display_center, radius, fill=False, color='lime', linewidth=1)
        ax.add_patch(circle_patch)
        ax.plot(display_center[0], display_center[1], '+', color='lime', markersize=15, markeredgewidth=3)

        ax.set_title(f'IC Circle Detection at {wavelength}nm {title_suffix}', fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{results_dir}/02_circle_detection.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_registration(self, ref_band, registered_band, ref_fiducials, results_dir):
        """Plot registration assets"""
        diff = np.abs(ref_band.astype(np.float32) - registered_band.astype(np.float32))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Registered sample
        ax1.imshow(registered_band, cmap='gray')
        for i, (x, y) in enumerate(ref_fiducials):
            ax1.plot(x, y, 'go', markersize=8)
            ax1.text(x + 10, y - 10, f'R{i+1}', color='lime', fontweight='bold')
        ax1.set_title('Registered Sample', fontweight='bold')
        ax1.axis('off')

        # Difference map
        im = ax2.imshow(diff, cmap='hot')
        ax2.set_title('Difference Map (|Ref - Sample|)', fontweight='bold')
        ax2.axis('off')
        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Absolute Intensity Difference', rotation=270, labelpad=15)

        plt.suptitle('Homography Registration Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/03_homography.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_reflectance(self, ref_band, registered_band, roi_mask,
                        reference_reflectance, sample_reflectance, band_idx,
                        sample_name, wavelength, results_dir):
        """Plot reflectance analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Reference with ROI
        ax1.imshow(ref_band, cmap='gray')
        roi_overlay = np.ma.masked_where(roi_mask == 0, roi_mask)
        ax1.imshow(roi_overlay, cmap='viridis', alpha=0.5)
        ax1.set_title('Reference with IC ROI', fontweight='bold')
        ax1.axis('off')

        # Registered sample with ROI
        ax2.imshow(registered_band, cmap='gray')
        ax2.imshow(roi_overlay, cmap='viridis', alpha=0.5)
        ax2.set_title(f'Registered {sample_name} with IC ROI', fontweight='bold')
        ax2.axis('off')

        # Reference reflectance (ROI only)
        ref_refl_band = reference_reflectance[:, :, band_idx]
        ref_display = np.where(roi_mask, ref_refl_band, np.nan)
        im3 = ax3.imshow(ref_display, cmap='viridis', vmin=0, vmax=1)
        ax3.set_title('Reference Reflectance (ROI)', fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # Sample reflectance (ROI only)
        sample_refl_band = sample_reflectance[:, :, band_idx]
        sample_display = np.where(roi_mask, sample_refl_band, np.nan)
        im4 = ax4.imshow(sample_display, cmap='viridis', vmin=0, vmax=1)
        ax4.set_title(f'{sample_name} Reflectance (ROI)', fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        plt.suptitle(f'IC ROI Reflectance Analysis at {wavelength}nm', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/04_reflectance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_analysis_maps(self, rmse_map, sam_map, roi_mask, center,
                          rmse_results, sam_results, ring_results, sample_name, results_dir):
        """Plot RMSE and SAM analysis maps"""
        # Calculate zoom bounds
        roi_coords = np.where(roi_mask)
        roi_distances = np.sqrt((roi_coords[1] - center[0])**2 + (roi_coords[0] - center[1])**2)
        max_radius = np.max(roi_distances)

        padded_radius = int(max_radius * self.padding_factor)
        x_min = max(0, int(center[0] - padded_radius))
        x_max = min(rmse_map.shape[1], int(center[0] + padded_radius))
        y_min = max(0, int(center[1] - padded_radius))
        y_max = min(rmse_map.shape[0], int(center[1] + padded_radius))

        # Create cropped arrays
        rmse_crop = rmse_map[y_min:y_max, x_min:x_max]
        sam_crop = sam_map[y_min:y_max, x_min:x_max]
        roi_crop = roi_mask[y_min:y_max, x_min:x_max]

        # Ring boundaries
        inner_radius = max_radius * self.inner_radius_fraction
        outer_radius_start = max_radius * self.outer_radius_fraction

        fig_maps, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot RMSE map
        ax1 = axes[0]
        rmse_masked = np.where(roi_crop, rmse_crop, np.nan)
        im1 = ax1.imshow(rmse_masked, cmap='hot', alpha=0.8)
        ax1.set_title(f'RMSE Map\nOverall: {rmse_results["overall_rmse"]:.4f}, Per-Pixel Mean: {rmse_results["rmse_per_pixel_mean"]:.4f}, Per-Pixel P95: {rmse_results["rmse_per_pixel_p95"]:.4f}',
                      fontweight='bold', fontsize=14)

        # Add circles and rings
        circle_center_crop = (center[0] - x_min, center[1] - y_min)
        self._add_circles_and_rings(ax1, circle_center_crop, max_radius, inner_radius, outer_radius_start)

        ax1.axis('off')
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('RMSE', fontsize=12)

        # Plot SAM map
        ax2 = axes[1]
        sam_masked = np.where(roi_crop, sam_crop, np.nan)
        im2 = ax2.imshow(sam_masked, cmap='viridis', alpha=0.8)

        # Use actual SAM mean from the analysis results
        ax2.set_title(f'SAM Map\nSAM Mean: {sam_results["sam_mean"]:.4f} rad, Ring Delta: {ring_results["delta_ring"]:.4f} rad',
                      fontweight='bold', fontsize=14)

        self._add_circles_and_rings(ax2, circle_center_crop, max_radius, inner_radius, outer_radius_start)

        ax2.axis('off')
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('SAM (radians)', fontsize=12)

        fig_maps.suptitle(f'Hyperspectral Analysis Maps - {sample_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/05_analysis_maps.png', dpi=300, bbox_inches='tight')
        plt.close()

        return x_min, y_min, max_radius  # Return for uniformity plot

    def plot_uniformity_analysis(self, sam_map, roi_mask, center, uniformity_results,
                                ring_results, sample_name, results_dir, crop_params):
        """Plot uniformity analysis with sector divisions and polar chart"""
        x_min, y_min, max_radius = crop_params

        # Create cropped arrays
        sam_crop = sam_map[y_min:y_min + int(max_radius * self.padding_factor * 2),
                          x_min:x_min + int(max_radius * self.padding_factor * 2)]
        roi_crop = roi_mask[y_min:y_min + int(max_radius * self.padding_factor * 2),
                          x_min:x_min + int(max_radius * self.padding_factor * 2)]

        uniformity_score = uniformity_results['U']
        sector_medians = uniformity_results['sector_meds']
        num_sectors = len(sector_medians)

        fig_sectors = plt.figure(figsize=(18, 7))

        # Left subplot: SAM map with sector divisions
        ax1 = plt.subplot(1, 2, 1)
        sam_masked = np.where(roi_crop, sam_crop, np.nan)
        im = ax1.imshow(sam_masked, cmap='viridis', alpha=0.9)

        # Draw sector lines and labels
        circle_center_crop = (center[0] - x_min, center[1] - y_min)
        for i in range(num_sectors):
            angle = i * 2 * np.pi / num_sectors
            x_end = circle_center_crop[0] + max_radius * np.cos(angle)
            y_end = circle_center_crop[1] + max_radius * np.sin(angle)
            ax1.plot([circle_center_crop[0], x_end], [circle_center_crop[1], y_end],
                     'white', linewidth=1.5, alpha=0.8, linestyle='--')

            # Add sector labels
            mid_angle = angle + np.pi / num_sectors
            label_radius = max_radius * 0.65
            label_x = circle_center_crop[0] + label_radius * np.cos(mid_angle)
            label_y = circle_center_crop[1] + label_radius * np.sin(mid_angle)
            ax1.text(label_x, label_y, f'S{i+1}', color='white', fontsize=14,
                     fontweight='bold', ha='center', va='center',
                     bbox=dict(boxstyle='circle,pad=0.3', facecolor='black', alpha=0.7,
                              edgecolor='white', linewidth=2))

        # Add outer circle
        circle_patch = plt.Circle(circle_center_crop, max_radius, fill=False, color='white', linewidth=2)
        ax1.add_patch(circle_patch)

        ax1.set_title(f'SAM Map with Sector Divisions (S1-S{num_sectors})', fontweight='bold', fontsize=14)
        ax1.set_xlabel('X Position (pixels)', fontsize=12)
        ax1.set_ylabel('Y Position (pixels)', fontsize=12)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1, shrink=0.85, pad=0.02)
        cbar.set_label('SAM (radians)', fontsize=11)

        # Right subplot: Polar pie chart with sector uniformity
        ax2 = plt.subplot(1, 2, 2, projection='polar')

        if len(sector_medians) > 0:
            theta_edges = np.linspace(0, 2*np.pi, num_sectors + 1)
            theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2

            # Normalize sector values for color mapping and radial extent
            vmin, vmax = sector_medians.min(), sector_medians.max()
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.RdYlGn_r

            radial_values = 0.7 + 0.2 * (sector_medians - vmin) / (vmax - vmin + 1e-10)

            # Draw pie sectors with variable radii
            width = 2 * np.pi / num_sectors
            for i, (theta, value, radius) in enumerate(zip(theta_centers, sector_medians, radial_values)):
                color = cmap(norm(value))
                ax2.bar(theta, radius, width=width, bottom=0, color=color,
                        edgecolor='black', linewidth=2, alpha=0.8)

            # Add sector labels with values
            for i, (theta, value, radius) in enumerate(zip(theta_centers, sector_medians, radial_values)):
                label_radius = radius * 0.6
                label_text = f'S{i+1}\n{value:.4f}'
                ax2.text(theta, label_radius, label_text, ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor='black', linewidth=1.5, alpha=0.9))

            # Configure polar plot
            ax2.set_ylim(0, 1.0)
            ax2.set_theta_direction(-1)
            ax2.set_theta_zero_location('E')

            # Add radial grid
            radial_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
            ax2.set_yticks(radial_ticks)
            ax2.set_yticklabels([f'{t:.2f}' for t in radial_ticks], fontsize=9)
            ax2.grid(True, alpha=0.3, linestyle='--')

            # Set angle labels
            angle_labels = ['0°', '315°', '270°', '225°', '180°', '135°', '90°', '45°']
            ax2.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
            ax2.set_xticklabels(angle_labels, fontsize=11)

            # Calculate and display statistics
            sector_cv = (np.std(sector_medians) / np.mean(sector_medians)) * 100
            sector_range = sector_medians.max() - sector_medians.min()

            stats_text = (f'Sector Uniformity Analysis\n'
                         f'Index: {uniformity_score:.3f} | '
                         f'CV: {sector_cv:.2f}% | '
                         f'Range: {sector_range:.4f}')

            ax2.text(0.5, 1.15, stats_text, transform=ax2.transAxes,
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue',
                             edgecolor='black', linewidth=2, alpha=0.9))

        fig_sectors.suptitle(f'Uniformity Analysis - {sample_name}',
                            fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/06_uniformity.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _add_circles_and_rings(self, ax, center_crop, max_radius, inner_radius, outer_radius):
        """Helper method to add circles and rings to plots"""
        # Main circle
        circle_patch = plt.Circle(center_crop, max_radius, fill=False, color='cyan', linewidth=2, linestyle='--')
        ax.add_patch(circle_patch)

        # Inner and outer rings
        inner_circle = plt.Circle(center_crop, inner_radius, fill=False, color='lime', linewidth=1.5, linestyle=':')
        outer_circle = plt.Circle(center_crop, outer_radius, fill=False, color='orange', linewidth=1.5, linestyle=':')
        ax.add_patch(inner_circle)
        ax.add_patch(outer_circle)
