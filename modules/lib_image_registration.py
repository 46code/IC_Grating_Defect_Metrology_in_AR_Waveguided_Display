#!/usr/bin/env python3
"""
Image Registration Library
Handles homography computation and spatial alignment of hyperspectral cubes
"""

import numpy as np
import cv2

class ImageRegistration:
    """Library for image registration using homography"""

    def __init__(self):
        """Initialize the image registration"""
        self.homography_matrix = None
        self.registration_error = None

    def compute_homography(self, source_points, target_points):
        """
        Compute homography matrix from corresponding points

        Args:
            source_points (list): List of (x, y) points in source image
            target_points (list): List of (x, y) points in target image

        Returns:
            np.ndarray: 3x3 homography matrix
        """
        print("🔄 Computing homography matrix...")

        if len(source_points) != len(target_points) or len(source_points) < 4:
            print(f"❌ Need at least 4 point pairs, got {len(source_points)}")
            return None

        # Convert to numpy arrays
        src_pts = np.array(source_points, dtype=np.float32)
        dst_pts = np.array(target_points, dtype=np.float32)

        # Compute homography
        self.homography_matrix, mask = cv2.findHomography(
            src_pts, dst_pts,
            cv2.RANSAC,
            ransacReprojThreshold=5.0
        )

        if self.homography_matrix is None:
            print("❌ Failed to compute homography")
            return None

        # Calculate reprojection error
        reprojected = cv2.perspectiveTransform(
            src_pts.reshape(-1, 1, 2),
            self.homography_matrix
        ).reshape(-1, 2)

        errors = np.sqrt(np.sum((dst_pts - reprojected) ** 2, axis=1))
        self.registration_error = np.mean(errors)

        print(f"✅ Homography computed")
        print(f"   Mean reprojection error: {self.registration_error:.2f} pixels")
        print(f"   Individual errors: {[f'{e:.2f}' for e in errors]}")

        return self.homography_matrix

    def register_image(self, image, homography_matrix=None):
        """
        Apply homography to register a single image

        Args:
            image (np.ndarray): Input image to register
            homography_matrix (np.ndarray): Optional homography matrix

        Returns:
            np.ndarray: Registered image
        """
        if homography_matrix is None:
            homography_matrix = self.homography_matrix

        if homography_matrix is None:
            print("❌ No homography matrix available")
            return None

        height, width = image.shape
        registered = cv2.warpPerspective(
            image,
            homography_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        return registered

    def register_cube(self, cube, homography_matrix=None, progress_callback=None):
        """
        Apply homography to register entire spectral cube

        Args:
            cube (np.ndarray): Input spectral cube (height, width, bands)
            homography_matrix (np.ndarray): Optional homography matrix
            progress_callback (callable): Optional progress callback function

        Returns:
            np.ndarray: Registered spectral cube
        """
        print("🔄 Registering spectral cube...")

        if homography_matrix is None:
            homography_matrix = self.homography_matrix

        if homography_matrix is None:
            print("❌ No homography matrix available")
            return None

        height, width, num_bands = cube.shape
        registered_cube = np.zeros_like(cube)

        print(f"   Processing {num_bands} wavelength bands...")

        for band_idx in range(num_bands):
            if progress_callback:
                progress_callback(band_idx, num_bands)
            elif band_idx % 10 == 0:
                print(f"   Band {band_idx+1}/{num_bands}")

            registered_cube[:, :, band_idx] = cv2.warpPerspective(
                cube[:, :, band_idx],
                homography_matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        print("✅ Cube registration complete")
        return registered_cube

    def create_roi_mask(self, image_shape, circle_params):
        """
        Create circular ROI mask from circle parameters

        Args:
            image_shape (tuple): (height, width) of target image
            circle_params (dict): Circle parameters with 'center' and 'radius'

        Returns:
            np.ndarray: Binary mask (uint8, 255=ROI, 0=background)
        """
        print("🎯 Creating circular ROI mask...")

        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)

        center = circle_params['center']
        radius = circle_params['radius']

        # Create circular mask
        cv2.circle(mask, center, radius, 255, -1)

        # Statistics
        roi_pixels = np.sum(mask > 0)
        total_pixels = height * width
        coverage = (roi_pixels / total_pixels) * 100

        print(f"✅ ROI mask created:")
        print(f"   Center: {center}, Radius: {radius}")
        print(f"   ROI pixels: {roi_pixels:,} ({coverage:.2f}% coverage)")

        return mask

    def get_registration_quality(self):
        """
        Get registration quality metrics

        Returns:
            dict: Quality metrics
        """
        if self.registration_error is None:
            return None

        quality = "Excellent" if self.registration_error < 1.0 else \
                 "Good" if self.registration_error < 2.0 else \
                 "Fair" if self.registration_error < 5.0 else "Poor"

        return {
            'mean_error': self.registration_error,
            'quality': quality,
            'matrix': self.homography_matrix.tolist() if self.homography_matrix is not None else None
        }
