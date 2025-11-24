#!/usr/bin/env python3
"""
Feature Detection Library
Handles fiducial point and circle detection in hyperspectral images
"""

import numpy as np
import cv2

class FeatureDetector:
    """Library for detecting fiducials and circles in hyperspectral images"""

    def __init__(self):
        """Initialize the feature detector"""
        self.fiducial_params = {
            'min_area': 50,
            'max_area': 500,
            'circularity_thresh': 0.7,
            'cross_threshold': 0.3
        }

        self.circle_params = {
            'min_radius': 10,
            'max_radius': 100,
            'hough_param1': 25,
            'hough_param2': 30,
            'circle_threshold': 0.3
        }

    def detect_fiducials(self, image, num_fiducials=4, percentile=3):
        """
        Detect fiducial markers using EXACT algorithm from fiducial_detection_pipeline.py

        Args:
            image (np.ndarray): Input grayscale image
            num_fiducials (int): Expected number of fiducials
            percentile (float): Percentile threshold for binary thresholding

        Returns:
            tuple: (fiducials, binary_map) where fiducials is list of (x, y) coordinates
                   and binary_map is the thresholded binary image used for detection
        """
        print(f"🎯 Detecting {num_fiducials} fiducials...")

        # EXACT PARAMETERS FROM ORIGINAL PIPELINE:
        # Normalize and blur (using Gaussian, not bilateral)
        image_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(image_norm, (3, 3), 1)

        # Percentile thresholding
        binary = cv2.threshold(blurred, np.percentile(blurred, percentile), 255, cv2.THRESH_BINARY_INV)[1]

        # Find and filter candidates using exact original method
        candidates = self._find_cross_candidates(binary, image.shape)
        print(f"   Found {len(candidates)} initial candidates")

        if not candidates:
            print(f"   ❌ No candidates found")
            return [], binary

        # Select best 4 fiducials using exact original method
        selected = self._select_best_fiducials(candidates, image.shape)

        if selected:
            print(f"   ✅ Selected {len(selected)} fiducials")
        else:
            print(f"   ❌ No fiducials selected")

        return selected, binary

    def _find_cross_candidates(self, binary_image, image_shape):
        """EXACT copy from original fiducial_detection_pipeline.py"""
        # Clean up binary image with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_clean = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for contour in contours:
            # Filter by area (crosses should be reasonably sized) - EXACT RANGE
            area = cv2.contourArea(contour)
            if area < 30 or area > 300:  # EXACT: 30-300
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by aspect ratio (crosses should be roughly square) - EXACT RANGE
            aspect_ratio = w / h
            if aspect_ratio < 0.4 or aspect_ratio > 2.5:  # EXACT: 0.4-2.5
                continue

            # Extract the binary patch for this contour
            binary_patch = binary_clean[y:y+h, x:x+w]

            # Check if the shape could be a cross and get cross-shape score
            if self._is_potential_cross(binary_patch, area):
                center_x = x + w // 2
                center_y = y + h // 2

                # Calculate cross-shape score using EXACT algorithm
                cross_shape_score = self._calculate_cross_shape_score(binary_patch)

                # Calculate traditional features for comparison
                compactness = area / (w * h)
                extent = area / cv2.contourArea(cv2.convexHull(contour))

                candidates.append({
                    'position': (center_x, center_y),
                    'area': area,
                    'cross_shape_score': cross_shape_score,
                    'compactness': compactness,
                    'extent': extent,
                    'aspect_ratio': aspect_ratio
                })

        return candidates

    def _calculate_cross_shape_score(self, binary_patch):
        """EXACT copy from original fiducial_detection_pipeline.py"""
        if binary_patch.size == 0:
            return 0.0

        h, w = binary_patch.shape
        if h < 8 or w < 8:
            return 0.0

        center_y, center_x = h // 2, w // 2
        tolerance = max(1, min(h, w) // 6)

        # Vertical and horizontal line coverage
        vertical_region = binary_patch[:, max(0, center_x-tolerance):min(w, center_x+tolerance+1)]
        horizontal_region = binary_patch[max(0, center_y-tolerance):min(h, center_y+tolerance+1), :]

        vertical_coverage = np.sum(vertical_region > 0) / (h * vertical_region.shape[1])
        horizontal_coverage = np.sum(horizontal_region > 0) / (horizontal_region.shape[0] * w)

        # Center intersection and corner emptiness
        center_region = binary_patch[max(0, center_y-2):min(h, center_y+3), max(0, center_x-2):min(w, center_x+3)]
        center_score = np.sum(center_region > 0) / center_region.size

        corner_size = min(3, min(h, w) // 4)
        corners = [
            binary_patch[0:corner_size, 0:corner_size],
            binary_patch[0:corner_size, -corner_size:],
            binary_patch[-corner_size:, 0:corner_size],
            binary_patch[-corner_size:, -corner_size:]
        ]
        corner_emptiness = sum(1.0 - (np.sum(corner > 0) / corner.size) for corner in corners if corner.size > 0) / 4

        # EXACT combination weights from original
        cross_score = (
            vertical_coverage * 0.25 +      # 25% - vertical line presence
            horizontal_coverage * 0.25 +    # 25% - horizontal line presence
            center_score * 0.20 +           # 20% - center intersection
            corner_emptiness * 0.20 +       # 20% - empty corners
            0.10                            # 10% - line thickness similarity
        )

        return min(1.0, cross_score)

    def _is_potential_cross(self, binary_patch, area):
        """EXACT copy from original fiducial_detection_pipeline.py"""
        if binary_patch.size == 0:
            return False

        h, w = binary_patch.shape
        if h < 8 or w < 8:
            return False

        # Check center lines coverage
        center_y, center_x = h // 2, w // 2
        tolerance = max(1, min(h, w) // 6)  # Adaptive tolerance

        # Vertical line check (with tolerance)
        vertical_region = binary_patch[:, max(0, center_x-tolerance):min(w, center_x+tolerance+1)]
        vertical_coverage = np.sum(vertical_region > 0) / (h * vertical_region.shape[1])

        # Horizontal line check (with tolerance)
        horizontal_region = binary_patch[max(0, center_y-tolerance):min(h, center_y+tolerance+1), :]
        horizontal_coverage = np.sum(horizontal_region > 0) / (horizontal_region.shape[0] * w)

        # Cross should have good coverage in both directions - EXACT THRESHOLD
        min_coverage = 0.4  # EXACT: 0.4
        cross_like = vertical_coverage > min_coverage and horizontal_coverage > min_coverage

        # Additional check: center should be filled
        center_region = binary_patch[max(0, center_y-1):min(h, center_y+2),
                                   max(0, center_x-1):min(w, center_x+2)]
        center_filled = np.sum(center_region > 0) > (center_region.size * 0.5)

        return cross_like and center_filled

    def _select_best_fiducials(self, candidates, image_shape):
        """EXACT copy from original fiducial_detection_pipeline.py"""
        MIN_CROSS_THRESHOLD = 0.3  # EXACT: 0.3
        height, width = image_shape

        # Filter by cross-shape threshold
        valid_candidates = [c for c in candidates if c['cross_shape_score'] >= MIN_CROSS_THRESHOLD]
        print(f"   {len(valid_candidates)}/{len(candidates)} passed cross-shape threshold")

        if len(valid_candidates) < 4:
            return [(c['position'][0], c['position'][1]) for c in valid_candidates]

        # Define regions - EXACT coordinates from original
        regions = {
            'top_right': (0.55, 1.0, 0.0, 0.45),
            'top_left': (0.0, 0.45, 0.0, 0.45),
            'bottom_left': (0.0, 0.45, 0.55, 1.0),
            'bottom_right': (0.55, 1.0, 0.55, 1.0)
        }

        selected_fiducials = []
        used_candidates = set()

        # Select best candidate from each region
        for region_name, (x_min, x_max, y_min, y_max) in regions.items():
            region_candidates = []
            for i, candidate in enumerate(valid_candidates):
                if i in used_candidates:
                    continue
                x, y = candidate['position']
                norm_x, norm_y = x / width, y / height
                if x_min <= norm_x <= x_max and y_min <= norm_y <= y_max:
                    score = candidate['cross_shape_score']
                    region_candidates.append((i, candidate, score))

            if region_candidates:
                best_idx, best_candidate, _ = max(region_candidates, key=lambda x: x[2])
                selected_fiducials.append(best_candidate['position'])
                used_candidates.add(best_idx)

        # Fill remaining with best cross-shape scores
        if len(selected_fiducials) < 4:
            remaining = [(i, c) for i, c in enumerate(valid_candidates) if i not in used_candidates]
            remaining.sort(key=lambda x: x[1]['cross_shape_score'], reverse=True)
            for i, candidate in remaining[:4-len(selected_fiducials)]:
                selected_fiducials.append(candidate['position'])

        return selected_fiducials[:4]

    def detect_circle(self, image, crop_region=None):
        """
        Detect circular features in an image

        Args:
            image (np.ndarray): Input grayscale image
            crop_region (tuple): Optional (x_min, x_max, y_min, y_max) for cropping

        Returns:
            dict: Circle parameters {'center': (x, y), 'radius': r, 'score': s}
        """
        print("🎯 Detecting circular features...")

        # Crop if specified
        if crop_region:
            x_min, x_max, y_min, y_max = crop_region
            height, width = image.shape
            x_start = int(x_min * width)
            x_end = int(x_max * width)
            y_start = int(y_min * height)
            y_end = int(y_max * height)

            cropped = image[y_start:y_end, x_start:x_end]
            offset = (x_start, y_start)
        else:
            cropped = image
            offset = (0, 0)

        # Normalize and filter
        image_norm = ((cropped - cropped.min()) / (cropped.max() - cropped.min()) * 255).astype(np.uint8)
        blurred = cv2.bilateralFilter(image_norm, 9, 75, 75)

        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=self.circle_params['hough_param1'],
            param2=self.circle_params['hough_param2'],
            minRadius=self.circle_params['min_radius'],
            maxRadius=self.circle_params['max_radius']
        )

        if circles is None:
            print("❌ No circles detected")
            return None

        circles = np.round(circles[0, :]).astype("int")
        print(f"✅ Found {len(circles)} circle candidates")

        # Score circles and select best one
        best_circle = None
        best_score = 0

        for (x, y, radius) in circles:
            # Validate bounds
            if (x - radius < 0 or x + radius >= blurred.shape[1] or
                y - radius < 0 or y + radius >= blurred.shape[0]):
                continue

            score = self._score_circle(blurred, x, y, radius)
            print(f"  Circle at ({x},{y}) radius {radius}: score {score:.3f}")

            if score >= self.circle_params['circle_threshold'] and score > best_score:
                # Adjust coordinates back to original image
                global_x = x + offset[0] + 5 # small horizontal adjustment
                global_y = y + offset[1] # small vertical adjustment
                best_circle = {
                    'center': (global_x, global_y),
                    'radius': radius - 5,
                    'score': score
                }
                best_score = score

        if best_circle:
            print(f"✅ Best circle: center {best_circle['center']}, radius {best_circle['radius']}")
            return best_circle
        else:
            print("❌ No valid circles found")
            return None

    def _score_circle(self, image, cx, cy, radius):
        """Score how well a circle fits the image features"""
        # Get edge magnitude
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        # Sample points around circle
        num_samples = max(60, int(2 * np.pi * radius))
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)

        circle_x = cx + radius * np.cos(angles)
        circle_y = cy + radius * np.sin(angles)

        circle_x = np.clip(circle_x.astype(int), 0, image.shape[1] - 1)
        circle_y = np.clip(circle_y.astype(int), 0, image.shape[0] - 1)

        # Edge alignment score
        edge_values = edge_magnitude[circle_y, circle_x]
        edge_threshold = np.percentile(edge_magnitude.flatten(), 75)
        edge_alignment = np.sum(edge_values > edge_threshold) / len(edge_values)

        # Brightness contrast
        inner_mask = np.zeros_like(image, dtype=bool)
        cv2.circle(inner_mask.astype(np.uint8), (cx, cy), max(1, radius - 5), 1, -1)

        outer_mask = np.zeros_like(image, dtype=bool)
        cv2.circle(outer_mask.astype(np.uint8), (cx, cy), radius + 10, 1, -1)
        outer_mask = outer_mask & ~inner_mask

        if np.any(inner_mask) and np.any(outer_mask):
            inner_mean = np.mean(image[inner_mask])
            outer_mean = np.mean(image[outer_mask])
            brightness_contrast = max(0, min(1, (inner_mean - outer_mean) / 255.0))
        else:
            brightness_contrast = 0

        # Combined score
        return edge_alignment * 0.6 + brightness_contrast * 0.4
