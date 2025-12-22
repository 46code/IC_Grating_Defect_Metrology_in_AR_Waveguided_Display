#!/usr/bin/env python3
"""
Spectral Data Loader Library
Handles loading and management of hyperspectral data cubes
"""

import numpy as np
from pathlib import Path
import tifffile

class SpectralDataLoader:
    """Library for loading and managing hyperspectral data cubes"""

    def __init__(self, data_paths):
        """
        Initialize the spectral data loader

        Args:
            data_paths (dict): Dictionary containing paths for different datasets:
                - sample_white_path: Path to white reference data (for samples)
                - sample_dark_path: Path to Dark reference data (for samples)
                - reference_path: Path to reference data
                - reference_white_path: Path to white reference data (for reference)
                - reference_dark_path: Path to Dark reference data (for reference)
                - sample_path: Base path for sample data
        """
        # Process path variable substitution
        processed_paths = self._process_path_variables(data_paths)

        self.reference_path = Path(processed_paths['reference_path'])
        self.reference_white_path = Path(processed_paths['reference_white_path'])
        self.reference_dark_path = Path(processed_paths['reference_dark_path'])

        self.sample_white_path = Path(processed_paths['sample_white_path'])
        self.sample_dark_path = Path(processed_paths['sample_dark_path'])
        self.sample_path = Path(processed_paths['sample_path'])
        self.wavelengths = list(range(450, 951, 10))  # 450-950nm in 10nm steps

    def _process_path_variables(self, data_paths):
        """Process variable substitution in paths (e.g., ${base_path})"""
        processed = {}
        base_path = data_paths.get('base_path', '')

        for key, value in data_paths.items():
            if isinstance(value, str) and '${base_path}' in value:
                processed[key] = value.replace('${base_path}', base_path)
            else:
                processed[key] = value

        return processed

    def load_cube(self, dataset_name, sample_name=None):
        """
        Load a complete spectral cube for a given dataset

        Args:
            dataset_name (str): Type of dataset ('reference', 'white', 'Dark', 'sample')
            sample_name (str): Name of sample (only used for 'sample' dataset_name)

        Returns:
            np.ndarray: Spectral cube with shape (height, width, num_bands)
        """
        # Determine the correct path based on dataset type
        if dataset_name == 'reference':
            dataset_path = self.reference_path
        elif dataset_name == 'white':
            dataset_path = self.sample_white_path
        elif dataset_name == 'dark':
            dataset_path = self.sample_dark_path
        elif dataset_name == 'sample':
            if sample_name is None:
                print(f"ERROR: Sample name required for sample dataset")
                return None
            dataset_path = self.sample_path / sample_name
        else:
            print(f"ERROR: Unknown dataset type: {dataset_name}")
            return None

        if not dataset_path.exists():
            print(f"ERROR: Dataset path not found: {dataset_path}")
            return None

        print(f"Loading Loading {dataset_name} spectral cube from {dataset_path}...")

        # Get list of TIFF files
        tiff_files = sorted(list(dataset_path.glob("*.tif")))
        if not tiff_files:
            print(f"ERROR: No TIFF files found in {dataset_path}")
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

        print(f"SUCCESS: Loaded {dataset_name}: {cube.shape}")
        return cube

    def get_wavelength_band(self, cube, wavelength_nm):
        """
        Get specific wavelength band from cube

        Args:
            cube (np.ndarray): Spectral cube
            wavelength_nm (int): Desired wavelength in nm

        Returns:
            tuple: (band_image, band_index)
        """
        if wavelength_nm not in self.wavelengths:
            print(f"ERROR: Wavelength {wavelength_nm}nm not available")
            return None, None

        band_idx = self.wavelengths.index(wavelength_nm)
        if band_idx >= cube.shape[2]:
            print(f"ERROR: Band index {band_idx} out of range")
            return None, None

        return cube[:, :, band_idx], band_idx

    def load_reference_calibration_data(self):
        """
        Load white and Dark reference data specifically for reference reflectance computation

        Returns:
            tuple: (reference_white_cube, reference_dark_cube)
        """
        print(f"Loading Loading reference calibration data...")

        # Load reference white data
        if not self.reference_white_path.exists():
            print(f"ERROR: Reference white path not found: {self.reference_white_path}")
            return None, None

        print(f"Loading Loading reference white cube from {self.reference_white_path}...")
        tiff_files = sorted(list(self.reference_white_path.glob("*.tif")))
        if not tiff_files:
            print(f"ERROR: No TIFF files found in {self.reference_white_path}")
            return None, None

        # Load reference white cube
        first_img = tifffile.imread(tiff_files[0])
        height, width = first_img.shape
        num_bands = len(tiff_files)

        reference_white_cube = np.zeros((height, width, num_bands), dtype=first_img.dtype)
        for i, tiff_file in enumerate(tiff_files):
            reference_white_cube[:, :, i] = tifffile.imread(tiff_file)
        print(f"SUCCESS: Loaded reference white: {reference_white_cube.shape}")

        # Load reference Dark data
        if not self.reference_dark_path.exists():
            print(f"ERROR: Reference Dark path not found: {self.reference_dark_path}")
            return None, None

        print(f"Loading Loading reference Dark cube from {self.reference_dark_path}...")
        tiff_files = sorted(list(self.reference_dark_path.glob("*.tif")))
        if not tiff_files:
            print(f"ERROR: No TIFF files found in {self.reference_dark_path}")
            return None, None

        reference_dark_cube = np.zeros((height, width, num_bands), dtype=first_img.dtype)
        for i, tiff_file in enumerate(tiff_files):
            reference_dark_cube[:, :, i] = tifffile.imread(tiff_file)
        print(f"SUCCESS: Loaded reference Dark: {reference_dark_cube.shape}")

        return reference_white_cube, reference_dark_cube

    def load_all_datasets(self, sample_name):
        """
        Load all required datasets for analysis

        Args:
            sample_name (str): Name of the sample dataset

        Returns:
            dict: Dictionary containing all loaded cubes
        """
        datasets = {}

        # Load all required datasets using new path structure
        dataset_configs = [
            ('reference', 'reference'),
            ('white', 'white'),
            ('dark', 'dark'),
            ('sample', sample_name)
        ]

        for dataset_type, dataset_key in dataset_configs:
            if dataset_type == 'sample':
                cube = self.load_cube('sample', sample_name)
            else:
                cube = self.load_cube(dataset_type)

            if cube is None:
                print(f"ERROR: Failed to load {dataset_type}")
                return None
            datasets[dataset_key] = cube

        print(f"SUCCESS: All datasets loaded successfully")
        return datasets
