#!/usr/bin/env python3
"""
Spectral Data Loader Library
Handles loading and management of hyperspectral data cubes
"""

import numpy as np
import os
from pathlib import Path
import tifffile

class SpectralDataLoader:
    """Library for loading and managing hyperspectral data cubes"""

    def __init__(self, data_path):
        """
        Initialize the spectral data loader

        Args:
            data_path (str): Path to the directory containing spectral data
        """
        self.data_path = Path(data_path)
        self.wavelengths = list(range(450, 951, 10))  # 450-950nm in 10nm steps

    def load_cube(self, dataset_name):
        """
        Load a complete spectral cube for a given dataset

        Args:
            dataset_name (str): Name of dataset ('reference', 'white', 'Darkreference', etc.)

        Returns:
            np.ndarray: Spectral cube with shape (height, width, num_bands)
        """
        dataset_path = self.data_path / dataset_name
        if not dataset_path.exists():
            print(f"❌ Dataset path not found: {dataset_path}")
            return None

        print(f"📁 Loading {dataset_name} spectral cube...")

        # Get list of TIFF files
        tiff_files = sorted(list(dataset_path.glob("*.tif")))
        if not tiff_files:
            print(f"❌ No TIFF files found in {dataset_path}")
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

        print(f"✅ Loaded {dataset_name}: {cube.shape}")
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
            print(f"❌ Wavelength {wavelength_nm}nm not available")
            return None, None

        band_idx = self.wavelengths.index(wavelength_nm)
        if band_idx >= cube.shape[2]:
            print(f"❌ Band index {band_idx} out of range")
            return None, None

        return cube[:, :, band_idx], band_idx

    def load_all_datasets(self, sample_name):
        """
        Load all required datasets for analysis

        Args:
            sample_name (str): Name of the sample dataset

        Returns:
            dict: Dictionary containing all loaded cubes
        """
        datasets = {}

        # Load all required datasets
        for name in ['reference', 'white', 'Darkreference', sample_name]:
            cube = self.load_cube(name)
            if cube is None:
                print(f"❌ Failed to load {name}")
                return None
            datasets[name] = cube

        print(f"✅ All datasets loaded successfully")
        return datasets
