# IC Grating Defect Metrology in AR Waveguided Display

Hyperspectral analysis pipeline for detecting and quantifying defects in waveguide integrated circuit (IC) gratings used in augmented reality displays. This system processes hyperspectral TIFF cubes to compute reflectance, register samples to reference, and output quality metrics for defect detection.

---

## 1. Setup

### Environment and Libraries

```bash
# Clone or download the repository
cd IC_Grating_Defect_Metrology_in_AR_Waveguided_Display

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Make Python scripts executable (macOS/Linux)
chmod +x run_8_trials.py main.py generate_score_plots.py analysis_8_trials.py
```

**Note**: The `chmod +x` command makes the Python files executable, which is particularly important for `run_8_trials.py` as it calls `main.py` and `generate_score_plots.py` as subprocess commands.

**Required libraries** (installed via requirements.txt):
- numpy, pandas - Data processing
- opencv-python - Image processing and registration
- matplotlib, seaborn - Visualization
- tifffile - Hyperspectral TIFF loading
- scikit-image - Image analysis
- scipy - Scientific computing

### Data Folder Structure

Your data should be organized as follows:
```
Gage R&R/
  <operator>/                # e.g., KhangT1, KiduT2, LuelT1, AnirbanT2
    Reference/ (or reference/)  # Reference device hyperspectral cubes
    White/                   # White calibration cubes  
    Dark/                    # Dark calibration cubes
    Sample01/                # Sample folders
    Sample02/
    Sample04/
    ...
```

**Important Notes:**
- Change the names to match the instruction (e,g, `smple01 -> Sample01`, `whiteref` -> `White`)
- Folder names may have slight variations (e.g., `sample01`, `Sample01`, `reference`, `Reference`)
- The code handles common naming variations automatically
- Inside each folder should be wavelength TIFF files: `Image_Cube_450.tif`, `Image_Cube_460.tif`, ..., `Image_Cube_950.tif`

### Configuration File (`config.json`)

The `config.json` file controls all analysis parameters. Key sections:

#### Sample and Reference Paths
```json
"data_paths": {
  "base_path": "./Gage R&R",
  "reference_path": "${base_path}/KhangT2/Reference",
  "reference_white_path": "${base_path}/KhangT2/White",
  "reference_dark_path": "${base_path}/KhangT2/Dark",
  "sample_path": "${base_path}/LuelT2",
  "samples": ["Sample01", "Sample02", "Sample04", ...]
}
```

#### Analysis Parameters
```json
"analysis_parameters": {
  "percentile_threshold": 3.9,    // Fiducial detection sensitivity (2-5 range)
  "wavelength_range": {
    "min": 450,                    // Start wavelength (nm)
    "max": 950,                    // End wavelength (nm)
    "step": 10                     // Wavelength step (nm)
  },
  "circle_crop_region": [0.05, 0.5, 0.0, 0.5],  // [top, bottom, left, right]
  "num_sectors": 8,                // Angular sectors for uniformity analysis
  "num_fiducials": 4               // Number of fiducial markers
}
```

#### Quality Thresholds (Pass/Fail Criteria)
```json
"quality_thresholds": {
  "rmse_per_pixel_p95_max": 0.08,     // Max RMSE (95th percentile)
  "sam_mean_max": 2.6,                 // Max spectral angle (radians)
  "uniformity_score_min": 0.6,        // Min uniformity score
}
```

**Key Parameters Explained:**
- **`percentile_threshold`**: Controls fiducial marker detection sensitivity. Lower values (2-3) detect more features, higher values (4-5) are more selective. Adjust if fiducial detection fails.
- **`wavelength_range`**: Spectral bands to analyze. Default 450-950 nm covers full visible-NIR range.
- **`samples`**: List of sample folders to process. Use exact folder names.

---

## 2. Quick Start - Run 8 Trials (Gage R&R Analysis)

For rapid Gage R&R validation across multiple operators:

```bash
python run_8_trials.py
```

**What this does:**
1. Automatically processes **8 operators** (4 people × 2 trials each):
   - KhangT1, KhangT2
   - KiduT1, KiduT2  
   - LuelT1, LuelT2
   - AnirbanT1, AnirbanT2

2. For each operator:
   - Updates `config.json` with operator-specific paths and optimized percentile thresholds
   - Runs `main.py` to process all samples
   - Runs `generate_score_plots.py` to create pass/fail visualizations

3. Saves results to `results/<operator>/` for each trial

**Expected Output:**
```
🔬 Gage R&R - 8 Trial Runner
============================================================
Operators:
   1. KhangT1 (threshold: 3.4)
   2. KhangT2 (threshold: 3.8)
   ...
   8. AnirbanT2 (threshold: 3.9)
============================================================
✅ Successful trials: 8/8
🎉 All trials completed successfully!
📊 Results saved in results/ directories
```

**Then run comprehensive analysis:**
```bash
python analysis_8_trials.py
```

This generates a Gage R&R report in `gage_rr_analysis/` with:
- Combined CTQ data across all operators
- Defect detection performance metrics
- Pass/fail rates by operator
- Statistical analysis (repeatability, reproducibility)
- Comparative visualizations

---

## 3. Detailed Manual Analysis

If you want to analyze specific samples with custom parameters:

### Step 1: Modify `config.json`

Edit the configuration file to specify:
- Your operator/dataset paths
- Samples to process
- Analysis parameters
- Quality thresholds

Example for a single operator:
```json
"data_paths": {
  "sample_path": "${base_path}/KhangT1",
  "samples": ["Sample01", "Sample04", "Sample10"]
},
"analysis_parameters": {
  "percentile_threshold": 3.5,
  "wavelength_range": {"min": 450, "max": 800, "step": 10}
}
```

### Step 2: Run Main Analysis Pipeline

```bash
python main.py
```

**Pipeline Workflow:**
1. **Data Loading** - Loads hyperspectral TIFF cubes for reference and samples
2. **Reflectance Computation** - Calculates calibrated reflectance using white/dark references
3. **Projection Creation** - Averages reflectance over wavelength range for robust feature detection
4. **Fiducial Detection** - Locates 4 fiducial markers using adaptive thresholding
5. **IC Circle Detection** - Identifies the IC grating circular region
6. **Image Registration** - Computes homography transformation to align sample with reference
7. **ROI Mask Creation** - Defines analysis region from IC circle boundaries
8. **Hyperspectral Analysis** - Calculates quality metrics within ROI:
   - RMSE (Root Mean Square Error)
   - SAM (Spectral Angle Mapper)
   - Ring Delta (radial uniformity)
   - Uniformity Score (angular uniformity)
9. **Visualization** - Generates diagnostic plots
10. **Export** - Saves metrics to CSV

**Output Location:**
```
results/<operator>/
  analysis_Sample01/
    analysis_summary.csv
    Sample01_fiducials.png
    Sample01_circle.png
    Sample01_registration.png
    Sample01_reflectance.png
    Sample01_analysis_maps.png
    Sample01_uniformity.png
  analysis_Sample04/
    ...
```

### Step 3: Generate Pass/Fail Visualizations

```bash
python generate_score_plots.py
```

Creates scatter plots with pass/fail zones based on thresholds defined in `config.json`:
- **RMSE Per Pixel P95** vs **SAM Mean**
- **RMSE Per Pixel P95** vs **Uniformity Score**
- **SAM Mean** vs **Uniformity Score**

Samples are color-coded:
- 🟢 **Green**: Pass all thresholds
- 🔴 **Red**: Fail one or more thresholds

**Output Location:**
```
results/<operator>/scatter_plots/
  rmse_vs_sam_scatter.png
  rmse_vs_uniformity_scatter.png
  sam_vs_uniformity_scatter.png
```

---

## 4. Metrics

All metrics are computed **only within the IC ROI** after registration and reflectance calibration.

### RMSE (Root Mean Square Error)
```
RMSE = sqrt(mean((R_ref - R_sample)²))  over all ROI pixels and wavelength bands
```
- **Range**: 0 → ∞ (typically 0-0.2 for reflectance)
- **Meaning**: Global spectral difference magnitude
- **Interpretation**: Lower = closer match to reference
- **Threshold**: Typically < 0.08 for passing samples

### SAM (Spectral Angle Mapper)
Per pixel:
```
SAM(pixel) = arccos((r · s) / (||r|| × ||s||))
where r = reference spectrum, s = sample spectrum
```
- **Range**: 0 → π radians (0 → 180°)
- **Meaning**: Angular difference between spectral vectors
- **Interpretation**: 
  - SAM = 0: Identical spectral shape
  - Lower angle = better spectral similarity (shape and proportions)
- **Threshold**: Typically mean < 2.6 radians for passing samples

### Ring Delta (Radial Uniformity)
```
inner_mean = mean(SAM within inner 20% of radius)
outer_mean = mean(SAM within outer 80% of radius)
Ring Delta = |outer_mean - inner_mean|
```
- **Range**: 0 → π radians
- **Meaning**: Spectral consistency from center to edge
- **Interpretation**: Near zero = uniform radial behavior (no center-edge variation)

### Uniformity Score
1. Divide ROI into 8 angular sectors (45° each)
2. Compute median SAM per sector
3. Calculate robust coefficient of variation (rCV) and relative range (RR)
4. Normalize and blend: `U = 1 - (0.6 × rCVn + 0.4 × RRn)`

- **Range**: 0-1
- **Meaning**: Angular uniformity of spectral response
- **Interpretation**:
  - 1.0 = Perfect uniformity across all sectors
  - < 0.6 = Significant angular variation (likely defect)
- **Threshold**: Typically > 0.6 for passing samples

### Additional Metrics
- **RMSE Per Pixel P95**: 95th percentile of per-pixel RMSE (captures worst-case regions)
- **SAM Median**: Median spectral angle (robust to outliers)
- **SAM P95**: 95th percentile of SAM (identifies problematic regions)

---

## 5. Outputs

### Per-Sample Analysis Results

Each processed sample generates:

**`analysis_summary.csv`** - Comprehensive metrics table:
```
Sample,RMSE_Overall,RMSE_Per_Pixel_Mean,RMSE_Per_Pixel_P95,
SAM_Mean,SAM_Median,SAM_P95,Ring_Delta,Uniformity_Score,
Pass_RMSE,Pass_SAM,Pass_Uniformity,Overall_Pass
```

**Visualization Files:**
- `*_fiducials.png` - Fiducial marker detection (4 points on reference and sample)
- `*_circle.png` - IC circle detection with Hough transform results
- `*_registration.png` - Registration quality check (overlay of aligned images)
- `*_reflectance.png` - Mean reflectance spectra comparison (reference vs sample)
- `*_analysis_maps.png` - Spatial heatmaps of RMSE and SAM across ROI
- `*_uniformity.png` - 8-sector uniformity analysis with polar plot

### Batch Analysis Results

**Scatter Plots** (`results/<operator>/scatter_plots/`):
- Cross-sample comparisons with pass/fail zones
- Color-coded by quality status
- Threshold lines clearly marked

### Gage R&R Analysis Results

**`gage_rr_analysis/`** folder contains:

**Data Files:**
- `combined_ctq_data.csv` - All CTQ metrics from all 8 operators
- `gage_rr_defect_detection_report.txt` - Statistical summary and validation results

**Visualizations:**
- `ctq_boxplots_by_operator.png` - Distribution of metrics per operator
- `defect_detection_performance.png` - Pass/fail rates across samples
- `pass_fail_rates_by_operator.png` - Operator consistency analysis
- `person_trial_comparison.png` - Repeatability check (Trial 1 vs Trial 2)
- `rmse_per_pixel_p95_by_samples.png` - Sample-by-sample RMSE comparison
- `sam_mean_by_samples.png` - Sample-by-sample SAM comparison
- `uniformity_score_by_samples.png` - Sample-by-sample uniformity comparison
- `sample_pass_fail_analysis.png` - Comprehensive pass/fail matrix

**Interpretation:**
- **Repeatability**: How consistent is each person across their 2 trials?
- **Reproducibility**: How consistent are different people?
- **Defect Detection**: Does the system correctly identify defective samples?

---

## Troubleshooting

**"No fiducials detected"**
- Adjust `percentile_threshold` in config (try 2-5 range)
- Lower values (2-3) = more sensitive, higher values (4-5) = more selective
- Check that images have clear fiducial markers

**"Circle detection fails"**
- Adjust `circle_crop_region` to focus search area
- Default `[0.05, 0.5, 0.0, 0.5]` searches top-left quadrant
- Ensure IC circle is visible in projection images

**"Registration poor quality"**
- Verify that 4 fiducials are detected in both reference and sample
- Check `*_registration.png` output for alignment quality
- May need to adjust fiducial detection threshold

**"Samples have slight name variations (sample01 vs Sample01)"**
- The code is designed to handle common variations
- If issues persist, ensure exact name matching between folders and config

---

## Contact

For questions or issues, contact: khangtm99@gmail.com
