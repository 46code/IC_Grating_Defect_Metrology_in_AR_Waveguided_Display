# IC Grating Defect Metrology in AR Waveguided Display

Hyperspectral analysis pipeline for detecting and quantifying defects in waveguide integrated circuit (IC) gratings used in augmented reality displays. This system processes hyperspectral TIFF cubes to compute reflectance, register samples to reference, and output quality metrics for defect detection.

---

## Project Structure

```
IC_Grating_Defect_Metrology_in_AR_Waveguided_Display/
├── main.py                          # Main analysis pipeline
├── generate_score_plots.py          # Pass/fail visualization generator
├── run_8_trials.py                  # Automated 8-operator runner
├── analysis_8_trials.py             # Comprehensive Gage R&R analysis
├── config.json                      # Configuration file (paths, thresholds, parameters)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── modules/                         # Reusable library modules
│   ├── lib_gage_rr.py              # Gage R&R analysis functions
│   ├── lib_spectral_loader.py      # Hyperspectral data loading
│   ├── lib_reflectance_analysis.py # Reflectance computation
│   ├── lib_feature_detection.py    # Fiducial & circle detection
│   ├── lib_image_registration.py   # Image alignment
│   ├── lib_ctq_analysis.py         # Quality metrics calculation
│   └── lib_plotting.py             # Visualization utilities
├── results/                         # Per-operator analysis results
│   └── <operator>/                 # e.g., KhangT1, LuelT2
│       ├── analysis_<sample>/      # Per-sample results
│       └── scatter_plots/          # Pass/fail visualizations
└── analysis_results/                # Comprehensive Gage R&R output (organized)
    ├── README.md                   # Output directory documentation
    ├── data/                       # CSV data files
    ├── reports/                    # Text summaries
    └── plots/                      # All visualizations
        ├── gage_rr_dashboards/     # Professional summary dashboards
        ├── diagnostics/            # Diagnostic plots
        ├── distinct_categories/    # Sample discrimination analysis
        └── sample_analysis/        # Sample performance plots
```

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
  "rmse_overall_max": 0.08,           // Max overall RMSE
  "rmse_per_pixel_mean_max": 0.06,   // Max mean RMSE per pixel
  "rmse_per_pixel_median_max": 0.09,  // Max median RMSE per pixel
  "sam_median_max": 3.0,              // Max median SAM
  "sam_p95_max": 5.2                  // Max 95th percentile SAM
}
```

**Important:** Quality thresholds are now **dynamically loaded** from `config.json`. 
- Modify thresholds in `config.json` without touching code
- Changes take effect immediately on next analysis run
- All visualizations automatically use updated thresholds
- Both `analysis_8_trials.py` and plotting functions read from config

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

**New Features (Dec 2025):**
- ✅ **Dynamic threshold loading** - Quality thresholds read directly from `config.json`
- ✅ **Organized output structure** - Results grouped by type (data, reports, plots)
- ✅ **Professional dashboards** - Multi-panel Gage R&R summary visualizations
- ✅ **Advanced diagnostics** - Three-panel diagnostic plots with statistical analysis
- ✅ **Sample discrimination** - NDC (Number of Distinct Categories) analysis
- ✅ **Modular architecture** - Core functions moved to `lib_gage_rr.py` for reusability

**Output:** Creates organized `analysis_results/` directory with:
- Data files (CSV with all CTQ metrics)
- Reports (comprehensive text summaries)
- Plots organized by category:
  - Gage R&R dashboards (3 professional summary panels)
  - Diagnostics (3 three-panel diagnostic plots)
  - Distinct categories (3 sample discrimination analyses)
  - Sample analysis (9 sample-level performance plots)

**Statistics Provided:**
- Measurement system capability (repeatability, reproducibility)
- Defect detection performance (signal-to-noise ratios)
- Sample discrimination (NDC calculation)
- Pass/fail rates by operator and sample

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

## 4. Outputs

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

### Gage R&R Analysis Results (Updated Structure)

Running `analysis_8_trials.py` creates an organized `analysis_results/` directory:

```
analysis_results/
├── README.md                           # Complete documentation
├── data/                               # CSV data files
│   └── all_ctq_data_with_thresholds.csv
├── reports/                            # Text reports
│   └── comprehensive_analysis_summary.txt
└── plots/                              # All visualizations (21 files)
    ├── gage_rr_dashboards/             # Professional dashboards (3 files)
    │   ├── RMSE_Per_Pixel_P95_gage_rr_summary_dashboard.png
    │   ├── SAM_Mean_gage_rr_summary_dashboard.png
    │   └── Uniformity_Score_gage_rr_summary_dashboard.png
    ├── diagnostics/                    # Diagnostic plots (3 files)
    │   ├── RMSE_Per_Pixel_P95_gage_rr_analysis.png
    │   ├── SAM_Mean_gage_rr_analysis.png
    │   └── Uniformity_Score_gage_rr_analysis.png
    ├── distinct_categories/            # Sample discrimination (3 files)
    │   ├── RMSE_Per_Pixel_P95_distinct_categories_analysis.png
    │   ├── SAM_Mean_distinct_categories_analysis.png
    │   └── Uniformity_Score_distinct_categories_analysis.png
    └── sample_analysis/                # Sample performance (9 files)
        ├── ctq_boxplots_by_operator.png
        ├── rmse_overall_by_samples.png
        ├── rmse_per_pixel_mean_by_samples.png
        └── ... (other CTQ metrics)
```

**Key Features:**

**1. Gage R&R Dashboards** - Professional multi-panel summaries with:
- Input summary & ANOVA tables
- Variance component breakdown
- Color-coded assessment zones (green/yellow/red)
- Sample × Operator interaction plots

**2. Diagnostic Plots** - Three-panel analysis showing:
- Box plots by operator with jittered points
- Sample means line plot
- Run chart with ±2σ control limits

**3. Distinct Categories Analysis** - Sample discrimination capability:
- NDC (Number of Distinct Categories) calculation
- Color-coded category visualization
- Assessment: Excellent (NDC≥5), Adequate (3-5), Poor (<3)

**4. Sample Analysis** - Detailed performance tracking:
- CTQ box plots by operator
- Individual metric scatter plots by sample
- Threshold lines for pass/fail assessment

**Analysis Metrics:**
- **Repeatability**: Consistency within each person across 2 trials
- **Reproducibility**: Consistency between different people
- **Measurement System %**: Total variation due to measurement system (target: <10% excellent, <30% acceptable)
- **Defect Detection**: Signal-to-noise ratio for distinguishing good vs defective samples

**Quality Thresholds** (loaded from `config.json`):
- RMSE Per Pixel P95: ≤ 0.08
- SAM Mean: ≤ 2.6 radians
- Uniformity Score: ≥ 0.6

*Note: Thresholds are dynamically loaded from config.json and can be adjusted without code changes.*

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

**"Want to change quality thresholds"**
- Edit `config.json` under `quality_thresholds` section
- No code changes needed - thresholds are loaded dynamically
- Changes apply to all analysis and visualization functions
- Example: To make SAM threshold stricter, change `"sam_mean_max": 2.6` to `"sam_mean_max": 2.0`

**"Need to find specific analysis results"**
- Results are organized in `analysis_results/` by type:
  - **Data**: `analysis_results/data/` (CSV files)
  - **Reports**: `analysis_results/reports/` (text summaries)
  - **Dashboards**: `analysis_results/plots/gage_rr_dashboards/`
  - **Diagnostics**: `analysis_results/plots/diagnostics/`
  - **Sample plots**: `analysis_results/plots/sample_analysis/`
- See `analysis_results/README.md` for complete directory documentation

---

## Recent Updates (December 2025)

### Version 2.0 - Enhanced Analysis & Organization

**Major Improvements:**
1. **Dynamic Configuration**
   - Quality thresholds loaded from `config.json`
   - No code changes needed to adjust pass/fail criteria
   - Automatic propagation to all analysis functions

2. **Organized Output Structure**
   - Hierarchical directory organization
   - Results grouped by type (data, reports, plots)
   - Subdirectories for different plot categories
   - Comprehensive README in output directory

3. **Advanced Gage R&R Visualizations**
   - Professional multi-panel summary dashboards
   - Three-panel diagnostic plots
   - Distinct categories analysis with NDC
   - Color-coded assessment zones

4. **Modular Architecture**
   - Core functions moved to `modules/lib_gage_rr.py`
   - Improved code reusability
   - Better separation of concerns
   - Easier maintenance and testing

5. **Enhanced Documentation**
   - Auto-generated README in results directory
   - Detailed interpretation guides
   - Clear directory structure documentation

**Bug Fixes:**
- Fixed type compatibility issues in matplotlib
- Resolved variable initialization bugs
- Corrected colormap access methods
- Improved error handling and fallbacks

---

## Contact

For questions or issues, contact: khangtm99@gmail.com
