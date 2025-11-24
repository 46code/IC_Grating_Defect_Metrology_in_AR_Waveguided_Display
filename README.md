# Hyperspectral Defect Analysis

Fast pipeline to analyze waveguide IC grating defects from hyperspectral TIFF cubes. It computes reflectance first, creates projections for robust feature detection, registers samples to reference, and outputs quality metrics (RMSE, SAM, Ring Delta, Uniformity). Then you can make cross‑sample scatter plots.

---
## 1. Install
```bash
# Clone (or unzip) repo, then inside project root
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

---
## 2. Data Folder Layout
Set `base_path` in `config.json` to the root that holds all spectral folders.
Expected structure (names are case‑sensitive):
```
<base_path>/
  <operator>/
    reference/          # Reference device hyperspectral cubes
    White/              # White calibration cubes  
    Dark/               # Dark calibration cubes
    Sample1/            # Sample folders (Sample1, Sample2, etc.)
    Sample2/
    ...
```
Inside each folder are wavelength TIFFs named:
```
Image_Cube_<wl>.tif   # e.g. Image_Cube_450.tif, 460, ..., 950 (step 10 nm)
```
You can restrict processing to a wavelength sub‑range using `analysis_parameters.wavelength_range` in `config.json` (e.g. 450–800 nm). The code loads cubes, then crops only bands in that range.

---
## 3. Configure
Edit `config.json` before running.
Key fields:
```json
"data_paths": {
  "base_path": "./Gage R&R",
  "reference_path": "${base_path}/LuelT2/reference",
  "reference_white_path": "${base_path}/LuelT2/White", 
  "reference_dark_path": "${base_path}/LuelT2/Dark",
  "sample_path": "${base_path}/LuelT2",
  "samples": ["Sample1", "Sample2", "Sample4"]
},
"analysis_parameters": {
  "percentile_threshold": 3.9,        # Fiducial detection sensitivity (lower = more sensitive)
  "wavelength_range": {"min":450, "max":800, "step":10},
  "circle_crop_region": [0.05, 0.5, 0.0, 0.5],  # [top, bottom, left, right] fractions
  "num_sectors": 8,                   # For uniformity analysis
  "num_fiducials": 4
},
"visualization_parameters": {
  "generate_plots": true,
  "padding_factor": 1.4,
  "inner_radius_fraction": 0.2,
  "outer_radius_fraction": 0.8
}
```

### Key Parameters:
- **`percentile_threshold`**: Controls fiducial detection sensitivity. Lower values (e.g., 2-3) detect more features, higher values (e.g., 5-8) are more selective. Start with 3.9.
- **`wavelength_range`**: Spectral bands to use. Projections average over this entire range for robust detection.
- **`circle_crop_region`**: Where to search for IC circle [top, bottom, left, right] as fractions of image size.
- **`samples`**: List of sample folder names to process.

Quality thresholds in `quality_thresholds` control pass/fail marking:
- Pass if `RMSE_Overall < rmse_overall_max`  
- Pass if `SAM_Mean < sam_mean_max`
- Pass if `Uniformity_Score > uniformity_score_min`

---
## 4. Run Pipeline

### Step 1: Process Samples
```bash
python main.py
```

**Pipeline Workflow:**
1. **Data Loading & Reflectance Computation** - Loads cubes and computes reflectance immediately
2. **Projection Creation** - Averages reflectance over wavelength range for robust feature detection
3. **Fiducial Detection** - Finds 4 fiducial markers using projections (not single wavelength)
4. **IC Circle Detection** - Locates integrated circuit circle using projections  
5. **Image Registration** - Aligns sample to reference using homography
6. **ROI Mask Creation** - Creates analysis region from IC circle
7. **Hyperspectral Analysis** - Computes RMSE, SAM, Ring Delta, Uniformity
8. **Visualization** - Generates analysis plots and maps
9. **Summary** - Saves results to CSV

Results per sample saved to:
```
results/<operator>/analysis_<SampleN>/analysis_summary.csv
```

### Step 2: Generate Cross-Sample Plots  
```bash
python generate_score_plots.py
```
Creates scatter plots in `results/<operator>/scatter_plots/` comparing all samples.

---
## 5. Metrics
All metrics computed only inside the IC ROI after registration and reflectance correction.

### RMSE Overall
```
RMSE = sqrt( mean( (R_ref - R_sample)^2 ) )  over all ROI pixels & bands
```
**Meaning**: Global spectral difference magnitude. Lower = closer match.

### SAM (Spectral Angle Mapper)
Per pixel:
```
SAM = arccos( (r · s) / (||r|| * ||s||) )   
```
**Range**: 0 → π radians. **Meaning**: Lower angle = spectra align better (shape & proportions).

### Ring Delta  
```
inner_mean = mean(SAM within inner 20% of radius)
outer_mean = mean(SAM in outer 80% of radius) 
Ring Delta = |outer_mean - inner_mean|
```
**Meaning**: Edge vs center spectral consistency. Near zero = uniform radial behavior.

### Uniformity Score (U)
1. Divide ROI into 8 angular sectors
2. Compute median SAM per sector  
3. Calculate robust coefficient of variation (rCV) and relative range (RR)
4. Normalize against thresholds and blend: `U = 1 - (0.6 * rCVn + 0.4 * RRn)`

**Range**: 0-1. **Meaning**: 1.0 = very uniform, lower = more angular variation.

---
## 6. Troubleshooting

**"Error processing Sample: 'dark'"** - Check case sensitivity. Use 'Dark' not 'dark' in folder names.

**No fiducials detected** - Adjust `percentile_threshold` (try 2-5 range). Lower = more sensitive.

**Circle detection fails** - Adjust `circle_crop_region` to focus search area.

**Registration poor** - Ensure 4 fiducials detected in both reference and sample.

---
## 7. Output Files

Per sample:
- `analysis_summary.csv` - All metrics for the sample
- `*_fiducials.png` - Fiducial detection visualization  
- `*_circle.png` - Circle detection visualization
- `*_registration.png` - Registration quality check
- `*_reflectance.png` - Reflectance analysis
- `*_analysis_maps.png` - RMSE & SAM spatial maps
- `*_uniformity.png` - Uniformity sector analysis

Batch results:
- `scatter_plots/` folder with cross-sample comparisons
- `batch_analysis_summary.csv` - All samples combined
- `high_quality_samples.csv` - Samples passing all thresholds
