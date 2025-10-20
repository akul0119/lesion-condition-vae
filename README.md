# lesion-condition-vae
A comprehensive Python framework for analyzing white matter tract geometry, lesion morphology, and their relationships in traumatic brain injury (TBI) and post-traumatic epilepsy (PTE) rodent models.

Overview

This framework provides tools for:
- **Global tract geometry analysis** - Quantifying white matter tract shape and organization
- **Along-tract analysis using VAE** - Dimensionality reduction and pattern discovery
- **Lesion shape analysis** - Spherical harmonic descriptors for lesion morphology
- **TBI vs PTE classification** - Machine learning models for group discrimination
- **Correlation analysis** - Lesion-tract relationship quantification
- **Temporal analysis** - Longitudinal tracking across timepoints

## Key Features

### 1. Tract Geometry Processing
- **17 geometric metrics** including:
  - Length, tortuosity, curvature
  - Elongation, planarity, anisotropy ratios
  - Angular dispersion and torsion
- Processes VTK streamline files
- Handles 16+ white matter tracts
- Supports multiple timepoints (2d, 9d, 1mo, 5mo)

### 2. Lesion Shape Analysis
- **Spherical harmonic (SH) decomposition** up to degree L=6
- Rotation-invariant power spectrum (P0-P6)
- Unit volume normalization
- 3D visualization and reconstruction
- Heme content quantification

### 3. VAE-Based Along-Tract Analysis
- Variational Autoencoder for microstructure profiles
- Latent dimension: 10
- Handles 12+ microstructure features:
  - DTI: FA, MD, AD, RD
  - MWI: R2*, T2*
  - XFIB: crossing fractions, fiber densities
- Segment-level analysis (20 segments per tract)

### 4. Machine Learning Classification
- Random Forest, SVM, Elastic Net
- 10-fold cross-validation
- Class-weighted for imbalanced data
- Performance visualization (ROC, AUC, confusion matrices)
- Feature importance ranking

### 5. Statistical Analysis
- Pearson correlations (lesion ↔ tract)
- Group comparisons (TBI vs PTE)
- Temporal trends
- Within-subject displacement analysis

## Installation

### Requirements
- Python 3.8+
- PyVista
- PyTorch
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- nibabel
- scipy

### Install from source
```bash
git clone https://github.com/yourusername/tbi-pte-tract-analysis.git
cd tbi-pte-tract-analysis
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Tract Geometry Analysis
```python
from src.geometry import compute_streamline_metrics

# Analyze single tract
df_sl, df_bundle = compute_streamline_metrics('path/to/tract.vtk', max_streamlines=1000)

# Or run comprehensive analysis
python scripts/run_tract_geometry.py
```

### 2. Lesion Shape Analysis
```python
from src.lesion import analyze_single_lesion

# Compute SH descriptors
result, success = analyze_single_lesion(
    subject_id='1008',
    timepoint='1mo',
    data_dir='./data',
    output_dir='./results',
    max_l=6
)
```

### 3. VAE Analysis
```python
from src.vae import run_vae_analysis

# Train VAE on tract profiles
results = run_vae_analysis(
    tract='atr_left',
    latent_dim=10,
    epochs=40
)
```

### 4. Classification (TBI vs PTE)
```python
python scripts/run_classification.py
# Outputs: ROC curves, confusion matrices, feature importance
```

### 5. Correlation Analysis
```python
python scripts/run_correlation_analysis.py
# Outputs: Scatter plots, heatmaps, significance tables
```

## Data Structure

Expected directory structure:
```
data/
├── {subject_id}/
│   ├── {timepoint}/
│   │   ├── bundles/
│   │   │   ├── {tract}_curves.vtk.gz
│   │   │   ├── {tract}_curves.vtk.gz
│   │   │   └── ...
│   │   ├── lesion.nii.gz
│   │   ├── heme.nii.gz
│   │   ├── tissue.nii.gz
│   │   └── tract_profiles/
│   │       └── {tract}_profiles.csv
```

## Configuration
Edit `configs/tract_config.json`:
```json
{
  "groups": {
    "TBI": [1043, 1046, 1095, ...],
    "PTE": [1008, 1012, 1024, ...],
    "Sham": [1017, 1035, ...]
  },
  "timepoints": ["2d", "9d", "1mo", "5mo"],
  "tracts": ["atr_left", "atr_right", ...],
  "model_params": {
    "latent_dim": 10,
    "epochs": 40
  }
}
```

## Outputs

### Tract Geometry
- `comprehensive_tract_geometry_metrics.csv` - All geometric features
- Summary statistics by group/timepoint

### Lesion Analysis
- `lesion_sh_heme_comprehensive.csv` - SH descriptors + heme content
- 3D surface visualizations
- Power spectrum plots

### Classification
- ROC curves and confusion matrices (PNG)
- Feature importance rankings
- Classification summary (CSV)
- Top predictor boxplots

### Correlation Analysis
- Scatter plots with regression lines (PNG)
- Correlation heatmaps by group/timepoint
- Significant correlations table (CSV)

## Key Analysis Results

### Tract Geometry Metrics
- **2,356 records** (37 subjects × 4 timepoints × 16 tracts)
- 11 geometric features per tract
- Aggregated at subject level for ML

### Classification Performance (Best: 5mo)
- Random Forest: 69% accuracy, AUC=0.653
- Most important features: elongation_ratio, planarity_ratio

### Lesion-Tract Correlations
- **28 significant correlations** (p<0.05)
- Strongest: P5 ↔ anisotropy_ratio (r=0.832, PTE @ 1mo)
- Temporal patterns differ: PTE peaks at 1mo, TBI at 5mo

### Centroid Displacement
- Peak at 9d: TBI=10.8mm, PTE=14.2mm
- PTE shows greater displacement throughout

## Citation

If you use this framework, please cite:

```bibtex
@software{tbi_pte_tract_analysis,
  author = Akul Sharma,
  title = {White Matter Tract Analysis Framework: TBI vs PTE},
  year = {2025},
  url = {https://github.com/akul0119/lesion-condition-vae}
}
```
