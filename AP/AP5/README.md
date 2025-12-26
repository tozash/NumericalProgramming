# Parametric Spline Reconstruction and Comparison

## Overview
This project explores parametric spline reconstruction for three characters: **T**, **O**, and **Z**. 
It implements methods to fit curves to digital glyphs represented by ordered node lists, comparing different spline techniques and the effect of node density.

## Key Features
- **Data**: Character strokes normalized in a 10x10 drawing box.
- **Methods**:
    - **Cubic Spline Interpolation**: With *Natural*, *Clamped*, and *Periodic* boundary conditions.
    - **B-Splines**: Quadratic and Cubic B-spline approximations (using `splprep`).
- **Experiments**:
    - Comparison of fit quality under node downsampling (using every 2nd or 3rd node).
    - Parametric chord-length parameterization for planar curves `(x(t), y(t))`.
- **Metrics**: Mean and Max Euclidean distance from the original dense source nodes.

## Structure
- `src/`: Source code for data, fitting, metrics, and plotting.
- `outputs/plots/`: Generated visualizations of the fits.
- `outputs/tables/`: CSV summary of error metrics.
- `paper/`: A short report explaining the math, methods, and results.
- `video/`: Script for the project video presentation.

## Usage
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Analysis**:
   ```bash
   python src/run_all.py
   ```
   This command will:
   - Generate plots for each character/stroke in `outputs/plots`.
   - Compute error metrics and save them to `outputs/tables/metrics.csv`.
   - Print a summary to the console.

## Dependencies
- Python 3.10+
- `numpy`, `matplotlib`, `scipy`
