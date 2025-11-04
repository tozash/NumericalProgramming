# Airline Reliability Clustering

This project clusters U.S. airlines by operational reliability using k-means clustering on Bureau of Transportation Statistics (BTS) On-Time Performance data.

## Quick Start

### Requirements

- Python 3.7+
- Required packages:
  ```bash
  pip install -r requirements.txt
  ```
  Or manually:
  ```bash
  pip install pandas numpy scikit-learn matplotlib scipy
  ```
  (Optional: `jupyter` for notebook)

### Data

Place your BTS On-Time Reporting CSV file at `./T_ONTIME_REPORTING.csv`.

The script automatically handles common column name variations. It expects columns for:
- **Reporting airline**: `Reporting_Airline`, `OP_UNIQUE_CARRIER`, or `OP_CARRIER`
- **Delays**: `DepDelay`/`DEP_DELAY`, `ArrDelay`/`ARR_DELAY`
- **Delay indicators**: `DepDel15`/`DEP_DEL15`, `ArrDel15`/`ARR_DEL15`
- **Status flags**: `Cancelled`/`CANCELLED`, `Diverted`/`DIVERTED`

The script will automatically map these to standard names internally.

### Run the Script

**Basic run:**
```bash
python airline_clustering.py
```

**With custom parameters:**
```bash
python airline_clustering.py --min-flights 2000 --k-min 3 --k-max 8 --random-state 42
```

**With hierarchical clustering (optional):**
```bash
python airline_clustering.py --with-hier
```

### Run the Notebook

```bash
jupyter notebook analysis.ipynb
```

Then run all cells.

## Outputs

The script generates:

1. **CSV files:**
   - `airline_clusters.csv` - Airlines with cluster assignments and feature values
   - `cluster_centroids.csv` - Cluster centroids in original units
   - (Optional) `airline_clusters_hierarchical.csv` - Hierarchical clustering results

2. **Visualizations:**
   - `fig_elbow.png` - Elbow plot (inertia vs k)
   - `fig_silhouette.png` - Silhouette score vs k
   - `fig_pca.png` - 2D PCA projection colored by cluster
   - (Optional) `fig_dendrogram.png` - Hierarchical clustering dendrogram

## Command-Line Arguments

- `--data`: Path to CSV file (default: `./T_ONTIME_REPORTING.csv`)
- `--min-flights`: Minimum flights per airline (default: 2000)
- `--k-min`: Minimum k to test (default: 3)
- `--k-max`: Maximum k to test (default: 8)
- `--random-state`: Random seed (default: 42)
- `--with-hier`: Also run Ward hierarchical clustering

## Project Structure

```
.
├── airline_clustering.py    # Main script
├── analysis.ipynb           # Jupyter notebook version
├── paper.md                 # 2-page academic report
├── README.md                # This file
└── T_ONTIME_REPORTING.csv   # Input data (user-provided)
```

## Methodology

1. **Data cleaning:** Coerce numeric types, clip extreme delays to [-30, 180] minutes, drop missing values
2. **Feature engineering:** Aggregate flight-level metrics to airline-level features (mean delays, delay frequency, cancellation/divertion rates)
3. **Filtering:** Exclude airlines with < 2,000 flights
4. **Scaling:** Standardize features (zero mean, unit variance)
5. **Clustering:** k-means++ with k selected via silhouette score (tests k ∈ [3, 8])
6. **Visualization:** Elbow, silhouette, and PCA plots

## Documentation

See `paper.md` for the full academic report including methodology, results, and conclusions.

## Reproducibility

All analyses use `random_state=42` for deterministic results. The script includes a unit test for feature engineering and validates required columns at load time.

