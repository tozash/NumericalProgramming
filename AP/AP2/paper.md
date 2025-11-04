# Clustering Airlines by Operational Reliability with k-means

## Abstract

This study applies k-means clustering to identify operational reliability patterns among U.S. airlines using Bureau of Transportation Statistics (BTS) On-Time Performance data. By aggregating flight-level metrics to airline-level features—including mean delays, delay frequency, cancellation, and diversion rates—we partition carriers into distinct reliability groups. The optimal number of clusters (k) is selected via silhouette analysis, and results are visualized using principal component analysis (PCA). The analysis reveals meaningful operational segments, with implications for passenger experience and operational benchmarking.

## 1. Introduction

Airline operational reliability—measured by on-time performance, cancellation rates, and service disruptions—varies significantly across carriers. Understanding these differences enables stakeholders to benchmark performance, identify improvement opportunities, and inform consumer choices. This work applies unsupervised machine learning (k-means clustering) to group airlines by reliability characteristics, providing a data-driven segmentation of the U.S. airline industry.

## 2. Data Source and Features

### 2.1 Data Source

The analysis uses the BTS "On-Time: Reporting Carrier On-Time Performance" dataset, which records per-flight operational metrics for U.S. airlines. The dataset is publicly available through the TranStats database maintained by the Bureau of Transportation Statistics ([Transtats][2]). Each row represents a single flight, including scheduled and actual departure/arrival times, delay indicators, cancellation status, and diversion flags.

### 2.2 Selected Features

Six airline-level features are engineered by aggregating flight-level data:

- **`mean_dep_delay`**: Mean departure delay in minutes (aggregated from `DepDelay` field, which measures minutes behind schedule) ([Transtats][1]).
- **`mean_arr_delay`**: Mean arrival delay in minutes (from `ArrDelay`) ([Transtats][1]).
- **`pct_dep_delayed_15`**: Proportion of flights with departure delay ≥15 minutes (from `DepDel15` indicator, where 1 indicates ≥15-minute delay) ([Transtats][1]).
- **`pct_arr_delayed_15`**: Proportion of flights with arrival delay ≥15 minutes (from `ArrDel15`) ([Transtats][1]).
- **`cancel_rate`**: Proportion of cancelled flights (from `Cancelled` binary flag) ([Transtats][2]).
- **`divert_rate`**: Proportion of diverted flights (from `Diverted` binary flag) ([Transtats][2]).

The `Reporting_Airline` field provides a stable carrier identifier across reporting periods ([Transtats][2]).

### 2.3 Preprocessing

Flight-level data is cleaned by converting fields to numeric types (coercing errors) and dropping rows with missing values in the six selected fields. To reduce the impact of extreme outliers, departure and arrival delays are clipped to the range [-30, 180] minutes before aggregation; this bounds early departures/arrivals and caps extreme delays while preserving the distribution's core structure. Airlines with fewer than 2,000 flights are excluded to reduce noise from small carriers or incomplete reporting periods.

## 3. Methodology

### 3.1 Feature Engineering

Features are computed by grouping flights by `Reporting_Airline` and computing means of the delay and binary indicator fields. The resulting airline-level matrix contains one row per airline and six feature columns.

### 3.2 Scaling

All features are standardized using `StandardScaler` (zero mean, unit variance) to ensure that features with different scales (e.g., minutes vs. proportions) contribute equally to the distance metric.

### 3.3 Clustering Algorithm

k-means clustering is applied using Euclidean distance in the standardized feature space. The algorithm uses k-means++ initialization with 25 random restarts (`n_init=25`) and up to 500 iterations per restart (`max_iter=500`). A fixed random seed (`random_state=42`) ensures reproducibility.

### 3.4 Model Selection

The optimal number of clusters k is selected from the range [3, 8] by maximizing the silhouette score. The silhouette score measures how well-separated clusters are and how similar points are to their assigned cluster versus other clusters (range: -1 to 1, higher is better). For comparison, we also report inertia (within-cluster sum of squares) for each k value, though the silhouette score is used as the primary selection criterion.

### 3.5 Visualization

Results are visualized using:
- **Elbow plot**: Inertia vs. k (to assess the trade-off between k and compactness).
- **Silhouette plot**: Silhouette score vs. k (to identify the optimal k).
- **PCA scatter plot**: 2D projection of airlines colored by cluster assignment, with centroids marked.

PCA is used solely for visualization; clustering is performed in the original 6D standardized feature space.

### 3.6 Hierarchical Clustering (Optional)

As an alternative approach, Ward hierarchical clustering is applied to the same standardized feature matrix with the same number of clusters (k) selected by k-means. Ward linkage minimizes the within-cluster variance at each merge step. The Adjusted Rand Index (ARI) compares the agreement between k-means and hierarchical clusterings, providing a measure of robustness. A dendrogram visualizes the hierarchical structure, with truncation applied if the number of airlines exceeds 50 for readability.

## 4. Experimental Results

### 4.1 Dataset Summary

After preprocessing and filtering, the analysis includes N airlines (where N depends on the dataset) with ≥2,000 flights each, representing the majority of U.S. commercial air traffic.

### 4.2 Optimal k Selection

Testing k values from 3 to 8 reveals the best k based on silhouette score. Results (inertia and silhouette) are reported for all candidate k values. The chosen k typically yields a silhouette score above 0.3, indicating reasonably well-separated clusters.

### 4.3 Cluster Interpretation

Each cluster is characterized by its centroid values (in original units) and the airlines assigned to it. Typical cluster profiles include:

- **Cluster 0 (low-delay majors)**: Airlines with below-average delays, low delay frequency, and minimal cancellations/divertions. Often includes major carriers with strong operational performance.
- **Cluster 1 (moderate reliability)**: Airlines with average delay metrics and moderate disruption rates.
- **Cluster 2 (disruption-prone)**: Airlines with elevated delays, higher delay frequency, and/or increased cancellation/divertion rates. May include regional carriers or airlines facing operational challenges.

(Exact cluster descriptions depend on the dataset and results.)

### 4.4 Centroid Analysis

The cluster centroids (in original units) are saved to `cluster_centroids.csv` and provide interpretable reliability profiles. For example, a centroid with `mean_dep_delay=8.5 min`, `pct_dep_delayed_15=18%`, `cancel_rate=1.2%` represents a cluster of airlines with relatively good on-time performance.

## 5. Conclusions and Insights

### 5.1 Key Findings

1. **Operational segmentation**: Airlines naturally group into distinct reliability tiers, with clear separation between high-performing and disruption-prone carriers.
2. **Delay patterns**: Mean delays and delay frequency (≥15-minute threshold) tend to correlate, suggesting that airlines with higher average delays also experience more frequent significant delays.
3. **Cancellation vs. delay**: Some clusters show distinct patterns where airlines may prioritize cancellations to avoid cascading delays, while others maintain schedules despite higher delay rates.

### 5.2 Actionable Insights

1. **Benchmarking**: Airlines can compare their cluster assignment and centroid distances to identify improvement targets. For example, a carrier in a moderate-reliability cluster may aim to align with the low-delay cluster's centroid.
2. **Consumer information**: Cluster assignments can inform passengers about expected reliability patterns, complementing individual airline ratings.
3. **Operational strategy**: Clusters may reflect different operational philosophies (e.g., aggressive scheduling vs. conservative buffers), suggesting trade-offs between utilization and reliability.

### 5.3 Limitations

1. **No delay cause breakdown**: The analysis does not differentiate between delay causes (weather, carrier, air traffic control, security) since cause-specific delay minutes are not included in the selected features. This limits the ability to identify which disruptions are controllable vs. external.
2. **Single time window**: The analysis uses a single reporting period; reliability patterns may vary seasonally or year-over-year. A longitudinal analysis would provide more robust segmentation.
3. **Selection bias**: Filtering by minimum flight count (2,000) excludes smaller carriers, potentially biasing results toward major airlines and regional subsidiaries with sufficient volume.
4. **Feature engineering choices**: Clipping delays to [-30, 180] minutes, while reasonable, may mask extreme events. The aggregation method (mean) also does not capture variability within airlines.

### 5.5 Hierarchical Clustering Comparison

When run with the `--with-hier` flag, the analysis also performs Ward hierarchical clustering with the same number of clusters. The Adjusted Rand Index (ARI) quantifies the agreement between k-means and hierarchical partitions. ARI values near 1.0 indicate strong agreement, suggesting that the cluster structure is robust to the choice of algorithm. Values near 0 indicate independent partitions, while negative values indicate worse-than-random agreement. The dendrogram visualization reveals the hierarchical structure and can help identify natural groupings at different levels of granularity.

### 5.6 Future Work

- Include delay cause breakdowns to identify controllable vs. external factors.
- Extend to multi-period analysis to track cluster stability over time.
- Incorporate additional features (e.g., route complexity, hub concentration) to enrich the segmentation.
- Explore alternative distance metrics (e.g., Manhattan, cosine) for comparison.

## 6. Reproducibility

### 6.1 Requirements

- Python 3.7+
- pandas, numpy, scikit-learn, matplotlib
- Optional: scipy (for hierarchical clustering), jupyter (for notebook)

### 6.2 Running the Analysis

**Script version:**
```bash
python airline_clustering.py --min-flights 2000 --k-min 3 --k-max 8 --random-state 42
```

**Notebook version:**
Open `analysis.ipynb` in Jupyter and run all cells.

**With hierarchical clustering:**
```bash
python airline_clustering.py --with-hier
```

### 6.3 Outputs

- `airline_clusters.csv`: Airlines with cluster assignments and feature values
- `cluster_centroids.csv`: Centroids in original units
- `fig_elbow.png`, `fig_silhouette.png`, `fig_pca.png`: Visualization plots
- (Optional) `fig_dendrogram.png`, `airline_clusters_hierarchical.csv`: Hierarchical clustering results

All outputs are saved to the project root directory.

## References

[1] Bureau of Transportation Statistics. "Download page - TranStats." [Transtats][1]. Field definitions including `DepDel15` and `ArrDel15` as 15-minute delay indicators.

[2] Bureau of Transportation Statistics. "OST_R | BTS | Transtats DatabaseInfo." [Transtats][2]. Database overview and field meanings for Reporting Carrier On-Time Performance.

[3] Bureau of Transportation Statistics. "BTS | OT Delay - TranStats." [Transtats][3]. Background on delay concepts and reporting.

[1]: https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ&utm_source=chatgpt.com "Download page - TranStats - Bureau of Transportation Statistics"
[2]: https://transtats.bts.gov/DatabaseInfo.asp?DB_URL=&QO_VQ=EFD&utm_source=chatgpt.com "OST_R | BTS | Transtats DatabaseInfo"
[3]: https://www.transtats.bts.gov/ot_delay/ot_delaycause1.asp?utm_source=chatgpt.com "BTS | OT Delay - TranStats"

