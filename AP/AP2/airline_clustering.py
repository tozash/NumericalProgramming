"""
Airline Reliability Clustering with k-means
Clusters U.S. airlines by operational reliability metrics.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """
    Load the BTS On-Time Reporting CSV.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with validated columns (standardized names)
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    
    # Map common column name variations to standard names
    column_mapping = {}
    
    # Reporting airline column
    for col in ['Reporting_Airline', 'OP_UNIQUE_CARRIER', 'OP_CARRIER']:
        if col in df.columns:
            column_mapping[col] = 'Reporting_Airline'
            break
    
    # Delay columns
    for std_name, variations in [
        ('DepDelay', ['DepDelay', 'DEP_DELAY', 'DEPARTURE_DELAY']),
        ('ArrDelay', ['ArrDelay', 'ARR_DELAY', 'ARRIVAL_DELAY']),
        ('DepDel15', ['DepDel15', 'DEP_DEL15', 'DEP_DELAY_15']),
        ('ArrDel15', ['ArrDel15', 'ARR_DEL15', 'ARR_DELAY_15']),
        ('Cancelled', ['Cancelled', 'CANCELLED', 'CANCELED']),
        ('Diverted', ['Diverted', 'DIVERTED'])
    ]:
        for var in variations:
            if var in df.columns:
                column_mapping[var] = std_name
                break
    
    # Check if we found all required columns
    required_std_cols = ['Reporting_Airline', 'DepDelay', 'ArrDelay', 
                         'DepDel15', 'ArrDel15', 'Cancelled', 'Diverted']
    found_cols = set(column_mapping.values())
    missing_cols = [col for col in required_std_cols if col not in found_cols]
    
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"Available columns: {list(df.columns)[:20]}...\n"
            f"Please ensure the CSV contains delay and airline information."
        )
    
    # Rename columns to standard names
    df = df.rename(columns=column_mapping)
    
    # Keep only needed columns (already renamed, so we have the standard names)
    df = df[required_std_cols]
    
    print(f"Loaded {len(df):,} flight records")
    print(f"  Using columns: {required_std_cols}")
    return df


def clean_data(df):
    """
    Clean and prepare flight-level data.
    
    Args:
        df: Raw flight data
        
    Returns:
        Cleaned DataFrame
    """
    print("Cleaning data...")
    df = df.copy()
    
    # Convert to numeric, coercing errors
    numeric_cols = ['DepDelay', 'ArrDelay', 'DepDel15', 'ArrDel15', 
                    'Cancelled', 'Diverted']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clip extreme delays: [-30, 180] minutes
    df['DepDelay'] = df['DepDelay'].clip(-30, 180)
    df['ArrDelay'] = df['ArrDelay'].clip(-30, 180)
    
    # Drop rows with missing values in required fields
    initial_rows = len(df)
    df = df.dropna(subset=numeric_cols + ['Reporting_Airline'])
    dropped = initial_rows - len(df)
    
    if dropped > 0:
        print(f"  Dropped {dropped:,} rows with missing values")
    
    print(f"  Retained {len(df):,} clean flight records")
    return df


def engineer_features(df, min_flights=2000):
    """
    Engineer airline-level reliability features.
    
    Args:
        df: Cleaned flight data
        min_flights: Minimum flights per airline to include
        
    Returns:
        DataFrame with airline-level features
    """
    print(f"Engineering airline-level features (min_flights={min_flights})...")
    
    # Group by airline
    airline_stats = df.groupby('Reporting_Airline').agg({
        'DepDelay': 'mean',
        'ArrDelay': 'mean',
        'DepDel15': 'mean',
        'ArrDel15': 'mean',
        'Cancelled': 'mean',
        'Diverted': 'mean',
        'Reporting_Airline': 'count'  # Count flights
    }).rename(columns={
        'DepDelay': 'mean_dep_delay',
        'ArrDelay': 'mean_arr_delay',
        'DepDel15': 'pct_dep_delayed_15',
        'ArrDel15': 'pct_arr_delayed_15',
        'Cancelled': 'cancel_rate',
        'Diverted': 'divert_rate',
        'Reporting_Airline': 'flight_count'
    })
    
    print(f"  Found {len(airline_stats)} airlines before filtering")
    
    # Filter by minimum flights
    airline_stats = airline_stats[airline_stats['flight_count'] >= min_flights]
    print(f"  Retained {len(airline_stats)} airlines after filtering")
    
    # Select feature columns
    feature_cols = ['mean_dep_delay', 'mean_arr_delay', 'pct_dep_delayed_15',
                    'pct_arr_delayed_15', 'cancel_rate', 'divert_rate']
    
    return airline_stats[['flight_count'] + feature_cols], feature_cols


def test_feature_engineering():
    """Unit test for feature engineering."""
    test_df = pd.DataFrame({
        'Reporting_Airline': ['AA', 'AA', 'AA', 'UA', 'UA'],
        'DepDelay': [10, 20, 30, 5, 15],
        'ArrDelay': [12, 22, 32, 7, 17],
        'DepDel15': [0, 1, 1, 0, 1],
        'ArrDel15': [0, 1, 1, 0, 1],
        'Cancelled': [0, 0, 0, 0, 1],
        'Diverted': [0, 0, 0, 0, 0]
    })
    
    stats, _ = engineer_features(test_df, min_flights=2)
    
    assert len(stats) == 2, "Should have 2 airlines"
    assert 'AA' in stats.index and 'UA' in stats.index
    assert abs(stats.loc['AA', 'mean_dep_delay'] - 20.0) < 0.1
    assert abs(stats.loc['AA', 'pct_dep_delayed_15'] - 2/3) < 0.01
    print("  ✓ Feature engineering test passed")


def find_best_k(X_scaled, k_min=3, k_max=8, random_state=42):
    """
    Find best k using silhouette score.
    
    Args:
        X_scaled: Scaled feature matrix
        k_min: Minimum k to test
        k_max: Maximum k to test
        random_state: Random seed
        
    Returns:
        best_k, results dict with metrics for each k
    """
    print(f"\nTesting k values from {k_min} to {k_max}...")
    
    results = []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, n_init=25, max_iter=500, 
                       random_state=random_state, init='k-means++')
        labels = kmeans.fit_predict(X_scaled)
        
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X_scaled, labels)
        
        results.append({
            'k': k,
            'inertia': inertia,
            'silhouette': silhouette
        })
        
        print(f"  k={k}: inertia={inertia:.2f}, silhouette={silhouette:.4f}")
    
    # Sort by silhouette (descending)
    results_sorted = sorted(results, key=lambda x: x['silhouette'], reverse=True)
    
    best_k = results_sorted[0]['k']
    print(f"\nBest k={best_k} (silhouette={results_sorted[0]['silhouette']:.4f})")
    
    return best_k, results


def plot_elbow(results, output_path='fig_elbow.png'):
    """Plot elbow curve (inertia vs k)."""
    k_vals = [r['k'] for r in results]
    inertias = [r['inertia'] for r in results]
    
    plt.figure(figsize=(8, 6))
    plt.plot(k_vals, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def plot_silhouette(results, output_path='fig_silhouette.png'):
    """Plot silhouette scores vs k."""
    k_vals = [r['k'] for r in results]
    silhouettes = [r['silhouette'] for r in results]
    
    plt.figure(figsize=(8, 6))
    plt.plot(k_vals, silhouettes, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def plot_pca_clusters(X_scaled, labels, centroids_scaled, output_path='fig_pca.png'):
    """
    Plot 2D PCA projection colored by cluster.
    
    Args:
        X_scaled: Scaled feature matrix
        labels: Cluster labels
        centroids_scaled: Scaled centroids
        output_path: Output file path
    """
    # Fit PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(centroids_scaled)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                         cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
               c='red', marker='x', s=200, linewidths=3, label='Centroids')
    
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)', 
               fontsize=12)
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)', 
               fontsize=12)
    plt.title('Airline Clusters (2D PCA Projection)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def describe_clusters(airline_stats, labels, feature_cols):
    """
    Print cluster descriptions based on centroids.
    
    Args:
        airline_stats: DataFrame with airline features
        labels: Cluster labels
        feature_cols: Feature column names
    """
    print("\n" + "="*60)
    print("CLUSTER DESCRIPTIONS")
    print("="*60)
    
    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        cluster_airlines = airline_stats[mask]
        
        # Calculate mean features for this cluster
        mean_features = cluster_airlines[feature_cols].mean()
        
        print(f"\nCluster {cluster_id} ({np.sum(mask)} airlines):")
        print(f"  Airlines: {', '.join(cluster_airlines.index.tolist())}")
        print(f"  Mean features:")
        print(f"    Departure delay: {mean_features['mean_dep_delay']:.2f} min")
        print(f"    Arrival delay: {mean_features['mean_arr_delay']:.2f} min")
        print(f"    % Dep delayed ≥15min: {mean_features['pct_dep_delayed_15']:.1%}")
        print(f"    % Arr delayed ≥15min: {mean_features['pct_arr_delayed_15']:.1%}")
        print(f"    Cancel rate: {mean_features['cancel_rate']:.2%}")
        print(f"    Divert rate: {mean_features['divert_rate']:.2%}")
    
    print("\n" + "="*60)


def hierarchical_clustering(X_scaled, n_clusters, random_state=42):
    """
    Perform Ward hierarchical clustering.
    
    Args:
        X_scaled: Scaled feature matrix
        n_clusters: Number of clusters
        random_state: Random seed (for consistency)
        
    Returns:
        labels, linkage matrix
    """
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.metrics import adjusted_rand_score
    
    print("\nPerforming Ward hierarchical clustering...")
    
    # Fit hierarchical clustering
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    hc_labels = hc.fit_predict(X_scaled)
    
    # Compute linkage matrix for dendrogram
    linkage_matrix = linkage(X_scaled, method='ward')
    
    return hc_labels, linkage_matrix


def plot_dendrogram(linkage_matrix, output_path='fig_dendrogram.png', max_display=50):
    """
    Plot dendrogram (truncated if too many airlines).
    
    Args:
        linkage_matrix: Linkage matrix from hierarchical clustering
        output_path: Output file path
        max_display: Maximum leaves to show (truncate if more)
    """
    from scipy.cluster.hierarchy import dendrogram
    
    plt.figure(figsize=(12, 8))
    
    # Truncate if needed
    n_samples = len(linkage_matrix) + 1
    if max_display and n_samples > max_display:
        dendrogram(linkage_matrix, truncate_mode='lastp', p=max_display,
                  leaf_rotation=90., leaf_font_size=12., show_contracted=True)
        plt.title(f'Dendrogram (Truncated to {max_display} leaves)', 
                 fontsize=14, fontweight='bold')
    else:
        dendrogram(linkage_matrix, leaf_rotation=90., leaf_font_size=10.)
        plt.title('Dendrogram (Ward Linkage)', fontsize=14, fontweight='bold')
    
    plt.xlabel('Airline (or Cluster)', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Cluster airlines by operational reliability using k-means'
    )
    parser.add_argument('--data', default='./T_ONTIME_REPORTING.csv',
                       help='Path to BTS CSV file')
    parser.add_argument('--min-flights', type=int, default=2000,
                       help='Minimum flights per airline (default: 2000)')
    parser.add_argument('--k-min', type=int, default=3,
                       help='Minimum k to test (default: 3)')
    parser.add_argument('--k-max', type=int, default=8,
                       help='Maximum k to test (default: 8)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--with-hier', action='store_true',
                       help='Also run hierarchical clustering')
    
    args = parser.parse_args()
    
    # Run unit test
    test_feature_engineering()
    
    # Load and clean data
    df = load_data(args.data)
    df_clean = clean_data(df)
    
    # Engineer features
    airline_stats, feature_cols = engineer_features(df_clean, args.min_flights)
    
    # Extract feature matrix
    X = airline_stats[feature_cols].values
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find best k
    best_k, results = find_best_k(X_scaled, args.k_min, args.k_max, args.random_state)
    
    # Fit final k-means model
    print(f"\nFitting final k-means model with k={best_k}...")
    kmeans = KMeans(n_clusters=best_k, n_init=25, max_iter=500,
                   random_state=args.random_state, init='k-means++')
    labels = kmeans.fit_predict(X_scaled)
    
    # Get centroids in original scale
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    
    # Create outputs
    print("\nGenerating outputs...")
    
    # Save airline clusters
    output_df = airline_stats.copy()
    output_df['cluster'] = labels
    output_df.to_csv('airline_clusters.csv')
    print(f"  Saved airline_clusters.csv")
    
    # Save centroids
    centroids_df = pd.DataFrame(
        centroids_original,
        columns=feature_cols,
        index=[f'Cluster_{i}' for i in range(best_k)]
    )
    centroids_df.to_csv('cluster_centroids.csv')
    print(f"  Saved cluster_centroids.csv")
    
    # Generate plots
    plot_elbow(results)
    plot_silhouette(results)
    plot_pca_clusters(X_scaled, labels, centroids_scaled)
    
    # Describe clusters
    describe_clusters(airline_stats, labels, feature_cols)
    
    # Optional: Hierarchical clustering
    ari = None
    if args.with_hier:
        print("\n" + "="*60)
        print("HIERARCHICAL CLUSTERING")
        print("="*60)
        
        hc_labels, linkage_matrix = hierarchical_clustering(X_scaled, best_k, args.random_state)
        
        # Compute Adjusted Rand Index
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(labels, hc_labels)
        print(f"Adjusted Rand Index (k-means vs hierarchical): {ari:.4f}")
        
        # Plot dendrogram
        plot_dendrogram(linkage_matrix)
        
        # Save hierarchical results
        output_df_hc = airline_stats.copy()
        output_df_hc['cluster_hierarchical'] = hc_labels
        output_df_hc.to_csv('airline_clusters_hierarchical.csv')
        print(f"  Saved airline_clusters_hierarchical.csv")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    if ari is not None:
        return {'best_k': best_k, 'ari': ari}
    return {'best_k': best_k}


if __name__ == '__main__':
    main()

