"""Experiment V3: Unsupervised clustering to discover natural music groups.

Instead of using genre labels (which don't cluster well in audio space),
discover what natural groupings emerge from audio features alone.

Key questions:
1. How many natural clusters exist in music audio feature space?
2. What characterizes each cluster? (What do they "sound like"?)
3. Do these clusters align with genres, or reveal a different structure?
4. Can 3 dimensions capture these natural clusters better than genres?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUT_DIR, K_NEIGHBORS
from data_loader import load_and_prepare


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def aggregate_features(features_df):
    """Aggregate 518 features into 6 feature groups (best from V2)."""
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(
        scaler.fit_transform(features_df),
        columns=features_df.columns,
        index=features_df.index
    )

    groups = {}
    for col in features_df.columns:
        group = col.split('_')[0]
        if group not in groups:
            groups[group] = []
        groups[group].append(col)

    agg = pd.DataFrame(index=features_df.index)
    for group, cols in sorted(groups.items()):
        agg[group] = scaled_df[cols].mean(axis=1)

    return agg, list(groups.keys())


def find_optimal_clusters(data_scaled, max_k=15):
    """Find optimal cluster count using elbow + silhouette methods."""
    print_section("Finding Optimal Cluster Count")

    inertias = []
    silhouettes = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(data_scaled)
        inertias.append(kmeans.inertia_)

        sil = silhouette_score(data_scaled, labels, sample_size=5000, random_state=42)
        silhouettes.append(sil)
        print(f"  k={k:2d}: inertia={kmeans.inertia_:12.1f}  silhouette={sil:.4f}")

    best_k = k_range[np.argmax(silhouettes)]
    print(f"\nBest k by silhouette: {best_k} (score={max(silhouettes):.4f})")

    return list(k_range), inertias, silhouettes, best_k


def analyze_clusters(data_agg, labels, group_names, genres):
    """Analyze what characterizes each cluster."""
    print_section("Cluster Analysis")

    df = data_agg.copy()
    df['cluster'] = labels
    df['genre'] = genres.values

    n_clusters = len(set(labels))

    # Cluster sizes
    print("Cluster sizes:")
    for c in range(n_clusters):
        count = (labels == c).sum()
        pct = count / len(labels) * 100
        print(f"  Cluster {c}: {count:6d} tracks ({pct:.1f}%)")

    # Feature profile per cluster
    print("\nCluster feature profiles (mean of standardized group features):")
    print(f"  {'Cluster':>8}", end='')
    for g in group_names:
        print(f"  {g:>10}", end='')
    print()

    cluster_profiles = []
    for c in range(n_clusters):
        mask = labels == c
        profile = data_agg[mask].mean()
        cluster_profiles.append(profile)
        print(f"  {c:>8}", end='')
        for g in group_names:
            print(f"  {profile[g]:>10.3f}", end='')
        print()

    # Dominant characteristic per cluster
    print("\nCluster interpretations:")
    for c, profile in enumerate(cluster_profiles):
        high = profile.nlargest(2)
        low = profile.nsmallest(1)
        desc_parts = []
        for feat, val in high.items():
            if val > 0.3:
                desc_parts.append(f"high {feat}")
        for feat, val in low.items():
            if val < -0.3:
                desc_parts.append(f"low {feat}")
        desc = ", ".join(desc_parts) if desc_parts else "moderate all"
        print(f"  Cluster {c}: {desc}")

    return cluster_profiles


def cluster_genre_comparison(labels, genres):
    """Compare discovered clusters with genre labels."""
    print_section("Cluster vs Genre Comparison")

    # Adjusted Rand Index
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    genre_encoded = le.fit_transform(genres)
    ari = adjusted_rand_score(genre_encoded, labels)
    print(f"Adjusted Rand Index (cluster vs genre): {ari:.4f}")
    print(f"  (1.0 = perfect match, 0.0 = random, <0 = worse than random)")

    # Cross-tabulation
    ct = pd.crosstab(
        pd.Series(labels, name='Cluster'),
        genres.reset_index(drop=True).rename('Genre'),
        normalize='index'  # Normalize by cluster (row percentages)
    )
    # Show top 3 genres per cluster
    print("\nTop genres per cluster (% of cluster):")
    for c in range(len(ct)):
        row = ct.iloc[c].sort_values(ascending=False)
        top3 = row.head(3)
        parts = [f"{genre}({pct*100:.0f}%)" for genre, pct in top3.items()]
        print(f"  Cluster {c}: {', '.join(parts)}")

    # Reverse: which cluster dominates each genre?
    ct_by_genre = pd.crosstab(
        genres.reset_index(drop=True).rename('Genre'),
        pd.Series(labels, name='Cluster'),
        normalize='index'
    )
    print("\nDominant cluster per genre:")
    for genre in sorted(ct_by_genre.index):
        row = ct_by_genre.loc[genre]
        dom = row.idxmax()
        pct = row.max()
        spread = (row > 0.15).sum()
        print(f"  {genre:20s}: Cluster {dom} ({pct*100:.0f}%), spread across {spread} clusters")

    return ari


def evaluate_3d_clustering(pca_3d, labels, genres):
    """Compare cluster-based vs genre-based similarity in 3D."""
    print_section("3D Similarity Evaluation")

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    genre_encoded = le.fit_transform(genres)

    # Silhouette with clusters vs genres
    sil_cluster = silhouette_score(pca_3d, labels, sample_size=5000, random_state=42)
    sil_genre = silhouette_score(pca_3d, genre_encoded, sample_size=5000, random_state=42)

    print(f"Silhouette score in 3D PCA space:")
    print(f"  With cluster labels: {sil_cluster:.4f}")
    print(f"  With genre labels:   {sil_genre:.4f}")
    print(f"  Improvement: {(sil_cluster - sil_genre):.4f} ({(sil_cluster/sil_genre - 1)*100:+.1f}%)"
          if sil_genre != 0 else "")

    # k-NN purity: what fraction of k nearest neighbors share the same label?
    knn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1)
    knn.fit(pca_3d)
    _, indices = knn.kneighbors(pca_3d)
    indices = indices[:, 1:]  # Exclude self

    def knn_purity(all_labels):
        purities = []
        for i in range(len(all_labels)):
            neighbor_labels = all_labels[indices[i]]
            purity = np.mean(neighbor_labels == all_labels[i])
            purities.append(purity)
        return np.mean(purities)

    purity_cluster = knn_purity(labels)
    purity_genre = knn_purity(genre_encoded)

    print(f"\nk-NN purity (k={K_NEIGHBORS}):")
    print(f"  With cluster labels: {purity_cluster:.4f} ({purity_cluster*100:.1f}%)")
    print(f"  With genre labels:   {purity_genre:.4f} ({purity_genre*100:.1f}%)")

    return sil_cluster, sil_genre, purity_cluster, purity_genre


def plot_results(k_range, inertias, silhouettes, best_k,
                 pca_3d, labels, genres, cluster_profiles, group_names):
    """Generate all plots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Elbow + Silhouette plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(k_range, inertias, 'bo-')
    ax1.axvline(x=best_k, color='red', linestyle='--', label=f'Best k={best_k}')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.legend()

    ax2.plot(k_range, silhouettes, 'go-')
    ax2.axvline(x=best_k, color='red', linestyle='--', label=f'Best k={best_k}')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'v3_optimal_k.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # 2. Side-by-side 3D: clusters vs genres
    fig = plt.figure(figsize=(14, 6))

    unique_clusters = sorted(set(labels))
    cluster_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))

    ax1 = fig.add_subplot(121, projection='3d')
    for c, color in zip(unique_clusters, cluster_colors):
        mask = labels == c
        ax1.scatter(pca_3d[mask, 0], pca_3d[mask, 1], pca_3d[mask, 2],
                    c=[color], alpha=0.3, s=5, label=f'Cluster {c}')
    ax1.set_title('Natural Clusters (Unsupervised)')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.legend(fontsize=7, markerscale=3, loc='upper left')

    unique_genres = sorted(genres.unique())
    genre_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_genres)))

    ax2 = fig.add_subplot(122, projection='3d')
    for genre, color in zip(unique_genres, genre_colors):
        mask = genres.values == genre
        ax2.scatter(pca_3d[mask, 0], pca_3d[mask, 1], pca_3d[mask, 2],
                    c=[color], alpha=0.3, s=5, label=genre)
    ax2.set_title('Genre Labels (Supervised)')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    ax2.legend(fontsize=6, markerscale=3, loc='upper left')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'v3_clusters_vs_genres.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # 3. Cluster feature profile heatmap
    profile_df = pd.DataFrame(cluster_profiles, columns=group_names,
                              index=[f'Cluster {i}' for i in range(len(cluster_profiles))])

    fig, ax = plt.subplots(figsize=(8, max(4, len(cluster_profiles) * 0.6)))
    sns.heatmap(profile_df, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                ax=ax, cbar_kws={'label': 'Standardized Mean'})
    ax.set_title('Cluster Feature Profiles')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'v3_cluster_profiles.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    print_section("Experiment V3: Unsupervised Music Clustering")
    print("Goal: Discover natural music groups from audio features alone,")
    print("without relying on genre labels.")

    # Load and prepare data
    features_df, genres = load_and_prepare()

    # Aggregate to 6 feature groups (best approach from V2)
    print_section("Feature Aggregation (6 Groups)")
    agg_data, group_names = aggregate_features(features_df)
    print(f"Aggregated: {features_df.shape[1]} features -> {agg_data.shape[1]} groups: {group_names}")

    # Scale aggregated features
    scaler = StandardScaler()
    agg_scaled = scaler.fit_transform(agg_data)

    # PCA to 3D on aggregated features
    pca = PCA(n_components=3)
    pca_3d = pca.fit_transform(agg_scaled)
    cumvar = sum(pca.explained_variance_ratio_)
    print(f"3-PC cumulative variance (on aggregated): {cumvar*100:.1f}%")

    # Find optimal clusters
    k_range, inertias, silhouettes, best_k = find_optimal_clusters(agg_scaled)

    # Run K-Means with optimal k
    print_section(f"K-Means Clustering (k={best_k})")
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(agg_scaled)

    # Analyze clusters
    profiles = analyze_clusters(agg_data, labels, group_names, genres)

    # Compare with genres
    ari = cluster_genre_comparison(labels, genres)

    # Evaluate in 3D
    sil_c, sil_g, pur_c, pur_g = evaluate_3d_clustering(pca_3d, labels, genres)

    # Generate plots
    print_section("Generating Visualizations")
    plot_results(k_range, inertias, silhouettes, best_k,
                 pca_3d, labels, genres, profiles, group_names)

    # Final conclusion
    print_section("CONCLUSION")
    print(f"Natural clusters found: {best_k}")
    print(f"Adjusted Rand Index (clusters vs genres): {ari:.4f}")
    print()
    print(f"In 3D PCA space:")
    print(f"  Cluster silhouette: {sil_c:.4f}  vs  Genre silhouette: {sil_g:.4f}")
    print(f"  Cluster k-NN purity: {pur_c*100:.1f}%  vs  Genre k-NN purity: {pur_g*100:.1f}%")
    print()

    if sil_c > sil_g:
        improvement = (sil_c - sil_g) / abs(sil_g) * 100 if sil_g != 0 else float('inf')
        print(f"  Natural clusters fit 3D space {improvement:.0f}% better than genres.")
        print(f"  This confirms: genre labels are a poor match for audio similarity.")
        print(f"  Music naturally groups by SOUND characteristics, not cultural genre labels.")
    else:
        print(f"  Neither clusters nor genres separate well in 3D.")
        print(f"  Music similarity may require more nuanced representation.")

    print()
    print(f"  The {best_k} natural clusters represent groups of music that")
    print(f"  actually sound similar, rather than sharing a genre label.")
    print(f"  This supports using continuous audio features instead of")
    print(f"  discrete genre categories for music recommendation.")


if __name__ == '__main__':
    main()
