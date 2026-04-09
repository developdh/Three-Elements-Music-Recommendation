"""Experiment V4: Cluster scalability — do natural clusters hold up as k increases?

V3 showed k=2 clusters dramatically outperform 16 genres in 3D.
But k=2 is very coarse. This experiment tests whether finer-grained
clusters (k=3~10) still outperform genres, or if the advantage disappears
as we approach genre-level granularity.

Key questions:
1. At what k does cluster quality start degrading?
2. Is there a "sweet spot" k that balances granularity and separation?
3. Do higher-k clusters still beat genres on all metrics?
4. What do the finer clusters represent?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
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


def aggregate_and_pca(features_df):
    """Aggregate to 6 groups, scale, PCA to 3D."""
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(
        scaler.fit_transform(features_df),
        columns=features_df.columns, index=features_df.index
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

    agg_scaler = StandardScaler()
    agg_scaled = agg_scaler.fit_transform(agg)

    pca = PCA(n_components=3)
    pca_3d = pca.fit_transform(agg_scaled)

    return agg, agg_scaled, pca_3d, list(sorted(groups.keys()))


def compute_metrics(pca_3d, labels):
    """Compute silhouette and k-NN purity for given labels."""
    sil = silhouette_score(pca_3d, labels, sample_size=5000, random_state=42)

    knn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1)
    knn.fit(pca_3d)
    _, indices = knn.kneighbors(pca_3d)
    indices = indices[:, 1:]

    purities = []
    for i in range(len(labels)):
        neighbor_labels = labels[indices[i]]
        purity = np.mean(neighbor_labels == labels[i])
        purities.append(purity)

    return sil, np.mean(purities)


def run_all_k(agg_scaled, pca_3d, genres, max_k=10):
    """Run K-Means for k=2..max_k and compute all metrics."""
    print_section("Scaling k from 2 to 10")

    le = LabelEncoder()
    genre_encoded = le.fit_transform(genres)

    # Genre baseline
    genre_sil, genre_purity = compute_metrics(pca_3d, genre_encoded)
    print(f"Genre baseline (k=16): silhouette={genre_sil:.4f}, purity={genre_purity*100:.1f}%")
    print()

    results = []
    all_labels = {}

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(agg_scaled)
        all_labels[k] = labels

        sil, purity = compute_metrics(pca_3d, labels)
        ari = adjusted_rand_score(genre_encoded, labels)

        # Advantage over genre
        sil_advantage = sil - genre_sil
        purity_ratio = purity / genre_purity if genre_purity > 0 else float('inf')

        results.append({
            'k': k,
            'silhouette': sil,
            'purity': purity,
            'ari_vs_genre': ari,
            'sil_advantage': sil_advantage,
            'purity_ratio': purity_ratio,
            'beats_genre_sil': sil > genre_sil,
            'beats_genre_purity': purity > genre_purity,
        })

        status = "BETTER" if sil > genre_sil else "WORSE"
        print(f"  k={k:2d}: sil={sil:+.4f} ({status} by {sil_advantage:+.4f}), "
              f"purity={purity*100:.1f}%, ARI={ari:.3f}")

    return results, all_labels, genre_sil, genre_purity


def analyze_sweet_spot(results, genre_sil):
    """Find the optimal k that balances granularity and quality."""
    print_section("Sweet Spot Analysis")

    # All k values that beat genre
    beating = [r for r in results if r['beats_genre_sil']]
    print(f"k values that beat genre silhouette ({genre_sil:.4f}):")
    if beating:
        for r in beating:
            print(f"  k={r['k']}: silhouette={r['silhouette']:.4f} "
                  f"(+{r['sil_advantage']:.4f}), purity={r['purity']*100:.1f}%")
    else:
        print("  None — genre beats all cluster counts in silhouette")

    # Best by silhouette
    best_sil = max(results, key=lambda r: r['silhouette'])
    # Best by purity
    best_pur = max(results, key=lambda r: r['purity'])
    # Best balanced (silhouette * purity)
    best_bal = max(results, key=lambda r: r['silhouette'] * r['purity']
                   if r['silhouette'] > 0 else -1)

    print(f"\nBest by silhouette:  k={best_sil['k']} (sil={best_sil['silhouette']:.4f})")
    print(f"Best by purity:      k={best_pur['k']} (purity={best_pur['purity']*100:.1f}%)")
    print(f"Best balanced:       k={best_bal['k']} (sil={best_bal['silhouette']:.4f}, "
          f"purity={best_bal['purity']*100:.1f}%)")

    return best_bal['k']


def analyze_best_k_clusters(agg_data, labels, group_names, genres, k):
    """Detailed analysis of the sweet-spot k."""
    print_section(f"Detailed Cluster Analysis (k={k})")

    df = agg_data.copy()
    df['cluster'] = labels
    df['genre'] = genres.values

    # Cluster profiles
    print("Cluster profiles:")
    print(f"  {'Cluster':>8} {'Size':>7}", end='')
    for g in group_names:
        print(f" {g:>9}", end='')
    print(f"  {'Interpretation'}")

    for c in range(k):
        mask = labels == c
        size = mask.sum()
        profile = agg_data[mask].mean()

        # Interpret
        high = [(g, v) for g, v in profile.items() if v > 0.2]
        low = [(g, v) for g, v in profile.items() if v < -0.2]
        parts = [f"{g}+" for g, _ in sorted(high, key=lambda x: -x[1])]
        parts += [f"{g}-" for g, _ in sorted(low, key=lambda x: x[1])]
        interp = " ".join(parts) if parts else "neutral"

        print(f"  {c:>8} {size:>7}", end='')
        for g in group_names:
            print(f" {profile[g]:>+9.3f}", end='')
        print(f"  {interp}")

    # Top genres per cluster
    print(f"\nTop 3 genres per cluster:")
    ct = pd.crosstab(
        pd.Series(labels, name='Cluster'),
        genres.reset_index(drop=True).rename('Genre'),
        normalize='index'
    )
    for c in range(k):
        row = ct.iloc[c].sort_values(ascending=False).head(3)
        parts = [f"{genre}({pct*100:.0f}%)" for genre, pct in row.items()]
        print(f"  Cluster {c}: {', '.join(parts)}")


def plot_scalability(results, genre_sil, genre_purity, pca_3d, all_labels, genres, best_k):
    """Generate comprehensive scalability plots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Metrics comparison chart
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ks = [r['k'] for r in results]
    sils = [r['silhouette'] for r in results]
    purs = [r['purity'] * 100 for r in results]
    aris = [r['ari_vs_genre'] for r in results]

    # Silhouette
    ax1.plot(ks, sils, 'bo-', linewidth=2, label='Cluster')
    ax1.axhline(y=genre_sil, color='red', linestyle='--', linewidth=2, label=f'Genre ({genre_sil:.3f})')
    ax1.axvline(x=best_k, color='green', linestyle=':', alpha=0.7, label=f'Sweet spot k={best_k}')
    ax1.fill_between(ks, genre_sil, sils, alpha=0.15,
                     color=['green' if s > genre_sil else 'red' for s in sils])
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Silhouette: Clusters vs Genre')
    ax1.legend(fontsize=8)
    ax1.set_xticks(ks)

    # Purity
    ax2.plot(ks, purs, 'go-', linewidth=2, label='Cluster')
    ax2.axhline(y=genre_purity * 100, color='red', linestyle='--', linewidth=2,
                label=f'Genre ({genre_purity*100:.1f}%)')
    ax2.axvline(x=best_k, color='green', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('k-NN Purity (%)')
    ax2.set_title('k-NN Purity: Clusters vs Genre')
    ax2.legend(fontsize=8)
    ax2.set_xticks(ks)

    # ARI
    ax3.plot(ks, aris, 'ro-', linewidth=2)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax3.axvline(x=best_k, color='green', linestyle=':', alpha=0.7)
    ax3.set_xlabel('Number of Clusters (k)')
    ax3.set_ylabel('Adjusted Rand Index')
    ax3.set_title('Cluster-Genre Agreement')
    ax3.set_xticks(ks)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'v4_scalability.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # 2. 3D scatter for best k
    labels = all_labels[best_k]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.Set1(np.linspace(0, 1, best_k))
    for c in range(best_k):
        mask = labels == c
        ax.scatter(pca_3d[mask, 0], pca_3d[mask, 1], pca_3d[mask, 2],
                   c=[colors[c]], alpha=0.3, s=5, label=f'Cluster {c}')

    ax.set_title(f'Natural Clusters (k={best_k}) in 3D PCA Space')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend(fontsize=8, markerscale=3, loc='upper left')

    path = os.path.join(OUTPUT_DIR, f'v4_clusters_k{best_k}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    print_section("Experiment V4: Cluster Scalability")
    print("Does the advantage of natural clusters over genres hold")
    print("as we increase the number of clusters?")

    # Load data
    features_df, genres = load_and_prepare()

    # Aggregate and PCA
    agg_data, agg_scaled, pca_3d, group_names = aggregate_and_pca(features_df)
    print(f"Aggregated to {len(group_names)} groups, projected to 3D")

    # Run all k values
    results, all_labels, genre_sil, genre_purity = run_all_k(agg_scaled, pca_3d, genres)

    # Find sweet spot
    best_k = analyze_sweet_spot(results, genre_sil)

    # Detailed analysis of sweet spot
    analyze_best_k_clusters(agg_data, all_labels[best_k], group_names, genres, best_k)

    # Plots
    print_section("Generating Visualizations")
    plot_scalability(results, genre_sil, genre_purity, pca_3d, all_labels, genres, best_k)

    # Final conclusion
    print_section("CONCLUSION")

    beats_count = sum(1 for r in results if r['beats_genre_sil'])
    total = len(results)
    max_k_beating = max((r['k'] for r in results if r['beats_genre_sil']), default=0)

    print(f"Clusters that beat genre silhouette: {beats_count}/{total}")
    print(f"Highest k still beating genre: k={max_k_beating}")
    print()

    if beats_count == total:
        print("  ALL cluster counts (k=2~10) outperform genre labels.")
        print("  The advantage of unsupervised clustering is robust across granularities.")
    elif beats_count > total / 2:
        print(f"  Majority ({beats_count}/{total}) of cluster counts outperform genres.")
        print(f"  The advantage holds up to k={max_k_beating}.")
    else:
        print(f"  Only {beats_count}/{total} cluster counts beat genres.")
        print(f"  The advantage is limited to coarse granularity (k <= {max_k_beating}).")

    print()
    print(f"  Sweet spot: k={best_k}")
    sweet = next(r for r in results if r['k'] == best_k)
    print(f"    Silhouette: {sweet['silhouette']:.4f} (genre: {genre_sil:.4f})")
    print(f"    k-NN purity: {sweet['purity']*100:.1f}% (genre: {genre_purity*100:.1f}%)")
    print(f"    ARI vs genre: {sweet['ari_vs_genre']:.3f}")


if __name__ == '__main__':
    main()
