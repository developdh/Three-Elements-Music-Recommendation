"""Visualization functions for PCA analysis results."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from config import OUTPUT_DIR, N_COMPONENTS, VARIANCE_THRESHOLD, N_TOP_FEATURES


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_scree(variance_analysis):
    """Plot individual + cumulative explained variance ratio (scree plot)."""
    ensure_output_dir()

    individual = variance_analysis['individual']
    cumulative = variance_analysis['cumulative']
    n_show = min(20, len(individual))  # Show up to 20 components

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Individual variance
    x = range(1, n_show + 1)
    ax1.bar(x, individual[:n_show], alpha=0.6, color='steelblue', label='Individual')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_xticks(range(1, n_show + 1))

    # Cumulative variance
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative[:n_show], 'ro-', linewidth=2, label='Cumulative')
    ax2.axhline(y=VARIANCE_THRESHOLD, color='green', linestyle='--',
                label=f'{VARIANCE_THRESHOLD*100:.0f}% threshold')
    ax2.axvline(x=N_COMPONENTS, color='orange', linestyle='--', alpha=0.7,
                label=f'{N_COMPONENTS} components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_ylim(0, 1.05)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    three_pc = variance_analysis['three_pc_cumulative']
    plt.title(f'PCA Scree Plot — 3 Components: {three_pc*100:.1f}% '
              f'({"PASS" if variance_analysis["is_sufficient"] else "FAIL"} ≥{VARIANCE_THRESHOLD*100:.0f}%)')
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'scree_plot.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_3d_scatter(projected_data, genres):
    """Plot 3D scatter of tracks colored by genre."""
    ensure_output_dir()

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    unique_genres = sorted(genres.unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_genres)))

    for genre, color in zip(unique_genres, colors):
        mask = genres.values == genre
        ax.scatter(projected_data[mask, 0], projected_data[mask, 1], projected_data[mask, 2],
                   c=[color], label=genre, alpha=0.4, s=8)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Music Tracks in 3-Component PCA Space (by Genre)')
    ax.legend(loc='upper left', fontsize=8, markerscale=3)

    path = os.path.join(OUTPUT_DIR, '3d_scatter.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_loadings_heatmap(loadings_analysis, pca, feature_names):
    """Plot heatmap of PCA loadings for top contributing features."""
    ensure_output_dir()

    # Collect all unique top feature indices across components
    n = min(N_COMPONENTS, len(loadings_analysis))
    top_feature_names = []
    for comp in loadings_analysis[:n]:
        for name, _ in comp['top_features']:
            if name not in top_feature_names:
                top_feature_names.append(name)

    # Build loadings matrix for these features
    feature_indices = [feature_names.index(name) for name in top_feature_names]
    loadings_matrix = pca.components_[:n, feature_indices].T

    fig, ax = plt.subplots(figsize=(8, max(6, len(top_feature_names) * 0.4)))
    sns.heatmap(loadings_matrix,
                xticklabels=[f'PC{i+1}' for i in range(n)],
                yticklabels=top_feature_names,
                cmap='RdBu_r', center=0, annot=True, fmt='.2f', ax=ax)
    ax.set_title(f'PCA Loadings — Top {N_TOP_FEATURES} Features per Component')
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'loadings_heatmap.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
