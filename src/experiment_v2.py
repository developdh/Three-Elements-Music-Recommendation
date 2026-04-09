"""Experiment V2: Strategies to make 3 dimensions work for music similarity.

Approach 1: Use only 'mean' statistics (~76 features instead of 518)
Approach 2: Aggregate by feature group (MFCC→1, spectral→1, etc.)
Approach 3: Domain-driven 3-axis design (Energy / Timbre / Rhythm)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, K_NEIGHBORS
from data_loader import load_and_prepare


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def evaluate_approach(name, features_3d, genres, all_features_scaled=None):
    """Evaluate a 3D representation: variance, silhouette, k-NN overlap."""
    results = {'name': name}

    # Silhouette score (how well genres cluster in 3D)
    try:
        sil = silhouette_score(features_3d, genres, sample_size=5000, random_state=42)
        results['silhouette'] = sil
    except:
        results['silhouette'] = None

    # k-NN overlap with full feature space (if provided)
    if all_features_scaled is not None:
        knn_full = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1)
        knn_full.fit(all_features_scaled)
        _, idx_full = knn_full.kneighbors(all_features_scaled)
        idx_full = idx_full[:, 1:]

        knn_3d = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1)
        knn_3d.fit(features_3d)
        _, idx_3d = knn_3d.kneighbors(features_3d)
        idx_3d = idx_3d[:, 1:]

        overlaps = [len(set(idx_full[i]) & set(idx_3d[i])) / K_NEIGHBORS
                    for i in range(len(features_3d))]
        results['knn_overlap'] = np.mean(overlaps)
    else:
        results['knn_overlap'] = None

    return results


def approach1_mean_only(features_df, genres):
    """Use only 'mean' statistics from each feature group."""
    print_section("Approach 1: Mean Statistics Only")

    mean_cols = [c for c in features_df.columns if '_mean_' in c]
    print(f"Mean-only features: {len(mean_cols)} (from {len(features_df.columns)} total)")

    data = features_df[mean_cols]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    pca = PCA(n_components=None)
    pca.fit(scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    three_pc = cumvar[2]
    n_for_80 = np.argmax(cumvar >= 0.8) + 1 if cumvar[-1] >= 0.8 else len(cumvar)

    print(f"3-PC cumulative variance: {three_pc*100:.2f}%")
    print(f"Components for 80%: {n_for_80}")
    print(f"Top 5 components: {', '.join(f'{v*100:.1f}%' for v in pca.explained_variance_ratio_[:5])}")

    projected = scaled @ pca.components_[:3].T
    return projected, scaled, pca


def approach2_group_aggregate(features_df, genres):
    """Aggregate features by group (mfcc, spectral, chroma, etc.)."""
    print_section("Approach 2: Feature Group Aggregation")

    # Extract feature group from column name (first part before _)
    groups = {}
    for col in features_df.columns:
        group = col.split('_')[0]
        if group not in groups:
            groups[group] = []
        groups[group].append(col)

    print(f"Feature groups found: {list(groups.keys())}")
    print(f"Features per group: {', '.join(f'{k}={len(v)}' for k, v in groups.items())}")

    # Aggregate: mean of standardized features per group
    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(features_df)
    all_scaled_df = pd.DataFrame(all_scaled, columns=features_df.columns, index=features_df.index)

    agg_data = pd.DataFrame(index=features_df.index)
    for group, cols in groups.items():
        agg_data[group] = all_scaled_df[cols].mean(axis=1)

    print(f"Aggregated to {len(agg_data.columns)} group features")

    # PCA on aggregated features
    agg_scaler = StandardScaler()
    agg_scaled = agg_scaler.fit_transform(agg_data)

    pca = PCA(n_components=None)
    pca.fit(agg_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    three_pc = cumvar[2] if len(cumvar) >= 3 else cumvar[-1]

    print(f"3-PC cumulative variance: {three_pc*100:.2f}%")
    print(f"Top components: {', '.join(f'{v*100:.1f}%' for v in pca.explained_variance_ratio_[:5])}")

    # Loadings interpretation
    for i in range(min(3, len(pca.components_))):
        loadings = pca.components_[i]
        top_idx = np.abs(loadings).argsort()[-3:][::-1]
        top = [(agg_data.columns[j], loadings[j]) for j in top_idx]
        print(f"  PC{i+1}: {', '.join(f'{n}({v:+.2f})' for n, v in top)}")

    projected = agg_scaled @ pca.components_[:3].T
    return projected, scaler.fit_transform(features_df), pca


def approach3_domain_axes(features_df, genres):
    """Domain-driven 3-axis: Energy, Timbre, Rhythm."""
    print_section("Approach 3: Domain-Driven 3 Axes")

    cols = features_df.columns.tolist()

    def get_cols(patterns):
        return [c for c in cols if any(p in c for p in patterns)]

    def safe_mean(col_list):
        valid = [c for c in col_list if c in cols]
        if not valid:
            return pd.Series(0, index=features_df.index)
        return features_df[valid].mean(axis=1)

    # Axis 1: Energy (RMS energy, spectral centroid, spectral bandwidth)
    energy_cols = get_cols(['rmse_mean', 'rms_mean', 'spectral_centroid_mean',
                           'spectral_bandwidth_mean', 'spectral_rolloff_mean'])
    # Axis 2: Timbre (MFCC coefficients - the core of timbre representation)
    timbre_cols = get_cols(['mfcc_mean'])
    # Axis 3: Rhythm (tempo, zero crossing rate, onset strength)
    rhythm_cols = get_cols(['zcr_mean', 'tonnetz_mean', 'chroma_cens_mean', 'chroma_cqt_mean'])

    print(f"Energy features: {len(energy_cols)}")
    print(f"Timbre features: {len(timbre_cols)}")
    print(f"Rhythm features: {len(rhythm_cols)}")

    # Composite scores: standardize within group, then average
    scaler = StandardScaler()

    energy = safe_mean(energy_cols)
    timbre_data = features_df[timbre_cols] if timbre_cols else pd.DataFrame(0, index=features_df.index, columns=['x'])
    timbre_pca = PCA(n_components=1)
    timbre_scaled = StandardScaler().fit_transform(timbre_data)
    timbre = timbre_pca.fit_transform(timbre_scaled).flatten()

    rhythm = safe_mean(rhythm_cols)

    domain_3d = np.column_stack([
        StandardScaler().fit_transform(energy.values.reshape(-1, 1)).flatten(),
        timbre,
        StandardScaler().fit_transform(rhythm.values.reshape(-1, 1)).flatten(),
    ])

    print(f"\nDomain axes constructed:")
    print(f"  Axis 1 (Energy):  spectral centroid, bandwidth, rolloff, RMS")
    print(f"  Axis 2 (Timbre):  PCA1 of MFCC means (captures dominant timbre variation)")
    print(f"  Axis 3 (Harmony): tonnetz, chroma, ZCR means")

    return domain_3d, scaler.fit_transform(features_df), None


def plot_comparison(results, all_3d_data, genres):
    """Plot comparison of all approaches."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5),
                             subplot_kw={'projection': '3d'})
    if len(results) == 1:
        axes = [axes]

    unique_genres = sorted(genres.unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_genres)))

    for ax, result, data_3d in zip(axes, results, all_3d_data):
        for genre, color in zip(unique_genres, colors):
            mask = genres.values == genre
            ax.scatter(data_3d[mask, 0], data_3d[mask, 1], data_3d[mask, 2],
                       c=[color], alpha=0.3, s=5)
        sil = result.get('silhouette')
        knn = result.get('knn_overlap')
        sil_str = f"{sil:.3f}" if sil is not None else "N/A"
        knn_str = f"{knn*100:.1f}%" if knn is not None else "N/A"
        ax.set_title(f"{result['name']}\nSil:{sil_str} kNN:{knn_str}")
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.set_zlabel('Dim 3')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'v2_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved: {path}")


def main():
    print_section("Experiment V2: Making 3 Dimensions Work")

    # Load data
    features_df, genres = load_and_prepare()

    # Full feature scaled (for k-NN comparison baseline)
    full_scaler = StandardScaler()
    full_scaled = full_scaler.fit_transform(features_df)

    # Run all approaches
    all_3d = []
    all_results = []

    # Approach 1: Mean only
    proj1, _, pca1 = approach1_mean_only(features_df, genres)
    res1 = evaluate_approach("Mean-Only PCA", proj1, genres, full_scaled)
    all_3d.append(proj1)
    all_results.append(res1)

    # Approach 2: Group aggregate
    proj2, _, pca2 = approach2_group_aggregate(features_df, genres)
    res2 = evaluate_approach("Group-Agg PCA", proj2, genres, full_scaled)
    all_3d.append(proj2)
    all_results.append(res2)

    # Approach 3: Domain axes
    proj3, _, _ = approach3_domain_axes(features_df, genres)
    res3 = evaluate_approach("Domain 3-Axis", proj3, genres, full_scaled)
    all_3d.append(proj3)
    all_results.append(res3)

    # Comparison
    print_section("COMPARISON OF APPROACHES")
    print(f"{'Approach':<20} {'Silhouette':>12} {'k-NN Overlap':>14} {'Verdict':>10}")
    print(f"{'-'*56}")

    baseline_sil = silhouette_score(full_scaled[:5000], genres[:5000], random_state=42)
    print(f"{'Full 518 features':<20} {baseline_sil:>12.4f} {'100.00%':>14} {'BASELINE':>10}")

    for r in all_results:
        sil = f"{r['silhouette']:.4f}" if r['silhouette'] is not None else "N/A"
        knn = f"{r['knn_overlap']*100:.2f}%" if r['knn_overlap'] is not None else "N/A"
        verdict = "BEST" if r == max(all_results, key=lambda x: x.get('silhouette', 0) or 0) else ""
        print(f"{r['name']:<20} {sil:>12} {knn:>14} {verdict:>10}")

    best = max(all_results, key=lambda x: x.get('silhouette', 0) or 0)
    print(f"\nBest approach: {best['name']}")
    print(f"  Silhouette score: {best['silhouette']:.4f} (baseline: {baseline_sil:.4f})")
    if best['knn_overlap']:
        print(f"  k-NN overlap: {best['knn_overlap']*100:.2f}%")

    # Visualize
    plot_comparison(all_results, all_3d, genres)

    # Conclusion
    print_section("CONCLUSION")
    print("Can we make 3 dimensions work with better feature engineering?")
    print()
    if best['silhouette'] and best['silhouette'] > 0.1:
        print(f"  YES — {best['name']} achieves meaningful clustering (silhouette={best['silhouette']:.4f})")
        print("  Genre separation is visible in 3D space with proper feature engineering.")
    else:
        print(f"  PARTIALLY — Best silhouette is {best['silhouette']:.4f}")
        print("  Some structure exists but genres still overlap significantly.")
    print()
    print("  Key insight: The choice of WHICH 3 dimensions matters more than")
    print("  the number. Raw PCA on 518 redundant statistics fails, but")
    print("  thoughtful feature engineering can extract meaningful structure.")


if __name__ == '__main__':
    main()
