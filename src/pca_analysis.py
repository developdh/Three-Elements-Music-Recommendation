"""PCA analysis and principal component interpretation."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from config import N_COMPONENTS, VARIANCE_THRESHOLD, N_TOP_FEATURES


def run_pca(features_df):
    """Standardize features and run PCA with all components.

    Returns: (pca, scaler, scaled_data, feature_names)
    """
    feature_names = features_df.columns.tolist()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_df)

    pca = PCA(n_components=None)
    pca.fit(scaled)

    return pca, scaler, scaled, feature_names


def get_variance_analysis(pca):
    """Analyze explained variance ratios.

    Returns dict with:
        - individual: per-component variance ratios
        - cumulative: cumulative variance ratios
        - n_components_for_threshold: min components needed for VARIANCE_THRESHOLD
        - three_pc_cumulative: cumulative variance for first 3 components
        - is_sufficient: whether 3 components meet the threshold
    """
    individual = pca.explained_variance_ratio_
    cumulative = np.cumsum(individual)

    three_pc = cumulative[N_COMPONENTS - 1] if len(cumulative) >= N_COMPONENTS else cumulative[-1]

    # Find minimum components for threshold
    n_for_threshold = np.argmax(cumulative >= VARIANCE_THRESHOLD) + 1
    if cumulative[-1] < VARIANCE_THRESHOLD:
        n_for_threshold = len(cumulative)

    return {
        'individual': individual,
        'cumulative': cumulative,
        'n_components_for_threshold': int(n_for_threshold),
        'three_pc_cumulative': float(three_pc),
        'is_sufficient': bool(three_pc >= VARIANCE_THRESHOLD),
    }


def get_loadings_analysis(pca, feature_names):
    """Analyze PCA loadings to interpret what each component represents.

    Returns list of dicts, one per component (up to N_COMPONENTS), each with:
        - component: component index (1-based)
        - top_features: list of (feature_name, loading_value) tuples
        - interpretation: string summary
    """
    results = []
    n = min(N_COMPONENTS, pca.n_components_)

    for i in range(n):
        loadings = pca.components_[i]
        abs_loadings = np.abs(loadings)
        top_indices = abs_loadings.argsort()[-N_TOP_FEATURES:][::-1]

        top_features = [(feature_names[idx], float(loadings[idx])) for idx in top_indices]

        # Group by feature type prefix for interpretation
        prefixes = [name.split('_')[0] for name, _ in top_features]
        dominant = max(set(prefixes), key=prefixes.count)

        results.append({
            'component': i + 1,
            'top_features': top_features,
            'dominant_type': dominant,
        })

    return results


def transform_to_n_components(pca, scaled_data, n=None):
    """Project data onto first n principal components."""
    if n is None:
        n = N_COMPONENTS
    return scaled_data @ pca.components_[:n].T
