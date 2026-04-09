"""k-NN overlap evaluation for information loss quantification."""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from config import K_NEIGHBORS, KNN_OVERLAP_THRESHOLD


def compute_knn_overlap(full_scaled, reduced_features, genres):
    """Compare k-NN neighbors in full feature space vs reduced (3-PC) space.

    Args:
        full_scaled: Already-scaled full feature matrix (same scaler used for PCA)
        reduced_features: PCA-projected data (N x n_components)
        genres: genre labels Series

    Returns dict with:
        - overall_overlap: mean overlap ratio across all tracks
        - per_genre_overlap: {genre: mean_overlap}
        - is_sufficient: whether overall overlap >= threshold
    """

    knn_full = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric='euclidean')
    knn_full.fit(full_scaled)
    _, indices_full = knn_full.kneighbors(full_scaled)
    # Exclude self (first neighbor)
    indices_full = indices_full[:, 1:]

    # k-NN in reduced space
    knn_reduced = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric='euclidean')
    knn_reduced.fit(reduced_features)
    _, indices_reduced = knn_reduced.kneighbors(reduced_features)
    indices_reduced = indices_reduced[:, 1:]

    # Compute overlap per track
    overlaps = []
    for i in range(len(full_scaled)):
        full_set = set(indices_full[i])
        reduced_set = set(indices_reduced[i])
        overlap = len(full_set & reduced_set) / K_NEIGHBORS
        overlaps.append(overlap)

    overlaps = np.array(overlaps)
    overall = float(overlaps.mean())

    # Per-genre overlap
    genres_arr = genres.values
    unique_genres = sorted(set(genres_arr))
    per_genre = {}
    for genre in unique_genres:
        mask = genres_arr == genre
        per_genre[genre] = float(overlaps[mask].mean())

    return {
        'overall_overlap': overall,
        'per_genre_overlap': per_genre,
        'is_sufficient': overall >= KNN_OVERLAP_THRESHOLD,
    }
