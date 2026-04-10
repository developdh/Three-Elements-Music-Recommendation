"""Export PCA coordinates + metadata to JSON for the web prototype."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from data_loader import load_and_prepare


def main():
    features_df, genres = load_and_prepare()

    # Group-aggregate
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

    # Quantile normalization: each axis gets uniform distribution 0-100
    # This ensures slider movement always covers equal density of tracks
    # (vs min-max which clusters everything in the center)
    normalized = np.zeros_like(pca_3d)
    for axis in range(3):
        ranks = rankdata(pca_3d[:, axis], method='average')
        normalized[:, axis] = (ranks - 1) / (len(ranks) - 1) * 100

    # Build track data
    tracks = []
    for i in range(len(features_df)):
        tracks.append({
            'id': int(features_df.index[i]),
            'genre': genres.values[i],
            'x': round(float(normalized[i, 0]), 2),
            'y': round(float(normalized[i, 1]), 2),
            'z': round(float(normalized[i, 2]), 2),
        })

    # Metadata
    data = {
        'axes': {
            'x': {'name': 'Brightness / Energy', 'low': 'Dark / Muted', 'high': 'Bright / Sharp'},
            'y': {'name': 'Intensity / Loudness', 'low': 'Quiet / Soft', 'high': 'Loud / Intense'},
            'z': {'name': 'Harmonic Complexity', 'low': 'Simple', 'high': 'Complex'},
        },
        'genres': sorted(genres.unique().tolist()),
        'stats': {
            'total_tracks': len(tracks),
            'variance_explained': round(float(sum(pca.explained_variance_ratio_) * 100), 1),
        },
        'tracks': tracks,
    }

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'music_data.json')

    with open(out_path, 'w') as f:
        json.dump(data, f)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Exported {len(tracks)} tracks to {out_path} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
