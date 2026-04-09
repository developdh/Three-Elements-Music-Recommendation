"""FMA dataset loading and preprocessing."""

import pandas as pd
from config import FEATURES_CSV, TRACKS_CSV, NAN_ROW_THRESHOLD


def load_features():
    """Load FMA features.csv with multi-level headers."""
    features = pd.read_csv(FEATURES_CSV, index_col=0, header=[0, 1, 2])
    # Flatten MultiIndex columns: ('mfcc', 'mean', '01') -> 'mfcc_mean_01'
    features.columns = ['_'.join(col).strip() for col in features.columns.values]
    print(f"Features loaded: {features.shape[0]} tracks x {features.shape[1]} features")
    return features


def load_genres():
    """Load genre labels from tracks.csv."""
    tracks = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])
    genres = tracks[('track', 'genre_top')]
    genres.name = 'genre'
    print(f"Genres loaded: {genres.notna().sum()} tracks with genre labels")
    return genres


def clean_data(features, genres):
    """Clean and merge features with genre labels.

    1. Filter out tracks with no genre label
    2. Drop rows with >50% NaN features
    3. Median-impute remaining NaN values
    """
    # Merge on index (track_id)
    df = features.join(genres, how='inner')
    print(f"After genre join: {len(df)} tracks")

    # Remove rows without genre
    df = df.dropna(subset=['genre'])
    print(f"After genre NaN filter: {len(df)} tracks")

    # Separate features and genre
    genre_col = df['genre']
    feature_cols = df.drop(columns=['genre'])

    # Convert to numeric, coercing errors
    feature_cols = feature_cols.apply(pd.to_numeric, errors='coerce')

    # Drop rows with >50% NaN features
    nan_ratio = feature_cols.isna().mean(axis=1)
    mask = nan_ratio <= NAN_ROW_THRESHOLD
    feature_cols = feature_cols[mask]
    genre_col = genre_col[mask]
    print(f"After NaN row filter (>{NAN_ROW_THRESHOLD*100:.0f}% NaN): {len(feature_cols)} tracks")

    # Median impute remaining NaN
    nan_count_before = feature_cols.isna().sum().sum()
    feature_cols = feature_cols.fillna(feature_cols.median())
    print(f"Median imputed {nan_count_before} NaN values")

    # Drop any columns that are still all-NaN (constant features)
    feature_cols = feature_cols.dropna(axis=1, how='all')

    print(f"Final dataset: {feature_cols.shape[0]} tracks x {feature_cols.shape[1]} features, "
          f"{genre_col.nunique()} genres")
    print(f"Genre distribution:\n{genre_col.value_counts().to_string()}\n")

    return feature_cols, genre_col


def load_and_prepare():
    """Full data loading pipeline. Returns (features_df, genre_series)."""
    features = load_features()
    genres = load_genres()
    return clean_data(features, genres)
