"""Central configuration for the Three Elements Music Recommendation experiment."""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "fma_metadata")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

FEATURES_CSV = os.path.join(DATA_DIR, "features.csv")
TRACKS_CSV = os.path.join(DATA_DIR, "tracks.csv")

# PCA parameters
N_COMPONENTS = 3
VARIANCE_THRESHOLD = 0.8  # 80% cumulative explained variance

# Evaluation parameters
K_NEIGHBORS = 10
KNN_OVERLAP_THRESHOLD = 0.7  # 70% overlap for "similarity preserved"

# Interpretation
N_TOP_FEATURES = 5  # Top contributing features per principal component

# NaN handling
NAN_ROW_THRESHOLD = 0.5  # Drop rows with >50% NaN features
