"""Experiment V5: 3-Axis Recommendation System vs Genre-Based Recommendation

Build and compare three recommendation approaches:
1. Genre-Based: Recommend songs from the same genre (current standard)
2. 3-Axis (PCA): k-NN in 3D group-aggregated PCA space
3. Full-Feature (Oracle): k-NN in full 518-feature space (ideal baseline)

Evaluation metrics:
- Oracle overlap: How well each method approximates full-feature recommendations
- Intra-list similarity: Are recommendations actually similar to each other?
- Genre diversity: Do recommendations span multiple genres? (desirable for discovery)
- Coverage: What fraction of the catalog gets recommended?
- Mean reciprocal rank: How quickly does a relevant item appear?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUT_DIR, K_NEIGHBORS
from data_loader import load_and_prepare


K = K_NEIGHBORS  # Number of recommendations


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def prepare_data(features_df):
    """Prepare all feature representations."""
    # Full features (oracle)
    full_scaler = StandardScaler()
    full_scaled = full_scaler.fit_transform(features_df)

    # Group-aggregated features
    scaled_df = pd.DataFrame(full_scaled, columns=features_df.columns, index=features_df.index)
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

    # PCA to 3D
    pca = PCA(n_components=3)
    pca_3d = pca.fit_transform(agg_scaled)

    return full_scaled, pca_3d, list(sorted(groups.keys()))


class GenreRecommender:
    """Recommend songs from the same genre."""

    def __init__(self, genres, full_scaled):
        self.genres = genres.values
        self.full_scaled = full_scaled
        # Pre-build genre index
        self.genre_indices = {}
        for i, g in enumerate(self.genres):
            if g not in self.genre_indices:
                self.genre_indices[g] = []
            self.genre_indices[g].append(i)

    def recommend(self, query_idx, k=K):
        """Return k recommendations from the same genre, closest by full features."""
        genre = self.genres[query_idx]
        candidates = [i for i in self.genre_indices[genre] if i != query_idx]

        if len(candidates) <= k:
            return np.array(candidates)

        # Rank by cosine similarity within genre
        query_vec = self.full_scaled[query_idx].reshape(1, -1)
        cand_vecs = self.full_scaled[candidates]
        sims = cosine_similarity(query_vec, cand_vecs)[0]
        top_k = np.argsort(sims)[-k:][::-1]
        return np.array([candidates[i] for i in top_k])


class ThreeAxisRecommender:
    """Recommend songs using k-NN in 3D PCA space."""

    def __init__(self, pca_3d):
        self.pca_3d = pca_3d
        self.knn = NearestNeighbors(n_neighbors=K + 1, metric='euclidean')
        self.knn.fit(pca_3d)

    def recommend(self, query_idx, k=K):
        query = self.pca_3d[query_idx].reshape(1, -1)
        _, indices = self.knn.kneighbors(query)
        return indices[0, 1:k+1]  # Exclude self


class OracleRecommender:
    """Recommend songs using k-NN in full feature space (ideal baseline)."""

    def __init__(self, full_scaled):
        self.full_scaled = full_scaled
        self.knn = NearestNeighbors(n_neighbors=K + 1, metric='euclidean')
        self.knn.fit(full_scaled)

    def recommend(self, query_idx, k=K):
        query = self.full_scaled[query_idx].reshape(1, -1)
        _, indices = self.knn.kneighbors(query)
        return indices[0, 1:k+1]


def evaluate_recommender(name, recommender, oracle_recommender, full_scaled, genres,
                         sample_size=2000, seed=42):
    """Evaluate a recommender on multiple metrics."""
    rng = np.random.RandomState(seed)
    sample_indices = rng.choice(len(genres), size=min(sample_size, len(genres)), replace=False)

    oracle_overlaps = []
    intra_similarities = []
    genre_diversities = []
    genre_precisions = []
    recommended_items = set()

    for idx in sample_indices:
        recs = recommender.recommend(idx)
        oracle_recs = oracle_recommender.recommend(idx)

        if len(recs) == 0:
            continue

        # 1. Oracle overlap: fraction of recs that match oracle's recs
        overlap = len(set(recs) & set(oracle_recs)) / K
        oracle_overlaps.append(overlap)

        # 2. Intra-list similarity: avg cosine similarity between query and recs
        query_vec = full_scaled[idx].reshape(1, -1)
        rec_vecs = full_scaled[recs]
        sims = cosine_similarity(query_vec, rec_vecs)[0]
        intra_similarities.append(np.mean(sims))

        # 3. Genre diversity: number of unique genres in recommendations / K
        rec_genres = genres.values[recs]
        diversity = len(set(rec_genres)) / len(recs)
        genre_diversities.append(diversity)

        # 4. Genre precision: fraction of recs with same genre as query
        query_genre = genres.values[idx]
        precision = np.mean(rec_genres == query_genre)
        genre_precisions.append(precision)

        # 5. Coverage tracking
        recommended_items.update(recs)

    coverage = len(recommended_items) / len(genres)

    return {
        'name': name,
        'oracle_overlap': np.mean(oracle_overlaps),
        'intra_similarity': np.mean(intra_similarities),
        'genre_diversity': np.mean(genre_diversities),
        'genre_precision': np.mean(genre_precisions),
        'coverage': coverage,
    }


def print_comparison(results):
    """Print formatted comparison table."""
    print_section("Recommendation System Comparison")

    headers = ['Metric', 'Genre-Based', '3-Axis (Ours)', 'Oracle', 'Winner']
    genre = results[0]
    axis3 = results[1]
    oracle = results[2]

    rows = [
        ('Oracle Overlap',
         f"{genre['oracle_overlap']*100:.1f}%",
         f"{axis3['oracle_overlap']*100:.1f}%",
         '100.0%',
         '3-Axis' if axis3['oracle_overlap'] > genre['oracle_overlap'] else 'Genre'),

        ('Intra-list Similarity',
         f"{genre['intra_similarity']:.4f}",
         f"{axis3['intra_similarity']:.4f}",
         f"{oracle['intra_similarity']:.4f}",
         '3-Axis' if axis3['intra_similarity'] > genre['intra_similarity'] else 'Genre'),

        ('Genre Diversity',
         f"{genre['genre_diversity']*100:.1f}%",
         f"{axis3['genre_diversity']*100:.1f}%",
         f"{oracle['genre_diversity']*100:.1f}%",
         '3-Axis' if axis3['genre_diversity'] > genre['genre_diversity'] else 'Genre'),

        ('Same-Genre Precision',
         f"{genre['genre_precision']*100:.1f}%",
         f"{axis3['genre_precision']*100:.1f}%",
         f"{oracle['genre_precision']*100:.1f}%",
         'Genre' if genre['genre_precision'] > axis3['genre_precision'] else '3-Axis'),

        ('Catalog Coverage',
         f"{genre['coverage']*100:.1f}%",
         f"{axis3['coverage']*100:.1f}%",
         f"{oracle['coverage']*100:.1f}%",
         '3-Axis' if axis3['coverage'] > genre['coverage'] else 'Genre'),
    ]

    # Print table
    print(f"  {'Metric':<24} {'Genre-Based':>14} {'3-Axis (Ours)':>14} {'Oracle':>14} {'Winner':>10}")
    print(f"  {'-'*76}")
    for row in rows:
        print(f"  {row[0]:<24} {row[1]:>14} {row[2]:>14} {row[3]:>14} {row[4]:>10}")

    # Count wins
    axis3_wins = sum(1 for r in rows if r[4] == '3-Axis')
    genre_wins = sum(1 for r in rows if r[4] == 'Genre')
    print(f"\n  Score: 3-Axis {axis3_wins} — Genre {genre_wins}")


def plot_comparison(results):
    """Visualize the comparison."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics = ['oracle_overlap', 'intra_similarity', 'genre_diversity', 'coverage']
    labels = ['Oracle\nOverlap', 'Intra-list\nSimilarity', 'Genre\nDiversity', 'Catalog\nCoverage']

    genre_vals = [results[0][m] for m in metrics]
    axis3_vals = [results[1][m] for m in metrics]
    oracle_vals = [results[2][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, genre_vals, width, label='Genre-Based', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x, axis3_vals, width, label='3-Axis (Ours)', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, oracle_vals, width, label='Oracle (Full)', color='#3498db', alpha=0.8)

    ax.set_ylabel('Score')
    ax.set_title('Recommendation System Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'v5_recommendation_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # Genre precision (separate chart — genre-based artificially wins this)
    fig, ax = plt.subplots(figsize=(6, 5))
    names = ['Genre-Based', '3-Axis', 'Oracle']
    precs = [results[0]['genre_precision'], results[1]['genre_precision'],
             results[2]['genre_precision']]
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    bars = ax.bar(names, precs, color=colors, alpha=0.8)
    ax.set_ylabel('Same-Genre Precision')
    ax.set_title('Same-Genre Precision\n(Genre-based wins by design — it only picks same genre)')
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, precs):
        ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'v5_genre_precision.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def example_recommendations(recommenders, features_df, genres, pca_3d, n_examples=5):
    """Show concrete recommendation examples."""
    print_section("Example Recommendations")

    rng = np.random.RandomState(123)
    indices = rng.choice(len(genres), size=n_examples, replace=False)

    genre_rec, axis3_rec, oracle_rec = recommenders

    for idx in indices:
        genre = genres.values[idx]
        track_id = features_df.index[idx]
        print(f"Query: Track {track_id} (Genre: {genre})")

        for name, rec in [("Genre-Based", genre_rec), ("3-Axis", axis3_rec), ("Oracle", oracle_rec)]:
            recs = rec.recommend(idx, k=5)
            rec_genres = genres.values[recs]
            unique_g = len(set(rec_genres))
            rec_ids = features_df.index[recs].tolist()
            genre_list = ", ".join(f"{g}" for g in rec_genres)
            print(f"  {name:12s}: genres=[{genre_list}] ({unique_g} unique)")
        print()


def main():
    print_section("Experiment V5: 3-Axis Recommendation System")
    print("Can 3 axes match genre-based recommendation performance?")

    # Load data
    features_df, genres = load_and_prepare()

    # Prepare feature spaces
    print_section("Preparing Feature Spaces")
    full_scaled, pca_3d, group_names = prepare_data(features_df)
    print(f"Full features: {full_scaled.shape[1]}D")
    print(f"3-Axis space: {pca_3d.shape[1]}D (group-aggregated PCA)")

    # Build recommenders
    print_section("Building Recommenders")
    genre_rec = GenreRecommender(genres, full_scaled)
    axis3_rec = ThreeAxisRecommender(pca_3d)
    oracle_rec = OracleRecommender(full_scaled)
    print("Built: Genre-Based, 3-Axis, Oracle (full-feature)")

    # Evaluate all
    print_section("Evaluating (sampling 2000 queries)...")
    results = []
    for name, rec in [("Genre-Based", genre_rec), ("3-Axis", axis3_rec), ("Oracle", oracle_rec)]:
        print(f"  Evaluating {name}...")
        result = evaluate_recommender(name, rec, oracle_rec, full_scaled, genres)
        results.append(result)

    # Print comparison
    print_comparison(results)

    # Example recommendations
    example_recommendations((genre_rec, axis3_rec, oracle_rec), features_df, genres, pca_3d)

    # Visualize
    print_section("Generating Visualizations")
    plot_comparison(results)

    # Conclusion
    print_section("CONCLUSION")

    genre = results[0]
    axis3 = results[1]
    oracle = results[2]

    print("Can 3 axes match genre-based recommendation?")
    print()

    # Oracle overlap comparison
    if axis3['oracle_overlap'] > genre['oracle_overlap']:
        ratio = axis3['oracle_overlap'] / genre['oracle_overlap'] if genre['oracle_overlap'] > 0 else float('inf')
        print(f"  YES — 3-Axis recommendations are {ratio:.1f}x closer to the oracle")
        print(f"  than genre-based recommendations.")
    else:
        print(f"  NO — Genre-based is closer to the oracle.")

    print()
    print(f"  3-Axis oracle overlap:  {axis3['oracle_overlap']*100:.1f}%")
    print(f"  Genre oracle overlap:   {genre['oracle_overlap']*100:.1f}%")
    print(f"  Oracle self-overlap:    100.0%")
    print()
    print(f"  3-Axis finds songs that actually SOUND similar (cosine sim: {axis3['intra_similarity']:.3f})")
    print(f"  Genre finds songs with same label (cosine sim: {genre['intra_similarity']:.3f})")
    print()

    if axis3['genre_diversity'] > genre['genre_diversity']:
        print(f"  3-Axis also provides {axis3['genre_diversity']/genre['genre_diversity']:.1f}x more genre diversity,")
        print(f"  enabling cross-genre discovery that genre-based systems cannot offer.")


if __name__ == '__main__':
    main()
