"""Main pipeline: FMA audio feature PCA analysis for Three Elements Music Recommendation."""

import sys
import os

# Allow running from project root: python src/main.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_loader import load_and_prepare
from pca_analysis import run_pca, get_variance_analysis, get_loadings_analysis, transform_to_n_components
from visualization import plot_scree, plot_3d_scatter, plot_loadings_heatmap
from evaluation import compute_knn_overlap


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_variance_results(variance):
    print_section("PCA Variance Analysis")
    print(f"3-component cumulative explained variance: {variance['three_pc_cumulative']*100:.2f}%")
    print(f"Threshold: {config.VARIANCE_THRESHOLD*100:.0f}%")
    print(f"Verdict: {'PASS' if variance['is_sufficient'] else 'FAIL'}")
    print(f"\nMinimum components for {config.VARIANCE_THRESHOLD*100:.0f}% variance: "
          f"{variance['n_components_for_threshold']}")

    print("\nTop 10 components:")
    for i in range(min(10, len(variance['individual']))):
        print(f"  PC{i+1}: {variance['individual'][i]*100:6.2f}%  "
              f"(cumulative: {variance['cumulative'][i]*100:6.2f}%)")


def print_loadings_results(loadings):
    print_section("Principal Component Interpretation")
    for comp in loadings:
        print(f"PC{comp['component']} (dominant type: {comp['dominant_type']}):")
        for name, value in comp['top_features']:
            print(f"  {name:30s}  {value:+.4f}")
        print()


def print_knn_results(knn):
    print_section("k-NN Overlap Analysis (Information Loss)")
    print(f"Overall overlap (k={config.K_NEIGHBORS}): {knn['overall_overlap']*100:.2f}%")
    print(f"Threshold: {config.KNN_OVERLAP_THRESHOLD*100:.0f}%")
    print(f"Verdict: {'PASS' if knn['is_sufficient'] else 'FAIL'}")
    print("\nPer-genre overlap:")
    for genre, overlap in sorted(knn['per_genre_overlap'].items()):
        print(f"  {genre:20s}  {overlap*100:.2f}%")


def print_conclusion(variance, knn):
    print_section("CONCLUSION")
    var_pass = variance['is_sufficient']
    knn_pass = knn['is_sufficient']

    print(f"Research question: Can 3 dimensions sufficiently represent music similarity?")
    print(f"")
    print(f"  Variance explained:  {variance['three_pc_cumulative']*100:.2f}%  "
          f"({'PASS' if var_pass else 'FAIL'} >= {config.VARIANCE_THRESHOLD*100:.0f}%)")
    print(f"  k-NN overlap:        {knn['overall_overlap']*100:.2f}%  "
          f"({'PASS' if knn_pass else 'FAIL'} >= {config.KNN_OVERLAP_THRESHOLD*100:.0f}%)")
    print()

    if var_pass and knn_pass:
        print("  ANSWER: YES - 3 dimensions are sufficient.")
        print("  Both variance explanation and similarity preservation meet thresholds.")
    elif var_pass:
        print("  ANSWER: PARTIALLY - 3 dimensions capture enough variance but lose ")
        print("  significant neighbor relationships.")
    elif knn_pass:
        print("  ANSWER: PARTIALLY - 3 dimensions preserve neighbors but don't capture")
        print("  enough total variance.")
    else:
        n = variance['n_components_for_threshold']
        print(f"  ANSWER: NO - 3 dimensions are NOT sufficient.")
        print(f"  Minimum {n} components needed for {config.VARIANCE_THRESHOLD*100:.0f}% variance.")


def main():
    # Step 1: Load and prepare data
    print_section("Data Loading")
    features_df, genres = load_and_prepare()

    # Step 2: Run PCA
    print_section("Running PCA")
    pca, scaler, scaled_data, feature_names = run_pca(features_df)
    variance = get_variance_analysis(pca)
    print_variance_results(variance)

    # Step 3: Interpret components
    loadings = get_loadings_analysis(pca, feature_names)
    print_loadings_results(loadings)

    # Step 4: Visualizations
    print_section("Generating Visualizations")
    projected = transform_to_n_components(pca, scaled_data)
    plot_scree(variance)
    plot_3d_scatter(projected, genres)
    plot_loadings_heatmap(loadings, pca, feature_names)

    # Step 5: k-NN overlap evaluation
    print_section("Running k-NN Overlap Evaluation")
    knn = compute_knn_overlap(scaled_data, projected, genres)
    print_knn_results(knn)

    # Step 6: Conclusion
    print_conclusion(variance, knn)


if __name__ == '__main__':
    main()
