"""Cluster Linkin Park songs in emotion, lyrics, audio, and hybrid spaces.

Input
-----
data/processed/master_dataset.parquet

Outputs
-------
data/processed/song_clusters.parquet
data/processed/song_clusters.csv

outputs/tables/clustering_model_summary.csv
outputs/tables/clustering_feature_manifest.csv
outputs/tables/cluster_profiles.csv
outputs/tables/cluster_album_composition.csv
outputs/tables/cluster_membership.csv

outputs/figures/emotion_clusters_pca.png
outputs/figures/lyrics_style_clusters_pca.png
outputs/figures/audio_clusters_pca.png
outputs/figures/hybrid_clusters_pca.png

Optional outputs when umap-learn is installed
----------------------------------------------
outputs/figures/emotion_clusters_umap.png
outputs/figures/lyrics_style_clusters_umap.png
outputs/figures/audio_clusters_umap.png
outputs/figures/hybrid_clusters_umap.png

Method
------
- Median imputation
- Standard scaling
- Removal of constant and nearly empty features
- PCA for reproducible two-dimensional projection
- KMeans with silhouette-based selection of k
- Optional UMAP visualization
- Separate feature spaces for emotion, lyrics style, audio, and hybrid

Notes
-----
- Instrumental tracks are excluded from lyrics-based spaces.
- From Zero is included in emotion and lyrics spaces.
- From Zero is excluded from audio space because AcousticBrainz coverage
  ends before the album's 2024 release.
- Hybrid clustering uses tracks with lyrics features. Missing audio values
  are median-imputed, so From Zero remains in the hybrid space.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_dataset.parquet"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

CLUSTERS_PARQUET_PATH = (
    PROCESSED_DIR
    / "song_clusters.parquet"
)

CLUSTERS_CSV_PATH = (
    PROCESSED_DIR
    / "song_clusters.csv"
)

MODEL_SUMMARY_PATH = (
    TABLES_DIR
    / "clustering_model_summary.csv"
)

FEATURE_MANIFEST_PATH = (
    TABLES_DIR
    / "clustering_feature_manifest.csv"
)

CLUSTER_PROFILES_PATH = (
    TABLES_DIR
    / "cluster_profiles.csv"
)

CLUSTER_ALBUM_COMPOSITION_PATH = (
    TABLES_DIR
    / "cluster_album_composition.csv"
)

CLUSTER_MEMBERSHIP_PATH = (
    TABLES_DIR
    / "cluster_membership.csv"
)


RANDOM_STATE = 42
MIN_CLUSTERS = 2
MAX_CLUSTERS = 8
PCA_COMPONENTS_FOR_CLUSTERING = 0.90
UMAP_N_NEIGHBORS = 12
UMAP_MIN_DIST = 0.15


IDENTITY_COLUMNS = [
    "master_track_id",
    "album_order",
    "album",
    "release_year",
    "track_position",
    "track_title",
    "canonical_title",
    "era",
    "is_instrumental",
    "has_lyrics_features",
    "has_audio_features",
]


EMOTION_FEATURES = [
    "lyrics_vader_negative",
    "lyrics_vader_neutral",
    "lyrics_vader_positive",
    "lyrics_vader_compound",
    "lyrics_nrc_anger_ratio",
    "lyrics_nrc_anticipation_ratio",
    "lyrics_nrc_disgust_ratio",
    "lyrics_nrc_fear_ratio",
    "lyrics_nrc_joy_ratio",
    "lyrics_nrc_sadness_ratio",
    "lyrics_nrc_surprise_ratio",
    "lyrics_nrc_trust_ratio",
    "lyrics_nrc_positive_ratio",
    "lyrics_nrc_negative_ratio",
    "lyrics_theme_body_ratio",
    "lyrics_theme_darkness_ratio",
    "lyrics_theme_pain_ratio",
    "lyrics_theme_hope_ratio",
    "lyrics_theme_conflict_ratio",
    "lyrics_theme_isolation_ratio",
    "lyrics_theme_time_ratio",
    "lyrics_negation_ratio",
    "lyrics_absolutist_ratio",
]


LYRICS_STYLE_FEATURES = [
    *EMOTION_FEATURES,
    "lyrics_word_count",
    "lyrics_unique_word_count",
    "lyrics_type_token_ratio",
    "lyrics_root_type_token_ratio",
    "lyrics_corrected_type_token_ratio",
    "lyrics_hapax_ratio",
    "lyrics_dis_legomena_ratio",
    "lyrics_token_repetition_ratio",
    "lyrics_avg_word_length",
    "lyrics_word_length_std",
    "lyrics_long_word_ratio",
    "lyrics_very_long_word_ratio",
    "lyrics_lexical_density_proxy",
    "lyrics_line_count",
    "lyrics_sentence_count",
    "lyrics_avg_words_per_line",
    "lyrics_line_word_count_std",
    "lyrics_avg_words_per_sentence",
    "lyrics_sentence_word_count_std",
    "lyrics_question_mark_count",
    "lyrics_exclamation_mark_count",
    "lyrics_ellipsis_count",
    "lyrics_punctuation_density",
    "lyrics_uppercase_ratio",
    "lyrics_unique_line_count",
    "lyrics_duplicate_line_occurrences",
    "lyrics_duplicate_line_types",
    "lyrics_line_repetition_ratio",
    "lyrics_most_repeated_line_count",
    "lyrics_first_person_singular_ratio",
    "lyrics_first_person_plural_ratio",
    "lyrics_second_person_ratio",
    "lyrics_third_person_ratio",
    "lyrics_top_5_word_share",
    "lyrics_readability_flesch_reading_ease",
    "lyrics_readability_flesch_kincaid_grade",
    "lyrics_readability_gunning_fog",
    "lyrics_readability_smog_index",
    "lyrics_readability_automated_readability_index",
    "lyrics_readability_coleman_liau_index",
    "lyrics_readability_dale_chall_score",
]


AUDIO_FEATURES = [
    "audio_ab_duration_seconds",
    "audio_ab_average_loudness",
    "audio_ab_dynamic_complexity",
    "audio_ab_loudness_ebu128_integrated",
    "audio_ab_loudness_ebu128_loudness_range",
    "audio_ab_spectral_centroid_mean",
    "audio_ab_spectral_centroid_stdev",
    "audio_ab_spectral_entropy_mean",
    "audio_ab_spectral_flux_mean",
    "audio_ab_spectral_rolloff_mean",
    "audio_ab_zero_crossing_rate_mean",
    "audio_ab_bpm",
    "audio_ab_bpm_histogram_first_peak_bpm",
    "audio_ab_beats_count",
    "audio_ab_danceability",
    "audio_ab_onset_rate",
    "audio_ab_key_strength_edma",
    "audio_ab_key_strength_krumhansl",
    "audio_ab_chords_changes_rate",
    "audio_ab_tuning_frequency",
    "audio_ab_highlevel_danceable_probability",
    "audio_ab_mood_acoustic_probability",
    "audio_ab_mood_aggressive_probability",
    "audio_ab_mood_electronic_probability",
    "audio_ab_mood_happy_probability",
    "audio_ab_mood_party_probability",
    "audio_ab_mood_relaxed_probability",
    "audio_ab_mood_sad_probability",
    "audio_ab_voice_probability",
    "audio_ab_instrumental_probability",
]


HYBRID_FEATURES = [
    *LYRICS_STYLE_FEATURES,
    *AUDIO_FEATURES,
]


@dataclass
class ClusteringResult:
    """Container for one feature-space clustering result."""

    space: str
    track_ids: list[str]
    feature_names: list[str]
    scaled_matrix: np.ndarray
    clustering_matrix: np.ndarray
    pca_2d: np.ndarray
    cluster_labels: np.ndarray
    selected_k: int
    silhouette: float
    pca_components_used: int
    pca_variance_explained: float
    umap_2d: np.ndarray | None


def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    """Raise a clear error when required columns are missing."""

    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(
            "Master dataset is missing required columns: "
            f"{sorted(missing)}"
        )


def load_dataset() -> pd.DataFrame:
    """Load and validate the master analytical dataset."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Master dataset not found: {INPUT_PATH}\n"
            "Run build_master_dataset.py first."
        )

    df = pd.read_parquet(INPUT_PATH)

    require_columns(
        df,
        IDENTITY_COLUMNS,
    )

    if df["master_track_id"].duplicated().any():
        raise ValueError(
            "master_dataset contains duplicate master_track_id values."
        )

    if "lyrics_word_count" in df.columns:
        no_lyric_words = (
            pd.to_numeric(
                df["lyrics_word_count"],
                errors="coerce",
            )
            .fillna(0)
            .eq(0)
        )
    else:
        no_lyric_words = pd.Series(
            False,
            index=df.index,
        )

    known_instrumental = (
        df["canonical_title"]
        .astype("string")
        .str.casefold()
        .eq("drawbar")
    )

    df["is_instrumental"] = (
        df["is_instrumental"]
        .fillna(False)
        .astype(bool)
        | no_lyric_words
        | known_instrumental
    )

    return (
        df.sort_values(
            ["album_order", "track_position"],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def select_features(
    df: pd.DataFrame,
    requested_features: list[str],
) -> list[str]:
    """Keep numeric features with sufficient data and variation."""

    selected: list[str] = []

    for column in requested_features:
        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if values.notna().sum() < 3:
            continue

        if values.nunique(dropna=True) <= 1:
            continue

        selected.append(column)

    return selected


def preprocess_matrix(
    df: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    """Median-impute and standardize feature values."""

    matrix = (
        df[feature_names]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    transformed = pipeline.fit_transform(
        matrix
    )

    return transformed


def reduce_for_clustering(
    scaled_matrix: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Reduce dimensionality while retaining target variance."""

    if scaled_matrix.shape[1] <= 2:
        return (
            scaled_matrix,
            scaled_matrix.shape[1],
            1.0,
        )

    pca = PCA(
        n_components=PCA_COMPONENTS_FOR_CLUSTERING,
        svd_solver="full",
        random_state=RANDOM_STATE,
    )

    reduced = pca.fit_transform(
        scaled_matrix
    )

    return (
        reduced,
        int(pca.n_components_),
        float(
            pca.explained_variance_ratio_.sum()
        ),
    )


def create_pca_2d(
    scaled_matrix: np.ndarray,
) -> np.ndarray:
    """Create a reproducible two-dimensional PCA projection."""

    if scaled_matrix.shape[1] == 1:
        return np.column_stack(
            [
                scaled_matrix[:, 0],
                np.zeros(
                    scaled_matrix.shape[0]
                ),
            ]
        )

    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    )

    return pca.fit_transform(
        scaled_matrix
    )


def create_umap_2d(
    scaled_matrix: np.ndarray,
) -> np.ndarray | None:
    """Create an optional UMAP projection when the package is installed."""

    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(
                UMAP_N_NEIGHBORS,
                max(
                    2,
                    scaled_matrix.shape[0] - 1,
                ),
            ),
            min_dist=UMAP_MIN_DIST,
            metric="euclidean",
            random_state=RANDOM_STATE,
        )

        return reducer.fit_transform(
            scaled_matrix
        )

    except ImportError:
        return None


def select_best_k(
    clustering_matrix: np.ndarray,
) -> tuple[int, float, np.ndarray]:
    """Choose KMeans k using the highest silhouette score."""

    max_k = min(
        MAX_CLUSTERS,
        clustering_matrix.shape[0] - 1,
    )

    if max_k < MIN_CLUSTERS:
        raise ValueError(
            "Not enough tracks for clustering."
        )

    candidates: list[
        tuple[float, int, np.ndarray]
    ] = []

    for k in range(
        MIN_CLUSTERS,
        max_k + 1,
    ):
        model = KMeans(
            n_clusters=k,
            n_init=50,
            random_state=RANDOM_STATE,
        )

        labels = model.fit_predict(
            clustering_matrix
        )

        score = float(
            silhouette_score(
                clustering_matrix,
                labels,
            )
        )

        candidates.append(
            (
                score,
                k,
                labels,
            )
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            -item[1],
        ),
        reverse=True,
    )

    best_score, best_k, best_labels = (
        candidates[0]
    )

    return (
        best_k,
        best_score,
        best_labels,
    )


def fit_clustering_space(
    df: pd.DataFrame,
    space: str,
    requested_features: list[str],
    eligibility_mask: pd.Series,
) -> ClusteringResult:
    """Fit one clustering model and projection space."""

    model_df = df.loc[
        eligibility_mask
    ].copy()

    feature_names = select_features(
        model_df,
        requested_features,
    )

    if not feature_names:
        raise ValueError(
            f"No usable features were found for clustering space: {space}"
        )

    scaled_matrix = preprocess_matrix(
        model_df,
        feature_names,
    )

    (
        clustering_matrix,
        pca_components_used,
        pca_variance_explained,
    ) = reduce_for_clustering(
        scaled_matrix
    )

    (
        selected_k,
        silhouette,
        cluster_labels,
    ) = select_best_k(
        clustering_matrix
    )

    pca_2d = create_pca_2d(
        scaled_matrix
    )

    umap_2d = create_umap_2d(
        scaled_matrix
    )

    return ClusteringResult(
        space=space,
        track_ids=(
            model_df["master_track_id"]
            .astype(str)
            .tolist()
        ),
        feature_names=feature_names,
        scaled_matrix=scaled_matrix,
        clustering_matrix=clustering_matrix,
        pca_2d=pca_2d,
        cluster_labels=cluster_labels,
        selected_k=selected_k,
        silhouette=silhouette,
        pca_components_used=pca_components_used,
        pca_variance_explained=pca_variance_explained,
        umap_2d=umap_2d,
    )


def build_membership_table(
    df: pd.DataFrame,
    results: list[ClusteringResult],
) -> pd.DataFrame:
    """Create one wide table containing all cluster memberships."""

    membership = df[
        IDENTITY_COLUMNS
    ].copy()

    for result in results:
        result_df = pd.DataFrame(
            {
                "master_track_id": result.track_ids,
                f"{result.space}_cluster": (
                    result.cluster_labels + 1
                ),
                f"{result.space}_pca_x": (
                    result.pca_2d[:, 0]
                ),
                f"{result.space}_pca_y": (
                    result.pca_2d[:, 1]
                ),
            }
        )

        if result.umap_2d is not None:
            result_df[
                f"{result.space}_umap_x"
            ] = result.umap_2d[:, 0]

            result_df[
                f"{result.space}_umap_y"
            ] = result.umap_2d[:, 1]

        membership = membership.merge(
            result_df,
            on="master_track_id",
            how="left",
            validate="one_to_one",
        )

    return membership


def build_feature_manifest(
    results: list[ClusteringResult],
) -> pd.DataFrame:
    """Create a transparent feature manifest."""

    records: list[dict[str, Any]] = []

    for result in results:
        for order, feature_name in enumerate(
            result.feature_names,
            start=1,
        ):
            records.append(
                {
                    "space": result.space,
                    "feature_order": order,
                    "feature_name": feature_name,
                    "track_coverage": len(
                        result.track_ids
                    ),
                }
            )

    return pd.DataFrame(records)


def build_model_summary(
    results: list[ClusteringResult],
) -> pd.DataFrame:
    """Summarize clustering configuration and quality."""

    records = []

    for result in results:
        records.append(
            {
                "space": result.space,
                "feature_count": len(
                    result.feature_names
                ),
                "track_coverage": len(
                    result.track_ids
                ),
                "selected_k": result.selected_k,
                "silhouette_score": round(
                    result.silhouette,
                    5,
                ),
                "pca_components_used": (
                    result.pca_components_used
                ),
                "pca_variance_explained": round(
                    result.pca_variance_explained,
                    5,
                ),
                "umap_available": (
                    result.umap_2d is not None
                ),
            }
        )

    return pd.DataFrame(records)


def build_cluster_profiles(
    df: pd.DataFrame,
    membership: pd.DataFrame,
    results: list[ClusteringResult],
) -> pd.DataFrame:
    """Build standardized mean feature profiles for every cluster."""

    profile_records: list[dict[str, Any]] = []

    for result in results:
        feature_df = pd.DataFrame(
            result.scaled_matrix,
            columns=result.feature_names,
        )

        feature_df["cluster"] = (
            result.cluster_labels + 1
        )

        cluster_means = (
            feature_df.groupby(
                "cluster"
            )
            .mean()
        )

        for cluster_id, row in cluster_means.iterrows():
            top_positive = (
                row.sort_values(
                    ascending=False
                )
                .head(8)
            )

            top_negative = (
                row.sort_values(
                    ascending=True
                )
                .head(8)
            )

            cluster_members = membership[
                membership[
                    f"{result.space}_cluster"
                ].eq(cluster_id)
            ]

            profile_records.append(
                {
                    "space": result.space,
                    "cluster": int(
                        cluster_id
                    ),
                    "track_count": len(
                        cluster_members
                    ),
                    "top_positive_features": " | ".join(
                        f"{name}:{value:.3f}"
                        for name, value
                        in top_positive.items()
                    ),
                    "top_negative_features": " | ".join(
                        f"{name}:{value:.3f}"
                        for name, value
                        in top_negative.items()
                    ),
                }
            )

    return pd.DataFrame(
        profile_records
    )


def build_album_composition(
    membership: pd.DataFrame,
    results: list[ClusteringResult],
) -> pd.DataFrame:
    """Calculate album representation inside each cluster."""

    records: list[dict[str, Any]] = []

    for result in results:
        cluster_column = (
            f"{result.space}_cluster"
        )

        available = membership[
            membership[cluster_column].notna()
        ].copy()

        grouped = (
            available.groupby(
                [
                    cluster_column,
                    "album_order",
                    "album",
                    "era",
                ],
                dropna=False,
            )
            .size()
            .reset_index(
                name="track_count"
            )
        )

        grouped["cluster_total"] = (
            grouped.groupby(
                cluster_column
            )["track_count"]
            .transform("sum")
        )

        grouped["cluster_share"] = (
            grouped["track_count"]
            / grouped["cluster_total"]
        )

        grouped.insert(
            0,
            "space",
            result.space,
        )

        grouped = grouped.rename(
            columns={
                cluster_column: "cluster",
            }
        )

        records.extend(
            grouped.to_dict(
                orient="records"
            )
        )

    return pd.DataFrame(records)


def plot_projection(
    membership: pd.DataFrame,
    result: ClusteringResult,
    method: str,
    output_path: Path,
) -> None:
    """Plot one two-dimensional clustering projection."""

    cluster_column = (
        f"{result.space}_cluster"
    )

    x_column = (
        f"{result.space}_{method}_x"
    )

    y_column = (
        f"{result.space}_{method}_y"
    )

    plot_df = membership[
        membership[cluster_column].notna()
    ].copy()

    figure, axis = plt.subplots(
        figsize=(12, 8)
    )

    scatter = axis.scatter(
        plot_df[x_column],
        plot_df[y_column],
        c=plot_df[cluster_column],
        s=70,
        alpha=0.85,
    )

    for _, row in plot_df.iterrows():
        axis.annotate(
            row["track_title"],
            (
                row[x_column],
                row[y_column],
            ),
            fontsize=7,
            alpha=0.75,
        )

    axis.set_title(
        f"Linkin Park Song Clusters: "
        f"{result.space.replace('_', ' ').title()} "
        f"({method.upper()})"
    )

    axis.set_xlabel(
        f"{method.upper()} dimension 1"
    )

    axis.set_ylabel(
        f"{method.upper()} dimension 2"
    )

    legend = axis.legend(
        *scatter.legend_elements(),
        title="Cluster",
    )

    axis.add_artist(
        legend
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_figures(
    membership: pd.DataFrame,
    results: list[ClusteringResult],
) -> None:
    """Save PCA and optional UMAP figures for each space."""

    for result in results:
        pca_path = (
            FIGURES_DIR
            / f"{result.space}_clusters_pca.png"
        )

        plot_projection(
            membership=membership,
            result=result,
            method="pca",
            output_path=pca_path,
        )

        if result.umap_2d is not None:
            umap_path = (
                FIGURES_DIR
                / f"{result.space}_clusters_umap.png"
            )

            plot_projection(
                membership=membership,
                result=result,
                method="umap",
                output_path=umap_path,
            )


def print_cluster_members(
    membership: pd.DataFrame,
    results: list[ClusteringResult],
) -> None:
    """Print cluster membership lists by feature space."""

    for result in results:
        cluster_column = (
            f"{result.space}_cluster"
        )

        print(
            f"\n{result.space.upper()} CLUSTERS"
        )

        for cluster_id in range(
            1,
            result.selected_k + 1,
        ):
            members = (
                membership[
                    membership[
                        cluster_column
                    ].eq(cluster_id)
                ]
                .sort_values(
                    [
                        "album_order",
                        "track_position",
                    ]
                )
            )

            labels = [
                (
                    f"{row['track_title']} "
                    f"— {row['album']}"
                )
                for _, row
                in members.iterrows()
            ]

            print(
                f"\nCluster {cluster_id} "
                f"({len(labels)} tracks)"
            )

            for label in labels:
                print(
                    f"- {label}"
                )


def validate_outputs(
    membership: pd.DataFrame,
    results: list[ClusteringResult],
    source_df: pd.DataFrame,
) -> None:
    """Run critical clustering output checks."""

    if len(membership) != len(source_df):
        raise ValueError(
            "Cluster membership row count does not match source dataset."
        )

    if membership["master_track_id"].duplicated().any():
        raise ValueError(
            "Duplicate master_track_id values found in cluster membership."
        )

    for result in results:
        cluster_column = (
            f"{result.space}_cluster"
        )

        actual_coverage = int(
            membership[cluster_column]
            .notna()
            .sum()
        )

        expected_coverage = len(
            result.track_ids
        )

        if actual_coverage != expected_coverage:
            raise ValueError(
                f"{result.space} cluster coverage mismatch: "
                f"expected {expected_coverage}, "
                f"found {actual_coverage}."
            )


def main() -> None:
    """Fit all clustering spaces and save outputs."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset()

    lyrics_eligible = (
        df["has_lyrics_features"]
        & ~df["is_instrumental"]
    )

    audio_eligible = (
        df["has_audio_features"]
    )

    hybrid_eligible = (
        df["has_lyrics_features"]
        & ~df["is_instrumental"]
    )

    print("\nClustering eligibility:")
    print(
        "Emotion tracks: "
        f"{int(lyrics_eligible.sum())}"
    )
    print(
        "Lyrics-style tracks: "
        f"{int(lyrics_eligible.sum())}"
    )
    print(
        "Audio tracks: "
        f"{int(audio_eligible.sum())}"
    )
    print(
        "Hybrid tracks: "
        f"{int(hybrid_eligible.sum())}"
    )

    results = [
        fit_clustering_space(
            df=df,
            space="emotion",
            requested_features=EMOTION_FEATURES,
            eligibility_mask=lyrics_eligible,
        ),
        fit_clustering_space(
            df=df,
            space="lyrics_style",
            requested_features=LYRICS_STYLE_FEATURES,
            eligibility_mask=lyrics_eligible,
        ),
        fit_clustering_space(
            df=df,
            space="audio",
            requested_features=AUDIO_FEATURES,
            eligibility_mask=audio_eligible,
        ),
        fit_clustering_space(
            df=df,
            space="hybrid",
            requested_features=HYBRID_FEATURES,
            eligibility_mask=hybrid_eligible,
        ),
    ]

    membership = build_membership_table(
        df=df,
        results=results,
    )

    validate_outputs(
        membership=membership,
        results=results,
        source_df=df,
    )

    feature_manifest = build_feature_manifest(
        results
    )

    model_summary = build_model_summary(
        results
    )

    cluster_profiles = build_cluster_profiles(
        df=df,
        membership=membership,
        results=results,
    )

    album_composition = build_album_composition(
        membership=membership,
        results=results,
    )

    membership.to_parquet(
        CLUSTERS_PARQUET_PATH,
        index=False,
    )

    membership.to_csv(
        CLUSTERS_CSV_PATH,
        index=False,
        encoding="utf-8",
    )

    membership.to_csv(
        CLUSTER_MEMBERSHIP_PATH,
        index=False,
        encoding="utf-8",
    )

    feature_manifest.to_csv(
        FEATURE_MANIFEST_PATH,
        index=False,
        encoding="utf-8",
    )

    model_summary.to_csv(
        MODEL_SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )

    cluster_profiles.to_csv(
        CLUSTER_PROFILES_PATH,
        index=False,
        encoding="utf-8",
    )

    album_composition.to_csv(
        CLUSTER_ALBUM_COMPOSITION_PATH,
        index=False,
        encoding="utf-8",
    )

    save_figures(
        membership=membership,
        results=results,
    )

    print("\nClustering model summary:")
    print(
        model_summary.to_string(
            index=False
        )
    )

    print_cluster_members(
        membership=membership,
        results=results,
    )

    print("\nSaved:")
    print(f"- {CLUSTERS_PARQUET_PATH}")
    print(f"- {CLUSTERS_CSV_PATH}")
    print(f"- {MODEL_SUMMARY_PATH}")
    print(f"- {FEATURE_MANIFEST_PATH}")
    print(f"- {CLUSTER_PROFILES_PATH}")
    print(f"- {CLUSTER_ALBUM_COMPOSITION_PATH}")
    print(f"- {CLUSTER_MEMBERSHIP_PATH}")

    print("\nClustering analysis completed.")


if __name__ == "__main__":
    main()
