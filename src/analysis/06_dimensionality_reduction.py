"""Create PCA, UMAP, and t-SNE maps of the Linkin Park catalogue.

Input
-----
data/processed/master_dataset.parquet

Outputs
-------
data/processed/song_embeddings.parquet
data/processed/song_embeddings.csv

outputs/tables/dimensionality_reduction_summary.csv
outputs/tables/dimensionality_reduction_feature_manifest.csv
outputs/tables/album_centroids.csv
outputs/tables/era_centroids.csv
outputs/tables/pca_loadings.csv

outputs/figures/pca_2d_albums.png
outputs/figures/pca_2d_eras.png
outputs/figures/pca_3d_albums.png
outputs/figures/umap_2d_albums.png
outputs/figures/umap_2d_eras.png
outputs/figures/tsne_2d_albums.png
outputs/figures/tsne_2d_eras.png
outputs/figures/album_centroids_pca.png
outputs/figures/album_trajectories_pca.png

Purpose
-------
This script projects the high-dimensional Linkin Park feature space into
two and three dimensions. It provides complementary global and local views:

- PCA shows the dominant linear directions of variation.
- UMAP preserves local neighbourhood structure and broad manifold shape.
- t-SNE emphasizes local clusters and nearest-neighbour relationships.

Notes
-----
- Instrumental tracks are excluded from lyrics-based embeddings.
- The hybrid feature space includes From Zero. Missing AcousticBrainz
  features are median-imputed.
- PCA is the most interpretable projection.
- UMAP and t-SNE are exploratory visualizations and should not be treated as
  definitive evidence of discrete natural clusters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
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

EMBEDDINGS_PARQUET_PATH = (
    PROCESSED_DIR
    / "song_embeddings.parquet"
)

EMBEDDINGS_CSV_PATH = (
    PROCESSED_DIR
    / "song_embeddings.csv"
)

SUMMARY_PATH = (
    TABLES_DIR
    / "dimensionality_reduction_summary.csv"
)

FEATURE_MANIFEST_PATH = (
    TABLES_DIR
    / "dimensionality_reduction_feature_manifest.csv"
)

ALBUM_CENTROIDS_PATH = (
    TABLES_DIR
    / "album_centroids.csv"
)

ERA_CENTROIDS_PATH = (
    TABLES_DIR
    / "era_centroids.csv"
)

PCA_LOADINGS_PATH = (
    TABLES_DIR
    / "pca_loadings.csv"
)


RANDOM_STATE = 42
TSNE_PERPLEXITY = 20
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
class EmbeddingResult:
    """Container for one feature-space embedding."""

    space: str
    track_ids: list[str]
    feature_names: list[str]
    scaled_matrix: np.ndarray
    pca_2d: np.ndarray
    pca_3d: np.ndarray
    pca_variance_2d: float
    pca_variance_3d: float
    pca_loadings: np.ndarray
    umap_2d: np.ndarray | None
    tsne_2d: np.ndarray


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
    """Keep numeric features with enough data and useful variation."""

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
    """Median-impute and standardize a feature matrix."""

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

    return pipeline.fit_transform(
        matrix
    )


def create_umap(
    scaled_matrix: np.ndarray,
) -> np.ndarray | None:
    """Create an optional UMAP embedding."""

    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(
                UMAP_N_NEIGHBORS,
                max(
                    2,
                    len(scaled_matrix) - 1,
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


def create_tsne(
    scaled_matrix: np.ndarray,
) -> np.ndarray:
    """Create a deterministic two-dimensional t-SNE embedding."""

    perplexity = min(
        TSNE_PERPLEXITY,
        max(
            5,
            (len(scaled_matrix) - 1) // 3,
        ),
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=2000,
        random_state=RANDOM_STATE,
        metric="euclidean",
    )

    return tsne.fit_transform(
        scaled_matrix
    )


def fit_embedding_space(
    df: pd.DataFrame,
    space: str,
    requested_features: list[str],
    eligibility_mask: pd.Series,
) -> EmbeddingResult:
    """Fit PCA, optional UMAP, and t-SNE for one feature space."""

    model_df = df.loc[
        eligibility_mask
    ].copy()

    feature_names = select_features(
        model_df,
        requested_features,
    )

    if not feature_names:
        raise ValueError(
            f"No usable features found for embedding space: {space}"
        )

    scaled_matrix = preprocess_matrix(
        model_df,
        feature_names,
    )

    pca_2_model = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    )

    pca_2d = pca_2_model.fit_transform(
        scaled_matrix
    )

    pca_3_model = PCA(
        n_components=3,
        random_state=RANDOM_STATE,
    )

    pca_3d = pca_3_model.fit_transform(
        scaled_matrix
    )

    umap_2d = create_umap(
        scaled_matrix
    )

    tsne_2d = create_tsne(
        scaled_matrix
    )

    return EmbeddingResult(
        space=space,
        track_ids=(
            model_df["master_track_id"]
            .astype(str)
            .tolist()
        ),
        feature_names=feature_names,
        scaled_matrix=scaled_matrix,
        pca_2d=pca_2d,
        pca_3d=pca_3d,
        pca_variance_2d=float(
            pca_2_model.explained_variance_ratio_.sum()
        ),
        pca_variance_3d=float(
            pca_3_model.explained_variance_ratio_.sum()
        ),
        pca_loadings=pca_3_model.components_,
        umap_2d=umap_2d,
        tsne_2d=tsne_2d,
    )


def build_embeddings_table(
    df: pd.DataFrame,
    results: list[EmbeddingResult],
) -> pd.DataFrame:
    """Build one wide track-level embedding table."""

    embeddings = df[
        IDENTITY_COLUMNS
    ].copy()

    for result in results:
        result_df = pd.DataFrame(
            {
                "master_track_id": result.track_ids,
                f"{result.space}_pca_x": result.pca_2d[:, 0],
                f"{result.space}_pca_y": result.pca_2d[:, 1],
                f"{result.space}_pca_3d_x": result.pca_3d[:, 0],
                f"{result.space}_pca_3d_y": result.pca_3d[:, 1],
                f"{result.space}_pca_3d_z": result.pca_3d[:, 2],
                f"{result.space}_tsne_x": result.tsne_2d[:, 0],
                f"{result.space}_tsne_y": result.tsne_2d[:, 1],
            }
        )

        if result.umap_2d is not None:
            result_df[
                f"{result.space}_umap_x"
            ] = result.umap_2d[:, 0]

            result_df[
                f"{result.space}_umap_y"
            ] = result.umap_2d[:, 1]

        embeddings = embeddings.merge(
            result_df,
            on="master_track_id",
            how="left",
            validate="one_to_one",
        )

    return embeddings


def build_summary(
    results: list[EmbeddingResult],
) -> pd.DataFrame:
    """Summarize feature counts and projection coverage."""

    records: list[dict[str, Any]] = []

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
                "pca_2d_variance_explained": round(
                    result.pca_variance_2d,
                    5,
                ),
                "pca_3d_variance_explained": round(
                    result.pca_variance_3d,
                    5,
                ),
                "umap_available": (
                    result.umap_2d is not None
                ),
                "tsne_perplexity": min(
                    TSNE_PERPLEXITY,
                    max(
                        5,
                        (len(result.track_ids) - 1) // 3,
                    ),
                ),
            }
        )

    return pd.DataFrame(records)


def build_feature_manifest(
    results: list[EmbeddingResult],
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


def build_pca_loadings(
    results: list[EmbeddingResult],
) -> pd.DataFrame:
    """Store PCA loadings for interpretability."""

    records: list[dict[str, Any]] = []

    for result in results:
        for component_index in range(
            min(
                3,
                result.pca_loadings.shape[0],
            )
        ):
            component_name = (
                f"PC{component_index + 1}"
            )

            for feature_name, loading in zip(
                result.feature_names,
                result.pca_loadings[
                    component_index
                ],
            ):
                records.append(
                    {
                        "space": result.space,
                        "component": component_name,
                        "feature_name": feature_name,
                        "loading": float(loading),
                        "absolute_loading": abs(
                            float(loading)
                        ),
                    }
                )

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "space",
                "component",
                "absolute_loading",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )


def build_centroids(
    embeddings: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Build PCA centroids for each album or era."""

    coordinate_columns = [
        column
        for column in embeddings.columns
        if column.endswith(
            (
                "_pca_x",
                "_pca_y",
            )
        )
        and "_3d_" not in column
    ]

    grouped = (
        embeddings.groupby(
            group_column,
            dropna=False,
        )[coordinate_columns]
        .mean()
        .reset_index()
    )

    return grouped


def plot_2d(
    embeddings: pd.DataFrame,
    space: str,
    method: str,
    color_column: str,
    output_path: Path,
    annotate: bool = True,
) -> None:
    """Plot one two-dimensional embedding."""

    x_column = f"{space}_{method}_x"
    y_column = f"{space}_{method}_y"

    plot_df = embeddings[
        embeddings[x_column].notna()
        & embeddings[y_column].notna()
    ].copy()

    categories = (
        plot_df[color_column]
        .astype("category")
    )

    codes = categories.cat.codes

    figure, axis = plt.subplots(
        figsize=(13, 9)
    )

    scatter = axis.scatter(
        plot_df[x_column],
        plot_df[y_column],
        c=codes,
        s=75,
        alpha=0.82,
    )

    if annotate:
        for _, row in plot_df.iterrows():
            axis.annotate(
                row["track_title"],
                (
                    row[x_column],
                    row[y_column],
                ),
                fontsize=7,
                alpha=0.72,
            )

    axis.set_title(
        f"Linkin Park {space.replace('_', ' ').title()} Space "
        f"— {method.upper()} by {color_column.replace('_', ' ').title()}"
    )

    axis.set_xlabel(
        f"{method.upper()} dimension 1"
    )

    axis.set_ylabel(
        f"{method.upper()} dimension 2"
    )

    handles, _ = scatter.legend_elements(
        num=len(
            categories.cat.categories
        )
    )

    axis.legend(
        handles,
        categories.cat.categories,
        title=color_column.replace(
            "_",
            " ",
        ).title(),
        loc="best",
        fontsize=8,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_pca_3d(
    embeddings: pd.DataFrame,
    space: str,
    output_path: Path,
) -> None:
    """Plot a three-dimensional PCA embedding by album."""

    x_column = f"{space}_pca_3d_x"
    y_column = f"{space}_pca_3d_y"
    z_column = f"{space}_pca_3d_z"

    plot_df = embeddings[
        embeddings[x_column].notna()
        & embeddings[y_column].notna()
        & embeddings[z_column].notna()
    ].copy()

    album_codes = (
        plot_df["album"]
        .astype("category")
    )

    codes = album_codes.cat.codes

    figure = plt.figure(
        figsize=(13, 10)
    )

    axis = figure.add_subplot(
        111,
        projection="3d",
    )

    scatter = axis.scatter(
        plot_df[x_column],
        plot_df[y_column],
        plot_df[z_column],
        c=codes,
        s=65,
        alpha=0.82,
    )

    axis.set_title(
        f"Linkin Park {space.replace('_', ' ').title()} Space — PCA 3D"
    )

    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_zlabel("PC3")

    handles, _ = scatter.legend_elements(
        num=len(
            album_codes.cat.categories
        )
    )

    axis.legend(
        handles,
        album_codes.cat.categories,
        title="Album",
        fontsize=8,
        loc="best",
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_album_centroids(
    album_centroids: pd.DataFrame,
    space: str,
    output_path: Path,
) -> None:
    """Plot album centroids in PCA space."""

    x_column = f"{space}_pca_x"
    y_column = f"{space}_pca_y"

    plot_df = album_centroids[
        album_centroids[x_column].notna()
        & album_centroids[y_column].notna()
    ].copy()

    figure, axis = plt.subplots(
        figsize=(11, 8)
    )

    axis.scatter(
        plot_df[x_column],
        plot_df[y_column],
        s=130,
    )

    for _, row in plot_df.iterrows():
        axis.annotate(
            row["album"],
            (
                row[x_column],
                row[y_column],
            ),
            fontsize=10,
        )

    axis.set_title(
        f"Album Centroids in {space.replace('_', ' ').title()} PCA Space"
    )

    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_album_trajectory(
    album_centroids: pd.DataFrame,
    space: str,
    output_path: Path,
) -> None:
    """Plot the chronological movement of album centroids."""

    x_column = f"{space}_pca_x"
    y_column = f"{space}_pca_y"

    plot_df = (
        album_centroids[
            album_centroids[x_column].notna()
            & album_centroids[y_column].notna()
        ]
        .sort_values(
            "album_order"
        )
        .copy()
    )

    figure, axis = plt.subplots(
        figsize=(11, 8)
    )

    axis.plot(
        plot_df[x_column],
        plot_df[y_column],
        marker="o",
    )

    for _, row in plot_df.iterrows():
        axis.annotate(
            f"{int(row['album_order'])}. {row['album']}",
            (
                row[x_column],
                row[y_column],
            ),
            fontsize=9,
        )

    axis.set_title(
        f"Chronological Album Trajectory in "
        f"{space.replace('_', ' ').title()} PCA Space"
    )

    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def validate_embeddings(
    embeddings: pd.DataFrame,
    source_df: pd.DataFrame,
) -> None:
    """Run critical structural checks."""

    if len(embeddings) != len(source_df):
        raise ValueError(
            "Embedding output row count does not match source dataset."
        )

    if embeddings["master_track_id"].duplicated().any():
        raise ValueError(
            "Duplicate master_track_id values found in embeddings."
        )


def print_summary(
    summary: pd.DataFrame,
    pca_loadings: pd.DataFrame,
) -> None:
    """Print embedding quality and dominant PCA drivers."""

    print("\nDimensionality reduction summary:")
    print(
        summary.to_string(
            index=False
        )
    )

    print("\nTop PCA drivers:")

    for space in summary["space"]:
        print(
            f"\n{space.upper()}"
        )

        for component in [
            "PC1",
            "PC2",
            "PC3",
        ]:
            top = (
                pca_loadings[
                    pca_loadings["space"].eq(
                        space
                    )
                    & pca_loadings[
                        "component"
                    ].eq(
                        component
                    )
                ]
                .head(6)
            )

            if top.empty:
                continue

            labels = " | ".join(
                (
                    f"{row['feature_name']}:"
                    f"{row['loading']:+.3f}"
                )
                for _, row
                in top.iterrows()
            )

            print(
                f"{component}: {labels}"
            )


def main() -> None:
    """Run PCA, UMAP, and t-SNE across all feature spaces."""

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

    print("\nEmbedding eligibility:")
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
        fit_embedding_space(
            df=df,
            space="emotion",
            requested_features=EMOTION_FEATURES,
            eligibility_mask=lyrics_eligible,
        ),
        fit_embedding_space(
            df=df,
            space="lyrics_style",
            requested_features=LYRICS_STYLE_FEATURES,
            eligibility_mask=lyrics_eligible,
        ),
        fit_embedding_space(
            df=df,
            space="audio",
            requested_features=AUDIO_FEATURES,
            eligibility_mask=audio_eligible,
        ),
        fit_embedding_space(
            df=df,
            space="hybrid",
            requested_features=HYBRID_FEATURES,
            eligibility_mask=hybrid_eligible,
        ),
    ]

    embeddings = build_embeddings_table(
        df=df,
        results=results,
    )

    validate_embeddings(
        embeddings=embeddings,
        source_df=df,
    )

    summary = build_summary(
        results
    )

    feature_manifest = build_feature_manifest(
        results
    )

    pca_loadings = build_pca_loadings(
        results
    )

    album_centroids = build_centroids(
        embeddings,
        "album",
    )

    album_metadata = (
        df[
            [
                "album",
                "album_order",
                "release_year",
                "era",
            ]
        ]
        .drop_duplicates(
            subset=["album"]
        )
    )

    album_centroids = album_metadata.merge(
        album_centroids,
        on="album",
        how="left",
        validate="one_to_one",
    )

    album_centroids = (
        album_centroids
        .sort_values(
            "album_order"
        )
        .reset_index(drop=True)
    )

    era_centroids = build_centroids(
        embeddings,
        "era",
    )

    embeddings.to_parquet(
        EMBEDDINGS_PARQUET_PATH,
        index=False,
    )

    embeddings.to_csv(
        EMBEDDINGS_CSV_PATH,
        index=False,
        encoding="utf-8",
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )

    feature_manifest.to_csv(
        FEATURE_MANIFEST_PATH,
        index=False,
        encoding="utf-8",
    )

    album_centroids.to_csv(
        ALBUM_CENTROIDS_PATH,
        index=False,
        encoding="utf-8",
    )

    era_centroids.to_csv(
        ERA_CENTROIDS_PATH,
        index=False,
        encoding="utf-8",
    )

    pca_loadings.to_csv(
        PCA_LOADINGS_PATH,
        index=False,
        encoding="utf-8",
    )

    plot_2d(
        embeddings=embeddings,
        space="hybrid",
        method="pca",
        color_column="album",
        output_path=(
            FIGURES_DIR
            / "pca_2d_albums.png"
        ),
    )

    plot_2d(
        embeddings=embeddings,
        space="hybrid",
        method="pca",
        color_column="era",
        output_path=(
            FIGURES_DIR
            / "pca_2d_eras.png"
        ),
    )

    plot_pca_3d(
        embeddings=embeddings,
        space="hybrid",
        output_path=(
            FIGURES_DIR
            / "pca_3d_albums.png"
        ),
    )

    if any(
        result.space == "hybrid"
        and result.umap_2d is not None
        for result in results
    ):
        plot_2d(
            embeddings=embeddings,
            space="hybrid",
            method="umap",
            color_column="album",
            output_path=(
                FIGURES_DIR
                / "umap_2d_albums.png"
            ),
        )

        plot_2d(
            embeddings=embeddings,
            space="hybrid",
            method="umap",
            color_column="era",
            output_path=(
                FIGURES_DIR
                / "umap_2d_eras.png"
            ),
        )

    plot_2d(
        embeddings=embeddings,
        space="hybrid",
        method="tsne",
        color_column="album",
        output_path=(
            FIGURES_DIR
            / "tsne_2d_albums.png"
        ),
    )

    plot_2d(
        embeddings=embeddings,
        space="hybrid",
        method="tsne",
        color_column="era",
        output_path=(
            FIGURES_DIR
            / "tsne_2d_eras.png"
        ),
    )

    plot_album_centroids(
        album_centroids=album_centroids,
        space="hybrid",
        output_path=(
            FIGURES_DIR
            / "album_centroids_pca.png"
        ),
    )

    plot_album_trajectory(
        album_centroids=album_centroids,
        space="hybrid",
        output_path=(
            FIGURES_DIR
            / "album_trajectories_pca.png"
        ),
    )

    print_summary(
        summary=summary,
        pca_loadings=pca_loadings,
    )

    print("\nSaved:")
    print(f"- {EMBEDDINGS_PARQUET_PATH}")
    print(f"- {EMBEDDINGS_CSV_PATH}")
    print(f"- {SUMMARY_PATH}")
    print(f"- {FEATURE_MANIFEST_PATH}")
    print(f"- {ALBUM_CENTROIDS_PATH}")
    print(f"- {ERA_CENTROIDS_PATH}")
    print(f"- {PCA_LOADINGS_PATH}")

    print("\nDimensionality reduction analysis completed.")


if __name__ == "__main__":
    main()
