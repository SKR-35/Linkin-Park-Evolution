"""Analyze temporal dynamics across Linkin Park's studio-album history.

Inputs
------
data/processed/master_dataset.parquet
data/processed/song_embeddings.parquet

Outputs
-------
outputs/tables/temporal_album_profiles.csv
outputs/tables/temporal_album_transitions.csv
outputs/tables/temporal_feature_drifts.csv
outputs/tables/temporal_era_transition.csv
outputs/tables/temporal_song_distances.csv

outputs/figures/temporal_evolution_velocity.png
outputs/figures/temporal_album_trajectory.png
outputs/figures/temporal_feature_drift_heatmap.png
outputs/figures/temporal_era_transition.png
outputs/figures/temporal_song_distance_distribution.png

Purpose
-------
This script treats Linkin Park's catalogue as a time-ordered sequence and
measures how quickly the band's lyrical, emotional, and musical profile
changes from one studio album to the next.

Core ideas
----------
- Album profile:
  the standardized mean feature vector for one studio album.

- Evolution distance:
  Euclidean distance between consecutive album profiles.

- Evolution velocity:
  profile distance divided by elapsed years between releases.

- Feature drift:
  signed standardized change in each feature between consecutive albums.

- Era transition:
  the aggregate profile change from Chester Era to Emily Era.

Notes
-----
- Instrumental tracks are excluded from lyrics-derived album profiles.
- From Zero remains fully included in lyrics and emotion features.
- Missing From Zero audio features are median-imputed only for the hybrid
  temporal profile. Audio-only conclusions should still exclude From Zero.
- PCA embedding coordinates are used only for trajectory visualization,
  not for the primary drift calculations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MASTER_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_dataset.parquet"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "song_embeddings.parquet"
)

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

ALBUM_PROFILES_PATH = (
    TABLES_DIR
    / "temporal_album_profiles.csv"
)

TRANSITIONS_PATH = (
    TABLES_DIR
    / "temporal_album_transitions.csv"
)

FEATURE_DRIFTS_PATH = (
    TABLES_DIR
    / "temporal_feature_drifts.csv"
)

ERA_TRANSITION_PATH = (
    TABLES_DIR
    / "temporal_era_transition.csv"
)

SONG_DISTANCES_PATH = (
    TABLES_DIR
    / "temporal_song_distances.csv"
)

VELOCITY_FIGURE_PATH = (
    FIGURES_DIR
    / "temporal_evolution_velocity.png"
)

TRAJECTORY_FIGURE_PATH = (
    FIGURES_DIR
    / "temporal_album_trajectory.png"
)

DRIFT_HEATMAP_PATH = (
    FIGURES_DIR
    / "temporal_feature_drift_heatmap.png"
)

ERA_TRANSITION_FIGURE_PATH = (
    FIGURES_DIR
    / "temporal_era_transition.png"
)

SONG_DISTANCE_FIGURE_PATH = (
    FIGURES_DIR
    / "temporal_song_distance_distribution.png"
)


IDENTITY_COLUMNS = [
    "master_track_id",
    "album_order",
    "album",
    "release_year",
    "track_title",
    "canonical_title",
    "era",
    "is_instrumental",
    "has_lyrics_features",
    "has_audio_features",
]


TEMPORAL_FEATURES = [
    # Sentiment and emotion
    "lyrics_vader_compound",
    "lyrics_vader_negative",
    "lyrics_vader_positive",
    "lyrics_nrc_anger_ratio",
    "lyrics_nrc_anticipation_ratio",
    "lyrics_nrc_disgust_ratio",
    "lyrics_nrc_fear_ratio",
    "lyrics_nrc_joy_ratio",
    "lyrics_nrc_sadness_ratio",
    "lyrics_nrc_surprise_ratio",
    "lyrics_nrc_trust_ratio",
    # Themes
    "lyrics_theme_darkness_ratio",
    "lyrics_theme_pain_ratio",
    "lyrics_theme_hope_ratio",
    "lyrics_theme_conflict_ratio",
    "lyrics_theme_isolation_ratio",
    "lyrics_theme_time_ratio",
    # Style
    "lyrics_word_count",
    "lyrics_type_token_ratio",
    "lyrics_line_repetition_ratio",
    "lyrics_first_person_singular_ratio",
    "lyrics_second_person_ratio",
    "lyrics_readability_flesch_reading_ease",
    # Audio
    "audio_ab_bpm",
    "audio_ab_average_loudness",
    "audio_ab_dynamic_complexity",
    "audio_ab_danceability",
    "audio_ab_mood_aggressive_probability",
    "audio_ab_mood_happy_probability",
    "audio_ab_mood_relaxed_probability",
    "audio_ab_mood_sad_probability",
    "audio_ab_mood_electronic_probability",
    "audio_ab_spectral_centroid_mean",
    "audio_ab_onset_rate",
]


def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:
    """Raise a clear error for missing required columns."""

    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{label} is missing required columns: "
            f"{sorted(missing)}"
        )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load master data and embedding coordinates."""

    if not MASTER_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Master dataset not found: {MASTER_DATASET_PATH}"
        )

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Song embeddings not found: {EMBEDDINGS_PATH}\n"
            "Run 06_dimensionality_reduction.py first."
        )

    master = pd.read_parquet(
        MASTER_DATASET_PATH
    )

    embeddings = pd.read_parquet(
        EMBEDDINGS_PATH
    )

    require_columns(
        master,
        IDENTITY_COLUMNS,
        "Master dataset",
    )

    require_columns(
        embeddings,
        [
            "master_track_id",
            "hybrid_pca_x",
            "hybrid_pca_y",
        ],
        "Song embeddings",
    )

    if master["master_track_id"].duplicated().any():
        raise ValueError(
            "Master dataset contains duplicate track IDs."
        )

    if embeddings["master_track_id"].duplicated().any():
        raise ValueError(
            "Embedding dataset contains duplicate track IDs."
        )

    df = master.merge(
        embeddings[
            [
                "master_track_id",
                "hybrid_pca_x",
                "hybrid_pca_y",
                "hybrid_tsne_x",
                "hybrid_tsne_y",
            ]
        ],
        on="master_track_id",
        how="left",
        validate="one_to_one",
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
            [
                "album_order",
                "track_position",
            ],
            na_position="last",
        )
        .reset_index(drop=True),
        embeddings,
    )


def select_temporal_features(
    df: pd.DataFrame,
) -> list[str]:
    """Keep usable numeric temporal features."""

    selected: list[str] = []

    for column in TEMPORAL_FEATURES:
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

    if not selected:
        raise ValueError(
            "No usable temporal features were found."
        )

    return selected


def build_album_profiles(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build standardized album-level hybrid profiles."""

    analysis_df = df[
        ~df["is_instrumental"]
    ].copy()

    grouped = (
        analysis_df.groupby(
            [
                "album_order",
                "album",
                "release_year",
                "era",
            ],
            dropna=False,
        )[features]
        .mean()
        .reset_index()
        .sort_values(
            "album_order"
        )
        .reset_index(drop=True)
    )

    matrix = grouped[
        features
    ].apply(
        pd.to_numeric,
        errors="coerce",
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

    standardized = pipeline.fit_transform(
        matrix
    )

    profile_df = grouped[
        [
            "album_order",
            "album",
            "release_year",
            "era",
        ]
    ].copy()

    for index, feature in enumerate(
        features
    ):
        profile_df[
            f"z_{feature}"
        ] = standardized[:, index]

    return profile_df, standardized


def build_transitions(
    profiles: pd.DataFrame,
    standardized: np.ndarray,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build consecutive album distances and feature drift details."""

    transition_records: list[
        dict[str, object]
    ] = []

    drift_records: list[
        dict[str, object]
    ] = []

    for index in range(
        1,
        len(profiles),
    ):
        previous = standardized[
            index - 1
        ]

        current = standardized[
            index
        ]

        difference = current - previous

        distance = float(
            np.linalg.norm(
                difference
            )
        )

        years_elapsed = int(
            profiles.iloc[index][
                "release_year"
            ]
            - profiles.iloc[index - 1][
                "release_year"
            ]
        )

        velocity = (
            distance / years_elapsed
            if years_elapsed > 0
            else np.nan
        )

        top_indices = (
            np.argsort(
                np.abs(
                    difference
                )
            )[::-1][:8]
        )

        transition_records.append(
            {
                "from_album": profiles.iloc[
                    index - 1
                ]["album"],
                "to_album": profiles.iloc[
                    index
                ]["album"],
                "from_year": int(
                    profiles.iloc[
                        index - 1
                    ]["release_year"]
                ),
                "to_year": int(
                    profiles.iloc[
                        index
                    ]["release_year"]
                ),
                "years_elapsed": years_elapsed,
                "profile_distance": round(
                    distance,
                    6,
                ),
                "evolution_velocity": round(
                    velocity,
                    6,
                ),
                "largest_feature_shifts": " | ".join(
                    (
                        f"{features[feature_index]}:"
                        f"{difference[feature_index]:+.3f}"
                    )
                    for feature_index
                    in top_indices
                ),
            }
        )

        for feature_index, feature in enumerate(
            features
        ):
            drift_records.append(
                {
                    "from_album": profiles.iloc[
                        index - 1
                    ]["album"],
                    "to_album": profiles.iloc[
                        index
                    ]["album"],
                    "transition_order": index,
                    "feature": feature,
                    "standardized_change": round(
                        float(
                            difference[
                                feature_index
                            ]
                        ),
                        6,
                    ),
                    "absolute_change": round(
                        abs(
                            float(
                                difference[
                                    feature_index
                                ]
                            )
                        ),
                        6,
                    ),
                }
            )

    transitions = pd.DataFrame(
        transition_records
    )

    drifts = pd.DataFrame(
        drift_records
    )

    return transitions, drifts


def build_era_transition(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Compare standardized Chester and Emily era profiles."""

    analysis_df = df[
        ~df["is_instrumental"]
    ].copy()

    era_means = (
        analysis_df.groupby(
            "era",
            dropna=False,
        )[features]
        .mean()
    )

    if not {
        "Chester Era",
        "Emily Era",
    }.issubset(
        era_means.index
    ):
        raise ValueError(
            "Both Chester Era and Emily Era are required."
        )

    matrix = era_means.loc[
        [
            "Chester Era",
            "Emily Era",
        ],
        features,
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    matrix = (
        matrix.T
        .fillna(
            matrix.median(
                axis=0
            )
        )
        .T
    )

    changes = (
        matrix.loc["Emily Era"]
        - matrix.loc["Chester Era"]
    )

    result = pd.DataFrame(
        {
            "feature": features,
            "chester_mean": (
                matrix.loc[
                    "Chester Era"
                ].values
            ),
            "emily_mean": (
                matrix.loc[
                    "Emily Era"
                ].values
            ),
            "raw_change": changes.values,
            "absolute_change": (
                changes.abs().values
            ),
        }
    )

    return (
        result.sort_values(
            "absolute_change",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_song_temporal_distances(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Measure each song's distance from its album and prior-album centroids."""

    analysis_df = df[
        ~df["is_instrumental"]
    ].copy()

    matrix = analysis_df[
        features
    ].apply(
        pd.to_numeric,
        errors="coerce",
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

    standardized = pipeline.fit_transform(
        matrix
    )

    standardized_df = pd.DataFrame(
        standardized,
        columns=features,
        index=analysis_df.index,
    )

    album_centroids = (
        standardized_df.assign(
            album=analysis_df["album"].values
        )
        .groupby("album")
        .mean()
    )

    previous_album = (
        analysis_df[
            [
                "album_order",
                "album",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "album_order"
        )
    )

    previous_map = {
        current_album: previous_album_name
        for current_album, previous_album_name
        in zip(
            previous_album[
                "album"
            ].iloc[1:],
            previous_album[
                "album"
            ].iloc[:-1],
        )
    }

    records: list[
        dict[str, object]
    ] = []

    for index, row in analysis_df.iterrows():
        vector = standardized_df.loc[
            index
        ].values

        album = row["album"]

        own_centroid = album_centroids.loc[
            album
        ].values

        own_distance = float(
            np.linalg.norm(
                vector - own_centroid
            )
        )

        prior_album = previous_map.get(
            album
        )

        prior_distance = np.nan

        if prior_album is not None:
            prior_centroid = (
                album_centroids.loc[
                    prior_album
                ].values
            )

            prior_distance = float(
                np.linalg.norm(
                    vector - prior_centroid
                )
            )

        records.append(
            {
                "master_track_id": row[
                    "master_track_id"
                ],
                "track_title": row[
                    "track_title"
                ],
                "album": album,
                "album_order": row[
                    "album_order"
                ],
                "release_year": row[
                    "release_year"
                ],
                "era": row["era"],
                "distance_to_own_album_centroid": round(
                    own_distance,
                    6,
                ),
                "previous_album": prior_album,
                "distance_to_previous_album_centroid": (
                    round(
                        prior_distance,
                        6,
                    )
                    if pd.notna(
                        prior_distance
                    )
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(records)


def build_embedding_album_centroids(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build chronological hybrid PCA album centroids."""

    return (
        df.groupby(
            [
                "album_order",
                "album",
                "release_year",
                "era",
            ],
            dropna=False,
        )
        .agg(
            pca_x=(
                "hybrid_pca_x",
                "mean",
            ),
            pca_y=(
                "hybrid_pca_y",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "album_order"
        )
        .reset_index(drop=True)
    )


def plot_velocity(
    transitions: pd.DataFrame,
) -> None:
    """Plot album-to-album evolution velocity."""

    labels = (
        transitions["from_album"]
        + " → "
        + transitions["to_album"]
    )

    figure, axis = plt.subplots(
        figsize=(13, 7)
    )

    axis.bar(
        labels,
        transitions[
            "evolution_velocity"
        ],
    )

    axis.set_title(
        "Linkin Park Evolution Velocity by Album Transition"
    )

    axis.set_xlabel(
        "Album transition"
    )

    axis.set_ylabel(
        "Standardized profile distance per year"
    )

    axis.tick_params(
        axis="x",
        rotation=40,
    )

    figure.tight_layout()

    figure.savefig(
        VELOCITY_FIGURE_PATH,
        dpi=230,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_album_trajectory(
    centroids: pd.DataFrame,
) -> None:
    """Plot chronological album movement in hybrid PCA space."""

    figure, axis = plt.subplots(
        figsize=(12, 9)
    )

    axis.plot(
        centroids["pca_x"],
        centroids["pca_y"],
        marker="o",
    )

    for _, row in centroids.iterrows():
        axis.annotate(
            (
                f"{int(row['album_order'])}. "
                f"{row['album']}"
            ),
            (
                row["pca_x"],
                row["pca_y"],
            ),
            fontsize=9,
        )

    axis.set_title(
        "Chronological Linkin Park Album Trajectory"
    )

    axis.set_xlabel(
        "Hybrid PCA dimension 1"
    )

    axis.set_ylabel(
        "Hybrid PCA dimension 2"
    )

    figure.tight_layout()

    figure.savefig(
        TRAJECTORY_FIGURE_PATH,
        dpi=230,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_drift_heatmap(
    drifts: pd.DataFrame,
) -> None:
    """Plot the strongest feature drifts across transitions."""

    top_features = (
        drifts.groupby(
            "feature"
        )["absolute_change"]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(18)
        .index
    )

    heatmap_df = (
        drifts[
            drifts["feature"].isin(
                top_features
            )
        ]
        .pivot(
            index="feature",
            columns="transition_order",
            values="standardized_change",
        )
        .reindex(
            top_features
        )
    )

    transition_labels = (
        drifts[
            [
                "transition_order",
                "from_album",
                "to_album",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "transition_order"
        )
    )

    figure, axis = plt.subplots(
        figsize=(15, 10)
    )

    image = axis.imshow(
        heatmap_df.values,
        aspect="auto",
        interpolation="nearest",
    )

    axis.set_title(
        "Largest Standardized Feature Drifts"
    )

    axis.set_yticks(
        np.arange(
            len(
                heatmap_df.index
            )
        )
    )

    axis.set_yticklabels(
        [
            feature.replace(
                "lyrics_",
                "",
            ).replace(
                "audio_ab_",
                "audio_",
            ).replace(
                "_",
                " ",
            )
            for feature in heatmap_df.index
        ]
    )

    axis.set_xticks(
        np.arange(
            len(
                transition_labels
            )
        )
    )

    axis.set_xticklabels(
        [
            (
                f"{row['from_album']} → "
                f"{row['to_album']}"
            )
            for _, row
            in transition_labels.iterrows()
        ],
        rotation=45,
        ha="right",
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Standardized change",
    )

    figure.tight_layout()

    figure.savefig(
        DRIFT_HEATMAP_PATH,
        dpi=230,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_era_transition(
    era_transition: pd.DataFrame,
) -> None:
    """Plot the largest Chester-to-Emily feature changes."""

    top = (
        era_transition.head(15)
        .sort_values(
            "raw_change"
        )
    )

    labels = [
        feature.replace(
            "lyrics_",
            "",
        ).replace(
            "audio_ab_",
            "audio_",
        ).replace(
            "_",
            " ",
        )
        for feature in top[
            "feature"
        ]
    ]

    figure, axis = plt.subplots(
        figsize=(12, 8)
    )

    axis.barh(
        labels,
        top["raw_change"],
    )

    axis.axvline(
        0,
        linewidth=1,
    )

    axis.set_title(
        "Largest Chester Era → Emily Era Feature Changes"
    )

    axis.set_xlabel(
        "Emily mean minus Chester mean"
    )

    figure.tight_layout()

    figure.savefig(
        ERA_TRANSITION_FIGURE_PATH,
        dpi=230,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_song_distances(
    song_distances: pd.DataFrame,
) -> None:
    """Plot song distance to own album centroid."""

    grouped = [
        group[
            "distance_to_own_album_centroid"
        ].dropna().values
        for _, group
        in song_distances.groupby(
            "album",
            sort=False,
        )
    ]

    labels = [
        album
        for album, _
        in song_distances.groupby(
            "album",
            sort=False,
        )
    ]

    figure, axis = plt.subplots(
        figsize=(13, 7)
    )

    axis.boxplot(
        grouped,
        tick_labels=labels,
        showfliers=True,
    )

    axis.set_title(
        "Within-Album Song Dispersion"
    )

    axis.set_xlabel(
        "Studio album"
    )

    axis.set_ylabel(
        "Distance to album centroid"
    )

    axis.tick_params(
        axis="x",
        rotation=35,
    )

    figure.tight_layout()

    figure.savefig(
        SONG_DISTANCE_FIGURE_PATH,
        dpi=230,
        bbox_inches="tight",
    )

    plt.close(figure)


def print_findings(
    transitions: pd.DataFrame,
    era_transition: pd.DataFrame,
    song_distances: pd.DataFrame,
) -> None:
    """Print the most important temporal findings."""

    fastest = transitions.sort_values(
        "evolution_velocity",
        ascending=False,
    ).iloc[0]

    largest = transitions.sort_values(
        "profile_distance",
        ascending=False,
    ).iloc[0]

    most_cohesive = (
        song_distances.groupby(
            "album"
        )[
            "distance_to_own_album_centroid"
        ]
        .mean()
        .sort_values()
    )

    print(
        "\nTemporal dynamics findings:"
    )

    print(
        "- Largest absolute album shift: "
        f"{largest['from_album']} → "
        f"{largest['to_album']} "
        f"({largest['profile_distance']:.4f})"
    )

    print(
        "- Fastest evolution velocity: "
        f"{fastest['from_album']} → "
        f"{fastest['to_album']} "
        f"({fastest['evolution_velocity']:.4f} per year)"
    )

    print(
        "- Most internally cohesive album: "
        f"{most_cohesive.index[0]} "
        f"({most_cohesive.iloc[0]:.4f})"
    )

    print(
        "- Most internally diverse album: "
        f"{most_cohesive.index[-1]} "
        f"({most_cohesive.iloc[-1]:.4f})"
    )

    print(
        "- Largest Chester-to-Emily feature changes:"
    )

    for _, row in era_transition.head(8).iterrows():
        print(
            f"  {row['feature']}: "
            f"{row['raw_change']:+.4f}"
        )


def validate_outputs(
    profiles: pd.DataFrame,
    transitions: pd.DataFrame,
) -> None:
    """Run critical temporal-output checks."""

    if len(profiles) != 8:
        raise ValueError(
            f"Expected 8 album profiles, found {len(profiles)}."
        )

    if len(transitions) != 7:
        raise ValueError(
            f"Expected 7 album transitions, found {len(transitions)}."
        )

    if profiles["album_order"].duplicated().any():
        raise ValueError(
            "Duplicate album order values found."
        )


def main() -> None:
    """Run the complete temporal dynamics analysis."""

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df, _ = load_inputs()

    features = select_temporal_features(
        df
    )

    profiles, standardized = (
        build_album_profiles(
            df=df,
            features=features,
        )
    )

    transitions, drifts = build_transitions(
        profiles=profiles,
        standardized=standardized,
        features=features,
    )

    era_transition = build_era_transition(
        df=df,
        features=features,
    )

    song_distances = (
        build_song_temporal_distances(
            df=df,
            features=features,
        )
    )

    embedding_centroids = (
        build_embedding_album_centroids(
            df
        )
    )

    validate_outputs(
        profiles=profiles,
        transitions=transitions,
    )

    profiles.to_csv(
        ALBUM_PROFILES_PATH,
        index=False,
        encoding="utf-8",
    )

    transitions.to_csv(
        TRANSITIONS_PATH,
        index=False,
        encoding="utf-8",
    )

    drifts.to_csv(
        FEATURE_DRIFTS_PATH,
        index=False,
        encoding="utf-8",
    )

    era_transition.to_csv(
        ERA_TRANSITION_PATH,
        index=False,
        encoding="utf-8",
    )

    song_distances.to_csv(
        SONG_DISTANCES_PATH,
        index=False,
        encoding="utf-8",
    )

    plot_velocity(
        transitions
    )

    plot_album_trajectory(
        embedding_centroids
    )

    plot_drift_heatmap(
        drifts
    )

    plot_era_transition(
        era_transition
    )

    plot_song_distances(
        song_distances
    )

    print("\nTemporal album transitions:")
    print(
        transitions.to_string(
            index=False
        )
    )

    print_findings(
        transitions=transitions,
        era_transition=era_transition,
        song_distances=song_distances,
    )

    print("\nSaved:")
    print(f"- {ALBUM_PROFILES_PATH}")
    print(f"- {TRANSITIONS_PATH}")
    print(f"- {FEATURE_DRIFTS_PATH}")
    print(f"- {ERA_TRANSITION_PATH}")
    print(f"- {SONG_DISTANCES_PATH}")
    print(f"- {VELOCITY_FIGURE_PATH}")
    print(f"- {TRAJECTORY_FIGURE_PATH}")
    print(f"- {DRIFT_HEATMAP_PATH}")
    print(f"- {ERA_TRANSITION_FIGURE_PATH}")
    print(f"- {SONG_DISTANCE_FIGURE_PATH}")

    print("\nTemporal dynamics analysis completed.")


if __name__ == "__main__":
    main()
