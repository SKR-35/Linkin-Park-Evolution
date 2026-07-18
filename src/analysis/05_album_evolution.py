"""Analyze Linkin Park's album-level evolution across lyrics and audio.

Input
-----
data/processed/master_dataset.parquet

Outputs
-------
outputs/tables/album_evolution_full.csv
outputs/tables/album_evolution_rankings.csv
outputs/tables/era_comparison.csv
outputs/tables/album_change_points.csv

outputs/figures/album_evolution_sentiment.png
outputs/figures/album_evolution_emotions.png
outputs/figures/album_evolution_lyrics_style.png
outputs/figures/album_evolution_themes.png
outputs/figures/album_evolution_audio.png
outputs/figures/album_evolution_heatmap.png

Purpose
-------
This script measures how Linkin Park's lyrical and musical profile changed
across eight studio albums. It aggregates track-level features, compares
Chester and Emily eras, identifies the largest album-to-album shifts, and
produces publication-ready figures.

Notes
-----
- From Zero is fully included in lyrics and emotion analyses.
- AcousticBrainz audio features are unavailable for From Zero and are left
  missing rather than imputed at album level.
- Instrumental tracks are excluded from lyrics-based averages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_dataset.parquet"
)

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

ALBUM_EVOLUTION_PATH = (
    TABLES_DIR
    / "album_evolution_full.csv"
)

RANKINGS_PATH = (
    TABLES_DIR
    / "album_evolution_rankings.csv"
)

ERA_COMPARISON_PATH = (
    TABLES_DIR
    / "era_comparison.csv"
)

CHANGE_POINTS_PATH = (
    TABLES_DIR
    / "album_change_points.csv"
)

SENTIMENT_FIGURE_PATH = (
    FIGURES_DIR
    / "album_evolution_sentiment.png"
)

EMOTIONS_FIGURE_PATH = (
    FIGURES_DIR
    / "album_evolution_emotions.png"
)

LYRICS_STYLE_FIGURE_PATH = (
    FIGURES_DIR
    / "album_evolution_lyrics_style.png"
)

THEMES_FIGURE_PATH = (
    FIGURES_DIR
    / "album_evolution_themes.png"
)

AUDIO_FIGURE_PATH = (
    FIGURES_DIR
    / "album_evolution_audio.png"
)

HEATMAP_FIGURE_PATH = (
    FIGURES_DIR
    / "album_evolution_heatmap.png"
)


IDENTITY_COLUMNS = [
    "master_track_id",
    "album_order",
    "album",
    "release_year",
    "track_title",
    "era",
    "is_instrumental",
    "has_lyrics_features",
    "has_audio_features",
]


LYRICS_METRICS = {
    "lyrics_word_count": "avg_word_count",
    "lyrics_type_token_ratio": "avg_lexical_diversity",
    "lyrics_line_repetition_ratio": "avg_line_repetition",
    "lyrics_first_person_singular_ratio": "avg_first_person_singular",
    "lyrics_second_person_ratio": "avg_second_person",
    "lyrics_readability_flesch_reading_ease": "avg_flesch_reading_ease",
    "lyrics_vader_compound": "avg_vader_compound",
    "lyrics_vader_negative": "avg_vader_negative",
    "lyrics_vader_positive": "avg_vader_positive",
    "lyrics_nrc_anger_ratio": "avg_nrc_anger",
    "lyrics_nrc_anticipation_ratio": "avg_nrc_anticipation",
    "lyrics_nrc_disgust_ratio": "avg_nrc_disgust",
    "lyrics_nrc_fear_ratio": "avg_nrc_fear",
    "lyrics_nrc_joy_ratio": "avg_nrc_joy",
    "lyrics_nrc_sadness_ratio": "avg_nrc_sadness",
    "lyrics_nrc_surprise_ratio": "avg_nrc_surprise",
    "lyrics_nrc_trust_ratio": "avg_nrc_trust",
    "lyrics_theme_darkness_ratio": "avg_theme_darkness",
    "lyrics_theme_pain_ratio": "avg_theme_pain",
    "lyrics_theme_hope_ratio": "avg_theme_hope",
    "lyrics_theme_conflict_ratio": "avg_theme_conflict",
    "lyrics_theme_isolation_ratio": "avg_theme_isolation",
    "lyrics_theme_time_ratio": "avg_theme_time",
}


AUDIO_METRICS = {
    "audio_ab_bpm": "avg_audio_bpm",
    "audio_ab_average_loudness": "avg_audio_loudness",
    "audio_ab_dynamic_complexity": "avg_audio_dynamic_complexity",
    "audio_ab_danceability": "avg_audio_danceability",
    "audio_ab_mood_aggressive_probability": "avg_audio_aggressive",
    "audio_ab_mood_happy_probability": "avg_audio_happy",
    "audio_ab_mood_relaxed_probability": "avg_audio_relaxed",
    "audio_ab_mood_sad_probability": "avg_audio_sad",
    "audio_ab_mood_electronic_probability": "avg_audio_electronic",
    "audio_ab_spectral_centroid_mean": "avg_audio_spectral_centroid",
    "audio_ab_onset_rate": "avg_audio_onset_rate",
}


HEATMAP_METRICS = [
    "avg_vader_compound",
    "avg_nrc_anger",
    "avg_nrc_fear",
    "avg_nrc_joy",
    "avg_nrc_sadness",
    "avg_theme_pain",
    "avg_theme_hope",
    "avg_theme_conflict",
    "avg_theme_isolation",
    "avg_lexical_diversity",
    "avg_line_repetition",
    "avg_audio_aggressive",
    "avg_audio_sad",
]


def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    """Raise a clear error if required columns are missing."""

    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(
            "Master dataset is missing required columns: "
            f"{sorted(missing)}"
        )


def load_dataset() -> pd.DataFrame:
    """Load and validate the analytical dataset."""

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
        df["track_title"]
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


def existing_metric_map(
    df: pd.DataFrame,
    requested: dict[str, str],
) -> dict[str, str]:
    """Keep only source metrics present in the dataset."""

    return {
        source: target
        for source, target in requested.items()
        if source in df.columns
    }


def build_album_evolution(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate lyrics and audio metrics by album."""

    lyrics_metric_map = existing_metric_map(
        df,
        LYRICS_METRICS,
    )

    audio_metric_map = existing_metric_map(
        df,
        AUDIO_METRICS,
    )

    lyrics_df = df[
        (~df["is_instrumental"])
        & df["has_lyrics_features"]
    ].copy()

    lyrics_named_agg = {
        target: (source, "mean")
        for source, target in lyrics_metric_map.items()
    }

    lyrics_summary = (
        lyrics_df.groupby(
            [
                "album_order",
                "album",
                "release_year",
                "era",
            ],
            dropna=False,
        )
        .agg(
            tracks_with_lyrics=("master_track_id", "count"),
            **lyrics_named_agg,
        )
        .reset_index()
    )

    album_track_counts = (
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
            tracks_total=("master_track_id", "count"),
            instrumentals=("is_instrumental", "sum"),
            tracks_with_audio=("has_audio_features", "sum"),
        )
        .reset_index()
    )

    summary = album_track_counts.merge(
        lyrics_summary,
        on=[
            "album_order",
            "album",
            "release_year",
            "era",
        ],
        how="left",
        validate="one_to_one",
    )

    if audio_metric_map:
        audio_df = df[
            df["has_audio_features"]
        ].copy()

        audio_named_agg = {
            target: (source, "mean")
            for source, target in audio_metric_map.items()
        }

        audio_summary = (
            audio_df.groupby(
                [
                    "album_order",
                    "album",
                ],
                dropna=False,
            )
            .agg(**audio_named_agg)
            .reset_index()
        )

        summary = summary.merge(
            audio_summary,
            on=["album_order", "album"],
            how="left",
            validate="one_to_one",
        )

    summary["lyrics_coverage_rate"] = (
        summary["tracks_with_lyrics"]
        / summary["tracks_total"]
    )

    summary["audio_coverage_rate"] = (
        summary["tracks_with_audio"]
        / summary["tracks_total"]
    )

    numeric_columns = summary.select_dtypes(
        include="number"
    ).columns

    summary[numeric_columns] = (
        summary[numeric_columns]
        .round(5)
    )

    return (
        summary
        .sort_values("album_order")
        .reset_index(drop=True)
    )


def build_rankings(
    album_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Rank albums for every analytical metric."""

    excluded_columns = {
        "album_order",
        "release_year",
        "tracks_total",
        "instrumentals",
        "tracks_with_lyrics",
        "tracks_with_audio",
        "lyrics_coverage_rate",
        "audio_coverage_rate",
    }

    metric_columns = [
        column
        for column in album_summary.select_dtypes(
            include="number"
        ).columns
        if column not in excluded_columns
    ]

    records: list[dict[str, object]] = []

    for metric in metric_columns:
        available = album_summary[
            album_summary[metric].notna()
        ][
            [
                "album_order",
                "album",
                "release_year",
                "era",
                metric,
            ]
        ].copy()

        if available.empty:
            continue

        descending = (
            available.sort_values(
                metric,
                ascending=False,
            )
            .reset_index(drop=True)
        )

        ascending = (
            available.sort_values(
                metric,
                ascending=True,
            )
            .reset_index(drop=True)
        )

        highest = descending.iloc[0]
        lowest = ascending.iloc[0]

        records.extend(
            [
                {
                    "metric": metric,
                    "extreme": "highest",
                    "rank": 1,
                    "album": highest["album"],
                    "release_year": highest["release_year"],
                    "era": highest["era"],
                    "value": highest[metric],
                },
                {
                    "metric": metric,
                    "extreme": "lowest",
                    "rank": 1,
                    "album": lowest["album"],
                    "release_year": lowest["release_year"],
                    "era": lowest["era"],
                    "value": lowest[metric],
                },
            ]
        )

    return pd.DataFrame(records)


def build_era_comparison(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare Chester and Emily eras using lyrics and available audio."""

    lyric_sources = existing_metric_map(
        df,
        LYRICS_METRICS,
    )

    audio_sources = existing_metric_map(
        df,
        AUDIO_METRICS,
    )

    records: list[dict[str, object]] = []

    for era, era_df in df.groupby(
        "era",
        dropna=False,
    ):
        lyrics_df = era_df[
            (~era_df["is_instrumental"])
            & era_df["has_lyrics_features"]
        ]

        audio_df = era_df[
            era_df["has_audio_features"]
        ]

        record: dict[str, object] = {
            "era": era,
            "albums": int(
                era_df["album"].nunique()
            ),
            "tracks_total": len(era_df),
            "tracks_with_lyrics": len(lyrics_df),
            "tracks_with_audio": len(audio_df),
        }

        for source, target in lyric_sources.items():
            record[target] = float(
                pd.to_numeric(
                    lyrics_df[source],
                    errors="coerce",
                ).mean()
            )

        for source, target in audio_sources.items():
            record[target] = float(
                pd.to_numeric(
                    audio_df[source],
                    errors="coerce",
                ).mean()
            )

        records.append(record)

    result = pd.DataFrame(records)

    numeric_columns = result.select_dtypes(
        include="number"
    ).columns

    result[numeric_columns] = (
        result[numeric_columns]
        .round(5)
    )

    return result


def build_change_points(
    album_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Measure largest standardized album-to-album profile shifts."""

    metric_columns = [
        column
        for column in HEATMAP_METRICS
        if column in album_summary.columns
    ]

    matrix = (
        album_summary[metric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    # Standardize only on available values, then use zero for missing
    # standardized audio values so missingness does not dominate distance.
    standardized = pd.DataFrame(
        index=matrix.index,
        columns=metric_columns,
        dtype=float,
    )

    for column in metric_columns:
        values = matrix[column]

        mean = values.mean()
        std = values.std(ddof=0)

        if pd.isna(std) or std == 0:
            standardized[column] = 0.0
        else:
            standardized[column] = (
                values - mean
            ) / std

    standardized = standardized.fillna(0.0)

    records: list[dict[str, object]] = []

    for index in range(
        1,
        len(album_summary),
    ):
        previous = standardized.iloc[
            index - 1
        ]

        current = standardized.iloc[
            index
        ]

        differences = current - previous

        distance = float(
            np.sqrt(
                np.square(
                    differences
                ).sum()
            )
        )

        top_changes = (
            differences.abs()
            .sort_values(
                ascending=False
            )
            .head(5)
        )

        records.append(
            {
                "from_album": album_summary.iloc[
                    index - 1
                ]["album"],
                "to_album": album_summary.iloc[
                    index
                ]["album"],
                "from_year": album_summary.iloc[
                    index - 1
                ]["release_year"],
                "to_year": album_summary.iloc[
                    index
                ]["release_year"],
                "standardized_profile_distance": round(
                    distance,
                    5,
                ),
                "largest_metric_changes": " | ".join(
                    (
                        f"{metric}:"
                        f"{differences[metric]:+.3f}"
                    )
                    for metric in top_changes.index
                ),
            }
        )

    result = pd.DataFrame(records)

    return (
        result.sort_values(
            "standardized_profile_distance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def plot_sentiment(
    album_summary: pd.DataFrame,
) -> None:
    """Plot VADER sentiment evolution."""

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    axis.plot(
        album_summary["album_order"],
        album_summary["avg_vader_compound"],
        marker="o",
        label="Compound",
    )

    axis.plot(
        album_summary["album_order"],
        album_summary["avg_vader_negative"],
        marker="o",
        label="Negative",
    )

    axis.plot(
        album_summary["album_order"],
        album_summary["avg_vader_positive"],
        marker="o",
        label="Positive",
    )

    axis.axhline(
        0,
        linewidth=1,
    )

    axis.set_title(
        "Linkin Park Album Evolution: Sentiment"
    )

    axis.set_xlabel("Studio album")
    axis.set_ylabel("Average VADER score")

    axis.set_xticks(
        album_summary["album_order"]
    )

    axis.set_xticklabels(
        album_summary["album"],
        rotation=35,
        ha="right",
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        SENTIMENT_FIGURE_PATH,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_emotions(
    album_summary: pd.DataFrame,
) -> None:
    """Plot NRC emotion evolution."""

    emotion_columns = {
        "avg_nrc_anger": "Anger",
        "avg_nrc_fear": "Fear",
        "avg_nrc_joy": "Joy",
        "avg_nrc_sadness": "Sadness",
        "avg_nrc_trust": "Trust",
        "avg_nrc_anticipation": "Anticipation",
    }

    available = {
        column: label
        for column, label in emotion_columns.items()
        if column in album_summary.columns
    }

    figure, axis = plt.subplots(
        figsize=(13, 7)
    )

    for column, label in available.items():
        axis.plot(
            album_summary["album_order"],
            album_summary[column],
            marker="o",
            label=label,
        )

    axis.set_title(
        "Linkin Park Album Evolution: NRC Emotions"
    )

    axis.set_xlabel("Studio album")
    axis.set_ylabel("Average emotion ratio")

    axis.set_xticks(
        album_summary["album_order"]
    )

    axis.set_xticklabels(
        album_summary["album"],
        rotation=35,
        ha="right",
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        EMOTIONS_FIGURE_PATH,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_lyrics_style(
    album_summary: pd.DataFrame,
) -> None:
    """Plot lexical richness, repetition, and word count."""

    figure, axis = plt.subplots(
        figsize=(12, 7)
    )

    axis.plot(
        album_summary["album_order"],
        album_summary["avg_lexical_diversity"],
        marker="o",
        label="Lexical diversity",
    )

    axis.plot(
        album_summary["album_order"],
        album_summary["avg_line_repetition"],
        marker="o",
        label="Line repetition",
    )

    axis.set_title(
        "Linkin Park Album Evolution: Lyrics Style"
    )

    axis.set_xlabel("Studio album")
    axis.set_ylabel("Average ratio")

    axis.set_xticks(
        album_summary["album_order"]
    )

    axis.set_xticklabels(
        album_summary["album"],
        rotation=35,
        ha="right",
    )

    secondary_axis = axis.twinx()

    secondary_axis.plot(
        album_summary["album_order"],
        album_summary["avg_word_count"],
        marker="s",
        linestyle="--",
        label="Word count",
    )

    secondary_axis.set_ylabel(
        "Average words per song"
    )

    handles_1, labels_1 = axis.get_legend_handles_labels()
    handles_2, labels_2 = (
        secondary_axis.get_legend_handles_labels()
    )

    axis.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="best",
    )

    figure.tight_layout()

    figure.savefig(
        LYRICS_STYLE_FIGURE_PATH,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_themes(
    album_summary: pd.DataFrame,
) -> None:
    """Plot hand-built lyrical theme evolution."""

    theme_columns = {
        "avg_theme_pain": "Pain",
        "avg_theme_darkness": "Darkness",
        "avg_theme_hope": "Hope",
        "avg_theme_conflict": "Conflict",
        "avg_theme_isolation": "Isolation",
        "avg_theme_time": "Time",
    }

    available = {
        column: label
        for column, label in theme_columns.items()
        if column in album_summary.columns
    }

    figure, axis = plt.subplots(
        figsize=(13, 7)
    )

    for column, label in available.items():
        axis.plot(
            album_summary["album_order"],
            album_summary[column],
            marker="o",
            label=label,
        )

    axis.set_title(
        "Linkin Park Album Evolution: Lyrical Themes"
    )

    axis.set_xlabel("Studio album")
    axis.set_ylabel("Average theme ratio")

    axis.set_xticks(
        album_summary["album_order"]
    )

    axis.set_xticklabels(
        album_summary["album"],
        rotation=35,
        ha="right",
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        THEMES_FIGURE_PATH,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_audio(
    album_summary: pd.DataFrame,
) -> None:
    """Plot available AcousticBrainz album evolution."""

    required_columns = [
        "avg_audio_aggressive",
        "avg_audio_happy",
        "avg_audio_relaxed",
        "avg_audio_sad",
    ]

    available_columns = [
        column
        for column in required_columns
        if column in album_summary.columns
    ]

    if not available_columns:
        print(
            "Audio evolution figure skipped: "
            "no usable audio mood columns."
        )
        return

    audio_summary = album_summary[
        album_summary[
            available_columns
        ]
        .notna()
        .any(axis=1)
    ].copy()

    if audio_summary.empty:
        print(
            "Audio evolution figure skipped: "
            "no album has audio data."
        )
        return

    labels = {
        "avg_audio_aggressive": "Aggressive",
        "avg_audio_happy": "Happy",
        "avg_audio_relaxed": "Relaxed",
        "avg_audio_sad": "Sad",
    }

    figure, axis = plt.subplots(
        figsize=(12, 7)
    )

    for column in available_columns:
        axis.plot(
            audio_summary["album_order"],
            audio_summary[column],
            marker="o",
            label=labels[column],
        )

    axis.set_title(
        "Linkin Park Album Evolution: AcousticBrainz Moods"
    )

    axis.set_xlabel("Studio album")
    axis.set_ylabel("Average classifier probability")

    axis.set_xticks(
        audio_summary["album_order"]
    )

    axis.set_xticklabels(
        audio_summary["album"],
        rotation=35,
        ha="right",
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        AUDIO_FIGURE_PATH,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_heatmap(
    album_summary: pd.DataFrame,
) -> None:
    """Plot standardized album profiles as a heatmap."""

    metric_columns = [
        column
        for column in HEATMAP_METRICS
        if (
            column in album_summary.columns
            and album_summary[column].notna().sum() >= 2
        )
    ]

    matrix = (
        album_summary[metric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    imputed = matrix.copy()

    for column in metric_columns:
        imputed[column] = (
            imputed[column]
            .fillna(
                imputed[column].median()
            )
        )

    scaler = StandardScaler()

    standardized = scaler.fit_transform(
        imputed
    )

    figure, axis = plt.subplots(
        figsize=(14, 8)
    )

    image = axis.imshow(
        standardized,
        aspect="auto",
        interpolation="nearest",
    )

    axis.set_title(
        "Standardized Album Evolution Profile"
    )

    axis.set_yticks(
        np.arange(
            len(album_summary)
        )
    )

    axis.set_yticklabels(
        album_summary["album"]
    )

    axis.set_xticks(
        np.arange(
            len(metric_columns)
        )
    )

    axis.set_xticklabels(
        [
            column.replace(
                "avg_",
                "",
            ).replace(
                "_",
                " ",
            )
            for column in metric_columns
        ],
        rotation=45,
        ha="right",
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Standardized album score",
    )

    figure.tight_layout()

    figure.savefig(
        HEATMAP_FIGURE_PATH,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def print_findings(
    album_summary: pd.DataFrame,
    rankings: pd.DataFrame,
    change_points: pd.DataFrame,
) -> None:
    """Print the most important evolution findings."""

    print("\nKey album evolution findings:")

    metrics_to_report = [
        "avg_vader_compound",
        "avg_nrc_anger",
        "avg_nrc_sadness",
        "avg_nrc_joy",
        "avg_lexical_diversity",
        "avg_line_repetition",
        "avg_theme_pain",
        "avg_theme_hope",
        "avg_theme_conflict",
    ]

    for metric in metrics_to_report:
        metric_rows = rankings[
            rankings["metric"].eq(metric)
        ]

        if metric_rows.empty:
            continue

        highest = metric_rows[
            metric_rows["extreme"].eq(
                "highest"
            )
        ].iloc[0]

        lowest = metric_rows[
            metric_rows["extreme"].eq(
                "lowest"
            )
        ].iloc[0]

        print(
            f"- {metric}: highest "
            f"{highest['album']} "
            f"({highest['value']:.4f}), "
            f"lowest {lowest['album']} "
            f"({lowest['value']:.4f})"
        )

    if not change_points.empty:
        largest_shift = change_points.iloc[0]

        print(
            "- Largest album-to-album profile shift: "
            f"{largest_shift['from_album']} → "
            f"{largest_shift['to_album']} "
            f"({largest_shift['standardized_profile_distance']:.4f})"
        )


def validate_outputs(
    album_summary: pd.DataFrame,
) -> None:
    """Run critical album-summary checks."""

    if len(album_summary) != 8:
        raise ValueError(
            f"Expected 8 studio albums, found {len(album_summary)}."
        )

    if album_summary["album_order"].duplicated().any():
        raise ValueError(
            "Duplicate album_order values found."
        )

    if album_summary["album"].duplicated().any():
        raise ValueError(
            "Duplicate album names found."
        )


def main() -> None:
    """Run the complete album-evolution analysis."""

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset()

    album_summary = build_album_evolution(
        df
    )

    validate_outputs(
        album_summary
    )

    rankings = build_rankings(
        album_summary
    )

    era_comparison = build_era_comparison(
        df
    )

    change_points = build_change_points(
        album_summary
    )

    album_summary.to_csv(
        ALBUM_EVOLUTION_PATH,
        index=False,
        encoding="utf-8",
    )

    rankings.to_csv(
        RANKINGS_PATH,
        index=False,
        encoding="utf-8",
    )

    era_comparison.to_csv(
        ERA_COMPARISON_PATH,
        index=False,
        encoding="utf-8",
    )

    change_points.to_csv(
        CHANGE_POINTS_PATH,
        index=False,
        encoding="utf-8",
    )

    plot_sentiment(
        album_summary
    )

    plot_emotions(
        album_summary
    )

    plot_lyrics_style(
        album_summary
    )

    plot_themes(
        album_summary
    )

    plot_audio(
        album_summary
    )

    plot_heatmap(
        album_summary
    )

    print("\nAlbum evolution summary:")
    print(
        album_summary.to_string(
            index=False
        )
    )

    print("\nEra comparison:")
    print(
        era_comparison.to_string(
            index=False
        )
    )

    print("\nLargest album-to-album shifts:")
    print(
        change_points.head(7).to_string(
            index=False
        )
    )

    print_findings(
        album_summary=album_summary,
        rankings=rankings,
        change_points=change_points,
    )

    print("\nSaved:")
    print(f"- {ALBUM_EVOLUTION_PATH}")
    print(f"- {RANKINGS_PATH}")
    print(f"- {ERA_COMPARISON_PATH}")
    print(f"- {CHANGE_POINTS_PATH}")
    print(f"- {SENTIMENT_FIGURE_PATH}")
    print(f"- {EMOTIONS_FIGURE_PATH}")
    print(f"- {LYRICS_STYLE_FIGURE_PATH}")
    print(f"- {THEMES_FIGURE_PATH}")

    if AUDIO_FIGURE_PATH.exists():
        print(f"- {AUDIO_FIGURE_PATH}")

    print(f"- {HEATMAP_FIGURE_PATH}")
    print("\nAlbum evolution analysis completed.")


if __name__ == "__main__":
    main()
