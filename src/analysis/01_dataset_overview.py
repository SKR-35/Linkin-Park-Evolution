"""Create the first analytical overview of the Linkin Park master dataset.

Input
-----
data/processed/master_dataset.parquet

Outputs
-------
outputs/tables/dataset_overview_summary.csv
outputs/tables/album_evolution_summary.csv
outputs/tables/top_negative_songs.csv
outputs/tables/top_positive_songs.csv
outputs/tables/top_anger_songs.csv
outputs/tables/top_sadness_songs.csv
outputs/figures/album_vader_evolution.png
outputs/figures/album_emotion_evolution.png
outputs/figures/album_lexical_diversity.png
outputs/figures/album_audio_mood_evolution.png

Purpose
-------
This script provides the first reproducible analytical layer after the
data pipeline. It summarizes catalogue coverage, lyrical evolution,
sentiment, NRC emotions, lexical richness, and available audio moods.

Notes
-----
- From Zero is included in lyrics analyses.
- From Zero is excluded only from AcousticBrainz audio averages because
  the source stopped collecting new analyses in 2022.
- Instrumental tracks are excluded from lyrics-based averages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_dataset.parquet"
)

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

DATASET_SUMMARY_PATH = (
    TABLES_DIR
    / "dataset_overview_summary.csv"
)

ALBUM_SUMMARY_PATH = (
    TABLES_DIR
    / "album_evolution_summary.csv"
)

TOP_NEGATIVE_PATH = (
    TABLES_DIR
    / "top_negative_songs.csv"
)

TOP_POSITIVE_PATH = (
    TABLES_DIR
    / "top_positive_songs.csv"
)

TOP_ANGER_PATH = (
    TABLES_DIR
    / "top_anger_songs.csv"
)

TOP_SADNESS_PATH = (
    TABLES_DIR
    / "top_sadness_songs.csv"
)

VADER_FIGURE_PATH = (
    FIGURES_DIR
    / "album_vader_evolution.png"
)

EMOTION_FIGURE_PATH = (
    FIGURES_DIR
    / "album_emotion_evolution.png"
)

LEXICAL_FIGURE_PATH = (
    FIGURES_DIR
    / "album_lexical_diversity.png"
)

AUDIO_FIGURE_PATH = (
    FIGURES_DIR
    / "album_audio_mood_evolution.png"
)


def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    """Raise a clear error when required columns are absent."""

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
        [
            "master_track_id",
            "album_order",
            "album",
            "release_year",
            "track_title",
            "era",
            "is_instrumental",
            "has_lyrics_features",
            "has_audio_features",
            "analysis_coverage",
            "lyrics_word_count",
            "lyrics_type_token_ratio",
            "lyrics_vader_compound",
            "lyrics_nrc_anger_ratio",
            "lyrics_nrc_joy_ratio",
            "lyrics_nrc_fear_ratio",
            "lyrics_nrc_sadness_ratio",
            "lyrics_nrc_trust_ratio",
            "lyrics_nrc_anticipation_ratio",
        ],
    )

    return (
        df.sort_values(
            ["album_order", "track_position"],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def build_dataset_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build high-level catalogue and source-coverage metrics."""

    non_instrumental = df[
        ~df["is_instrumental"]
    ].copy()

    records = [
        {
            "metric": "tracks_total",
            "value": len(df),
        },
        {
            "metric": "albums_total",
            "value": df["album"].nunique(),
        },
        {
            "metric": "release_year_min",
            "value": int(df["release_year"].min()),
        },
        {
            "metric": "release_year_max",
            "value": int(df["release_year"].max()),
        },
        {
            "metric": "tracks_with_lyrics_features",
            "value": int(df["has_lyrics_features"].sum()),
        },
        {
            "metric": "tracks_with_audio_features",
            "value": int(df["has_audio_features"].sum()),
        },
        {
            "metric": "instrumental_tracks",
            "value": int(df["is_instrumental"].sum()),
        },
        {
            "metric": "total_lyric_words",
            "value": int(
                non_instrumental[
                    "lyrics_word_count"
                ].fillna(0).sum()
            ),
        },
        {
            "metric": "average_words_per_song",
            "value": round(
                float(
                    non_instrumental[
                        "lyrics_word_count"
                    ].mean()
                ),
                4,
            ),
        },
        {
            "metric": "average_vader_compound",
            "value": round(
                float(
                    non_instrumental[
                        "lyrics_vader_compound"
                    ].mean()
                ),
                4,
            ),
        },
    ]

    coverage_counts = (
        df["analysis_coverage"]
        .value_counts(dropna=False)
    )

    for coverage_name, count in coverage_counts.items():
        records.append(
            {
                "metric": (
                    "coverage_"
                    + str(coverage_name)
                ),
                "value": int(count),
            }
        )

    return pd.DataFrame(records)


def build_album_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate the main lyrical and audio features by album."""

    lyrics_df = df[
        (~df["is_instrumental"])
        & df["has_lyrics_features"]
    ].copy()

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
            tracks=("master_track_id", "count"),
            avg_word_count=(
                "lyrics_word_count",
                "mean",
            ),
            median_word_count=(
                "lyrics_word_count",
                "median",
            ),
            avg_lexical_diversity=(
                "lyrics_type_token_ratio",
                "mean",
            ),
            avg_line_repetition=(
                "lyrics_line_repetition_ratio",
                "mean",
            ),
            avg_vader_compound=(
                "lyrics_vader_compound",
                "mean",
            ),
            avg_vader_negative=(
                "lyrics_vader_negative",
                "mean",
            ),
            avg_vader_positive=(
                "lyrics_vader_positive",
                "mean",
            ),
            avg_nrc_anger=(
                "lyrics_nrc_anger_ratio",
                "mean",
            ),
            avg_nrc_anticipation=(
                "lyrics_nrc_anticipation_ratio",
                "mean",
            ),
            avg_nrc_disgust=(
                "lyrics_nrc_disgust_ratio",
                "mean",
            ),
            avg_nrc_fear=(
                "lyrics_nrc_fear_ratio",
                "mean",
            ),
            avg_nrc_joy=(
                "lyrics_nrc_joy_ratio",
                "mean",
            ),
            avg_nrc_sadness=(
                "lyrics_nrc_sadness_ratio",
                "mean",
            ),
            avg_nrc_surprise=(
                "lyrics_nrc_surprise_ratio",
                "mean",
            ),
            avg_nrc_trust=(
                "lyrics_nrc_trust_ratio",
                "mean",
            ),
            avg_first_person_singular=(
                "lyrics_first_person_singular_ratio",
                "mean",
            ),
            avg_second_person=(
                "lyrics_second_person_ratio",
                "mean",
            ),
            avg_pain_theme=(
                "lyrics_theme_pain_ratio",
                "mean",
            ),
            avg_darkness_theme=(
                "lyrics_theme_darkness_ratio",
                "mean",
            ),
            avg_hope_theme=(
                "lyrics_theme_hope_ratio",
                "mean",
            ),
            avg_isolation_theme=(
                "lyrics_theme_isolation_ratio",
                "mean",
            ),
        )
        .reset_index()
    )

    audio_columns = {
        "audio_ab_bpm": "avg_audio_bpm",
        "audio_ab_danceability": "avg_audio_danceability",
        "audio_ab_mood_aggressive_probability": (
            "avg_audio_aggressive_probability"
        ),
        "audio_ab_mood_happy_probability": (
            "avg_audio_happy_probability"
        ),
        "audio_ab_mood_relaxed_probability": (
            "avg_audio_relaxed_probability"
        ),
        "audio_ab_mood_sad_probability": (
            "avg_audio_sad_probability"
        ),
        "audio_ab_average_loudness": (
            "avg_audio_loudness"
        ),
        "audio_ab_dynamic_complexity": (
            "avg_audio_dynamic_complexity"
        ),
    }

    available_audio_columns = {
        source: target
        for source, target in audio_columns.items()
        if source in df.columns
    }

    if available_audio_columns:
        audio_df = df[
            df["has_audio_features"]
        ].copy()

        aggregation = {
            source: "mean"
            for source in available_audio_columns
        }

        audio_summary = (
            audio_df.groupby(
                ["album_order", "album"],
                dropna=False,
            )
            .agg(aggregation)
            .reset_index()
            .rename(
                columns=available_audio_columns
            )
        )

        album_summary = lyrics_summary.merge(
            audio_summary,
            on=["album_order", "album"],
            how="left",
            validate="one_to_one",
        )
    else:
        album_summary = lyrics_summary

    numeric_columns = album_summary.select_dtypes(
        include="number"
    ).columns

    album_summary[numeric_columns] = (
        album_summary[numeric_columns]
        .round(5)
    )

    return (
        album_summary
        .sort_values("album_order")
        .reset_index(drop=True)
    )


def build_ranked_song_table(
    df: pd.DataFrame,
    score_column: str,
    ascending: bool,
    output_path: Path,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank non-instrumental songs by one analytical metric."""

    require_columns(df, [score_column])

    selected_columns = [
        "master_track_id",
        "album_order",
        "album",
        "release_year",
        "track_title",
        "era",
        "lyrics_word_count",
        "lyrics_vader_compound",
        "lyrics_nrc_anger_ratio",
        "lyrics_nrc_joy_ratio",
        "lyrics_nrc_fear_ratio",
        "lyrics_nrc_sadness_ratio",
        "lyrics_nrc_trust_ratio",
        "lyrics_nrc_anticipation_ratio",
    ]

    if score_column not in selected_columns:
        selected_columns.append(score_column)

    ranked = (
        df[
            (~df["is_instrumental"])
            & df[score_column].notna()
        ][selected_columns]
        .sort_values(
            score_column,
            ascending=ascending,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    ranked.insert(
        0,
        "rank",
        np.arange(
            1,
            len(ranked) + 1,
        ),
    )

    ranked.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return ranked


def plot_album_vader(
    album_summary: pd.DataFrame,
) -> None:
    """Plot average VADER compound sentiment across albums."""

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    axis.plot(
        album_summary["album_order"],
        album_summary["avg_vader_compound"],
        marker="o",
    )

    axis.axhline(
        0,
        linewidth=1,
    )

    axis.set_title(
        "Linkin Park Album Evolution: VADER Sentiment"
    )

    axis.set_xlabel("Studio album")
    axis.set_ylabel("Average VADER compound score")

    axis.set_xticks(
        album_summary["album_order"]
    )

    axis.set_xticklabels(
        album_summary["album"],
        rotation=35,
        ha="right",
    )

    figure.tight_layout()

    figure.savefig(
        VADER_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_album_emotions(
    album_summary: pd.DataFrame,
) -> None:
    """Plot selected NRC emotion ratios across albums."""

    figure, axis = plt.subplots(
        figsize=(13, 7)
    )

    emotion_columns = {
        "avg_nrc_anger": "Anger",
        "avg_nrc_fear": "Fear",
        "avg_nrc_joy": "Joy",
        "avg_nrc_sadness": "Sadness",
        "avg_nrc_trust": "Trust",
        "avg_nrc_anticipation": "Anticipation",
    }

    for column, label in emotion_columns.items():
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
        EMOTION_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_lexical_diversity(
    album_summary: pd.DataFrame,
) -> None:
    """Plot average type-token ratio by album."""

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    axis.bar(
        album_summary["album"],
        album_summary["avg_lexical_diversity"],
    )

    axis.set_title(
        "Linkin Park Album Evolution: Lexical Diversity"
    )

    axis.set_xlabel("Studio album")
    axis.set_ylabel("Average type-token ratio")

    axis.tick_params(
        axis="x",
        rotation=35,
    )

    figure.tight_layout()

    figure.savefig(
        LEXICAL_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_audio_moods(
    album_summary: pd.DataFrame,
) -> None:
    """Plot available AcousticBrainz mood probabilities."""

    required_audio_columns = [
        "avg_audio_aggressive_probability",
        "avg_audio_happy_probability",
        "avg_audio_relaxed_probability",
        "avg_audio_sad_probability",
    ]

    if not set(required_audio_columns).issubset(
        album_summary.columns
    ):
        print(
            "Audio mood figure skipped: "
            "required AcousticBrainz columns are unavailable."
        )
        return

    audio_summary = album_summary[
        album_summary[
            "avg_audio_aggressive_probability"
        ].notna()
    ].copy()

    if audio_summary.empty:
        print(
            "Audio mood figure skipped: "
            "no album has audio mood data."
        )
        return

    figure, axis = plt.subplots(
        figsize=(12, 7)
    )

    mood_columns = {
        "avg_audio_aggressive_probability": "Aggressive",
        "avg_audio_happy_probability": "Happy",
        "avg_audio_relaxed_probability": "Relaxed",
        "avg_audio_sad_probability": "Sad",
    }

    for column, label in mood_columns.items():
        axis.plot(
            audio_summary["album_order"],
            audio_summary[column],
            marker="o",
            label=label,
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
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def print_key_findings(
    album_summary: pd.DataFrame,
    top_negative: pd.DataFrame,
    top_positive: pd.DataFrame,
) -> None:
    """Print a compact, data-driven overview."""

    darkest_album = album_summary.loc[
        album_summary[
            "avg_vader_compound"
        ].idxmin()
    ]

    most_positive_album = album_summary.loc[
        album_summary[
            "avg_vader_compound"
        ].idxmax()
    ]

    richest_album = album_summary.loc[
        album_summary[
            "avg_lexical_diversity"
        ].idxmax()
    ]

    most_anger_album = album_summary.loc[
        album_summary[
            "avg_nrc_anger"
        ].idxmax()
    ]

    most_sad_album = album_summary.loc[
        album_summary[
            "avg_nrc_sadness"
        ].idxmax()
    ]

    print("\nInitial analytical findings:")
    print(
        "Most negative album by average VADER compound: "
        f"{darkest_album['album']} "
        f"({darkest_album['avg_vader_compound']:.4f})"
    )

    print(
        "Most positive album by average VADER compound: "
        f"{most_positive_album['album']} "
        f"({most_positive_album['avg_vader_compound']:.4f})"
    )

    print(
        "Highest average lexical diversity: "
        f"{richest_album['album']} "
        f"({richest_album['avg_lexical_diversity']:.4f})"
    )

    print(
        "Highest NRC anger ratio: "
        f"{most_anger_album['album']} "
        f"({most_anger_album['avg_nrc_anger']:.4f})"
    )

    print(
        "Highest NRC sadness ratio: "
        f"{most_sad_album['album']} "
        f"({most_sad_album['avg_nrc_sadness']:.4f})"
    )

    if not top_negative.empty:
        print(
            "Most negative song by VADER compound: "
            f"{top_negative.iloc[0]['track_title']} "
            f"— {top_negative.iloc[0]['album']} "
            f"({top_negative.iloc[0]['lyrics_vader_compound']:.4f})"
        )

    if not top_positive.empty:
        print(
            "Most positive song by VADER compound: "
            f"{top_positive.iloc[0]['track_title']} "
            f"— {top_positive.iloc[0]['album']} "
            f"({top_positive.iloc[0]['lyrics_vader_compound']:.4f})"
        )


def main() -> None:
    """Run the first complete analytical overview."""

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset()

    dataset_summary = build_dataset_summary(df)
    album_summary = build_album_summary(df)

    dataset_summary.to_csv(
        DATASET_SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )

    album_summary.to_csv(
        ALBUM_SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )

    top_negative = build_ranked_song_table(
        df=df,
        score_column="lyrics_vader_compound",
        ascending=True,
        output_path=TOP_NEGATIVE_PATH,
    )

    top_positive = build_ranked_song_table(
        df=df,
        score_column="lyrics_vader_compound",
        ascending=False,
        output_path=TOP_POSITIVE_PATH,
    )

    build_ranked_song_table(
        df=df,
        score_column="lyrics_nrc_anger_ratio",
        ascending=False,
        output_path=TOP_ANGER_PATH,
    )

    build_ranked_song_table(
        df=df,
        score_column="lyrics_nrc_sadness_ratio",
        ascending=False,
        output_path=TOP_SADNESS_PATH,
    )

    plot_album_vader(album_summary)
    plot_album_emotions(album_summary)
    plot_lexical_diversity(album_summary)
    plot_audio_moods(album_summary)

    print("\nDataset overview:")
    print(dataset_summary.to_string(index=False))

    print("\nAlbum evolution summary:")
    print(album_summary.to_string(index=False))

    print_key_findings(
        album_summary=album_summary,
        top_negative=top_negative,
        top_positive=top_positive,
    )

    print("\nSaved tables:")
    print(f"- {DATASET_SUMMARY_PATH}")
    print(f"- {ALBUM_SUMMARY_PATH}")
    print(f"- {TOP_NEGATIVE_PATH}")
    print(f"- {TOP_POSITIVE_PATH}")
    print(f"- {TOP_ANGER_PATH}")
    print(f"- {TOP_SADNESS_PATH}")

    print("\nSaved figures:")
    print(f"- {VADER_FIGURE_PATH}")
    print(f"- {EMOTION_FIGURE_PATH}")
    print(f"- {LEXICAL_FIGURE_PATH}")

    if AUDIO_FIGURE_PATH.exists():
        print(f"- {AUDIO_FIGURE_PATH}")

    print("\nDataset overview analysis completed.")


if __name__ == "__main__":
    main()
