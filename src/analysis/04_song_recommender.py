"""Query the precomputed Linkin Park song similarity engine.

Inputs
------
data/processed/song_similarity_pairs.parquet

Optional supporting input
-------------------------
data/processed/master_dataset.parquet

Usage examples
--------------
python src/analysis/04_song_recommender.py --song "Numb"
python src/analysis/04_song_recommender.py --song "Faint" --model audio --top-n 5
python src/analysis/04_song_recommender.py --song "Heavy Is the Crown" --model hybrid
python src/analysis/04_song_recommender.py --list-songs
python src/analysis/04_song_recommender.py --export-all --model hybrid --top-n 10

Outputs
-------
outputs/recommendations/<song>_<model>_top<n>.csv
outputs/recommendations/all_song_recommendations_<model>_top<n>.csv

Purpose
-------
This script is a lightweight user-facing layer over the similarity model.
It does not recompute embeddings or similarity matrices. Instead, it reads
the pair-level results created by 02_song_similarity.py and returns the
nearest songs for any catalogue track.

Notes
-----
- Models: emotion, lyrics_style, audio, hybrid.
- Audio recommendations are unavailable for From Zero because AcousticBrainz
  coverage ends before the album's release.
- Hybrid recommendations are missing-aware and may use two or three models
  depending on source and target coverage.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PAIRS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "song_similarity_pairs.parquet"
)

MASTER_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_dataset.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "recommendations"
)

VALID_MODELS = {
    "emotion": "emotion_similarity",
    "lyrics_style": "lyrics_style_similarity",
    "audio": "audio_similarity",
    "hybrid": "hybrid_similarity",
}

DEFAULT_MODEL = "hybrid"
DEFAULT_TOP_N = 10


def normalize_text(value: Any) -> str:
    """Normalize a title for robust matching."""

    if value is None:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(value),
    )

    text = text.casefold().strip()
    text = text.replace("&", " and ")

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    text = re.sub(
        r"_",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def slugify(value: str) -> str:
    """Create a filesystem-safe lowercase slug."""

    text = normalize_text(value)
    text = re.sub(
        r"\s+",
        "-",
        text,
    )

    return text or "song"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Find songs similar to any Linkin Park track "
            "using precomputed emotion, lyrics, audio, or hybrid models."
        )
    )

    parser.add_argument(
        "--song",
        type=str,
        help=(
            "Track title to query, for example: "
            '--song "Numb"'
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=sorted(VALID_MODELS),
        default=DEFAULT_MODEL,
        help=(
            "Similarity model to use. "
            f"Default: {DEFAULT_MODEL}"
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=(
            "Number of recommendations to return. "
            f"Default: {DEFAULT_TOP_N}"
        ),
    )

    parser.add_argument(
        "--list-songs",
        action="store_true",
        help="List all available catalogue tracks and exit.",
    )

    parser.add_argument(
        "--export-all",
        action="store_true",
        help=(
            "Export top-N recommendations for every available source song."
        ),
    )

    parser.add_argument(
        "--same-era-only",
        action="store_true",
        help=(
            "Restrict recommendations to the same vocalist era "
            "as the source track."
        ),
    )

    parser.add_argument(
        "--exclude-same-album",
        action="store_true",
        help=(
            "Exclude tracks from the same album as the source song."
        ),
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help=(
            "Optional minimum similarity score. "
            "Cosine similarity usually ranges from -1 to 1."
        ),
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print results without writing a CSV file.",
    )

    args = parser.parse_args()

    if args.top_n < 1:
        parser.error("--top-n must be at least 1.")

    if not (
        args.song
        or args.list_songs
        or args.export_all
    ):
        parser.error(
            "Provide --song, --list-songs, or --export-all."
        )

    return args


def load_pairs() -> pd.DataFrame:
    """Load and validate the precomputed similarity pairs."""

    if not PAIRS_PATH.exists():
        raise FileNotFoundError(
            f"Similarity pairs not found: {PAIRS_PATH}\n"
            "Run 02_song_similarity.py first."
        )

    df = pd.read_parquet(PAIRS_PATH)

    required_columns = {
        "source_track_id",
        "target_track_id",
        "source_track_title",
        "target_track_title",
        "source_album",
        "target_album",
        "source_era",
        "target_era",
        *VALID_MODELS.values(),
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Similarity pairs are missing required columns: "
            f"{sorted(missing)}"
        )

    return df


def build_catalogue(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Build a unique source-song catalogue from the pair table."""

    source_columns = [
        "source_track_id",
        "source_track_title",
        "source_canonical_title",
        "source_album_order",
        "source_album",
        "source_release_year",
        "source_track_position",
        "source_era",
        "source_is_instrumental",
        "source_has_audio_features",
    ]

    existing_columns = [
        column
        for column in source_columns
        if column in pairs.columns
    ]

    catalogue = (
        pairs[existing_columns]
        .drop_duplicates(
            subset=["source_track_id"]
        )
        .rename(
            columns={
                column: column.replace(
                    "source_",
                    "",
                    1,
                )
                for column in existing_columns
            }
        )
    )

    catalogue["normalized_title"] = (
        catalogue["track_title"]
        .map(normalize_text)
    )

    if "canonical_title" in catalogue.columns:
        catalogue["normalized_canonical_title"] = (
            catalogue["canonical_title"]
            .map(normalize_text)
        )
    else:
        catalogue["normalized_canonical_title"] = (
            catalogue["normalized_title"]
        )

    sort_columns = [
        column
        for column in [
            "album_order",
            "track_position",
            "track_title",
        ]
        if column in catalogue.columns
    ]

    return (
        catalogue
        .sort_values(
            sort_columns,
            na_position="last",
        )
        .reset_index(drop=True)
    )


def resolve_song(
    catalogue: pd.DataFrame,
    query: str,
) -> pd.Series:
    """Resolve a user-provided title to exactly one track."""

    normalized_query = normalize_text(query)

    exact = catalogue[
        catalogue["normalized_title"].eq(
            normalized_query
        )
        | catalogue[
            "normalized_canonical_title"
        ].eq(
            normalized_query
        )
    ]

    if len(exact) == 1:
        return exact.iloc[0]

    if len(exact) > 1:
        raise ValueError(
            "The title is ambiguous. Matching tracks:\n"
            + exact[
                [
                    "track_title",
                    "album",
                    "release_year",
                ]
            ].to_string(index=False)
        )

    contains = catalogue[
        catalogue["normalized_title"].str.contains(
            normalized_query,
            regex=False,
            na=False,
        )
        | catalogue[
            "normalized_canonical_title"
        ].str.contains(
            normalized_query,
            regex=False,
            na=False,
        )
    ]

    if len(contains) == 1:
        return contains.iloc[0]

    if not contains.empty:
        raise ValueError(
            "No unique exact match was found. Possible tracks:\n"
            + contains[
                [
                    "track_title",
                    "album",
                    "release_year",
                ]
            ]
            .head(15)
            .to_string(index=False)
        )

    # Lightweight token-overlap suggestions.
    query_tokens = set(
        normalized_query.split()
    )

    suggestions = catalogue.copy()

    suggestions["query_overlap"] = (
        suggestions["normalized_title"]
        .map(
            lambda value: len(
                query_tokens
                & set(value.split())
            )
        )
    )

    suggestions = (
        suggestions[
            suggestions["query_overlap"] > 0
        ]
        .sort_values(
            [
                "query_overlap",
                "track_title",
            ],
            ascending=[False, True],
        )
        .head(10)
    )

    message = f"Track not found: {query}"

    if not suggestions.empty:
        message += (
            "\nPossible matches:\n"
            + suggestions[
                [
                    "track_title",
                    "album",
                    "release_year",
                ]
            ].to_string(index=False)
        )

    raise ValueError(message)


def get_recommendations(
    pairs: pd.DataFrame,
    source_track: pd.Series,
    model: str,
    top_n: int,
    same_era_only: bool = False,
    exclude_same_album: bool = False,
    min_score: float | None = None,
) -> pd.DataFrame:
    """Return ranked recommendations for one source song."""

    score_column = VALID_MODELS[model]

    source_id = str(
        source_track["track_id"]
    )

    result = pairs[
        pairs["source_track_id"]
        .astype(str)
        .eq(source_id)
    ].copy()

    result = result[
        result[score_column].notna()
    ]

    if same_era_only:
        result = result[
            result["target_era"].eq(
                source_track["era"]
            )
        ]

    if exclude_same_album:
        result = result[
            ~result["target_album"].eq(
                source_track["album"]
            )
        ]

    if min_score is not None:
        result = result[
            result[score_column] >= min_score
        ]

    result = (
        result.sort_values(
            score_column,
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    if result.empty:
        raise ValueError(
            "No recommendations matched the requested filters."
        )

    result.insert(
        0,
        "rank",
        np.arange(
            1,
            len(result) + 1,
        ),
    )

    result.insert(
        1,
        "model",
        model,
    )

    result.insert(
        2,
        "query_track",
        source_track["track_title"],
    )

    result.insert(
        3,
        "query_album",
        source_track["album"],
    )

    output_columns = [
        "rank",
        "model",
        "query_track",
        "query_album",
        "target_track_title",
        "target_album",
        "target_release_year",
        "target_era",
        score_column,
        "emotion_similarity",
        "lyrics_style_similarity",
        "audio_similarity",
        "hybrid_similarity",
        "hybrid_models_used",
        "target_has_audio_features",
    ]

    output_columns = list(
        dict.fromkeys(
            column
            for column in output_columns
            if column in result.columns
        )
    )

    return result[output_columns]


def print_recommendations(
    recommendations: pd.DataFrame,
    model: str,
) -> None:
    """Print recommendations in a readable terminal format."""

    score_column = VALID_MODELS[model]

    source_song = recommendations.iloc[0][
        "query_track"
    ]

    source_album = recommendations.iloc[0][
        "query_album"
    ]

    print(
        f"\nRecommendations for "
        f"{source_song} — {source_album}"
    )

    print(
        f"Model: {model}"
    )

    for _, row in recommendations.iterrows():
        coverage_note = ""

        if model == "hybrid":
            models_used = row.get(
                "hybrid_models_used"
            )

            if pd.notna(models_used):
                coverage_note = (
                    f", {int(models_used)} views"
                )

        print(
            f"{int(row['rank']):>2}. "
            f"{row['target_track_title']} "
            f"— {row['target_album']} "
            f"({row[score_column]:.4f}{coverage_note})"
        )


def save_single_recommendations(
    recommendations: pd.DataFrame,
    source_track: pd.Series,
    model: str,
    top_n: int,
) -> Path:
    """Save one song's recommendations to CSV."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{slugify(source_track['track_title'])}"
        f"_{model}_top{top_n}.csv"
    )

    output_path = OUTPUT_DIR / filename

    recommendations.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return output_path


def export_all_recommendations(
    pairs: pd.DataFrame,
    catalogue: pd.DataFrame,
    model: str,
    top_n: int,
    same_era_only: bool,
    exclude_same_album: bool,
    min_score: float | None,
) -> Path:
    """Export recommendations for every eligible source song."""

    frames: list[pd.DataFrame] = []

    score_column = VALID_MODELS[model]

    for _, source_track in catalogue.iterrows():
        source_id = str(
            source_track["track_id"]
        )

        source_rows = pairs[
            pairs["source_track_id"]
            .astype(str)
            .eq(source_id)
        ]

        if source_rows[score_column].notna().sum() == 0:
            continue

        try:
            recommendations = get_recommendations(
                pairs=pairs,
                source_track=source_track,
                model=model,
                top_n=top_n,
                same_era_only=same_era_only,
                exclude_same_album=exclude_same_album,
                min_score=min_score,
            )

            frames.append(
                recommendations
            )

        except ValueError:
            continue

    if not frames:
        raise ValueError(
            "No recommendations could be exported."
        )

    all_recommendations = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / (
            "all_song_recommendations_"
            f"{model}_top{top_n}.csv"
        )
    )

    all_recommendations.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return output_path


def list_songs(
    catalogue: pd.DataFrame,
) -> None:
    """Print all catalogue songs in album order."""

    print("\nAvailable songs:")

    for album, album_df in catalogue.groupby(
        "album",
        sort=False,
    ):
        print(f"\n{album}")

        for _, row in album_df.iterrows():
            print(
                f"- {row['track_title']}"
            )


def main() -> None:
    """Run the command-line recommendation interface."""

    args = parse_args()

    pairs = load_pairs()
    catalogue = build_catalogue(pairs)

    if args.list_songs:
        list_songs(catalogue)
        return

    if args.export_all:
        output_path = export_all_recommendations(
            pairs=pairs,
            catalogue=catalogue,
            model=args.model,
            top_n=args.top_n,
            same_era_only=args.same_era_only,
            exclude_same_album=args.exclude_same_album,
            min_score=args.min_score,
        )

        print(
            f"Saved all recommendations: "
            f"{output_path}"
        )

        return

    source_track = resolve_song(
        catalogue=catalogue,
        query=args.song,
    )

    if (
        args.model == "audio"
        and not bool(
            source_track.get(
                "has_audio_features",
                False,
            )
        )
    ):
        raise ValueError(
            f"Audio recommendations are unavailable for "
            f"{source_track['track_title']} because the track "
            "has no AcousticBrainz audio features. "
            "Use --model emotion, lyrics_style, or hybrid."
        )

    recommendations = get_recommendations(
        pairs=pairs,
        source_track=source_track,
        model=args.model,
        top_n=args.top_n,
        same_era_only=args.same_era_only,
        exclude_same_album=args.exclude_same_album,
        min_score=args.min_score,
    )

    print_recommendations(
        recommendations=recommendations,
        model=args.model,
    )

    if not args.no_save:
        output_path = save_single_recommendations(
            recommendations=recommendations,
            source_track=source_track,
            model=args.model,
            top_n=args.top_n,
        )

        print(
            f"\nSaved: {output_path}"
        )


if __name__ == "__main__":
    try:
        main()

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        print(
            f"\nError: {error}",
            file=sys.stderr,
        )

        raise SystemExit(1)
