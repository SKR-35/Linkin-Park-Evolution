"""Build the canonical Linkin Park master track catalogue.

Input
-----
data/raw/musicbrainz/tracks.parquet

Outputs
-------
data/processed/master_tracks.parquet
data/processed/master_tracks.csv
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "musicbrainz"
    / "tracks.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
PARQUET_OUTPUT_PATH = OUTPUT_DIR / "master_tracks.parquet"
CSV_OUTPUT_PATH = OUTPUT_DIR / "master_tracks.csv"


ALBUM_METADATA = {
    "Hybrid Theory": {
        "album_order": 1,
        "era": "Chester Era",
        "primary_vocalist": "Chester Bennington",
        "standard_track_count": 12,
    },
    "Meteora": {
        "album_order": 2,
        "era": "Chester Era",
        "primary_vocalist": "Chester Bennington",
        "standard_track_count": 13,
    },
    "Minutes to Midnight": {
        "album_order": 3,
        "era": "Chester Era",
        "primary_vocalist": "Chester Bennington",
        "standard_track_count": 12,
    },
    "A Thousand Suns": {
        "album_order": 4,
        "era": "Chester Era",
        "primary_vocalist": "Chester Bennington",
        "standard_track_count": 15,
    },
    "Living Things": {
        "album_order": 5,
        "era": "Chester Era",
        "primary_vocalist": "Chester Bennington",
        "standard_track_count": 12,
    },
    "The Hunting Party": {
        "album_order": 6,
        "era": "Chester Era",
        "primary_vocalist": "Chester Bennington",
        "standard_track_count": 12,
    },
    "One More Light": {
        "album_order": 7,
        "era": "Chester Era",
        "primary_vocalist": "Chester Bennington",
        "standard_track_count": 10,
    },
    "From Zero": {
        "album_order": 8,
        "era": "Emily Era",
        "primary_vocalist": "Emily Armstrong",
        "standard_track_count": 11,
    },
}


VERSION_PATTERNS = {
    "is_live": r"\blive\b",
    "is_remaster": r"\bremaster(?:ed)?\b",
    "is_demo": r"\bdemo\b",
    "is_remix": r"\bremix\b|\breanimation\b",
    "is_instrumental": r"\binstrumental\b",
    "is_acoustic": r"\bacoustic\b",
}


def normalize_whitespace(value: str) -> str:
    """Collapse repeated whitespace and trim surrounding spaces."""

    return re.sub(r"\s+", " ", value).strip()


def normalize_unicode(value: str) -> str:
    """Normalize visually equivalent Unicode characters."""

    return unicodedata.normalize("NFKC", value)


def canonicalize_title(title: str) -> str:
    """Create a conservative canonical track title.

    Removes common version suffixes while preserving the underlying song name.

    Examples
    --------
    Numb (Live) -> Numb
    Crawling - Remastered -> Crawling
    Faint [Demo] -> Faint
    """

    value = normalize_unicode(str(title))
    value = normalize_whitespace(value)

    version_words = (
        r"live"
        r"|remaster(?:ed)?"
        r"|demo"
        r"|remix"
        r"|instrumental"
        r"|acoustic"
        r"|radio edit"
        r"|album version"
    )

    # Remove parenthesized or bracketed version descriptions.
    value = re.sub(
        rf"\s*[\(\[][^)\]]*\b(?:{version_words})\b[^)\]]*[\)\]]\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    )

    # Remove dash-separated version descriptions.
    value = re.sub(
        rf"\s+[-–—]\s+.*\b(?:{version_words})\b.*$",
        "",
        value,
        flags=re.IGNORECASE,
    )

    return normalize_whitespace(value)


def create_match_title(title: str) -> str:
    """Create a simplified title used for cross-source matching."""

    value = canonicalize_title(title).lower()
    value = value.replace("&", " and ")

    # Keep letters and numbers while removing punctuation.
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"_", " ", value)

    return normalize_whitespace(value)


def contains_pattern(
    series: pd.Series,
    pattern: str,
) -> pd.Series:
    """Return a boolean Series indicating whether titles match a pattern."""

    return series.str.contains(
        pattern,
        case=False,
        regex=True,
        na=False,
    )


def add_album_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add album order, era, vocalist, and standard track count."""

    metadata_df = pd.DataFrame.from_dict(
        ALBUM_METADATA,
        orient="index",
    )

    metadata_df.index.name = "album"
    metadata_df = metadata_df.reset_index()

    return df.merge(
        metadata_df,
        on="album",
        how="left",
        validate="many_to_one",
    )


def build_master_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw MusicBrainz tracks into a canonical catalogue."""

    required_columns = {
        "album",
        "album_first_release_date",
        "release_group_id",
        "release_id",
        "track_id",
        "track_title",
        "track_position",
        "recording_id",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Input data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    result = df.copy()

    # Basic string cleanup.
    string_columns = [
        "album",
        "release_title",
        "track_title",
        "recording_title",
        "artist_credit",
    ]

    for column in string_columns:
        if column in result.columns:
            result[column] = (
                result[column]
                .astype("string")
                .str.strip()
            )

    # Dates and release year.
    result["album_first_release_date"] = pd.to_datetime(
        result["album_first_release_date"],
        errors="coerce",
    )

    result["release_date"] = pd.to_datetime(
        result.get("release_date"),
        errors="coerce",
    )

    result["release_year"] = (
        result["album_first_release_date"]
        .dt.year
        .astype("Int64")
    )

    # Numeric fields.
    numeric_columns = [
        "disc_number",
        "track_position",
        "track_length_ms",
        "recording_length_ms",
    ]

    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            ).astype("Int64")

    # Prefer track duration; fall back to recording duration.
    result["duration_ms"] = result["track_length_ms"].fillna(
        result["recording_length_ms"]
    )

    result["duration_seconds"] = (
        result["duration_ms"] / 1000
    ).round(2)

    result["duration_minutes"] = (
        result["duration_ms"] / 60_000
    ).round(2)

    # Canonical and matching titles.
    result["canonical_title"] = result["track_title"].map(
        canonicalize_title
    )

    result["match_title"] = result["track_title"].map(
        create_match_title
    )

    # Version flags.
    for column, pattern in VERSION_PATTERNS.items():
        result[column] = contains_pattern(
            result["track_title"],
            pattern,
        )

    # Album-level metadata.
    result = add_album_metadata(result)

    result["has_chester"] = result["era"].eq("Chester Era")
    result["has_emily"] = result["era"].eq("Emily Era")

    # A bonus track is anything beyond the expected standard edition count.
    result["is_bonus_track"] = (
        result["track_position"]
        > result["standard_track_count"]
    ).fillna(False)

    # This first dataset contains one representative release per album.
    result["is_original_album_track"] = ~(
        result[
            [
                "is_live",
                "is_remaster",
                "is_demo",
                "is_remix",
                "is_instrumental",
                "is_acoustic",
                "is_bonus_track",
            ]
        ].any(axis=1)
    )

    # Stable project-specific identifier.
    result["master_track_id"] = (
        "lp-"
        + result["album_order"].astype("Int64").astype(str).str.zfill(2)
        + "-"
        + result["track_position"].astype("Int64").astype(str).str.zfill(2)
    )

    # Availability flags will be updated as new sources are merged.
    result["lyrics_available"] = False
    result["audio_features_available"] = False

    # Remove exact duplicate MusicBrainz track records.
    result = result.drop_duplicates(
        subset=["release_id", "track_id"],
        keep="first",
    )

    column_order = [
        "master_track_id",
        "album_order",
        "album",
        "album_first_release_date",
        "release_year",
        "era",
        "primary_vocalist",
        "has_chester",
        "has_emily",
        "disc_number",
        "track_number",
        "track_position",
        "track_title",
        "canonical_title",
        "match_title",
        "duration_ms",
        "duration_seconds",
        "duration_minutes",
        "is_original_album_track",
        "is_bonus_track",
        "is_live",
        "is_remaster",
        "is_demo",
        "is_remix",
        "is_instrumental",
        "is_acoustic",
        "lyrics_available",
        "audio_features_available",
        "release_group_id",
        "release_id",
        "track_id",
        "recording_id",
        "isrcs",
        "release_title",
        "release_date",
        "release_country",
        "release_status",
        "recording_title",
        "artist_credit",
        "standard_track_count",
    ]

    existing_columns = [
        column
        for column in column_order
        if column in result.columns
    ]

    result = result[existing_columns]

    result = result.sort_values(
        ["album_order", "disc_number", "track_position"],
        na_position="last",
    ).reset_index(drop=True)

    return result


def validate_master_tracks(df: pd.DataFrame) -> None:
    """Run essential quality checks before saving."""

    expected_albums = set(ALBUM_METADATA)
    actual_albums = set(df["album"].dropna())

    missing_albums = expected_albums - actual_albums
    unexpected_albums = actual_albums - expected_albums

    if missing_albums:
        raise ValueError(
            f"Expected albums are missing: {sorted(missing_albums)}"
        )

    if unexpected_albums:
        raise ValueError(
            f"Unexpected albums found: {sorted(unexpected_albums)}"
        )

    if df["master_track_id"].duplicated().any():
        duplicates = df.loc[
            df["master_track_id"].duplicated(keep=False),
            ["master_track_id", "album", "track_title"],
        ]

        raise ValueError(
            "Duplicate master_track_id values found:\n"
            f"{duplicates.to_string(index=False)}"
        )

    if df["recording_id"].isna().any():
        raise ValueError("Some tracks have no MusicBrainz recording ID.")

    if df["album_order"].isna().any():
        raise ValueError("Some tracks could not be mapped to an album order.")

    print("\nQuality checks passed.")


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact summary of the resulting catalogue."""

    album_summary = (
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
            tracks=("master_track_id", "count"),
            bonus_tracks=("is_bonus_track", "sum"),
            total_minutes=("duration_minutes", "sum"),
        )
        .reset_index()
        .sort_values("album_order")
    )

    album_summary["total_minutes"] = album_summary[
        "total_minutes"
    ].round(1)

    print("\nMaster catalogue summary:")
    print(album_summary.to_string(index=False))

    print(f"\nTotal albums: {df['album'].nunique()}")
    print(f"Total tracks: {len(df)}")
    print(
        "Unique canonical titles: "
        f"{df['canonical_title'].nunique()}"
    )


def main() -> None:
    """Build, validate, and save the master track catalogue."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {INPUT_PATH}\n"
            "Run download_musicbrainz.py first."
        )

    print(f"Reading: {INPUT_PATH}")

    raw_tracks = pd.read_parquet(INPUT_PATH)
    master_tracks = build_master_tracks(raw_tracks)

    validate_master_tracks(master_tracks)
    print_summary(master_tracks)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    master_tracks.to_parquet(
        PARQUET_OUTPUT_PATH,
        index=False,
    )

    master_tracks.to_csv(
        CSV_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print(f"\nSaved: {PARQUET_OUTPUT_PATH}")
    print(f"Saved: {CSV_OUTPUT_PATH}")
    print("Master track catalogue completed.")


if __name__ == "__main__":
    main()