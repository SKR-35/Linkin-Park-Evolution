"""Clean locally stored lyrics and build a canonical NLP-ready table.

Inputs
------
data/raw/lyrics/lyrics.json
data/processed/master_tracks.parquet

Outputs
-------
data/processed/lyrics_clean.parquet
data/processed/lyrics_clean.csv
data/interim/lyrics/lyrics_cleaning_report.csv

Notes
-----
- Raw lyrics remain local and excluded from Git.
- Cleaned lyrics are also copyrighted text. Keep the processed files local.
- The script preserves both raw and cleaned text in memory, but only writes
  the cleaned text and derived metadata to the processed outputs.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LYRICS_JSON_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "lyrics"
    / "lyrics.json"
)

MASTER_TRACKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_tracks.parquet"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "lyrics"

PARQUET_OUTPUT_PATH = PROCESSED_DIR / "lyrics_clean.parquet"
CSV_OUTPUT_PATH = PROCESSED_DIR / "lyrics_clean.csv"
REPORT_OUTPUT_PATH = INTERIM_DIR / "lyrics_cleaning_report.csv"


SECTION_HEADER_PATTERN = re.compile(
    r"^\s*[\[\(\{].{0,80}[\]\)\}]\s*$"
)

COMMON_SECTION_WORDS = re.compile(
    r"\b("
    r"intro|verse|pre[- ]?chorus|chorus|hook|refrain|bridge|"
    r"outro|interlude|instrumental|breakdown|spoken|rap|"
    r"post[- ]?chorus|repeat|solo|sample"
    r")\b",
    flags=re.IGNORECASE,
)

CREDIT_LINE_PATTERNS = [
    re.compile(r"^\s*lyrics?\s*[:\-]", re.IGNORECASE),
    re.compile(r"^\s*written by\s*[:\-]", re.IGNORECASE),
    re.compile(r"^\s*produced by\s*[:\-]", re.IGNORECASE),
    re.compile(r"^\s*translation\s*[:\-]", re.IGNORECASE),
    re.compile(r"^\s*contributors?\s*[:\-]", re.IGNORECASE),
]

TIMESTAMP_PATTERN = re.compile(
    r"^\s*\[\d{1,2}:\d{2}(?:\.\d{1,3})?\]\s*"
)

MULTISPACE_PATTERN = re.compile(r"[ \t]+")
MULTIBLANK_PATTERN = re.compile(r"\n{3,}")


def load_lyrics_records() -> list[dict[str, Any]]:
    """Load raw lyric records from the local checkpoint file."""

    if not LYRICS_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Lyrics file not found: {LYRICS_JSON_PATH}\n"
            "Run download_lyrics.py first."
        )

    with LYRICS_JSON_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError(
            f"Expected a list in {LYRICS_JSON_PATH}"
        )

    return payload


def normalize_unicode(text: str) -> str:
    """Normalize Unicode and standardize common punctuation."""

    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def is_section_header(line: str) -> bool:
    """Return True when a line appears to be a structural lyrics label."""

    stripped = line.strip()

    if not stripped:
        return False

    if SECTION_HEADER_PATTERN.match(stripped):
        inner = stripped[1:-1].strip()
        return bool(COMMON_SECTION_WORDS.search(inner))

    return False


def is_credit_line(line: str) -> bool:
    """Return True for metadata or credit lines embedded in lyrics."""

    return any(pattern.search(line) for pattern in CREDIT_LINE_PATTERNS)


def clean_lyrics_text(text: str | None) -> str:
    """Clean one lyrics string conservatively.

    The function removes:
    - synced timestamp prefixes,
    - section headers such as [Chorus] or (Verse 2),
    - obvious credits and annotations,
    - repeated spaces and excessive blank lines.

    It intentionally preserves punctuation, contractions, line order,
    and repeated lyric lines because repetition is analytically useful.
    """

    if text is None:
        return ""

    text = normalize_unicode(str(text))
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines: list[str] = []

    for raw_line in text.split("\n"):
        line = TIMESTAMP_PATTERN.sub("", raw_line)
        line = MULTISPACE_PATTERN.sub(" ", line).strip()

        if not line:
            cleaned_lines.append("")
            continue

        if is_section_header(line):
            continue

        if is_credit_line(line):
            continue

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = MULTIBLANK_PATTERN.sub("\n\n", cleaned_text)
    cleaned_text = cleaned_text.strip()

    return cleaned_text


def count_nonempty_lines(text: str) -> int:
    """Count non-empty lyric lines."""

    return sum(
        1
        for line in text.splitlines()
        if line.strip()
    )


def count_words(text: str) -> int:
    """Count word-like tokens while preserving contractions."""

    return len(
        re.findall(
            r"\b[\w']+\b",
            text,
            flags=re.UNICODE,
        )
    )


def build_clean_table(
    lyric_records: list[dict[str, Any]],
    master_tracks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build cleaned lyrics and a cleaning report."""

    master_columns = [
        "master_track_id",
        "album_order",
        "album",
        "release_year",
        "track_position",
        "track_title",
        "canonical_title",
        "era",
        "primary_vocalist",
    ]

    master_subset = master_tracks[master_columns].copy()

    lyric_df = pd.DataFrame(lyric_records)

    if lyric_df.empty:
        raise ValueError("Lyrics JSON contains no records.")

    required_columns = {
        "master_track_id",
        "match_status",
        "lyrics_available",
        "plain_lyrics",
        "synced_lyrics",
        "instrumental",
    }

    missing_columns = required_columns - set(lyric_df.columns)

    if missing_columns:
        raise ValueError(
            "Lyrics data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    lyric_df = lyric_df.drop_duplicates(
        subset=["master_track_id"],
        keep="last",
    )

    merged = master_subset.merge(
        lyric_df,
        on="master_track_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_lyrics"),
    )

    rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []

    for _, row in merged.iterrows():
        plain_lyrics = row.get("plain_lyrics")
        synced_lyrics = row.get("synced_lyrics")

        source_text = (
            plain_lyrics
            if isinstance(plain_lyrics, str) and plain_lyrics.strip()
            else synced_lyrics
        )

        cleaned_text = clean_lyrics_text(source_text)

        raw_text = "" if source_text is None else str(source_text)

        raw_char_count = len(raw_text)
        clean_char_count = len(cleaned_text)

        raw_line_count = count_nonempty_lines(raw_text)
        clean_line_count = count_nonempty_lines(cleaned_text)

        raw_word_count = count_words(raw_text)
        clean_word_count = count_words(cleaned_text)

        is_instrumental = bool(row.get("instrumental")) or (
            row["canonical_title"] == "Drawbar"
        )

        lyrics_available = bool(cleaned_text.strip())

        output_row = {
            "master_track_id": row["master_track_id"],
            "album_order": row["album_order"],
            "album": row["album"],
            "release_year": row["release_year"],
            "track_position": row["track_position"],
            "track_title": row["track_title"],
            "canonical_title": row["canonical_title"],
            "era": row["era"],
            "primary_vocalist": row["primary_vocalist"],
            "match_status": row.get("match_status"),
            "match_score": row.get("match_score"),
            "lyrics_source": row.get("source"),
            "lrclib_id": row.get("lrclib_id"),
            "is_instrumental": is_instrumental,
            "lyrics_available": lyrics_available,
            "lyrics_clean": cleaned_text,
            "line_count": clean_line_count,
            "word_count": clean_word_count,
            "character_count": clean_char_count,
        }

        rows.append(output_row)

        report_rows.append(
            {
                "master_track_id": row["master_track_id"],
                "album": row["album"],
                "track_title": row["track_title"],
                "is_instrumental": is_instrumental,
                "match_status": row.get("match_status"),
                "raw_char_count": raw_char_count,
                "clean_char_count": clean_char_count,
                "chars_removed": raw_char_count - clean_char_count,
                "raw_line_count": raw_line_count,
                "clean_line_count": clean_line_count,
                "lines_removed": raw_line_count - clean_line_count,
                "raw_word_count": raw_word_count,
                "clean_word_count": clean_word_count,
                "words_removed": raw_word_count - clean_word_count,
                "lyrics_available_after_cleaning": lyrics_available,
            }
        )

    clean_df = (
        pd.DataFrame(rows)
        .sort_values(
            ["album_order", "track_position"],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    report_df = (
        pd.DataFrame(report_rows)
        .sort_values(
            ["album", "track_title"]
        )
        .reset_index(drop=True)
    )

    return clean_df, report_df


def validate_clean_table(df: pd.DataFrame) -> None:
    """Run essential quality checks."""

    if df["master_track_id"].duplicated().any():
        raise ValueError(
            "Duplicate master_track_id values found in clean lyrics."
        )

    expected_rows = 97

    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} tracks, found {len(df)}."
        )

    non_instrumental_missing = df[
        (~df["is_instrumental"])
        & (~df["lyrics_available"])
    ]

    if not non_instrumental_missing.empty:
        raise ValueError(
            "Some non-instrumental tracks have no cleaned lyrics:\n"
            + non_instrumental_missing[
                ["album", "track_title", "match_status"]
            ].to_string(index=False)
        )


def print_summary(
    clean_df: pd.DataFrame,
    report_df: pd.DataFrame,
) -> None:
    """Print a compact cleaning summary."""

    print("\nLyrics cleaning summary:")
    print(f"Total tracks: {len(clean_df)}")
    print(
        "Tracks with cleaned lyrics: "
        f"{int(clean_df['lyrics_available'].sum())}"
    )
    print(
        "Instrumental tracks: "
        f"{int(clean_df['is_instrumental'].sum())}"
    )
    print(
        "Total cleaned words: "
        f"{int(clean_df['word_count'].sum())}"
    )
    print(
        "Total cleaned lines: "
        f"{int(clean_df['line_count'].sum())}"
    )
    print(
        "Characters removed: "
        f"{int(report_df['chars_removed'].sum())}"
    )
    print(
        "Lines removed: "
        f"{int(report_df['lines_removed'].sum())}"
    )


def main() -> None:
    """Build, validate, and save the cleaned lyrics table."""

    if not MASTER_TRACKS_PATH.exists():
        raise FileNotFoundError(
            f"Master track catalogue not found: {MASTER_TRACKS_PATH}"
        )

    lyric_records = load_lyrics_records()
    master_tracks = pd.read_parquet(MASTER_TRACKS_PATH)

    clean_df, report_df = build_clean_table(
        lyric_records,
        master_tracks,
    )

    validate_clean_table(clean_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    clean_df.to_parquet(
        PARQUET_OUTPUT_PATH,
        index=False,
    )

    clean_df.to_csv(
        CSV_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    report_df.to_csv(
        REPORT_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print_summary(clean_df, report_df)

    print(f"\nSaved: {PARQUET_OUTPUT_PATH}")
    print(f"Saved: {CSV_OUTPUT_PATH}")
    print(f"Saved: {REPORT_OUTPUT_PATH}")
    print("Lyrics cleaning completed.")


if __name__ == "__main__":
    main()
