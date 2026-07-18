"""Merge metadata, lyrics features, and AcousticBrainz features.

Inputs
------
data/processed/master_tracks.parquet
data/processed/lyrics_features.parquet
data/processed/audio_features_acousticbrainz.parquet

Outputs
-------
data/processed/master_dataset.parquet
data/processed/master_dataset.csv
data/interim/master_dataset/master_dataset_coverage.csv
data/interim/master_dataset/master_dataset_validation.csv

Notes
-----
- The master track catalogue is the left/base table.
- All 97 canonical tracks are preserved.
- From Zero remains present with missing AcousticBrainz features.
- Derived NLP features contain no copyrighted lyrics text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MASTER_TRACKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_tracks.parquet"
)

LYRICS_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lyrics_features.parquet"
)

AUDIO_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "audio_features_acousticbrainz.parquet"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "master_dataset"
)

PARQUET_OUTPUT_PATH = (
    PROCESSED_DIR
    / "master_dataset.parquet"
)

CSV_OUTPUT_PATH = (
    PROCESSED_DIR
    / "master_dataset.csv"
)

COVERAGE_OUTPUT_PATH = (
    INTERIM_DIR
    / "master_dataset_coverage.csv"
)

VALIDATION_OUTPUT_PATH = (
    INTERIM_DIR
    / "master_dataset_validation.csv"
)


BASE_KEY = "master_track_id"

DUPLICATE_METADATA_COLUMNS = {
    "album_order",
    "album",
    "release_year",
    "track_position",
    "track_title",
    "canonical_title",
    "era",
    "primary_vocalist",
    "is_instrumental",
    "lyrics_available",
    "audio_features_available",
    "recording_id",
}


def load_parquet(
    path: Path,
    label: str,
) -> pd.DataFrame:
    """Load and minimally validate one Parquet dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"{label} file not found: {path}"
        )

    df = pd.read_parquet(path)

    if BASE_KEY not in df.columns:
        raise ValueError(
            f"{label} is missing key column: {BASE_KEY}"
        )

    if df[BASE_KEY].duplicated().any():
        duplicates = (
            df.loc[
                df[BASE_KEY].duplicated(keep=False),
                BASE_KEY,
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            f"{label} contains duplicate track IDs: "
            f"{sorted(set(duplicates))}"
        )

    return df


def rename_feature_columns(
    df: pd.DataFrame,
    prefix: str,
    preserve: set[str],
) -> pd.DataFrame:
    """Prefix feature columns while preserving identifiers and metadata."""

    rename_map: dict[str, str] = {}

    for column in df.columns:
        if column == BASE_KEY or column in preserve:
            continue

        if column.startswith(prefix):
            continue

        rename_map[column] = f"{prefix}{column}"

    return df.rename(columns=rename_map)


def remove_duplicate_metadata(
    df: pd.DataFrame,
    base_columns: set[str],
) -> pd.DataFrame:
    """Remove feature-table metadata already supplied by the master table."""

    columns_to_drop = [
        column
        for column in df.columns
        if (
            column != BASE_KEY
            and column in base_columns
            and column in DUPLICATE_METADATA_COLUMNS
        )
    ]

    return df.drop(
        columns=columns_to_drop,
        errors="ignore",
    )


def compare_shared_metadata(
    master_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    dataset_name: str,
) -> list[dict[str, Any]]:
    """Compare shared metadata before duplicate columns are removed."""

    checks: list[dict[str, Any]] = []

    shared_columns = sorted(
        (
            set(master_df.columns)
            & set(feature_df.columns)
            & DUPLICATE_METADATA_COLUMNS
        )
        - {BASE_KEY}
    )

    merged = master_df.merge(
        feature_df[
            [BASE_KEY, *shared_columns]
        ],
        on=BASE_KEY,
        how="left",
        suffixes=("_master", "_feature"),
        validate="one_to_one",
    )

    for column in shared_columns:
        master_column = f"{column}_master"
        feature_column = f"{column}_feature"

        master_values = merged[master_column]
        feature_values = merged[feature_column]

        comparable = (
            master_values.notna()
            & feature_values.notna()
        )

        mismatch_count = int(
            (
                master_values[comparable].astype(str)
                != feature_values[comparable].astype(str)
            ).sum()
        )

        checks.append(
            {
                "dataset": dataset_name,
                "check_type": "shared_metadata_consistency",
                "column": column,
                "rows_compared": int(comparable.sum()),
                "mismatch_count": mismatch_count,
                "status": (
                    "passed"
                    if mismatch_count == 0
                    else "warning"
                ),
            }
        )

    return checks


def build_master_dataset(
    master_df: pd.DataFrame,
    lyrics_df: pd.DataFrame,
    audio_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge all track-level data sources into one canonical dataset."""

    validation_rows: list[dict[str, Any]] = []

    validation_rows.extend(
        compare_shared_metadata(
            master_df=master_df,
            feature_df=lyrics_df,
            dataset_name="lyrics_features",
        )
    )

    validation_rows.extend(
        compare_shared_metadata(
            master_df=master_df,
            feature_df=audio_df,
            dataset_name="audio_features_acousticbrainz",
        )
    )

    base_columns = set(master_df.columns)

    lyrics_merge = remove_duplicate_metadata(
        lyrics_df,
        base_columns,
    )

    audio_merge = remove_duplicate_metadata(
        audio_df,
        base_columns,
    )

    # Prefix remaining feature-table columns to make provenance explicit.
    lyrics_merge = rename_feature_columns(
        lyrics_merge,
        prefix="lyrics_",
        preserve=set(),
    )

    audio_merge = rename_feature_columns(
        audio_merge,
        prefix="audio_",
        preserve=set(),
    )

    merged = master_df.merge(
        lyrics_merge,
        on=BASE_KEY,
        how="left",
        validate="one_to_one",
    )

    merged = merged.merge(
        audio_merge,
        on=BASE_KEY,
        how="left",
        validate="one_to_one",
    )

    merged["has_lyrics_features"] = (
        merged["lyrics_word_count"].notna()
        if "lyrics_word_count" in merged.columns
        else False
    )

    if "audio_audio_features_available" in merged.columns:
        merged["has_audio_features"] = (
            merged["audio_audio_features_available"]
            .fillna(False)
            .astype(bool)
        )
    else:
        audio_feature_columns = [
            column
            for column in merged.columns
            if column.startswith("audio_ab_")
        ]

        merged["has_audio_features"] = (
            merged[audio_feature_columns]
            .notna()
            .any(axis=1)
            if audio_feature_columns
            else False
        )

    merged["analysis_coverage"] = np.select(
        [
            (
                merged["has_lyrics_features"]
                & merged["has_audio_features"]
            ),
            (
                merged["has_lyrics_features"]
                & ~merged["has_audio_features"]
            ),
            (
                ~merged["has_lyrics_features"]
                & merged["has_audio_features"]
            ),
        ],
        [
            "lyrics_and_audio",
            "lyrics_only",
            "audio_only",
        ],
        default="metadata_only",
    )

    merged = (
        merged
        .sort_values(
            [
                "album_order",
                "disc_number",
                "track_position",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    validation_df = pd.DataFrame(validation_rows)

    return merged, validation_df


def validate_master_dataset(
    df: pd.DataFrame,
    master_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Run critical structural and coverage checks."""

    checks: list[dict[str, Any]] = []

    checks.append(
        {
            "dataset": "master_dataset",
            "check_type": "row_count",
            "column": None,
            "expected": len(master_df),
            "actual": len(df),
            "status": (
                "passed"
                if len(df) == len(master_df)
                else "failed"
            ),
        }
    )

    checks.append(
        {
            "dataset": "master_dataset",
            "check_type": "unique_track_ids",
            "column": BASE_KEY,
            "expected": len(df),
            "actual": int(df[BASE_KEY].nunique()),
            "status": (
                "passed"
                if df[BASE_KEY].is_unique
                else "failed"
            ),
        }
    )

    missing_master_ids = (
        set(master_df[BASE_KEY])
        - set(df[BASE_KEY])
    )

    checks.append(
        {
            "dataset": "master_dataset",
            "check_type": "master_id_preservation",
            "column": BASE_KEY,
            "expected": 0,
            "actual": len(missing_master_ids),
            "status": (
                "passed"
                if not missing_master_ids
                else "failed"
            ),
        }
    )

    album_count = int(df["album"].nunique())

    checks.append(
        {
            "dataset": "master_dataset",
            "check_type": "album_count",
            "column": "album",
            "expected": 8,
            "actual": album_count,
            "status": (
                "passed"
                if album_count == 8
                else "failed"
            ),
        }
    )

    lyrics_coverage = int(
        df["has_lyrics_features"].sum()
    )

    checks.append(
        {
            "dataset": "master_dataset",
            "check_type": "lyrics_feature_coverage",
            "column": "has_lyrics_features",
            "expected": 97,
            "actual": lyrics_coverage,
            "status": (
                "passed"
                if lyrics_coverage == 97
                else "warning"
            ),
        }
    )

    audio_coverage = int(
        df["has_audio_features"].sum()
    )

    checks.append(
        {
            "dataset": "master_dataset",
            "check_type": "audio_feature_coverage",
            "column": "has_audio_features",
            "expected": 86,
            "actual": audio_coverage,
            "status": (
                "passed"
                if audio_coverage == 86
                else "warning"
            ),
        }
    )

    from_zero_audio = int(
        df.loc[
            df["album"].eq("From Zero"),
            "has_audio_features",
        ].sum()
    )

    checks.append(
        {
            "dataset": "master_dataset",
            "check_type": "from_zero_audio_exclusion",
            "column": "has_audio_features",
            "expected": 0,
            "actual": from_zero_audio,
            "status": (
                "passed"
                if from_zero_audio == 0
                else "warning"
            ),
        }
    )

    failed_checks = [
        check
        for check in checks
        if check["status"] == "failed"
    ]

    if failed_checks:
        raise ValueError(
            "Critical master dataset validation failed:\n"
            + pd.DataFrame(failed_checks).to_string(index=False)
        )

    return checks


def build_coverage_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build album-level source coverage metrics."""

    coverage = (
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
            tracks=(BASE_KEY, "count"),
            lyrics_features=("has_lyrics_features", "sum"),
            audio_features=("has_audio_features", "sum"),
            full_coverage=(
                "analysis_coverage",
                lambda values: int(
                    (values == "lyrics_and_audio").sum()
                ),
            ),
        )
        .reset_index()
        .sort_values("album_order")
    )

    coverage["lyrics_coverage_rate"] = (
        coverage["lyrics_features"]
        / coverage["tracks"]
    )

    coverage["audio_coverage_rate"] = (
        coverage["audio_features"]
        / coverage["tracks"]
    )

    coverage["full_coverage_rate"] = (
        coverage["full_coverage"]
        / coverage["tracks"]
    )

    return coverage


def print_summary(
    df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> None:
    """Print final dataset summary."""

    print("\nMaster dataset summary:")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Albums: {df['album'].nunique()}")
    print(
        "Tracks with lyrics features: "
        f"{int(df['has_lyrics_features'].sum())}"
    )
    print(
        "Tracks with audio features: "
        f"{int(df['has_audio_features'].sum())}"
    )

    print("\nAnalysis coverage:")
    print(
        df["analysis_coverage"]
        .value_counts(dropna=False)
        .to_string()
    )

    display_df = coverage_df.copy()

    rate_columns = [
        "lyrics_coverage_rate",
        "audio_coverage_rate",
        "full_coverage_rate",
    ]

    display_df[rate_columns] = (
        display_df[rate_columns]
        .round(3)
    )

    print("\nAlbum-level coverage:")
    print(display_df.to_string(index=False))


def main() -> None:
    """Build, validate, and save the final analytical dataset."""

    master_df = load_parquet(
        MASTER_TRACKS_PATH,
        "Master tracks",
    )

    lyrics_df = load_parquet(
        LYRICS_FEATURES_PATH,
        "Lyrics features",
    )

    audio_df = load_parquet(
        AUDIO_FEATURES_PATH,
        "AcousticBrainz features",
    )

    merged_df, metadata_validation_df = (
        build_master_dataset(
            master_df=master_df,
            lyrics_df=lyrics_df,
            audio_df=audio_df,
        )
    )

    structural_checks = validate_master_dataset(
        df=merged_df,
        master_df=master_df,
    )

    structural_validation_df = pd.DataFrame(
        structural_checks
    )

    validation_df = pd.concat(
        [
            metadata_validation_df,
            structural_validation_df,
        ],
        ignore_index=True,
        sort=False,
    )

    coverage_df = build_coverage_table(
        merged_df
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged_df.to_parquet(
        PARQUET_OUTPUT_PATH,
        index=False,
    )

    merged_df.to_csv(
        CSV_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    coverage_df.to_csv(
        COVERAGE_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    validation_df.to_csv(
        VALIDATION_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print_summary(
        df=merged_df,
        coverage_df=coverage_df,
    )

    print(f"\nSaved: {PARQUET_OUTPUT_PATH}")
    print(f"Saved: {CSV_OUTPUT_PATH}")
    print(f"Saved: {COVERAGE_OUTPUT_PATH}")
    print(f"Saved: {VALIDATION_OUTPUT_PATH}")
    print("Master analytical dataset completed.")


if __name__ == "__main__":
    main()
