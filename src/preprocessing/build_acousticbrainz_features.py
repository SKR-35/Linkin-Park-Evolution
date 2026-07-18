"""Flatten AcousticBrainz JSON files into a track-level feature table.

Inputs
------
data/processed/master_tracks.parquet
data/raw/acousticbrainz/lowlevel/<recording_id>.json
data/raw/acousticbrainz/highlevel/<recording_id>.json

Outputs
-------
data/processed/audio_features_acousticbrainz.parquet
data/processed/audio_features_acousticbrainz.csv
data/interim/audio_features/acousticbrainz_feature_coverage.csv

Notes
-----
- AcousticBrainz stopped collecting new analyses in 2022.
- From Zero (2024) is therefore expected to remain unavailable.
- The extractor is defensive: missing nested fields become nulls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MASTER_TRACKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_tracks.parquet"
)

LOWLEVEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "acousticbrainz"
    / "lowlevel"
)

HIGHLEVEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "acousticbrainz"
    / "highlevel"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "audio_features"
)

PARQUET_OUTPUT_PATH = (
    PROCESSED_DIR
    / "audio_features_acousticbrainz.parquet"
)

CSV_OUTPUT_PATH = (
    PROCESSED_DIR
    / "audio_features_acousticbrainz.csv"
)

COVERAGE_OUTPUT_PATH = (
    INTERIM_DIR
    / "acousticbrainz_feature_coverage.csv"
)


def load_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")

    return payload


def get_nested(
    payload: dict[str, Any] | None,
    path: str,
    default: Any = None,
) -> Any:
    """Safely retrieve a dotted nested field."""

    if payload is None:
        return default

    current: Any = payload

    for key in path.split("."):
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def get_probability(
    payload: dict[str, Any] | None,
    classifier_name: str,
    positive_label: str,
) -> float | None:
    """Return a positive-class probability from a high-level classifier."""

    probabilities = get_nested(
        payload,
        f"highlevel.{classifier_name}.all",
    )

    if not isinstance(probabilities, dict):
        return None

    value = probabilities.get(positive_label)

    if value is None:
        return None

    return float(value)


def get_classifier_value(
    payload: dict[str, Any] | None,
    classifier_name: str,
) -> Any:
    """Return the winning label for a high-level classifier."""

    return get_nested(
        payload,
        f"highlevel.{classifier_name}.value",
    )


def extract_lowlevel_features(
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Extract selected reproducible low-level and tonal descriptors."""

    return {
        "ab_duration_seconds": get_nested(
            payload,
            "metadata.audio_properties.length",
        ),
        "ab_sample_rate": get_nested(
            payload,
            "metadata.audio_properties.sample_rate",
        ),
        "ab_bit_rate": get_nested(
            payload,
            "metadata.audio_properties.bit_rate",
        ),
        "ab_channels": get_nested(
            payload,
            "metadata.audio_properties.channels",
        ),
        "ab_codec": get_nested(
            payload,
            "metadata.audio_properties.codec",
        ),
        "ab_average_loudness": get_nested(
            payload,
            "lowlevel.average_loudness",
        ),
        "ab_dynamic_complexity": get_nested(
            payload,
            "lowlevel.dynamic_complexity",
        ),
        "ab_loudness_ebu128_integrated": get_nested(
            payload,
            "lowlevel.loudness_ebu128.integrated",
        ),
        "ab_loudness_ebu128_loudness_range": get_nested(
            payload,
            "lowlevel.loudness_ebu128.loudness_range",
        ),
        "ab_spectral_centroid_mean": get_nested(
            payload,
            "lowlevel.spectral_centroid.mean",
        ),
        "ab_spectral_centroid_stdev": get_nested(
            payload,
            "lowlevel.spectral_centroid.stdev",
        ),
        "ab_spectral_entropy_mean": get_nested(
            payload,
            "lowlevel.spectral_entropy.mean",
        ),
        "ab_spectral_flux_mean": get_nested(
            payload,
            "lowlevel.spectral_flux.mean",
        ),
        "ab_spectral_rolloff_mean": get_nested(
            payload,
            "lowlevel.spectral_rolloff.mean",
        ),
        "ab_zero_crossing_rate_mean": get_nested(
            payload,
            "lowlevel.zerocrossingrate.mean",
        ),
        "ab_bpm": get_nested(
            payload,
            "rhythm.bpm",
        ),
        "ab_bpm_histogram_first_peak_bpm": get_nested(
            payload,
            "rhythm.bpm_histogram_first_peak_bpm.mean",
        ),
        "ab_beats_count": get_nested(
            payload,
            "rhythm.beats_count",
        ),
        "ab_danceability": get_nested(
            payload,
            "rhythm.danceability",
        ),
        "ab_onset_rate": get_nested(
            payload,
            "rhythm.onset_rate",
        ),
        "ab_key_edma": get_nested(
            payload,
            "tonal.key_edma.key",
        ),
        "ab_scale_edma": get_nested(
            payload,
            "tonal.key_edma.scale",
        ),
        "ab_key_strength_edma": get_nested(
            payload,
            "tonal.key_edma.strength",
        ),
        "ab_key_krumhansl": get_nested(
            payload,
            "tonal.key_krumhansl.key",
        ),
        "ab_scale_krumhansl": get_nested(
            payload,
            "tonal.key_krumhansl.scale",
        ),
        "ab_key_strength_krumhansl": get_nested(
            payload,
            "tonal.key_krumhansl.strength",
        ),
        "ab_chords_key": get_nested(
            payload,
            "tonal.chords_key",
        ),
        "ab_chords_scale": get_nested(
            payload,
            "tonal.chords_scale",
        ),
        "ab_chords_changes_rate": get_nested(
            payload,
            "tonal.chords_changes_rate",
        ),
        "ab_tuning_frequency": get_nested(
            payload,
            "tonal.tuning_frequency",
        ),
    }


def extract_highlevel_features(
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Extract selected mood, voice, and style classifiers."""

    return {
        "ab_highlevel_danceability": get_classifier_value(
            payload,
            "danceability",
        ),
        "ab_highlevel_danceable_probability": get_probability(
            payload,
            "danceability",
            "danceable",
        ),
        "ab_gender": get_classifier_value(
            payload,
            "gender",
        ),
        "ab_mood_acoustic": get_classifier_value(
            payload,
            "mood_acoustic",
        ),
        "ab_mood_acoustic_probability": get_probability(
            payload,
            "mood_acoustic",
            "acoustic",
        ),
        "ab_mood_aggressive": get_classifier_value(
            payload,
            "mood_aggressive",
        ),
        "ab_mood_aggressive_probability": get_probability(
            payload,
            "mood_aggressive",
            "aggressive",
        ),
        "ab_mood_electronic": get_classifier_value(
            payload,
            "mood_electronic",
        ),
        "ab_mood_electronic_probability": get_probability(
            payload,
            "mood_electronic",
            "electronic",
        ),
        "ab_mood_happy": get_classifier_value(
            payload,
            "mood_happy",
        ),
        "ab_mood_happy_probability": get_probability(
            payload,
            "mood_happy",
            "happy",
        ),
        "ab_mood_party": get_classifier_value(
            payload,
            "mood_party",
        ),
        "ab_mood_party_probability": get_probability(
            payload,
            "mood_party",
            "party",
        ),
        "ab_mood_relaxed": get_classifier_value(
            payload,
            "mood_relaxed",
        ),
        "ab_mood_relaxed_probability": get_probability(
            payload,
            "mood_relaxed",
            "relaxed",
        ),
        "ab_mood_sad": get_classifier_value(
            payload,
            "mood_sad",
        ),
        "ab_mood_sad_probability": get_probability(
            payload,
            "mood_sad",
            "sad",
        ),
        "ab_timbre": get_classifier_value(
            payload,
            "timbre",
        ),
        "ab_tonal_atonal": get_classifier_value(
            payload,
            "tonal_atonal",
        ),
        "ab_voice_instrumental": get_classifier_value(
            payload,
            "voice_instrumental",
        ),
        "ab_voice_probability": get_probability(
            payload,
            "voice_instrumental",
            "voice",
        ),
        "ab_instrumental_probability": get_probability(
            payload,
            "voice_instrumental",
            "instrumental",
        ),
        "ab_genre_dortmund": get_classifier_value(
            payload,
            "genre_dortmund",
        ),
        "ab_genre_electronic": get_classifier_value(
            payload,
            "genre_electronic",
        ),
        "ab_genre_rosamerica": get_classifier_value(
            payload,
            "genre_rosamerica",
        ),
    }


def build_feature_table(
    master_tracks: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per canonical track."""

    records: list[dict[str, Any]] = []

    for _, track in master_tracks.iterrows():
        recording_id = str(track["recording_id"])

        lowlevel_path = LOWLEVEL_DIR / f"{recording_id}.json"
        highlevel_path = HIGHLEVEL_DIR / f"{recording_id}.json"

        lowlevel_payload = (
            load_json(lowlevel_path)
            if lowlevel_path.exists()
            else None
        )

        highlevel_payload = (
            load_json(highlevel_path)
            if highlevel_path.exists()
            else None
        )

        record: dict[str, Any] = {
            "master_track_id": track["master_track_id"],
            "recording_id": recording_id,
            "album_order": track["album_order"],
            "album": track["album"],
            "release_year": track["release_year"],
            "track_position": track["track_position"],
            "track_title": track["track_title"],
            "canonical_title": track["canonical_title"],
            "era": track["era"],
            "lowlevel_available": lowlevel_payload is not None,
            "highlevel_available": highlevel_payload is not None,
            "audio_features_available": (
                lowlevel_payload is not None
                or highlevel_payload is not None
            ),
            **extract_lowlevel_features(lowlevel_payload),
            **extract_highlevel_features(highlevel_payload),
        }

        records.append(record)

    return (
        pd.DataFrame(records)
        .sort_values(
            ["album_order", "track_position"],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def print_summary(df: pd.DataFrame) -> None:
    """Print feature coverage by album."""

    summary = (
        df.groupby(
            [
                "album_order",
                "album",
                "release_year",
            ],
            dropna=False,
        )
        .agg(
            tracks=("master_track_id", "count"),
            audio_available=("audio_features_available", "sum"),
            lowlevel_available=("lowlevel_available", "sum"),
            highlevel_available=("highlevel_available", "sum"),
        )
        .reset_index()
        .sort_values("album_order")
    )

    print("\nAcousticBrainz feature coverage:")
    print(summary.to_string(index=False))

    print(f"\nTotal tracks: {len(df)}")
    print(
        "Tracks with audio features: "
        f"{int(df['audio_features_available'].sum())}"
    )
    print(
        "Tracks without audio features: "
        f"{int((~df['audio_features_available']).sum())}"
    )


def main() -> None:
    """Build and save the AcousticBrainz feature table."""

    if not MASTER_TRACKS_PATH.exists():
        raise FileNotFoundError(
            f"Master track catalogue not found: {MASTER_TRACKS_PATH}"
        )

    master_tracks = pd.read_parquet(MASTER_TRACKS_PATH)

    required_columns = {
        "master_track_id",
        "recording_id",
        "album_order",
        "album",
        "release_year",
        "track_position",
        "track_title",
        "canonical_title",
        "era",
    }

    missing_columns = required_columns - set(master_tracks.columns)

    if missing_columns:
        raise ValueError(
            "Master catalogue is missing columns: "
            f"{sorted(missing_columns)}"
        )

    feature_df = build_feature_table(master_tracks)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    feature_df.to_parquet(
        PARQUET_OUTPUT_PATH,
        index=False,
    )

    feature_df.to_csv(
        CSV_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    coverage_df = feature_df[
        [
            "master_track_id",
            "album",
            "track_title",
            "lowlevel_available",
            "highlevel_available",
            "audio_features_available",
        ]
    ].copy()

    coverage_df.to_csv(
        COVERAGE_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print_summary(feature_df)

    print(f"\nSaved: {PARQUET_OUTPUT_PATH}")
    print(f"Saved: {CSV_OUTPUT_PATH}")
    print(f"Saved: {COVERAGE_OUTPUT_PATH}")
    print("AcousticBrainz feature table completed.")


if __name__ == "__main__":
    main()
