"""Extract reproducible audio features from local audio files.

Inputs
------
data/raw/audio/**/*.mp3
data/raw/audio/**/*.flac
data/raw/audio/**/*.wav
data/raw/audio/**/*.m4a
data/processed/master_tracks.parquet

Outputs
-------
data/processed/audio_features_librosa.parquet
data/processed/audio_features_librosa.csv
data/interim/audio_features/librosa_matches.csv
data/interim/audio_features/librosa_unmatched.csv
data/interim/audio_features/librosa_errors.csv

Notes
-----
- Audio files must remain local and excluded from Git.
- The script supports MP3, FLAC, WAV, OGG, M4A, AAC, and AIFF.
- Features are extracted with librosa using mono audio at 22,050 Hz.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIO_ROOT = PROJECT_ROOT / "data" / "raw" / "audio"

MASTER_TRACKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_tracks.parquet"
)

PROCESSED_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

INTERIM_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "audio_features"
)

FEATURES_PARQUET_PATH = (
    PROCESSED_OUTPUT_DIR
    / "audio_features_librosa.parquet"
)

FEATURES_CSV_PATH = (
    PROCESSED_OUTPUT_DIR
    / "audio_features_librosa.csv"
)

MATCHES_CSV_PATH = (
    INTERIM_OUTPUT_DIR
    / "librosa_matches.csv"
)

UNMATCHED_CSV_PATH = (
    INTERIM_OUTPUT_DIR
    / "librosa_unmatched.csv"
)

ERRORS_CSV_PATH = (
    INTERIM_OUTPUT_DIR
    / "librosa_errors.csv"
)

CHECKPOINT_JSON_PATH = (
    INTERIM_OUTPUT_DIR
    / "librosa_checkpoint.json"
)

SUPPORTED_EXTENSIONS = {
    ".mp3",
    ".flac",
    ".wav",
    ".ogg",
    ".m4a",
    ".aac",
    ".aiff",
    ".aif",
}

TARGET_SAMPLE_RATE = 22_050
HOP_LENGTH = 512
N_MFCC = 20
MATCH_THRESHOLD = 80.0


def normalize_text(value: Any) -> str:
    """Normalize text for filename-to-track matching."""

    if value is None:
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.lower().strip()
    text = text.replace("&", " and ")

    # Remove common track-number prefixes.
    text = re.sub(r"^\s*\d{1,3}[\s._-]+", "", text)

    # Remove common version/quality suffixes.
    text = re.sub(
        r"\b("
        r"official|audio|video|lyrics?|lyric video|"
        r"hd|hq|remaster(?:ed)?|deluxe|explicit"
        r")\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"_", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def discover_audio_files() -> list[Path]:
    """Discover supported local audio files recursively."""

    if not AUDIO_ROOT.exists():
        return []

    return sorted(
        path
        for path in AUDIO_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_master_tracks() -> pd.DataFrame:
    """Load the canonical track catalogue."""

    if not MASTER_TRACKS_PATH.exists():
        raise FileNotFoundError(
            f"Master track catalogue not found: {MASTER_TRACKS_PATH}"
        )

    df = pd.read_parquet(MASTER_TRACKS_PATH)

    required_columns = {
        "master_track_id",
        "album",
        "track_title",
        "canonical_title",
        "duration_seconds",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Master track catalogue is missing columns: "
            f"{sorted(missing)}"
        )

    return df


def duration_score(
    expected_seconds: float | int | None,
    actual_seconds: float | int | None,
) -> float:
    """Score duration similarity from zero to one hundred."""

    if expected_seconds is None or actual_seconds is None:
        return 50.0

    if pd.isna(expected_seconds) or pd.isna(actual_seconds):
        return 50.0

    difference = abs(float(expected_seconds) - float(actual_seconds))

    if difference <= 2:
        return 100.0
    if difference <= 5:
        return 90.0
    if difference <= 10:
        return 75.0
    if difference <= 20:
        return 50.0
    if difference <= 40:
        return 25.0

    return 0.0


def match_audio_file(
    audio_path: Path,
    master_tracks: pd.DataFrame,
    measured_duration_seconds: float | None,
) -> dict[str, Any]:
    """Match one audio filename to the canonical track catalogue."""

    file_title = normalize_text(audio_path.stem)
    folder_album = normalize_text(audio_path.parent.name)

    scored_rows: list[dict[str, Any]] = []

    for _, track in master_tracks.iterrows():
        title_similarity = float(
            ratio(
                file_title,
                normalize_text(track["canonical_title"]),
            )
        )

        album_similarity = float(
            ratio(
                folder_album,
                normalize_text(track["album"]),
            )
        )

        track_duration_score = duration_score(
            track.get("duration_seconds"),
            measured_duration_seconds,
        )

        total_score = (
            0.70 * title_similarity
            + 0.15 * album_similarity
            + 0.15 * track_duration_score
        )

        scored_rows.append(
            {
                "master_track_id": track["master_track_id"],
                "album": track["album"],
                "track_title": track["track_title"],
                "canonical_title": track["canonical_title"],
                "title_score": round(title_similarity, 2),
                "album_score": round(album_similarity, 2),
                "duration_score": round(track_duration_score, 2),
                "match_score": round(total_score, 2),
            }
        )

    scored_rows.sort(
        key=lambda item: item["match_score"],
        reverse=True,
    )

    best = scored_rows[0]

    best["match_status"] = (
        "matched"
        if best["match_score"] >= MATCH_THRESHOLD
        else "review"
    )

    return best


def safe_mean(values: np.ndarray) -> float:
    """Return a finite mean."""

    value = float(np.nanmean(values))
    return value if math.isfinite(value) else float("nan")


def safe_std(values: np.ndarray) -> float:
    """Return a finite standard deviation."""

    value = float(np.nanstd(values))
    return value if math.isfinite(value) else float("nan")


def estimate_key_and_mode(
    chroma_mean: np.ndarray,
) -> tuple[str, str, float]:
    """Estimate musical key and mode using Krumhansl-style profiles."""

    major_profile = np.array(
        [
            6.35,
            2.23,
            3.48,
            2.33,
            4.38,
            4.09,
            2.52,
            5.19,
            2.39,
            3.66,
            2.29,
            2.88,
        ]
    )

    minor_profile = np.array(
        [
            6.33,
            2.68,
            3.52,
            5.38,
            2.60,
            3.53,
            2.54,
            4.75,
            3.98,
            2.69,
            3.34,
            3.17,
        ]
    )

    pitch_names = [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    ]

    scores: list[tuple[float, str, str]] = []

    for tonic in range(12):
        major_score = float(
            np.corrcoef(
                chroma_mean,
                np.roll(major_profile, tonic),
            )[0, 1]
        )

        minor_score = float(
            np.corrcoef(
                chroma_mean,
                np.roll(minor_profile, tonic),
            )[0, 1]
        )

        scores.append(
            (major_score, pitch_names[tonic], "major")
        )

        scores.append(
            (minor_score, pitch_names[tonic], "minor")
        )

    scores.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    confidence, key_name, mode = scores[0]

    return key_name, mode, round(confidence, 4)


def extract_features(audio_path: Path) -> dict[str, Any]:
    """Extract reproducible track-level audio features."""

    y, sample_rate = librosa.load(
        audio_path,
        sr=TARGET_SAMPLE_RATE,
        mono=True,
    )

    if y.size == 0:
        raise ValueError("Decoded audio signal is empty.")

    duration_seconds = float(
        librosa.get_duration(
            y=y,
            sr=sample_rate,
        )
    )

    tempo_result, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
    )

    tempo_bpm = float(np.asarray(tempo_result).squeeze())

    rms = librosa.feature.rms(
        y=y,
        hop_length=HOP_LENGTH,
    )[0]

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y,
        hop_length=HOP_LENGTH,
    )[0]

    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
    )[0]

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
    )[0]

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
        roll_percent=0.85,
    )[0]

    spectral_flatness = librosa.feature.spectral_flatness(
        y=y,
        hop_length=HOP_LENGTH,
    )[0]

    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
    )

    chroma_mean = np.nanmean(
        chroma,
        axis=1,
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH,
    )

    onset_strength = librosa.onset.onset_strength(
        y=y,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
    )

    key_name, mode, key_confidence = estimate_key_and_mode(
        chroma_mean
    )

    result: dict[str, Any] = {
        "audio_file": str(audio_path.relative_to(PROJECT_ROOT)),
        "audio_filename": audio_path.name,
        "audio_extension": audio_path.suffix.lower(),
        "sample_rate": sample_rate,
        "duration_seconds_measured": round(duration_seconds, 3),
        "tempo_bpm": round(tempo_bpm, 3),
        "beat_count": int(len(beat_frames)),
        "rms_mean": safe_mean(rms),
        "rms_std": safe_std(rms),
        "zero_crossing_rate_mean": safe_mean(zero_crossing_rate),
        "zero_crossing_rate_std": safe_std(zero_crossing_rate),
        "spectral_centroid_mean": safe_mean(spectral_centroid),
        "spectral_centroid_std": safe_std(spectral_centroid),
        "spectral_bandwidth_mean": safe_mean(spectral_bandwidth),
        "spectral_bandwidth_std": safe_std(spectral_bandwidth),
        "spectral_rolloff_mean": safe_mean(spectral_rolloff),
        "spectral_rolloff_std": safe_std(spectral_rolloff),
        "spectral_flatness_mean": safe_mean(spectral_flatness),
        "spectral_flatness_std": safe_std(spectral_flatness),
        "onset_strength_mean": safe_mean(onset_strength),
        "onset_strength_std": safe_std(onset_strength),
        "estimated_key": key_name,
        "estimated_mode": mode,
        "key_confidence": key_confidence,
    }

    for index in range(12):
        result[f"chroma_{index:02d}_mean"] = float(chroma_mean[index])

    for index in range(N_MFCC):
        result[f"mfcc_{index + 1:02d}_mean"] = safe_mean(
            mfcc[index]
        )

        result[f"mfcc_{index + 1:02d}_std"] = safe_std(
            mfcc[index]
        )

    return result


def load_checkpoint() -> list[dict[str, Any]]:
    """Load a prior extraction checkpoint."""

    if not CHECKPOINT_JSON_PATH.exists():
        return []

    with CHECKPOINT_JSON_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError(
            f"Checkpoint is not a list: {CHECKPOINT_JSON_PATH}"
        )

    return payload


def save_checkpoint(records: list[dict[str, Any]]) -> None:
    """Save extraction progress atomically."""

    INTERIM_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = CHECKPOINT_JSON_PATH.with_suffix(
        ".json.tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_path.replace(CHECKPOINT_JSON_PATH)


def save_outputs(
    feature_records: list[dict[str, Any]],
    match_records: list[dict[str, Any]],
    error_records: list[dict[str, Any]],
) -> None:
    """Save all extraction and review outputs."""

    PROCESSED_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    INTERIM_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if feature_records:
        features_df = pd.DataFrame(feature_records)

        features_df = (
            features_df
            .drop_duplicates(
                subset=["audio_file"],
                keep="last",
            )
            .sort_values(
                ["album_order", "track_position"],
                na_position="last",
            )
            .reset_index(drop=True)
        )

        features_df.to_parquet(
            FEATURES_PARQUET_PATH,
            index=False,
        )

        features_df.to_csv(
            FEATURES_CSV_PATH,
            index=False,
            encoding="utf-8",
        )

    matches_df = pd.DataFrame(match_records)

    if not matches_df.empty:
        matches_df = (
            matches_df
            .drop_duplicates(
                subset=["audio_file"],
                keep="last",
            )
            .sort_values(
                ["match_status", "match_score"],
                ascending=[True, False],
            )
            .reset_index(drop=True)
        )

        matches_df.to_csv(
            MATCHES_CSV_PATH,
            index=False,
            encoding="utf-8",
        )

        unmatched_df = matches_df[
            matches_df["match_status"].ne("matched")
        ].copy()

        unmatched_df.to_csv(
            UNMATCHED_CSV_PATH,
            index=False,
            encoding="utf-8",
        )

    if error_records:
        pd.DataFrame(error_records).to_csv(
            ERRORS_CSV_PATH,
            index=False,
            encoding="utf-8",
        )


def main() -> None:
    """Run local audio matching and feature extraction."""

    master_tracks = load_master_tracks()
    audio_files = discover_audio_files()

    if not audio_files:
        raise FileNotFoundError(
            "No supported audio files were found.\n"
            f"Place files under: {AUDIO_ROOT}"
        )

    existing_records = load_checkpoint()

    processed_files = {
        item["audio_file"]
        for item in existing_records
        if item.get("audio_file")
        and item.get("processing_status") == "completed"
    }

    feature_records = [
        item["features"]
        for item in existing_records
        if item.get("processing_status") == "completed"
        and item.get("features")
    ]

    match_records = [
        item["match"]
        for item in existing_records
        if item.get("match")
    ]

    error_records = [
        item
        for item in existing_records
        if item.get("processing_status") == "error"
    ]

    pending_files = [
        path
        for path in audio_files
        if str(path.relative_to(PROJECT_ROOT)) not in processed_files
    ]

    print(f"Audio files discovered: {len(audio_files)}")
    print(f"Already processed: {len(processed_files)}")
    print(f"Pending files: {len(pending_files)}")

    checkpoint_records = list(existing_records)

    for counter, audio_path in enumerate(
        pending_files,
        start=1,
    ):
        relative_path = str(
            audio_path.relative_to(PROJECT_ROOT)
        )

        print(
            f"\n[{counter}/{len(pending_files)}] "
            f"{relative_path}"
        )

        try:
            features = extract_features(audio_path)

            match = match_audio_file(
                audio_path=audio_path,
                master_tracks=master_tracks,
                measured_duration_seconds=features[
                    "duration_seconds_measured"
                ],
            )

            matched_track = master_tracks.loc[
                master_tracks["master_track_id"].eq(
                    match["master_track_id"]
                )
            ].iloc[0]

            features.update(
                {
                    "master_track_id": match["master_track_id"],
                    "album_order": matched_track["album_order"],
                    "album": matched_track["album"],
                    "track_position": matched_track["track_position"],
                    "track_title": matched_track["track_title"],
                    "canonical_title": matched_track["canonical_title"],
                    "match_status": match["match_status"],
                    "match_score": match["match_score"],
                }
            )

            match_record = {
                "audio_file": relative_path,
                "audio_filename": audio_path.name,
                **match,
            }

            checkpoint_record = {
                "audio_file": relative_path,
                "processing_status": "completed",
                "features": features,
                "match": match_record,
            }

            feature_records.append(features)
            match_records.append(match_record)

            print(
                f"  Match: {match['album']} — "
                f"{match['track_title']}"
            )

            print(
                f"  Status: {match['match_status']} | "
                f"Score: {match['match_score']}"
            )

            print(
                f"  Tempo: {features['tempo_bpm']} BPM | "
                f"Key: {features['estimated_key']} "
                f"{features['estimated_mode']}"
            )

        except Exception as error:
            checkpoint_record = {
                "audio_file": relative_path,
                "processing_status": "error",
                "error": str(error),
            }

            error_records.append(checkpoint_record)

            print(f"  Error: {error}")

        checkpoint_records = [
            item
            for item in checkpoint_records
            if item.get("audio_file") != relative_path
        ]

        checkpoint_records.append(checkpoint_record)
        save_checkpoint(checkpoint_records)

    save_outputs(
        feature_records=feature_records,
        match_records=match_records,
        error_records=error_records,
    )

    print("\nAudio feature extraction summary:")
    print(f"Completed files: {len(feature_records)}")
    print(f"Match records: {len(match_records)}")
    print(f"Errors: {len(error_records)}")

    if feature_records:
        print(f"\nSaved: {FEATURES_PARQUET_PATH}")
        print(f"Saved: {FEATURES_CSV_PATH}")

    print(f"Saved match report: {MATCHES_CSV_PATH}")
    print(f"Saved unmatched report: {UNMATCHED_CSV_PATH}")

    if error_records:
        print(f"Saved error report: {ERRORS_CSV_PATH}")

    print("Audio feature extraction completed.")


if __name__ == "__main__":
    main()
