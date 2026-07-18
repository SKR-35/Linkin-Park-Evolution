"""Download frozen AcousticBrainz features by MusicBrainz recording ID.

Input
-----
data/processed/master_tracks.parquet

Local-only outputs
------------------
data/raw/acousticbrainz/lowlevel/<recording_id>.json
data/raw/acousticbrainz/highlevel/<recording_id>.json

Review outputs
--------------
data/interim/audio_features/acousticbrainz_status.parquet
data/interim/audio_features/acousticbrainz_status.csv
data/interim/audio_features/acousticbrainz_unavailable.csv

Important
---------
AcousticBrainz was discontinued in 2022. Its historical API may return
404/410 or otherwise be unavailable. This script checks availability,
downloads any still-accessible records, saves checkpoints, and exits
cleanly with a useful report when the service is unavailable.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_tracks.parquet"
)

RAW_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "acousticbrainz"
LOWLEVEL_DIR = RAW_OUTPUT_DIR / "lowlevel"
HIGHLEVEL_DIR = RAW_OUTPUT_DIR / "highlevel"

INTERIM_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "audio_features"
)

STATUS_PARQUET_PATH = (
    INTERIM_OUTPUT_DIR
    / "acousticbrainz_status.parquet"
)

STATUS_CSV_PATH = (
    INTERIM_OUTPUT_DIR
    / "acousticbrainz_status.csv"
)

UNAVAILABLE_CSV_PATH = (
    INTERIM_OUTPUT_DIR
    / "acousticbrainz_unavailable.csv"
)

BASE_URL = "https://acousticbrainz.org/api/v1"

HEADERS = {
    "User-Agent": (
        "Linkin-Park-Evolution/0.1 "
        "(https://github.com/skr-35/Linkin-Park-Evolution)"
    )
}

REQUEST_TIMEOUT_SECONDS = 60
REQUEST_DELAY_SECONDS = 1.2
MAX_RETRIES = 3
SAVE_EVERY = 10

TERMINAL_HTTP_STATUSES = {
    404,
    410,
}


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def ensure_directories() -> None:
    """Create output directories."""

    LOWLEVEL_DIR.mkdir(parents=True, exist_ok=True)
    HIGHLEVEL_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_json(
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    """Save JSON atomically."""

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_path.replace(output_path)


def request_feature_payload(
    recording_id: str,
    feature_level: str,
) -> tuple[str, dict[str, Any] | None, str | None]:
    """Request one AcousticBrainz feature payload.

    Returns
    -------
    status, payload, error_message
    """

    url = (
        f"{BASE_URL}/{recording_id}/"
        f"{feature_level}"
    )

    last_error: str | None = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            if response.status_code == 200:
                payload = response.json()

                if not isinstance(payload, dict):
                    return (
                        "invalid_payload",
                        None,
                        "Response JSON was not an object.",
                    )

                return "found", payload, None

            if response.status_code in TERMINAL_HTTP_STATUSES:
                return (
                    "not_found",
                    None,
                    f"HTTP {response.status_code}",
                )

            if response.status_code >= 500:
                last_error = (
                    f"HTTP {response.status_code}"
                )
            else:
                return (
                    "http_error",
                    None,
                    f"HTTP {response.status_code}: "
                    f"{response.text[:200]}",
                )

        except (
            requests.RequestException,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            last_error = str(error)

        if attempt < MAX_RETRIES:
            wait_seconds = min(
                2 ** attempt,
                20,
            )

            print(
                f"    Retry in {wait_seconds}s "
                f"({attempt}/{MAX_RETRIES})..."
            )

            time.sleep(wait_seconds)

    return (
        "request_error",
        None,
        last_error,
    )


def build_status_record(
    track: pd.Series,
    lowlevel_status: str,
    highlevel_status: str,
    lowlevel_error: str | None,
    highlevel_error: str | None,
) -> dict[str, Any]:
    """Create one status row."""

    return {
        "master_track_id": track["master_track_id"],
        "album": track["album"],
        "track_title": track["track_title"],
        "recording_id": track["recording_id"],
        "lowlevel_status": lowlevel_status,
        "highlevel_status": highlevel_status,
        "lowlevel_available": (
            lowlevel_status == "found"
        ),
        "highlevel_available": (
            highlevel_status == "found"
        ),
        "any_audio_features_available": (
            lowlevel_status == "found"
            or highlevel_status == "found"
        ),
        "lowlevel_error": lowlevel_error,
        "highlevel_error": highlevel_error,
        "checked_at": utc_now(),
        "source": "AcousticBrainz",
    }


def load_existing_status() -> pd.DataFrame:
    """Load an existing checkpoint table."""

    if not STATUS_PARQUET_PATH.exists():
        return pd.DataFrame()

    status_df = pd.read_parquet(
        STATUS_PARQUET_PATH
    )

    print(
        f"Loaded {len(status_df)} "
        "existing status records."
    )

    return status_df


def save_status_outputs(
    records: list[dict[str, Any]],
) -> None:
    """Save checkpoint and review outputs."""

    if not records:
        return

    status_df = pd.DataFrame(records)

    status_df = (
        status_df
        .drop_duplicates(
            subset=["master_track_id"],
            keep="last",
        )
        .sort_values(
            ["album", "master_track_id"]
        )
        .reset_index(drop=True)
    )

    status_df.to_parquet(
        STATUS_PARQUET_PATH,
        index=False,
    )

    status_df.to_csv(
        STATUS_CSV_PATH,
        index=False,
        encoding="utf-8",
    )

    unavailable_df = status_df[
        ~status_df[
            "any_audio_features_available"
        ]
    ].copy()

    unavailable_df.to_csv(
        UNAVAILABLE_CSV_PATH,
        index=False,
        encoding="utf-8",
    )


def print_summary(
    records: list[dict[str, Any]],
) -> None:
    """Print collection summary."""

    if not records:
        print("No tracks were processed.")
        return

    df = pd.DataFrame(records)

    df = df.drop_duplicates(
        subset=["master_track_id"],
        keep="last",
    )

    print("\nAcousticBrainz summary:")
    print(
        f"Total tracks: {len(df)}"
    )
    print(
        "Low-level available: "
        f"{int(df['lowlevel_available'].sum())}"
    )
    print(
        "High-level available: "
        f"{int(df['highlevel_available'].sum())}"
    )
    print(
        "Any features available: "
        f"{int(df['any_audio_features_available'].sum())}"
    )
    print(
        "Unavailable: "
        f"{int((~df['any_audio_features_available']).sum())}"
    )

    print("\nLow-level statuses:")
    print(
        df["lowlevel_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nHigh-level statuses:")
    print(
        df["highlevel_status"]
        .value_counts(dropna=False)
        .to_string()
    )


def service_preflight(
    recording_id: str,
) -> None:
    """Check whether the historical API appears reachable."""

    print(
        "Checking AcousticBrainz API availability..."
    )

    status, _, error = request_feature_payload(
        recording_id,
        "low-level",
    )

    print(
        f"Preflight result: {status}"
        + (
            f" ({error})"
            if error
            else ""
        )
    )

    if status in {
        "http_error",
        "request_error",
    }:
        print(
            "\nWarning: the historical AcousticBrainz "
            "API may be unavailable. The script will "
            "continue and create a status report."
        )


def main() -> None:
    """Download all available AcousticBrainz features."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {INPUT_PATH}\n"
            "Run build_master_tracks.py first."
        )

    ensure_directories()

    tracks = pd.read_parquet(
        INPUT_PATH
    )

    required_columns = {
        "master_track_id",
        "album",
        "track_title",
        "recording_id",
    }

    missing_columns = (
        required_columns
        - set(tracks.columns)
    )

    if missing_columns:
        raise ValueError(
            "Master catalogue is missing columns: "
            f"{sorted(missing_columns)}"
        )

    tracks = tracks.dropna(
        subset=["recording_id"]
    ).copy()

    if tracks.empty:
        raise ValueError(
            "No MusicBrainz recording IDs were found."
        )

    existing_status_df = (
        load_existing_status()
    )

    if existing_status_df.empty:
        records: list[dict[str, Any]] = []
        completed_ids: set[str] = set()
    else:
        records = (
            existing_status_df
            .to_dict(orient="records")
        )
        completed_ids = set(
            existing_status_df[
                "master_track_id"
            ].astype(str)
        )

    pending_tracks = tracks[
        ~tracks["master_track_id"]
        .astype(str)
        .isin(completed_ids)
    ].copy()

    print(
        f"Total catalogue tracks: {len(tracks)}"
    )
    print(
        f"Already processed: {len(completed_ids)}"
    )
    print(
        f"Pending tracks: {len(pending_tracks)}"
    )

    if pending_tracks.empty:
        save_status_outputs(records)
        print_summary(records)
        return

    service_preflight(
        str(
            pending_tracks.iloc[0][
                "recording_id"
            ]
        )
    )

    for counter, (_, track) in enumerate(
        pending_tracks.iterrows(),
        start=1,
    ):
        recording_id = str(
            track["recording_id"]
        )

        print(
            f"\n[{counter}/{len(pending_tracks)}] "
            f"{track['album']} — "
            f"{track['track_title']}"
        )

        lowlevel_path = (
            LOWLEVEL_DIR
            / f"{recording_id}.json"
        )

        highlevel_path = (
            HIGHLEVEL_DIR
            / f"{recording_id}.json"
        )

        if lowlevel_path.exists():
            lowlevel_status = "found"
            lowlevel_error = None
        else:
            (
                lowlevel_status,
                lowlevel_payload,
                lowlevel_error,
            ) = request_feature_payload(
                recording_id,
                "low-level",
            )

            if (
                lowlevel_status == "found"
                and lowlevel_payload is not None
            ):
                save_json(
                    lowlevel_payload,
                    lowlevel_path,
                )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

        if highlevel_path.exists():
            highlevel_status = "found"
            highlevel_error = None
        else:
            (
                highlevel_status,
                highlevel_payload,
                highlevel_error,
            ) = request_feature_payload(
                recording_id,
                "high-level",
            )

            if (
                highlevel_status == "found"
                and highlevel_payload is not None
            ):
                save_json(
                    highlevel_payload,
                    highlevel_path,
                )

        print(
            "  Low-level: "
            f"{lowlevel_status} | "
            "High-level: "
            f"{highlevel_status}"
        )

        record = build_status_record(
            track=track,
            lowlevel_status=lowlevel_status,
            highlevel_status=highlevel_status,
            lowlevel_error=lowlevel_error,
            highlevel_error=highlevel_error,
        )

        records.append(record)

        if counter % SAVE_EVERY == 0:
            save_status_outputs(records)

            print(
                "  Checkpoint saved."
            )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    save_status_outputs(records)
    print_summary(records)

    print(
        f"\nSaved status table: "
        f"{STATUS_PARQUET_PATH}"
    )
    print(
        f"Saved status CSV: "
        f"{STATUS_CSV_PATH}"
    )
    print(
        f"Saved unavailable report: "
        f"{UNAVAILABLE_CSV_PATH}"
    )
    print(
        "AcousticBrainz collection completed."
    )


if __name__ == "__main__":
    main()
