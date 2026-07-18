"""Download Linkin Park lyrics from LRCLIB.

Inputs
------
data/processed/master_tracks.parquet

Local-only outputs
------------------
data/raw/lyrics/lyrics.json

Review outputs
--------------
data/interim/lyrics/lyrics_matches.parquet
data/interim/lyrics/lyrics_matches.csv
data/interim/lyrics/unmatched_tracks.csv

Notes
-----
Raw lyrics are copyrighted material and must remain excluded from Git.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from rapidfuzz.fuzz import ratio


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_tracks.parquet"
)

RAW_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "lyrics"
INTERIM_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "lyrics"

LYRICS_JSON_PATH = RAW_OUTPUT_DIR / "lyrics.json"
MATCHES_PARQUET_PATH = INTERIM_OUTPUT_DIR / "lyrics_matches.parquet"
MATCHES_CSV_PATH = INTERIM_OUTPUT_DIR / "lyrics_matches.csv"
UNMATCHED_CSV_PATH = INTERIM_OUTPUT_DIR / "unmatched_tracks.csv"

BASE_URL = "https://lrclib.net/api/search"

HEADERS = {
    "User-Agent": (
        "Linkin-Park-Evolution/0.1 "
        "(https://github.com/skr-35/Linkin-Park-Evolution)"
    )
}

ARTIST_NAME = "Linkin Park"

REQUEST_DELAY_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 5
SAVE_EVERY = 5

# Scores below this threshold are retained for manual review,
# but they are not treated as confident matches.
CONFIDENT_MATCH_THRESHOLD = 80.0


def normalize_text(value: Any) -> str:
    """Normalize text for fuzzy matching."""

    if value is None:
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.lower().strip()
    text = text.replace("&", " and ")

    text = re.sub(
        r"\bfeat(?:uring)?\.?\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"_", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_existing_results() -> list[dict[str, Any]]:
    """Load the checkpoint file when a previous run exists."""

    if not LYRICS_JSON_PATH.exists():
        return []

    with LYRICS_JSON_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a list in checkpoint file: {LYRICS_JSON_PATH}"
        )

    print(f"Loaded {len(data)} existing lyric results.")
    return data


def save_json(data: list[dict[str, Any]]) -> None:
    """Save raw search results and lyrics locally."""

    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    temporary_path = LYRICS_JSON_PATH.with_suffix(".json.tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_path.replace(LYRICS_JSON_PATH)


def request_candidates(
    track_name: str,
    album_name: str,
) -> list[dict[str, Any]]:
    """Search LRCLIB and return candidate lyric records."""

    params = {
        "track_name": track_name,
        "artist_name": ARTIST_NAME,
        "album_name": album_name,
    }

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                BASE_URL,
                params=params,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            response.raise_for_status()
            payload = response.json()

            if not isinstance(payload, list):
                raise ValueError(
                    "LRCLIB search response was not a list."
                )

            return payload

        except (
            requests.RequestException,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            last_error = error

            if attempt == MAX_RETRIES:
                break

            wait_seconds = min(2 ** attempt, 30) #with backoff

            print(
                f"  Request failed, retrying in {wait_seconds}s "
                f"({attempt}/{MAX_RETRIES})..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"LRCLIB request failed after {MAX_RETRIES} attempts."
    ) from last_error


def request_fallback_candidates(
    track_name: str,
) -> list[dict[str, Any]]:
    """Search LRCLIB without an album constraint."""

    params = {
        "track_name": track_name,
        "artist_name": ARTIST_NAME,
    }

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                BASE_URL,
                params=params,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()

            if not isinstance(payload, list):
                raise ValueError(
                    "LRCLIB fallback response was not a list."
                )

            return payload

        except (
            requests.RequestException,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            last_error = error

            if attempt == MAX_RETRIES:
                break

            wait_seconds = min(2 ** attempt, 30)
            print(
                f"  Fallback failed, retrying in {wait_seconds}s "
                f"({attempt}/{MAX_RETRIES})..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"LRCLIB fallback failed after {MAX_RETRIES} attempts."
    ) from last_error


def duration_similarity(
    expected_seconds: float | int | None,
    candidate_seconds: float | int | None,
) -> float:
    """Score duration similarity between zero and one hundred."""

    if pd.isna(expected_seconds) or candidate_seconds is None:
        return 50.0

    expected = float(expected_seconds)
    candidate = float(candidate_seconds)

    difference = abs(expected - candidate)

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


def score_candidate(
    track: pd.Series,
    candidate: dict[str, Any],
) -> dict[str, float]:
    """Calculate component and overall candidate match scores."""

    expected_title = normalize_text(track["canonical_title"])
    expected_album = normalize_text(track["album"])
    expected_artist = normalize_text(ARTIST_NAME)

    candidate_title = normalize_text(candidate.get("trackName"))
    candidate_album = normalize_text(candidate.get("albumName"))
    candidate_artist = normalize_text(candidate.get("artistName"))

    title_score = float(ratio(expected_title, candidate_title))
    album_score = float(ratio(expected_album, candidate_album))
    artist_score = float(ratio(expected_artist, candidate_artist))

    duration_score = duration_similarity(
        track.get("duration_seconds"),
        candidate.get("duration"),
    )

    # Title is the strongest signal. Album and duration reduce the
    # likelihood of selecting live, deluxe, remix, or unrelated versions.
    total_score = (
        0.50 * title_score
        + 0.20 * album_score
        + 0.15 * artist_score
        + 0.15 * duration_score
    )

    return {
        "title_score": round(title_score, 2),
        "album_score": round(album_score, 2),
        "artist_score": round(artist_score, 2),
        "duration_score": round(duration_score, 2),
        "match_score": round(total_score, 2),
    }


def select_best_candidate(
    track: pd.Series,
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, float] | None]:
    """Select the strongest candidate, preferring records with lyrics."""

    if not candidates:
        return None, None

    scored_candidates: list[tuple[dict[str, Any], dict[str, float]]] = []

    for candidate in candidates:
        scores = score_candidate(track, candidate)
        has_lyrics = bool(
            candidate.get("plainLyrics")
            or candidate.get("syncedLyrics")
        )
        scores["selection_score"] = (
            scores["match_score"] + (5.0 if has_lyrics else 0.0)
        )
        scored_candidates.append((candidate, scores))

    candidates_with_lyrics = [
        item
        for item in scored_candidates
        if item[0].get("plainLyrics") or item[0].get("syncedLyrics")
    ]

    candidate_pool = candidates_with_lyrics or scored_candidates
    candidate_pool.sort(
        key=lambda item: item[1]["selection_score"],
        reverse=True,
    )

    best_candidate, best_scores = candidate_pool[0]
    best_scores.pop("selection_score", None)
    return best_candidate, best_scores

def build_result(
    track: pd.Series,
    candidate: dict[str, Any] | None,
    scores: dict[str, float] | None,
    candidate_count: int,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Build one checkpoint record."""

    retrieved_at = datetime.now(timezone.utc).isoformat()

    base_record: dict[str, Any] = {
        "master_track_id": track["master_track_id"],
        "album": track["album"],
        "track_title": track["track_title"],
        "canonical_title": track["canonical_title"],
        "artist_name": ARTIST_NAME,
        "expected_duration_seconds": (
            None
            if pd.isna(track.get("duration_seconds"))
            else float(track["duration_seconds"])
        ),
        "candidate_count": candidate_count,
        "retrieved_at": retrieved_at,
        "source": "LRCLIB",
        "error": error_message,
    }

    if candidate is None or scores is None:
        return {
            **base_record,
            "match_status": (
                "error" if error_message else "not_found"
            ),
            "match_score": None,
            "title_score": None,
            "album_score": None,
            "artist_score": None,
            "duration_score": None,
            "lrclib_id": None,
            "matched_track_name": None,
            "matched_artist_name": None,
            "matched_album_name": None,
            "matched_duration_seconds": None,
            "instrumental": None,
            "plain_lyrics": None,
            "synced_lyrics": None,
            "lyrics_available": False,
        }

    plain_lyrics = candidate.get("plainLyrics")
    synced_lyrics = candidate.get("syncedLyrics")

    lyrics_available = bool(
        (plain_lyrics and str(plain_lyrics).strip())
        or (synced_lyrics and str(synced_lyrics).strip())
    )

    match_score = scores["match_score"]

    if match_score >= CONFIDENT_MATCH_THRESHOLD:
        match_status = "matched"
    else:
        match_status = "review"

    if not lyrics_available:
        match_status = "no_lyrics"

    return {
        **base_record,
        "match_status": match_status,
        **scores,
        "lrclib_id": candidate.get("id"),
        "matched_track_name": candidate.get("trackName"),
        "matched_artist_name": candidate.get("artistName"),
        "matched_album_name": candidate.get("albumName"),
        "matched_duration_seconds": candidate.get("duration"),
        "instrumental": candidate.get("instrumental"),
        "plain_lyrics": plain_lyrics,
        "synced_lyrics": synced_lyrics,
        "lyrics_available": lyrics_available,
    }


def create_review_dataframe(
    results: list[dict[str, Any]],
) -> pd.DataFrame:
    """Create a shareable review table without lyric text."""

    df = pd.DataFrame(results)

    lyric_columns = [
        "plain_lyrics",
        "synced_lyrics",
    ]

    existing_lyric_columns = [
        column
        for column in lyric_columns
        if column in df.columns
    ]

    return df.drop(columns=existing_lyric_columns)


def save_review_outputs(
    results: list[dict[str, Any]],
) -> None:
    """Save matching metadata without raw lyric text."""

    INTERIM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    review_df = create_review_dataframe(results)

    review_df = review_df.sort_values(
        ["album", "master_track_id"],
        na_position="last",
    ).reset_index(drop=True)

    review_df.to_parquet(
        MATCHES_PARQUET_PATH,
        index=False,
    )

    review_df.to_csv(
        MATCHES_CSV_PATH,
        index=False,
        encoding="utf-8",
    )

    unmatched_df = review_df[
        review_df["match_status"].ne("matched")
    ].copy()

    unmatched_df.to_csv(
        UNMATCHED_CSV_PATH,
        index=False,
        encoding="utf-8",
    )


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print collection and matching summary."""

    df = pd.DataFrame(results)

    print("\nLyrics collection summary:")

    if df.empty:
        print("No tracks were processed.")
        return

    status_counts = (
        df["match_status"]
        .value_counts(dropna=False)
        .rename_axis("status")
        .reset_index(name="tracks")
    )

    print(status_counts.to_string(index=False))

    print(f"\nTotal processed: {len(df)}")
    print(
        "Lyrics available: "
        f"{int(df['lyrics_available'].fillna(False).sum())}"
    )

    matched_scores = df.loc[
        df["match_status"].eq("matched"),
        "match_score",
    ]

    if not matched_scores.empty:
        print(
            "Average matched score: "
            f"{matched_scores.mean():.2f}"
        )


def main() -> None:
    """Download lyrics for all canonical album tracks."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {INPUT_PATH}\n"
            "Run build_master_tracks.py first."
        )

    print(f"Reading: {INPUT_PATH}")

    tracks = pd.read_parquet(INPUT_PATH)

    required_columns = {
        "master_track_id",
        "album",
        "track_title",
        "canonical_title",
        "duration_seconds",
    }

    missing_columns = required_columns - set(tracks.columns)

    if missing_columns:
        raise ValueError(
            "Master track catalogue is missing columns: "
            f"{sorted(missing_columns)}"
        )

    existing_results = load_existing_results()

    # Only successful or definitive results are considered complete.
    # Temporary request errors will be retried on the next run.
    completed_statuses = {
        "matched",
        "review",
        "not_found",
    }

    processed_ids = {
        item["master_track_id"]
        for item in existing_results
        if (
            item.get("master_track_id")
            and item.get("match_status") in completed_statuses
        )
    }

    # Remove old error records before retrying them, preventing duplicates.
    results = [
        item
        for item in existing_results
        if item.get("match_status") in completed_statuses
    ]

    pending_tracks = tracks[
        ~tracks["master_track_id"].isin(processed_ids)
    ].copy()

    print(f"Total catalogue tracks: {len(tracks)}")
    print(f"Already processed: {len(processed_ids)}")
    print(f"Pending tracks: {len(pending_tracks)}")

    if pending_tracks.empty:
        print("No pending tracks. Rebuilding review outputs.")
        save_review_outputs(results)
        print_summary(results)
        return

    for counter, (_, track) in enumerate(
        pending_tracks.iterrows(),
        start=1,
    ):
        title = track["canonical_title"]
        album = track["album"]

        print(
            f"\n[{counter}/{len(pending_tracks)}] "
            f"{album} — {title}"
        )

        try:
            candidates = request_candidates(
                track_name=title,
                album_name=album,
            )

            candidate, scores = select_best_candidate(
                track,
                candidates,
            )

            has_usable_lyrics = bool(
                candidate
                and (
                    candidate.get("plainLyrics")
                    or candidate.get("syncedLyrics")
                )
            )

            if not has_usable_lyrics:
                print(
                    "  Exact search has no lyrics; "
                    "trying fallback search..."
                )
                fallback_candidates = request_fallback_candidates(
                    track_name=title,
                )
                combined_candidates = candidates + fallback_candidates
                unique_candidates: dict[Any, dict[str, Any]] = {}

                for index, item in enumerate(combined_candidates):
                    candidate_key = item.get("id")
                    if candidate_key is None:
                        candidate_key = (
                            "anonymous",
                            normalize_text(item.get("trackName")),
                            normalize_text(item.get("artistName")),
                            normalize_text(item.get("albumName")),
                            item.get("duration"),
                            index,
                        )
                    unique_candidates[candidate_key] = item

                candidates = list(unique_candidates.values())
                candidate, scores = select_best_candidate(
                    track,
                    candidates,
                )

            result = build_result(
                track=track,
                candidate=candidate,
                scores=scores,
                candidate_count=len(candidates),
            )

            print(
                f"  Status: {result['match_status']} | "
                f"Score: {result['match_score']} | "
                f"Candidates: {len(candidates)}"
            )

            if result.get("matched_track_name"):
                print(
                    "  Match: "
                    f"{result['matched_track_name']} — "
                    f"{result['matched_album_name']}"
                )

        except Exception as error:
            print(f"  Error: {error}")

            result = build_result(
                track=track,
                candidate=None,
                scores=None,
                candidate_count=0,
                error_message=str(error),
            )

        results.append(result)

        if counter % SAVE_EVERY == 0:
            save_json(results)
            save_review_outputs(results)
            print(f"  Checkpoint saved after {counter} pending tracks.")

        time.sleep(REQUEST_DELAY_SECONDS)

    save_json(results)
    save_review_outputs(results)
    print_summary(results)

    print(f"\nSaved local lyrics: {LYRICS_JSON_PATH}")
    print(f"Saved review table: {MATCHES_PARQUET_PATH}")
    print(f"Saved review CSV: {MATCHES_CSV_PATH}")
    print(f"Saved unmatched report: {UNMATCHED_CSV_PATH}")
    print("Lyrics collection completed.")


if __name__ == "__main__":
    main()