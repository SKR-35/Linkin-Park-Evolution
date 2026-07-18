"""Download Linkin Park album and track metadata from MusicBrainz."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ARTIST_ID = "f59c5520-5f46-4d2c-b2c4-822eabf53419"
BASE_URL = "https://musicbrainz.org/ws/2"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "musicbrainz"

HEADERS = {
    "User-Agent": (
        "Linkin-Park-Evolution/0.1 "
        "(https://github.com/skr-35/Linkin-Park-Evolution)"
    )
}

STUDIO_ALBUMS = {
    "Hybrid Theory",
    "Meteora",
    "Minutes to Midnight",
    "A Thousand Suns",
    "Living Things",
    "The Hunting Party",
    "One More Light",
    "From Zero",
}


def get_json(
    endpoint: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send a rate-limited GET request to MusicBrainz."""

    request_params: dict[str, Any] = {"fmt": "json"}

    if params:
        request_params.update(params)

    response = requests.get(
        f"{BASE_URL}/{endpoint}",
        params=request_params,
        headers=HEADERS,
        timeout=60,
    )
    response.raise_for_status()

    # MusicBrainz requests responsible rate limiting.
    time.sleep(1.1)

    return response.json()


def save_json(data: Any, filename: str) -> None:
    """Save JSON data."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"Saved: {output_path}")


def download_artist() -> dict[str, Any]:
    """Download artist metadata."""

    artist = get_json(
        f"artist/{ARTIST_ID}",
        params={"inc": "aliases+tags+ratings+genres+url-rels"},
    )

    save_json(artist, "artist.json")
    return artist


def download_release_groups() -> list[dict[str, Any]]:
    """Download all release groups associated with Linkin Park."""

    release_groups: list[dict[str, Any]] = []
    offset = 0
    limit = 100

    while True:
        response = get_json(
            "release-group",
            params={
                "artist": ARTIST_ID,
                "limit": limit,
                "offset": offset,
            },
        )

        batch = response.get("release-groups", [])
        release_groups.extend(batch)

        total = response.get("release-group-count", "?")
        print(f"Release groups downloaded: {len(release_groups)} / {total}")

        if len(batch) < limit:
            break

        offset += limit

    save_json(release_groups, "release_groups.json")
    return release_groups


def select_studio_album_groups(
    release_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select the eight principal studio-album release groups."""

    selected = []

    for item in release_groups:
        title = item.get("title")
        primary_type = item.get("primary-type")
        secondary_types = item.get("secondary-types", [])

        if (
            title in STUDIO_ALBUMS
            and primary_type == "Album"
            and not secondary_types
        ):
            selected.append(item)

    selected.sort(key=lambda item: item.get("first-release-date", ""))

    found_titles = {item.get("title") for item in selected}
    missing_titles = STUDIO_ALBUMS - found_titles

    if missing_titles:
        print(f"Warning: studio albums not found: {sorted(missing_titles)}")

    save_json(selected, "studio_album_release_groups.json")
    return selected


def download_releases_for_group(
    release_group_id: str,
) -> list[dict[str, Any]]:
    """Download releases belonging to one release group."""

    releases: list[dict[str, Any]] = []
    offset = 0
    limit = 100

    while True:
        response = get_json(
            "release",
            params={
                "release-group": release_group_id,
                "limit": limit,
                "offset": offset,
            },
        )

        batch = response.get("releases", [])
        releases.extend(batch)

        if len(batch) < limit:
            break

        offset += limit

    return releases


def choose_representative_release(
    releases: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Choose one release edition to represent an album.

    Preference:
    1. Official release
    2. Worldwide or US release
    3. Earliest dated release
    """

    if not releases:
        return None

    def score(release: dict[str, Any]) -> tuple[int, int, str]:
        official_score = 0 if release.get("status") == "Official" else 1
        country_score = 0 if release.get("country") in {"XW", "US"} else 1
        date = release.get("date") or "9999-99-99"

        return official_score, country_score, date

    return sorted(releases, key=score)[0]


def download_release_with_tracks(
    release_id: str,
) -> dict[str, Any]:
    """Download one release with media, tracks, recordings and ISRCs."""

    return get_json(
        f"release/{release_id}",
        params={
            "inc": "recordings+isrcs+artist-credits+release-groups",
        },
    )


def extract_tracks(
    album_group: dict[str, Any],
    release: dict[str, Any],
) -> list[dict[str, Any]]:
    """Flatten MusicBrainz media and track data."""

    rows: list[dict[str, Any]] = []

    for medium in release.get("media", []):
        disc_number = medium.get("position")

        for track in medium.get("tracks", []):
            recording = track.get("recording", {})
            artist_credit = recording.get("artist-credit", [])

            rows.append(
                {
                    "release_group_id": album_group.get("id"),
                    "album": album_group.get("title"),
                    "album_first_release_date": album_group.get(
                        "first-release-date"
                    ),
                    "release_id": release.get("id"),
                    "release_title": release.get("title"),
                    "release_date": release.get("date"),
                    "release_country": release.get("country"),
                    "release_status": release.get("status"),
                    "disc_number": disc_number,
                    "track_number": track.get("number"),
                    "track_position": track.get("position"),
                    "track_id": track.get("id"),
                    "track_title": track.get("title"),
                    "track_length_ms": track.get("length"),
                    "recording_id": recording.get("id"),
                    "recording_title": recording.get("title"),
                    "recording_length_ms": recording.get("length"),
                    "isrcs": ", ".join(recording.get("isrcs", [])),
                    "artist_credit": " & ".join(
                        credit.get("name", "")
                        for credit in artist_credit
                    ),
                }
            )

    return rows


def main() -> None:
    """Run the complete MusicBrainz collection pipeline."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading Linkin Park metadata from MusicBrainz...")

    artist = download_artist()
    release_groups = download_release_groups()
    studio_groups = select_studio_album_groups(release_groups)

    selected_releases: list[dict[str, Any]] = []
    detailed_releases: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []

    for album_group in studio_groups:
        album_title = album_group.get("title", "Unknown album")
        release_group_id = album_group["id"]

        print(f"\nProcessing album: {album_title}")

        releases = download_releases_for_group(release_group_id)
        selected_release = choose_representative_release(releases)

        if selected_release is None:
            print(f"No release found for: {album_title}")
            continue

        print(
            "Selected release: "
            f"{selected_release.get('title')} "
            f"({selected_release.get('date')}, "
            f"{selected_release.get('country')})"
        )

        detailed_release = download_release_with_tracks(
            selected_release["id"]
        )

        selected_releases.append(selected_release)
        detailed_releases.append(detailed_release)
        track_rows.extend(
            extract_tracks(album_group, detailed_release)
        )

    save_json(selected_releases, "releases.json")
    save_json(detailed_releases, "recordings.json")

    release_groups_df = pd.DataFrame(
        [
            {
                "release_group_id": item.get("id"),
                "title": item.get("title"),
                "first_release_date": item.get("first-release-date"),
                "primary_type": item.get("primary-type"),
                "secondary_types": ", ".join(
                    item.get("secondary-types", [])
                ),
            }
            for item in release_groups
        ]
    )

    tracks_df = pd.DataFrame(track_rows)

    release_groups_path = OUTPUT_DIR / "release_groups.parquet"
    tracks_path = OUTPUT_DIR / "tracks.parquet"

    release_groups_df.to_parquet(release_groups_path, index=False)
    tracks_df.to_parquet(tracks_path, index=False)

    print("\n----------------------------------------")
    print(f"Artist: {artist.get('name')}")
    print(f"Studio albums selected: {len(studio_groups)}")
    print(f"Representative releases: {len(selected_releases)}")
    print(f"Tracks downloaded: {len(tracks_df)}")
    print(f"Saved: {release_groups_path}")
    print(f"Saved: {tracks_path}")
    print("MusicBrainz metadata download completed.")


if __name__ == "__main__":
    main()