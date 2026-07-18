"""Turn Linkin Park's album embeddings into a chronological visual story.

Inputs
------
data/processed/song_embeddings.parquet
outputs/tables/temporal_album_transitions.csv
outputs/tables/temporal_feature_drifts.csv
outputs/tables/album_evolution_full.csv

Outputs
-------
outputs/tables/album_embedding_story.csv
outputs/tables/album_embedding_nearest_neighbours.csv
outputs/tables/album_embedding_turning_points.csv
outputs/tables/album_embedding_story_cards.csv

outputs/figures/album_embedding_story_pca.png
outputs/figures/album_embedding_story_umap.png
outputs/figures/album_embedding_story_tsne.png
outputs/figures/album_embedding_distance_matrix.png
outputs/figures/album_embedding_storyline.png

Purpose
-------
This script converts dimensionality-reduction outputs into a readable
chronological story of Linkin Park's studio-album evolution.

It answers questions such as:
- Which albums occupy nearby regions of the feature space?
- Which album-to-album moves represent the largest stylistic turns?
- Which albums act as transitions between earlier and later periods?
- How does From Zero relate to the Chester-era catalogue?
- Which album pairs are closest despite being far apart in time?

Interpretation notes
--------------------
- PCA is used for the main chronological story because it is linear and
  interpretable.
- UMAP and t-SNE are supporting local-neighbourhood views.
- Distances in t-SNE should not be interpreted globally.
- Album centroids summarize the average position of songs in each album.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "song_embeddings.parquet"
)

TRANSITIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "temporal_album_transitions.csv"
)

DRIFTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "temporal_feature_drifts.csv"
)

ALBUM_EVOLUTION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "album_evolution_full.csv"
)

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

STORY_PATH = (
    TABLES_DIR
    / "album_embedding_story.csv"
)

NEIGHBOURS_PATH = (
    TABLES_DIR
    / "album_embedding_nearest_neighbours.csv"
)

TURNING_POINTS_PATH = (
    TABLES_DIR
    / "album_embedding_turning_points.csv"
)

STORY_CARDS_PATH = (
    TABLES_DIR
    / "album_embedding_story_cards.csv"
)

PCA_FIGURE_PATH = (
    FIGURES_DIR
    / "album_embedding_story_pca.png"
)

UMAP_FIGURE_PATH = (
    FIGURES_DIR
    / "album_embedding_story_umap.png"
)

TSNE_FIGURE_PATH = (
    FIGURES_DIR
    / "album_embedding_story_tsne.png"
)

DISTANCE_MATRIX_PATH = (
    FIGURES_DIR
    / "album_embedding_distance_matrix.png"
)

STORYLINE_PATH = (
    FIGURES_DIR
    / "album_embedding_storyline.png"
)


REQUIRED_EMBEDDING_COLUMNS = [
    "master_track_id",
    "album_order",
    "album",
    "release_year",
    "era",
    "hybrid_pca_x",
    "hybrid_pca_y",
    "hybrid_umap_x",
    "hybrid_umap_y",
    "hybrid_tsne_x",
    "hybrid_tsne_y",
]


def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:
    """Raise a clear error if required columns are missing."""

    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{label} is missing required columns: "
            f"{sorted(missing)}"
        )


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load embeddings and supporting temporal analysis tables."""

    for path in [
        EMBEDDINGS_PATH,
        TRANSITIONS_PATH,
        DRIFTS_PATH,
        ALBUM_EVOLUTION_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}"
            )

    embeddings = pd.read_parquet(
        EMBEDDINGS_PATH
    )

    transitions = pd.read_csv(
        TRANSITIONS_PATH
    )

    drifts = pd.read_csv(
        DRIFTS_PATH
    )

    album_evolution = pd.read_csv(
        ALBUM_EVOLUTION_PATH
    )

    require_columns(
        embeddings,
        REQUIRED_EMBEDDING_COLUMNS,
        "Song embeddings",
    )

    require_columns(
        transitions,
        [
            "from_album",
            "to_album",
            "profile_distance",
            "evolution_velocity",
            "largest_feature_shifts",
        ],
        "Temporal transitions",
    )

    require_columns(
        drifts,
        [
            "from_album",
            "to_album",
            "feature",
            "standardized_change",
            "absolute_change",
        ],
        "Temporal feature drifts",
    )

    require_columns(
        album_evolution,
        [
            "album_order",
            "album",
            "release_year",
            "era",
        ],
        "Album evolution",
    )

    return (
        embeddings,
        transitions,
        drifts,
        album_evolution,
    )


def build_album_centroids(
    embeddings: pd.DataFrame,
) -> pd.DataFrame:
    """Build album centroids for PCA, UMAP, and t-SNE spaces."""

    centroids = (
        embeddings.groupby(
            [
                "album_order",
                "album",
                "release_year",
                "era",
            ],
            dropna=False,
        )
        .agg(
            song_count=("master_track_id", "count"),
            pca_x=("hybrid_pca_x", "mean"),
            pca_y=("hybrid_pca_y", "mean"),
            umap_x=("hybrid_umap_x", "mean"),
            umap_y=("hybrid_umap_y", "mean"),
            tsne_x=("hybrid_tsne_x", "mean"),
            tsne_y=("hybrid_tsne_y", "mean"),
            pca_dispersion_x=("hybrid_pca_x", "std"),
            pca_dispersion_y=("hybrid_pca_y", "std"),
        )
        .reset_index()
        .sort_values("album_order")
        .reset_index(drop=True)
    )

    centroids["pca_dispersion"] = np.sqrt(
        centroids[
            "pca_dispersion_x"
        ].fillna(0) ** 2
        + centroids[
            "pca_dispersion_y"
        ].fillna(0) ** 2
    )

    return centroids


def build_distance_table(
    centroids: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
]:
    """Calculate all album-to-album PCA centroid distances."""

    coordinates = centroids[
        [
            "pca_x",
            "pca_y",
        ]
    ].to_numpy()

    distances = pairwise_distances(
        coordinates,
        metric="euclidean",
    )

    records: list[dict[str, object]] = []

    for source_index, source_row in centroids.iterrows():
        for target_index, target_row in centroids.iterrows():
            if source_index == target_index:
                continue

            records.append(
                {
                    "source_album": source_row["album"],
                    "source_order": int(
                        source_row["album_order"]
                    ),
                    "target_album": target_row["album"],
                    "target_order": int(
                        target_row["album_order"]
                    ),
                    "years_apart": abs(
                        int(
                            target_row[
                                "release_year"
                            ]
                        )
                        - int(
                            source_row[
                                "release_year"
                            ]
                        )
                    ),
                    "album_order_gap": abs(
                        int(
                            target_row[
                                "album_order"
                            ]
                        )
                        - int(
                            source_row[
                                "album_order"
                            ]
                        )
                    ),
                    "pca_centroid_distance": float(
                        distances[
                            source_index,
                            target_index,
                        ]
                    ),
                }
            )

    return (
        pd.DataFrame(records),
        distances,
    )


def build_nearest_neighbours(
    distance_table: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """Find each album's nearest neighbours in PCA centroid space."""

    return (
        distance_table.sort_values(
            [
                "source_album",
                "pca_centroid_distance",
            ]
        )
        .groupby(
            "source_album",
            group_keys=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def build_turning_points(
    transitions: pd.DataFrame,
    drifts: pd.DataFrame,
) -> pd.DataFrame:
    """Create a ranked table of chronological turning points."""

    top_drifts = (
        drifts.sort_values(
            [
                "from_album",
                "to_album",
                "absolute_change",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .groupby(
            [
                "from_album",
                "to_album",
            ],
            group_keys=False,
        )
        .head(5)
    )

    drift_summary = (
        top_drifts.groupby(
            [
                "from_album",
                "to_album",
            ]
        )
        .apply(
            lambda group: " | ".join(
                (
                    f"{row['feature']}:"
                    f"{row['standardized_change']:+.3f}"
                )
                for _, row
                in group.iterrows()
            ),
            include_groups=False,
        )
        .reset_index(
            name="top_feature_drifts"
        )
    )

    turning_points = transitions.merge(
        drift_summary,
        on=[
            "from_album",
            "to_album",
        ],
        how="left",
        validate="one_to_one",
    )

    turning_points["distance_rank"] = (
        turning_points[
            "profile_distance"
        ]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    turning_points["velocity_rank"] = (
        turning_points[
            "evolution_velocity"
        ]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    turning_points["turning_point_score"] = (
        0.6
        * turning_points[
            "profile_distance"
        ].rank(
            pct=True
        )
        + 0.4
        * turning_points[
            "evolution_velocity"
        ].rank(
            pct=True
        )
    )

    return (
        turning_points.sort_values(
            "turning_point_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_story_table(
    centroids: pd.DataFrame,
    neighbours: pd.DataFrame,
    turning_points: pd.DataFrame,
    album_evolution: pd.DataFrame,
) -> pd.DataFrame:
    """Build one chronological album-story table."""

    nearest_lookup = (
        neighbours.sort_values(
            [
                "source_album",
                "pca_centroid_distance",
            ]
        )
        .groupby(
            "source_album"
        )
        .first()
    )

    transition_lookup = (
        turning_points.set_index(
            "to_album"
        )
    )

    story = centroids.merge(
        album_evolution,
        on=[
            "album_order",
            "album",
            "release_year",
            "era",
        ],
        how="left",
        validate="one_to_one",
    )

    story["nearest_album"] = (
        story["album"]
        .map(
            nearest_lookup[
                "target_album"
            ]
        )
    )

    story[
        "nearest_album_distance"
    ] = (
        story["album"]
        .map(
            nearest_lookup[
                "pca_centroid_distance"
            ]
        )
    )

    story["previous_album"] = (
        story["album"]
        .map(
            transition_lookup[
                "from_album"
            ]
        )
    )

    story[
        "distance_from_previous"
    ] = (
        story["album"]
        .map(
            transition_lookup[
                "profile_distance"
            ]
        )
    )

    story[
        "velocity_from_previous"
    ] = (
        story["album"]
        .map(
            transition_lookup[
                "evolution_velocity"
            ]
        )
    )

    story[
        "largest_feature_shifts"
    ] = (
        story["album"]
        .map(
            transition_lookup[
                "largest_feature_shifts"
            ]
        )
    )

    return story


def build_story_cards(
    story: pd.DataFrame,
    turning_points: pd.DataFrame,
) -> pd.DataFrame:
    """Create concise human-readable story cards for each album."""

    turning_lookup = (
        turning_points.set_index(
            "to_album"
        )
    )

    cards: list[dict[str, object]] = []

    for _, row in story.iterrows():
        album = row["album"]

        if int(
            row["album_order"]
        ) == 1:
            transition_text = (
                "Starting point of the studio-album timeline."
            )
        else:
            transition = turning_lookup.loc[
                album
            ]

            transition_text = (
                f"Moved {transition['profile_distance']:.2f} "
                f"standardized units from {transition['from_album']} "
                f"at {transition['evolution_velocity']:.2f} units/year."
            )

        nearest_text = (
            f"Closest album in PCA space: "
            f"{row['nearest_album']} "
            f"(distance {row['nearest_album_distance']:.2f})."
        )

        cohesion_text = (
            f"Album centroid dispersion: "
            f"{row['pca_dispersion']:.2f}."
        )

        cards.append(
            {
                "album_order": row["album_order"],
                "album": album,
                "release_year": row["release_year"],
                "era": row["era"],
                "headline": (
                    f"{int(row['album_order'])}. "
                    f"{album} ({int(row['release_year'])})"
                ),
                "transition_story": transition_text,
                "nearest_neighbour_story": nearest_text,
                "cohesion_story": cohesion_text,
                "key_shift_story": (
                    row[
                        "largest_feature_shifts"
                    ]
                    if pd.notna(
                        row[
                            "largest_feature_shifts"
                        ]
                    )
                    else ""
                ),
            }
        )

    return pd.DataFrame(cards)


def plot_story_space(
    centroids: pd.DataFrame,
    x_column: str,
    y_column: str,
    method_name: str,
    output_path: Path,
) -> None:
    """Plot album centroids with chronological arrows."""

    figure, axis = plt.subplots(
        figsize=(12, 9)
    )

    axis.plot(
        centroids[x_column],
        centroids[y_column],
        marker="o",
    )

    for index, row in centroids.iterrows():
        axis.annotate(
            (
                f"{int(row['album_order'])}. "
                f"{row['album']}"
            ),
            (
                row[x_column],
                row[y_column],
            ),
            fontsize=9,
        )

        if index > 0:
            previous = centroids.iloc[
                index - 1
            ]

            axis.annotate(
                "",
                xy=(
                    row[x_column],
                    row[y_column],
                ),
                xytext=(
                    previous[x_column],
                    previous[y_column],
                ),
                arrowprops={
                    "arrowstyle": "->",
                    "linewidth": 1.2,
                },
            )

    axis.set_title(
        f"Linkin Park Album Embedding Story — {method_name}"
    )

    axis.set_xlabel(
        f"{method_name} dimension 1"
    )

    axis.set_ylabel(
        f"{method_name} dimension 2"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=230,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_distance_matrix(
    centroids: pd.DataFrame,
    distances: np.ndarray,
) -> None:
    """Plot album-to-album PCA centroid distances."""

    figure, axis = plt.subplots(
        figsize=(10, 9)
    )

    image = axis.imshow(
        distances,
        aspect="auto",
        interpolation="nearest",
    )

    labels = centroids[
        "album"
    ].tolist()

    axis.set_xticks(
        np.arange(
            len(labels)
        )
    )

    axis.set_yticks(
        np.arange(
            len(labels)
        )
    )

    axis.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(
        labels
    )

    for row_index in range(
        len(labels)
    ):
        for column_index in range(
            len(labels)
        ):
            axis.text(
                column_index,
                row_index,
                f"{distances[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    axis.set_title(
        "Album Centroid Distance Matrix — Hybrid PCA Space"
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Euclidean distance",
    )

    figure.tight_layout()

    figure.savefig(
        DISTANCE_MATRIX_PATH,
        dpi=230,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_storyline(
    story: pd.DataFrame,
) -> None:
    """Create a narrative timeline combining distance and dispersion."""

    figure, axis = plt.subplots(
        figsize=(14, 7)
    )

    axis.plot(
        story["album_order"],
        story[
            "distance_from_previous"
        ],
        marker="o",
        label="Distance from previous album",
    )

    axis.plot(
        story["album_order"],
        story[
            "pca_dispersion"
        ],
        marker="s",
        label="Within-album dispersion",
    )

    axis.set_title(
        "Linkin Park Album Storyline: Change vs Internal Diversity"
    )

    axis.set_xlabel(
        "Studio album"
    )

    axis.set_ylabel(
        "Distance / dispersion"
    )

    axis.set_xticks(
        story["album_order"]
    )

    axis.set_xticklabels(
        story["album"],
        rotation=35,
        ha="right",
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        STORYLINE_PATH,
        dpi=230,
        bbox_inches="tight",
    )

    plt.close(figure)


def print_findings(
    neighbours: pd.DataFrame,
    turning_points: pd.DataFrame,
    story: pd.DataFrame,
) -> None:
    """Print the main embedding-story findings."""

    strongest_turn = turning_points.iloc[
        0
    ]

    far_apart_close = (
        neighbours[
            neighbours[
                "album_order_gap"
            ] >= 3
        ]
        .sort_values(
            "pca_centroid_distance"
        )
    )

    most_cohesive = (
        story.sort_values(
            "pca_dispersion"
        )
        .iloc[0]
    )

    most_diverse = (
        story.sort_values(
            "pca_dispersion",
            ascending=False,
        )
        .iloc[0]
    )

    print(
        "\nAlbum embedding story findings:"
    )

    print(
        "- Strongest chronological turning point: "
        f"{strongest_turn['from_album']} → "
        f"{strongest_turn['to_album']} "
        f"(score {strongest_turn['turning_point_score']:.3f})"
    )

    if not far_apart_close.empty:
        row = far_apart_close.iloc[0]

        print(
            "- Closest non-adjacent album pair: "
            f"{row['source_album']} ↔ "
            f"{row['target_album']} "
            f"(distance {row['pca_centroid_distance']:.3f})"
        )

    print(
        "- Most internally cohesive album in PCA space: "
        f"{most_cohesive['album']} "
        f"({most_cohesive['pca_dispersion']:.3f})"
    )

    print(
        "- Most internally diverse album in PCA space: "
        f"{most_diverse['album']} "
        f"({most_diverse['pca_dispersion']:.3f})"
    )

    print(
        "\nNearest album neighbours:"
    )

    for album, group in neighbours.groupby(
        "source_album",
        sort=False,
    ):
        nearest = group.iloc[0]

        print(
            f"- {album}: "
            f"{nearest['target_album']} "
            f"({nearest['pca_centroid_distance']:.3f})"
        )


def validate_outputs(
    centroids: pd.DataFrame,
    story: pd.DataFrame,
) -> None:
    """Run critical output checks."""

    if len(centroids) != 8:
        raise ValueError(
            f"Expected 8 album centroids, found {len(centroids)}."
        )

    if len(story) != 8:
        raise ValueError(
            f"Expected 8 story rows, found {len(story)}."
        )

    if centroids["album"].duplicated().any():
        raise ValueError(
            "Duplicate album names found in centroids."
        )


def main() -> None:
    """Build the complete album embedding story."""

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        embeddings,
        transitions,
        drifts,
        album_evolution,
    ) = load_inputs()

    centroids = build_album_centroids(
        embeddings
    )

    (
        distance_table,
        distance_matrix,
    ) = build_distance_table(
        centroids
    )

    neighbours = build_nearest_neighbours(
        distance_table
    )

    turning_points = build_turning_points(
        transitions=transitions,
        drifts=drifts,
    )

    story = build_story_table(
        centroids=centroids,
        neighbours=neighbours,
        turning_points=turning_points,
        album_evolution=album_evolution,
    )

    story_cards = build_story_cards(
        story=story,
        turning_points=turning_points,
    )

    validate_outputs(
        centroids=centroids,
        story=story,
    )

    story.to_csv(
        STORY_PATH,
        index=False,
        encoding="utf-8",
    )

    neighbours.to_csv(
        NEIGHBOURS_PATH,
        index=False,
        encoding="utf-8",
    )

    turning_points.to_csv(
        TURNING_POINTS_PATH,
        index=False,
        encoding="utf-8",
    )

    story_cards.to_csv(
        STORY_CARDS_PATH,
        index=False,
        encoding="utf-8",
    )

    plot_story_space(
        centroids=centroids,
        x_column="pca_x",
        y_column="pca_y",
        method_name="PCA",
        output_path=PCA_FIGURE_PATH,
    )

    plot_story_space(
        centroids=centroids,
        x_column="umap_x",
        y_column="umap_y",
        method_name="UMAP",
        output_path=UMAP_FIGURE_PATH,
    )

    plot_story_space(
        centroids=centroids,
        x_column="tsne_x",
        y_column="tsne_y",
        method_name="t-SNE",
        output_path=TSNE_FIGURE_PATH,
    )

    plot_distance_matrix(
        centroids=centroids,
        distances=distance_matrix,
    )

    plot_storyline(
        story
    )

    print(
        "\nAlbum embedding story:"
    )

    print(
        story[
            [
                "album_order",
                "album",
                "release_year",
                "era",
                "nearest_album",
                "nearest_album_distance",
                "distance_from_previous",
                "velocity_from_previous",
                "pca_dispersion",
            ]
        ].to_string(
            index=False
        )
    )

    print_findings(
        neighbours=neighbours,
        turning_points=turning_points,
        story=story,
    )

    print("\nSaved:")
    print(f"- {STORY_PATH}")
    print(f"- {NEIGHBOURS_PATH}")
    print(f"- {TURNING_POINTS_PATH}")
    print(f"- {STORY_CARDS_PATH}")
    print(f"- {PCA_FIGURE_PATH}")
    print(f"- {UMAP_FIGURE_PATH}")
    print(f"- {TSNE_FIGURE_PATH}")
    print(f"- {DISTANCE_MATRIX_PATH}")
    print(f"- {STORYLINE_PATH}")

    print("\nAlbum embedding story completed.")


if __name__ == "__main__":
    main()
