"""Build a Linkin Park song-similarity network and export it for Gephi.

Inputs
------
data/processed/song_similarity_pairs.parquet
data/processed/master_dataset.parquet

Outputs
-------
data/processed/song_network_nodes.parquet
data/processed/song_network_edges.parquet
data/processed/song_network_metrics.parquet
data/processed/song_network_nodes.csv
data/processed/song_network_edges.csv
data/processed/song_network_metrics.csv

outputs/tables/network_model_summary.csv
outputs/tables/network_hub_songs.csv
outputs/tables/network_bridge_songs.csv
outputs/tables/network_communities.csv
outputs/tables/network_album_interactions.csv
outputs/tables/network_era_interactions.csv
outputs/tables/network_mst_edges.csv

outputs/figures/song_similarity_network.png
outputs/figures/song_similarity_network_communities.png
outputs/figures/song_network_mst.png
outputs/figures/album_interaction_network.png
outputs/figures/era_interaction_network.png

outputs/gephi/linkin_park_song_network.gexf
outputs/gephi/linkin_park_song_network.graphml
outputs/gephi/linkin_park_song_network_nodes.csv
outputs/gephi/linkin_park_song_network_edges.csv

Purpose
-------
This script converts pairwise song similarity into a sparse graph where:

- each node is a Linkin Park song,
- each edge represents a sufficiently strong hybrid similarity,
- edge weights store similarity strength,
- node metrics identify hubs, bridges, authorities, and communities.

Network metrics
---------------
- degree and weighted degree
- PageRank
- eigenvector centrality
- betweenness centrality
- closeness centrality
- harmonic centrality
- clustering coefficient
- k-core number
- community membership
- bridge score
- hub score

Graph construction
------------------
The graph uses a union of two rules:

1. Mutual/nearest-neighbour rule:
   keep each track's top-k hybrid neighbours.

2. Minimum similarity rule:
   keep any pair whose hybrid similarity exceeds the configured threshold.

This prevents a few dense regions from dominating while retaining important
local relationships across albums and eras.

Notes
-----
- The similarity table contains directed pairs; this script converts them
  into one undirected edge per song pair.
- Hybrid similarity may use two or three underlying views depending on audio
  coverage. The edge table retains this information.
- Negative cosine similarities are excluded.
- The minimum spanning tree is built on distance = 1 - similarity so that
  stronger similarities are preferred.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import networkx as nx
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

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
GEPHI_DIR = PROJECT_ROOT / "outputs" / "gephi"

NODES_PARQUET_PATH = (
    PROCESSED_DIR
    / "song_network_nodes.parquet"
)

EDGES_PARQUET_PATH = (
    PROCESSED_DIR
    / "song_network_edges.parquet"
)

METRICS_PARQUET_PATH = (
    PROCESSED_DIR
    / "song_network_metrics.parquet"
)

NODES_CSV_PATH = (
    PROCESSED_DIR
    / "song_network_nodes.csv"
)

EDGES_CSV_PATH = (
    PROCESSED_DIR
    / "song_network_edges.csv"
)

METRICS_CSV_PATH = (
    PROCESSED_DIR
    / "song_network_metrics.csv"
)

MODEL_SUMMARY_PATH = (
    TABLES_DIR
    / "network_model_summary.csv"
)

HUBS_PATH = (
    TABLES_DIR
    / "network_hub_songs.csv"
)

BRIDGES_PATH = (
    TABLES_DIR
    / "network_bridge_songs.csv"
)

COMMUNITIES_PATH = (
    TABLES_DIR
    / "network_communities.csv"
)

ALBUM_INTERACTIONS_PATH = (
    TABLES_DIR
    / "network_album_interactions.csv"
)

ERA_INTERACTIONS_PATH = (
    TABLES_DIR
    / "network_era_interactions.csv"
)

MST_EDGES_PATH = (
    TABLES_DIR
    / "network_mst_edges.csv"
)

NETWORK_FIGURE_PATH = (
    FIGURES_DIR
    / "song_similarity_network.png"
)

COMMUNITY_FIGURE_PATH = (
    FIGURES_DIR
    / "song_similarity_network_communities.png"
)

MST_FIGURE_PATH = (
    FIGURES_DIR
    / "song_network_mst.png"
)

ALBUM_NETWORK_FIGURE_PATH = (
    FIGURES_DIR
    / "album_interaction_network.png"
)

ERA_NETWORK_FIGURE_PATH = (
    FIGURES_DIR
    / "era_interaction_network.png"
)

GEXF_PATH = (
    GEPHI_DIR
    / "linkin_park_song_network.gexf"
)

GRAPHML_PATH = (
    GEPHI_DIR
    / "linkin_park_song_network.graphml"
)

GEPHI_NODES_PATH = (
    GEPHI_DIR
    / "linkin_park_song_network_nodes.csv"
)

GEPHI_EDGES_PATH = (
    GEPHI_DIR
    / "linkin_park_song_network_edges.csv"
)


SIMILARITY_COLUMN = "hybrid_similarity"
TOP_K_NEIGHBOURS = 5
MIN_SIMILARITY = 0.32
RANDOM_STATE = 42
TOP_RESULTS = 15


NODE_COLUMNS = [
    "master_track_id",
    "album_order",
    "album",
    "release_year",
    "track_position",
    "track_title",
    "canonical_title",
    "era",
    "is_instrumental",
    "has_lyrics_features",
    "has_audio_features",
]


def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:
    """Raise a clear error when required columns are absent."""

    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{label} is missing required columns: "
            f"{sorted(missing)}"
        )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pairwise similarities and canonical node metadata."""

    if not PAIRS_PATH.exists():
        raise FileNotFoundError(
            f"Similarity pairs not found: {PAIRS_PATH}\n"
            "Run 02_song_similarity.py first."
        )

    if not MASTER_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Master dataset not found: {MASTER_DATASET_PATH}"
        )

    pairs = pd.read_parquet(PAIRS_PATH)
    master = pd.read_parquet(MASTER_DATASET_PATH)

    require_columns(
        pairs,
        [
            "source_track_id",
            "target_track_id",
            SIMILARITY_COLUMN,
            "emotion_similarity",
            "lyrics_style_similarity",
            "audio_similarity",
            "hybrid_models_used",
        ],
        "Similarity pairs",
    )

    require_columns(
        master,
        NODE_COLUMNS,
        "Master dataset",
    )

    if master["master_track_id"].duplicated().any():
        raise ValueError(
            "Master dataset contains duplicate track IDs."
        )

    return pairs, master


def build_sparse_edges(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Convert directed similarities into a sparse undirected edge table."""

    usable = pairs[
        pairs[SIMILARITY_COLUMN].notna()
        & (pairs[SIMILARITY_COLUMN] > 0)
    ].copy()

    usable["source_rank"] = (
        usable.groupby(
            "source_track_id"
        )[SIMILARITY_COLUMN]
        .rank(
            method="first",
            ascending=False,
        )
    )

    usable["selected_top_k"] = (
        usable["source_rank"]
        <= TOP_K_NEIGHBOURS
    )

    usable["selected_threshold"] = (
        usable[SIMILARITY_COLUMN]
        >= MIN_SIMILARITY
    )

    selected = usable[
        usable["selected_top_k"]
        | usable["selected_threshold"]
    ].copy()

    selected["node_a"] = selected[
        [
            "source_track_id",
            "target_track_id",
        ]
    ].min(axis=1)

    selected["node_b"] = selected[
        [
            "source_track_id",
            "target_track_id",
        ]
    ].max(axis=1)

    edge_records: list[dict[str, Any]] = []

    for (node_a, node_b), group in selected.groupby(
        ["node_a", "node_b"],
        sort=False,
    ):
        similarity_row = group.loc[
            group[SIMILARITY_COLUMN].idxmax()
        ]

        reciprocal = {
            str(row["source_track_id"]): int(
                bool(row["selected_top_k"])
            )
            for _, row in group.iterrows()
        }

        edge_records.append(
            {
                "source": str(node_a),
                "target": str(node_b),
                "weight": float(
                    similarity_row[
                        SIMILARITY_COLUMN
                    ]
                ),
                "distance": float(
                    1.0
                    - similarity_row[
                        SIMILARITY_COLUMN
                    ]
                ),
                "emotion_similarity": (
                    float(
                        similarity_row[
                            "emotion_similarity"
                        ]
                    )
                    if pd.notna(
                        similarity_row[
                            "emotion_similarity"
                        ]
                    )
                    else np.nan
                ),
                "lyrics_style_similarity": (
                    float(
                        similarity_row[
                            "lyrics_style_similarity"
                        ]
                    )
                    if pd.notna(
                        similarity_row[
                            "lyrics_style_similarity"
                        ]
                    )
                    else np.nan
                ),
                "audio_similarity": (
                    float(
                        similarity_row[
                            "audio_similarity"
                        ]
                    )
                    if pd.notna(
                        similarity_row[
                            "audio_similarity"
                        ]
                    )
                    else np.nan
                ),
                "hybrid_models_used": int(
                    similarity_row[
                        "hybrid_models_used"
                    ]
                ),
                "selected_by_threshold": bool(
                    group[
                        "selected_threshold"
                    ].any()
                ),
                "selected_by_top_k": bool(
                    group[
                        "selected_top_k"
                    ].any()
                ),
                "mutual_top_k": bool(
                    reciprocal.get(
                        str(node_a),
                        0,
                    )
                    and reciprocal.get(
                        str(node_b),
                        0,
                    )
                ),
            }
        )

    edges = pd.DataFrame(edge_records)

    if edges.empty:
        raise ValueError(
            "No graph edges survived the selection rules."
        )

    return (
        edges.sort_values(
            "weight",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> nx.Graph:
    """Build a weighted undirected NetworkX graph."""

    graph = nx.Graph()

    for _, row in nodes.iterrows():
        node_id = str(
            row["master_track_id"]
        )

        attributes = {
            column: (
                row[column].item()
                if hasattr(
                    row[column],
                    "item",
                )
                else row[column]
            )
            for column in NODE_COLUMNS
            if column != "master_track_id"
        }

        # GEXF/GraphML do not handle pandas NA consistently.
        attributes = {
            key: (
                ""
                if pd.isna(value)
                else value
            )
            for key, value in attributes.items()
        }

        graph.add_node(
            node_id,
            **attributes,
        )

    for _, row in edges.iterrows():
        graph.add_edge(
            str(row["source"]),
            str(row["target"]),
            weight=float(
                row["weight"]
            ),
            distance=float(
                row["distance"]
            ),
            emotion_similarity=(
                float(
                    row[
                        "emotion_similarity"
                    ]
                )
                if pd.notna(
                    row[
                        "emotion_similarity"
                    ]
                )
                else 0.0
            ),
            lyrics_style_similarity=(
                float(
                    row[
                        "lyrics_style_similarity"
                    ]
                )
                if pd.notna(
                    row[
                        "lyrics_style_similarity"
                    ]
                )
                else 0.0
            ),
            audio_similarity=(
                float(
                    row[
                        "audio_similarity"
                    ]
                )
                if pd.notna(
                    row[
                        "audio_similarity"
                    ]
                )
                else 0.0
            ),
            hybrid_models_used=int(
                row["hybrid_models_used"]
            ),
            mutual_top_k=bool(
                row["mutual_top_k"]
            ),
        )

    return graph


def detect_communities(
    graph: nx.Graph,
) -> dict[str, int]:
    """Detect weighted communities with Louvain when available."""

    try:
        communities = nx.community.louvain_communities(
            graph,
            weight="weight",
            resolution=1.0,
            seed=RANDOM_STATE,
        )

    except AttributeError:
        communities = (
            nx.community.greedy_modularity_communities(
                graph,
                weight="weight",
            )
        )

    mapping: dict[str, int] = {}

    ordered = sorted(
        communities,
        key=len,
        reverse=True,
    )

    for community_id, members in enumerate(
        ordered,
        start=1,
    ):
        for node_id in members:
            mapping[str(node_id)] = (
                community_id
            )

    return mapping


def safe_eigenvector_centrality(
    graph: nx.Graph,
) -> dict[str, float]:
    """Calculate eigenvector centrality with a robust fallback."""

    try:
        return nx.eigenvector_centrality(
            graph,
            max_iter=5000,
            tol=1e-10,
            weight="weight",
        )

    except nx.PowerIterationFailedConvergence:
        return nx.eigenvector_centrality_numpy(
            graph,
            weight="weight",
        )


def calculate_metrics(
    graph: nx.Graph,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate node-level network metrics and composite scores."""

    degree = dict(
        graph.degree()
    )

    weighted_degree = dict(
        graph.degree(
            weight="weight"
        )
    )

    pagerank = nx.pagerank(
        graph,
        alpha=0.85,
        weight="weight",
    )

    eigenvector = (
        safe_eigenvector_centrality(
            graph
        )
    )

    betweenness = (
        nx.betweenness_centrality(
            graph,
            weight="distance",
            normalized=True,
        )
    )

    closeness = nx.closeness_centrality(
        graph,
        distance="distance",
    )

    harmonic = nx.harmonic_centrality(
        graph,
        distance="distance",
    )

    clustering = nx.clustering(
        graph,
        weight="weight",
    )

    core_number = nx.core_number(
        graph
    )

    communities = detect_communities(
        graph
    )

    metric_df = nodes.copy()

    metric_df["degree"] = (
        metric_df["master_track_id"]
        .astype(str)
        .map(degree)
        .fillna(0)
        .astype(int)
    )

    metric_df["weighted_degree"] = (
        metric_df["master_track_id"]
        .astype(str)
        .map(weighted_degree)
        .fillna(0.0)
    )

    metric_df["pagerank"] = (
        metric_df["master_track_id"]
        .astype(str)
        .map(pagerank)
        .fillna(0.0)
    )

    metric_df["eigenvector_centrality"] = (
        metric_df["master_track_id"]
        .astype(str)
        .map(eigenvector)
        .fillna(0.0)
    )

    metric_df["betweenness_centrality"] = (
        metric_df["master_track_id"]
        .astype(str)
        .map(betweenness)
        .fillna(0.0)
    )

    metric_df["closeness_centrality"] = (
        metric_df["master_track_id"]
        .astype(str)
        .map(closeness)
        .fillna(0.0)
    )

    metric_df["harmonic_centrality"] = (
        metric_df["master_track_id"]
        .astype(str)
        .map(harmonic)
        .fillna(0.0)
    )

    metric_df["clustering_coefficient"] = (
        metric_df["master_track_id"]
        .astype(str)
        .map(clustering)
        .fillna(0.0)
    )

    metric_df["core_number"] = (
        metric_df["master_track_id"]
        .astype(str)
        .map(core_number)
        .fillna(0)
        .astype(int)
    )

    metric_df["community"] = (
        metric_df["master_track_id"]
        .astype(str)
        .map(communities)
        .fillna(0)
        .astype(int)
    )

    # Composite scores are percentile ranks to keep different centrality
    # scales comparable.
    hub_components = [
        "pagerank",
        "eigenvector_centrality",
        "weighted_degree",
        "core_number",
    ]

    bridge_components = [
        "betweenness_centrality",
        "closeness_centrality",
        "harmonic_centrality",
    ]

    hub_percentiles = metric_df[
        hub_components
    ].rank(
        pct=True,
        method="average",
    )

    bridge_percentiles = metric_df[
        bridge_components
    ].rank(
        pct=True,
        method="average",
    )

    metric_df["hub_score"] = (
        hub_percentiles.mean(axis=1)
    )

    metric_df["bridge_score"] = (
        bridge_percentiles.mean(axis=1)
    )

    numeric_columns = metric_df.select_dtypes(
        include="number"
    ).columns

    metric_df[numeric_columns] = (
        metric_df[numeric_columns]
        .round(8)
    )

    return (
        metric_df.sort_values(
            "pagerank",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_summary(
    graph: nx.Graph,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build graph-level summary metrics."""

    components = list(
        nx.connected_components(
            graph
        )
    )

    largest_component = max(
        components,
        key=len,
    )

    largest_subgraph = graph.subgraph(
        largest_component
    ).copy()

    records = [
        {
            "metric": "nodes",
            "value": graph.number_of_nodes(),
        },
        {
            "metric": "edges",
            "value": graph.number_of_edges(),
        },
        {
            "metric": "density",
            "value": nx.density(graph),
        },
        {
            "metric": "connected_components",
            "value": len(components),
        },
        {
            "metric": "largest_component_nodes",
            "value": len(largest_component),
        },
        {
            "metric": "average_degree",
            "value": safe_mean(
                dict(
                    graph.degree()
                ).values()
            ),
        },
        {
            "metric": "average_weighted_degree",
            "value": safe_mean(
                dict(
                    graph.degree(
                        weight="weight"
                    )
                ).values()
            ),
        },
        {
            "metric": "average_clustering",
            "value": nx.average_clustering(
                graph,
                weight="weight",
            ),
        },
        {
            "metric": "communities",
            "value": int(
                metrics["community"]
                .nunique()
            ),
        },
        {
            "metric": "modularity",
            "value": calculate_modularity(
                graph,
                metrics,
            ),
        },
    ]

    if (
        largest_subgraph.number_of_nodes()
        > 1
    ):
        records.extend(
            [
                {
                    "metric": "largest_component_diameter",
                    "value": nx.diameter(
                        largest_subgraph
                    ),
                },
                {
                    "metric": "largest_component_average_shortest_path",
                    "value": (
                        nx.average_shortest_path_length(
                            largest_subgraph,
                            weight="distance",
                        )
                    ),
                },
            ]
        )

    summary = pd.DataFrame(records)

    # pandas 3.x removed errors="ignore" from pd.to_numeric.
    # All summary values are expected to be numeric, so coerce safely.
    summary["value"] = pd.to_numeric(
        summary["value"],
        errors="coerce",
    )

    return summary


def safe_mean(
    values: Iterable[float],
) -> float:
    """Return a safe arithmetic mean."""

    values_list = list(values)

    if not values_list:
        return 0.0

    return float(
        np.mean(values_list)
    )


def calculate_modularity(
    graph: nx.Graph,
    metrics: pd.DataFrame,
) -> float:
    """Calculate modularity from assigned community IDs."""

    communities = []

    for _, group in metrics.groupby(
        "community"
    ):
        communities.append(
            set(
                group[
                    "master_track_id"
                ].astype(str)
            )
        )

    return float(
        nx.community.modularity(
            graph,
            communities,
            weight="weight",
        )
    )


def build_community_table(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize community membership and dominant albums."""

    records: list[dict[str, Any]] = []

    for community_id, group in metrics.groupby(
        "community",
        sort=True,
    ):
        album_counts = (
            group["album"]
            .value_counts()
        )

        era_counts = (
            group["era"]
            .value_counts()
        )

        top_pagerank = (
            group.sort_values(
                "pagerank",
                ascending=False,
            )
            .head(5)
        )

        records.append(
            {
                "community": int(
                    community_id
                ),
                "track_count": len(group),
                "dominant_album": (
                    album_counts.index[0]
                    if not album_counts.empty
                    else None
                ),
                "dominant_album_share": (
                    album_counts.iloc[0]
                    / len(group)
                    if not album_counts.empty
                    else np.nan
                ),
                "dominant_era": (
                    era_counts.index[0]
                    if not era_counts.empty
                    else None
                ),
                "dominant_era_share": (
                    era_counts.iloc[0]
                    / len(group)
                    if not era_counts.empty
                    else np.nan
                ),
                "top_songs_by_pagerank": " | ".join(
                    (
                        f"{row['track_title']} "
                        f"({row['album']})"
                    )
                    for _, row
                    in top_pagerank.iterrows()
                ),
                "albums_present": " | ".join(
                    sorted(
                        group["album"]
                        .dropna()
                        .astype(str)
                        .unique()
                    )
                ),
            }
        )

    return pd.DataFrame(records)


def build_group_interactions(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Aggregate song edges into album- or era-level interactions."""

    metadata = nodes[
        [
            "master_track_id",
            group_column,
        ]
    ].copy()

    source_meta = metadata.rename(
        columns={
            "master_track_id": "source",
            group_column: "source_group",
        }
    )

    target_meta = metadata.rename(
        columns={
            "master_track_id": "target",
            group_column: "target_group",
        }
    )

    enriched = (
        edges.merge(
            source_meta,
            on="source",
            how="left",
            validate="many_to_one",
        )
        .merge(
            target_meta,
            on="target",
            how="left",
            validate="many_to_one",
        )
    )

    enriched["group_a"] = enriched[
        [
            "source_group",
            "target_group",
        ]
    ].min(axis=1)

    enriched["group_b"] = enriched[
        [
            "source_group",
            "target_group",
        ]
    ].max(axis=1)

    interaction = (
        enriched.groupby(
            ["group_a", "group_b"],
            dropna=False,
        )
        .agg(
            edge_count=("weight", "count"),
            average_similarity=("weight", "mean"),
            maximum_similarity=("weight", "max"),
            total_similarity=("weight", "sum"),
        )
        .reset_index()
        .sort_values(
            "total_similarity",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    numeric_columns = interaction.select_dtypes(
        include="number"
    ).columns

    interaction[numeric_columns] = (
        interaction[numeric_columns]
        .round(6)
    )

    return interaction


def build_mst(
    graph: nx.Graph,
    metrics: pd.DataFrame,
) -> tuple[nx.Graph, pd.DataFrame]:
    """Build a minimum spanning forest from similarity distances."""

    mst = nx.minimum_spanning_tree(
        graph,
        weight="distance",
    )

    metadata = metrics.set_index(
        metrics[
            "master_track_id"
        ].astype(str)
    )

    records: list[dict[str, Any]] = []

    for source, target, attributes in mst.edges(
        data=True
    ):
        records.append(
            {
                "source": source,
                "source_track": metadata.loc[
                    source,
                    "track_title",
                ],
                "source_album": metadata.loc[
                    source,
                    "album",
                ],
                "target": target,
                "target_track": metadata.loc[
                    target,
                    "track_title",
                ],
                "target_album": metadata.loc[
                    target,
                    "album",
                ],
                "similarity": float(
                    attributes["weight"]
                ),
                "distance": float(
                    attributes["distance"]
                ),
            }
        )

    return (
        mst,
        pd.DataFrame(records)
        .sort_values(
            "similarity",
            ascending=False,
        )
        .reset_index(drop=True),
    )


def create_gephi_exports(
    graph: nx.Graph,
    metrics: pd.DataFrame,
    edges: pd.DataFrame,
) -> None:
    """Write Gephi-compatible GEXF, GraphML, node, and edge files."""

    metrics_lookup = metrics.set_index(
        metrics[
            "master_track_id"
        ].astype(str)
    )

    graph_for_export = graph.copy()

    metric_columns = [
        "degree",
        "weighted_degree",
        "pagerank",
        "eigenvector_centrality",
        "betweenness_centrality",
        "closeness_centrality",
        "harmonic_centrality",
        "clustering_coefficient",
        "core_number",
        "community",
        "hub_score",
        "bridge_score",
    ]

    for node_id in graph_for_export.nodes:
        row = metrics_lookup.loc[
            str(node_id)
        ]

        graph_for_export.nodes[
            node_id
        ]["label"] = str(
            row["track_title"]
        )

        for column in metric_columns:
            graph_for_export.nodes[
                node_id
            ][column] = (
                float(row[column])
                if column
                not in {
                    "degree",
                    "core_number",
                    "community",
                }
                else int(row[column])
            )

    nx.write_gexf(
        graph_for_export,
        GEXF_PATH,
    )

    nx.write_graphml(
        graph_for_export,
        GRAPHML_PATH,
    )

    gephi_nodes = metrics.rename(
        columns={
            "master_track_id": "Id",
            "track_title": "Label",
        }
    ).copy()

    gephi_nodes.to_csv(
        GEPHI_NODES_PATH,
        index=False,
        encoding="utf-8",
    )

    gephi_edges = edges.rename(
        columns={
            "source": "Source",
            "target": "Target",
            "weight": "Weight",
        }
    ).copy()

    gephi_edges.insert(
        0,
        "Id",
        np.arange(
            1,
            len(gephi_edges) + 1,
        ),
    )

    gephi_edges["Type"] = "Undirected"

    gephi_edges.to_csv(
        GEPHI_EDGES_PATH,
        index=False,
        encoding="utf-8",
    )


def plot_song_network(
    graph: nx.Graph,
    metrics: pd.DataFrame,
    output_path: Path,
    color_by: str,
    title: str,
) -> None:
    """Plot the song network with centrality-scaled nodes."""

    position = nx.spring_layout(
        graph,
        seed=RANDOM_STATE,
        weight="weight",
        k=0.6,
        iterations=300,
    )

    metric_lookup = metrics.set_index(
        metrics[
            "master_track_id"
        ].astype(str)
    )

    node_sizes = [
        250
        + 3500
        * float(
            metric_lookup.loc[
                str(node_id),
                "pagerank",
            ]
        )
        for node_id in graph.nodes
    ]

    if color_by == "community":
        node_colors = [
            int(
                metric_lookup.loc[
                    str(node_id),
                    "community",
                ]
            )
            for node_id in graph.nodes
        ]
    else:
        categories = (
            metrics[color_by]
            .astype("category")
        )

        category_map = {
            value: code
            for code, value in enumerate(
                categories.cat.categories
            )
        }

        node_colors = [
            category_map[
                metric_lookup.loc[
                    str(node_id),
                    color_by,
                ]
            ]
            for node_id in graph.nodes
        ]

    edge_widths = [
        0.5
        + 3.0
        * float(
            attributes["weight"]
        )
        for _, _, attributes
        in graph.edges(data=True)
    ]

    figure, axis = plt.subplots(
        figsize=(16, 12)
    )

    nx.draw_networkx_edges(
        graph,
        position,
        width=edge_widths,
        alpha=0.25,
        ax=axis,
    )

    nodes_artist = nx.draw_networkx_nodes(
        graph,
        position,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.88,
        ax=axis,
    )

    top_labels = (
        metrics.sort_values(
            "pagerank",
            ascending=False,
        )
        .head(18)
    )

    labels = {
        str(row["master_track_id"]): (
            row["track_title"]
        )
        for _, row
        in top_labels.iterrows()
    }

    nx.draw_networkx_labels(
        graph,
        position,
        labels=labels,
        font_size=8,
        ax=axis,
    )

    axis.set_title(title)
    axis.axis("off")

    figure.colorbar(
        nodes_artist,
        ax=axis,
        shrink=0.7,
        label=color_by.replace(
            "_",
            " ",
        ).title(),
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_mst(
    mst: nx.Graph,
    metrics: pd.DataFrame,
) -> None:
    """Plot the minimum spanning forest."""

    position = nx.spring_layout(
        mst,
        seed=RANDOM_STATE,
        weight="weight",
        k=0.8,
        iterations=300,
    )

    metric_lookup = metrics.set_index(
        metrics[
            "master_track_id"
        ].astype(str)
    )

    node_sizes = [
        180
        + 2600
        * float(
            metric_lookup.loc[
                str(node_id),
                "pagerank",
            ]
        )
        for node_id in mst.nodes
    ]

    album_categories = (
        metrics["album"]
        .astype("category")
    )

    album_map = {
        value: code
        for code, value in enumerate(
            album_categories.cat.categories
        )
    }

    node_colors = [
        album_map[
            metric_lookup.loc[
                str(node_id),
                "album",
            ]
        ]
        for node_id in mst.nodes
    ]

    figure, axis = plt.subplots(
        figsize=(16, 12)
    )

    nx.draw_networkx_edges(
        mst,
        position,
        width=1.2,
        alpha=0.5,
        ax=axis,
    )

    nodes_artist = nx.draw_networkx_nodes(
        mst,
        position,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.9,
        ax=axis,
    )

    labels = {
        str(row["master_track_id"]): (
            row["track_title"]
        )
        for _, row
        in metrics.iterrows()
    }

    nx.draw_networkx_labels(
        mst,
        position,
        labels=labels,
        font_size=6,
        ax=axis,
    )

    axis.set_title(
        "Linkin Park Similarity Minimum Spanning Tree"
    )

    axis.axis("off")

    figure.colorbar(
        nodes_artist,
        ax=axis,
        shrink=0.7,
        label="Album code",
    )

    figure.tight_layout()

    figure.savefig(
        MST_FIGURE_PATH,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_group_network(
    interaction: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Plot album- or era-level interaction networks."""

    group_graph = nx.Graph()

    for _, row in interaction.iterrows():
        group_a = str(
            row["group_a"]
        )

        group_b = str(
            row["group_b"]
        )

        if group_a == group_b:
            continue

        group_graph.add_edge(
            group_a,
            group_b,
            weight=float(
                row["total_similarity"]
            ),
            average_similarity=float(
                row["average_similarity"]
            ),
            edge_count=int(
                row["edge_count"]
            ),
        )

    if group_graph.number_of_edges() == 0:
        return

    position = nx.spring_layout(
        group_graph,
        seed=RANDOM_STATE,
        weight="weight",
        k=1.0,
    )

    weighted_degree = dict(
        group_graph.degree(
            weight="weight"
        )
    )

    node_sizes = [
        600
        + 150
        * weighted_degree[node]
        for node in group_graph.nodes
    ]

    max_weight = max(
        attributes["weight"]
        for _, _, attributes
        in group_graph.edges(data=True)
    )

    edge_widths = [
        1.0
        + 7.0
        * attributes["weight"]
        / max_weight
        for _, _, attributes
        in group_graph.edges(data=True)
    ]

    figure, axis = plt.subplots(
        figsize=(12, 9)
    )

    nx.draw_networkx_edges(
        group_graph,
        position,
        width=edge_widths,
        alpha=0.45,
        ax=axis,
    )

    nx.draw_networkx_nodes(
        group_graph,
        position,
        node_size=node_sizes,
        alpha=0.9,
        ax=axis,
    )

    nx.draw_networkx_labels(
        group_graph,
        position,
        font_size=9,
        ax=axis,
    )

    axis.set_title(title)
    axis.axis("off")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=230,
        bbox_inches="tight",
    )

    plt.close(figure)


def validate_outputs(
    graph: nx.Graph,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    """Run critical graph integrity checks."""

    if graph.number_of_nodes() != len(
        nodes
    ):
        raise ValueError(
            "Graph node count does not match node table."
        )

    if graph.number_of_edges() != len(
        edges
    ):
        raise ValueError(
            "Graph edge count does not match edge table."
        )

    if len(metrics) != len(nodes):
        raise ValueError(
            "Network metrics row count does not match node count."
        )

    if edges.duplicated(
        subset=["source", "target"]
    ).any():
        raise ValueError(
            "Duplicate undirected edges found."
        )

    self_edges = edges[
        edges["source"].eq(
            edges["target"]
        )
    ]

    if not self_edges.empty:
        raise ValueError(
            "Self-edges are not allowed."
        )


def print_top_results(
    metrics: pd.DataFrame,
    communities: pd.DataFrame,
) -> None:
    """Print hubs, bridges, PageRank leaders, and communities."""

    print("\nTop PageRank songs:")

    for rank, (_, row) in enumerate(
        metrics.sort_values(
            "pagerank",
            ascending=False,
        )
        .head(TOP_RESULTS)
        .iterrows(),
        start=1,
    ):
        print(
            f"{rank:>2}. "
            f"{row['track_title']} "
            f"— {row['album']} "
            f"({row['pagerank']:.6f})"
        )

    print("\nTop hub songs:")

    for rank, (_, row) in enumerate(
        metrics.sort_values(
            "hub_score",
            ascending=False,
        )
        .head(TOP_RESULTS)
        .iterrows(),
        start=1,
    ):
        print(
            f"{rank:>2}. "
            f"{row['track_title']} "
            f"— {row['album']} "
            f"({row['hub_score']:.4f})"
        )

    print("\nTop bridge songs:")

    for rank, (_, row) in enumerate(
        metrics.sort_values(
            "bridge_score",
            ascending=False,
        )
        .head(TOP_RESULTS)
        .iterrows(),
        start=1,
    ):
        print(
            f"{rank:>2}. "
            f"{row['track_title']} "
            f"— {row['album']} "
            f"({row['bridge_score']:.4f})"
        )

    print("\nCommunities:")
    print(
        communities.to_string(
            index=False
        )
    )


def main() -> None:
    """Build the complete song similarity network."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    GEPHI_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs, master = load_inputs()

    nodes = master[
        NODE_COLUMNS
    ].copy()

    nodes["master_track_id"] = (
        nodes["master_track_id"]
        .astype(str)
    )

    edges = build_sparse_edges(
        pairs
    )

    graph = build_graph(
        nodes=nodes,
        edges=edges,
    )

    metrics = calculate_metrics(
        graph=graph,
        nodes=nodes,
    )

    summary = build_summary(
        graph=graph,
        metrics=metrics,
    )

    communities = build_community_table(
        metrics
    )

    album_interactions = (
        build_group_interactions(
            edges=edges,
            nodes=nodes,
            group_column="album",
        )
    )

    era_interactions = (
        build_group_interactions(
            edges=edges,
            nodes=nodes,
            group_column="era",
        )
    )

    mst, mst_edges = build_mst(
        graph=graph,
        metrics=metrics,
    )

    validate_outputs(
        graph=graph,
        nodes=nodes,
        edges=edges,
        metrics=metrics,
    )

    hub_songs = (
        metrics.sort_values(
            "hub_score",
            ascending=False,
        )
        .head(TOP_RESULTS)
        .reset_index(drop=True)
    )

    bridge_songs = (
        metrics.sort_values(
            "bridge_score",
            ascending=False,
        )
        .head(TOP_RESULTS)
        .reset_index(drop=True)
    )

    nodes.to_parquet(
        NODES_PARQUET_PATH,
        index=False,
    )

    edges.to_parquet(
        EDGES_PARQUET_PATH,
        index=False,
    )

    metrics.to_parquet(
        METRICS_PARQUET_PATH,
        index=False,
    )

    nodes.to_csv(
        NODES_CSV_PATH,
        index=False,
        encoding="utf-8",
    )

    edges.to_csv(
        EDGES_CSV_PATH,
        index=False,
        encoding="utf-8",
    )

    metrics.to_csv(
        METRICS_CSV_PATH,
        index=False,
        encoding="utf-8",
    )

    summary.to_csv(
        MODEL_SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )

    hub_songs.to_csv(
        HUBS_PATH,
        index=False,
        encoding="utf-8",
    )

    bridge_songs.to_csv(
        BRIDGES_PATH,
        index=False,
        encoding="utf-8",
    )

    communities.to_csv(
        COMMUNITIES_PATH,
        index=False,
        encoding="utf-8",
    )

    album_interactions.to_csv(
        ALBUM_INTERACTIONS_PATH,
        index=False,
        encoding="utf-8",
    )

    era_interactions.to_csv(
        ERA_INTERACTIONS_PATH,
        index=False,
        encoding="utf-8",
    )

    mst_edges.to_csv(
        MST_EDGES_PATH,
        index=False,
        encoding="utf-8",
    )

    create_gephi_exports(
        graph=graph,
        metrics=metrics,
        edges=edges,
    )

    plot_song_network(
        graph=graph,
        metrics=metrics,
        output_path=NETWORK_FIGURE_PATH,
        color_by="album",
        title=(
            "Linkin Park Song Similarity Network "
            "— Node Size by PageRank"
        ),
    )

    plot_song_network(
        graph=graph,
        metrics=metrics,
        output_path=COMMUNITY_FIGURE_PATH,
        color_by="community",
        title=(
            "Linkin Park Song Similarity Network "
            "— Louvain Communities"
        ),
    )

    plot_mst(
        mst=mst,
        metrics=metrics,
    )

    plot_group_network(
        interaction=album_interactions,
        output_path=ALBUM_NETWORK_FIGURE_PATH,
        title="Linkin Park Album Interaction Network",
    )

    plot_group_network(
        interaction=era_interactions,
        output_path=ERA_NETWORK_FIGURE_PATH,
        title="Linkin Park Era Interaction Network",
    )

    print("\nNetwork model summary:")
    print(
        summary.to_string(
            index=False
        )
    )

    print_top_results(
        metrics=metrics,
        communities=communities,
    )

    print("\nSaved:")
    print(f"- {NODES_PARQUET_PATH}")
    print(f"- {EDGES_PARQUET_PATH}")
    print(f"- {METRICS_PARQUET_PATH}")
    print(f"- {MODEL_SUMMARY_PATH}")
    print(f"- {HUBS_PATH}")
    print(f"- {BRIDGES_PATH}")
    print(f"- {COMMUNITIES_PATH}")
    print(f"- {ALBUM_INTERACTIONS_PATH}")
    print(f"- {ERA_INTERACTIONS_PATH}")
    print(f"- {MST_EDGES_PATH}")
    print(f"- {GEXF_PATH}")
    print(f"- {GRAPHML_PATH}")
    print(f"- {GEPHI_NODES_PATH}")
    print(f"- {GEPHI_EDGES_PATH}")
    print(f"- {NETWORK_FIGURE_PATH}")
    print(f"- {COMMUNITY_FIGURE_PATH}")
    print(f"- {MST_FIGURE_PATH}")
    print(f"- {ALBUM_NETWORK_FIGURE_PATH}")
    print(f"- {ERA_NETWORK_FIGURE_PATH}")

    print("\nNetwork analysis completed.")


if __name__ == "__main__":
    main()
