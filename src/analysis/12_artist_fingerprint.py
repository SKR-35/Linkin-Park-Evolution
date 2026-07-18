"""Estimate Linkin Park's stable artist-level feature fingerprint.

Input
-----
data/processed/master_dataset.parquet

Outputs
-------
outputs/tables/artist_fingerprint.csv
outputs/tables/artist_core_features.csv
outputs/tables/artist_stable_features.csv
outputs/tables/artist_variable_features.csv
outputs/tables/artist_album_deviation.csv
outputs/tables/artist_emotional_balance.csv
outputs/tables/artist_bootstrap_confidence.csv

outputs/figures/artist_fingerprint_radar.png
outputs/figures/artist_feature_stability.png
outputs/figures/artist_album_distance.png
outputs/figures/artist_dna_heatmap.png
outputs/figures/artist_feature_bootstrap.png

Purpose
-------
This script asks a different question from the album-level analyses:

    Which measurable traits remain relatively stable across Linkin Park's
    studio albums and therefore form a plausible artist-level fingerprint?

The fingerprint combines:
- prevalence: the feature is meaningfully present across the catalogue,
- stability: album means do not fluctuate excessively,
- ubiquity: the feature is observed across most albums,
- interpretive importance: the feature separates the artist's catalogue
  from a feature-neutral zero baseline in standardized space.

Important limitation
--------------------
This is an internal fingerprint, not a comparative artist classifier.
Without other artists, the script can identify stable catalogue traits but
cannot prove that they are unique to Linkin Park.

Method summary
--------------
1. Exclude instrumental tracks from lyrics-derived features.
2. Aggregate feature means by studio album.
3. Standardize album profiles across the eight albums.
4. Calculate:
   - overall catalogue mean,
   - between-album standard deviation,
   - coefficient of variation,
   - intraclass-style stability proxy,
   - album coverage,
   - bootstrap confidence intervals.
5. Combine normalized components into a fingerprint score.
6. Calculate each album's distance from the artist fingerprint centroid.

Bootstrap
---------
Albums are resampled with replacement. This quantifies how sensitive each
feature's artist-level mean is to the particular set of albums observed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_dataset.parquet"
)

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

FINGERPRINT_PATH = TABLES_DIR / "artist_fingerprint.csv"
CORE_FEATURES_PATH = TABLES_DIR / "artist_core_features.csv"
STABLE_FEATURES_PATH = TABLES_DIR / "artist_stable_features.csv"
VARIABLE_FEATURES_PATH = TABLES_DIR / "artist_variable_features.csv"
ALBUM_DEVIATION_PATH = TABLES_DIR / "artist_album_deviation.csv"
EMOTIONAL_BALANCE_PATH = TABLES_DIR / "artist_emotional_balance.csv"
BOOTSTRAP_PATH = TABLES_DIR / "artist_bootstrap_confidence.csv"

RADAR_FIGURE_PATH = FIGURES_DIR / "artist_fingerprint_radar.png"
STABILITY_FIGURE_PATH = FIGURES_DIR / "artist_feature_stability.png"
ALBUM_DISTANCE_FIGURE_PATH = FIGURES_DIR / "artist_album_distance.png"
DNA_HEATMAP_PATH = FIGURES_DIR / "artist_dna_heatmap.png"
BOOTSTRAP_FIGURE_PATH = FIGURES_DIR / "artist_feature_bootstrap.png"


RANDOM_STATE = 42
BOOTSTRAP_ITERATIONS = 2000
TOP_FEATURES = 20
TOP_RADAR_FEATURES = 10


IDENTITY_COLUMNS = [
    "master_track_id",
    "album_order",
    "album",
    "release_year",
    "track_title",
    "canonical_title",
    "era",
    "is_instrumental",
    "has_lyrics_features",
]


FINGERPRINT_FEATURES = [
    # Sentiment and emotion
    "lyrics_vader_compound",
    "lyrics_vader_negative",
    "lyrics_vader_positive",
    "lyrics_nrc_anger_ratio",
    "lyrics_nrc_anticipation_ratio",
    "lyrics_nrc_disgust_ratio",
    "lyrics_nrc_fear_ratio",
    "lyrics_nrc_joy_ratio",
    "lyrics_nrc_sadness_ratio",
    "lyrics_nrc_surprise_ratio",
    "lyrics_nrc_trust_ratio",
    "lyrics_nrc_positive_ratio",
    "lyrics_nrc_negative_ratio",
    # Themes
    "lyrics_theme_body_ratio",
    "lyrics_theme_darkness_ratio",
    "lyrics_theme_pain_ratio",
    "lyrics_theme_hope_ratio",
    "lyrics_theme_conflict_ratio",
    "lyrics_theme_isolation_ratio",
    "lyrics_theme_time_ratio",
    # Linguistic style
    "lyrics_negation_ratio",
    "lyrics_absolutist_ratio",
    "lyrics_word_count",
    "lyrics_type_token_ratio",
    "lyrics_root_type_token_ratio",
    "lyrics_hapax_ratio",
    "lyrics_dis_legomena_ratio",
    "lyrics_token_repetition_ratio",
    "lyrics_avg_word_length",
    "lyrics_lexical_density_proxy",
    "lyrics_line_count",
    "lyrics_sentence_count",
    "lyrics_avg_words_per_line",
    "lyrics_avg_words_per_sentence",
    "lyrics_punctuation_density",
    "lyrics_line_repetition_ratio",
    "lyrics_first_person_singular_ratio",
    "lyrics_first_person_plural_ratio",
    "lyrics_second_person_ratio",
    "lyrics_third_person_ratio",
    "lyrics_top_5_word_share",
    "lyrics_readability_flesch_reading_ease",
]


RADAR_CANDIDATES = [
    "lyrics_nrc_anger_ratio",
    "lyrics_nrc_fear_ratio",
    "lyrics_nrc_joy_ratio",
    "lyrics_nrc_sadness_ratio",
    "lyrics_nrc_trust_ratio",
    "lyrics_theme_pain_ratio",
    "lyrics_theme_hope_ratio",
    "lyrics_theme_conflict_ratio",
    "lyrics_theme_isolation_ratio",
    "lyrics_negation_ratio",
    "lyrics_line_repetition_ratio",
    "lyrics_type_token_ratio",
]


def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    """Raise a clear error if required columns are missing."""

    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(
            "Master dataset is missing required columns: "
            f"{sorted(missing)}"
        )


def load_dataset() -> pd.DataFrame:
    """Load and validate the master analytical dataset."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Master dataset not found: {INPUT_PATH}"
        )

    df = pd.read_parquet(INPUT_PATH)

    require_columns(df, IDENTITY_COLUMNS)

    if df["master_track_id"].duplicated().any():
        raise ValueError(
            "Master dataset contains duplicate track IDs."
        )

    if "lyrics_word_count" in df.columns:
        no_words = (
            pd.to_numeric(
                df["lyrics_word_count"],
                errors="coerce",
            )
            .fillna(0)
            .eq(0)
        )
    else:
        no_words = pd.Series(False, index=df.index)

    known_instrumental = (
        df["canonical_title"]
        .astype("string")
        .str.casefold()
        .eq("drawbar")
    )

    df["is_instrumental"] = (
        df["is_instrumental"]
        .fillna(False)
        .astype(bool)
        | no_words
        | known_instrumental
    )

    return (
        df.sort_values(
            ["album_order", "track_position"],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def select_features(df: pd.DataFrame) -> list[str]:
    """Keep numeric features with useful variation and sufficient coverage."""

    selected: list[str] = []

    for column in FINGERPRINT_FEATURES:
        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if values.notna().sum() < 20:
            continue

        if values.nunique(dropna=True) <= 1:
            continue

        selected.append(column)

    if not selected:
        raise ValueError(
            "No usable artist fingerprint features were found."
        )

    return selected


def build_album_profiles(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Build raw album-level feature profiles."""

    analysis_df = df[
        df["has_lyrics_features"]
        & ~df["is_instrumental"]
    ].copy()

    return (
        analysis_df.groupby(
            [
                "album_order",
                "album",
                "release_year",
                "era",
            ],
            dropna=False,
        )[features]
        .mean()
        .reset_index()
        .sort_values("album_order")
        .reset_index(drop=True)
    )


def robust_minmax(series: pd.Series) -> pd.Series:
    """Scale values to [0, 1] while handling constant series."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    minimum = values.min()
    maximum = values.max()

    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(
            0.5,
            index=series.index,
            dtype=float,
        )

    return (values - minimum) / (maximum - minimum)


def bootstrap_album_means(
    album_profiles: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Bootstrap artist-level means by resampling albums."""

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    matrix = (
        album_profiles[features]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    n_albums = len(album_profiles)
    bootstrap_means = np.empty(
        (
            BOOTSTRAP_ITERATIONS,
            len(features),
        ),
        dtype=float,
    )

    for iteration in range(
        BOOTSTRAP_ITERATIONS
    ):
        sampled_indices = rng.integers(
            0,
            n_albums,
            size=n_albums,
        )

        sample = matrix[
            sampled_indices
        ]

        bootstrap_means[
            iteration
        ] = np.nanmean(
            sample,
            axis=0,
        )

    records: list[dict[str, float | str]] = []

    for feature_index, feature in enumerate(
        features
    ):
        values = bootstrap_means[
            :,
            feature_index,
        ]

        records.append(
            {
                "feature": feature,
                "bootstrap_mean": float(
                    np.nanmean(values)
                ),
                "bootstrap_std": float(
                    np.nanstd(values, ddof=1)
                ),
                "ci_2_5": float(
                    np.nanpercentile(values, 2.5)
                ),
                "ci_50": float(
                    np.nanpercentile(values, 50)
                ),
                "ci_97_5": float(
                    np.nanpercentile(values, 97.5)
                ),
                "ci_width": float(
                    np.nanpercentile(values, 97.5)
                    - np.nanpercentile(values, 2.5)
                ),
            }
        )

    return pd.DataFrame(records)


def build_fingerprint_table(
    album_profiles: pd.DataFrame,
    features: list[str],
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate stability, ubiquity, and artist fingerprint scores."""

    raw = (
        album_profiles[features]
        .apply(pd.to_numeric, errors="coerce")
    )

    overall_mean = raw.mean(axis=0)
    album_std = raw.std(axis=0, ddof=1)
    album_mad = raw.sub(
        raw.median(axis=0),
        axis=1,
    ).abs().median(axis=0)

    coverage = raw.notna().mean(axis=0)

    absolute_mean = overall_mean.abs()

    coefficient_of_variation = (
        album_std
        / absolute_mean.replace(0, np.nan)
    )

    stability_raw = 1.0 / (
        1.0
        + coefficient_of_variation.replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    stability_raw = stability_raw.fillna(
        1.0 / (1.0 + album_std)
    )

    result = pd.DataFrame(
        {
            "feature": features,
            "catalogue_mean": overall_mean.values,
            "album_std": album_std.values,
            "album_mad": album_mad.values,
            "coefficient_of_variation": (
                coefficient_of_variation.values
            ),
            "album_coverage": coverage.values,
            "stability_raw": stability_raw.values,
        }
    )

    result = result.merge(
        bootstrap,
        on="feature",
        how="left",
        validate="one_to_one",
    )

    result["prevalence_score"] = robust_minmax(
        result["catalogue_mean"].abs()
    )

    result["stability_score"] = robust_minmax(
        result["stability_raw"]
    )

    result["confidence_score"] = (
        1.0
        - robust_minmax(
            result["ci_width"]
        )
    )

    result["ubiquity_score"] = (
        result["album_coverage"]
        .clip(0, 1)
    )

    result["fingerprint_score"] = (
        0.35 * result["stability_score"]
        + 0.25 * result["prevalence_score"]
        + 0.20 * result["confidence_score"]
        + 0.20 * result["ubiquity_score"]
    )

    result["feature_role"] = np.select(
        [
            result["fingerprint_score"] >= 0.75,
            result["stability_score"] >= 0.70,
            result["stability_score"] <= 0.30,
        ],
        [
            "core_fingerprint",
            "stable_trait",
            "album_variable",
        ],
        default="supporting_trait",
    )

    numeric_columns = result.select_dtypes(
        include="number"
    ).columns

    result[numeric_columns] = (
        result[numeric_columns]
        .round(8)
    )

    return (
        result.sort_values(
            "fingerprint_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_album_deviation(
    album_profiles: pd.DataFrame,
    features: list[str],
    fingerprint: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure each album's standardized distance from the artist centroid."""

    matrix = (
        album_profiles[features]
        .apply(pd.to_numeric, errors="coerce")
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    scaler = StandardScaler()

    imputed = imputer.fit_transform(
        matrix
    )

    standardized = scaler.fit_transform(
        imputed
    )

    feature_weights = (
        fingerprint.set_index("feature")
        .loc[features, "fingerprint_score"]
        .to_numpy(dtype=float)
    )

    feature_weights = (
        feature_weights
        / feature_weights.sum()
    )

    artist_centroid = np.average(
        standardized,
        axis=0,
        weights=None,
    )

    differences = (
        standardized
        - artist_centroid
    )

    weighted_squared = (
        np.square(differences)
        * feature_weights
    )

    weighted_distance = np.sqrt(
        weighted_squared.sum(axis=1)
    )

    unweighted_distance = np.linalg.norm(
        differences,
        axis=1,
    )

    deviation = album_profiles[
        [
            "album_order",
            "album",
            "release_year",
            "era",
        ]
    ].copy()

    deviation[
        "weighted_distance_from_artist_fingerprint"
    ] = weighted_distance

    deviation[
        "unweighted_distance_from_artist_centroid"
    ] = unweighted_distance

    strongest_deviations: list[str] = []

    for row_index in range(
        len(album_profiles)
    ):
        contributions = weighted_squared[
            row_index
        ]

        top_indices = np.argsort(
            contributions
        )[::-1][:8]

        strongest_deviations.append(
            " | ".join(
                (
                    f"{features[index]}:"
                    f"{differences[row_index, index]:+.3f}"
                )
                for index in top_indices
            )
        )

    deviation[
        "largest_feature_deviations"
    ] = strongest_deviations

    standardized_profiles = pd.DataFrame(
        standardized,
        columns=features,
    )

    standardized_profiles.insert(
        0,
        "album",
        album_profiles["album"].values,
    )

    numeric_columns = deviation.select_dtypes(
        include="number"
    ).columns

    deviation[numeric_columns] = (
        deviation[numeric_columns]
        .round(6)
    )

    return (
        deviation.sort_values(
            "weighted_distance_from_artist_fingerprint",
            ascending=False,
        )
        .reset_index(drop=True),
        standardized_profiles,
    )


def build_emotional_balance(
    album_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate catalogue-level opposing emotion balances."""

    pairs = [
        (
            "hope_vs_pain",
            "lyrics_theme_hope_ratio",
            "lyrics_theme_pain_ratio",
        ),
        (
            "joy_vs_sadness",
            "lyrics_nrc_joy_ratio",
            "lyrics_nrc_sadness_ratio",
        ),
        (
            "trust_vs_fear",
            "lyrics_nrc_trust_ratio",
            "lyrics_nrc_fear_ratio",
        ),
        (
            "positive_vs_negative",
            "lyrics_nrc_positive_ratio",
            "lyrics_nrc_negative_ratio",
        ),
    ]

    records: list[dict[str, object]] = []

    for balance_name, positive_feature, negative_feature in pairs:
        if (
            positive_feature not in album_profiles.columns
            or negative_feature not in album_profiles.columns
        ):
            continue

        positive_mean = float(
            album_profiles[
                positive_feature
            ].mean()
        )

        negative_mean = float(
            album_profiles[
                negative_feature
            ].mean()
        )

        denominator = (
            abs(positive_mean)
            + abs(negative_mean)
        )

        balance = (
            (positive_mean - negative_mean)
            / denominator
            if denominator > 0
            else 0.0
        )

        records.append(
            {
                "balance": balance_name,
                "positive_feature": positive_feature,
                "negative_feature": negative_feature,
                "positive_mean": positive_mean,
                "negative_mean": negative_mean,
                "normalized_balance": balance,
                "dominant_side": (
                    positive_feature
                    if balance > 0
                    else negative_feature
                    if balance < 0
                    else "balanced"
                ),
            }
        )

    return pd.DataFrame(records)


def plot_radar(
    fingerprint: pd.DataFrame,
) -> None:
    """Plot a compact radar chart of selected fingerprint traits."""

    available = fingerprint[
        fingerprint["feature"].isin(
            RADAR_CANDIDATES
        )
    ].copy()

    available = (
        available.sort_values(
            "fingerprint_score",
            ascending=False,
        )
        .head(TOP_RADAR_FEATURES)
    )

    if len(available) < 3:
        return

    labels = [
        feature.replace(
            "lyrics_",
            "",
        ).replace(
            "_ratio",
            "",
        ).replace(
            "_",
            " ",
        )
        for feature in available["feature"]
    ]

    values = (
        available[
            "fingerprint_score"
        ].to_numpy()
    )

    angles = np.linspace(
        0,
        2 * np.pi,
        len(labels),
        endpoint=False,
    )

    values = np.concatenate(
        [values, values[:1]]
    )

    angles = np.concatenate(
        [angles, angles[:1]]
    )

    figure = plt.figure(
        figsize=(10, 10)
    )

    axis = figure.add_subplot(
        111,
        polar=True,
    )

    axis.plot(
        angles,
        values,
        marker="o",
    )

    axis.fill(
        angles,
        values,
        alpha=0.20,
    )

    axis.set_xticks(
        angles[:-1]
    )

    axis.set_xticklabels(
        labels,
        fontsize=8,
    )

    axis.set_ylim(
        0,
        1,
    )

    axis.set_title(
        "Linkin Park Artist Fingerprint",
        pad=25,
    )

    figure.tight_layout()

    figure.savefig(
        RADAR_FIGURE_PATH,
        dpi=230,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_feature_stability(
    fingerprint: pd.DataFrame,
) -> None:
    """Plot stable and variable catalogue features."""

    plot_df = pd.concat(
        [
            fingerprint.head(12),
            fingerprint.tail(8),
        ]
    ).drop_duplicates(
        subset=["feature"]
    )

    plot_df = plot_df.sort_values(
        "stability_score"
    )

    labels = [
        feature.replace(
            "lyrics_",
            "",
        ).replace(
            "_",
            " ",
        )
        for feature in plot_df["feature"]
    ]

    figure, axis = plt.subplots(
        figsize=(12, 10)
    )

    axis.barh(
        labels,
        plot_df["stability_score"],
    )

    axis.set_title(
        "Artist Feature Stability Across Albums"
    )

    axis.set_xlabel(
        "Stability score"
    )

    figure.tight_layout()

    figure.savefig(
        STABILITY_FIGURE_PATH,
        dpi=230,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_album_distance(
    deviation: pd.DataFrame,
) -> None:
    """Plot each album's distance from the artist fingerprint."""

    plot_df = deviation.sort_values(
        "album_order"
    )

    figure, axis = plt.subplots(
        figsize=(13, 7)
    )

    axis.bar(
        plot_df["album"],
        plot_df[
            "weighted_distance_from_artist_fingerprint"
        ],
    )

    axis.set_title(
        "Album Distance from Linkin Park's Artist Fingerprint"
    )

    axis.set_xlabel(
        "Studio album"
    )

    axis.set_ylabel(
        "Weighted standardized distance"
    )

    axis.tick_params(
        axis="x",
        rotation=35,
    )

    figure.tight_layout()

    figure.savefig(
        ALBUM_DISTANCE_FIGURE_PATH,
        dpi=230,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_dna_heatmap(
    standardized_profiles: pd.DataFrame,
    fingerprint: pd.DataFrame,
) -> None:
    """Plot top fingerprint features across albums."""

    top_features = (
        fingerprint.head(18)[
            "feature"
        ].tolist()
    )

    matrix = (
        standardized_profiles.set_index(
            "album"
        )[top_features]
    )

    figure, axis = plt.subplots(
        figsize=(15, 9)
    )

    image = axis.imshow(
        matrix.values,
        aspect="auto",
        interpolation="nearest",
    )

    axis.set_yticks(
        np.arange(
            len(matrix.index)
        )
    )

    axis.set_yticklabels(
        matrix.index
    )

    axis.set_xticks(
        np.arange(
            len(top_features)
        )
    )

    axis.set_xticklabels(
        [
            feature.replace(
                "lyrics_",
                "",
            ).replace(
                "_",
                " ",
            )
            for feature in top_features
        ],
        rotation=45,
        ha="right",
    )

    axis.set_title(
        "Artist DNA Across Studio Albums"
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Album feature z-score",
    )

    figure.tight_layout()

    figure.savefig(
        DNA_HEATMAP_PATH,
        dpi=230,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_bootstrap(
    fingerprint: pd.DataFrame,
) -> None:
    """Plot bootstrap intervals for top fingerprint features."""

    top = (
        fingerprint.head(15)
        .sort_values(
            "bootstrap_mean"
        )
    )

    labels = [
        feature.replace(
            "lyrics_",
            "",
        ).replace(
            "_",
            " ",
        )
        for feature in top["feature"]
    ]

    lower = (
        top["bootstrap_mean"]
        - top["ci_2_5"]
    )

    upper = (
        top["ci_97_5"]
        - top["bootstrap_mean"]
    )

    figure, axis = plt.subplots(
        figsize=(12, 9)
    )

    axis.errorbar(
        top["bootstrap_mean"],
        labels,
        xerr=np.vstack(
            [lower, upper]
        ),
        fmt="o",
        capsize=4,
    )

    axis.set_title(
        "Bootstrap Confidence Intervals for Core Artist Features"
    )

    axis.set_xlabel(
        "Album-resampled catalogue mean"
    )

    figure.tight_layout()

    figure.savefig(
        BOOTSTRAP_FIGURE_PATH,
        dpi=230,
        bbox_inches="tight",
    )

    plt.close(figure)


def print_findings(
    fingerprint: pd.DataFrame,
    deviation: pd.DataFrame,
    emotional_balance: pd.DataFrame,
) -> None:
    """Print the most important artist fingerprint findings."""

    print(
        "\nCore artist fingerprint features:"
    )

    for _, row in fingerprint.head(15).iterrows():
        print(
            f"- {row['feature']}: "
            f"fingerprint={row['fingerprint_score']:.3f}, "
            f"stability={row['stability_score']:.3f}, "
            f"mean={row['catalogue_mean']:.4f}"
        )

    print(
        "\nMost album-variable features:"
    )

    for _, row in (
        fingerprint.sort_values(
            "stability_score"
        )
        .head(10)
        .iterrows()
    ):
        print(
            f"- {row['feature']}: "
            f"stability={row['stability_score']:.3f}"
        )

    closest = deviation.sort_values(
        "weighted_distance_from_artist_fingerprint"
    ).iloc[0]

    farthest = deviation.sort_values(
        "weighted_distance_from_artist_fingerprint",
        ascending=False,
    ).iloc[0]

    print(
        "\nAlbum relationship to artist fingerprint:"
    )

    print(
        "- Closest album to artist centroid: "
        f"{closest['album']} "
        f"({closest['weighted_distance_from_artist_fingerprint']:.3f})"
    )

    print(
        "- Most distinctive album: "
        f"{farthest['album']} "
        f"({farthest['weighted_distance_from_artist_fingerprint']:.3f})"
    )

    if not emotional_balance.empty:
        print(
            "\nEmotional balances:"
        )

        for _, row in emotional_balance.iterrows():
            print(
                f"- {row['balance']}: "
                f"{row['normalized_balance']:+.3f} "
                f"({row['dominant_side']})"
            )


def validate_outputs(
    album_profiles: pd.DataFrame,
    fingerprint: pd.DataFrame,
) -> None:
    """Run critical structural checks."""

    if len(album_profiles) != 8:
        raise ValueError(
            f"Expected 8 album profiles, found {len(album_profiles)}."
        )

    if fingerprint["feature"].duplicated().any():
        raise ValueError(
            "Duplicate features found in fingerprint table."
        )

    invalid_scores = fingerprint[
        ~fingerprint[
            "fingerprint_score"
        ].between(0, 1)
    ]

    if not invalid_scores.empty:
        raise ValueError(
            "Fingerprint scores outside [0, 1] were found."
        )


def main() -> None:
    """Run the complete artist fingerprint analysis."""

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset()

    features = select_features(
        df
    )

    album_profiles = build_album_profiles(
        df=df,
        features=features,
    )

    bootstrap = bootstrap_album_means(
        album_profiles=album_profiles,
        features=features,
    )

    fingerprint = build_fingerprint_table(
        album_profiles=album_profiles,
        features=features,
        bootstrap=bootstrap,
    )

    (
        album_deviation,
        standardized_profiles,
    ) = build_album_deviation(
        album_profiles=album_profiles,
        features=features,
        fingerprint=fingerprint,
    )

    emotional_balance = build_emotional_balance(
        album_profiles
    )

    validate_outputs(
        album_profiles=album_profiles,
        fingerprint=fingerprint,
    )

    core_features = fingerprint[
        fingerprint["feature_role"].eq(
            "core_fingerprint"
        )
    ].copy()

    stable_features = fingerprint[
        fingerprint["feature_role"].isin(
            [
                "core_fingerprint",
                "stable_trait",
            ]
        )
    ].copy()

    variable_features = (
        fingerprint.sort_values(
            "stability_score"
        )
        .head(TOP_FEATURES)
        .copy()
    )

    fingerprint.to_csv(
        FINGERPRINT_PATH,
        index=False,
        encoding="utf-8",
    )

    core_features.to_csv(
        CORE_FEATURES_PATH,
        index=False,
        encoding="utf-8",
    )

    stable_features.to_csv(
        STABLE_FEATURES_PATH,
        index=False,
        encoding="utf-8",
    )

    variable_features.to_csv(
        VARIABLE_FEATURES_PATH,
        index=False,
        encoding="utf-8",
    )

    album_deviation.to_csv(
        ALBUM_DEVIATION_PATH,
        index=False,
        encoding="utf-8",
    )

    emotional_balance.to_csv(
        EMOTIONAL_BALANCE_PATH,
        index=False,
        encoding="utf-8",
    )

    bootstrap.to_csv(
        BOOTSTRAP_PATH,
        index=False,
        encoding="utf-8",
    )

    plot_radar(
        fingerprint
    )

    plot_feature_stability(
        fingerprint
    )

    plot_album_distance(
        album_deviation
    )

    plot_dna_heatmap(
        standardized_profiles=standardized_profiles,
        fingerprint=fingerprint,
    )

    plot_bootstrap(
        fingerprint
    )

    print(
        "\nArtist fingerprint summary:"
    )

    print(
        f"- Albums: {len(album_profiles)}"
    )

    print(
        f"- Features analyzed: {len(features)}"
    )

    print(
        f"- Core fingerprint features: {len(core_features)}"
    )

    print(
        f"- Bootstrap iterations: {BOOTSTRAP_ITERATIONS}"
    )

    print_findings(
        fingerprint=fingerprint,
        deviation=album_deviation,
        emotional_balance=emotional_balance,
    )

    print("\nSaved:")
    print(f"- {FINGERPRINT_PATH}")
    print(f"- {CORE_FEATURES_PATH}")
    print(f"- {STABLE_FEATURES_PATH}")
    print(f"- {VARIABLE_FEATURES_PATH}")
    print(f"- {ALBUM_DEVIATION_PATH}")
    print(f"- {EMOTIONAL_BALANCE_PATH}")
    print(f"- {BOOTSTRAP_PATH}")
    print(f"- {RADAR_FIGURE_PATH}")
    print(f"- {STABILITY_FIGURE_PATH}")
    print(f"- {ALBUM_DISTANCE_FIGURE_PATH}")
    print(f"- {DNA_HEATMAP_PATH}")
    print(f"- {BOOTSTRAP_FIGURE_PATH}")

    print(
        "\nArtist fingerprint analysis completed."
    )


if __name__ == "__main__":
    main()
