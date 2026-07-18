"""Build multi-view song similarity models for the Linkin Park catalogue.

Input
-----
data/processed/master_dataset.parquet

Outputs
-------
data/processed/song_similarity_pairs.parquet
data/processed/song_similarity_pairs.csv
outputs/tables/numb_similarity.csv
outputs/tables/song_similarity_feature_manifest.csv
outputs/tables/song_similarity_model_summary.csv
outputs/figures/numb_similarity_top10.png

Models
------
1. emotion:
   VADER sentiment, NRC emotions, and hand-built lyrical themes.

2. lyrics_style:
   Emotion features plus lexical richness, repetition, pronouns,
   readability, and structural properties.

3. audio:
   AcousticBrainz rhythm, tonal, mood, loudness, and spectral features.
   This model covers only tracks with AcousticBrainz data.

4. hybrid:
   A weighted combination of emotion, lyrics style, and audio similarity.
   For pairs where audio is unavailable, the available model weights are
   renormalized rather than treating missing audio as zero similarity.

Notes
-----
- Cosine similarity is calculated after robust preprocessing.
- Numerical features are median-imputed and standardized.
- Constant and nearly empty features are removed.
- Instrumental tracks remain available for audio similarity, but they are
  excluded from lyrics-based models.
- From Zero participates in emotion and lyrics similarity, but not audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_dataset.parquet"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

PAIRS_PARQUET_PATH = (
    PROCESSED_DIR
    / "song_similarity_pairs.parquet"
)

PAIRS_CSV_PATH = (
    PROCESSED_DIR
    / "song_similarity_pairs.csv"
)

NUMB_OUTPUT_PATH = (
    TABLES_DIR
    / "numb_similarity.csv"
)

FEATURE_MANIFEST_PATH = (
    TABLES_DIR
    / "song_similarity_feature_manifest.csv"
)

MODEL_SUMMARY_PATH = (
    TABLES_DIR
    / "song_similarity_model_summary.csv"
)

NUMB_FIGURE_PATH = (
    FIGURES_DIR
    / "numb_similarity_top10.png"
)


HYBRID_WEIGHTS = {
    "emotion_similarity": 0.45,
    "lyrics_style_similarity": 0.30,
    "audio_similarity": 0.25,
}

TOP_N_DEFAULT = 10

IDENTITY_COLUMNS = [
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


EMOTION_FEATURES = [
    "lyrics_vader_negative",
    "lyrics_vader_neutral",
    "lyrics_vader_positive",
    "lyrics_vader_compound",
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
    "lyrics_theme_body_ratio",
    "lyrics_theme_darkness_ratio",
    "lyrics_theme_pain_ratio",
    "lyrics_theme_hope_ratio",
    "lyrics_theme_conflict_ratio",
    "lyrics_theme_isolation_ratio",
    "lyrics_theme_time_ratio",
    "lyrics_negation_ratio",
    "lyrics_absolutist_ratio",
]


LYRICS_STYLE_FEATURES = [
    *EMOTION_FEATURES,
    "lyrics_word_count",
    "lyrics_unique_word_count",
    "lyrics_type_token_ratio",
    "lyrics_root_type_token_ratio",
    "lyrics_corrected_type_token_ratio",
    "lyrics_hapax_ratio",
    "lyrics_dis_legomena_ratio",
    "lyrics_token_repetition_ratio",
    "lyrics_avg_word_length",
    "lyrics_word_length_std",
    "lyrics_long_word_ratio",
    "lyrics_very_long_word_ratio",
    "lyrics_lexical_density_proxy",
    "lyrics_line_count",
    "lyrics_sentence_count",
    "lyrics_avg_words_per_line",
    "lyrics_line_word_count_std",
    "lyrics_avg_words_per_sentence",
    "lyrics_sentence_word_count_std",
    "lyrics_question_mark_count",
    "lyrics_exclamation_mark_count",
    "lyrics_ellipsis_count",
    "lyrics_punctuation_density",
    "lyrics_uppercase_ratio",
    "lyrics_unique_line_count",
    "lyrics_duplicate_line_occurrences",
    "lyrics_duplicate_line_types",
    "lyrics_line_repetition_ratio",
    "lyrics_most_repeated_line_count",
    "lyrics_first_person_singular_ratio",
    "lyrics_first_person_plural_ratio",
    "lyrics_second_person_ratio",
    "lyrics_third_person_ratio",
    "lyrics_top_5_word_share",
    "lyrics_readability_flesch_reading_ease",
    "lyrics_readability_flesch_kincaid_grade",
    "lyrics_readability_gunning_fog",
    "lyrics_readability_smog_index",
    "lyrics_readability_automated_readability_index",
    "lyrics_readability_coleman_liau_index",
    "lyrics_readability_dale_chall_score",
]


AUDIO_FEATURES = [
    "audio_ab_duration_seconds",
    "audio_ab_average_loudness",
    "audio_ab_dynamic_complexity",
    "audio_ab_loudness_ebu128_integrated",
    "audio_ab_loudness_ebu128_loudness_range",
    "audio_ab_spectral_centroid_mean",
    "audio_ab_spectral_centroid_stdev",
    "audio_ab_spectral_entropy_mean",
    "audio_ab_spectral_flux_mean",
    "audio_ab_spectral_rolloff_mean",
    "audio_ab_zero_crossing_rate_mean",
    "audio_ab_bpm",
    "audio_ab_bpm_histogram_first_peak_bpm",
    "audio_ab_beats_count",
    "audio_ab_danceability",
    "audio_ab_onset_rate",
    "audio_ab_key_strength_edma",
    "audio_ab_key_strength_krumhansl",
    "audio_ab_chords_changes_rate",
    "audio_ab_tuning_frequency",
    "audio_ab_highlevel_danceable_probability",
    "audio_ab_mood_acoustic_probability",
    "audio_ab_mood_aggressive_probability",
    "audio_ab_mood_electronic_probability",
    "audio_ab_mood_happy_probability",
    "audio_ab_mood_party_probability",
    "audio_ab_mood_relaxed_probability",
    "audio_ab_mood_sad_probability",
    "audio_ab_voice_probability",
    "audio_ab_instrumental_probability",
]


@dataclass
class SimilarityModelResult:
    """Container for one fitted similarity view."""

    name: str
    track_ids: list[str]
    feature_names: list[str]
    similarity_matrix: np.ndarray
    coverage_count: int


def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    """Raise a clear error for missing required columns."""

    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(
            "Master dataset is missing required columns: "
            f"{sorted(missing)}"
        )


def load_dataset() -> pd.DataFrame:
    """Load and validate the analytical dataset."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Master dataset not found: {INPUT_PATH}\n"
            "Run build_master_dataset.py first."
        )

    df = pd.read_parquet(INPUT_PATH)

    require_columns(
        df,
        IDENTITY_COLUMNS,
    )

    if df["master_track_id"].duplicated().any():
        raise ValueError(
            "master_dataset contains duplicate master_track_id values."
        )

    # The base master table may contain an older instrumental flag.
    # Reconcile it with the derived NLP evidence so Drawbar is excluded
    # from lyrics-based models while remaining eligible for audio similarity.
    if "lyrics_word_count" in df.columns:
        no_lyric_words = (
            pd.to_numeric(
                df["lyrics_word_count"],
                errors="coerce",
            )
            .fillna(0)
            .eq(0)
        )
    else:
        no_lyric_words = pd.Series(
            False,
            index=df.index,
        )

    known_instrumental_title = (
        df["canonical_title"]
        .astype("string")
        .str.casefold()
        .eq("drawbar")
    )

    df["is_instrumental"] = (
        df["is_instrumental"]
        .fillna(False)
        .astype(bool)
        | no_lyric_words
        | known_instrumental_title
    )

    return (
        df.sort_values(
            ["album_order", "track_position"],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def select_existing_numeric_features(
    df: pd.DataFrame,
    requested_features: list[str],
) -> list[str]:
    """Keep available numeric features with useful variation."""

    selected: list[str] = []

    for column in requested_features:
        if column not in df.columns:
            continue

        numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if numeric.notna().sum() < 3:
            continue

        if numeric.nunique(dropna=True) <= 1:
            continue

        selected.append(column)

    return selected


def fit_similarity_model(
    df: pd.DataFrame,
    name: str,
    requested_features: list[str],
    eligibility_mask: pd.Series,
) -> SimilarityModelResult:
    """Fit one standardized cosine-similarity model."""

    model_df = df.loc[
        eligibility_mask
    ].copy()

    feature_names = select_existing_numeric_features(
        model_df,
        requested_features,
    )

    if not feature_names:
        raise ValueError(
            f"No usable features were found for model: {name}"
        )

    matrix = model_df[
        feature_names
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    transformed = pipeline.fit_transform(matrix)

    similarity = cosine_similarity(
        transformed
    )

    # Numerical noise can produce values just outside [-1, 1].
    similarity = np.clip(
        similarity,
        -1.0,
        1.0,
    )

    return SimilarityModelResult(
        name=name,
        track_ids=(
            model_df["master_track_id"]
            .astype(str)
            .tolist()
        ),
        feature_names=feature_names,
        similarity_matrix=similarity,
        coverage_count=len(model_df),
    )


def similarity_to_pairs(
    result: SimilarityModelResult,
) -> pd.DataFrame:
    """Convert a similarity matrix into directed song-pair rows."""

    records: list[dict[str, Any]] = []

    for source_index, source_id in enumerate(
        result.track_ids
    ):
        for target_index, target_id in enumerate(
            result.track_ids
        ):
            if source_id == target_id:
                continue

            records.append(
                {
                    "source_track_id": source_id,
                    "target_track_id": target_id,
                    f"{result.name}_similarity": float(
                        result.similarity_matrix[
                            source_index,
                            target_index,
                        ]
                    ),
                }
            )

    return pd.DataFrame(records)


def merge_similarity_views(
    model_results: list[SimilarityModelResult],
    all_track_ids: list[str],
) -> pd.DataFrame:
    """Merge model views onto the complete directed song-pair universe.

    Some pairs have neither lyrics similarity nor audio similarity, such as
    Drawbar versus a From Zero track. They are still retained with null model
    scores so the output always contains every directed pair.
    """

    pair_records = [
        {
            "source_track_id": source_id,
            "target_track_id": target_id,
        }
        for source_id in all_track_ids
        for target_id in all_track_ids
        if source_id != target_id
    ]

    pairs = pd.DataFrame(pair_records)

    for result in model_results:
        view_df = similarity_to_pairs(result)

        pairs = pairs.merge(
            view_df,
            on=[
                "source_track_id",
                "target_track_id",
            ],
            how="left",
            validate="one_to_one",
        )

    return pairs

def calculate_hybrid_similarity(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate a weighted, missing-aware hybrid similarity."""

    available_columns = [
        column
        for column in HYBRID_WEIGHTS
        if column in pairs.columns
    ]

    weighted_sum = pd.Series(
        0.0,
        index=pairs.index,
    )

    weight_sum = pd.Series(
        0.0,
        index=pairs.index,
    )

    for column in available_columns:
        weight = HYBRID_WEIGHTS[column]
        available = pairs[column].notna()

        weighted_sum.loc[available] += (
            pairs.loc[available, column]
            * weight
        )

        weight_sum.loc[available] += weight

    pairs["hybrid_similarity"] = (
        weighted_sum
        / weight_sum.replace(0, np.nan)
    )

    pairs["hybrid_models_used"] = sum(
        pairs[column].notna().astype(int)
        for column in available_columns
    )

    return pairs


def add_pair_metadata(
    pairs: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach source and target track metadata."""

    metadata = df[
        [
            "master_track_id",
            "album_order",
            "album",
            "release_year",
            "track_position",
            "track_title",
            "canonical_title",
            "era",
            "is_instrumental",
            "has_audio_features",
        ]
    ].copy()

    source_metadata = metadata.rename(
        columns={
            column: f"source_{column}"
            for column in metadata.columns
            if column != "master_track_id"
        }
    ).rename(
        columns={
            "master_track_id": "source_track_id",
        }
    )

    target_metadata = metadata.rename(
        columns={
            column: f"target_{column}"
            for column in metadata.columns
            if column != "master_track_id"
        }
    ).rename(
        columns={
            "master_track_id": "target_track_id",
        }
    )

    pairs = pairs.merge(
        source_metadata,
        on="source_track_id",
        how="left",
        validate="many_to_one",
    )

    pairs = pairs.merge(
        target_metadata,
        on="target_track_id",
        how="left",
        validate="many_to_one",
    )

    return pairs


def add_ranks(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Rank targets within each source song for every model."""

    similarity_columns = [
        column
        for column in [
            "emotion_similarity",
            "lyrics_style_similarity",
            "audio_similarity",
            "hybrid_similarity",
        ]
        if column in pairs.columns
    ]

    for column in similarity_columns:
        rank_column = column.replace(
            "_similarity",
            "_rank",
        )

        pairs[rank_column] = (
            pairs.groupby(
                "source_track_id"
            )[column]
            .rank(
                method="first",
                ascending=False,
                na_option="bottom",
            )
            .astype("Int64")
        )

    return pairs


def build_feature_manifest(
    model_results: list[SimilarityModelResult],
) -> pd.DataFrame:
    """Create a transparent model-to-feature manifest."""

    records: list[dict[str, Any]] = []

    for result in model_results:
        for feature_order, feature_name in enumerate(
            result.feature_names,
            start=1,
        ):
            records.append(
                {
                    "model": result.name,
                    "feature_order": feature_order,
                    "feature_name": feature_name,
                    "track_coverage": result.coverage_count,
                }
            )

    return pd.DataFrame(records)


def build_model_summary(
    model_results: list[SimilarityModelResult],
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize feature count, track coverage, and pair coverage."""

    records: list[dict[str, Any]] = []

    for result in model_results:
        similarity_column = f"{result.name}_similarity"

        records.append(
            {
                "model": result.name,
                "feature_count": len(result.feature_names),
                "track_coverage": result.coverage_count,
                "directed_pair_coverage": int(
                    pairs[similarity_column]
                    .notna()
                    .sum()
                ),
            }
        )

    records.append(
        {
            "model": "hybrid",
            "feature_count": sum(
                len(result.feature_names)
                for result in model_results
            ),
            "track_coverage": int(
                pairs["source_track_id"].nunique()
            ),
            "directed_pair_coverage": int(
                pairs["hybrid_similarity"]
                .notna()
                .sum()
            ),
        }
    )

    return pd.DataFrame(records)


def find_track_id(
    df: pd.DataFrame,
    title: str,
) -> str:
    """Resolve one canonical title to its track ID."""

    matches = df[
        df["canonical_title"]
        .str.casefold()
        .eq(title.casefold())
    ]

    if matches.empty:
        raise ValueError(
            f"Track not found: {title}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Track title is ambiguous: {title}"
        )

    return str(
        matches.iloc[0]["master_track_id"]
    )


def get_top_similar_songs(
    pairs: pd.DataFrame,
    source_track_id: str,
    similarity_column: str,
    top_n: int = TOP_N_DEFAULT,
) -> pd.DataFrame:
    """Return the top N targets for one source song."""

    require_columns(
        pairs,
        [
            "source_track_id",
            "target_track_id",
            similarity_column,
        ],
    )

    result = (
        pairs[
            pairs["source_track_id"]
            .eq(source_track_id)
            & pairs[similarity_column]
            .notna()
        ]
        .sort_values(
            similarity_column,
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    result.insert(
        0,
        "rank",
        np.arange(
            1,
            len(result) + 1,
        ),
    )

    return result


def save_numb_results(
    pairs: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Save Numb's nearest neighbors under each similarity view."""

    numb_id = find_track_id(
        df,
        "Numb",
    )

    model_columns = [
        column
        for column in [
            "emotion_similarity",
            "lyrics_style_similarity",
            "audio_similarity",
            "hybrid_similarity",
        ]
        if column in pairs.columns
    ]

    result_frames: list[pd.DataFrame] = []

    for similarity_column in model_columns:
        model_name = similarity_column.replace(
            "_similarity",
            "",
        )

        ranked = get_top_similar_songs(
            pairs=pairs,
            source_track_id=numb_id,
            similarity_column=similarity_column,
            top_n=TOP_N_DEFAULT,
        )

        ranked.insert(
            1,
            "model",
            model_name,
        )

        selected_columns = [
            "rank",
            "model",
            "source_track_title",
            "source_album",
            "target_track_title",
            "target_album",
            "target_release_year",
            similarity_column,
            "emotion_similarity",
            "lyrics_style_similarity",
            "audio_similarity",
            "hybrid_similarity",
            "hybrid_models_used",
        ]

        selected_columns = list(
            dict.fromkeys(
                column
                for column in selected_columns
                if column in ranked.columns
            )
        )

        result_frames.append(
            ranked[selected_columns]
        )

    numb_results = pd.concat(
        result_frames,
        ignore_index=True,
        sort=False,
    )

    numb_results.to_csv(
        NUMB_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    return numb_results


def plot_numb_hybrid_top10(
    numb_results: pd.DataFrame,
) -> None:
    """Plot the ten strongest hybrid neighbors of Numb."""

    hybrid = numb_results[
        numb_results["model"].eq("hybrid")
    ].copy()

    if hybrid.empty:
        print(
            "Numb figure skipped: no hybrid results were available."
        )
        return

    hybrid = hybrid.sort_values(
        "hybrid_similarity",
        ascending=True,
    )

    labels = (
        hybrid["target_track_title"]
        + " — "
        + hybrid["target_album"]
    )

    figure, axis = plt.subplots(
        figsize=(11, 7)
    )

    axis.barh(
        labels,
        hybrid["hybrid_similarity"],
    )

    axis.set_title(
        "Songs Most Similar to Numb: Hybrid Model"
    )

    axis.set_xlabel(
        "Cosine similarity"
    )

    figure.tight_layout()

    figure.savefig(
        NUMB_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def validate_pairs(
    pairs: pd.DataFrame,
    df: pd.DataFrame,
) -> None:
    """Run critical integrity checks on similarity output."""

    expected_directed_pairs = (
        len(df)
        * (len(df) - 1)
    )

    if len(pairs) != expected_directed_pairs:
        raise ValueError(
            "Unexpected number of directed song pairs. "
            f"Expected {expected_directed_pairs}, found {len(pairs)}."
        )

    duplicate_pairs = pairs.duplicated(
        subset=[
            "source_track_id",
            "target_track_id",
        ]
    )

    if duplicate_pairs.any():
        raise ValueError(
            "Duplicate directed song pairs were found."
        )

    self_pairs = pairs[
        pairs["source_track_id"]
        .eq(pairs["target_track_id"])
    ]

    if not self_pairs.empty:
        raise ValueError(
            "Self-similarity rows should not be present."
        )

    similarity_columns = [
        column
        for column in [
            "emotion_similarity",
            "lyrics_style_similarity",
            "audio_similarity",
            "hybrid_similarity",
        ]
        if column in pairs.columns
    ]

    for column in similarity_columns:
        values = pairs[column].dropna()

        invalid = values[
            (values < -1.000001)
            | (values > 1.000001)
        ]

        if not invalid.empty:
            raise ValueError(
                f"{column} contains values outside [-1, 1]."
            )


def print_numb_summary(
    numb_results: pd.DataFrame,
) -> None:
    """Print Numb's neighbors for each model."""

    print("\nSongs most similar to Numb:")

    for model_name in numb_results["model"].unique():
        model_df = numb_results[
            numb_results["model"].eq(model_name)
        ]

        score_column = (
            f"{model_name}_similarity"
            if model_name != "hybrid"
            else "hybrid_similarity"
        )

        print(f"\n{model_name.upper()} MODEL")

        for _, row in model_df.iterrows():
            print(
                f"{int(row['rank']):>2}. "
                f"{row['target_track_title']} "
                f"— {row['target_album']} "
                f"({row[score_column]:.4f})"
            )


def main() -> None:
    """Fit all similarity views and save pair-level results."""

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

    df = load_dataset()

    lyrics_eligible = (
        df["has_lyrics_features"]
        & ~df["is_instrumental"]
    )

    audio_eligible = (
        df["has_audio_features"]
    )

    print("\nSimilarity eligibility:")
    print(
        "Lyrics/emotion tracks: "
        f"{int(lyrics_eligible.sum())}"
    )
    print(
        "Instrumental tracks excluded from lyrics models: "
        f"{int(df['is_instrumental'].sum())}"
    )
    print(
        "Audio tracks: "
        f"{int(audio_eligible.sum())}"
    )

    model_results = [
        fit_similarity_model(
            df=df,
            name="emotion",
            requested_features=EMOTION_FEATURES,
            eligibility_mask=lyrics_eligible,
        ),
        fit_similarity_model(
            df=df,
            name="lyrics_style",
            requested_features=LYRICS_STYLE_FEATURES,
            eligibility_mask=lyrics_eligible,
        ),
        fit_similarity_model(
            df=df,
            name="audio",
            requested_features=AUDIO_FEATURES,
            eligibility_mask=audio_eligible,
        ),
    ]

    pairs = merge_similarity_views(
        model_results=model_results,
        all_track_ids=(
            df["master_track_id"]
            .astype(str)
            .tolist()
        ),
    )

    pairs = calculate_hybrid_similarity(
        pairs
    )

    pairs = add_pair_metadata(
        pairs=pairs,
        df=df,
    )

    pairs = add_ranks(
        pairs
    )

    validate_pairs(
        pairs=pairs,
        df=df,
    )

    feature_manifest = build_feature_manifest(
        model_results
    )

    model_summary = build_model_summary(
        model_results=model_results,
        pairs=pairs,
    )

    numb_results = save_numb_results(
        pairs=pairs,
        df=df,
    )

    plot_numb_hybrid_top10(
        numb_results
    )

    pairs.to_parquet(
        PAIRS_PARQUET_PATH,
        index=False,
    )

    pairs.to_csv(
        PAIRS_CSV_PATH,
        index=False,
        encoding="utf-8",
    )

    feature_manifest.to_csv(
        FEATURE_MANIFEST_PATH,
        index=False,
        encoding="utf-8",
    )

    model_summary.to_csv(
        MODEL_SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\nSimilarity model summary:")
    print(model_summary.to_string(index=False))

    print_numb_summary(
        numb_results
    )

    print("\nSaved:")
    print(f"- {PAIRS_PARQUET_PATH}")
    print(f"- {PAIRS_CSV_PATH}")
    print(f"- {NUMB_OUTPUT_PATH}")
    print(f"- {FEATURE_MANIFEST_PATH}")
    print(f"- {MODEL_SUMMARY_PATH}")

    if NUMB_FIGURE_PATH.exists():
        print(f"- {NUMB_FIGURE_PATH}")

    print("\nSong similarity analysis completed.")


if __name__ == "__main__":
    main()
