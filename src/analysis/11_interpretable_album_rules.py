"""Build interpretable album-classification rules for Linkin Park songs.

Input
-----
data/processed/master_dataset.parquet

Outputs
-------
outputs/tables/interpretable_album_rules_summary.csv
outputs/tables/interpretable_album_rules_global_importance.csv
outputs/tables/interpretable_album_rules_tree_rules.txt
outputs/tables/interpretable_album_rules_local_explanations.csv
outputs/tables/interpretable_album_rules_album_signatures.csv
outputs/tables/interpretable_album_rules_counterfactuals.csv

outputs/figures/interpretable_album_rules_tree.png
outputs/figures/interpretable_album_rules_global_importance.png
outputs/figures/interpretable_album_rules_album_signatures.png

Purpose
-------
This script explains what makes Linkin Park songs look like they belong to
particular studio albums.

It complements 10_predict_album.py by focusing on interpretation rather than
maximizing predictive performance.

Methods
-------
1. Primary predictive model:
   multinomial logistic regression on eight-album lyrics features.

2. Global interpretation:
   - standardized logistic-regression coefficients
   - permutation importance

3. Human-readable rule model:
   - shallow decision tree trained on the same feature set
   - exported textual rules
   - plotted tree

4. Local explanation:
   - per-song feature contributions to the predicted album
   - strongest supporting and opposing features

5. Album signatures:
   - standardized album-level mean feature profiles
   - strongest positive and negative defining features

6. Simple counterfactual guidance:
   - minimum feature shifts, under a linear approximation, that would move
     a song toward the second-most likely album

Scientific safeguards
---------------------
- Album metadata, year, era, title, and album order are excluded.
- The primary eight-album task uses lyrics-derived features only.
- Instrumental tracks are excluded.
- Interpretations are descriptive, not causal.
- Counterfactuals are linear approximations and not claims about songwriting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import (
    DecisionTreeClassifier,
    export_text,
    plot_tree,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_dataset.parquet"
)

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

SUMMARY_PATH = (
    TABLES_DIR
    / "interpretable_album_rules_summary.csv"
)

GLOBAL_IMPORTANCE_PATH = (
    TABLES_DIR
    / "interpretable_album_rules_global_importance.csv"
)

TREE_RULES_PATH = (
    TABLES_DIR
    / "interpretable_album_rules_tree_rules.txt"
)

LOCAL_EXPLANATIONS_PATH = (
    TABLES_DIR
    / "interpretable_album_rules_local_explanations.csv"
)

ALBUM_SIGNATURES_PATH = (
    TABLES_DIR
    / "interpretable_album_rules_album_signatures.csv"
)

COUNTERFACTUALS_PATH = (
    TABLES_DIR
    / "interpretable_album_rules_counterfactuals.csv"
)

TREE_FIGURE_PATH = (
    FIGURES_DIR
    / "interpretable_album_rules_tree.png"
)

GLOBAL_IMPORTANCE_FIGURE_PATH = (
    FIGURES_DIR
    / "interpretable_album_rules_global_importance.png"
)

ALBUM_SIGNATURES_FIGURE_PATH = (
    FIGURES_DIR
    / "interpretable_album_rules_album_signatures.png"
)


RANDOM_STATE = 42
CV_SPLITS = 5
TOP_GLOBAL_FEATURES = 20
TOP_LOCAL_FEATURES = 8
TOP_SIGNATURE_FEATURES = 8
TREE_MAX_DEPTH = 4
TREE_MIN_SAMPLES_LEAF = 4


IDENTITY_COLUMNS = [
    "master_track_id",
    "album",
    "track_title",
    "canonical_title",
    "era",
    "is_instrumental",
    "has_lyrics_features",
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


def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    """Raise a clear error when required columns are absent."""

    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(
            "Master dataset is missing required columns: "
            f"{sorted(missing)}"
        )


def load_dataset() -> pd.DataFrame:
    """Load and validate the master dataset."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Master dataset not found: {INPUT_PATH}"
        )

    df = pd.read_parquet(INPUT_PATH)

    require_columns(
        df,
        IDENTITY_COLUMNS,
    )

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
        no_words = pd.Series(
            False,
            index=df.index,
        )

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

    return df.reset_index(drop=True)


def select_features(
    df: pd.DataFrame,
) -> list[str]:
    """Select usable lyrics features without leakage."""

    selected: list[str] = []

    for column in LYRICS_STYLE_FEATURES:
        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if values.notna().sum() < 5:
            continue

        if values.nunique(dropna=True) <= 1:
            continue

        selected.append(column)

    if not selected:
        raise ValueError(
            "No usable lyrics features were found."
        )

    return selected


def build_analysis_frame(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
]:
    """Build eligible metadata, X, and y."""

    eligible = (
        df["has_lyrics_features"]
        & ~df["is_instrumental"]
    )

    analysis_df = (
        df.loc[eligible]
        .copy()
        .reset_index(drop=True)
    )

    X = (
        analysis_df[features]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    y = (
        analysis_df["album"]
        .astype(str)
    )

    class_counts = y.value_counts()

    if class_counts.min() < CV_SPLITS:
        raise ValueError(
            "At least one album has too few tracks for "
            f"{CV_SPLITS}-fold CV:\n{class_counts}"
        )

    return analysis_df, X, y


def build_logistic_pipeline() -> Pipeline:
    """Create the interpretable primary model."""

    return Pipeline(
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
            (
                "model",
                LogisticRegression(
                    max_iter=10000,
                    class_weight="balanced",
                    C=0.5,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def create_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
) -> pd.DataFrame:
    """Create deterministic out-of-fold predictions."""

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    predictions = np.empty(
        len(X),
        dtype=int,
    )

    probabilities = np.zeros(
        (
            len(X),
            len(encoder.classes_),
        ),
        dtype=float,
    )

    folds = np.zeros(
        len(X),
        dtype=int,
    )

    for fold_number, (
        train_index,
        test_index,
    ) in enumerate(
        cv.split(X, y_encoded),
        start=1,
    ):
        fitted = build_logistic_pipeline()

        fitted.fit(
            X.iloc[train_index],
            y_encoded[train_index],
        )

        predictions[test_index] = (
            fitted.predict(
                X.iloc[test_index]
            )
        )

        probabilities[test_index] = (
            fitted.predict_proba(
                X.iloc[test_index]
            )
        )

        folds[test_index] = fold_number

    result = pd.DataFrame(
        {
            "fold": folds,
            "actual_album": y.values,
            "predicted_album": (
                encoder.inverse_transform(
                    predictions
                )
            ),
            "prediction_confidence": (
                probabilities.max(axis=1)
            ),
        }
    )

    result["correct"] = (
        result["actual_album"]
        == result["predicted_album"]
    )

    return result


def fit_primary_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> Pipeline:
    """Fit the final logistic model on all eligible tracks."""

    pipeline = build_logistic_pipeline()
    pipeline.fit(X, y)

    return pipeline


def extract_standardized_matrix(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return the imputed and standardized matrix."""

    imputer = pipeline.named_steps[
        "imputer"
    ]

    scaler = pipeline.named_steps[
        "scaler"
    ]

    imputed = imputer.transform(X)
    scaled = scaler.transform(imputed)

    return scaled


def build_global_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
) -> pd.DataFrame:
    """Combine logistic coefficients and permutation importance."""

    model = pipeline.named_steps[
        "model"
    ]

    coefficient_strength = (
        np.abs(model.coef_)
        .mean(axis=0)
    )

    permutation = permutation_importance(
        estimator=pipeline,
        X=X,
        y=y,
        scoring="f1_macro",
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    result = pd.DataFrame(
        {
            "feature": features,
            "mean_absolute_logistic_coefficient": (
                coefficient_strength
            ),
            "permutation_importance_mean": (
                permutation.importances_mean
            ),
            "permutation_importance_std": (
                permutation.importances_std
            ),
        }
    )

    result[
        "coefficient_percentile"
    ] = result[
        "mean_absolute_logistic_coefficient"
    ].rank(
        pct=True
    )

    result[
        "permutation_percentile"
    ] = result[
        "permutation_importance_mean"
    ].rank(
        pct=True
    )

    result["global_importance_score"] = (
        0.5
        * result["coefficient_percentile"]
        + 0.5
        * result["permutation_percentile"]
    )

    return (
        result.sort_values(
            "global_importance_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_tree_model(
    X_scaled: np.ndarray,
    y: pd.Series,
    features: list[str],
) -> DecisionTreeClassifier:
    """Fit a shallow human-readable surrogate tree."""

    tree = DecisionTreeClassifier(
        max_depth=TREE_MAX_DEPTH,
        min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    tree.fit(
        X_scaled,
        y,
    )

    return tree


def save_tree_rules(
    tree: DecisionTreeClassifier,
    features: list[str],
) -> None:
    """Export the shallow tree as readable text."""

    rules = export_text(
        tree,
        feature_names=features,
        decimals=3,
    )

    TREE_RULES_PATH.write_text(
        rules,
        encoding="utf-8",
    )


def build_local_explanations(
    pipeline: Pipeline,
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Explain each prediction with linear feature contributions."""

    model = pipeline.named_steps[
        "model"
    ]

    scaled = extract_standardized_matrix(
        pipeline,
        X,
    )

    probabilities = pipeline.predict_proba(
        X
    )

    predicted_labels = pipeline.predict(
        X
    )

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(
            model.classes_
        )
    }

    records: list[dict[str, Any]] = []

    for row_index in range(
        len(metadata)
    ):
        predicted_album = (
            predicted_labels[row_index]
        )

        class_index = class_to_index[
            predicted_album
        ]

        contributions = (
            scaled[row_index]
            * model.coef_[class_index]
        )

        positive_indices = np.argsort(
            contributions
        )[::-1][
            :TOP_LOCAL_FEATURES
        ]

        negative_indices = np.argsort(
            contributions
        )[
            :TOP_LOCAL_FEATURES
        ]

        records.append(
            {
                "master_track_id": metadata.iloc[
                    row_index
                ]["master_track_id"],
                "track_title": metadata.iloc[
                    row_index
                ]["track_title"],
                "actual_album": metadata.iloc[
                    row_index
                ]["album"],
                "predicted_album": predicted_album,
                "prediction_confidence": float(
                    probabilities[
                        row_index,
                        class_index,
                    ]
                ),
                "supporting_features": " | ".join(
                    (
                        f"{features[index]}:"
                        f"{contributions[index]:+.3f}"
                    )
                    for index
                    in positive_indices
                ),
                "opposing_features": " | ".join(
                    (
                        f"{features[index]}:"
                        f"{contributions[index]:+.3f}"
                    )
                    for index
                    in negative_indices
                ),
            }
        )

    return pd.DataFrame(records)


def build_album_signatures(
    X_scaled: np.ndarray,
    y: pd.Series,
    features: list[str],
) -> pd.DataFrame:
    """Build standardized mean feature signatures for each album."""

    scaled_df = pd.DataFrame(
        X_scaled,
        columns=features,
    )

    scaled_df["album"] = y.values

    album_means = (
        scaled_df.groupby(
            "album"
        )
        .mean()
    )

    records: list[dict[str, Any]] = []

    for album, row in album_means.iterrows():
        positive = (
            row.sort_values(
                ascending=False
            )
            .head(TOP_SIGNATURE_FEATURES)
        )

        negative = (
            row.sort_values(
                ascending=True
            )
            .head(TOP_SIGNATURE_FEATURES)
        )

        records.append(
            {
                "album": album,
                "top_positive_signature": " | ".join(
                    f"{name}:{value:+.3f}"
                    for name, value
                    in positive.items()
                ),
                "top_negative_signature": " | ".join(
                    f"{name}:{value:+.3f}"
                    for name, value
                    in negative.items()
                ),
            }
        )

    return pd.DataFrame(records)


def build_counterfactuals(
    pipeline: Pipeline,
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Create simple linear counterfactual directions."""

    model = pipeline.named_steps[
        "model"
    ]

    scaled = extract_standardized_matrix(
        pipeline,
        X,
    )

    probabilities = pipeline.predict_proba(
        X
    )

    class_names = np.array(
        model.classes_
    )

    records: list[dict[str, Any]] = []

    for row_index in range(
        len(metadata)
    ):
        order = np.argsort(
            probabilities[row_index]
        )[::-1]

        predicted_index = order[0]
        alternative_index = order[1]

        predicted_album = (
            class_names[
                predicted_index
            ]
        )

        alternative_album = (
            class_names[
                alternative_index
            ]
        )

        direction = (
            model.coef_[
                alternative_index
            ]
            - model.coef_[
                predicted_index
            ]
        )

        contribution_shift = (
            scaled[row_index]
            * direction
        )

        top_indices = np.argsort(
            np.abs(
                contribution_shift
            )
        )[::-1][
            :TOP_LOCAL_FEATURES
        ]

        records.append(
            {
                "master_track_id": metadata.iloc[
                    row_index
                ]["master_track_id"],
                "track_title": metadata.iloc[
                    row_index
                ]["track_title"],
                "actual_album": metadata.iloc[
                    row_index
                ]["album"],
                "predicted_album": predicted_album,
                "alternative_album": alternative_album,
                "predicted_probability": float(
                    probabilities[
                        row_index,
                        predicted_index,
                    ]
                ),
                "alternative_probability": float(
                    probabilities[
                        row_index,
                        alternative_index,
                    ]
                ),
                "features_most_relevant_to_switch": " | ".join(
                    (
                        f"{features[index]}:"
                        f"{contribution_shift[index]:+.3f}"
                    )
                    for index
                    in top_indices
                ),
            }
        )

    return pd.DataFrame(records)


def build_summary(
    oof_predictions: pd.DataFrame,
    tree: DecisionTreeClassifier,
    feature_count: int,
) -> pd.DataFrame:
    """Build high-level interpretation summary."""

    actual = oof_predictions[
        "actual_album"
    ]

    predicted = oof_predictions[
        "predicted_album"
    ]

    return pd.DataFrame(
        [
            {
                "metric": "tracks",
                "value": len(
                    oof_predictions
                ),
            },
            {
                "metric": "albums",
                "value": actual.nunique(),
            },
            {
                "metric": "feature_count",
                "value": feature_count,
            },
            {
                "metric": "oof_accuracy",
                "value": accuracy_score(
                    actual,
                    predicted,
                ),
            },
            {
                "metric": "oof_balanced_accuracy",
                "value": (
                    balanced_accuracy_score(
                        actual,
                        predicted,
                    )
                ),
            },
            {
                "metric": "oof_macro_f1",
                "value": f1_score(
                    actual,
                    predicted,
                    average="macro",
                ),
            },
            {
                "metric": "tree_depth",
                "value": tree.get_depth(),
            },
            {
                "metric": "tree_leaves",
                "value": tree.get_n_leaves(),
            },
        ]
    )


def plot_global_importance(
    importance: pd.DataFrame,
) -> None:
    """Plot strongest global features."""

    top = (
        importance.head(
            TOP_GLOBAL_FEATURES
        )
        .sort_values(
            "global_importance_score"
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
        for feature in top[
            "feature"
        ]
    ]

    figure, axis = plt.subplots(
        figsize=(12, 9)
    )

    axis.barh(
        labels,
        top[
            "global_importance_score"
        ],
    )

    axis.set_title(
        "Global Album-Classification Feature Importance"
    )

    axis.set_xlabel(
        "Combined coefficient and permutation score"
    )

    figure.tight_layout()

    figure.savefig(
        GLOBAL_IMPORTANCE_FIGURE_PATH,
        dpi=230,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_tree_model(
    tree: DecisionTreeClassifier,
    features: list[str],
) -> None:
    """Plot the shallow surrogate tree."""

    figure, axis = plt.subplots(
        figsize=(22, 12)
    )

    plot_tree(
        tree,
        feature_names=features,
        class_names=[
            str(name)
            for name in tree.classes_
        ],
        filled=True,
        rounded=True,
        fontsize=7,
        ax=axis,
    )

    axis.set_title(
        "Interpretable Album-Classification Rules"
    )

    figure.tight_layout()

    figure.savefig(
        TREE_FIGURE_PATH,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_album_signatures(
    X_scaled: np.ndarray,
    y: pd.Series,
    features: list[str],
    importance: pd.DataFrame,
) -> None:
    """Plot standardized album signatures for top global features."""

    top_features = (
        importance.head(
            15
        )["feature"]
        .tolist()
    )

    scaled_df = pd.DataFrame(
        X_scaled,
        columns=features,
    )

    scaled_df["album"] = y.values

    matrix = (
        scaled_df.groupby(
            "album"
        )[top_features]
        .mean()
    )

    figure, axis = plt.subplots(
        figsize=(14, 9)
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
        "Standardized Album Signatures"
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Album mean z-score",
    )

    figure.tight_layout()

    figure.savefig(
        ALBUM_SIGNATURES_FIGURE_PATH,
        dpi=230,
        bbox_inches="tight",
    )

    plt.close(figure)


def print_findings(
    summary: pd.DataFrame,
    importance: pd.DataFrame,
    album_signatures: pd.DataFrame,
    local_explanations: pd.DataFrame,
) -> None:
    """Print the most important interpretation findings."""

    summary_lookup = (
        summary.set_index(
            "metric"
        )["value"]
    )

    print(
        "\nInterpretable album-rules summary:"
    )

    print(
        f"- OOF macro F1: "
        f"{float(summary_lookup['oof_macro_f1']):.3f}"
    )

    print(
        f"- OOF balanced accuracy: "
        f"{float(summary_lookup['oof_balanced_accuracy']):.3f}"
    )

    print(
        "\nTop global features:"
    )

    for _, row in (
        importance.head(12)
        .iterrows()
    ):
        print(
            f"- {row['feature']}: "
            f"{row['global_importance_score']:.3f}"
        )

    print(
        "\nAlbum signatures:"
    )

    for _, row in album_signatures.iterrows():
        print(
            f"- {row['album']}: "
            f"{row['top_positive_signature']}"
        )

    print(
        "\nExample local explanations:"
    )

    examples = (
        local_explanations.sort_values(
            "prediction_confidence",
            ascending=False,
        )
        .head(8)
    )

    for _, row in examples.iterrows():
        print(
            f"- {row['track_title']}: "
            f"predicted={row['predicted_album']}, "
            f"support={row['supporting_features']}"
        )


def main() -> None:
    """Run the complete interpretation pipeline."""

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

    (
        analysis_df,
        X,
        y,
    ) = build_analysis_frame(
        df=df,
        features=features,
    )

    pipeline = fit_primary_model(
        X=X,
        y=y,
    )

    oof_predictions = (
        create_oof_predictions(
            X=X,
            y=y,
            pipeline=pipeline,
        )
    )

    global_importance = (
        build_global_importance(
            pipeline=pipeline,
            X=X,
            y=y,
            features=features,
        )
    )

    X_scaled = extract_standardized_matrix(
        pipeline,
        X,
    )

    tree = build_tree_model(
        X_scaled=X_scaled,
        y=y,
        features=features,
    )

    save_tree_rules(
        tree=tree,
        features=features,
    )

    metadata = analysis_df[
        [
            "master_track_id",
            "track_title",
            "album",
        ]
    ]

    local_explanations = (
        build_local_explanations(
            pipeline=pipeline,
            X=X,
            metadata=metadata,
            features=features,
        )
    )

    album_signatures = (
        build_album_signatures(
            X_scaled=X_scaled,
            y=y,
            features=features,
        )
    )

    counterfactuals = (
        build_counterfactuals(
            pipeline=pipeline,
            X=X,
            metadata=metadata,
            features=features,
        )
    )

    summary = build_summary(
        oof_predictions=oof_predictions,
        tree=tree,
        feature_count=len(
            features
        ),
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )

    global_importance.to_csv(
        GLOBAL_IMPORTANCE_PATH,
        index=False,
        encoding="utf-8",
    )

    local_explanations.to_csv(
        LOCAL_EXPLANATIONS_PATH,
        index=False,
        encoding="utf-8",
    )

    album_signatures.to_csv(
        ALBUM_SIGNATURES_PATH,
        index=False,
        encoding="utf-8",
    )

    counterfactuals.to_csv(
        COUNTERFACTUALS_PATH,
        index=False,
        encoding="utf-8",
    )

    plot_global_importance(
        global_importance
    )

    plot_tree_model(
        tree=tree,
        features=features,
    )

    plot_album_signatures(
        X_scaled=X_scaled,
        y=y,
        features=features,
        importance=global_importance,
    )

    print_findings(
        summary=summary,
        importance=global_importance,
        album_signatures=album_signatures,
        local_explanations=local_explanations,
    )

    print("\nSaved:")
    print(f"- {SUMMARY_PATH}")
    print(f"- {GLOBAL_IMPORTANCE_PATH}")
    print(f"- {TREE_RULES_PATH}")
    print(f"- {LOCAL_EXPLANATIONS_PATH}")
    print(f"- {ALBUM_SIGNATURES_PATH}")
    print(f"- {COUNTERFACTUALS_PATH}")
    print(f"- {TREE_FIGURE_PATH}")
    print(f"- {GLOBAL_IMPORTANCE_FIGURE_PATH}")
    print(f"- {ALBUM_SIGNATURES_FIGURE_PATH}")

    print(
        "\nInterpretable album-rules analysis completed."
    )


if __name__ == "__main__":
    main()
