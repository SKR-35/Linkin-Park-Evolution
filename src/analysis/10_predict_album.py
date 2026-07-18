"""Predict a song's studio album from lyrical and audio features.

Input
-----
data/processed/master_dataset.parquet

Outputs
-------
outputs/tables/album_prediction_model_summary.csv
outputs/tables/album_prediction_cv_predictions.csv
outputs/tables/album_prediction_confusion_matrix.csv
outputs/tables/album_prediction_feature_importance.csv
outputs/tables/album_prediction_errors.csv

outputs/figures/album_prediction_model_comparison.png
outputs/figures/album_prediction_confusion_matrix.png
outputs/figures/album_prediction_feature_importance.png

Purpose
-------
This script tests whether Linkin Park studio albums have sufficiently
distinct data signatures for a machine-learning model to identify the
album of an unseen song.

Scientific safeguards
---------------------
1. Album/year/era/title metadata are never used as predictors.
2. The primary eight-album task uses lyrics features only.
   This avoids exploiting the fact that From Zero has no AcousticBrainz data.
3. Audio and hybrid tasks are evaluated only on albums with audio coverage
   (the seven Chester-era albums).
4. Performance is compared against a majority-class baseline.
5. Repeated stratified cross-validation is used because the catalogue is
   small (roughly 10-15 tracks per album).
6. Macro F1 and balanced accuracy are emphasized instead of raw accuracy.
7. Permutation importance is calculated from out-of-fold predictions.

Tasks
-----
- emotion_all_8:
  Eight albums, emotion and lyrical-theme features.

- lyrics_all_8:
  Eight albums, full lyrics-style features. This is the primary task.

- audio_chester_7:
  Seven Chester-era albums with AcousticBrainz coverage.

- hybrid_chester_7:
  Seven Chester-era albums using lyrics plus audio features.

Models
------
- multinomial logistic regression
- linear support vector classifier
- random forest
- extra trees

The best model for each task is selected by mean macro F1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_dataset.parquet"
)

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

MODEL_SUMMARY_PATH = (
    TABLES_DIR
    / "album_prediction_model_summary.csv"
)

CV_PREDICTIONS_PATH = (
    TABLES_DIR
    / "album_prediction_cv_predictions.csv"
)

CONFUSION_MATRIX_PATH = (
    TABLES_DIR
    / "album_prediction_confusion_matrix.csv"
)

FEATURE_IMPORTANCE_PATH = (
    TABLES_DIR
    / "album_prediction_feature_importance.csv"
)

ERRORS_PATH = (
    TABLES_DIR
    / "album_prediction_errors.csv"
)

MODEL_COMPARISON_FIGURE_PATH = (
    FIGURES_DIR
    / "album_prediction_model_comparison.png"
)

CONFUSION_FIGURE_PATH = (
    FIGURES_DIR
    / "album_prediction_confusion_matrix.png"
)

IMPORTANCE_FIGURE_PATH = (
    FIGURES_DIR
    / "album_prediction_feature_importance.png"
)


RANDOM_STATE = 42
CV_SPLITS = 5
CV_REPEATS = 10
PERMUTATION_REPEATS = 30
TOP_FEATURES = 20


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


@dataclass(frozen=True)
class TaskSpec:
    """Configuration for one prediction task."""

    name: str
    requested_features: list[str]
    eligibility: str
    description: str


TASKS = [
    TaskSpec(
        name="emotion_all_8",
        requested_features=EMOTION_FEATURES,
        eligibility="lyrics_all",
        description="Eight albums using emotion and theme features.",
    ),
    TaskSpec(
        name="lyrics_all_8",
        requested_features=LYRICS_STYLE_FEATURES,
        eligibility="lyrics_all",
        description="Eight albums using full lyrics-style features.",
    ),
    TaskSpec(
        name="audio_chester_7",
        requested_features=AUDIO_FEATURES,
        eligibility="audio_chester",
        description="Seven Chester-era albums using audio features.",
    ),
    TaskSpec(
        name="hybrid_chester_7",
        requested_features=[
            *LYRICS_STYLE_FEATURES,
            *AUDIO_FEATURES,
        ],
        eligibility="hybrid_chester",
        description="Seven Chester-era albums using lyrics and audio.",
    ),
]


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
    """Load and validate the master analytical dataset."""

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
        | no_lyric_words
        | known_instrumental
    )

    return (
        df.sort_values(
            [
                "album_order",
                "track_position",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def eligibility_mask(
    df: pd.DataFrame,
    rule: str,
) -> pd.Series:
    """Return a scientifically appropriate task eligibility mask."""

    if rule == "lyrics_all":
        return (
            df["has_lyrics_features"]
            & ~df["is_instrumental"]
        )

    if rule == "audio_chester":
        return (
            df["has_audio_features"]
            & df["era"].eq("Chester Era")
        )

    if rule == "hybrid_chester":
        return (
            df["has_lyrics_features"]
            & df["has_audio_features"]
            & ~df["is_instrumental"]
            & df["era"].eq("Chester Era")
        )

    raise ValueError(
        f"Unknown eligibility rule: {rule}"
    )


def select_features(
    df: pd.DataFrame,
    requested: list[str],
) -> list[str]:
    """Keep available numeric features with useful variation."""

    selected: list[str] = []

    for column in requested:
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
            "No usable prediction features were found."
        )

    return selected


def build_models() -> dict[str, Pipeline]:
    """Create candidate classification pipelines."""

    scaled_preprocessing = [
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

    tree_preprocessing = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
            ),
        ),
    ]

    return {
        "multinomial_logistic": Pipeline(
            steps=[
                *scaled_preprocessing,
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
        ),
        "linear_svc": Pipeline(
            steps=[
                *scaled_preprocessing,
                (
                    "model",
                    SVC(
                        kernel="linear",
                        C=0.5,
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                *tree_preprocessing,
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=700,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                *tree_preprocessing,
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=700,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    task: TaskSpec,
    models: dict[str, Pipeline],
) -> pd.DataFrame:
    """Evaluate candidate models with repeated stratified CV."""

    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
    }

    majority_baseline = (
        y.value_counts(normalize=True)
        .max()
    )

    records: list[dict[str, Any]] = []

    for model_name, model in models.items():
        scores = cross_validate(
            estimator=model,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        records.append(
            {
                "task": task.name,
                "task_description": task.description,
                "model": model_name,
                "tracks": len(X),
                "classes": y.nunique(),
                "feature_count": X.shape[1],
                "majority_baseline_accuracy": round(
                    float(majority_baseline),
                    6,
                ),
                "mean_accuracy": round(
                    float(
                        scores[
                            "test_accuracy"
                        ].mean()
                    ),
                    6,
                ),
                "std_accuracy": round(
                    float(
                        scores[
                            "test_accuracy"
                        ].std()
                    ),
                    6,
                ),
                "mean_balanced_accuracy": round(
                    float(
                        scores[
                            "test_balanced_accuracy"
                        ].mean()
                    ),
                    6,
                ),
                "std_balanced_accuracy": round(
                    float(
                        scores[
                            "test_balanced_accuracy"
                        ].std()
                    ),
                    6,
                ),
                "mean_macro_f1": round(
                    float(
                        scores[
                            "test_macro_f1"
                        ].mean()
                    ),
                    6,
                ),
                "std_macro_f1": round(
                    float(
                        scores[
                            "test_macro_f1"
                        ].std()
                    ),
                    6,
                ),
            }
        )

    return pd.DataFrame(records)


def out_of_fold_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    model: Pipeline,
    task_name: str,
) -> pd.DataFrame:
    """Create one deterministic five-fold out-of-fold prediction set."""

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
            len(
                encoder.classes_
            ),
        ),
        dtype=float,
    )

    fold_numbers = np.zeros(
        len(X),
        dtype=int,
    )

    for fold_number, (
        train_index,
        test_index,
    ) in enumerate(
        cv.split(
            X,
            y_encoded,
        ),
        start=1,
    ):
        fitted = clone(model)

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

        fold_numbers[test_index] = (
            fold_number
        )

    predicted_labels = (
        encoder.inverse_transform(
            predictions
        )
    )

    result = metadata.reset_index(
        drop=True
    ).copy()

    result.insert(
        0,
        "task",
        task_name,
    )

    result["fold"] = fold_numbers
    result["actual_album"] = y.reset_index(
        drop=True
    )
    result["predicted_album"] = (
        predicted_labels
    )
    result["correct"] = (
        result["actual_album"]
        == result["predicted_album"]
    )
    result["prediction_confidence"] = (
        probabilities.max(axis=1)
    )

    sorted_probabilities = np.sort(
        probabilities,
        axis=1,
    )

    result["prediction_margin"] = (
        sorted_probabilities[:, -1]
        - sorted_probabilities[:, -2]
    )

    result["top_2_correct"] = [
        y_encoded[index]
        in np.argsort(
            probabilities[index]
        )[-2:]
        for index in range(
            len(result)
        )
    ]

    for class_index, class_name in enumerate(
        encoder.classes_
    ):
        safe_name = (
            str(class_name)
            .lower()
            .replace(" ", "_")
        )

        result[
            f"probability_{safe_name}"
        ] = probabilities[
            :,
            class_index,
        ]

    return result


def summarize_oof(
    predictions: pd.DataFrame,
) -> dict[str, float]:
    """Calculate final out-of-fold metrics."""

    actual = predictions[
        "actual_album"
    ]

    predicted = predictions[
        "predicted_album"
    ]

    return {
        "oof_accuracy": float(
            accuracy_score(
                actual,
                predicted,
            )
        ),
        "oof_balanced_accuracy": float(
            balanced_accuracy_score(
                actual,
                predicted,
            )
        ),
        "oof_macro_f1": float(
            f1_score(
                actual,
                predicted,
                average="macro",
            )
        ),
        "oof_top_2_accuracy": float(
            predictions[
                "top_2_correct"
            ].mean()
        ),
    }


def build_confusion_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Build a labeled confusion matrix."""

    labels = sorted(
        predictions[
            "actual_album"
        ].unique()
    )

    matrix = confusion_matrix(
        predictions[
            "actual_album"
        ],
        predictions[
            "predicted_album"
        ],
        labels=labels,
    )

    result = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    result.index.name = "actual_album"

    return result.reset_index()


def calculate_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    model: Pipeline,
    feature_names: list[str],
    task_name: str,
) -> pd.DataFrame:
    """Calculate repeated permutation importance on a fitted final model."""

    fitted = clone(model)

    fitted.fit(
        X,
        y,
    )

    importance = permutation_importance(
        estimator=fitted,
        X=X,
        y=y,
        scoring="f1_macro",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    result = pd.DataFrame(
        {
            "task": task_name,
            "feature": feature_names,
            "importance_mean": (
                importance.importances_mean
            ),
            "importance_std": (
                importance.importances_std
            ),
        }
    )

    return (
        result.sort_values(
            "importance_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def plot_model_comparison(
    summary: pd.DataFrame,
) -> None:
    """Plot macro F1 for every task and candidate model."""

    labels = (
        summary["task"]
        + "\n"
        + summary["model"]
    )

    figure, axis = plt.subplots(
        figsize=(15, 8)
    )

    axis.bar(
        labels,
        summary[
            "mean_macro_f1"
        ],
        yerr=summary[
            "std_macro_f1"
        ],
        capsize=3,
    )

    axis.set_title(
        "Album Prediction: Repeated-CV Macro F1"
    )

    axis.set_xlabel(
        "Task and model"
    )

    axis.set_ylabel(
        "Mean macro F1"
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    figure.tight_layout()

    figure.savefig(
        MODEL_COMPARISON_FIGURE_PATH,
        dpi=230,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_confusion_matrix(
    confusion: pd.DataFrame,
) -> None:
    """Plot the primary-task out-of-fold confusion matrix."""

    labels = confusion[
        "actual_album"
    ].tolist()

    matrix = confusion.drop(
        columns="actual_album"
    ).to_numpy()

    figure, axis = plt.subplots(
        figsize=(11, 10)
    )

    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
    )

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

    axis.set_xlabel(
        "Predicted album"
    )

    axis.set_ylabel(
        "Actual album"
    )

    axis.set_title(
        "Lyrics-Based Album Prediction Confusion Matrix"
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
                str(
                    int(
                        matrix[
                            row_index,
                            column_index,
                        ]
                    )
                ),
                ha="center",
                va="center",
                fontsize=8,
            )

    figure.colorbar(
        image,
        ax=axis,
        label="Song count",
    )

    figure.tight_layout()

    figure.savefig(
        CONFUSION_FIGURE_PATH,
        dpi=230,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_feature_importance(
    importance: pd.DataFrame,
) -> None:
    """Plot the strongest primary-task feature importances."""

    top = (
        importance.head(
            TOP_FEATURES
        )
        .sort_values(
            "importance_mean"
        )
    )

    labels = [
        feature.replace(
            "lyrics_",
            "",
        ).replace(
            "audio_ab_",
            "audio_",
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
            "importance_mean"
        ],
        xerr=top[
            "importance_std"
        ],
        capsize=3,
    )

    axis.set_title(
        "Most Important Features for Album Prediction"
    )

    axis.set_xlabel(
        "Permutation importance decrease in macro F1"
    )

    figure.tight_layout()

    figure.savefig(
        IMPORTANCE_FIGURE_PATH,
        dpi=230,
        bbox_inches="tight",
    )

    plt.close(figure)


def validate_task(
    task_df: pd.DataFrame,
    feature_names: list[str],
    task_name: str,
) -> None:
    """Run critical validation before model evaluation."""

    class_counts = task_df[
        "album"
    ].value_counts()

    if class_counts.min() < CV_SPLITS:
        raise ValueError(
            f"{task_name} has a class with fewer than "
            f"{CV_SPLITS} tracks:\n{class_counts}"
        )

    if any(
        column
        in {
            "album",
            "album_order",
            "release_year",
            "era",
            "track_title",
            "canonical_title",
        }
        for column in feature_names
    ):
        raise ValueError(
            f"Target leakage detected in {task_name}."
        )


def main() -> None:
    """Evaluate album-prediction tasks and save explainable results."""

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset()
    models = build_models()

    all_summaries: list[pd.DataFrame] = []
    best_task_artifacts: dict[
        str,
        dict[str, Any],
    ] = {}

    print("\nAlbum prediction tasks:")

    for task in TASKS:
        mask = eligibility_mask(
            df,
            task.eligibility,
        )

        task_df = df.loc[
            mask
        ].copy()

        feature_names = select_features(
            task_df,
            task.requested_features,
        )

        validate_task(
            task_df=task_df,
            feature_names=feature_names,
            task_name=task.name,
        )

        X = (
            task_df[
                feature_names
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .reset_index(drop=True)
        )

        y = (
            task_df["album"]
            .astype(str)
            .reset_index(drop=True)
        )

        task_summary = evaluate_models(
            X=X,
            y=y,
            task=task,
            models=models,
        )

        all_summaries.append(
            task_summary
        )

        best_row = (
            task_summary.sort_values(
                [
                    "mean_macro_f1",
                    "mean_balanced_accuracy",
                ],
                ascending=False,
            )
            .iloc[0]
        )

        best_model_name = str(
            best_row["model"]
        )

        best_task_artifacts[
            task.name
        ] = {
            "task_df": task_df.reset_index(
                drop=True
            ),
            "X": X,
            "y": y,
            "features": feature_names,
            "model_name": best_model_name,
            "model": models[
                best_model_name
            ],
        }

        print(
            f"- {task.name}: "
            f"{len(task_df)} tracks, "
            f"{y.nunique()} albums, "
            f"{len(feature_names)} features, "
            f"best={best_model_name}, "
            f"macro F1={best_row['mean_macro_f1']:.3f}"
        )

    summary = pd.concat(
        all_summaries,
        ignore_index=True,
        sort=False,
    )

    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for task_name, artifact in (
        best_task_artifacts.items()
    ):
        metadata = artifact[
            "task_df"
        ][
            [
                "master_track_id",
                "track_title",
                "album",
                "release_year",
                "era",
            ]
        ]

        predictions = out_of_fold_predictions(
            X=artifact["X"],
            y=artifact["y"],
            metadata=metadata,
            model=artifact["model"],
            task_name=task_name,
        )

        prediction_frames.append(
            predictions
        )

        oof_metrics = summarize_oof(
            predictions
        )

        best_mask = (
            summary["task"].eq(
                task_name
            )
            & summary["model"].eq(
                artifact["model_name"]
            )
        )

        for metric_name, value in (
            oof_metrics.items()
        ):
            summary.loc[
                best_mask,
                metric_name,
            ] = round(
                value,
                6,
            )

        importance_frames.append(
            calculate_feature_importance(
                X=artifact["X"],
                y=artifact["y"],
                model=artifact["model"],
                feature_names=artifact[
                    "features"
                ],
                task_name=task_name,
            )
        )

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
        sort=False,
    )

    all_importance = pd.concat(
        importance_frames,
        ignore_index=True,
        sort=False,
    )

    primary_predictions = (
        all_predictions[
            all_predictions[
                "task"
            ].eq(
                "lyrics_all_8"
            )
        ]
        .reset_index(drop=True)
    )

    primary_confusion = (
        build_confusion_table(
            primary_predictions
        )
    )

    primary_importance = (
        all_importance[
            all_importance[
                "task"
            ].eq(
                "lyrics_all_8"
            )
        ]
        .reset_index(drop=True)
    )

    errors = (
        primary_predictions[
            ~primary_predictions[
                "correct"
            ]
        ]
        .sort_values(
            [
                "prediction_confidence",
                "prediction_margin",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    summary.to_csv(
        MODEL_SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )

    all_predictions.to_csv(
        CV_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8",
    )

    primary_confusion.to_csv(
        CONFUSION_MATRIX_PATH,
        index=False,
        encoding="utf-8",
    )

    all_importance.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
        encoding="utf-8",
    )

    errors.to_csv(
        ERRORS_PATH,
        index=False,
        encoding="utf-8",
    )

    plot_model_comparison(
        summary
    )

    plot_confusion_matrix(
        primary_confusion
    )

    plot_feature_importance(
        primary_importance
    )

    print("\nModel summary:")
    print(
        summary.sort_values(
            [
                "task",
                "mean_macro_f1",
            ],
            ascending=[
                True,
                False,
            ],
        ).to_string(
            index=False
        )
    )

    print(
        "\nPrimary task: lyrics_all_8"
    )

    primary_best = (
        summary[
            summary["task"].eq(
                "lyrics_all_8"
            )
        ]
        .sort_values(
            "mean_macro_f1",
            ascending=False,
        )
        .iloc[0]
    )

    print(
        f"Best model: "
        f"{primary_best['model']}"
    )

    print(
        f"Repeated-CV macro F1: "
        f"{primary_best['mean_macro_f1']:.3f}"
    )

    print(
        f"OOF balanced accuracy: "
        f"{primary_best.get('oof_balanced_accuracy', np.nan):.3f}"
    )

    print(
        f"OOF top-2 accuracy: "
        f"{primary_best.get('oof_top_2_accuracy', np.nan):.3f}"
    )

    print(
        "\nMost informative features:"
    )

    for _, row in (
        primary_importance.head(12)
        .iterrows()
    ):
        print(
            f"- {row['feature']}: "
            f"{row['importance_mean']:.4f}"
        )

    print(
        "\nMost confident mistakes:"
    )

    for _, row in errors.head(10).iterrows():
        print(
            f"- {row['track_title']}: "
            f"actual={row['actual_album']}, "
            f"predicted={row['predicted_album']}, "
            f"confidence={row['prediction_confidence']:.3f}"
        )

    print("\nSaved:")
    print(f"- {MODEL_SUMMARY_PATH}")
    print(f"- {CV_PREDICTIONS_PATH}")
    print(f"- {CONFUSION_MATRIX_PATH}")
    print(f"- {FEATURE_IMPORTANCE_PATH}")
    print(f"- {ERRORS_PATH}")
    print(f"- {MODEL_COMPARISON_FIGURE_PATH}")
    print(f"- {CONFUSION_FIGURE_PATH}")
    print(f"- {IMPORTANCE_FIGURE_PATH}")

    print("\nAlbum prediction analysis completed.")


if __name__ == "__main__":
    main()
