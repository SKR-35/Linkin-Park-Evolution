"""Build track-level NLP, sentiment, emotion, style, and readability features.

Input
-----
data/processed/lyrics_clean.parquet

Outputs
-------
data/processed/lyrics_features.parquet
data/processed/lyrics_features.csv
data/interim/lyrics/nlp_feature_coverage.csv
data/interim/lyrics/nlp_feature_errors.csv

Feature groups
--------------
- Basic text structure
- Vocabulary richness and repetition
- Pronoun and stylistic usage
- Readability
- VADER sentiment
- NRC emotion lexicon scores
- Lightweight keyword/theme indicators

Notes
-----
- This script writes derived features only; it does not write lyrics text.
- The instrumental track is retained with null/zero NLP features.
- NRC features use NRCLex when available.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lyrics_clean.parquet"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "lyrics"

PARQUET_OUTPUT_PATH = (
    PROCESSED_DIR
    / "lyrics_features.parquet"
)

CSV_OUTPUT_PATH = (
    PROCESSED_DIR
    / "lyrics_features.csv"
)

COVERAGE_OUTPUT_PATH = (
    INTERIM_DIR
    / "nlp_feature_coverage.csv"
)

ERRORS_OUTPUT_PATH = (
    INTERIM_DIR
    / "nlp_feature_errors.csv"
)


WORD_PATTERN = re.compile(
    r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b"
)

SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?])\s+|\n+"
)

ELLIPSIS_PATTERN = re.compile(
    r"(?:\.\.\.|…)"
)

FIRST_PERSON_SINGULAR = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
}

FIRST_PERSON_PLURAL = {
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
}

SECOND_PERSON = {
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

THIRD_PERSON = {
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
}

NEGATIONS = {
    "no",
    "not",
    "never",
    "none",
    "nothing",
    "nowhere",
    "neither",
    "nor",
    "cannot",
    "can't",
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "wouldn't",
    "shouldn't",
    "couldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "haven't",
    "hasn't",
    "hadn't",
}

ABSOLUTIST_WORDS = {
    "always",
    "never",
    "completely",
    "totally",
    "absolutely",
    "everything",
    "nothing",
    "everyone",
    "nobody",
    "entirely",
    "forever",
}

BODY_WORDS = {
    "blood",
    "skin",
    "heart",
    "head",
    "eyes",
    "eye",
    "face",
    "hands",
    "hand",
    "bones",
    "body",
    "breath",
}

DARKNESS_WORDS = {
    "dark",
    "darkness",
    "shadow",
    "shadows",
    "night",
    "black",
    "empty",
    "emptiness",
    "cold",
    "dead",
    "death",
    "die",
    "dying",
}

PAIN_WORDS = {
    "pain",
    "hurt",
    "hurting",
    "ache",
    "broken",
    "break",
    "breaking",
    "wound",
    "wounded",
    "scar",
    "scars",
    "suffer",
    "suffering",
}

HOPE_WORDS = {
    "hope",
    "light",
    "rise",
    "rising",
    "alive",
    "tomorrow",
    "believe",
    "healing",
    "heal",
    "free",
    "freedom",
}

CONFLICT_WORDS = {
    "fight",
    "war",
    "battle",
    "enemy",
    "enemies",
    "rage",
    "anger",
    "angry",
    "hate",
    "attack",
}

ISOLATION_WORDS = {
    "alone",
    "lonely",
    "isolate",
    "isolated",
    "inside",
    "away",
    "apart",
    "lost",
    "nobody",
    "silence",
}

TIME_WORDS = {
    "time",
    "again",
    "before",
    "after",
    "past",
    "future",
    "today",
    "tonight",
    "tomorrow",
    "yesterday",
    "forever",
}

NRC_EMOTIONS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
    "positive",
    "negative",
]


def safe_divide(
    numerator: float | int,
    denominator: float | int,
) -> float:
    """Divide safely and return zero for a zero denominator."""

    if denominator == 0:
        return 0.0

    return float(numerator) / float(denominator)


def safe_mean(values: Iterable[float]) -> float:
    """Return a finite mean or zero."""

    values_list = [
        float(value)
        for value in values
        if value is not None
        and math.isfinite(float(value))
    ]

    if not values_list:
        return 0.0

    return float(statistics.mean(values_list))


def safe_std(values: Iterable[float]) -> float:
    """Return a finite population standard deviation or zero."""

    values_list = [
        float(value)
        for value in values
        if value is not None
        and math.isfinite(float(value))
    ]

    if len(values_list) < 2:
        return 0.0

    return float(statistics.pstdev(values_list))


def tokenize_words(text: str) -> list[str]:
    """Tokenize English word-like strings and preserve contractions."""

    return [
        token.lower()
        for token in WORD_PATTERN.findall(text)
    ]


def split_sentences(text: str) -> list[str]:
    """Split lyrics into approximate sentences/lines."""

    sentences = [
        segment.strip()
        for segment in SENTENCE_SPLIT_PATTERN.split(text)
        if segment.strip()
    ]

    return sentences


def nonempty_lines(text: str) -> list[str]:
    """Return non-empty stripped lyric lines."""

    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


def count_set_matches(
    tokens: list[str],
    vocabulary: set[str],
) -> int:
    """Count token occurrences belonging to a vocabulary."""

    return sum(
        1
        for token in tokens
        if token in vocabulary
    )


def count_syllables_fallback(word: str) -> int:
    """Estimate English syllables without external resources."""

    word = word.lower()
    word = re.sub(r"[^a-z]", "", word)

    if not word:
        return 0

    if len(word) <= 3:
        return 1

    vowels = "aeiouy"
    syllables = 0
    previous_is_vowel = False

    for character in word:
        is_vowel = character in vowels

        if is_vowel and not previous_is_vowel:
            syllables += 1

        previous_is_vowel = is_vowel

    if word.endswith("e") and syllables > 1:
        syllables -= 1

    if word.endswith("le") and len(word) > 2:
        if word[-3] not in vowels:
            syllables += 1

    return max(1, syllables)


def readability_fallback(
    tokens: list[str],
    sentences: list[str],
) -> dict[str, float]:
    """Calculate simple readability estimates without textstat."""

    word_count = len(tokens)
    sentence_count = max(1, len(sentences))

    syllable_count = sum(
        count_syllables_fallback(word)
        for word in tokens
    )

    complex_word_count = sum(
        count_syllables_fallback(word) >= 3
        for word in tokens
    )

    words_per_sentence = safe_divide(
        word_count,
        sentence_count,
    )

    syllables_per_word = safe_divide(
        syllable_count,
        word_count,
    )

    flesch_reading_ease = (
        206.835
        - 1.015 * words_per_sentence
        - 84.6 * syllables_per_word
    )

    flesch_kincaid_grade = (
        0.39 * words_per_sentence
        + 11.8 * syllables_per_word
        - 15.59
    )

    gunning_fog = (
        0.4
        * (
            words_per_sentence
            + 100
            * safe_divide(
                complex_word_count,
                word_count,
            )
        )
    )

    return {
        "readability_flesch_reading_ease": round(
            flesch_reading_ease,
            4,
        ),
        "readability_flesch_kincaid_grade": round(
            flesch_kincaid_grade,
            4,
        ),
        "readability_gunning_fog": round(
            gunning_fog,
            4,
        ),
        "readability_smog_index": float("nan"),
        "readability_automated_readability_index": float("nan"),
        "readability_coleman_liau_index": float("nan"),
        "readability_dale_chall_score": float("nan"),
    }


def calculate_readability(
    text: str,
    tokens: list[str],
    sentences: list[str],
) -> tuple[dict[str, float], str]:
    """Calculate readability with textstat when available."""

    try:
        import textstat

        return (
            {
                "readability_flesch_reading_ease": float(
                    textstat.flesch_reading_ease(text)
                ),
                "readability_flesch_kincaid_grade": float(
                    textstat.flesch_kincaid_grade(text)
                ),
                "readability_gunning_fog": float(
                    textstat.gunning_fog(text)
                ),
                "readability_smog_index": float(
                    textstat.smog_index(text)
                ),
                "readability_automated_readability_index": float(
                    textstat.automated_readability_index(text)
                ),
                "readability_coleman_liau_index": float(
                    textstat.coleman_liau_index(text)
                ),
                "readability_dale_chall_score": float(
                    textstat.dale_chall_readability_score(text)
                ),
            },
            "textstat",
        )

    except Exception:
        return (
            readability_fallback(
                tokens=tokens,
                sentences=sentences,
            ),
            "fallback",
        )


def create_vader_analyzer() -> Any | None:
    """Create a VADER sentiment analyzer, downloading the lexicon if needed."""

    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer

        try:
            return SentimentIntensityAnalyzer()

        except LookupError:
            nltk.download(
                "vader_lexicon",
                quiet=True,
            )

            return SentimentIntensityAnalyzer()

    except Exception:
        return None


def calculate_vader(
    text: str,
    analyzer: Any | None,
) -> dict[str, float]:
    """Calculate VADER sentiment scores."""

    if analyzer is None or not text.strip():
        return {
            "vader_negative": float("nan"),
            "vader_neutral": float("nan"),
            "vader_positive": float("nan"),
            "vader_compound": float("nan"),
        }

    scores = analyzer.polarity_scores(text)

    return {
        "vader_negative": float(scores["neg"]),
        "vader_neutral": float(scores["neu"]),
        "vader_positive": float(scores["pos"]),
        "vader_compound": float(scores["compound"]),
    }


def calculate_nrc(
    text: str,
    tokens: list[str],
) -> tuple[dict[str, float], str]:
    """Calculate NRC emotion counts and normalized scores with NRCLex."""

    empty_result: dict[str, float] = {}

    for emotion in NRC_EMOTIONS:
        empty_result[f"nrc_{emotion}_count"] = 0.0
        empty_result[f"nrc_{emotion}_ratio"] = 0.0

    if not text.strip():
        return empty_result, "empty"

    try:
        from nrclex import NRCLex

        emotion_object = NRCLex()
        emotion_object.load_token_list(tokens)

        raw_counts = {
            key.lower(): float(value)
            for key, value
            in emotion_object.raw_emotion_scores.items()
        }

        word_count = len(tokens)

        result: dict[str, float] = {}

        for emotion in NRC_EMOTIONS:
            count = float(
                raw_counts.get(
                    emotion,
                    0.0,
                )
            )

            result[f"nrc_{emotion}_count"] = count
            result[f"nrc_{emotion}_ratio"] = safe_divide(
                count,
                word_count,
            )

        return result, "nrclex"
        
    except Exception as error:
        return empty_result, f"unavailable: {type(error).__name__}: {error}"


def calculate_line_repetition(
    lines: list[str],
) -> dict[str, float]:
    """Calculate exact normalized line repetition metrics."""

    if not lines:
        return {
            "unique_line_count": 0,
            "duplicate_line_occurrences": 0,
            "duplicate_line_types": 0,
            "line_repetition_ratio": 0.0,
            "most_repeated_line_count": 0,
        }

    normalized_lines = [
        re.sub(
            r"\s+",
            " ",
            re.sub(
                r"[^\w\s']",
                "",
                line.lower(),
            ),
        ).strip()
        for line in lines
    ]

    normalized_lines = [
        line
        for line in normalized_lines
        if line
    ]

    counts = Counter(normalized_lines)

    duplicate_line_occurrences = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    duplicate_line_types = sum(
        1
        for count in counts.values()
        if count > 1
    )

    return {
        "unique_line_count": len(counts),
        "duplicate_line_occurrences": duplicate_line_occurrences,
        "duplicate_line_types": duplicate_line_types,
        "line_repetition_ratio": safe_divide(
            duplicate_line_occurrences,
            len(normalized_lines),
        ),
        "most_repeated_line_count": max(
            counts.values(),
            default=0,
        ),
    }


def calculate_token_features(
    tokens: list[str],
) -> dict[str, float]:
    """Calculate vocabulary, lexical, and token-frequency features."""

    token_count = len(tokens)
    token_counter = Counter(tokens)
    unique_word_count = len(token_counter)

    frequency_values = list(
        token_counter.values()
    )

    hapax_count = sum(
        count == 1
        for count in frequency_values
    )

    dis_legomena_count = sum(
        count == 2
        for count in frequency_values
    )

    repeated_token_occurrences = sum(
        max(0, count - 1)
        for count in frequency_values
    )

    avg_word_length = safe_mean(
        len(token)
        for token in tokens
    )

    word_length_std = safe_std(
        len(token)
        for token in tokens
    )

    long_word_count = sum(
        len(token) >= 7
        for token in tokens
    )

    very_long_word_count = sum(
        len(token) >= 10
        for token in tokens
    )

    lexical_density_proxy = safe_divide(
        sum(
            len(token) >= 4
            for token in tokens
        ),
        token_count,
    )

    return {
        "word_count": token_count,
        "unique_word_count": unique_word_count,
        "type_token_ratio": safe_divide(
            unique_word_count,
            token_count,
        ),
        "root_type_token_ratio": safe_divide(
            unique_word_count,
            math.sqrt(token_count)
            if token_count > 0
            else 0,
        ),
        "corrected_type_token_ratio": safe_divide(
            unique_word_count,
            math.sqrt(2 * token_count)
            if token_count > 0
            else 0,
        ),
        "hapax_count": hapax_count,
        "hapax_ratio": safe_divide(
            hapax_count,
            token_count,
        ),
        "dis_legomena_count": dis_legomena_count,
        "dis_legomena_ratio": safe_divide(
            dis_legomena_count,
            token_count,
        ),
        "repeated_token_occurrences": repeated_token_occurrences,
        "token_repetition_ratio": safe_divide(
            repeated_token_occurrences,
            token_count,
        ),
        "avg_word_length": avg_word_length,
        "word_length_std": word_length_std,
        "long_word_count": long_word_count,
        "long_word_ratio": safe_divide(
            long_word_count,
            token_count,
        ),
        "very_long_word_count": very_long_word_count,
        "very_long_word_ratio": safe_divide(
            very_long_word_count,
            token_count,
        ),
        "lexical_density_proxy": lexical_density_proxy,
    }


def calculate_structure_features(
    text: str,
    lines: list[str],
    sentences: list[str],
    tokens: list[str],
) -> dict[str, float]:
    """Calculate line, sentence, punctuation, and casing features."""

    alphabetic_characters = [
        character
        for character in text
        if character.isalpha()
    ]

    uppercase_characters = sum(
        character.isupper()
        for character in alphabetic_characters
    )

    line_word_counts = [
        len(tokenize_words(line))
        for line in lines
    ]

    sentence_word_counts = [
        len(tokenize_words(sentence))
        for sentence in sentences
    ]

    character_count = len(text)
    punctuation_count = sum(
        not character.isalnum()
        and not character.isspace()
        for character in text
    )

    return {
        "character_count": character_count,
        "line_count": len(lines),
        "sentence_count": len(sentences),
        "avg_words_per_line": safe_mean(line_word_counts),
        "line_word_count_std": safe_std(line_word_counts),
        "max_words_in_line": max(
            line_word_counts,
            default=0,
        ),
        "min_words_in_line": min(
            line_word_counts,
            default=0,
        ),
        "avg_words_per_sentence": safe_mean(
            sentence_word_counts
        ),
        "sentence_word_count_std": safe_std(
            sentence_word_counts
        ),
        "question_mark_count": text.count("?"),
        "exclamation_mark_count": text.count("!"),
        "ellipsis_count": len(
            ELLIPSIS_PATTERN.findall(text)
        ),
        "comma_count": text.count(","),
        "semicolon_count": text.count(";"),
        "colon_count": text.count(":"),
        "apostrophe_count": text.count("'"),
        "punctuation_density": safe_divide(
            punctuation_count,
            character_count,
        ),
        "uppercase_ratio": safe_divide(
            uppercase_characters,
            len(alphabetic_characters),
        ),
        "digit_count": sum(
            character.isdigit()
            for character in text
        ),
        "blank_line_count": len(
            re.findall(
                r"\n\s*\n",
                text,
            )
        ),
    }


def calculate_pronoun_and_theme_features(
    tokens: list[str],
) -> dict[str, float]:
    """Calculate pronoun, negation, absolutist, and theme indicators."""

    token_count = len(tokens)

    feature_sets = {
        "first_person_singular": FIRST_PERSON_SINGULAR,
        "first_person_plural": FIRST_PERSON_PLURAL,
        "second_person": SECOND_PERSON,
        "third_person": THIRD_PERSON,
        "negation": NEGATIONS,
        "absolutist": ABSOLUTIST_WORDS,
        "theme_body": BODY_WORDS,
        "theme_darkness": DARKNESS_WORDS,
        "theme_pain": PAIN_WORDS,
        "theme_hope": HOPE_WORDS,
        "theme_conflict": CONFLICT_WORDS,
        "theme_isolation": ISOLATION_WORDS,
        "theme_time": TIME_WORDS,
    }

    result: dict[str, float] = {}

    for feature_name, vocabulary in feature_sets.items():
        count = count_set_matches(
            tokens,
            vocabulary,
        )

        result[f"{feature_name}_count"] = count
        result[f"{feature_name}_ratio"] = safe_divide(
            count,
            token_count,
        )

    return result


def calculate_top_token_features(
    tokens: list[str],
) -> dict[str, Any]:
    """Store lightweight summaries of dominant tokens."""

    if not tokens:
        return {
            "most_common_word": None,
            "most_common_word_count": 0,
            "top_5_word_share": 0.0,
        }

    token_counter = Counter(tokens)
    most_common = token_counter.most_common(5)

    top_5_total = sum(
        count
        for _, count in most_common
    )

    return {
        "most_common_word": most_common[0][0],
        "most_common_word_count": most_common[0][1],
        "top_5_word_share": safe_divide(
            top_5_total,
            len(tokens),
        ),
    }


def build_feature_row(
    row: pd.Series,
    vader_analyzer: Any | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build all NLP features for one track."""

    text = str(
        row.get("lyrics_clean")
        if pd.notna(row.get("lyrics_clean"))
        else ""
    )

    is_instrumental = bool(
        row.get("is_instrumental")
    )

    tokens = tokenize_words(text)
    lines = nonempty_lines(text)
    sentences = split_sentences(text)

    token_features = calculate_token_features(tokens)
    structure_features = calculate_structure_features(
        text=text,
        lines=lines,
        sentences=sentences,
        tokens=tokens,
    )
    repetition_features = calculate_line_repetition(lines)
    pronoun_theme_features = (
        calculate_pronoun_and_theme_features(tokens)
    )
    top_token_features = calculate_top_token_features(tokens)

    readability_features, readability_source = (
        calculate_readability(
            text=text,
            tokens=tokens,
            sentences=sentences,
        )
    )

    vader_features = calculate_vader(
        text=text,
        analyzer=vader_analyzer,
    )

    nrc_features, nrc_source = calculate_nrc(
        text=text,
        tokens=tokens,
    )

    feature_row: dict[str, Any] = {
        "master_track_id": row["master_track_id"],
        "album_order": row["album_order"],
        "album": row["album"],
        "release_year": row["release_year"],
        "track_position": row["track_position"],
        "track_title": row["track_title"],
        "canonical_title": row["canonical_title"],
        "era": row["era"],
        "primary_vocalist": row["primary_vocalist"],
        "is_instrumental": is_instrumental,
        "lyrics_available": bool(
            row.get("lyrics_available")
        ),
        **token_features,
        **structure_features,
        **repetition_features,
        **pronoun_theme_features,
        **top_token_features,
        **readability_features,
        **vader_features,
        **nrc_features,
    }

    coverage_row = {
        "master_track_id": row["master_track_id"],
        "album": row["album"],
        "track_title": row["track_title"],
        "is_instrumental": is_instrumental,
        "lyrics_available": bool(
            row.get("lyrics_available")
        ),
        "readability_source": readability_source,
        "vader_available": vader_analyzer is not None,
        "nrc_source": nrc_source,
        "nlp_features_built": True,
    }

    return feature_row, coverage_row


def validate_features(
    features_df: pd.DataFrame,
    input_df: pd.DataFrame,
) -> None:
    """Validate row counts, IDs, and expected feature domains."""

    if len(features_df) != len(input_df):
        raise ValueError(
            "NLP feature row count does not match input row count."
        )

    if features_df["master_track_id"].duplicated().any():
        raise ValueError(
            "Duplicate master_track_id values found."
        )

    missing_ids = (
        set(input_df["master_track_id"])
        - set(features_df["master_track_id"])
    )

    if missing_ids:
        raise ValueError(
            f"NLP features are missing track IDs: {sorted(missing_ids)}"
        )

    bounded_ratio_columns = [
    "type_token_ratio",
    "hapax_ratio",
    "dis_legomena_ratio",
    "token_repetition_ratio",
    "long_word_ratio",
    "very_long_word_ratio",
    "lexical_density_proxy",
    "line_repetition_ratio",
    "punctuation_density",
    "uppercase_ratio",
    "top_5_word_share",
    "first_person_singular_ratio",
    "first_person_plural_ratio",
    "second_person_ratio",
    "third_person_ratio",
    "negation_ratio",
    "absolutist_ratio",
    "theme_body_ratio",
    "theme_darkness_ratio",
    "theme_pain_ratio",
    "theme_hope_ratio",
    "theme_conflict_ratio",
    "theme_isolation_ratio",
    "theme_time_ratio",]

    bounded_ratio_columns.extend(
        [
            f"nrc_{emotion}_ratio"
            for emotion in NRC_EMOTIONS
        ]
    )

    for column in bounded_ratio_columns:
        if column not in features_df.columns:
            continue

        values = pd.to_numeric(
            features_df[column],
            errors="coerce",
        ).dropna()

        invalid = values[
            (values < 0)
            | (values > 1)
        ]

        if not invalid.empty:
            raise ValueError(
                f"Feature {column} contains values outside [0, 1]."
            )


def print_summary(
    features_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> None:
    """Print a compact feature-generation summary."""

    print("\nNLP feature summary:")
    print(f"Tracks processed: {len(features_df)}")
    print(
        "Tracks with lyrics: "
        f"{int(features_df['lyrics_available'].sum())}"
    )
    print(
        "Instrumental tracks: "
        f"{int(features_df['is_instrumental'].sum())}"
    )
    print(
        "Total feature columns: "
        f"{len(features_df.columns)}"
    )
    print(
        "VADER available: "
        f"{bool(coverage_df['vader_available'].all())}"
    )

    nrc_counts = (
        coverage_df["nrc_source"]
        .value_counts(dropna=False)
    )

    print("\nNRC sources:")
    print(nrc_counts.to_string())

    print("\nAlbum-level averages:")

    album_summary = (
        features_df[
            ~features_df["is_instrumental"]
        ]
        .groupby(
            [
                "album_order",
                "album",
            ],
            dropna=False,
        )
        .agg(
            tracks=("master_track_id", "count"),
            avg_words=("word_count", "mean"),
            avg_lexical_diversity=(
                "type_token_ratio",
                "mean",
            ),
            avg_vader_compound=(
                "vader_compound",
                "mean",
            ),
            avg_nrc_anger=(
                "nrc_anger_ratio",
                "mean",
            ),
            avg_nrc_joy=(
                "nrc_joy_ratio",
                "mean",
            ),
            avg_nrc_fear=(
                "nrc_fear_ratio",
                "mean",
            ),
            avg_nrc_sadness=(
                "nrc_sadness_ratio",
                "mean",
            ),
        )
        .reset_index()
        .sort_values("album_order")
    )

    numeric_columns = album_summary.select_dtypes(
        include="number"
    ).columns

    album_summary[numeric_columns] = (
        album_summary[numeric_columns]
        .round(4)
    )

    print(album_summary.to_string(index=False))


def main() -> None:
    """Build and save all NLP feature tables."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Clean lyrics file not found: {INPUT_PATH}\n"
            "Run clean_lyrics.py first."
        )

    lyrics_df = pd.read_parquet(INPUT_PATH)

    required_columns = {
        "master_track_id",
        "album_order",
        "album",
        "release_year",
        "track_position",
        "track_title",
        "canonical_title",
        "era",
        "primary_vocalist",
        "is_instrumental",
        "lyrics_available",
        "lyrics_clean",
    }

    missing_columns = (
        required_columns
        - set(lyrics_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Clean lyrics data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    vader_analyzer = create_vader_analyzer()

    feature_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for counter, (_, row) in enumerate(
        lyrics_df.iterrows(),
        start=1,
    ):
        print(
            f"[{counter}/{len(lyrics_df)}] "
            f"{row['album']} — {row['track_title']}"
        )

        try:
            feature_row, coverage_row = build_feature_row(
                row=row,
                vader_analyzer=vader_analyzer,
            )

            feature_rows.append(feature_row)
            coverage_rows.append(coverage_row)

        except Exception as error:
            error_rows.append(
                {
                    "master_track_id": row["master_track_id"],
                    "album": row["album"],
                    "track_title": row["track_title"],
                    "error": str(error),
                }
            )

            print(f"  Error: {error}")

    if error_rows:
        ERRORS_OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(error_rows).to_csv(
            ERRORS_OUTPUT_PATH,
            index=False,
            encoding="utf-8",
        )

        raise RuntimeError(
            "NLP feature generation failed for one or more tracks. "
            f"See: {ERRORS_OUTPUT_PATH}"
        )

    features_df = (
        pd.DataFrame(feature_rows)
        .sort_values(
            ["album_order", "track_position"],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    coverage_df = (
        pd.DataFrame(coverage_rows)
        .sort_values(
            ["album", "track_title"]
        )
        .reset_index(drop=True)
    )

    validate_features(
        features_df=features_df,
        input_df=lyrics_df,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features_df.to_parquet(
        PARQUET_OUTPUT_PATH,
        index=False,
    )

    features_df.to_csv(
        CSV_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    coverage_df.to_csv(
        COVERAGE_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print_summary(
        features_df=features_df,
        coverage_df=coverage_df,
    )

    print(f"\nSaved: {PARQUET_OUTPUT_PATH}")
    print(f"Saved: {CSV_OUTPUT_PATH}")
    print(f"Saved: {COVERAGE_OUTPUT_PATH}")
    print("NLP feature generation completed.")


if __name__ == "__main__":
    main()
