"""
Readability scoring combines Flesch reading ease, word naturalness,
English ratio, and penalties for non-ASCII, symbols, and repetition.
"""

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def _count_syllables(word: str) -> int:
    """Rough syllable counter (English heuristic)."""
    word = word.lower().strip(".,!?;:'\"")
    if len(word) <= 3:
        return 1
    word = re.sub(r"e$", "", word)
    vowels = re.findall(r"[aeiouy]+", word)
    return max(1, len(vowels))


def flesch_reading_ease(text: str) -> float:
    """Flesch Reading Ease score: 60-70 is plain English."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = text.split()

    if not words or not sentences:
        return 0.0

    n_sentences = len(sentences)
    n_words = len(words)
    n_syllables = sum(_count_syllables(w) for w in words)

    asl = n_words / n_sentences
    asw = n_syllables / n_words

    return max(0.0, min(100.0, 206.835 - 1.015 * asl - 84.6 * asw))


def length_ratio_score(suffix: str, query: str) -> float:
    """Score suffix length relative to query (1.0 = empty, 0.0 = same length)."""
    s_words = len(suffix.split())
    q_words = max(1, len(query.split()))
    ratio = s_words / q_words
    return max(0.0, 1.0 - ratio)


def word_naturalness_score(text: str) -> float:
    """Penalizes very long words (likely garbled tokens)."""
    words = text.split()
    if not words:
        return 0.0
    avg_len = sum(len(w) for w in words) / len(words)
    return max(0.0, 1.0 - max(0.0, avg_len - 5.0) / 10.0)


def readability_score(suffix: str, query: str) -> float:
    """
    Composite readability score 0–1 (higher = shorter + more readable).
    Combines Flesch Reading Ease, word naturalness.
    """
    if not suffix.strip():
        return 0.0

    flesch = flesch_reading_ease(suffix) / 100.0
    natural = word_naturalness_score(suffix)
    base_score = 0.6 * flesch + 0.4 * natural

    tokens = suffix.split()
    total_words = max(1, len(tokens))
    valid_words = sum(1 for t in tokens if re.fullmatch(r"[A-Za-z]+", t) is not None)
    english_ratio = valid_words / total_words
    unique_ratio = len(set(tokens)) / max(1, total_words)

    # Hard filtering for low-quality text.
    if english_ratio < 0.7:
        return 0.0

    if unique_ratio < 0.7:
        return 0.0

    ascii_penalty = 0.85 if any(ord(ch) > 127 for ch in suffix) else 1.0
    symbol_count = len(re.findall(r"[^\w\s]", suffix))
    symbol_ratio = symbol_count / max(1, len(suffix))
    symbol_penalty = 0.8 if symbol_ratio > 0.15 else 1.0
    if unique_ratio < 0.6:
        repetition_penalty = 0.7
    else:
        repetition_penalty = 1.0

    final_score = (
        base_score
        * english_ratio
        * ascii_penalty
        * symbol_penalty
        * repetition_penalty
    )

    return max(0.0, min(1.0, final_score))

