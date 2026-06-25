"""
feature_engineering.py
Extra stylometric + linguistic signals for fake news detection.
These features capture writing-style patterns that TF-IDF alone misses.
"""

import re
import string


# ---------------------------------------------------------------------------
# Readability helpers
# ---------------------------------------------------------------------------

def count_syllables(word: str) -> int:
    """Rough English syllable count (Gunning-Fog approximation)."""
    word = word.lower().strip(string.punctuation)
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # silent -e
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def flesch_kincaid_grade(text: str) -> float:
    """Return Flesch-Kincaid Grade Level (higher = harder to read)."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = text.split()
    if not sentences or not words:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    asl = len(words) / len(sentences)       # avg sentence length
    asw = syllables / len(words)            # avg syllables per word
    return 0.39 * asl + 11.8 * asw - 15.59


# ---------------------------------------------------------------------------
# Stylometric signals
# ---------------------------------------------------------------------------

def exclamation_ratio(text: str) -> float:
    if not text:
        return 0.0
    return text.count("!") / max(len(text), 1) * 100


def question_ratio(text: str) -> float:
    if not text:
        return 0.0
    return text.count("?") / max(len(text), 1) * 100


def caps_ratio(text: str) -> float:
    """Ratio of ALL-CAPS words (sensationalism signal)."""
    words = text.split()
    if not words:
        return 0.0
    caps = sum(1 for w in words if w.isupper() and len(w) > 2)
    return caps / len(words)


def quote_count(text: str) -> int:
    return text.count('"') + text.count("'")


def avg_word_length(text: str) -> float:
    words = [w.strip(string.punctuation) for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def url_count(text: str) -> int:
    return len(re.findall(r"http\S+|www\S+", text))


def numeric_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(ch.isdigit() for ch in text)
    return digits / max(len(text), 1)


def ellipsis_count(text: str) -> int:
    return text.count("...") + text.count("…")


# ---------------------------------------------------------------------------
# Emotional / clickbait vocabulary
# ---------------------------------------------------------------------------

_EMOTION_WORDS = {
    "shocking", "unbelievable", "outrage", "outrageous", "horrifying",
    "disgusting", "terrifying", "explosive", "bombshell", "scandal",
    "exposed", "conspiracy", "hoax", "debunked", "leaked", "secret",
    "hidden", "suppressed", "censored", "banned", "urgent", "breaking",
    "viral", "exclusive", "truth", "revealed", "proof", "must read",
    "share now", "wake up", "they don't want you to know",
    "mainstream media", "fake news", "deep state",
}


def emotional_word_ratio(text: str) -> float:
    if not text:
        return 0.0
    words = set(text.lower().split())
    hits = sum(1 for w in _EMOTION_WORDS if w in text.lower())
    return hits / max(len(text.split()), 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(text: str) -> dict:
    """Return a dict of all stylometric features for a single text."""
    return {
        "flesch_kincaid_grade": flesch_kincaid_grade(text),
        "exclamation_ratio":    exclamation_ratio(text),
        "question_ratio":       question_ratio(text),
        "caps_ratio":           caps_ratio(text),
        "avg_word_length":      avg_word_length(text),
        "url_count":            url_count(text),
        "numeric_ratio":        numeric_ratio(text),
        "ellipsis_count":       ellipsis_count(text),
        "emotional_word_ratio": emotional_word_ratio(text),
        "quote_count":          quote_count(text),
        "text_length":          len(text),
        "word_count":           len(text.split()),
    }
