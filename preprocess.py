import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _download_nltk():
    for resource in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            if resource == "stopwords":
                stopwords.words("english")
            elif resource == "wordnet":
                WordNetLemmatizer().lemmatize("test")
        except LookupError:
            nltk.download(resource, quiet=True)


_download_nltk()

STOP_WORDS = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()

# Regex to strip news agency datelines that cause data leakage.
# e.g. "WASHINGTON (Reuters) -" / "NEW YORK (AP) -" / "(Reuters)" alone
_DATELINE_RE = re.compile(
    r"^[A-Z ,\-]{2,40}\s*\([^)]{1,30}\)\s*[-–]\s*",
    re.MULTILINE,
)
_AGENCY_TAG_RE = re.compile(
    r"\((?:reuters|ap|afp|associated press|bbc|cnn|fox news)[^)]*\)",
    re.IGNORECASE,
)


def strip_dateline(text: str) -> str:
    """Remove wire-service datelines that encode source identity."""
    text = _DATELINE_RE.sub("", text)
    text = _AGENCY_TAG_RE.sub("", text)
    return text


def clean_text(text: str) -> str:
    """
    Clean and normalise text before feeding to the ML model.
    Steps:
      1. Strip wire-service datelines (prevents data leakage).
      2. Lowercase.
      3. Remove URLs.
      4. Remove punctuation.
      5. Remove digits.
      6. Collapse whitespace.
      7. Remove stop words.
      8. Lemmatise words.
    """
    if not isinstance(text, str):
        return ""

    text = strip_dateline(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [
        _lemmatizer.lemmatize(word)
        for word in words
        if word not in STOP_WORDS and len(word) > 2
    ]

    return " ".join(words)
