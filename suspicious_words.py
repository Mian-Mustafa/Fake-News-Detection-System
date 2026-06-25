import html
import re


SUSPICIOUS_PHRASES = [
    "shocking",
    "unbelievable",
    "urgent",
    "share now",
    "hidden truth",
    "exposed",
    "breaking",
    "secret",
    "viral",
    "dangerous",
    "you won't believe",
    "government does not want you to know",
    "mainstream media",
    "must read",
    "truth revealed",
    "share before deleted",
]


def find_suspicious_phrases(text):
    """
    Return suspicious emotional or misleading phrases found in the text.
    """
    if not isinstance(text, str):
        return []

    text_lower = text.lower()
    found_phrases = []

    for phrase in SUSPICIOUS_PHRASES:
        if phrase in text_lower:
            found_phrases.append(phrase)

    return found_phrases


def highlight_suspicious_phrases(text):
    """
    Highlight suspicious phrases so they can be displayed in Streamlit.
    """
    if not isinstance(text, str):
        return ""

    escaped_text = html.escape(text)

    for phrase in find_suspicious_phrases(text):
        pattern = re.compile(re.escape(html.escape(phrase)), re.IGNORECASE)
        escaped_text = pattern.sub(
            lambda match: f"<mark>{match.group(0)}</mark>",
            escaped_text,
        )

    return escaped_text.replace("\n", "<br>")
