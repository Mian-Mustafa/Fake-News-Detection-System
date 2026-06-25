import concurrent.futures
import json
import re
import time
import base64
import io
import urllib.parse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from openai import OpenAI

from preprocess import clean_text
from suspicious_words import find_suspicious_phrases, highlight_suspicious_phrases


# ── Paths ─────────────────────────────────────────────────────────────────────

MODEL_PATH      = Path("models/best_model.pkl")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.pkl")
SCORES_PATH     = Path("results/model_scores.csv")

# ── API configuration ─────────────────────────────────────────────────────────

# Google API key — powers Gemini + Fact Check Tools API
_DEFAULT_API_KEY = "AIzaSyDCi6MWaSS8DXDWUpFBks8SDp41JRFy_SQ"
# NewsAPI key
_NEWS_API_KEY    = "bddd5a0c0b904e7cbfc0221f0e92f71c"
# OpenAI API key
_OPENAI_API_KEY  = "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"

GEMINI_V2_URL  = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
GEMINI_V1_URL  = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
FACT_CHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
NEWS_API_URL   = "https://newsapi.org/v2/everything"
HF_API_URL     = "https://api-inference.huggingface.co/models/hamzab/roberta-fake-news-classification"

# Domain keywords for reputable-source detection in NewsAPI results
_REPUTABLE_DOMAINS = frozenset({
    "bbc", "reuters", "apnews", "cnn", "nytimes", "theguardian", "washingtonpost",
    "aljazeera", "nbcnews", "cbsnews", "abcnews", "politico", "thehill", "axios",
    "bloomberg", "ft.com", "economist", "usatoday", "newsweek", "independent",
    "telegraph", "time.com", "wired", "fortune", "snopes", "politifact", "factcheck",
    "afp", "dw.com", "france24", "euronews", "npr", "pbs", "vox", "theatlantic",
    "foreignpolicy", "foreignaffairs", "sciencemag", "nature", "newscientist",
})

# ── Verdict constants ─────────────────────────────────────────────────────────

V_VERIFIED_TRUE  = "Verified True"
V_VERIFIED_FALSE = "Verified False"
V_MISLEADING     = "Misleading or Partly False"
V_REAL           = "Real News"
V_FAKE           = "Fake News"
V_INSUFFICIENT   = "Not Enough Evidence"

# ── Confidence thresholds ─────────────────────────────────────────────────────

FC_MIN     = 0.60
GEMINI_MIN = 0.58
HF_MIN     = 0.62
LOCAL_MIN  = 0.60

# ── Rating keyword sets ───────────────────────────────────────────────────────

_FALSE_KW = frozenset({
    "false", "fake", "wrong", "incorrect", "debunked", "fabricated",
    "pants on fire", "inaccurate", "scam", "hoax", "lie", "bogus",
    "unfounded", "baseless", "not true",
})
_TRUE_KW = frozenset({
    "true", "correct", "accurate", "verified", "confirmed",
    "factual", "real", "legitimate", "accurate",
})
_MIXED_KW = frozenset({
    "misleading", "partly false", "partially false", "mixed", "half true",
    "unverified", "lacks context", "out of context", "disputed",
    "needs context", "partially accurate", "mostly false", "mostly true",
    "exaggerated", "distorted", "spin",
})


def _classify_rating(rating: str) -> str:
    r = rating.lower()
    if any(k in r for k in _MIXED_KW):
        return "mixed"
    if any(k in r for k in _FALSE_KW):
        return "false"
    if any(k in r for k in _TRUE_KW):
        return "true"
    return "unknown"


# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_local_model():
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


# ── 1. Google Fact Check Tools API ───────────────────────────────────────────

_GEMINI_FC_DIRECT_PROMPT = """\
Search these fact-checking websites to find if this claim has been verified:
Snopes.com, PolitiFact.com, FactCheck.org, AFP Fact Check, Reuters Fact Check, BBC Reality Check, AP Fact Check.

Claim to check:
{query}

If you find a fact-check result, respond ONLY with this JSON:
{{"verdict": "Verified True" or "Verified False" or "Misleading or Partly False" or "Not Enough Evidence", \
"confidence": <integer 60-97>, \
"reason": "<which site rated this and what was the verdict>", \
"publisher": "<fact-check site name>", \
"rating": "<their exact rating label>"}}"""


def predict_fact_check_api(
    text: str, api_key: str
) -> tuple[str | None, float, list[dict]]:
    """
    Tries the real Google Fact Check Tools API first.
    If it's blocked (403 — API not enabled), falls back to a live Gemini search
    of Snopes, PolitiFact, AFP Fact Check and similar sites.
    Returns (verdict, confidence, fact_check_reports).
    """
    key = api_key.strip() or _DEFAULT_API_KEY
    query = _extract_query(text)
    if not query:
        return None, 0.0, []

    # ── Attempt 1: real Fact Check Tools API ──────────────────────────────
    if key:
        try:
            resp = requests.get(
                FACT_CHECK_URL,
                params={"query": query, "languageCode": "en", "pageSize": 10, "key": key},
                timeout=20,
            )
            if resp.status_code == 200:
                claims_raw = resp.json().get("claims", [])
                reports: list[dict] = []
                for claim in claims_raw:
                    claim_text = claim.get("text", "").strip()
                    for review in claim.get("claimReview", []):
                        rating = review.get("textualRating", "").strip()
                        if not rating:
                            continue
                        reports.append({
                            "claim":     claim_text,
                            "rating":    rating,
                            "category":  _classify_rating(rating),
                            "publisher": review.get("publisher", {}).get("name", "Unknown"),
                            "url":       review.get("url", ""),
                            "date":      review.get("reviewDate", ""),
                        })
                if reports:
                    false_n = sum(1 for r in reports if r["category"] == "false")
                    true_n  = sum(1 for r in reports if r["category"] == "true")
                    mixed_n = sum(1 for r in reports if r["category"] == "mixed")
                    total   = len(reports)
                    if mixed_n >= total * 0.4 or (false_n > 0 and true_n > 0):
                        return V_MISLEADING, 0.70, reports[:5]
                    elif false_n > true_n and false_n >= total * 0.5:
                        return V_VERIFIED_FALSE, min(0.62 + (false_n / total) * 0.33, 0.97), reports[:5]
                    elif true_n > false_n and true_n >= total * 0.5:
                        return V_VERIFIED_TRUE,  min(0.62 + (true_n  / total) * 0.33, 0.97), reports[:5]
        except Exception:
            pass

    # ── Attempt 2: Gemini live search of fact-check sites (fires when API is 403) ──
    if not key:
        return None, 0.0, []
    try:
        payload = {
            "contents": [{"parts": [{"text": _GEMINI_FC_DIRECT_PROMPT.format(query=query[:800])}]}],
            "tools":    [{"google_search": {}}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 300},
        }
        resp = requests.post(f"{GEMINI_V2_URL}?key={key}", json=payload, timeout=35)
        if resp.status_code == 200:
            raw_text = ""
            for part in resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", []):
                if "text" in part and part["text"].strip():
                    raw_text = part["text"]
                    break
            result = _extract_json_from_text(raw_text)
            if result:
                verdict   = result.get("verdict", "").strip()
                conf      = min(max(float(result.get("confidence", 60)), 50), 97) / 100
                reason    = result.get("reason", "").strip()
                publisher = result.get("publisher", "Gemini Fact Search")
                rating    = result.get("rating", verdict)
                if verdict in _VALID_VERDICTS and reason:
                    report = {
                        "claim":     query[:200],
                        "rating":    rating,
                        "category":  _classify_rating(rating),
                        "publisher": publisher,
                        "url":       "",
                        "date":      "",
                    }
                    return verdict, conf, [report]
    except Exception:
        pass

    return None, 0.0, []


def _extract_query(text: str) -> str:
    """Extract a short, search-friendly claim from article text."""
    cleaned = re.sub(r"^[A-Z\s]+\s*\([^)]+\)\s*[-–—]\s*", "", text.strip())
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return " ".join(sentences[:2])[:300].strip()


# ── 2. Gemini Fact-Check Search ───────────────────────────────────────────────

_FC_SEARCH_PROMPT = """\
You are an expert fact-checker. Search these fact-checking databases for this claim:
Snopes, PolitiFact, FactCheck.org, AFP Fact Check, Reuters Fact Check, BBC Reality Check.

Claim / Article:
{text}

After searching, answer:
1. Has this claim been fact-checked by any organisation?
2. What rating did they give? (True / False / Misleading / Partly False / Unverified)
3. What evidence did you find?

Respond ONLY with this exact JSON:
{{"verdict": "Verified True" or "Verified False" or "Misleading or Partly False" or "Real News" or "Fake News" or "Not Enough Evidence", \
"confidence": <integer 50-98>, \
"reason": "<2-3 sentences: which sites confirmed or denied this and what they found>", \
"fact_check_sites": ["site1", "site2"]}}"""

_WEB_SEARCH_PROMPT = """\
You are an expert fact-checker with real-time Google Search access.
Verify this news article by searching the web.

Search for:
• Coverage by BBC, Reuters, AP, CNN, NYT, The Guardian
• Fact-checks from Snopes, PolitiFact, FactCheck.org, AFP Fact Check
• Key claims, statistics, names, dates
• Official statements or retractions

Article:
{text}

Respond ONLY with this exact JSON:
{{"verdict": "Verified True" or "Verified False" or "Misleading or Partly False" or "Real News" or "Fake News", \
"confidence": <integer 50-98>, \
"reason": "<2-3 sentences on what you found online>"}}"""

_PLAIN_PROMPT = """\
You are a professional fact-checker. Analyse this article and decide if it is real or fake news.

Check for: sensationalist language, unverifiable claims, conspiracy framing, logical inconsistencies.

Article:
{text}

Respond ONLY with this JSON:
{{"verdict": "Real News" or "Fake News" or "Misleading or Partly False", \
"confidence": <integer 50-98>, \
"reason": "<one clear sentence>"}}"""

_WEB_SEARCH_WITH_CONTEXT_PROMPT = """\
You are an expert fact-checker with real-time Google Search access.
Verify this news article by searching the web.

NewsAPI has already found these related news articles:
{newsapi_context}

Search for:
• Coverage by BBC, Reuters, AP, CNN, NYT, The Guardian
• Fact-checks from Snopes, PolitiFact, FactCheck.org, AFP Fact Check
• Key claims, statistics, names, dates
• Official statements or retractions

Article to verify:
{text}

Respond ONLY with this exact JSON:
{{"verdict": "Verified True" or "Verified False" or "Misleading or Partly False" or "Real News" or "Fake News", \
"confidence": <integer 50-98>, \
"reason": "<2-3 sentences on what you found online, considering the news coverage above>"}}"""

_VALID_VERDICTS = (
    V_VERIFIED_TRUE, V_VERIFIED_FALSE, V_MISLEADING,
    V_REAL, V_FAKE, V_INSUFFICIENT,
)


def _extract_json_from_text(raw: str) -> dict | None:
    """
    Robust JSON extraction from Gemini text responses.
    Handles: plain JSON, markdown code blocks, text-wrapped JSON, nested objects.
    """
    # Strip markdown code fences
    stripped = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

    # Strategy 1: entire response is valid JSON
    try:
        return json.loads(stripped)
    except Exception:
        pass

    # Strategy 2: balanced-brace extraction (handles nested objects)
    depth, start = 0, -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(raw[start : i + 1])
                except Exception:
                    start = -1  # keep searching

    return None


def _parse_gemini_response(
    data: dict,
) -> tuple[str | None, float, str, list[dict], list[str]]:
    """Parse Gemini REST response → (verdict, conf, reason, web_sources, queries)."""
    try:
        candidate = data["candidates"][0]

        # Look through ALL parts for one that contains text (google_search tool
        # sometimes prepends a function-call part with no "text" key)
        raw = ""
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part and part["text"].strip():
                raw = part["text"]
                break
        if not raw:
            return None, 0.0, "", [], []

        sources: list[dict] = []
        queries: list[str]  = []
        grounding = candidate.get("groundingMetadata", {})
        for chunk in grounding.get("groundingChunks", []):
            web = chunk.get("web", {})
            if web.get("uri"):
                sources.append({"uri": web["uri"], "title": web.get("title", web["uri"])})
        queries = grounding.get("webSearchQueries", [])

        result = _extract_json_from_text(raw)
        if not result:
            return None, 0.0, "", sources, queries

        verdict = result.get("verdict", "").strip()
        conf    = min(max(float(result.get("confidence", 50)), 50), 98) / 100
        reason  = result.get("reason", "").strip()

        if verdict in _VALID_VERDICTS:
            if sources:
                conf = min(conf * 1.04, 0.98)
            return verdict, conf, reason, sources, queries
    except Exception:
        pass

    return None, 0.0, "", [], []


def predict_gemini_factcheck(
    text: str, api_key: str
) -> tuple[str | None, float, str, list[dict], list[str]]:
    """
    Gemini 2.5 Flash searching fact-check databases (Snopes, PolitiFact, AFP, etc.).
    Only uses gemini-2.5-flash + google_search — gemini-2.0-flash has 0 free-tier quota.
    """
    key = api_key.strip() or _DEFAULT_API_KEY
    if not key:
        return None, 0.0, "", [], []

    snippet = text[:2500]
    payload = {
        "contents": [{"parts": [{"text": _FC_SEARCH_PROMPT.format(text=snippet)}]}],
        "tools":    [{"google_search": {}}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 500},
    }
    try:
        resp = requests.post(f"{GEMINI_V2_URL}?key={key}", json=payload, timeout=35)
        if resp.status_code == 200:
            v, c, r, srcs, qs = _parse_gemini_response(resp.json())
            if v:
                return v, c, r, srcs, qs
    except Exception:
        pass

    return None, 0.0, "", [], []


def predict_gemini_web(
    text: str, api_key: str, newsapi_articles: list[dict] | None = None
) -> tuple[str | None, float, str, list[dict], list[str]]:
    """
    Gemini with broad web search — news outlets + evidence gathering.
    Accepts optional newsapi_articles from Phase 1 to enrich the prompt context.
    Returns (verdict, confidence, reason, web_sources, search_queries).
    """
    key = api_key.strip() or _DEFAULT_API_KEY
    if not key:
        return None, 0.0, "", [], []

    snippet = text[:3000]

    # Build prompt — use context-enriched version when NewsAPI articles available
    if newsapi_articles:
        context_lines = []
        for art in newsapi_articles[:5]:
            src   = art.get("source", "Unknown")
            rep   = " [REPUTABLE]" if art.get("reputable") else ""
            title = art.get("title", "")
            if title:
                context_lines.append(f"- {src}{rep}: {title}")
        newsapi_context = "\n".join(context_lines) if context_lines else "No related articles found."
        primary_prompt  = _WEB_SEARCH_WITH_CONTEXT_PROMPT.format(
            newsapi_context=newsapi_context, text=snippet
        )
    else:
        primary_prompt = _WEB_SEARCH_PROMPT.format(text=snippet)

    # Only gemini-2.5-flash + google_search is used — gemini-2.0-flash has 0 free quota.
    try:
        payload = {
            "contents": [{"parts": [{"text": primary_prompt}]}],
            "tools":    [{"google_search": {}}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 600},
        }
        resp = requests.post(f"{GEMINI_V2_URL}?key={key}", json=payload, timeout=35)
        if resp.status_code == 200:
            v, c, r, srcs, qs = _parse_gemini_response(resp.json())
            if v:
                return v, c, r, srcs, qs
    except Exception:
        pass

    return None, 0.0, "", [], []


# ── 3. NewsAPI Coverage Analysis ─────────────────────────────────────────────

def predict_newsapi(
    text: str, api_key: str = ""
) -> tuple[str | None, float, list[dict], int, int]:
    """
    NewsAPI.org — searches thousands of publishers for related articles.
    Measures how widely a story is covered by reputable sources.

    Returns (coverage_signal, confidence, articles, total_results, reputable_count).

    coverage_signal logic:
    • ≥3 reputable outlets found  → V_REAL   (0.68–0.80)
    • 1–2 reputable outlets found → V_REAL   (0.62–0.67)
    • 0 results at all            → V_FAKE   (0.55) — claim absent from all news
    • results but 0 reputable     → None     (ambiguous — could be fringe / new)
    """
    key = api_key.strip() or _NEWS_API_KEY
    if not key:
        return None, 0.0, [], 0, 0

    query = _extract_query(text)
    if len(query) < 10:
        return None, 0.0, [], 0, 0

    try:
        resp = requests.get(
            NEWS_API_URL,
            params={
                "q":        query,
                "language": "en",
                "sortBy":   "relevancy",
                "pageSize": 10,
                "apiKey":   key,
            },
            timeout=20,
        )
        if resp.status_code != 200:
            return None, 0.0, [], 0, 0

        data         = resp.json()
        total        = data.get("totalResults", 0)
        raw_articles = data.get("articles", [])

        articles: list[dict] = []
        reputable_count = 0

        for art in raw_articles:
            src_name  = (art.get("source") or {}).get("name", "") or ""
            src_url   = (art.get("url") or "").lower()
            title     = art.get("title") or ""
            desc      = art.get("description") or ""
            url       = art.get("url") or ""
            published = (art.get("publishedAt") or "")[:10]

            # Decide if source is reputable
            combined = (src_name + " " + src_url).lower()
            is_rep   = any(kw in combined for kw in _REPUTABLE_DOMAINS)
            if is_rep:
                reputable_count += 1

            articles.append({
                "title":      title,
                "source":     src_name,
                "url":        url,
                "published":  published,
                "description": desc[:200],
                "reputable":  is_rep,
            })

        # Coverage signal
        if total == 0:
            signal = V_FAKE
            conf   = 0.55
        elif reputable_count >= 3:
            signal = V_REAL
            conf   = min(0.68 + (reputable_count / 10) * 0.12, 0.80)
        elif reputable_count >= 1:
            signal = V_REAL
            conf   = 0.62 + (reputable_count - 1) * 0.025
        else:
            signal = None   # non-reputable coverage — ambiguous
            conf   = 0.0

        return signal, conf, articles[:8], total, reputable_count

    except Exception:
        return None, 0.0, [], 0, 0


# ── 4. HuggingFace RoBERTa ───────────────────────────────────────────────────

_HF_MODELS = [
    "hamzab/roberta-fake-news-classification",
    "jy46604790/Fake-News-Bert-Detect",
]


def predict_hf(text: str, token: str = "") -> tuple[str | None, float]:
    """
    HuggingFace classifier — tries two public models.
    Uses X-Wait-For-Model header to survive cold-start delays.
    Handles DNS/connection failures gracefully.
    """
    base_headers: dict = {"X-Wait-For-Model": "true"}
    if token.strip():
        base_headers["Authorization"] = f"Bearer {token}"

    for model_id in _HF_MODELS:
        url = f"https://api-inference.huggingface.co/models/{model_id}"

        def _post(url=url):
            try:
                r = requests.post(
                    url, headers=base_headers,
                    json={"inputs": text[:1500]}, timeout=30,
                )
                if r.status_code == 200:
                    return r.json()
                if r.content:
                    return r.json()
                return None
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    Exception):
                return None

        data = _post()

        # Model still loading — wait once then retry
        if isinstance(data, dict) and "error" in data:
            wait = min(float(data.get("estimated_time", 20)), 25)
            time.sleep(wait)
            data = _post()

        if not data:
            continue  # try next model

        try:
            inner = data[0] if isinstance(data[0], list) else data
            top   = max(inner, key=lambda x: x["score"])
            raw   = top["label"].lower()
            score = float(top["score"])
            label = "Fake News" if ("fake" in raw or raw in ("label_0", "0", "false")) else "Real News"
            return label, score
        except Exception:
            continue

    return None, 0.0


# ── 5. OpenAI GPT-4o Web Search Analysis ─────────────────────────────────────

_OPENAI_SYSTEM_PROMPT = (
    "You are a professional fact-checker and investigative journalist. "
    "Analyze news articles for authenticity. Be objective and evidence-based."
)

_OPENAI_USER_PROMPT = """\
Analyze the following news article or claim and determine if it is real or fake.

Evaluate:
1. Language: sensationalism, emotional manipulation, clickbait, exaggeration
2. Sources: named sources, official statements, expert quotes present or absent
3. Factual consistency: verifiable claims, statistics, dates, named entities
4. Logical coherence: sound reasoning, no logical fallacies
5. Misinformation patterns: conspiracy framing, fear-mongering, anti-establishment bias

Article / Claim:
{text}

Respond ONLY with this exact JSON (no markdown, no extra text):
{{"verdict": "Real News" or "Fake News" or "Misleading or Partly False" or "Not Enough Evidence", "confidence": <integer 50-98>, "reason": "<2-3 sentences with key indicators>"}}"""

_OPENAI_WEB_SEARCH_PROMPT = """\
Use your web search capability to verify whether the following news article or claim is real or fake.

Search for:
1. Coverage by BBC, Reuters, AP News, CNN, New York Times, The Guardian, Al Jazeera
2. Fact-checks from Snopes, PolitiFact, FactCheck.org, AFP Fact Check, Reuters Fact Check
3. Official statements, government sources, or expert commentary related to the claims
4. Any evidence that confirms, contradicts, or debunks the key claims
5. Related news stories or investigations published recently

News Article / Claim:
{text}

After your web search, respond ONLY with this exact JSON (no markdown):
{{"verdict": "Verified True" or "Verified False" or "Misleading or Partly False" or "Real News" or "Fake News" or "Not Enough Evidence", "confidence": <integer 50-98>, "reason": "<2-3 sentences citing specific sources you found and what they say>"}}"""


def predict_openai(text: str, api_key: str = "") -> tuple[str | None, float, str]:
    """
    OpenAI GPT-4o with real-time web search — searches news outlets and fact-check
    databases live before rendering a verdict.
    Falls back to GPT-4o/GPT-4o-mini linguistic analysis if web search fails.
    Returns (verdict, confidence, reason).
    """
    key = api_key.strip() or _OPENAI_API_KEY
    if not key:
        return None, 0.0, ""

    snippet = text[:3000]
    client  = OpenAI(api_key=key)

    # ── Path A: Responses API + GPT-4o with live web search ──────────────
    try:
        response = client.responses.create(
            model="gpt-4o",
            tools=[{"type": "web_search_preview"}],
            instructions=_OPENAI_SYSTEM_PROMPT,
            input=_OPENAI_WEB_SEARCH_PROMPT.format(text=snippet),
            temperature=0.1,
        )
        raw_text = getattr(response, "output_text", "") or ""
        if not raw_text:
            for item in (getattr(response, "output", []) or []):
                if getattr(item, "type", "") == "message":
                    for part in (getattr(item, "content", []) or []):
                        t = getattr(part, "text", "")
                        if t and t.strip():
                            raw_text = t
                            break
                if raw_text:
                    break
        result = _extract_json_from_text(raw_text)
        if result:
            verdict = result.get("verdict", "").strip()
            conf    = min(max(float(result.get("confidence", 50)), 50), 98) / 100
            reason  = result.get("reason", "").strip()
            if verdict in _VALID_VERDICTS:
                return verdict, min(conf * 1.05, 0.98), reason
    except Exception:
        pass

    # ── Path B: Chat Completions — forces strict JSON via response_format ─
    for model in ("gpt-4o", "gpt-4o-mini"):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _OPENAI_SYSTEM_PROMPT},
                    {"role": "user",   "content": _OPENAI_USER_PROMPT.format(text=snippet)},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=400,
            )
            raw    = resp.choices[0].message.content
            result = _extract_json_from_text(raw)
            if not result:
                continue
            verdict = result.get("verdict", "").strip()
            conf    = min(max(float(result.get("confidence", 50)), 50), 98) / 100
            reason  = result.get("reason", "").strip()
            if verdict in _VALID_VERDICTS:
                return verdict, conf, reason
        except Exception:
            continue

    return None, 0.0, ""


# ── 6. Local SVM ─────────────────────────────────────────────────────────────

def predict_local(text: str, model, vectorizer) -> tuple[str, float, str]:
    cleaned = clean_text(text)
    if not cleaned.strip():
        return V_INSUFFICIENT, 0.0, cleaned

    feats = vectorizer.transform([cleaned])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feats)[0]
        idx   = int(np.argmax(probs))
        conf  = float(probs[idx])
        raw   = model.classes_[idx]
    else:
        raw  = model.predict(feats)[0]
        conf = 1.0

    label = _fmt(raw)
    return (V_INSUFFICIENT if conf < LOCAL_MIN else label), conf, cleaned


def _fmt(raw) -> str:
    s = str(raw).lower().strip()
    if s in ("real", "true", "1"):  return V_REAL
    if s in ("fake", "false", "0"): return V_FAKE
    return s.title()


# ── Ensemble — parallel execution + weighted consensus voting ─────────────────

# Authority weights for each source (higher = more trusted)
_SOURCE_WEIGHTS = {
    "Fact Check API":    4.0,   # human fact-checkers — gold standard
    "Gemini FC Search":  3.0,   # AI targeting Snopes / PolitiFact / AFP
    "GPT-4o Web Search":   2.8,   # GPT-4o with live web search — real-time source verification
    "Gemini Web Search": 2.5,   # broad live web analysis
    "NewsAPI Coverage":  2.0,   # publisher breadth + reputable-source count
    "RoBERTa":           1.5,   # fine-tuned transformer classifier
    "Local SVM":         1.8,   # offline TF-IDF + SVM — 99.38% F1, always available
}


def _to_signal(verdict: str | None) -> float | None:
    """Map a verdict string to a numeric signal: +1 real, -1 fake, None = abstain."""
    if verdict in (V_VERIFIED_TRUE, V_REAL, "Real News"):       return +1.0
    if verdict in (V_VERIFIED_FALSE, V_FAKE, "Fake News"):      return -1.0
    if verdict == V_MISLEADING:                                   return -0.45
    if verdict == V_INSUFFICIENT:                                 return  0.0
    return None


def _weighted_vote(
    ballots: list[tuple[str, str | None, float]],
) -> tuple[str, float, dict]:
    """
    Weighted consensus vote across all 7 sources.

    ballots : [(source_name, verdict, confidence), ...]
    Returns : (final_verdict, final_confidence, breakdown_dict)

    breakdown_dict keys  → source name
    each value           → {"verdict", "conf", "signal", "weight", "pct"}
    plus "__meta__"      → {"avg_signal", "n_active", "n_agree", "total_weight"}
    """
    total_w  = 0.0
    signal_w = 0.0
    conf_w   = 0.0
    breakdown: dict = {}

    for name, verdict, conf in ballots:
        sig      = _to_signal(verdict)
        base_w   = _SOURCE_WEIGHTS.get(name, 1.0)
        eff_w    = base_w * max(conf, 0.0)  # confidence-scaled weight

        breakdown[name] = {
            "verdict": verdict,
            "conf":    conf,
            "signal":  sig,
            "weight":  eff_w if (sig is not None and conf >= 0.48) else 0.0,
        }

        if sig is None or conf < 0.48:
            continue

        signal_w += eff_w * sig
        conf_w   += eff_w * conf
        total_w  += eff_w

    # Compute percentage contribution for each active source
    for name in breakdown:
        w = breakdown[name]["weight"]
        breakdown[name]["pct"] = (w / total_w * 100) if total_w > 0 else 0.0

    if total_w == 0:
        breakdown["__meta__"] = {"avg_signal": 0.0, "n_active": 0, "n_agree": 0, "total_weight": 0.0}
        return V_INSUFFICIENT, 0.0, breakdown

    avg_signal = signal_w / total_w
    avg_conf   = conf_w   / total_w

    # Count agreeing sources
    n_active = sum(1 for n, d in breakdown.items() if n != "__meta__" and d["weight"] > 0)
    n_agree  = sum(
        1 for n, d in breakdown.items()
        if n != "__meta__" and d["weight"] > 0
        and d["signal"] is not None and d["signal"] * avg_signal > 0
    )

    # Consensus boost: every additional agreeing source adds 4% confidence
    consensus_boost = 1.0 + min(0.04 * max(n_agree - 1, 0), 0.22)
    final_conf = min(avg_conf * consensus_boost, 0.98)

    # Check if any gold-standard source gave a verified verdict
    fc_verified_true  = breakdown.get("Fact Check API", {}).get("verdict") == V_VERIFIED_TRUE
    fc_verified_false = breakdown.get("Fact Check API", {}).get("verdict") == V_VERIFIED_FALSE

    # Map weighted signal → final verdict
    if avg_signal >= 0.55:
        final_v = V_VERIFIED_TRUE if fc_verified_true else V_REAL
    elif avg_signal >= 0.12:
        final_v = V_REAL
    elif avg_signal <= -0.55:
        final_v = V_VERIFIED_FALSE if fc_verified_false else V_FAKE
    elif avg_signal <= -0.12:
        final_v = V_FAKE
    else:
        final_v = V_MISLEADING if abs(avg_signal) > 0.04 else V_INSUFFICIENT

    breakdown["__meta__"] = {
        "avg_signal":   avg_signal,
        "n_active":     n_active,
        "n_agree":      n_agree,
        "total_weight": total_w,
    }
    return final_v, final_conf, breakdown


def run_ensemble(
    text: str, model, vectorizer, api_key: str, hf_token: str,
    progress_cb=None,
):
    """
    3-phase sequential pipeline — search first, then analyse:

    Phase 1 — NewsAPI (news coverage search)
    Phase 2 — Gemini Web Search (web context, enriched by Phase 1 results)
    Phase 3 — All remaining sources in parallel:
               Fact Check API, Gemini FC Search, GPT-4o Web Search, RoBERTa, Local SVM

    progress_cb(phase: int, label: str) — optional UI callback called at each phase.

    Returns 29-tuple:
        final_verdict, final_conf,
        fc_verdict, fc_conf, fact_check_reports,
        gfc_verdict, gfc_conf, gfc_reason, gfc_sources,
        gweb_verdict, gweb_conf, gweb_reason, gweb_sources, gweb_queries,
        ns_verdict, ns_conf, ns_articles, ns_total, ns_reputable,
        hf_verdict, hf_conf,
        openai_verdict, openai_conf, openai_reason,
        local_verdict, local_conf,
        cleaned_text, primary_source, breakdown
    """
    # ── Phase 1: NewsAPI — gather news coverage first ─────────────────────
    if progress_cb:
        progress_cb(1, "Searching NewsAPI for news coverage…")
    ns_v, ns_c, ns_arts, ns_total, ns_rep = predict_newsapi(text)

    # ── Phase 2: Gemini Web Search — enriched with Phase 1 context ────────
    if progress_cb:
        progress_cb(2, "Running Gemini web search (with NewsAPI context)…")
    gweb_v, gweb_c, gweb_r, gweb_srcs, gweb_qs = predict_gemini_web(
        text, api_key, newsapi_articles=ns_arts
    )

    # ── Phase 3: Remaining sources in parallel — armed with search data ───
    if progress_cb:
        progress_cb(3, "Running Fact-Check API, Gemini FC, GPT-4o Web Search, RoBERTa & SVM…")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        fut_fc     = pool.submit(predict_fact_check_api,   text, api_key)
        fut_gfc    = pool.submit(predict_gemini_factcheck, text, api_key)
        fut_openai = pool.submit(predict_openai,           text)
        fut_hf     = pool.submit(predict_hf,               text, hf_token)
        fut_local  = pool.submit(predict_local,            text, model, vectorizer)

        fc_v,     fc_c,     fact_checks          = fut_fc.result()
        gfc_v,    gfc_c,    gfc_r, gfc_srcs, _   = fut_gfc.result()
        openai_v, openai_c, openai_r              = fut_openai.result()
        hf_v,     hf_c                            = fut_hf.result()
        local_v,  local_c,  cleaned               = fut_local.result()

    # ── Weighted consensus vote across all 7 results ──────────────────────
    ballots = [
        ("Fact Check API",    fc_v,     fc_c),
        ("Gemini FC Search",  gfc_v,    gfc_c),
        ("GPT-4o Web Search",   openai_v, openai_c),
        ("Gemini Web Search", gweb_v,   gweb_c),
        ("NewsAPI Coverage",  ns_v,     ns_c),
        ("RoBERTa",           hf_v,     hf_c),
        ("Local SVM",         local_v,  local_c),
    ]
    final_v, final_c, breakdown = _weighted_vote(ballots)

    # Primary source = the one carrying the largest share of the vote
    active = {k: v for k, v in breakdown.items() if k != "__meta__" and v["weight"] > 0}
    primary = max(active, key=lambda k: active[k]["weight"]) if active else "Local SVM"

    return (
        final_v, final_c,
        fc_v,     fc_c,    fact_checks,
        gfc_v,    gfc_c,   gfc_r,    gfc_srcs,
        gweb_v,   gweb_c,  gweb_r,   gweb_srcs, gweb_qs,
        ns_v,     ns_c,    ns_arts,  ns_total,  ns_rep,
        hf_v,     hf_c,
        openai_v, openai_c, openai_r,
        local_v,  local_c,
        cleaned,  primary, breakdown,
    )


# ── UI helpers ────────────────────────────────────────────────────────────────

# Maps verdict → (css_modifier_class, icon, subtitle)
_VERDICT_CFG = {
    V_VERIFIED_TRUE:  ("fn-v-green", "✅", "Confirmed by professional fact-checkers"),
    V_REAL:           ("fn-v-green", "✅", "Supported by AI analysis and web evidence"),
    V_VERIFIED_FALSE: ("fn-v-red",   "🚫", "Debunked by professional fact-checkers"),
    V_FAKE:           ("fn-v-red",   "🚫", "Identified as false by AI and web analysis"),
    V_MISLEADING:     ("fn-v-amber", "⚠️", "Contains misleading or inaccurate elements"),
    V_INSUFFICIENT:   ("fn-v-gray",  "🔍", "Insufficient evidence — verify from trusted sources"),
}


def _verdict_card(label: str, conf: float, source: str):
    cls, icon, subtitle = _VERDICT_CFG.get(label, ("fn-v-gray", "⚠️", ""))
    bar_pct = min(int(conf * 100), 100)
    st.markdown(
        f'<div class="fn-verdict {cls}">'
        f'  <div class="fn-v-row">'
        f'    <div>'
        f'      <div class="fn-v-label">{icon}&nbsp;{label}</div>'
        f'      <div class="fn-v-sub">{subtitle}</div>'
        f'      <div class="fn-v-src">Source: {source}</div>'
        f'    </div>'
        f'    <div class="fn-v-right">'
        f'      <div class="fn-v-pct">{conf * 100:.0f}%</div>'
        f'      <div class="fn-v-pct-lbl">Confidence</div>'
        f'    </div>'
        f'  </div>'
        f'  <div class="fn-v-track"><div class="fn-v-fill" style="width:{bar_pct}%;"></div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _mini_verdict(label: str | None, conf: float, unavailable: str = "—"):
    if label in (V_VERIFIED_TRUE, V_REAL):
        st.success(f"✅ {label}")
    elif label in (V_VERIFIED_FALSE, V_FAKE):
        st.error(f"🚫 {label}")
    elif label == V_MISLEADING:
        st.warning(f"⚠️ {label}")
    elif label is None:
        st.caption(f"_{unavailable}_")
        return
    else:
        st.warning(f"⚠️ {label}")
    if conf > 0:
        st.caption(f"{conf * 100:.0f}%")


def _rating_badge(rating: str, category: str):
    if category == "false":
        st.error(f"🚫 {rating}")
    elif category == "true":
        st.success(f"✅ {rating}")
    else:
        st.warning(f"⚠️ {rating}")


def _article_card(art: dict):
    rep_html = '<span class="fn-rep-badge">✓ Reputable</span>' if art.get("reputable") else ""
    title     = art.get("title", "Untitled")
    url       = art.get("url", "")
    src_name  = art.get("source", "")
    pub       = art.get("published", "")
    desc      = art.get("description", "")
    title_lnk = f'<a class="fn-link" href="{url}" target="_blank">{title}</a>' if url else title
    meta      = " &nbsp;·&nbsp; ".join(filter(None, [src_name, pub]))
    desc_html = f'<div class="fn-text-muted" style="font-size:0.83rem;margin-top:0.3rem;">{desc}</div>' if desc else ""
    st.markdown(
        f'<div class="fn-card">'
        f'  <div style="font-size:0.95rem;font-weight:600;line-height:1.4;">{title_lnk}{rep_html}</div>'
        f'  <div class="fn-text-faint" style="font-size:0.78rem;margin-top:0.2rem;">{meta}</div>'
        f'  {desc_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Web article scraper ───────────────────────────────────────────────────────

_SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_article_from_url(url: str) -> dict:
    """
    Fetch and extract a news article from a URL.
    Tries trafilatura first (best for news), then BeautifulSoup fallback.
    Returns dict: {title, text, domain, url, word_count, error}.
    """
    domain = urllib.parse.urlparse(url).netloc.replace("www.", "")
    result = {"title": "", "text": "", "domain": domain, "url": url,
              "word_count": 0, "error": ""}

    # ── Method 1: trafilatura ─────────────────────────────────────────────
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            meta = trafilatura.extract_metadata(downloaded)
            text = trafilatura.extract(
                downloaded,
                include_tables=False,
                include_comments=False,
                favor_precision=True,
            )
            if text and len(text) > 100:
                result["title"]      = (meta.title or "").strip() if meta else ""
                result["text"]       = text.strip()
                result["word_count"] = len(text.split())
                return result
    except Exception:
        pass

    # ── Method 2: requests + BeautifulSoup ───────────────────────────────
    try:
        from bs4 import BeautifulSoup

        resp = requests.get(url, headers=_SCRAPE_HEADERS, timeout=20, allow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Remove noise tags
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "iframe", "noscript", "form", "button"]):
            tag.decompose()

        # Title
        title_tag = soup.find("meta", property="og:title") or soup.find("title")
        if title_tag:
            result["title"] = (title_tag.get("content") or title_tag.text or "").strip()

        # Article body — prefer semantic article tag, then common class patterns
        body = (
            soup.find("article")
            or soup.find(attrs={"class": re.compile(r"article[_-]?body|post[_-]?content|story[_-]?body|entry[_-]?content|main[_-]?content", re.I)})
            or soup.find("main")
            or soup.body
        )
        if body:
            text = body.get_text(separator="\n", strip=True)
            # Collapse blank lines
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if len(text) > 100:
                result["text"]       = text[:8000]
                result["word_count"] = len(text.split())
                return result

        result["error"] = "Could not extract article body from page."
    except requests.exceptions.ConnectionError:
        result["error"] = "Could not connect to the URL. Check the address or your network."
    except requests.exceptions.Timeout:
        result["error"] = "The website took too long to respond."
    except requests.exceptions.HTTPError as e:
        result["error"] = f"HTTP {e.response.status_code} — website refused access."
    except Exception as e:
        result["error"] = f"Extraction failed: {str(e)[:120]}"

    return result


def _url_article_card(art: dict):
    """Render the scraped article preview card."""
    domain    = art.get("domain", "")
    title     = art.get("title") or "Untitled Article"
    url       = art.get("url", "")
    words     = art.get("word_count", 0)
    text      = art.get("text", "")
    excerpt   = " ".join(text.split()[:40]) + ("…" if words > 40 else "")
    favicon   = f"https://www.google.com/s2/favicons?domain={domain}&sz=32"

    st.markdown(
        f'<div class="fn-card fn-url-card">'
        f'  <div style="display:flex;align-items:flex-start;gap:0.75rem;">'
        f'    <img src="{favicon}" width="20" height="20" style="margin-top:3px;border-radius:3px;" onerror="this.style.display=\'none\'">'
        f'    <div style="flex:1;min-width:0;">'
        f'      <div style="font-size:0.98rem;font-weight:700;line-height:1.35;">'
        f'        <a class="fn-link" href="{url}" target="_blank">{title}</a>'
        f'      </div>'
        f'      <div class="fn-text-faint" style="font-size:0.78rem;margin-top:0.15rem;">'
        f'        {domain} &nbsp;·&nbsp; {words:,} words extracted'
        f'      </div>'
        f'      <div class="fn-text-muted" style="font-size:0.85rem;margin-top:0.4rem;line-height:1.5;">'
        f'        {excerpt}'
        f'      </div>'
        f'    </div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── File-upload text extraction ──────────────────────────────────────────────

def extract_text_from_image(file_bytes: bytes, filename: str) -> dict:
    """GPT-4o Vision — extract all visible text from an uploaded image."""
    key = _OPENAI_API_KEY
    if not key:
        return {"text": "", "title": "", "description": "",
                "error": "No OpenAI API key configured for image analysis."}

    ext = Path(filename).suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".webp": "image/webp", ".gif": "image/gif"}.get(ext, "image/jpeg")
    b64 = base64.b64encode(file_bytes).decode()

    client = OpenAI(api_key=key)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract every word of visible text from this image exactly as it appears. "
                            "Preserve headlines, body paragraphs, captions, and any other text. "
                            "Also produce a short title and a one-sentence description of what this image shows.\n\n"
                            "Respond ONLY with this JSON (no markdown):\n"
                            '{"title":"<headline or main text>","text":"<full extracted text>'
                            '","description":"<one sentence about the image content>"}'
                        ),
                    },
                ],
            }],
            response_format={"type": "json_object"},
            max_tokens=2500,
            temperature=0.0,
        )
        result = json.loads(resp.choices[0].message.content)
        text = result.get("text", "").strip() or result.get("title", "").strip()
        return {
            "text":        text,
            "title":       result.get("title", ""),
            "description": result.get("description", ""),
            "error":       "" if text else "No readable text found in image.",
        }
    except Exception as exc:
        return {"text": "", "title": "", "description": "", "error": str(exc)[:300]}


def extract_text_from_pdf(file_bytes: bytes) -> dict:
    """Extract text from a PDF — tries PyMuPDF then pdfplumber."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = [pg.get_text() for pg in doc]
        doc.close()
        full = "\n\n".join(p for p in pages if p.strip())
        if full.strip():
            return {"text": full.strip(), "pages": len(pages), "error": ""}
    except ImportError:
        pass
    except Exception as exc:
        return {"text": "", "pages": 0, "error": str(exc)[:300]}

    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for pg in pdf.pages:
                t = pg.extract_text()
                if t:
                    pages.append(t)
        full = "\n\n".join(pages)
        if full.strip():
            return {"text": full.strip(), "pages": len(pages), "error": ""}
    except ImportError:
        pass
    except Exception as exc:
        return {"text": "", "pages": 0, "error": str(exc)[:300]}

    return {"text": "", "pages": 0,
            "error": "No PDF library found. Run: pip install PyMuPDF"}


def extract_text_from_docx(file_bytes: bytes) -> dict:
    """Extract text from a DOCX file using python-docx."""
    try:
        import docx as _docx
        doc = _docx.Document(io.BytesIO(file_bytes))
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        full  = "\n\n".join(paras)
        return {"text": full.strip(), "paragraphs": len(paras), "error": ""}
    except ImportError:
        return {"text": "", "paragraphs": 0,
                "error": "python-docx not installed. Run: pip install python-docx"}
    except Exception as exc:
        return {"text": "", "paragraphs": 0, "error": str(exc)[:300]}


_FILE_ICONS = {".pdf": "📄", ".docx": "📝", ".doc": "📝",
               ".png": "🖼️", ".jpg": "🖼️", ".jpeg": "🖼️",
               ".webp": "🖼️", ".gif": "🖼️"}


def _file_card(meta: dict):
    """Render an uploaded-file info card (mirrors _url_article_card)."""
    name    = meta.get("name", "Uploaded file")
    ftype   = meta.get("type", "")
    size_kb = meta.get("size_kb", 0)
    extract = meta.get("extract_result", {})
    title   = extract.get("title") or name
    desc    = extract.get("description", "")
    text    = extract.get("text", "")
    excerpt = " ".join(text.split()[:45]) + ("…" if len(text.split()) > 45 else "")
    icon    = _FILE_ICONS.get(ftype, "📁")
    size_s  = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.2f} MB"
    extra   = (f" · {extract['pages']} pages" if extract.get("pages")
               else f" · {extract['paragraphs']} paragraphs" if extract.get("paragraphs")
               else "")
    words   = len(text.split())
    st.markdown(
        f'<div class="fn-card fn-file-card">'
        f'  <div style="display:flex;align-items:flex-start;gap:0.75rem;">'
        f'    <div style="font-size:1.9rem;line-height:1;margin-top:2px;">{icon}</div>'
        f'    <div style="flex:1;min-width:0;">'
        f'      <div style="font-size:0.98rem;font-weight:700;line-height:1.35;">{title}</div>'
        f'      <div class="fn-text-faint" style="font-size:0.78rem;margin-top:0.15rem;">'
        f'        {name} &nbsp;·&nbsp; {size_s}{extra} &nbsp;·&nbsp; {words:,} words extracted'
        f'      </div>'
        f'      <div class="fn-text-muted" style="font-size:0.85rem;margin-top:0.4rem;line-height:1.5;">'
        f'        {desc or excerpt}'
        f'      </div>'
        f'    </div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Pages ─────────────────────────────────────────────────────────────────────

def show_checker_page():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        st.warning("⚠️ Model files missing. Run `python train_model.py` first.")
        return

    model, vectorizer = load_local_model()

    # ── Hero header ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="fn-hero">'
        '  <div class="fn-hero-title">🔍 Fake News Detector</div>'
        '  <div class="fn-hero-sub">GPT-4o Web Search &nbsp;·&nbsp; Gemini 2.5 AI &nbsp;·&nbsp; Google Fact Check &nbsp;·&nbsp; NewsAPI &nbsp;·&nbsp; 100 000+ sources</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Input — three tabs: paste text | scan URL | upload file ─────────────
    tab_text, tab_url, tab_file = st.tabs(
        ["📝  Paste Text", "🌐  Scan Website URL", "📁  Upload File"]
    )

    user_text    = ""
    clicked      = False
    article_meta: dict | None = None   # populated when coming from URL tab
    file_meta:    dict | None = None   # populated when coming from file-upload tab

    with tab_text:
        user_text_input = st.text_area(
            "content",
            height=180,
            placeholder="Paste a news headline, article, or social media post here…",
            label_visibility="collapsed",
        )
        col_l, col_m, col_r = st.columns([1, 2, 1])
        with col_m:
            if st.button("Verify Now", type="primary", use_container_width=True, key="btn_text"):
                if user_text_input.strip():
                    user_text = user_text_input.strip()
                    clicked   = True
                else:
                    st.error("Please paste some text first.")

    with tab_url:
        st.markdown(
            '<div class="fn-url-hint">Enter any news article URL — the page will be fetched, '
            'its text extracted, then verified across all AI models and APIs.</div>',
            unsafe_allow_html=True,
        )
        url_input = st.text_input(
            "url",
            placeholder="https://www.bbc.com/news/...",
            label_visibility="collapsed",
        )
        col_l2, col_m2, col_r2 = st.columns([1, 2, 1])
        with col_m2:
            scan_clicked = st.button("Scan & Verify", type="primary",
                                     use_container_width=True, key="btn_url")

        if scan_clicked:
            if not url_input.strip():
                st.error("Please enter a URL first.")
            elif not url_input.strip().startswith(("http://", "https://")):
                st.error("URL must start with http:// or https://")
            else:
                with st.spinner("Fetching article from website…"):
                    article_meta = fetch_article_from_url(url_input.strip())

                if article_meta.get("error"):
                    st.error(f"Could not extract article: {article_meta['error']}")
                elif not article_meta.get("text"):
                    st.error("No readable text found at that URL. Try pasting the article text manually.")
                else:
                    user_text = article_meta["text"]
                    clicked   = True

    with tab_file:
        st.markdown(
            '<div class="fn-url-hint">'
            'Upload a <strong>screenshot / photo</strong>, <strong>PDF</strong>, or '
            '<strong>Word document (.docx)</strong> that contains a news article or claim. '
            'GPT-4o Vision reads text from images; the full 7-source detection pipeline '
            'then analyses the extracted content.'
            '</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "file_uploader",
            type=["png", "jpg", "jpeg", "webp", "gif", "pdf", "docx"],
            label_visibility="collapsed",
            help="Supported: PNG · JPG · WEBP · GIF (images)  |  PDF  |  DOCX (Word)",
        )

        if uploaded_file is not None:
            _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
            fext      = Path(uploaded_file.name).suffix.lower()
            fsize_kb  = len(uploaded_file.getvalue()) / 1024

            if fext in _IMAGE_EXTS:
                st.image(uploaded_file, caption=uploaded_file.name,
                         use_container_width=True)
            else:
                ficon = "📄" if fext == ".pdf" else "📝"
                st.markdown(
                    f'<div class="fn-file-preview-info">'
                    f'  <span style="font-size:1.6rem;">{ficon}</span>'
                    f'  <span style="font-size:0.9rem;color:var(--fn-text-muted);">'
                    f'    {uploaded_file.name} &nbsp;·&nbsp; {fsize_kb:.1f} KB'
                    f'  </span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            col_lf, col_mf, col_rf = st.columns([1, 2, 1])
            with col_mf:
                analyze_btn = st.button(
                    "Analyze File", type="primary",
                    use_container_width=True, key="btn_file",
                )

            if analyze_btn:
                raw_bytes = uploaded_file.getvalue()
                with st.spinner("Extracting content from file…"):
                    if fext in _IMAGE_EXTS:
                        extract_result = extract_text_from_image(raw_bytes, uploaded_file.name)
                    elif fext == ".pdf":
                        extract_result = extract_text_from_pdf(raw_bytes)
                    elif fext in {".docx", ".doc"}:
                        extract_result = extract_text_from_docx(raw_bytes)
                    else:
                        extract_result = {"text": "", "error": "Unsupported file type."}

                if extract_result.get("error"):
                    st.error(f"Could not extract content: {extract_result['error']}")
                elif not extract_result.get("text", "").strip():
                    st.warning("No readable text content found in this file.")
                else:
                    file_meta = {
                        "name":           uploaded_file.name,
                        "type":           fext,
                        "size_kb":        fsize_kb,
                        "extract_result": extract_result,
                    }
                    user_text = extract_result["text"].strip()
                    clicked   = True

    if not clicked:
        st.markdown(
            '<div class="fn-idle">'
            '  <div class="fn-idle-row">'
            '    <div class="fn-idle-item"><div style="font-size:1.6rem;">📋</div><div class="fn-text-faint fn-idle-lbl">Fact-Check DB</div></div>'
            '    <div class="fn-idle-item"><div style="font-size:1.6rem;">🧠</div><div class="fn-text-faint fn-idle-lbl">GPT-5.5</div></div>'
            '    <div class="fn-idle-item"><div style="font-size:1.6rem;">🌐</div><div class="fn-text-faint fn-idle-lbl">Web Search</div></div>'
            '    <div class="fn-idle-item"><div style="font-size:1.6rem;">📰</div><div class="fn-text-faint fn-idle-lbl">NewsAPI</div></div>'
            '    <div class="fn-idle-item"><div style="font-size:1.6rem;">🤖</div><div class="fn-text-faint fn-idle-lbl">AI Models</div></div>'
            '  </div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    _phase_placeholder = st.empty()
    _phase_step_labels = {
        1: "📰 Step 1 / 3 — Searching NewsAPI for news coverage…",
        2: "🌐 Step 2 / 3 — Running Gemini web search (with NewsAPI context)…",
        3: "🔬 Step 3 / 3 — Running Fact-Check API, Gemini FC, GPT-4o Web Search, RoBERTa & SVM…",
    }

    def _phase_cb(phase: int, _label: str):
        _phase_placeholder.info(_phase_step_labels[phase])

    with st.spinner("Analysing…"):
        (
            final_v, final_c,
            fc_v, fc_c, fact_checks,
            gfc_v, gfc_c, gfc_r, gfc_srcs,
            gweb_v, gweb_c, gweb_r, gweb_srcs, gweb_qs,
            ns_v, ns_c, ns_arts, ns_total, ns_rep,
            hf_v, hf_c,
            openai_v, openai_c, openai_r,
            local_v, local_c,
            cleaned, source, breakdown,
        ) = run_ensemble(user_text, model, vectorizer, "", "", progress_cb=_phase_cb)
        suspicious = find_suspicious_phrases(user_text)

    _phase_placeholder.empty()

    all_web_srcs = gfc_srcs + [s for s in gweb_srcs if s not in gfc_srcs]
    best_reason  = gfc_r or gweb_r or openai_r

    # ── Source card (URL mode or file-upload mode) ───────────────────────
    if article_meta and article_meta.get("text"):
        _url_article_card(article_meta)
    elif file_meta:
        _file_card(file_meta)

    # ── Verdict card ──────────────────────────────────────────────────────
    _verdict_card(final_v, final_c, source)

    if final_v == V_INSUFFICIENT:
        st.info("No fact-checker record found and AI confidence is low. Cross-check from a trusted outlet before sharing.")

    if best_reason:
        if all_web_srcs:
            src_tag = "&nbsp;&nbsp;<em>(web-grounded)</em>"
        elif openai_r and best_reason == openai_r:
            src_tag = "&nbsp;&nbsp;<em>(GPT-5.5)</em>"
        else:
            src_tag = ""
        st.markdown(
            f'<div class="fn-reasoning">🧠 <strong>AI Reasoning</strong>{src_tag}<br>{best_reason}</div>',
            unsafe_allow_html=True,
        )

    # ── Evidence stats strip ──────────────────────────────────────────────
    stat_items = []
    if fact_checks:
        stat_items.append(("📋", str(len(fact_checks)), "Fact-checks"))
    if all_web_srcs:
        stat_items.append(("🌐", str(len(all_web_srcs)), "Web sources"))
    if ns_total > 0:
        stat_items.append(("📰", str(ns_total), "News articles"))
    if ns_rep > 0:
        stat_items.append(("🏅", str(ns_rep), "Reputable outlets"))

    if stat_items:
        cols = st.columns(len(stat_items))
        for col, (icon, val, lbl) in zip(cols, stat_items):
            with col:
                st.markdown(
                    f'<div class="fn-stat">'
                    f'  <div style="font-size:1.3rem;">{icon}</div>'
                    f'  <div class="fn-stat-val">{val}</div>'
                    f'  <div class="fn-text-faint fn-stat-lbl">{lbl}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.write("")

    # ── Tabbed results ────────────────────────────────────────────────────
    tab_labels = []
    if fact_checks or all_web_srcs:
        tab_labels.append("📋 Fact-Checks & Web")
    tab_labels.append("📰 News Coverage")
    tab_labels.append("🔬 All Models")
    if suspicious:
        tab_labels.append("⚠️ Suspicious Phrases")

    tabs  = st.tabs(tab_labels)
    t_idx = 0

    # Tab: Fact-Checks & Web Sources
    if fact_checks or all_web_srcs:
        with tabs[t_idx]:
            if fact_checks:
                st.markdown("#### Fact-Check Reports")
                for fc in fact_checks:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        _rating_badge(fc["rating"], fc["category"])
                        st.caption(f"**{fc['publisher']}**" + (f"  ·  {fc['date'][:10]}" if fc.get("date") else ""))
                        if fc.get("claim"):
                            st.caption(f"_{fc['claim'][:200]}_")
                    with col2:
                        if fc.get("url"):
                            st.link_button("Open →", fc["url"])
                    st.divider()

            if all_web_srcs:
                st.markdown("#### Web Sources Consulted")
                if gweb_qs:
                    st.markdown("**Searches:** " + " &nbsp;·&nbsp; ".join(f"`{q}`" for q in gweb_qs))
                for src in all_web_srcs[:12]:
                    try:
                        domain = src["uri"].split("/")[2]
                    except (IndexError, KeyError):
                        domain = ""
                    title = src.get("title") or domain or src.get("uri", "")
                    uri   = src.get("uri", "#")
                    dom_tag = f'&nbsp;&nbsp;<code style="font-size:0.72rem;">{domain}</code>' if domain else ""
                    st.markdown(
                        f'<div class="fn-src-row">'
                        f'  <a class="fn-link" href="{uri}" target="_blank" style="font-size:0.88rem;">{title}</a>{dom_tag}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        t_idx += 1

    # Tab: News Coverage
    with tabs[t_idx]:
        if ns_rep >= 3:
            st.success(f"✅ {ns_rep} reputable outlets covered this story — strong corroboration.")
        elif ns_rep >= 1:
            st.info(f"ℹ️ {ns_rep} reputable outlet{'s' if ns_rep > 1 else ''} found.")
        elif ns_total == 0:
            st.warning("No news articles found for this claim anywhere in NewsAPI.")
        else:
            st.warning(f"{ns_total} articles found but none from reputable outlets.")
        for art in ns_arts:
            _article_card(art)
    t_idx += 1

    # Tab: All Models — consensus breakdown
    with tabs[t_idx]:
        meta      = breakdown.get("__meta__", {})
        avg_sig   = meta.get("avg_signal", 0.0)
        n_active  = meta.get("n_active",  0)
        n_agree   = meta.get("n_agree",   0)
        total_w   = meta.get("total_weight", 0.0)

        # ── Consensus meter ───────────────────────────────────────────────
        if n_active == 0:
            cons_label = "No sources responded"
            cons_cls   = "fn-v-gray"
        elif n_agree == n_active and n_active >= 3:
            cons_label = f"Full Consensus ({n_active}/{n_active} sources agree)"
            cons_cls   = "fn-v-green" if avg_sig > 0 else "fn-v-red"
        elif n_agree >= n_active * 0.7:
            cons_label = f"Strong Agreement ({n_agree}/{n_active} sources)"
            cons_cls   = "fn-v-green" if avg_sig > 0 else "fn-v-red"
        elif n_agree >= n_active * 0.5:
            cons_label = f"Moderate Agreement ({n_agree}/{n_active} sources)"
            cons_cls   = "fn-v-amber"
        else:
            cons_label = f"Conflicted ({n_agree}/{n_active} agree)"
            cons_cls   = "fn-v-amber"

        # Signal bar: -1 (fake) ← 0 → +1 (real)
        bar_fill = int((avg_sig + 1) / 2 * 100)   # 0–100%
        bar_color = "var(--fn-v-green-border)" if avg_sig >= 0 else "var(--fn-v-red-border)"
        st.markdown(
            f'<div class="fn-card" style="margin-bottom:1rem;">'
            f'  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">'
            f'    <span style="font-weight:600;color:var(--fn-text);">Consensus</span>'
            f'    <span class="fn-text-muted" style="font-size:0.85rem;">{cons_label}</span>'
            f'  </div>'
            f'  <div style="position:relative;height:12px;background:var(--fn-border);border-radius:8px;overflow:hidden;">'
            f'    <div style="position:absolute;left:0;top:0;height:100%;width:{bar_fill}%;'
            f'         background:{bar_color};border-radius:8px;transition:width .4s;"></div>'
            f'    <div style="position:absolute;left:50%;top:0;width:2px;height:100%;'
            f'         background:var(--fn-text-faint);"></div>'
            f'  </div>'
            f'  <div style="display:flex;justify-content:space-between;margin-top:0.25rem;">'
            f'    <span class="fn-text-faint" style="font-size:0.72rem;">← Fake</span>'
            f'    <span class="fn-text-faint" style="font-size:0.72rem;">Real →</span>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Per-source breakdown rows ─────────────────────────────────────
        source_rows = [
            ("📋 Fact Check API",    "Fact Check API",   fc_v,     fc_c,     "Not enabled / no match"),
            ("🔎 Gemini FC Search",  "Gemini FC Search", gfc_v,    gfc_c,    "No fact-check found"),
            ("🧠 GPT-4o Web Search",   "GPT-4o Web Search",  openai_v, openai_c, "Unavailable"),
            ("🌐 Gemini Web Search", "Gemini Web Search",gweb_v,   gweb_c,   "No web result"),
            ("📰 NewsAPI Coverage",  "NewsAPI Coverage", ns_v,     ns_c,     "No coverage signal"),
            ("🤖 RoBERTa",           "RoBERTa",          hf_v,     hf_c,     "Unavailable"),
            ("🗂 Local SVM",         "Local SVM",        local_v,  local_c,  "—"),
        ]
        for label, src_key, v, c, na in source_rows:
            bd    = breakdown.get(src_key, {})
            sig   = bd.get("signal")
            pct   = bd.get("pct", 0.0)
            w     = bd.get("weight", 0.0)

            # Row color indicator
            if sig is not None and w > 0:
                if sig > 0.1:
                    ind_color = "var(--fn-v-green-border)"
                    ind_icon  = "✅"
                elif sig < -0.1:
                    ind_color = "var(--fn-v-red-border)"
                    ind_icon  = "🚫"
                else:
                    ind_color = "var(--fn-v-amber-border)"
                    ind_icon  = "⚠️"
            else:
                ind_color = "var(--fn-border)"
                ind_icon  = "○"

            ca, cb, cc, cd = st.columns([2, 2, 1, 2])
            with ca:
                st.markdown(f"**{label}**")
            with cb:
                _mini_verdict(v, c, na)
            with cc:
                st.markdown(
                    f'<div style="font-size:0.75rem;color:var(--fn-text-faint);padding-top:0.4rem;">'
                    f'{pct:.0f}% vote</div>',
                    unsafe_allow_html=True,
                )
            with cd:
                if w > 0 and pct > 0:
                    st.markdown(
                        f'<div style="margin-top:0.45rem;height:8px;background:var(--fn-border);'
                        f'border-radius:6px;overflow:hidden;">'
                        f'<div style="height:100%;width:{min(pct,100):.0f}%;'
                        f'background:{ind_color};border-radius:6px;"></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("—")
    t_idx += 1

    # Tab: Suspicious Phrases
    if suspicious:
        with tabs[t_idx]:
            st.markdown(
                f'<div class="fn-suspicious">Found: <strong>{", ".join(suspicious)}</strong></div>',
                unsafe_allow_html=True,
            )
            st.markdown(highlight_suspicious_phrases(user_text), unsafe_allow_html=True)


def show_model_comparison_page():
    st.markdown(
        '<div style="padding:1.5rem 0 1rem 0;">'
        '  <div class="fn-page-title">📈 Model Performance</div>'
        '  <div class="fn-text-muted" style="margin-top:0.3rem;">Accuracy, precision, recall and F1-score of locally trained classifiers.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not SCORES_PATH.exists():
        st.warning("No results found. Run `python train_model.py` first.")
        return

    scores     = pd.read_csv(SCORES_PATH)
    pct_cols   = [c for c in ["accuracy", "precision", "recall", "f1_score"] if c in scores.columns]
    display_df = scores.copy()
    for col in pct_cols:
        display_df[col] = display_df[col].apply(lambda v: f"{v * 100:.2f}%")

    st.dataframe(display_df, use_container_width=True)

    if {"model", "f1_score"} <= set(scores.columns):
        best = scores.sort_values("f1_score", ascending=False).iloc[0]
        st.success(f"🏆 Best local model: **{best['model']}** ({best['f1_score'] * 100:.2f}% F1)")

    numeric = [c for c in ["accuracy", "precision", "recall", "f1_score"] if c in scores.columns]
    if numeric:
        st.bar_chart(scores.set_index("model")[numeric])

    st.markdown("---")
    st.markdown("#### How the detection pipeline works")
    pipeline_steps = [
        ("📋", "Google Fact Check API",    "Returns verdicts from Snopes, PolitiFact, FactCheck.org and other professional fact-checkers."),
        ("🔎", "Gemini Fact-Check Search", "Gemini 2.5 Flash searches Snopes, PolitiFact, AFP Fact Check and Reuters Fact Check directly."),
        ("🧠", "GPT-4o Web Search",           "GPT-4o searches the live web — BBC, Reuters, Snopes, PolitiFact and more — then analyses language, source credibility and logical coherence."),
        ("🌐", "Gemini Web Search",         "Broad search across BBC, Reuters, AP, CNN, NYT and more for corroborating coverage."),
        ("📰", "NewsAPI Coverage",          "Searches 100 000+ publishers. Counts how many reputable outlets covered the same story."),
        ("🤖", "RoBERTa (HuggingFace)",     "Transformer-based ML classifier — fallback when all web sources are unavailable."),
        ("🗂", "Local SVM",                 "Offline TF-IDF + SVM classifier — works without any internet connection."),
    ]
    for icon, name, desc in pipeline_steps:
        st.markdown(
            f'<div class="fn-pipeline">'
            f'  <div style="font-size:1.4rem;line-height:1;flex-shrink:0;">{icon}</div>'
            f'  <div>'
            f'    <div class="fn-pipeline-name">{name}</div>'
            f'    <div class="fn-text-muted fn-pipeline-desc">{desc}</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="🔍",
        layout="centered",
    )

    # ── Global theme-aware CSS ─────────────────────────────────────────────
    st.markdown("""
    <style>
    /* ── CSS custom properties — light mode defaults ── */
    :root {
        --fn-text:        var(--text-color,        #1e293b);
        --fn-text-muted:  color-mix(in srgb, var(--text-color, #64748b) 70%, transparent);
        --fn-text-faint:  color-mix(in srgb, var(--text-color, #94a3b8) 45%, transparent);
        --fn-card-bg:     var(--secondary-background-color, #ffffff);
        --fn-surface:     var(--secondary-background-color, #f8fafc);
        --fn-border:      rgba(148,163,184,0.25);
        --fn-link:        #3b82f6;
        --fn-reasoning-bg:     rgba(148,163,184,0.12);
        --fn-reasoning-border: #6366f1;

        /* Verdict — green */
        --fn-v-green-bg:    linear-gradient(135deg,#dcfce7,#bbf7d0);
        --fn-v-green-border:#16a34a;
        --fn-v-green-label: #14532d;
        --fn-v-green-pct:   #16a34a;
        /* Verdict — red */
        --fn-v-red-bg:      linear-gradient(135deg,#fee2e2,#fecaca);
        --fn-v-red-border:  #dc2626;
        --fn-v-red-label:   #7f1d1d;
        --fn-v-red-pct:     #dc2626;
        /* Verdict — amber */
        --fn-v-amber-bg:    linear-gradient(135deg,#fffbeb,#fef9c3);
        --fn-v-amber-border:#d97706;
        --fn-v-amber-label: #78350f;
        --fn-v-amber-pct:   #d97706;
        /* Verdict — gray */
        --fn-v-gray-bg:     linear-gradient(135deg,var(--secondary-background-color,#f8fafc),rgba(148,163,184,0.15));
        --fn-v-gray-border: #94a3b8;
        --fn-v-gray-label:  var(--text-color, #475569);
        --fn-v-gray-pct:    #64748b;

        /* Reputable badge */
        --fn-rep-bg:    #dbeafe;
        --fn-rep-color: #1d4ed8;

        /* Suspicious */
        --fn-sus-bg:     #fffbeb;
        --fn-sus-border: #fde68a;
        --fn-sus-color:  #92400e;
    }

    /* ── Dark mode override ── */
    @media (prefers-color-scheme: dark) {
        :root {
            --fn-link:        #93c5fd;
            --fn-reasoning-bg:     rgba(99,102,241,0.12);
            --fn-reasoning-border: #818cf8;

            --fn-v-green-bg:    linear-gradient(135deg,#052e16,#14532d);
            --fn-v-green-border:#22c55e;
            --fn-v-green-label: #bbf7d0;
            --fn-v-green-pct:   #22c55e;

            --fn-v-red-bg:      linear-gradient(135deg,#450a0a,#7f1d1d);
            --fn-v-red-border:  #f87171;
            --fn-v-red-label:   #fecaca;
            --fn-v-red-pct:     #f87171;

            --fn-v-amber-bg:    linear-gradient(135deg,#431407,#78350f);
            --fn-v-amber-border:#fbbf24;
            --fn-v-amber-label: #fef9c3;
            --fn-v-amber-pct:   #fbbf24;

            --fn-v-gray-bg:     linear-gradient(135deg,rgba(15,23,42,0.6),rgba(30,41,59,0.9));
            --fn-v-gray-border: #64748b;
            --fn-v-gray-label:  #cbd5e1;
            --fn-v-gray-pct:    #94a3b8;

            --fn-rep-bg:    #1e3a8a;
            --fn-rep-color: #93c5fd;

            --fn-sus-bg:     #431407;
            --fn-sus-border: #92400e;
            --fn-sus-color:  #fef9c3;
        }
    }

    /* ── Layout ── */
    .block-container { padding-top:1.5rem !important; padding-bottom:2rem !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] .stRadio label { font-size:0.95rem; font-weight:500; }

    /* ── Button ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg,#6366f1,#4f46e5);
        color:#fff; border:none; border-radius:12px;
        font-size:1.05rem; font-weight:700; padding:0.65rem 2rem;
        box-shadow:0 4px 14px rgba(99,102,241,0.35);
        transition:opacity .15s;
    }
    .stButton > button[kind="primary"]:hover { opacity:.88; }

    /* ── Text area ── */
    .stTextArea textarea {
        border-radius:12px; border:1.5px solid var(--fn-border);
        font-size:0.95rem; background:var(--fn-card-bg);
        color:var(--fn-text);
    }
    .stTextArea textarea:focus {
        border-color:#6366f1;
        box-shadow:0 0 0 3px rgba(99,102,241,0.15);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap:4px; }
    .stTabs [data-baseweb="tab"] { border-radius:8px 8px 0 0; font-weight:500; }

    /* ── Highlight ── */
    mark { background:#fef9c3; color:#713f12; padding:2px 4px; border-radius:4px; }

    /* ── Shared text utilities ── */
    .fn-text       { color: var(--fn-text); }
    .fn-text-muted { color: var(--fn-text-muted); }
    .fn-text-faint { color: var(--fn-text-faint); }
    .fn-link       { color: var(--fn-link); text-decoration:none; }
    .fn-link:hover { text-decoration:underline; }

    /* ── Hero ── */
    .fn-hero       { text-align:center; padding:2rem 0 1.5rem; }
    .fn-hero-title { font-size:2.6rem; font-weight:900; color:var(--fn-text); line-height:1.1; }
    .fn-hero-sub   { color:var(--fn-text-muted); font-size:1.05rem; margin-top:0.6rem; }

    /* ── Page title (model comparison) ── */
    .fn-page-title { font-size:2rem; font-weight:800; color:var(--fn-text); }

    /* ── Verdict card base ── */
    .fn-verdict {
        border-left:5px solid; border-radius:14px;
        padding:1.5rem 2rem; margin:0.75rem 0;
        box-shadow:0 3px 12px rgba(0,0,0,0.07);
    }
    .fn-v-row  { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem; }
    .fn-v-label { font-size:1.9rem; font-weight:800; line-height:1.15; }
    .fn-v-sub   { font-size:0.88rem; margin-top:0.25rem; color:var(--fn-text-muted); }
    .fn-v-src   { font-size:0.8rem; margin-top:0.15rem; color:var(--fn-text-faint); }
    .fn-v-right { text-align:right; }
    .fn-v-pct   { font-size:2.6rem; font-weight:800; }
    .fn-v-pct-lbl { font-size:0.82rem; color:var(--fn-text-faint); }
    .fn-v-track { margin-top:1rem; background:rgba(128,128,128,0.15); border-radius:8px; height:9px; overflow:hidden; }
    .fn-v-fill  { height:100%; border-radius:8px; }

    /* Verdict colour modifiers */
    .fn-v-green { background:var(--fn-v-green-bg); border-color:var(--fn-v-green-border); }
    .fn-v-green .fn-v-label { color:var(--fn-v-green-label); }
    .fn-v-green .fn-v-pct   { color:var(--fn-v-green-pct);   }
    .fn-v-green .fn-v-fill  { background:var(--fn-v-green-border); }

    .fn-v-red   { background:var(--fn-v-red-bg); border-color:var(--fn-v-red-border); }
    .fn-v-red   .fn-v-label { color:var(--fn-v-red-label); }
    .fn-v-red   .fn-v-pct   { color:var(--fn-v-red-pct);   }
    .fn-v-red   .fn-v-fill  { background:var(--fn-v-red-border); }

    .fn-v-amber { background:var(--fn-v-amber-bg); border-color:var(--fn-v-amber-border); }
    .fn-v-amber .fn-v-label { color:var(--fn-v-amber-label); }
    .fn-v-amber .fn-v-pct   { color:var(--fn-v-amber-pct);   }
    .fn-v-amber .fn-v-fill  { background:var(--fn-v-amber-border); }

    .fn-v-gray  { background:var(--fn-v-gray-bg); border-color:var(--fn-v-gray-border); }
    .fn-v-gray  .fn-v-label { color:var(--fn-v-gray-label); }
    .fn-v-gray  .fn-v-pct   { color:var(--fn-v-gray-pct);   }
    .fn-v-gray  .fn-v-fill  { background:var(--fn-v-gray-border); }

    /* ── AI Reasoning box ── */
    .fn-reasoning {
        background:var(--fn-reasoning-bg);
        border-left:4px solid var(--fn-reasoning-border);
        border-radius:0 8px 8px 0;
        padding:0.75rem 1rem;
        color:var(--fn-text);
        font-size:0.92rem;
        margin:0.5rem 0 1rem;
        line-height:1.55;
    }

    /* ── Stat strip ── */
    .fn-stat     { text-align:center; background:var(--fn-surface); border-radius:10px; padding:0.6rem 0.4rem; border:1px solid var(--fn-border); }
    .fn-stat-val { font-size:1.4rem; font-weight:700; color:var(--fn-text); }
    .fn-stat-lbl { font-size:0.72rem; }

    /* ── Article card ── */
    .fn-card { background:var(--fn-card-bg); border-radius:10px; padding:0.85rem 1.1rem; margin-bottom:0.6rem; box-shadow:0 1px 5px rgba(0,0,0,0.06); border:1px solid var(--fn-border); }

    /* ── Reputable badge ── */
    .fn-rep-badge { background:var(--fn-rep-bg); color:var(--fn-rep-color); padding:2px 8px; border-radius:20px; font-size:0.72rem; font-weight:600; margin-left:6px; display:inline; }

    /* ── Source row (web sources list) ── */
    .fn-src-row { padding:0.35rem 0; border-bottom:1px solid var(--fn-border); }

    /* ── Suspicious phrases ── */
    .fn-suspicious { background:var(--fn-sus-bg); border:1px solid var(--fn-sus-border); border-radius:10px; padding:0.8rem 1rem; margin-bottom:0.8rem; color:var(--fn-sus-color); font-size:0.9rem; }

    /* ── Pipeline steps (model comparison page) ── */
    .fn-pipeline      { display:flex; gap:0.8rem; align-items:flex-start; background:var(--fn-surface); border-radius:10px; padding:0.75rem 1rem; margin-bottom:0.5rem; border:1px solid var(--fn-border); }
    .fn-pipeline-name { color:var(--fn-text); font-weight:600; }
    .fn-pipeline-desc { font-size:0.85rem; margin-top:0.15rem; }

    /* ── Idle state ── */
    .fn-idle     { text-align:center; margin-top:2.5rem; }
    .fn-idle-row { display:inline-flex; gap:2rem; flex-wrap:wrap; justify-content:center; }
    .fn-idle-item { text-align:center; }
    .fn-idle-lbl { font-size:0.78rem; margin-top:0.2rem; }

    /* ── URL input hint ── */
    .fn-url-hint {
        background: var(--fn-surface);
        border: 1px solid var(--fn-border);
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-size: 0.88rem;
        color: var(--fn-text-muted);
        margin-bottom: 0.6rem;
    }

    /* ── Scraped article card ── */
    .fn-url-card {
        border-left: 4px solid #6366f1 !important;
        margin-bottom: 0.75rem;
    }

    /* ── Uploaded file card ── */
    .fn-file-card {
        border-left: 4px solid #8b5cf6 !important;
        margin-bottom: 0.75rem;
    }
    .fn-file-preview-info {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        background: var(--fn-surface);
        border: 1px solid var(--fn-border);
        border-radius: 10px;
        padding: 0.65rem 1rem;
        margin-bottom: 0.75rem;
    }

    /* ── Text input ── */
    .stTextInput input {
        border-radius: 10px;
        border: 1.5px solid var(--fn-border);
        font-size: 0.95rem;
        background: var(--fn-card-bg);
        color: var(--fn-text);
        padding: 0.55rem 0.85rem;
    }
    .stTextInput input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.15);
    }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="fn-text" style="font-size:1.15rem;font-weight:700;padding:0.5rem 0 1rem;">🔍 Fake News Detector</div>', unsafe_allow_html=True)
        page = st.radio("nav", ["Check News", "Model Performance"], label_visibility="collapsed")

    if page == "Check News":
        show_checker_page()
    else:
        show_model_comparison_page()


if __name__ == "__main__":
    main()
