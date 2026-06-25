# Fake News Detection System

A multi-source, AI-powered fake news detector built with Streamlit. It combines a locally-trained machine learning model with live AI APIs to deliver high-confidence verdicts on any news article, headline, or social media claim.

---

## Features

- **7-source ensemble** — Google Fact Check Tools, Gemini 2.5 Flash (fact-check search), Gemini 2.5 Flash (web search), GPT-4o (live web search), NewsAPI coverage analysis, HuggingFace RoBERTa, and a local TF-IDF + SVM model
- **Three input modes** — paste text, scan a URL, or upload a PDF / DOCX / image file
- **GPT-4o Vision** — extracts text from screenshots and photos before running the full detection pipeline
- **Suspicious phrase detection** — highlights emotional manipulation language in the source text
- **Weighted confidence voting** — each source is weighted by trustworthiness; confidence scales with inter-source agreement
- **Fully offline fallback** — the local SVM (99.45% F1) runs without any API key

---

## Detection Pipeline

Analysis runs in three sequential phases:

| Phase | Sources | Purpose |
|-------|---------|---------|
| 1 | NewsAPI | Measure news coverage breadth across 100,000+ publishers |
| 2 | Gemini 2.5 Web Search | Live web analysis, enriched by Phase 1 context |
| 3 (parallel) | Fact Check API · Gemini FC · GPT-4o · RoBERTa · Local SVM | Deep fact-checking and classification |

The final verdict is determined by a **weighted consensus vote**. Sources are weighted by authority:

| Source | Weight |
|--------|--------|
| Google Fact Check Tools API | 4.0 |
| Gemini FC Search (Snopes / PolitiFact / AFP) | 3.0 |
| GPT-4o Web Search | 2.8 |
| Gemini Web Search | 2.5 |
| NewsAPI Coverage | 2.0 |
| Local SVM | 1.8 |
| HuggingFace RoBERTa | 1.5 |

---

## Local Model Performance

Trained on the [ISOT Fake News Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets) (~44,000 articles).

| Model | Accuracy | F1 Score | Overfit Gap |
|-------|----------|----------|-------------|
| **Linear SVM** | 99.45% | **99.45%** | 0.0055 |
| Passive Aggressive | 99.42% | 99.42% | 0.0058 |
| Logistic Regression | 98.75% | 98.75% | 0.0034 |
| Random Forest | 98.16% | 98.16% | 0.0166 |
| Naive Bayes | 96.27% | 96.27% | 0.0094 |

The best model (Linear SVM) is saved automatically and loaded by the web app.

---

## Project Structure

```
Fake-News-Detection-System/
├── app.py                  # Streamlit web application (main entry point)
├── train_model.py          # ML training pipeline (11 steps)
├── preprocess.py           # Text cleaning & NLP preprocessing
├── suspicious_words.py     # Clickbait / manipulation phrase detector
├── requirements.txt        # Python dependencies
├── dataset/
│   ├── Fake.csv            # ISOT fake news articles
│   └── True.csv            # ISOT real news articles
├── models/
│   ├── best_model.pkl      # Trained classifier (auto-generated)
│   └── tfidf_vectorizer.pkl
└── results/
    ├── model_scores.csv    # Evaluation scores for all models
    └── *_confusion_matrix.png
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Fake-News-Detection-System.git
cd Fake-News-Detection-System
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Add the dataset

Place `Fake.csv` and `True.csv` inside the `dataset/` directory. The ISOT dataset is available from [Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets).

### 4. Train the local model

```bash
python train_model.py
```

This runs the full 11-step training pipeline (dataset analysis → cleaning → preprocessing → TF-IDF comparison → training → cross-validation → hyperparameter tuning → model export) and saves the best model to `models/`.

### 5. Launch the web app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## API Keys

The app works out of the box using pre-configured keys for demonstration. To use your own:

| API | Where to get it |
|-----|----------------|
| Google Gemini + Fact Check Tools | [Google AI Studio](https://aistudio.google.com/) |
| NewsAPI | [newsapi.org](https://newsapi.org/) |
| OpenAI (GPT-4o) | [platform.openai.com](https://platform.openai.com/) |
| HuggingFace (optional) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

Update the key constants at the top of `app.py` (`_DEFAULT_API_KEY`, `_NEWS_API_KEY`, `_OPENAI_API_KEY`).

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | Streamlit |
| ML / NLP | scikit-learn, NLTK, TF-IDF |
| AI APIs | Google Gemini 2.5 Flash, OpenAI GPT-4o, HuggingFace RoBERTa |
| News APIs | Google Fact Check Tools, NewsAPI |
| Web scraping | trafilatura, BeautifulSoup4, lxml |
| File parsing | PyMuPDF (PDF), python-docx (DOCX), GPT-4o Vision (images) |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn |

---

## Verdict Labels

| Label | Meaning |
|-------|---------|
| Verified True | Confirmed by professional fact-checkers |
| Real News | Supported by AI analysis and web evidence |
| Verified False | Debunked by professional fact-checkers |
| Fake News | Identified as false by AI and web analysis |
| Misleading or Partly False | Contains misleading or inaccurate elements |
| Not Enough Evidence | Insufficient evidence — verify from trusted sources |

---

## License

This project is released under the [MIT License](LICENSE).
