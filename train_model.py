"""
train_model.py - Improved Fake News Detection Trainer
======================================================
Steps covered:
  1.  Dataset analysis
  2.  Data cleaning
  3.  Text preprocessing (dateline removal, lemmatisation)
  4.  TF-IDF configuration comparison
  5.  Multi-model training (LR, NB, LinearSVM, PassiveAggressive, RandomForest)
  6.  Full evaluation (accuracy, precision, recall, F1, confusion matrix,
      classification report)
  7.  Cross-validation (5-fold, top 3 models)
  8.  Overfitting check (train vs test accuracy gap)
  9.  Hyperparameter tuning (RandomizedSearchCV on best model)
  10. Save best model + vectorizer
  11. Prediction test on sample headlines
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from preprocess import clean_text


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FAKE_PATH    = Path("dataset/Fake.csv")
TRUE_PATH    = Path("dataset/True.csv")
DATASET_PATH = Path("dataset/fake_news.csv")
MODELS_DIR   = Path("models")
RESULTS_DIR  = Path("results")

# Set to False for a faster run without hyperparameter tuning
TUNE_HYPERPARAMS = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(title: str):
    line = "=" * 70
    print(f"\n{line}\n  {title}\n{line}")


def normalize_label(raw) -> str:
    s = str(raw).lower().strip()
    return "real" if s in ("real", "true", "1") else "fake" if s in ("fake", "false", "0") else s


def save_confusion_matrix(model_name: str, y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig(RESULTS_DIR / fname)
    plt.close()


# ---------------------------------------------------------------------------
# STEP 1 — Dataset Analysis
# ---------------------------------------------------------------------------

def load_raw() -> pd.DataFrame:
    if DATASET_PATH.exists() and DATASET_PATH.stat().st_size > 20:
        df = pd.read_csv(DATASET_PATH)
        df.columns = df.columns.str.lower().str.strip()
    elif FAKE_PATH.exists() and TRUE_PATH.exists():
        fake = pd.read_csv(FAKE_PATH)
        true = pd.read_csv(TRUE_PATH)
        fake["label"] = "fake"
        true["label"] = "real"
        df = pd.concat([fake, true], ignore_index=True)
        df.columns = df.columns.str.lower().str.strip()
    else:
        raise FileNotFoundError(
            "Dataset not found. Add dataset/Fake.csv + dataset/True.csv "
            "or dataset/fake_news.csv"
        )
    return df


def analyse_dataset(df: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 1 - DATASET ANALYSIS")

    print(f"\n  Rows        : {df.shape[0]:,}")
    print(f"  Columns     : {list(df.columns)}")

    print(f"\n  Missing values:")
    for col, n in df.isnull().sum().items():
        flag = "  [!]" if n > 0 else "  [OK]"
        print(f"    {flag}  {col}: {n}")

    dups = df.duplicated().sum()
    print(f"\n  Duplicate rows : {dups:,}  {'[!] will be removed' if dups else '[OK] none'}")

    if "label" in df.columns:
        df["label"] = df["label"].apply(normalize_label)
        print("\n  Label distribution:")
        vc = df["label"].value_counts()
        for lbl, cnt in vc.items():
            bar = "#" * int(cnt / vc.max() * 30)
            print(f"    {lbl:6s}: {cnt:6,}  ({cnt/len(df)*100:.1f}%)  {bar}")
        ratio = vc.max() / vc.min()
        if ratio > 1.5:
            print(f"\n  [!] Class imbalance - ratio {ratio:.2f}. "
                  "class_weight='balanced' will be applied.")
        else:
            print(f"\n  [OK] Classes are nearly balanced (ratio {ratio:.2f}).")

    if "text" in df.columns:
        wc = df["text"].fillna("").str.split().str.len()
        print("\n  Text length (words):")
        for stat, val in wc.describe()[["mean","std","min","50%","max"]].items():
            print(f"    {stat:5s}: {val:.0f}")

    if "subject" in df.columns and "label" in df.columns:
        s0 = set(df[df["label"] == "fake"]["subject"].dropna().unique())
        s1 = set(df[df["label"] == "real"]["subject"].dropna().unique())
        overlap = s0 & s1
        if not overlap:
            print("\n  [!] 'subject' has ZERO overlap between fake/real --"
                  " perfect leakage feature, will be DROPPED.")
        else:
            print(f"\n  [i] 'subject' overlapping values: {overlap}")

    return df


# ---------------------------------------------------------------------------
# STEP 2 — Data Cleaning
# ---------------------------------------------------------------------------

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 2 - DATA CLEANING")
    n0 = len(df)

    # Drop leaky / irrelevant columns
    leaky = [c for c in ("subject", "date") if c in df.columns]
    if leaky:
        df = df.drop(columns=leaky)
        print(f"  Dropped leaky/irrelevant columns: {leaky}")

    # Combine title + text into 'content'
    if "title" in df.columns and "text" in df.columns:
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
        df = df.drop(columns=["title", "text"])
        print("  Combined title + text  ->  content")
    elif "text" in df.columns:
        df = df.rename(columns={"text": "content"})

    # Drop empty content / missing labels
    df = df.dropna(subset=["content", "label"])
    df = df[df["content"].str.strip() != ""]

    # Remove duplicate articles
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["content"])
    print(f"  Removed {before_dedup - len(df):,} duplicate articles")

    # Normalize labels, drop unknowns
    df["label"] = df["label"].apply(normalize_label)
    valid = {"real", "fake"}
    bad = ~df["label"].isin(valid)
    if bad.any():
        print(f"  Dropped {bad.sum()} rows with unknown labels: "
              f"{df.loc[bad, 'label'].unique()}")
        df = df[~bad]

    print(f"\n  Rows: {n0:,}  ->  {len(df):,}  (removed {n0 - len(df):,})")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# STEP 3 — Text Preprocessing
# ---------------------------------------------------------------------------

def preprocess_texts(df: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 3 - TEXT PREPROCESSING")
    print("  Applying: dateline stripping -> lowercase -> URL/punct removal "
          "-> stop-word removal -> lemmatisation")
    print("  (This may take 2-5 minutes for 44k articles...)")

    df["cleaned"] = df["content"].apply(clean_text)

    empty = (df["cleaned"].str.strip() == "").sum()
    if empty:
        print(f"  [!] {empty} articles became empty after cleaning -- dropped.")
        df = df[df["cleaned"].str.strip() != ""]

    avg_words = df["cleaned"].str.split().str.len().mean()
    print(f"  Done.  {len(df):,} articles  |  avg {avg_words:.0f} words after cleaning")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# STEP 4 — TF-IDF Configuration Comparison
# ---------------------------------------------------------------------------

TFIDF_CONFIGS = {
    "TF-IDF-A  50k unigrams      ": TfidfVectorizer(
        max_features=50_000, ngram_range=(1, 1),
        min_df=2, max_df=0.85, sublinear_tf=True, strip_accents="unicode",
    ),
    "TF-IDF-B  50k bigrams       ": TfidfVectorizer(
        max_features=50_000, ngram_range=(1, 2),
        min_df=2, max_df=0.85, sublinear_tf=True, strip_accents="unicode",
    ),
    "TF-IDF-C  100k bigrams      ": TfidfVectorizer(
        max_features=100_000, ngram_range=(1, 2),
        min_df=2, max_df=0.95, sublinear_tf=True, strip_accents="unicode",
    ),
}


def compare_vectorizers(x_train, x_test, y_train, y_test) -> TfidfVectorizer:
    banner("STEP 4 - TF-IDF CONFIGURATION COMPARISON")

    probe = LogisticRegression(
        max_iter=300, C=1.0, solver="lbfgs",
        class_weight="balanced", random_state=42,
    )
    best_name, best_vec, best_f1 = "", None, -1.0

    print(f"\n  {'Config':<36} {'Vocab':>8}  {'F1':>7}")
    print("  " + "-" * 54)

    for name, vec in TFIDF_CONFIGS.items():
        xtr = vec.fit_transform(x_train)
        xte = vec.transform(x_test)
        probe.fit(xtr, y_train)
        f1 = f1_score(y_test, probe.predict(xte), average="weighted")
        vocab = len(vec.vocabulary_)
        marker = "  <-- best" if f1 > best_f1 else ""
        print(f"  {name}  {vocab:>8,}  {f1:.4f}{marker}")
        if f1 > best_f1:
            best_f1, best_name, best_vec = f1, name.strip(), vec

    print(f"\n  [OK] Selected: {best_name} (F1 = {best_f1:.4f})")
    return best_vec


# ---------------------------------------------------------------------------
# STEP 5 — Model Definitions
# ---------------------------------------------------------------------------

def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=5.0, solver="lbfgs",
            class_weight="balanced", random_state=42,
        ),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(
                max_iter=3000, C=1.0,
                class_weight="balanced", random_state=42,
            ),
            cv=3,
        ),
        "Passive Aggressive": CalibratedClassifierCV(
            SGDClassifier(
                loss="hinge", penalty=None,
                learning_rate="pa1", eta0=1.0,
                max_iter=1000, class_weight="balanced", random_state=42,
            ),
            cv=3,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=30,
            class_weight="balanced", n_jobs=-1, random_state=42,
        ),
    }


# ---------------------------------------------------------------------------
# STEP 6 — Train, Evaluate, Overfitting Check
# ---------------------------------------------------------------------------

def train_and_evaluate(models, x_train_v, x_test_v, y_train, y_test, labels) -> list:
    banner("STEP 5 & 6 - TRAINING + FULL EVALUATION")
    scores = []

    for name, model in models.items():
        print(f"\n  -- {name} {'-'*(50 - len(name))}")
        model.fit(x_train_v, y_train)

        train_preds = model.predict(x_train_v)
        test_preds  = model.predict(x_test_v)

        train_acc = accuracy_score(y_train, train_preds)
        test_acc  = accuracy_score(y_test,  test_preds)
        precision = precision_score(y_test, test_preds, average="weighted", zero_division=0)
        recall    = recall_score(   y_test, test_preds, average="weighted", zero_division=0)
        f1        = f1_score(       y_test, test_preds, average="weighted", zero_division=0)
        gap       = train_acc - test_acc

        overfit_msg = "[!] OVERFIT" if gap > 0.05 else "[OK]"
        print(f"  Train acc : {train_acc*100:.2f}%  |  Test acc : {test_acc*100:.2f}%  "
              f"|  Gap : {gap:+.4f}  {overfit_msg}")
        print(f"  Precision : {precision*100:.2f}%  |  Recall   : {recall*100:.2f}%  "
              f"|  F1      : {f1*100:.2f}%")

        print(f"\n  Classification Report:")
        print(classification_report(y_test, test_preds, target_names=labels, digits=4,
                                    zero_division=0))

        save_confusion_matrix(name, y_test, test_preds, labels)

        scores.append({
            "model":          name,
            "train_accuracy": round(train_acc, 6),
            "accuracy":       round(test_acc,  6),
            "precision":      round(precision, 6),
            "recall":         round(recall,    6),
            "f1_score":       round(f1,        6),
            "overfit_gap":    round(gap,        6),
        })

    return scores


# ---------------------------------------------------------------------------
# STEP 7 — Cross-Validation (top 3 models)
# ---------------------------------------------------------------------------

def cross_validate_top(models: dict, scores: list, x_v, y):
    banner("STEP 7 - CROSS-VALIDATION (5-fold, top 3 models)")

    df = pd.DataFrame(scores).sort_values("f1_score", ascending=False)
    top3 = df.head(3)["model"].tolist()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n  {'Model':<25}  {'Mean F1':>8}  {'Std':>7}  Fold scores")
    print("  " + "-" * 72)

    for name in top3:
        cv_scores = cross_val_score(
            models[name], x_v, y, cv=cv, scoring="f1_weighted", n_jobs=-1
        )
        fold_str = "  ".join(f"{s:.4f}" for s in cv_scores)
        print(f"  {name:<25}  {cv_scores.mean():.4f}    ±{cv_scores.std():.4f}   {fold_str}")


# ---------------------------------------------------------------------------
# STEP 8 — Overfitting Summary
# ---------------------------------------------------------------------------

def overfitting_summary(scores: list):
    banner("STEP 8 - OVERFITTING SUMMARY")
    df = pd.DataFrame(scores)[["model","train_accuracy","accuracy","overfit_gap","f1_score"]]
    df = df.sort_values("f1_score", ascending=False)
    df["train_accuracy"] = df["train_accuracy"].apply(lambda v: f"{v*100:.2f}%")
    df["accuracy"]       = df["accuracy"].apply(lambda v: f"{v*100:.2f}%")
    df["f1_score"]       = df["f1_score"].apply(lambda v: f"{v*100:.2f}%")
    df["overfit_gap"]    = df["overfit_gap"].apply(lambda v: f"{v:+.4f}")
    print("\n" + df.to_string(index=False))
    print("""
  Interpretation:
    overfit_gap > 0.05  -> model memorises training data (overfitting)
    overfit_gap ~ 0.00  -> good generalisation
    overfit_gap < 0.00  -> possible underfitting or stochastic variation
  """)


# ---------------------------------------------------------------------------
# STEP 9 — Hyperparameter Tuning
# ---------------------------------------------------------------------------

TUNING_GRIDS = {
    "Logistic Regression": {
        "C":      [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
        "solver": ["lbfgs", "saga"],
    },
    "Linear SVM": {
        "estimator__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    },
    "Passive Aggressive": {
        "estimator__eta0": [0.1, 0.5, 1.0, 2.0, 5.0],
    },
}


def tune_model(name: str, model, x_train_v, y_train, x_test_v, y_test):
    banner(f"STEP 9 - HYPERPARAMETER TUNING ({name})")

    grid = TUNING_GRIDS.get(name)
    if grid is None:
        print(f"  No tuning grid defined for '{name}' -- using default parameters.")
        return model

    print(f"  Running RandomizedSearchCV (n_iter=10, cv=3)...")
    search = RandomizedSearchCV(
        model, grid,
        n_iter=10, cv=3, scoring="f1_weighted",
        n_jobs=-1, random_state=42, verbose=0,
    )
    search.fit(x_train_v, y_train)

    print(f"  Best params   : {search.best_params_}")
    print(f"  Best CV F1    : {search.best_score_*100:.2f}%")

    # Compare with default
    default_f1 = f1_score(y_test, model.predict(x_test_v), average="weighted")
    tuned_f1   = f1_score(y_test, search.best_estimator_.predict(x_test_v), average="weighted")
    tuned_acc  = accuracy_score(y_test, search.best_estimator_.predict(x_test_v))

    print(f"  Default test F1 : {default_f1*100:.2f}%")
    print(f"  Tuned   test F1 : {tuned_f1*100:.2f}%   Acc: {tuned_acc*100:.2f}%")
    improvement = (tuned_f1 - default_f1) * 100
    print(f"  Improvement     : {improvement:+.2f}%")

    return search.best_estimator_


# ---------------------------------------------------------------------------
# STEP 10 — Prediction Test
# ---------------------------------------------------------------------------

TEST_HEADLINES = [
    "Scientists confirm new COVID-19 vaccine is 95% effective in large clinical trial",
    "SHOCKING: Government secretly adding chemicals in water to control your mind, leaked proof!",
    "Federal Reserve raises interest rates by 0.25 percentage points amid inflation concerns",
    "BREAKING: Top celebrity admits to being part of a secret global lizard-people conspiracy",
    "White House says President Biden will attend G7 summit next month",
    "You WON'T BELIEVE what they found - this destroys everything mainstream media told you!!",
]


def prediction_test(model, vectorizer):
    banner("STEP 11 - PREDICTION TEST (sample headlines)")
    print(f"\n  {'Verdict':<22} {'Conf':>6}  Headline\n  " + "-" * 80)

    for text in TEST_HEADLINES:
        cleaned = clean_text(text)
        feats   = vectorizer.transform([cleaned])

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feats)[0]
            idx   = int(np.argmax(probs))
            conf  = float(probs[idx])
            label = model.classes_[idx]
        else:
            label = model.predict(feats)[0]
            conf  = 1.0

        tag = "[REAL] Real News" if label == "real" else "[FAKE] Fake News"
        print(f"  {tag:<22} {conf*100:>5.1f}%  {text[:65]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # -- 1. Load & analyse
    df = load_raw()
    df = analyse_dataset(df)

    # ── 2. Clean ──────────────────────────────────────────────────────────
    df = clean_dataset(df)

    # ── 3. Preprocess text ────────────────────────────────────────────────
    df = preprocess_texts(df)

    X = df["cleaned"]
    y = df["label"]
    labels = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}  "
          f"|  Classes: {labels}")

    # ── 4. Pick best TF-IDF ───────────────────────────────────────────────
    best_vec = compare_vectorizers(X_train, X_test, y_train, y_test)
    # Re-fit on clean train split for actual training run
    X_train_v = best_vec.fit_transform(X_train)
    X_test_v  = best_vec.transform(X_test)

    # ── 5-6. Train all models + evaluate ─────────────────────────────────
    models = build_models()
    scores = train_and_evaluate(models, X_train_v, X_test_v, y_train, y_test, labels)

    # ── 7. Cross-validation ───────────────────────────────────────────────
    cross_validate_top(models, scores, X_train_v, y_train)

    # ── 8. Overfitting summary ────────────────────────────────────────────
    overfitting_summary(scores)

    # ── 9. Hyperparameter tuning on best model ────────────────────────────
    scores_df   = pd.DataFrame(scores).sort_values("f1_score", ascending=False)
    best_name   = scores_df.iloc[0]["model"]
    final_model = models[best_name]

    if TUNE_HYPERPARAMS and best_name in TUNING_GRIDS:
        final_model = tune_model(
            best_name, final_model,
            X_train_v, y_train, X_test_v, y_test,
        )
        # Update scores row with tuned result
        tuned_preds = final_model.predict(X_test_v)
        scores_df.loc[scores_df["model"] == best_name, "f1_score"] = round(
            f1_score(y_test, tuned_preds, average="weighted"), 6
        )
        scores_df.loc[scores_df["model"] == best_name, "accuracy"] = round(
            accuracy_score(y_test, tuned_preds), 6
        )
    else:
        if TUNE_HYPERPARAMS:
            print(f"\n  [i] '{best_name}' has no tuning grid -- skipping tuning.")

    # ── Save model + scores ───────────────────────────────────────────────
    banner("STEP 10 - SAVING BEST MODEL")
    scores_df.to_csv(RESULTS_DIR / "model_scores.csv", index=False)
    joblib.dump(final_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(best_vec,    MODELS_DIR / "tfidf_vectorizer.pkl")

    best_row = scores_df.iloc[0]
    print(f"""
  Best model   : {best_name}
  Test accuracy: {best_row['accuracy']*100:.2f}%
  F1-score     : {best_row['f1_score']*100:.2f}%
  Overfit gap  : {best_row['overfit_gap']:+.4f}

  Saved:
    models/best_model.pkl
    models/tfidf_vectorizer.pkl
    results/model_scores.csv
    results/*_confusion_matrix.png
""")

    # ── 11. Prediction test ───────────────────────────────────────────────
    prediction_test(final_model, best_vec)

    banner("TRAINING COMPLETE")
    print(f"""
  [OK] Best model  : {best_name}
  [OK] F1-score    : {best_row['f1_score']*100:.2f}%
  [OK] Models saved to models/

  Next step: run  streamlit run app.py  to use the updated model in the web app.
""")


if __name__ == "__main__":
    train()
