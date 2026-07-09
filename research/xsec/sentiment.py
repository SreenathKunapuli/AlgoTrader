"""LLM-distilled news sentiment: Claude labels -> TF-IDF + logistic student.

Teacher: Claude (in-session) labeled 682 Benzinga headlines {-1, 0, +1} for
the tagged stock. Student: character+word TF-IDF into multinomial logistic
regression — runs in microseconds with no API dependency, so it can score
every headline the engine sees live.

Honesty constraints, in order of importance:
1. Teacher agreement is measured on a held-out split the student never saw.
2. The student's OUTPUT is a feature candidate, not alpha: its return
   predictiveness must be established in the same walk-forward harness as
   every other feature before it earns portfolio weight.
3. 682 labels is a pilot. The scaling path is labeling more headlines with
   the same rubric (cheap: one batch prompt per ~100 headlines).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

MODEL_PATH = "models/news_sentiment.pkl"


def build_student() -> Pipeline:
    return Pipeline([
        ("tfidf", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000,
                                     sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     min_df=2, max_features=30000, sublinear_tf=True)),
        ])),
        ("clf", LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced")),
    ])


def train_student(labels_csv: str, model_path: str = MODEL_PATH,
                  holdout_frac: float = 0.2, seed: int = 0) -> dict[str, float]:
    df = pd.read_csv(labels_csv)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_hold = int(len(df) * holdout_frac)
    hold, tr = df.iloc[idx[:n_hold]], df.iloc[idx[n_hold:]]

    student = build_student()
    student.fit(tr["headline"], tr["label"])
    pred = student.predict(hold["headline"])
    metrics = {
        "holdout_accuracy": float(accuracy_score(hold["label"], pred)),
        "holdout_macro_f1": float(f1_score(hold["label"], pred, average="macro")),
        "n_train": len(tr), "n_holdout": len(hold),
        # directional confusion is the costly error: teacher said +1, student says -1
        "sign_flip_rate": float(np.mean((hold["label"].values * pred) < 0)),
    }
    # refit on everything before saving — holdout was for measurement only
    final = build_student()
    final.fit(df["headline"], df["label"])
    Path(model_path).parent.mkdir(exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(final, f)
    return metrics


def load_student(model_path: str = MODEL_PATH):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def score_headlines(headlines: list[str], model_path: str = MODEL_PATH) -> np.ndarray:
    """Expected sentiment in [-1, 1]: sum(class * P(class))."""
    student = load_student(model_path)
    proba = student.predict_proba(headlines)
    classes = np.array(student.classes_, dtype=float)
    return proba @ classes
