import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

ROOT = Path(__file__).resolve().parents[2]
NUMERIC = ["prompt_chars", "word_count", "n_digits", "n_lines", "has_dollar"]
FEATURES = ["prompt"] + NUMERIC


def load():
    con = sqlite3.connect(ROOT / "data" / "router.db")
    df = pd.read_sql("""SELECT l.*, q.prompt, q.prompt_chars, q.word_count
                        FROM labels l JOIN queries q ON q.id = l.query_id""", con)
    df["n_digits"] = df.prompt.str.count(r"\d")
    df["n_lines"] = df.prompt.str.count("\n") + 1
    df["has_dollar"] = df.prompt.str.contains(r"[$€£]").astype(int)
    bucket = df.query_id % 10          # fixed split: 60% train, 20% val, 20% test
    df["split"] = np.where(bucket < 6, "train", np.where(bucket < 8, "val", "test"))
    df["y"] = df.cheap_correct.astype(int)   # 1 = the cheap model answered correctly
    return df


def make_model(clf):
    pre = ColumnTransformer([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000), "prompt"),
        ("num", StandardScaler(), NUMERIC)])
    return Pipeline([("pre", pre), ("clf", clf)])


def sweep(d, p):
    rows = []
    for t in np.arange(0.3, 0.96, 0.05):
        use_cheap = p >= t
        acc = np.where(use_cheap, d.cheap_correct, d.strong_correct).mean()
        cost = np.where(use_cheap, d.cheap_cost, d.strong_cost).sum()
        rows.append({"threshold": round(t, 2), "cheap_share": round(use_cheap.mean(), 2),
                     "accuracy": round(acc, 3),
                     "cost_vs_strong": round(cost / d.strong_cost.sum(), 2)})
    return pd.DataFrame(rows)


def main():
    df = load()
    tr, va = df[df.split == "train"], df[df.split == "val"]
    print("rows per split:", {s: int((df.split == s).sum()) for s in ("train", "val", "test")})

    models = {
        "majority baseline": DummyClassifier(strategy="prior"),
        "logistic regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                class_weight="balanced", random_state=42, n_jobs=-1),
    }
    results, probs = [], {}
    for name, clf in models.items():
        pipe = make_model(clf)
        cv_auc = cross_val_score(pipe, tr[FEATURES], tr.y, cv=5, scoring="roc_auc").mean()
        pipe.fit(tr[FEATURES], tr.y)
        p = pipe.predict_proba(va[FEATURES])[:, 1]
        pred = (p >= 0.5).astype(int)
        probs[name] = p
        results.append({"model": name, "cv_auc": round(cv_auc, 3),
                        "val_auc": round(roc_auc_score(va.y, p), 3),
                        "accuracy": round(accuracy_score(va.y, pred), 3),
                        "precision": round(precision_score(va.y, pred, zero_division=0), 3),
                        "recall": round(recall_score(va.y, pred, zero_division=0), 3),
                        "f1": round(f1_score(va.y, pred, zero_division=0), 3)})
    res = pd.DataFrame(results)
    print("\nvalidation metrics (label = cheap model is correct):")
    print(res.to_string(index=False))

    best = res[res.model != "majority baseline"].sort_values("val_auc").iloc[-1].model
    print(f"\nthreshold sweep on validation for: {best}")
    print(sweep(va, probs[best]).to_string(index=False))

    print("\nvalidation baselines:")
    print("  always cheap : accuracy", round(va.cheap_correct.mean(), 3), "| cost_vs_strong",
          round(va.cheap_cost.sum() / va.strong_cost.sum(), 2))
    print("  always strong: accuracy", round(va.strong_correct.mean(), 3), "| cost_vs_strong 1.0")
    print("  perfect      : accuracy", round((va.cheap_correct | va.strong_correct).mean(), 3),
          "| cost_vs_strong", round(np.where(va.cheap_correct == 1, va.cheap_cost,
                                              va.strong_cost).sum() / va.strong_cost.sum(), 2))


if __name__ == "__main__":
    main()