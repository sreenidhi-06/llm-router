import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_router import NUMERIC as PROMPT_NUMERIC, load

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

ANSWER_NUMERIC = ["ans_tokens", "ans_len", "ans_bare", "ans_has_final"]


def add_answer_features(df):
    reply = df.cheap_reply.fillna("")
    df["ans_tokens"] = df.cheap_completion_tokens
    df["ans_len"] = reply.str.len()
    df["ans_bare"] = reply.str.strip().str.fullmatch(r"[A-D]\.?").astype(int)
    df["ans_has_final"] = reply.str.contains("final answer", case=False).astype(int)
    return df


def make_model(clf, numeric):
    pre = ColumnTransformer([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000), "prompt"),
        ("num", StandardScaler(), numeric)])
    return Pipeline([("pre", pre), ("clf", clf)])


def cascade_sweep(d, p):
    c_cost, s_cost = d.cheap_cost.sum(), d.strong_cost.sum()
    c_acc, s_acc = d.cheap_correct.mean(), d.strong_correct.mean()
    rows = []
    for t in np.arange(0.3, 0.96, 0.05):
        escalate = p < t                      # low confidence in the cheap answer -> ask the strong model
        acc = np.where(escalate, d.strong_correct, d.cheap_correct).mean()
        cost = c_cost + d.strong_cost[escalate].sum()   # the cheap model is always called first
        rand_acc = c_acc + ((cost - c_cost) / s_cost) * (s_acc - c_acc)
        rows.append({"threshold": round(t, 2), "escalated": round(escalate.mean(), 2),
                     "accuracy": round(acc, 3), "cost_vs_strong": round(cost / s_cost, 2),
                     "random_escalation_acc": round(rand_acc, 3),
                     "gain_vs_random": round(acc - rand_acc, 3)})
    return pd.DataFrame(rows)


def main():
    df = add_answer_features(load())
    tr, va = df[df.split == "train"], df[df.split == "val"]
    sets = {"question only": PROMPT_NUMERIC, "question + cheap answer": PROMPT_NUMERIC + ANSWER_NUMERIC}
    models = {"logistic regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
              "random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                      class_weight="balanced", random_state=42, n_jobs=-1)}
    rows, probs = [], {}
    for fs, numeric in sets.items():
        cols = ["prompt"] + numeric
        for name, clf in models.items():
            pipe = make_model(clf, numeric)
            cv = cross_val_score(pipe, tr[cols], tr.y, cv=5, scoring="roc_auc").mean()
            pipe.fit(tr[cols], tr.y)
            p = pipe.predict_proba(va[cols])[:, 1]
            probs[(fs, name)] = p
            rows.append({"features": fs, "model": name, "cv_auc": round(cv, 3),
                         "val_auc": round(roc_auc_score(va.y, p), 3)})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))

    best = res[res.features == "question + cheap answer"].sort_values("cv_auc").iloc[-1]
    print(f"\ncascade sweep on validation: {best.model} with answer features")
    table = cascade_sweep(va, probs[("question + cheap answer", best.model)])
    print(table.to_string(index=False))
    print(f"\nbest gain over random escalation: {table.gain_vs_random.max():+.3f}")

    wrong = va.cheap_correct == 0
    print("\nvalidation references:")
    print("  always cheap   : accuracy", round(va.cheap_correct.mean(), 3), "| cost_vs_strong",
          round(va.cheap_cost.sum() / va.strong_cost.sum(), 2))
    print("  always strong  : accuracy", round(va.strong_correct.mean(), 3), "| cost_vs_strong 1.0")
    print("  perfect cascade: accuracy", round((va.cheap_correct | va.strong_correct).mean(), 3),
          "| cost_vs_strong", round((va.cheap_cost.sum() + va.strong_cost[wrong].sum())
                                    / va.strong_cost.sum(), 2))


if __name__ == "__main__":
    main()