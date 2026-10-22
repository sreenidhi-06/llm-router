import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "router.db"

# USD per 1M tokens (input, output). cheap = stand-in price for a small 7-8B model
# (Llama 3.1 8B Instant); strong = gpt-oss-120b. Confirm in the Groq console docs.
PRICES = {"cheap": (0.05, 0.08), "strong": (0.15, 0.60)}


def grade_letter(reply, gold):
    if not reply:
        return None
    text = reply.strip()
    bare = re.sub(r"[^A-Za-z]", "", text)
    if len(bare) == 1 and bare.upper() in "ABCD":
        return bare.upper() == gold
    m = re.search(r"(?i:answer|option|choice)\s*(?:is|:)?\s*\(?([A-D])\b", text)
    if m:
        return m.group(1) == gold
    m = re.search(r"\b([A-D])\b", text)
    return (m.group(1) == gold) if m else None


def grade_number(reply, gold):
    if not reply:
        return None
    m = re.search(r"final answer:?\s*\$?\s*(-?[\d,]*\.?\d+)", reply, re.I)
    if m:
        raw = m.group(1)
    else:
        nums = re.findall(r"-?[\d,]*\.?\d+", reply)
        if not nums:
            return None
        raw = nums[-1]
    try:
        return abs(float(raw.replace(",", "")) - float(gold)) < 1e-6
    except ValueError:
        return None


def main():
    con = sqlite3.connect(DB)
    q = pd.read_sql("SELECT id AS query_id, source, domain, subject, gold_answer FROM queries", con)
    r = pd.read_sql("SELECT * FROM responses", con)
    wide = r.pivot(index="query_id", columns="tier",
                   values=["reply", "finish_reason", "prompt_tokens",
                           "completion_tokens", "latency_s"])
    wide.columns = [f"{tier}_{field}" for field, tier in wide.columns]
    df = q.merge(wide.reset_index(), on="query_id")
    df = df.dropna(subset=["cheap_finish_reason", "strong_finish_reason"]).copy()

    for tier in ("cheap", "strong"):
        graded = [(grade_number if s == "gsm8k" else grade_letter)(rep, str(g).strip())
                  for s, rep, g in zip(df.source, df[f"{tier}_reply"], df.gold_answer)]
        df[f"{tier}_gradable"] = [x is not None for x in graded]
        df[f"{tier}_correct"] = [bool(x) for x in graded]
        pin, pout = PRICES[tier]
        df[f"{tier}_cost"] = (df[f"{tier}_prompt_tokens"] * pin
                              + df[f"{tier}_completion_tokens"] * pout) / 1e6
    df["oracle_correct"] = df.cheap_correct | df.strong_correct
    df.to_sql("labels", con, if_exists="replace", index=False)

    print(df.groupby("source").agg(
        n=("query_id", "count"),
        cheap_acc=("cheap_correct", "mean"),
        strong_acc=("strong_correct", "mean"),
        oracle_acc=("oracle_correct", "mean"),
        cheap_ungradable=("cheap_gradable", lambda s: int((~s).sum())),
        strong_ungradable=("strong_gradable", lambda s: int((~s).sum())),
    ).round(3))
    print("\noverall: cheap", round(df.cheap_correct.mean(), 3),
          "| strong", round(df.strong_correct.mean(), 3),
          "| oracle", round(df.oracle_correct.mean(), 3))
    print("cheap right, strong wrong:", int((df.cheap_correct & ~df.strong_correct).sum()),
          "| cheap wrong, strong right:", int((~df.cheap_correct & df.strong_correct).sum()))
    print("estimated list-price cost: cheap", round(df.cheap_cost.sum(), 6),
          "| strong", round(df.strong_cost.sum(), 6))

    bad = df[~df.cheap_gradable | ~df.strong_gradable]
    for row in bad.head(5).itertuples():
        print(row.query_id, row.source, "gold:", row.gold_answer,
              "| cheap:", repr(str(row.cheap_reply)[:80]),
              "| strong:", repr(str(row.strong_reply)[:80]))


if __name__ == "__main__":
    main()