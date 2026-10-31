import sqlite3

import pandas as pd
from grade import DB, PRICES, grade_letter, grade_number

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

if "mid" not in PRICES:
    raise SystemExit('Add a "mid" entry to PRICES in grade.py first.')

con = sqlite3.connect(DB)
q = pd.read_sql("SELECT id AS query_id, source, gold_answer FROM queries", con)
r = pd.read_sql("SELECT query_id, tier, reply, prompt_tokens, completion_tokens FROM responses", con)
r = r.merge(q, on="query_id")
r = r[r.groupby("query_id").tier.transform("nunique") == 3]   # keep questions answered by all three tiers

r["correct"] = [bool((grade_number if s == "gsm8k" else grade_letter)(rep, str(g).strip()))
                for s, rep, g in zip(r.source, r.reply, r.gold_answer)]
r["cost"] = [(pt * PRICES[t][0] + ct * PRICES[t][1]) / 1e6
             for t, pt, ct in zip(r.tier, r.prompt_tokens, r.completion_tokens)]

summary = r.groupby("tier").agg(accuracy=("correct", "mean"), cost_usd=("cost", "sum")).round(4)
summary["cost_vs_strong"] = (summary.cost_usd / summary.loc["strong", "cost_usd"]).round(2)
print(f"questions answered by all three tiers: {r.query_id.nunique()}\n")
print(summary.loc[["cheap", "mid", "strong"]])
print("\naccuracy by source:")
print(r.pivot_table(index="source", columns="tier", values="correct", aggfunc="mean")
      [["cheap", "mid", "strong"]].round(3))