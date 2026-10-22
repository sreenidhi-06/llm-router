import sqlite3

import pandas as pd
from grade import DB, grade_letter, grade_number

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

con = sqlite3.connect(DB)
q = pd.read_sql("SELECT id AS query_id, source, gold_answer FROM queries", con)
r = pd.read_sql("SELECT query_id, tier, reply FROM responses", con).merge(q, on="query_id")
r["correct"] = [(grade_number if s == "gsm8k" else grade_letter)(rep, str(g).strip())
                for s, rep, g in zip(r.source, r.reply, r.gold_answer)]
r["gradable"] = r.correct.notna()
r["correct"] = r.correct.fillna(False).astype(bool)

print(r.groupby(["tier", "source"]).agg(
    n=("query_id", "count"), acc=("correct", "mean"),
    ungradable=("gradable", lambda s: int((~s).sum()))).round(3))
print("\noverall accuracy by tier:")
print(r.groupby("tier").correct.mean().round(3))

wrong = r[r.gradable & ~r.correct]
print("\nwrong answers (first 10):")
for row in wrong.head(10).itertuples():
    print(row.tier, row.query_id, row.source, "gold:", row.gold_answer,
          "| reply:", repr(str(row.reply)[:80]))