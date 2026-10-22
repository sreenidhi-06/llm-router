import sqlite3
from pathlib import Path

import pandas as pd

con = sqlite3.connect(Path(__file__).resolve().parents[2] / "data" / "router.db")
df = pd.read_sql("SELECT * FROM labels", con)


def policy(name, use_cheap):
    acc = df.cheap_correct.where(use_cheap, df.strong_correct).mean()
    cost = df.cheap_cost.where(use_cheap, df.strong_cost).sum()
    return {"policy": name, "accuracy": round(acc, 3), "cost_usd": round(cost, 6)}


rows = [policy("always cheap", pd.Series(True, index=df.index)),
        policy("always strong", pd.Series(False, index=df.index)),
        policy("perfect router", df.cheap_correct.astype(bool))]
out = pd.DataFrame(rows)
out["cost_vs_strong"] = (out.cost_usd / out.loc[1, "cost_usd"]).round(2)
print(out.to_string(index=False))
print("\nmax possible saving vs always-strong:",
      f"{1 - out.loc[2, 'cost_usd'] / out.loc[1, 'cost_usd']:.0%}")