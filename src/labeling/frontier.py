import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from grade import DB, PRICES, grade_letter, grade_number

ROOT = Path(__file__).resolve().parents[2]
con = sqlite3.connect(DB)
q = pd.read_sql("SELECT id AS query_id, source, gold_answer FROM queries", con)
r = pd.read_sql("SELECT query_id, tier, reply, prompt_tokens, completion_tokens FROM responses",
                con).merge(q, on="query_id")
r = r[r.groupby("query_id").tier.transform("nunique") == 3].copy()
r["correct"] = [bool((grade_number if s == "gsm8k" else grade_letter)(rep, str(g).strip()))
                for s, rep, g in zip(r.source, r.reply, r.gold_answer)]
r["cost"] = [(pt * PRICES[t][0] + ct * PRICES[t][1]) / 1e6
             for t, pt, ct in zip(r.tier, r.prompt_tokens, r.completion_tokens)]

order = ["cheap", "mid", "strong"]
ok = r.pivot(index="query_id", columns="tier", values="correct")[order].astype(bool).to_numpy()
cost = r.pivot(index="query_id", columns="tier", values="cost")[order].to_numpy()
n, S = len(ok), cost[:, 2].sum()


def point(idx):
    rows = np.arange(n)
    return ok[rows, idx].mean(), cost[rows, idx].sum() / S


policies = {
    "always cheap": np.zeros(n, int),
    "always mid": np.ones(n, int),
    "always strong": np.full(n, 2),
    "perfect cheap->mid": np.where(ok[:, 0], 0, 1),
    "perfect mid->strong": np.where(ok[:, 1], 1, 2),
    "perfect 3-way": np.where(ok[:, 0], 0, np.where(ok[:, 1], 1, 2)),
}
pts = {name: point(idx) for name, idx in policies.items()}
print(f"questions: {n}")
print(pd.DataFrame([{"policy": k, "accuracy": round(a, 3), "cost_vs_strong": round(c, 2)}
                    for k, (a, c) in pts.items()]).to_string(index=False))
print("\nmid wrong, strong right:", int((~ok[:, 1] & ok[:, 2]).sum()),
      "| mid right, strong wrong:", int((ok[:, 1] & ~ok[:, 2]).sum()))
print("cheap right, mid wrong:", int((ok[:, 0] & ~ok[:, 1]).sum()),
      "| cheap wrong, mid right:", int((~ok[:, 0] & ok[:, 1]).sum()))

fig, ax = plt.subplots(figsize=(7, 5))
base = [pts["always cheap"], pts["always mid"], pts["always strong"]]
ax.plot([p[1] for p in base], [p[0] for p in base], "--", color="gray", label="random mix of neighbouring tiers")
for name, (a, c) in pts.items():
    perfect = name.startswith("perfect")
    ax.scatter(c, a, s=60, marker="*" if perfect else "o", color="tab:green" if perfect else "tab:blue", zorder=3)
    ax.annotate(name, (c, a), textcoords="offset points", xytext=(6, -12), fontsize=8)
ax.set_xlabel("cost relative to always-strong (list price)")
ax.set_ylabel("accuracy")
ax.set_title("Cost vs accuracy across model tiers")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
out = ROOT / "reports" / "figures" / "tier_frontier.png"
plt.savefig(out, dpi=150)
print("\nsaved", out)