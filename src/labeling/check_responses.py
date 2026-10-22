import sqlite3
from pathlib import Path

import pandas as pd

con = sqlite3.connect(Path(__file__).resolve().parents[2] / "data" / "router.db")
print(pd.read_sql("""SELECT tier, COUNT(*) AS calls,
    ROUND(AVG(latency_s), 2) AS avg_seconds,
    ROUND(AVG(total_tokens)) AS avg_tokens,
    SUM(finish_reason = 'length') AS truncated,
    SUM(reply IS NULL OR reply = '') AS empty
    FROM responses GROUP BY tier""", con))