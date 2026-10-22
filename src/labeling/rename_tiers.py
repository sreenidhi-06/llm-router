import sqlite3
from pathlib import Path

con = sqlite3.connect(Path(__file__).resolve().parents[2] / "data" / "router.db")
if con.execute("SELECT COUNT(*) FROM responses WHERE tier = 'small'").fetchone()[0] == 0:
    print("Nothing to do: tiers already renamed.")
else:
    con.execute("UPDATE responses SET tier = 'mid' WHERE tier = 'cheap'")
    con.execute("UPDATE responses SET tier = 'cheap' WHERE tier = 'small'")
    con.commit()
    print("Renamed: gpt-oss-20b is now 'mid'; allam-2-7b is now 'cheap'.")
print(con.execute("SELECT tier, model, COUNT(*) FROM responses GROUP BY tier, model").fetchall())