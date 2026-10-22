import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq, RateLimitError

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
DB = ROOT / "data" / "router.db"
MODELS = {"cheap": "allam-2-7b", "mid": "openai/gpt-oss-20b", "strong": "openai/gpt-oss-120b"}
PAUSE = 2.2  # seconds between calls, keeps us under about 30 requests per minute

client = Groq(api_key=os.environ["GROQ_API_KEY"])

LETTER_RULE = "Reply with only the letter (A, B, C, or D) of the correct answer.\n\n"
MATH_RULE = ("Solve the problem. Finish with a last line in exactly this form:\n"
             "Final answer: <number>\n\n")


def build_prompt(source, prompt):
    return (MATH_RULE if source == "gsm8k" else LETTER_RULE) + prompt


def init_db(con):
    con.execute("""CREATE TABLE IF NOT EXISTS responses (
        query_id INTEGER, tier TEXT, model TEXT, reply TEXT, finish_reason TEXT,
        prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER,
        latency_s REAL, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (query_id, tier))""")
    con.commit()


def call_model(model, text, max_tokens):
    for attempt in range(1, 6):
        try:
            t0 = time.time()
            kwargs = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
            r = client.chat.completions.create(
                model=model, temperature=0, max_tokens=max_tokens,
                messages=[{"role": "user", "content": text}], **kwargs)
            return r, time.time() - t0
        except RateLimitError as e:
            if "per day" in str(e).lower():
                print("Daily limit reached. Run the script again later; progress is saved.")
                sys.exit(0)
            wait = 20 * attempt
            print(f"  rate limited, waiting {wait}s (attempt {attempt}/5)")
            time.sleep(wait)
        except Exception as e:
            print(f"  error: {e}")
            time.sleep(5)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="size of the trial sample")
    ap.add_argument("--all", action="store_true", help="label every query")
    ap.add_argument("--tiers", default="cheap,strong", help="comma-separated tiers to run")
    args = ap.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    unknown = [t for t in tiers if t not in MODELS]
    if unknown:
        sys.exit(f"Unknown tier(s): {unknown}. Choose from {list(MODELS)}")

    con = sqlite3.connect(DB)
    init_db(con)
    q = pd.read_sql("SELECT id, source, prompt FROM queries", con)
    if not args.all:
        q = q.sample(n=min(args.n, len(q)), random_state=42)

    done = set(con.execute("SELECT query_id, tier FROM responses").fetchall())
    todo = [(int(r.id), r.source, r.prompt, tier)
            for r in q.itertuples() for tier in tiers
            if (int(r.id), tier) not in done]
    print(f"{len(todo)} calls to make")

    for i, (qid, source, prompt, tier) in enumerate(todo, 1):
        max_tokens = 2048 if source == "gsm8k" else 1024
        r, secs = call_model(MODELS[tier], build_prompt(source, prompt), max_tokens)
        if r is None:
            print(f"  skipped query {qid} ({tier}) after repeated errors")
            continue
        ch = r.choices[0]
        con.execute(
            """INSERT OR REPLACE INTO responses
               (query_id, tier, model, reply, finish_reason,
                prompt_tokens, completion_tokens, total_tokens, latency_s)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (qid, tier, MODELS[tier], ch.message.content, ch.finish_reason,
             r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.total_tokens, secs))
        con.commit()
        print(f"{i}/{len(todo)}  query {qid} ({tier})  {ch.finish_reason}  {secs:.1f}s")
        time.sleep(PAUSE)
    print("Finished.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped. Progress is saved; run again to resume.")