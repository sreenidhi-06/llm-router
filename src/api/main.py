import logging
import os
import sqlite3
import time
import uuid
from fastapi.responses import RedirectResponse
from contextlib import closing
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import Groq, RateLimitError
from pydantic import BaseModel, Field

from config import DEFAULT_POLICY, MAX_TOKENS, MODELS, PRICES

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
DB = ROOT / "data" / "router.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("router-api")

client = Groq(api_key=os.environ["GROQ_API_KEY"])
app = FastAPI(title="Cost-Aware LLM Router API")

Tier = Literal["cheap", "mid", "strong"]
SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    policy: Tier = DEFAULT_POLICY


class Answer(BaseModel):
    tier: Tier
    model: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float


class ChatResponse(BaseModel):
    request_id: str
    policy: Tier
    result: Answer


class CompareResponse(BaseModel):
    request_id: str
    results: list[Answer]


def ask(tier: str, query: str) -> Answer:
    model = MODELS[tier]
    kwargs = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
    for attempt in range(1, 4):
        try:
            t0 = time.time()
            r = client.chat.completions.create(
                model=model, temperature=0, max_tokens=MAX_TOKENS,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": query}], **kwargs)
            latency = time.time() - t0
            break
        except RateLimitError:
            log.warning("rate limited on %s (attempt %d)", tier, attempt)
            time.sleep(10 * attempt)
        except Exception:
            log.exception("model call failed for %s", tier)
            raise HTTPException(status_code=502, detail=f"{tier} model call failed")
    else:
        raise HTTPException(status_code=429, detail=f"{tier} model is rate limited; try again shortly")

    pin, pout = PRICES[tier]
    u = r.usage
    cost = (u.prompt_tokens * pin + u.completion_tokens * pout) / 1e6
    return Answer(tier=tier, model=model, answer=r.choices[0].message.content or "",
                  prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                  cost_usd=round(cost, 8), latency_s=round(latency, 3))


def log_request(request_id: str, endpoint: str, a: Answer, query_chars: int) -> None:
    """Stores usage only (no question text), so the log holds no user content."""
    try:
        with closing(sqlite3.connect(DB)) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS api_requests (
                request_id TEXT, ts TEXT, endpoint TEXT, tier TEXT, model TEXT,
                query_chars INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER,
                cost_usd REAL, latency_s REAL)""")
            con.execute("INSERT INTO api_requests VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?)",
                        (request_id, endpoint, a.tier, a.model, query_chars,
                         a.prompt_tokens, a.completion_tokens, a.cost_usd, a.latency_s))
            con.commit()
    except Exception:
        log.exception("could not write request log")

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")
@app.get("/health")
def health():
    return {"status": "ok", "models": MODELS, "default_policy": DEFAULT_POLICY}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    rid = str(uuid.uuid4())
    a = ask(req.policy, req.query)
    log.info("req=%s policy=%s tokens=%d cost=%.6f latency=%.2fs", rid, req.policy,
             a.prompt_tokens + a.completion_tokens, a.cost_usd, a.latency_s)
    log_request(rid, "chat", a, len(req.query))
    return ChatResponse(request_id=rid, policy=req.policy, result=a)


@app.post("/compare", response_model=CompareResponse)
def compare(req: ChatRequest):
    rid = str(uuid.uuid4())
    results = [ask(t, req.query) for t in ("cheap", "mid", "strong")]
    for a in results:
        log.info("req=%s compare tier=%s cost=%.6f latency=%.2fs", rid, a.tier, a.cost_usd, a.latency_s)
        log_request(rid, "compare", a, len(req.query))
    return CompareResponse(request_id=rid, results=results)