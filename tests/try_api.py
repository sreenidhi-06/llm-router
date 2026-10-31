import requests

base = "http://127.0.0.1:8000"
r = requests.post(f"{base}/chat", json={"query": "What is idempotency in a data pipeline?", "policy": "mid"})
print(r.status_code, r.json())

r = requests.post(f"{base}/compare", json={"query": "Which SQL clause filters groups after aggregation?"})
for a in r.json()["results"]:
    print(a["tier"], f'{a["latency_s"]}s', f'${a["cost_usd"]}', "|", a["answer"][:80])