import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="LLM Router Answerboard", layout="wide")
st.title("Cost-Aware LLM Router: Answerboard")
st.caption("Compare answers from three model tiers side by side, with cost and speed.")

LABELS = {"cheap": "Cheap", "mid": "Mid (default)", "strong": "Strong"}

with st.sidebar:
    api_url = st.text_input("API URL", "http://127.0.0.1:8000")
    mode = st.radio("Mode", ["Compare all tiers", "Single policy"])
    policy = st.selectbox("Policy", ["mid", "strong", "cheap"], disabled=(mode == "Compare all tiers"))
    st.markdown("**Why mid is the default:** on 700 benchmark questions it reached about 98% of the "
                "strong model's accuracy at about 43% of the cost. The cheap tier was correct on only "
                "about 59%, so check its answers.")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_area("Question", height=120,
                     placeholder="For example: Which SQL clause filters groups after aggregation?")
run = st.button("Ask", type="primary", disabled=not query.strip())


def call_api(path, payload):
    try:
        r = requests.post(f"{api_url}{path}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API. Start it with: python -m uvicorn --app-dir src\\api main:app")
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text[:300]}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None


def show(a, strong_cost=None):
    st.subheader(LABELS[a["tier"]])
    st.caption(a["model"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Cost", f"${a['cost_usd']:.6f}")
    c2.metric("Latency", f"{a['latency_s']:.2f}s")
    c3.metric("Tokens", a["prompt_tokens"] + a["completion_tokens"])
    if strong_cost:
        st.caption(f"{a['cost_usd'] / strong_cost:.0%} of the strong model's cost")
    st.write(a["answer"])


answers = []
if run:
    with st.spinner("Asking the models..."):
        if mode == "Compare all tiers":
            data = call_api("/compare", {"query": query})
            if data:
                answers = data["results"]
                by_tier = {a["tier"]: a for a in answers}
                cols = st.columns(3)
                for col, tier in zip(cols, ["cheap", "mid", "strong"]):
                    with col:
                        show(by_tier[tier], by_tier["strong"]["cost_usd"])
        else:
            data = call_api("/chat", {"query": query, "policy": policy})
            if data:
                answers = [data["result"]]
                show(data["result"])

for a in answers:
    st.session_state.history.append({"tier": a["tier"], "cost_usd": a["cost_usd"],
                                     "latency_s": a["latency_s"]})

if st.session_state.history:
    h = pd.DataFrame(st.session_state.history)
    st.divider()
    st.subheader("This session")
    st.dataframe(h.groupby("tier").agg(calls=("cost_usd", "count"),
                                       total_cost_usd=("cost_usd", "sum"),
                                       avg_latency_s=("latency_s", "mean")).round(6))