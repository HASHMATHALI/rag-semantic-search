"""Cartograph - semantic search that shows you the map, and where it ends."""
import html
import os
import re

import plotly.graph_objects as go
import requests
import streamlit as st

SEA, DEEP, SAND, CORAL, KELP, FOAM = "#0E2A2B", "#0A1F20", "#E9DFC7", "#FF6F59", "#7FB69A", "#9DB8B5"
VERDICTS = {
    "charted": (KELP, "Charted waters", "The index holds passages that are genuinely about this."),
    "frontier": ("#E3B95B", "Frontier", "Only loosely related passages exist. Read the results with care."),
    "uncharted": (CORAL, "Off the map", "Nothing indexed is about this. The closest passages are shown, but they are not answers."),
}

st.set_page_config(page_title="Cartograph", page_icon="🧭", layout="wide")
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,700;1,500&family=Karla:wght@400;500;700&display=swap');
.stApp {{ background: radial-gradient(1200px 600px at 80% -10%, #16403F 0%, {SEA} 55%, {DEEP} 100%); }}
html, body, [class*="css"], .stMarkdown p, label {{ font-family:'Karla',sans-serif; color:{SAND}; }}
.plate {{ display:flex; align-items:flex-end; gap:18px; margin-bottom:4px; }}
.plate .name {{ font-family:'Cormorant Garamond',serif; font-weight:700; font-size:3.4rem; line-height:.9; color:{SAND}; }}
.plate .tag {{ font-family:'Cormorant Garamond',serif; font-style:italic; font-size:1.25rem; color:{FOAM}; padding-bottom:6px; }}
.rule {{ height:1px; background:repeating-linear-gradient(90deg,{FOAM} 0 10px,transparent 10px 16px); opacity:.5; margin:6px 0 18px; }}
.bearing {{ border:1px solid; border-radius:2px; padding:12px 16px; margin:4px 0 14px; }}
.bearing .head {{ font-family:'Cormorant Garamond',serif; font-weight:700; font-size:1.5rem; }}
.passage {{ border-left:2px solid {FOAM}66; padding:6px 0 10px 16px; margin-bottom:14px; }}
.passage .meta {{ font-size:.74rem; letter-spacing:.1em; text-transform:uppercase; color:{FOAM}; margin-bottom:4px; }}
.passage .body {{ font-size:1rem; line-height:1.55; color:{SAND}; }}
.passage mark {{ background:{CORAL}33; color:{SAND}; border-bottom:2px solid {CORAL}; padding:0 2px; }}
.pill {{ display:inline-block; font-size:.68rem; letter-spacing:.08em; padding:1px 7px; border:1px solid {FOAM}; border-radius:10px; margin-left:6px; color:{FOAM}; }}
.pill.both {{ border-color:{KELP}; color:{KELP}; }}
.answer {{ background:{DEEP}; border:1px solid {SAND}33; padding:16px 20px; font-size:1.05rem; line-height:1.6; margin-bottom:18px; }}
.answer sup {{ color:{CORAL}; font-weight:700; }}
</style>
<div class="plate"><span class="name">Cartograph</span><span class="tag">search that knows where its map ends</span></div>
<div class="rule"></div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Instruments")
    mode = st.radio("Engines", ["hybrid", "semantic", "lexical"], captions=["meaning + exact words, fused", "meaning only", "exact words only (BM25)"])
    top_k = st.slider("Passages", 1, 20, 5)
    diversity = st.slider("Spread", 0.0, 1.0, 0.0, 0.1, help="0 ranks purely by relevance. Higher values push near-duplicate passages down so you see more of the neighbourhood.")
    use_rag = st.checkbox("Write an answer with citations", value=False, help="Needs GROQ_API_KEY on the backend. Never runs for off-the-map queries.")
    api_base = st.text_input("Backend", value=os.environ.get("BACKEND_URL", "http://localhost:8000")).rstrip("/")
    api_base = api_base[: -len("/search")] if api_base.endswith("/search") else api_base


@st.cache_data(ttl=600, show_spinner=False)
def fetch_map(base: str):
    return requests.get(f"{base}/map", timeout=30).json()


def highlight(text: str, evidence: str) -> str:
    """Escape the passage and mark the sentence the engine judged closest to the query."""
    safe, ev = html.escape(text), html.escape(evidence)
    return safe.replace(ev, f"<mark>{ev}</mark>", 1) if ev and ev != safe else safe


def cite(answer: str) -> str:
    return re.sub(r"\[(\d+)\]", r"<sup>[\1]</sup>", html.escape(answer))


query = st.text_input("Query", placeholder="Ask in your own words, e.g. how are hospitals using AI?", label_visibility="collapsed")

data = None
if query.strip():
    try:
        resp = requests.post(f"{api_base}/search", timeout=60,
                             json={"query": query, "top_k": top_k, "mode": mode, "diversity": diversity, "use_rag": use_rag})
        if resp.status_code == 200:
            data = resp.json()
        else:
            st.error(f"Backend error {resp.status_code}: {resp.text[:300]}")
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not reach the backend at {api_base}. Is FastAPI running? ({type(exc).__name__})")

chart_col, text_col = st.columns([5, 6])

with chart_col:
    try:
        corpus = fetch_map(api_base)
        pts = corpus["points"]
        fig = go.Figure()
        fig.add_scatter(x=[p["x"] for p in pts], y=[p["y"] for p in pts], mode="markers",
                        marker=dict(size=5, color=FOAM, opacity=0.45), text=[p["text"] for p in pts],
                        hovertemplate="%{text}<extra></extra>", name="indexed passages")
        if data:
            res = data["results"]
            qx, qy = data["query_xy"]
            for r in res:  # rhumb lines from the query to each hit
                fig.add_scatter(x=[qx, r["x"]], y=[qy, r["y"]], mode="lines", line=dict(color=SAND, width=0.6, dash="dot"), hoverinfo="skip", showlegend=False)
            fig.add_scatter(x=[r["x"] for r in res], y=[r["y"] for r in res], mode="markers+text", text=[str(i + 1) for i in range(len(res))],
                            textposition="top center", textfont=dict(color=SAND, size=12), marker=dict(size=11, color=SAND, line=dict(color=DEEP, width=1)),
                            hovertext=[r["text"][:90] for r in res], hoverinfo="text", name="retrieved")
            fig.add_scatter(x=[qx], y=[qy], mode="markers", marker=dict(size=16, color=CORAL, symbol="x-thin", line=dict(color=CORAL, width=3)), name="your query", hoverinfo="name")
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Karla, sans-serif", color=SAND), legend=dict(orientation="h", y=-0.04, x=0))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, scaleanchor="x")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Every indexed passage ({corpus['total']:,} in all, sampled) flattened to two dimensions. Nearby dots mean similar meaning. "
                   "The projection keeps only the two strongest directions, so treat distances as a guide.")
    except Exception:
        st.info("The map appears once the backend is reachable.")

with text_col:
    if not data:
        st.markdown("Type a question above. Cartograph retrieves with two engines at once, shows where your query lands "
                    "among everything indexed, and tells you plainly when the index has nothing to say.")
    else:
        cov = data["coverage"]
        colour, title, blurb = VERDICTS[cov["verdict"]]
        st.markdown(
            f"<div class='bearing' style='border-color:{colour}'><div class='head' style='color:{colour}'>{title}</div>{blurb}"
            f"<br><span style='font-size:.8rem;color:{FOAM}'>closest passage: similarity {cov['best_cosine']:.2f} · "
            f"farther than {(1 - cov['score']):.0%} of passages are from their own nearest neighbour</span></div>",
            unsafe_allow_html=True,
        )
        if data.get("rag_response"):
            st.markdown(f"<div class='answer'>{cite(data['rag_response'])}</div>", unsafe_allow_html=True)
        elif data.get("rag_skipped_reason"):
            st.markdown(f"<div class='answer' style='border-color:{CORAL}'>{html.escape(data['rag_skipped_reason'])}</div>", unsafe_allow_html=True)

        cited = set(data.get("cited") or [])
        for n, r in enumerate(data["results"], start=1):
            both = len(r["found_by"]) > 1
            pill = f"<span class='pill {'both' if both else ''}'>{' + '.join(r['found_by']) if both else r['found_by'][0] + ' only'}</span>"
            cited_pill = f"<span class='pill' style='border-color:{CORAL};color:{CORAL}'>cited</span>" if n in cited else ""
            st.markdown(
                f"<div class='passage'><div class='meta'>{n} · passage {int(r['id'])} · distance {r['score']:.2f}{pill}{cited_pill}</div>"
                f"<div class='body'>{highlight(r['text'], r['evidence'])}</div></div>",
                unsafe_allow_html=True,
            )
