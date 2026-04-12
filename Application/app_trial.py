"""
Biomedical Knowledge Exploration Pipeline
app.py — Fixed & Production-ready Streamlit application

Fixes applied:
  1. Network graph: uses fixed pixel dimensions (not window.innerWidth) so canvas renders inside iframe
  2. PubTator cache: moved to a module-level dict (thread-safe) instead of st.session_state
     (session_state is inaccessible from ThreadPoolExecutor threads)
  3. Ranking formula: matches network_module.py (strong > 70 & avg > 15 = Strong, etc.)
  4. New PMID-fetch tab: live PubMed lookup by PMID input
  5. RAG: two tabs — pipeline RAG (FAISS-grounded) and LLM-only with optional FAISS retrieval
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import time
import json
import os
import faiss
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.parse

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BioKnowledge Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

:root {
    --bg: #0a0f1e; --surface: #111827; --surface2: #1a2233;
    --border: #1e3a5f; --accent: #00d4ff; --accent2: #7c3aed;
    --success: #10b981; --warning: #f59e0b; --danger: #ef4444;
    --text: #e2e8f0; --muted: #64748b;
    --mono: 'IBM Plex Mono', monospace;
}

.stApp { background: var(--bg); color: var(--text); }
section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--text) !important; }

.bio-header {
    background: linear-gradient(135deg,#0a0f1e 0%,#111827 50%,#0d1b35 100%);
    border:1px solid var(--border); border-radius:12px;
    padding:2rem; margin-bottom:1.5rem; position:relative; overflow:hidden;
}
.bio-header::before {
    content:''; position:absolute; top:-50%; left:-50%; width:200%; height:200%;
    background: radial-gradient(circle at 30% 50%,rgba(0,212,255,.05) 0%,transparent 60%),
                radial-gradient(circle at 70% 50%,rgba(124,58,237,.05) 0%,transparent 60%);
    pointer-events:none;
}
.bio-title { font-family:var(--mono); font-size:2rem; font-weight:600; color:var(--accent); letter-spacing:-.02em; margin:0; }
.bio-subtitle { color:var(--muted); font-size:.9rem; margin-top:.3rem; font-family:var(--mono); }

.stage-label { font-family:var(--mono); font-size:.7rem; color:var(--accent); letter-spacing:.15em; text-transform:uppercase; margin-bottom:.5rem; }

.badge { display:inline-block; font-family:var(--mono); font-size:.7rem; padding:2px 8px; border-radius:4px; font-weight:600; letter-spacing:.05em; }
.badge-success { background:rgba(16,185,129,.15); color:var(--success); border:1px solid rgba(16,185,129,.3); }
.badge-warning { background:rgba(245,158,11,.15); color:var(--warning); border:1px solid rgba(245,158,11,.3); }
.badge-info    { background:rgba(0,212,255,.1);   color:var(--accent);  border:1px solid rgba(0,212,255,.2); }
.badge-error   { background:rgba(239,68,68,.15);  color:var(--danger);  border:1px solid rgba(239,68,68,.3); }

.metric-row { display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1rem; }
.metric-tile { flex:1; min-width:130px; background:var(--surface2); border:1px solid var(--border); border-radius:8px; padding:1rem; text-align:center; }
.metric-val  { font-family:var(--mono); font-size:1.8rem; font-weight:600; color:var(--accent); display:block; }
.metric-lbl  { font-size:.75rem; color:var(--muted); margin-top:.2rem; }

.pipeline-bar { display:flex; gap:4px; align-items:center; margin:1rem 0; flex-wrap:wrap; }
.pipe-step { font-family:var(--mono); font-size:.65rem; padding:4px 10px; border-radius:4px; letter-spacing:.05em; white-space:nowrap; }
.pipe-done   { background:rgba(16,185,129,.2);  color:var(--success); border:1px solid rgba(16,185,129,.4); }
.pipe-active { background:rgba(0,212,255,.15);  color:var(--accent);  border:1px solid rgba(0,212,255,.4); animation:pulse 1.5s infinite; }
.pipe-idle   { background:rgba(255,255,255,.04);color:var(--muted);   border:1px solid rgba(255,255,255,.08); }
.pipe-arrow  { color:var(--border); font-size:.8rem; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.6} }

.stButton>button {
    background:linear-gradient(135deg,rgba(0,212,255,.1),rgba(124,58,237,.1)) !important;
    border:1px solid var(--accent) !important; color:var(--accent) !important;
    font-family:var(--mono) !important; font-size:.8rem !important;
    letter-spacing:.05em !important; border-radius:6px !important; transition:all .2s !important;
}
.stButton>button:hover { background:rgba(0,212,255,.2) !important; box-shadow:0 0 20px rgba(0,212,255,.2) !important; }

.stTextInput>div>div>input,.stTextArea textarea {
    background:var(--surface2) !important; border:1px solid var(--border) !important;
    color:var(--text) !important; border-radius:6px !important; font-family:var(--mono) !important;
}
hr { border-color:var(--border) !important; }
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:var(--surface); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
iframe { border:1px solid var(--border) !important; border-radius:8px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FIX #2 — MODULE-LEVEL CACHE (thread-safe, no st.session_state in threads)
# ─────────────────────────────────────────────
_PUBTATOR_CACHE: dict = {}

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
STAGE_KEYS = [
    "stage_input", "stage_initial_viz", "stage_pubtator",
    "stage_evidence", "stage_ot", "stage_merged",
    "stage_ranked", "stage_filtered", "stage_kg",
    "stage_faiss", "stage_rag_ready",
]
for _k in STAGE_KEYS:
    if _k not in st.session_state:
        st.session_state[_k] = False

for _k in ["input_df", "validated_df", "enriched_df", "filtered_df",
           "faiss_documents", "ranked_df", "llm_pairs",
           "faiss_model", "faiss_index"]:
    if _k not in st.session_state:
        st.session_state[_k] = None

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
PUBTATOR_BASE = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
OT_GQL        = "https://api.platform.opentargets.org/api/v4/graphql"
OLS_URL       = "https://www.ebi.ac.uk/ols/api/search"
ENSEMBL_URL   = "https://rest.ensembl.org/xrefs/symbol/homo_sapiens"
PIPELINE_STAGES = ["INPUT","VISUALIZE","PUBTATOR","EVIDENCE",
                   "OPEN TARGETS","MERGE","RANK","FILTER",
                   "KNOWLEDGE GRAPH","FAISS","RAG"]

# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────
def safe_get(url, params=None, timeout=12, max_retries=5):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            time.sleep(0.40)
            return r
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response else 0
            if code == 429 or 500 <= code < 600:
                time.sleep(2 ** attempt)
            else:
                raise
        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)
    raise requests.exceptions.RequestException(f"Max retries exceeded for {url}")


def render_header():
    st.markdown("""
    <div class="bio-header">
        <div class="bio-title">🧬 BioKnowledge Explorer</div>
        <div class="bio-subtitle">Gene–Disease Network · PubTator · Open Targets · RAG</div>
    </div>""", unsafe_allow_html=True)


def render_pipeline_bar():
    done = sum(st.session_state[k] for k in STAGE_KEYS)
    html = '<div class="pipeline-bar">'
    for i, s in enumerate(PIPELINE_STAGES):
        cls = "pipe-done" if i < done else ("pipe-active" if i == done else "pipe-idle")
        html += f'<span class="pipe-step {cls}">{s}</span>'
        if i < len(PIPELINE_STAGES)-1:
            html += '<span class="pipe-arrow">›</span>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def badge(text, kind="info"):
    return f'<span class="badge badge-{kind}">{text}</span>'


def metric_html(val, lbl):
    return f'<div class="metric-tile"><span class="metric-val">{val}</span><div class="metric-lbl">{lbl}</div></div>'


# ─────────────────────────────────────────────
# FIX #2 — PUBTATOR API (uses module-level _PUBTATOR_CACHE, NOT st.session_state)
# ─────────────────────────────────────────────
def find_entity_id(entity, bioconcept=None, limit=5):
    """Thread-safe: uses module-level dict, not st.session_state."""
    key = f"ent:{entity}:{bioconcept}:{limit}"
    if key in _PUBTATOR_CACHE:
        return _PUBTATOR_CACHE[key]
    params = {"query": entity, "limit": limit}
    if bioconcept:
        params["concept"] = bioconcept
    try:
        r = safe_get(f"{PUBTATOR_BASE}/entity/autocomplete/", params=params)
        data = r.json()
        result = data if isinstance(data, list) else []
        _PUBTATOR_CACHE[key] = result
        return result
    except Exception:
        _PUBTATOR_CACHE[key] = []
        return []


def find_related_entities(entity_id, relation_types=("treats", "associate"), limit=100):
    """Thread-safe: uses module-level dict."""
    url = f"{PUBTATOR_BASE}/relations"
    for rel_type in relation_types:
        key = f"rel:{entity_id}:{rel_type}"
        if key in _PUBTATOR_CACHE:
            cached = _PUBTATOR_CACHE[key]
            if cached:
                return cached
            continue
        params = {"e1": entity_id, "type": rel_type, "e2": "disease", "limit": limit}
        try:
            r = safe_get(url, params=params)
            j = r.json()
            data = j.get("relations", []) if isinstance(j, dict) else (j if isinstance(j, list) else [])
            result = {"relations": data, "relation_type": rel_type}
            _PUBTATOR_CACHE[key] = result
            if data:
                return result
        except Exception:
            _PUBTATOR_CACHE[key] = {"relations": [], "relation_type": None}
            continue
    return {"relations": [], "relation_type": None}


def search_pmids(entity_id1, entity_id2, relation_type, max_results=5):
    """Search PubTator for PMIDs supporting a relation. Matches PubTator_API.py logic."""
    query = f"relations:{relation_type.lower()}|{entity_id1}|{entity_id2}"
    url = f"{PUBTATOR_BASE}/search/?text={urllib.parse.quote(query, safe='')}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        j = r.json()
        results = j.get("results", []) if isinstance(j, dict) else (j if isinstance(j, list) else [])
        pmids = []
        for item in results[:max_results]:
            pmid = item.get("pmid")
            if pmid:
                pmids.append(str(pmid))
        return pmids
    except Exception:
        return []


def clean_relations(relations):
    bad = ["Neoplasms","Drug_Related_Side_Effects",
           "Chemical_and_Drug_Induced_Liver_Injury","Disease_Models"]
    return [r for r in relations
            if not any(re.search(p, r.get("entity_id",""), re.IGNORECASE) for p in bad)]


def validate_pair(entity_name, disease_name):
    """Validate one gene–disease pair. Safe for ThreadPoolExecutor."""
    result = {
        "entity": entity_name, "disease": disease_name,
        "entity_id": None, "disease_id": None,
        "relation_found": False, "relation_type": None, "pmids": None
    }
    try:
        ent_data = find_entity_id(entity_name)
        dis_data = find_entity_id(disease_name)
        if not ent_data or not dis_data:
            return result

        entity_id  = ent_data[0]["_id"]
        disease_id = dis_data[0]["_id"]
        result["entity_id"]  = entity_id
        result["disease_id"] = disease_id

        related = find_related_entities(entity_id)
        cleaned = clean_relations(related.get("relations", []))

        for rel in cleaned:
            # PubTator_API.py uses rel.get("source") as the matched entity id
            rel_id   = rel.get("source") or ""
            rel_name = (rel.get("target") or rel.get("entity_name") or "").lower()
            rel_type = (rel.get("type") or "associate").lower()
            target_lc = disease_name.lower()

            matched = (rel_id == disease_id
                       or target_lc in rel_name
                       or rel_name in target_lc)
            if matched and rel_id:
                raw_pmids = search_pmids(entity_id, rel_id, rel_type, max_results=3)
                valid_pmids = [p for p in raw_pmids if p and str(p).strip().isdigit()]
                result.update({
                    "relation_found": True,
                    "relation_type": rel_type,
                    "pmids": ";".join(valid_pmids) if valid_pmids else None
                })
                break
    except Exception as e:
        result["error"] = str(e)
    return result


def run_pubtator_parallel(df, max_workers=4):
    results = []
    rows = [row for _, row in df.iterrows()]
    prog = st.progress(0, text="Validating gene–disease pairs…")
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(validate_pair, r["entity"], r["disease"]): i
                   for i, r in enumerate(rows)}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                results.append({"error": str(e)})
            done += 1
            prog.progress(done / len(rows), text=f"Validated {done}/{len(rows)} pairs")
    prog.empty()
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# OPEN TARGETS / EFO / ENSEMBL
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_efo_id(disease_name):
    try:
        r = requests.get(OLS_URL, params={"q": disease_name, "ontology": "efo"}, timeout=10)
        r.raise_for_status()
        docs = r.json().get("response", {}).get("docs", [])
        if not docs:
            return None, None
        return docs[0]["obo_id"].replace(":", "_"), docs[0]["label"]
    except Exception:
        return None, None


@st.cache_data(ttl=3600, show_spinner=False)
def get_ensembl_id(gene_name):
    try:
        r = requests.get(f"{ENSEMBL_URL}/{gene_name}",
                         params={"content-type": "application/json"}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None, None
        return data[0]["id"], data[0].get("description", "")
    except Exception:
        return None, None


@st.cache_data(ttl=3600, show_spinner=False)
def ot_evidence(ensembl_id, efo_id):
    query = """
    query($eid:String!,$did:String!){
      disease(efoId:$did){
        evidences(ensemblIds:[$eid]){
          count
          rows{ datasourceId resourceScore }
        }
      }
    }"""
    try:
        r = requests.post(OT_GQL,
                          json={"query": query, "variables": {"eid": ensembl_id, "did": efo_id}},
                          headers={"Content-Type": "application/json"}, timeout=20)
        r.raise_for_status()
        ev = (r.json().get("data", {}).get("disease") or {}).get("evidences") or {}
        rows = ev.get("rows", [])
        if rows:
            count = ev.get("count", len(rows))
            avg   = sum((x.get("resourceScore") or 0) for x in rows) / len(rows)
            srcs  = ";".join(sorted({x.get("datasourceId","") for x in rows if x.get("datasourceId")}))
            return count, round(avg, 3), srcs
        return 0, None, None
    except Exception:
        return None, None, None


def enrich_with_ot(validated_df):
    rows, total = [], len(validated_df)
    prog = st.progress(0, text="Fetching Open Targets data…")
    for idx, (_, row) in enumerate(validated_df.iterrows()):
        r = row.to_dict()
        ensembl_id, _  = get_ensembl_id(r["entity"])
        efo_id, efo_lbl = get_efo_id(r["disease"])
        r["ensembl_id"] = ensembl_id
        r["efo_id"]     = efo_id
        r["efo_label"]  = efo_lbl
        if ensembl_id and efo_id:
            cnt, avg, src = ot_evidence(ensembl_id, efo_id)
        else:
            cnt, avg, src = None, None, None
        r["ot_evidence_count"] = cnt
        r["ot_avg_score"]      = avg
        r["ot_sources"]        = src
        rows.append(r)
        prog.progress((idx+1)/total, text=f"OT enrichment {idx+1}/{total}")
        time.sleep(0.2)
    prog.empty()
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# FIX #3 — RANKING: mirrors network_module.py filter logic
# Strong: count>70 AND avg_score>15  → composite = avg_score * count (normalized)
# Medium: count>15 AND 10<=avg_score<=15
# Otherwise: lower composite
# ─────────────────────────────────────────────
def compute_composite_rank(df):
    df = df.copy()
    df["ot_evidence_count"] = pd.to_numeric(df.get("ot_evidence_count"), errors="coerce").fillna(0)
    df["ot_avg_score"]      = pd.to_numeric(df.get("ot_avg_score"),      errors="coerce").fillna(0)

    def score_row(row):
        cnt = row["ot_evidence_count"]
        avg = row["ot_avg_score"]
        rel = 1 if row.get("relation_found") else 0
        pmid_bonus = 1 if (pd.notna(row.get("pmids")) and str(row.get("pmids","")).strip()) else 0

        # Mirror network_module.py association categories for weighting
        if cnt > 70 and avg > 15:             # Strong
            score = avg * 1.5 + cnt * 0.05 + rel * 10 + pmid_bonus * 5
        elif cnt > 15 and 10 <= avg <= 15:    # Medium
            score = avg * 1.0 + cnt * 0.02 + rel * 8  + pmid_bonus * 3
        else:                                 # Weak / no relation
            score = avg * 0.5 + cnt * 0.005 + rel * 3 + pmid_bonus * 1
        return round(score, 3)

    df["composite_rank"] = df.apply(score_row, axis=1)
    return df.sort_values("composite_rank", ascending=False).reset_index(drop=True)


def apply_filter(df):
    """Mirrors network_module.filter_network exactly."""
    df = df.copy()
    df["ot_evidence_count"] = pd.to_numeric(df.get("ot_evidence_count"), errors="coerce").fillna(0)
    df["ot_avg_score"]      = pd.to_numeric(df.get("ot_avg_score"),      errors="coerce").fillna(0)

    strong = df[(df["ot_evidence_count"] > 70) & (df["ot_avg_score"] > 15)].copy()
    strong["association_type"] = "Strong"

    medium = df[
        (df["ot_evidence_count"] > 15) &
        (df["ot_avg_score"] >= 10) &
        (df["ot_avg_score"] <= 15)
    ].copy()
    medium["association_type"] = "Medium"

    filtered = pd.concat([strong, medium]).drop_duplicates(subset=["entity","disease"])
    if "composite_rank" in filtered.columns:
        filtered = filtered.sort_values("composite_rank", ascending=False)
    return filtered.reset_index(drop=True)


# ─────────────────────────────────────────────
# FIX #1 — NETWORK GRAPH: fixed pixel dimensions, not window.innerWidth
# ─────────────────────────────────────────────
def build_network_html(df, title="Gene–Disease Network", width=1100, height=580):
    import math
    nodes_genes    = df["entity"].dropna().unique().tolist()
    nodes_diseases = df["disease"].dropna().unique().tolist()
    all_nodes = list(dict.fromkeys(nodes_genes + nodes_diseases))
    n = len(all_nodes)
    if n == 0:
        return ("<html><body style='background:#0a0f1e;color:#64748b;"
                "font-family:monospace;padding:2rem'>No data to visualize.</body></html>")

    CX, CY = width // 2, height // 2

    pos = {}
    for i, nd in enumerate(all_nodes):
        angle = 2 * math.pi * i / n
        r = min(CX, CY) * 0.75 if nd in nodes_genes else min(CX, CY) * 0.45
        pos[nd] = (CX + r * math.cos(angle), CY + r * math.sin(angle))

    edges = []
    for _, row in df.iterrows():
        g, d = row.get("entity",""), row.get("disease","")
        if not g or not d:
            continue
        w        = float(row.get("composite_rank", 1) or 1)
        pmids_s  = str(row.get("pmids","") or "")[:100]
        tooltip  = (f"{g} → {d} | Relation: {row.get('relation_type','N/A')} | "
                    f"OT Score: {row.get('ot_avg_score','N/A')} | "
                    f"Count: {row.get('ot_evidence_count','N/A')} | PMIDs: {pmids_s}")
        assoc    = str(row.get("association_type",""))
        edges.append({"from": g, "to": d, "weight": w, "tooltip": tooltip, "assoc": assoc})

    max_w    = max((e["weight"] for e in edges), default=1)
    nodes_js = json.dumps([{"id": nd, "x": pos[nd][0], "y": pos[nd][1],
                             "type": "gene" if nd in nodes_genes else "disease"}
                            for nd in all_nodes])
    edges_js = json.dumps(edges)

    # ── CRITICAL FIX: hardcode canvas size, no window.innerWidth ──
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0f1e;font-family:'IBM Plex Mono',monospace;overflow:hidden;width:{width}px;height:{height}px}}
canvas{{display:block;position:absolute;top:0;left:0}}
#tip{{position:absolute;pointer-events:none;background:rgba(10,15,30,.96);
  border:1px solid #1e3a5f;border-radius:6px;padding:8px 12px;font-size:11px;
  color:#e2e8f0;max-width:320px;line-height:1.6;display:none;z-index:100;word-break:break-word}}
#ttl{{position:absolute;top:10px;left:50%;transform:translateX(-50%);
  color:#00d4ff;font-size:12px;letter-spacing:.1em;white-space:nowrap;z-index:10}}
#leg{{position:absolute;bottom:10px;left:10px;font-size:10px;color:#64748b;z-index:10}}
.leg{{display:flex;align-items:center;gap:6px;margin:2px 0}}
.dot{{width:9px;height:9px;border-radius:50%;display:inline-block;flex-shrink:0}}
</style></head>
<body>
<canvas id="c" width="{width}" height="{height}"></canvas>
<div id="tip"></div>
<div id="ttl">{title}</div>
<div id="leg">
  <div class="leg"><div class="dot" style="background:#00d4ff"></div>Gene</div>
  <div class="leg"><div class="dot" style="background:#7c3aed"></div>Disease</div>
  <div class="leg"><div class="dot" style="background:#f59e0b"></div>Strong</div>
  <div class="leg"><div class="dot" style="background:#10b981"></div>Medium</div>
  <div class="leg"><div class="dot" style="background:#475569"></div>Other</div>
</div>
<script>
const canvas=document.getElementById('c');
const ctx=canvas.getContext('2d');
const tip=document.getElementById('tip');

const nodes={nodes_js};
const edges={edges_js};
const mw={max_w};
const nm={{}};
nodes.forEach(n=>nm[n.id]=n);

function draw(){{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  // edges
  edges.forEach(e=>{{
    const a=nm[e.from],b=nm[e.to];
    if(!a||!b)return;
    const alpha=0.25+0.6*(e.weight/mw);
    const lw=0.8+2.5*(e.weight/mw);
    const col=e.assoc==='Strong' ? `rgba(245,158,11,${{alpha}})`
             :e.assoc==='Medium' ? `rgba(16,185,129,${{alpha}})`
             :`rgba(71,85,105,${{alpha}})`;
    ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);
    ctx.strokeStyle=col;ctx.lineWidth=lw;ctx.stroke();
  }});
  // nodes
  nodes.forEach(n=>{{
    const isG=n.type==='gene', r=isG?11:9;
    // glow
    const grd=ctx.createRadialGradient(n.x,n.y,0,n.x,n.y,r*2.5);
    grd.addColorStop(0,isG?'rgba(0,212,255,.15)':'rgba(124,58,237,.15)');
    grd.addColorStop(1,'rgba(0,0,0,0)');
    ctx.beginPath();ctx.arc(n.x,n.y,r*2.5,0,Math.PI*2);
    ctx.fillStyle=grd;ctx.fill();
    // circle
    ctx.beginPath();ctx.arc(n.x,n.y,r,0,Math.PI*2);
    ctx.fillStyle=isG?'#00d4ff':'#7c3aed';ctx.fill();
    ctx.strokeStyle=isG?'rgba(0,212,255,.5)':'rgba(124,58,237,.5)';
    ctx.lineWidth=2;ctx.stroke();
    // label
    ctx.fillStyle='#e2e8f0';ctx.font='bold 10px IBM Plex Mono';
    ctx.textAlign='center';
    ctx.fillText(n.id.length>15?n.id.slice(0,15)+'…':n.id,n.x,n.y+r+13);
  }});
}}
draw();

// tooltip on edge hover
canvas.addEventListener('mousemove',ev=>{{
  const rect=canvas.getBoundingClientRect();
  const mx=ev.clientX-rect.left,my=ev.clientY-rect.top;
  let found=null;
  // check node hover first
  for(const nd of nodes){{
    if(Math.hypot(mx-nd.x,my-nd.y)<14){{found={{tooltip:nd.id+' ('+nd.type+')'}};break;}}
  }}
  // then edge hover
  if(!found){{
    for(const edge of edges){{
      const a=nm[edge.from],b=nm[edge.to];
      if(!a||!b)continue;
      const dx=b.x-a.x,dy=b.y-a.y;
      const t=Math.max(0,Math.min(1,((mx-a.x)*dx+(my-a.y)*dy)/(dx*dx+dy*dy)));
      if(Math.hypot(mx-(a.x+t*dx),my-(a.y+t*dy))<9){{found=edge;break;}}
    }}
  }}
  if(found){{
    tip.style.display='block';
    const tx=mx+14,ty=Math.max(10,my-20);
    tip.style.left=Math.min(tx,canvas.width-330)+'px';
    tip.style.top=ty+'px';
    tip.textContent=found.tooltip;
  }}else tip.style.display='none';
}});
canvas.addEventListener('mouseleave',()=>tip.style.display='none');

// drag
let drag=null,ox=0,oy=0;
canvas.addEventListener('mousedown',ev=>{{
  const rect=canvas.getBoundingClientRect();
  const mx=ev.clientX-rect.left,my=ev.clientY-rect.top;
  for(const n of nodes){{if(Math.hypot(mx-n.x,my-n.y)<14){{drag=n;ox=mx-n.x;oy=my-n.y;break;}}}}
}});
canvas.addEventListener('mousemove',ev=>{{
  if(!drag)return;
  const rect=canvas.getBoundingClientRect();
  drag.x=ev.clientX-rect.left-ox;drag.y=ev.clientY-rect.top-oy;draw();
}});
canvas.addEventListener('mouseup',()=>drag=null);
</script></body></html>"""


# ─────────────────────────────────────────────
# PUBMED ABSTRACT FETCH
# ─────────────────────────────────────────────
def fetch_abstract(pmid, email="user@example.com"):
    try:
        from Bio import Entrez
        Entrez.email = email
        handle  = Entrez.efetch(db="pubmed", id=str(pmid), retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        art = records["PubmedArticle"][0]["MedlineCitation"]["Article"]
        ab  = art.get("Abstract", {})
        if "AbstractText" in ab:
            return " ".join(str(t) for t in ab["AbstractText"])
        return None
    except Exception:
        return None


def fetch_pubmed_details(pmid, email="user@example.com"):
    """Return title + abstract dict for a PMID."""
    try:
        from Bio import Entrez
        Entrez.email = email
        handle  = Entrez.efetch(db="pubmed", id=str(pmid), retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        art     = records["PubmedArticle"][0]["MedlineCitation"]["Article"]
        title   = str(art.get("ArticleTitle","N/A"))
        ab      = art.get("Abstract",{})
        abstract = " ".join(str(t) for t in ab.get("AbstractText",[])) if "AbstractText" in ab else "No abstract."
        journal = str(art.get("Journal",{}).get("Title","N/A"))
        # year
        pub_date = art.get("Journal",{}).get("JournalIssue",{}).get("PubDate",{})
        year = pub_date.get("Year","") or pub_date.get("MedlineDate","")[:4]
        return {"pmid": pmid, "title": title, "abstract": abstract,
                "journal": journal, "year": year}
    except Exception as e:
        return {"pmid": pmid, "title": "Error", "abstract": str(e), "journal":"","year":""}


def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def extract_valid_pmids(df):
    pmids = set()
    for val in df["pmids"].dropna():
        for p in str(val).split(";"):
            p = p.strip()
            if p and p.isdigit():
                pmids.add(p)
    return sorted(pmids)


def build_faiss_index_fn(documents):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [d["text"] for d in documents]
    if not texts:
        return None, None, []
    embeddings = model.encode(texts, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))
    return model, index, documents


def faiss_search(query, top_k=5):
    """Search FAISS index if available. Returns list of doc dicts."""
    model = st.session_state.get("faiss_model")
    index = st.session_state.get("faiss_index")
    docs  = st.session_state.get("faiss_documents") or []
    if model is None or index is None or not docs:
        return []
    q_emb = model.encode([query])
    D, I  = index.search(np.array(q_emb, dtype=np.float32), top_k)
    return [docs[i] for i in I[0] if i < len(docs)]


# ─────────────────────────────────────────────
# RAG — mirrors rag_module.py
# ─────────────────────────────────────────────
def build_prompt(query, retrieved_docs):
    """Mirrors rag_module.py build_prompt exactly."""
    context = ""
    for doc in retrieved_docs:
        context += f"""
        PMID: {doc.get('pmid','?')}
        Gene: {doc.get('gene','?')}
        Disease: {doc.get('disease','?')}
        Relation Type: {doc.get('relation_type','?')}
        Score: {doc.get('ot_score','?')}
        Text: {doc.get('text','')}
        ---
        """
    return f"""
    You are a biomedical reasoning assistant.

    Use the context primarily but you may reason moderately.
    Cite PMIDs when possible.

    Context:
    {context}

    Question:
    {query}

    Provide:
    - Evidence summary
    - Mechanistic insight
    - Strength of association
    - Confidence
    - Supporting PMIDs
    """


def build_prompt_llm_only(query, retrieved_docs=None):
    """For LLM-only tab: if FAISS docs exist, include them; else pure LLM reasoning."""
    if retrieved_docs:
        context = ""
        for doc in retrieved_docs:
            context += f"""
        PMID: {doc.get('pmid','?')}
        Gene: {doc.get('gene','?')}  Disease: {doc.get('disease','?')}
        Relation: {doc.get('relation_type','?')}  Score: {doc.get('ot_score','?')}
        Text: {doc.get('text','')}
        ---"""
        return f"""You are a senior biomedical research assistant with deep expertise.
You have access to retrieved evidence from a curated gene-disease knowledge base.

Retrieved Context:
{context}

Question: {query}

Using BOTH your training knowledge AND the retrieved context above, provide:
1. Evidence Summary (cite PMIDs from context where possible)
2. Mechanistic Insight
3. Known Pathway / Biological Process
4. Strength of Evidence
5. Confidence (High/Medium/Low)
6. Suggested follow-up research directions
"""
    else:
        return f"""You are a senior biomedical research assistant with deep expertise
in genetics, pharmacology, and molecular biology.

Question: {query}

Please provide a comprehensive answer including:
1. Biomedical Background
2. Known Gene–Disease Associations (cite key studies if known)
3. Mechanistic Insight
4. Strength of Evidence in Literature
5. Confidence Level
6. Suggested follow-up research
"""


def ask_gemini_rest(prompt, api_key):
    try:
        url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
        body = {"contents": [{"parts": [{"text": prompt}]}]}
        r    = requests.post(url, json=body, timeout=60)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Gemini error: {e}"


def ask_gemini(prompt, api_key):
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro")
        return model.generate_content(prompt).text
    except ImportError:
        return ask_gemini_rest(prompt, api_key)
    except Exception:
        return ask_gemini_rest(prompt, api_key)


# ─────────────────────────────────────────────
# LLM PAIR GENERATION
# ─────────────────────────────────────────────
def llm_generate_pairs(query, api_key):
    prompt = f"""Given the biomedical query: "{query}"
Generate a list of gene–disease associations that are scientifically well-established.
Return ONLY a JSON array (no markdown, no explanation):
[{{"entity":"GENE1","disease":"Disease1"}},{{"entity":"GENE2","disease":"Disease2"}}]
Include 8-15 pairs. Only valid JSON."""
    raw = ask_gemini(prompt, api_key)
    try:
        cleaned = re.sub(r"```json|```","",raw).strip()
        data = json.loads(cleaned)
        if isinstance(data, list) and data:
            return pd.DataFrame(data)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:IBM Plex Mono;color:#00d4ff;font-size:.9rem;letter-spacing:.1em;margin-bottom:1rem">⚙ CONFIGURATION</div>', unsafe_allow_html=True)
    st.markdown("**LLM**")
    gemini_key = st.text_input("Gemini API Key", type="password", key="gem_key")
    st.markdown("**PubMed**")
    ncbi_email = st.text_input("NCBI Email", value="user@example.com", key="ncbi_mail")
    st.markdown("**Processing**")
    max_workers = st.slider("Parallel workers", 1, 8, 4)

    st.divider()
    st.markdown('<div style="font-family:IBM Plex Mono;color:#64748b;font-size:.7rem;margin-bottom:.5rem">PIPELINE STATUS</div>', unsafe_allow_html=True)
    _labels = ["INPUT","INITIAL VIZ","PUBTATOR","EVIDENCE",
               "OPEN TARGETS","MERGED","RANKED","FILTERED",
               "KNOWLEDGE GRAPH","FAISS","RAG READY"]
    for _k, _lbl in zip(STAGE_KEYS, _labels):
        _ok = st.session_state[_k]
        _col = "#10b981" if _ok else "#1e3a5f"
        st.markdown(f'<div style="font-family:IBM Plex Mono;font-size:.7rem;color:{_col};padding:1px 0">{"●" if _ok else "○"} {_lbl}</div>', unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Reset Pipeline"):
        for _k in STAGE_KEYS:
            st.session_state[_k] = False
        for _k in ["input_df","validated_df","enriched_df","filtered_df",
                   "faiss_documents","ranked_df","llm_pairs","faiss_model","faiss_index"]:
            st.session_state[_k] = None
        _PUBTATOR_CACHE.clear()
        st.rerun()


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
render_header()
render_pipeline_bar()

# ──────────────────────────────────────────────
# STAGE 1 — INPUT
# ──────────────────────────────────────────────
with st.expander("📂 STAGE 1 — Input", expanded=not st.session_state.stage_input):
    st.markdown('<div class="stage-label">Data Ingestion</div>', unsafe_allow_html=True)
    input_mode = st.radio("Input mode", ["Upload CSV","Natural Language Query"], horizontal=True)

    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV (columns: entity, disease)", type="csv")
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                df_up.columns = [c.lower().strip() for c in df_up.columns]
                if "entity" not in df_up.columns or "disease" not in df_up.columns:
                    st.error("CSV must contain 'entity' and 'disease' columns.")
                else:
                    df_up = df_up[["entity","disease"]].dropna().drop_duplicates()
                    st.session_state.input_df = df_up
                    st.success(f"Loaded {len(df_up)} gene–disease pairs")
                    st.dataframe(df_up, use_container_width=True)
                    if st.button("✅ Confirm Input & Proceed"):
                        st.session_state.stage_input = True
                        st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        nlq = st.text_area("Enter biomedical query",
                           placeholder="What genes are associated with Type 2 Diabetes?")
        if st.button("🔍 Generate Pairs via Gemini"):
            if not gemini_key:
                st.warning("Enter Gemini API Key in sidebar.")
            elif nlq.strip():
                with st.spinner("Generating pairs…"):
                    df_gen = llm_generate_pairs(nlq.strip(), gemini_key)
                if df_gen is not None and not df_gen.empty:
                    df_gen.columns = [c.lower().strip() for c in df_gen.columns]
                    st.session_state.input_df  = df_gen
                    st.session_state.llm_pairs = nlq
                    st.success(f"Generated {len(df_gen)} pairs")
                    st.dataframe(df_gen, use_container_width=True)
                    st.session_state.stage_input = True
                    st.rerun()
                else:
                    st.error("LLM returned invalid data. Check API key.")
            else:
                st.warning("Please enter a query.")

# ──────────────────────────────────────────────
# STAGE 2 — INITIAL VISUALIZATION
# ──────────────────────────────────────────────
if st.session_state.stage_input and st.session_state.input_df is not None:
    with st.expander("🕸 STAGE 2 — Initial Network Visualization",
                     expanded=not st.session_state.stage_initial_viz):
        df_in = st.session_state.input_df.copy()
        df_in["composite_rank"]    = 1.0
        df_in["relation_type"]     = "unknown"
        df_in["association_type"]  = ""
        df_in["ot_avg_score"]      = None
        df_in["ot_evidence_count"] = None
        df_in["pmids"]             = None

        st.markdown('<div class="stage-label">Raw Gene–Disease Network (unvalidated)</div>', unsafe_allow_html=True)
        # FIX #1: height=600 for the iframe to match canvas
        st.components.v1.html(build_network_html(df_in, "Raw Input Network"), height=620, scrolling=False)

        genes    = df_in["entity"].nunique()
        diseases = df_in["disease"].nunique()
        st.markdown(f'<div class="metric-row">{metric_html(genes,"Unique Genes")}{metric_html(diseases,"Unique Diseases")}{metric_html(len(df_in),"Total Pairs")}</div>', unsafe_allow_html=True)

        if st.button("▶ Proceed to PubTator Validation"):
            st.session_state.stage_initial_viz = True
            st.rerun()

# ──────────────────────────────────────────────
# STAGE 3 — PUBTATOR VALIDATION
# ──────────────────────────────────────────────
if st.session_state.stage_initial_viz:
    with st.expander("🔬 STAGE 3 — PubTator Validation",
                     expanded=not st.session_state.stage_pubtator):
        if not st.session_state.stage_pubtator:
            st.info("Queries PubTator3 API for each pair. May take several minutes for large datasets.")
            if st.button("🚀 Run PubTator Validation"):
                df_in = st.session_state.input_df.copy()
                with st.spinner("Running parallel PubTator validation…"):
                    validated = run_pubtator_parallel(df_in, max_workers=max_workers)
                st.session_state.validated_df    = validated
                st.session_state.stage_pubtator = True
                st.rerun()
        else:
            validated = st.session_state.validated_df
            if validated is not None and "relation_found" in validated.columns:
                found = int(validated["relation_found"].sum())
                total = len(validated)
                st.markdown(f'<div class="metric-row">{metric_html(total,"Total")}{metric_html(found,"Relations Found")}{metric_html(total-found,"Not Found")}</div>', unsafe_allow_html=True)
            st.dataframe(validated, use_container_width=True)
            st.markdown(badge("COMPLETE","success"), unsafe_allow_html=True)

# ──────────────────────────────────────────────
# STAGE 4 — EVIDENCE EXTRACTION
# ──────────────────────────────────────────────
if st.session_state.stage_pubtator and st.session_state.validated_df is not None:
    with st.expander("📚 STAGE 4 — PMID Evidence Extraction",
                     expanded=not st.session_state.stage_evidence):
        validated = st.session_state.validated_df.copy()
        if "pmids" not in validated.columns:
            validated["pmids"] = None

        def _clean_pmids(val):
            if pd.isna(val) or not str(val).strip():
                return None
            parts = [p.strip() for p in str(val).split(";") if p.strip().isdigit()]
            return ";".join(parts) if parts else None

        validated["pmids"] = validated["pmids"].apply(_clean_pmids)
        st.session_state.validated_df = validated

        pmid_list = extract_valid_pmids(validated)
        with_pmids = validated["pmids"].notna().sum()
        st.markdown(f'<div class="metric-row">{metric_html(len(pmid_list),"Unique Valid PMIDs")}{metric_html(with_pmids,"Pairs w/ PMIDs")}</div>', unsafe_allow_html=True)
        if pmid_list:
            st.code(", ".join(pmid_list[:30]))
        with st.expander("📋 PMID per pair"):
            pmid_view = validated[validated["pmids"].notna()][["entity","disease","relation_type","pmids"]]
            st.dataframe(pmid_view, use_container_width=True)
        if st.button("▶ Continue to Open Targets"):
            st.session_state.stage_evidence = True
            st.rerun()

# ──────────────────────────────────────────────
# STAGE 5 — OPEN TARGETS ENRICHMENT
# ──────────────────────────────────────────────
if st.session_state.stage_evidence:
    with st.expander("🎯 STAGE 5 — Open Targets Enrichment",
                     expanded=not st.session_state.stage_ot):
        if not st.session_state.stage_ot:
            st.info("Maps genes → Ensembl ID, diseases → EFO ID, fetches OT evidence.")
            if st.button("🚀 Run Open Targets Enrichment"):
                with st.spinner("Enriching…"):
                    enriched = enrich_with_ot(st.session_state.validated_df.copy())
                st.session_state.enriched_df = enriched
                st.session_state.stage_ot = True
                st.rerun()
        else:
            enriched = st.session_state.enriched_df
            if enriched is not None:
                st.markdown(f'<div class="metric-row">{metric_html(enriched["ensembl_id"].notna().sum(),"Ensembl Mapped")}{metric_html(enriched["efo_id"].notna().sum(),"EFO Mapped")}</div>', unsafe_allow_html=True)
                show = ["entity","disease","ensembl_id","efo_id","ot_evidence_count","ot_avg_score","ot_sources"]
                st.dataframe(enriched[[c for c in show if c in enriched.columns]], use_container_width=True)
            st.markdown(badge("COMPLETE","success"), unsafe_allow_html=True)

# ──────────────────────────────────────────────
# STAGE 6 — DATA INTEGRATION
# ──────────────────────────────────────────────
if st.session_state.stage_ot and st.session_state.enriched_df is not None:
    with st.expander("🔗 STAGE 6 — Data Integration",
                     expanded=not st.session_state.stage_merged):
        enriched = st.session_state.enriched_df.copy()
        for col in ["entity","disease","pmids","relation_type","ot_evidence_count","ot_avg_score"]:
            if col not in enriched.columns:
                enriched[col] = None
        st.dataframe(enriched, use_container_width=True)
        st.download_button("⬇ Merged CSV", enriched.to_csv(index=False).encode(),
                           file_name="merged_data.csv", mime="text/csv")
        if st.button("▶ Compute Rankings"):
            st.session_state.enriched_df = enriched
            st.session_state.stage_merged = True
            st.rerun()

# ──────────────────────────────────────────────
# STAGE 7 — RANKING (FIX #3: mirrors network_module.py)
# ──────────────────────────────────────────────
if st.session_state.stage_merged:
    with st.expander("📊 STAGE 7 — Hybrid Ranking",
                     expanded=not st.session_state.stage_ranked):
        enriched = st.session_state.enriched_df.copy()
        ranked   = compute_composite_rank(enriched)
        st.session_state.ranked_df = ranked
        st.markdown("""**Composite Rank (mirrors network_module.py thresholds):**
- **Strong** (count>70, score>15): `score×1.5 + count×0.05 + relation×10 + pmid_bonus×5`
- **Medium** (count>15, score 10–15): `score×1.0 + count×0.02 + relation×8 + pmid_bonus×3`
- **Other**: `score×0.5 + count×0.005 + relation×3 + pmid_bonus×1`
""")
        show = ["entity","disease","relation_found","ot_evidence_count","ot_avg_score","composite_rank"]
        st.dataframe(ranked[[c for c in show if c in ranked.columns]].head(25), use_container_width=True)
        if st.button("▶ Apply Filtering"):
            st.session_state.stage_ranked = True
            st.rerun()

# ──────────────────────────────────────────────
# STAGE 8 — FILTERING
# ──────────────────────────────────────────────
if st.session_state.stage_ranked and st.session_state.ranked_df is not None:
    with st.expander("🧹 STAGE 8 — Filter Strong / Medium Associations",
                     expanded=not st.session_state.stage_filtered):
        ranked   = st.session_state.ranked_df.copy()
        filtered = apply_filter(ranked)
        if filtered.empty:
            st.warning("No associations passed thresholds. Showing top-10 by composite rank.")
            filtered = ranked.head(10).copy()
            filtered["association_type"] = "Below Threshold"

        strong_n = int((filtered.get("association_type","") == "Strong").sum()) if "association_type" in filtered else 0
        medium_n = int((filtered.get("association_type","") == "Medium").sum()) if "association_type" in filtered else 0
        st.markdown(f'<div class="metric-row">{metric_html(strong_n,"Strong")}{metric_html(medium_n,"Medium")}{metric_html(len(filtered),"Total Kept")}</div>', unsafe_allow_html=True)
        st.dataframe(filtered, use_container_width=True)
        st.session_state.filtered_df = filtered
        if st.button("▶ Build Knowledge Graph"):
            st.session_state.stage_filtered = True
            st.rerun()

# ──────────────────────────────────────────────
# STAGE 9 — KNOWLEDGE GRAPH (FIX #1)
# ──────────────────────────────────────────────
if st.session_state.stage_filtered and st.session_state.filtered_df is not None:
    with st.expander("🌐 STAGE 9 — Interactive Knowledge Graph",
                     expanded=not st.session_state.stage_kg):
        filtered = st.session_state.filtered_df.copy()
        st.markdown('<div class="stage-label">Filtered Gene–Disease Knowledge Graph</div>', unsafe_allow_html=True)
        # height=620 matches the 580px canvas + some chrome
        st.components.v1.html(build_network_html(filtered, "Filtered Knowledge Graph"), height=620, scrolling=False)
        st.caption("🖱 Drag nodes · Hover edges or nodes for details")
        if st.button("▶ Build FAISS Index"):
            st.session_state.stage_kg = True
            st.rerun()

# ──────────────────────────────────────────────
# STAGE 10 — FAISS INDEX
# ──────────────────────────────────────────────
if st.session_state.stage_kg and st.session_state.filtered_df is not None:
    with st.expander("🗄 STAGE 10 — FAISS Vector Index",
                     expanded=not st.session_state.stage_faiss):
        filtered  = st.session_state.filtered_df.copy()
        pmids_all = extract_valid_pmids(filtered)

        if not pmids_all:
            st.warning("No valid PMIDs in filtered data — cannot build FAISS index.")
        else:
            st.markdown(f"Found **{len(pmids_all)}** unique PMIDs to index.")

        if not st.session_state.stage_faiss and pmids_all:
            if st.button("🚀 Fetch Abstracts & Build FAISS Index"):
                pmid_meta = {}
                for _, row in filtered.iterrows():
                    if row.get("pmids"):
                        for p in str(row["pmids"]).split(";"):
                            p = p.strip()
                            if p and p.isdigit():
                                pmid_meta.setdefault(p, {
                                    "gene": row.get("entity",""),
                                    "disease": row.get("disease",""),
                                    "relation_type": row.get("relation_type",""),
                                    "ot_score": row.get("ot_avg_score","")
                                })

                documents = []
                prog = st.progress(0, text="Fetching abstracts from PubMed…")
                pmid_list_f = list(pmid_meta.keys())
                for idx, pmid in enumerate(pmid_list_f):
                    abstract = fetch_abstract(pmid, email=ncbi_email)
                    if abstract:
                        meta = pmid_meta[pmid]
                        for chunk in chunk_text(abstract):
                            documents.append({
                                "text": chunk, "pmid": pmid,
                                "gene": meta["gene"], "disease": meta["disease"],
                                "relation_type": meta["relation_type"],
                                "ot_score": meta["ot_score"]
                            })
                    prog.progress((idx+1)/len(pmid_list_f),
                                  text=f"Fetched {idx+1}/{len(pmid_list_f)} abstracts")
                    time.sleep(0.3)
                prog.empty()

                if not documents:
                    st.error("No abstracts retrieved. Check NCBI email and PMIDs.")
                else:
                    with st.spinner("Encoding & building FAISS index…"):
                        model, index, docs = build_faiss_index_fn(documents)
                    st.session_state["faiss_model"]    = model
                    st.session_state["faiss_index"]    = index
                    st.session_state.faiss_documents   = docs
                    st.session_state.stage_faiss       = True
                    st.success(f"✅ FAISS index: {len(docs)} chunks from {len(pmid_list_f)} PMIDs")
                    st.rerun()

        if st.session_state.stage_faiss:
            docs = st.session_state.faiss_documents or []
            st.markdown(f'<div class="metric-row">{metric_html(len(docs),"Indexed Chunks")}{metric_html(len({d["pmid"] for d in docs}),"PMIDs Indexed")}</div>', unsafe_allow_html=True)
            st.markdown(badge("INDEX READY — RAG ENABLED","success"), unsafe_allow_html=True)
            if st.button("▶ Enable RAG Interface"):
                st.session_state.stage_rag_ready = True
                st.rerun()

# ──────────────────────────────────────────────
# STAGE 11 — RAG (two tabs: pipeline RAG + LLM-only with optional FAISS)
# ──────────────────────────────────────────────
if st.session_state.stage_rag_ready:
    with st.expander("🤖 STAGE 11 — Biomedical Query Interface", expanded=True):
        if not gemini_key:
            st.warning("⚠ Enter Gemini API Key in sidebar.")
        else:
            rag_tab, llm_tab = st.tabs([
                "🔬 Pipeline RAG (FAISS-grounded)",
                "🧠 LLM Search (+ optional FAISS)"
            ])

            # ── Tab A: Pipeline RAG ──
            with rag_tab:
                st.markdown('<div class="stage-label">Evidence-Grounded Biomedical Reasoning</div>', unsafe_allow_html=True)
                st.caption("Answers grounded strictly in your FAISS-indexed PubMed abstracts.")

                rag_query = st.text_area("Ask about your gene–disease network:",
                                         placeholder="Explain the mechanism between CYP2C19 and Diabetes Mellitus",
                                         height=80, key="rag_q")
                top_k_rag = st.slider("Context chunks (top-k)", 3, 10, 5, key="rag_k")

                if st.button("🔍 Query Network", key="rag_btn"):
                    if not rag_query.strip():
                        st.warning("Please enter a question.")
                    else:
                        retrieved = faiss_search(rag_query, top_k=top_k_rag)
                        if not retrieved:
                            st.error("FAISS index empty or unavailable. Rebuild in Stage 10.")
                        else:
                            with st.spinner("Generating response via Gemini…"):
                                answer = ask_gemini(build_prompt(rag_query, retrieved), gemini_key)
                            st.markdown("### 📋 Analysis")
                            st.markdown(
                                f'<div style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;'
                                f'padding:1.5rem;white-space:pre-wrap;font-family:IBM Plex Sans,sans-serif;'
                                f'font-size:.9rem;line-height:1.75;color:#e2e8f0">{answer}</div>',
                                unsafe_allow_html=True)
                            with st.expander("📚 Retrieved Context Chunks"):
                                for chunk_doc in retrieved:
                                    c1, c2 = st.columns([1,3])
                                    with c1:
                                        st.markdown(
                                            f'{badge("PMID "+str(chunk_doc["pmid"]),"info")}<br>'
                                            f'{badge(str(chunk_doc["gene"])+" → "+str(chunk_doc["disease"]),"warning")}',
                                            unsafe_allow_html=True)
                                    with c2:
                                        st.caption(chunk_doc["text"][:500]+"…")
                                    st.divider()

            # ── Tab B: LLM Search with optional FAISS ──
            with llm_tab:
                st.markdown('<div class="stage-label">LLM Biomedical Search</div>', unsafe_allow_html=True)
                faiss_ready = st.session_state.stage_faiss and bool(st.session_state.faiss_documents)
                if faiss_ready:
                    st.success("✅ FAISS index available — LLM will also retrieve relevant context from your pipeline data.")
                else:
                    st.info("ℹ FAISS index not built yet — LLM will answer from general knowledge only.")

                use_faiss = st.checkbox("Use FAISS context if available", value=faiss_ready,
                                        disabled=not faiss_ready, key="llm_faiss_toggle")
                llm_query = st.text_area("Ask any biomedical question:",
                                          placeholder="What is the role of NFE2L2 in oxidative stress diseases?",
                                          height=80, key="llm_q")
                top_k_llm = st.slider("FAISS context chunks", 3, 10, 5, key="llm_k",
                                       disabled=not (faiss_ready and use_faiss))

                if st.button("🧠 Ask LLM", key="llm_btn"):
                    if not llm_query.strip():
                        st.warning("Please enter a question.")
                    else:
                        retrieved_llm = []
                        if use_faiss and faiss_ready:
                            with st.spinner("Retrieving FAISS context…"):
                                retrieved_llm = faiss_search(llm_query, top_k=top_k_llm)

                        prompt = build_prompt_llm_only(llm_query, retrieved_llm if retrieved_llm else None)
                        with st.spinner("Querying Gemini…"):
                            answer_llm = ask_gemini(prompt, gemini_key)

                        st.markdown("### 💡 LLM Response")
                        st.markdown(
                            f'<div style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;'
                            f'padding:1.5rem;white-space:pre-wrap;font-family:IBM Plex Sans,sans-serif;'
                            f'font-size:.9rem;line-height:1.75;color:#e2e8f0">{answer_llm}</div>',
                            unsafe_allow_html=True)

                        if retrieved_llm:
                            with st.expander("📚 FAISS Context Used"):
                                for chunk_doc in retrieved_llm:
                                    c1, c2 = st.columns([1,3])
                                    with c1:
                                        st.markdown(
                                            f'{badge("PMID "+str(chunk_doc["pmid"]),"info")}<br>'
                                            f'{badge(str(chunk_doc["gene"])+" → "+str(chunk_doc["disease"]),"warning")}',
                                            unsafe_allow_html=True)
                                    with c2:
                                        st.caption(chunk_doc["text"][:500]+"…")
                                    st.divider()

# ──────────────────────────────────────────────
# DASHBOARD TOOLS
# ──────────────────────────────────────────────
if st.session_state.stage_filtered and st.session_state.filtered_df is not None:
    st.markdown("---")
    st.markdown('<div style="font-family:IBM Plex Mono;color:#00d4ff;letter-spacing:.1em;margin:1rem 0 .5rem;font-size:.9rem">⚡ DASHBOARD TOOLS</div>', unsafe_allow_html=True)

    tab_lookup, tab_fetch, tab_curate, tab_export = st.tabs([
        "🔎 PMID Lookup (in-pipeline)",
        "🌐 PMID Fetch (live PubMed)",
        "✏️ Curation Table",
        "⬇ Export"
    ])

    # ── Tab 1: PMID lookup within pipeline results ──
    with tab_lookup:
        st.markdown("**Search whether a PMID appears in your current pipeline results.**")
        pmid_q = st.text_input("PMID to search", placeholder="e.g. 37920809", key="pmid_lookup")
        if pmid_q.strip():
            hits = pd.DataFrame()
            for src_df in [st.session_state.filtered_df, st.session_state.validated_df]:
                if src_df is not None and "pmids" in src_df.columns:
                    h = src_df[src_df["pmids"].fillna("").str.contains(pmid_q.strip())]
                    if not h.empty:
                        hits = h
                        break
            if not hits.empty:
                st.success(f"Found PMID {pmid_q} in {len(hits)} row(s)")
                st.dataframe(hits, use_container_width=True)
            else:
                st.info(f"PMID {pmid_q} not found in pipeline results.")

    # ── Tab 2 (NEW FIX #4): Live PMID fetch from PubMed ──
    with tab_fetch:
        st.markdown("**Fetch full details for any PMID directly from PubMed.**")
        st.caption("Enter one or more PMIDs (comma-separated). Does not require them to be in your pipeline.")
        fetch_input = st.text_input("Enter PMID(s)", placeholder="e.g. 37920809, 33738827", key="pmid_fetch_input")
        if st.button("🔎 Fetch from PubMed", key="pmid_fetch_btn"):
            raw_ids = [p.strip() for p in fetch_input.split(",") if p.strip().isdigit()]
            if not raw_ids:
                st.warning("Please enter valid numeric PMID(s).")
            else:
                results_fetch = []
                prog_f = st.progress(0, text="Fetching from PubMed…")
                for i, pmid_f in enumerate(raw_ids):
                    details = fetch_pubmed_details(pmid_f, email=ncbi_email)
                    results_fetch.append(details)
                    prog_f.progress((i+1)/len(raw_ids))
                    time.sleep(0.3)
                prog_f.empty()
                for det in results_fetch:
                    st.markdown(
                        f'<div style="background:#111827;border:1px solid #1e3a5f;border-radius:8px;padding:1.2rem;margin-bottom:.8rem">'
                        f'<div style="font-family:IBM Plex Mono;color:#00d4ff;font-size:.75rem;margin-bottom:.3rem">PMID {det["pmid"]} · {det["journal"]} · {det["year"]}</div>'
                        f'<div style="font-weight:600;color:#e2e8f0;margin-bottom:.5rem">{det["title"]}</div>'
                        f'<div style="color:#94a3b8;font-size:.85rem;line-height:1.6">{det["abstract"][:800]}{"…" if len(det["abstract"])>800 else ""}</div>'
                        f'</div>',
                        unsafe_allow_html=True)
                    # also check if PMID is in pipeline
                    in_pipe = False
                    for src_df in [st.session_state.filtered_df, st.session_state.validated_df]:
                        if src_df is not None and "pmids" in src_df.columns:
                            if src_df["pmids"].fillna("").str.contains(det["pmid"]).any():
                                in_pipe = True
                                break
                    tag = badge("IN PIPELINE","success") if in_pipe else badge("NOT IN PIPELINE","warning")
                    st.markdown(tag, unsafe_allow_html=True)

    # ── Tab 3: Curation ──
    with tab_curate:
        st.markdown("Edit table, then click **Recompute Ranking**.")
        fdf = st.session_state.filtered_df.copy()
        ecols = [c for c in ["entity","disease","relation_found","relation_type",
                              "ot_evidence_count","ot_avg_score","pmids"] if c in fdf.columns]
        edited = st.data_editor(fdf[ecols], use_container_width=True,
                                num_rows="dynamic", key="curator_table")
        if st.button("♻ Recompute Ranking"):
            for col in ecols:
                if col in edited.columns:
                    fdf[col] = edited[col].values
            fdf["pmids"] = fdf["pmids"].apply(
                lambda v: ";".join(p.strip() for p in str(v).split(";") if str(p).strip().isdigit())
                if pd.notna(v) else None)
            reranked = compute_composite_rank(fdf)
            st.session_state.filtered_df = reranked
            st.success("✅ Ranking recomputed.")
            st.dataframe(reranked[["entity","disease","composite_rank","ot_evidence_count","ot_avg_score"]],
                         use_container_width=True)

    # ── Tab 4: Export ──
    with tab_export:
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.session_state.ranked_df is not None:
                st.download_button("⬇ Full Ranked",
                                   st.session_state.ranked_df.to_csv(index=False).encode(),
                                   file_name="ranked_results.csv", mime="text/csv")
        with c2:
            if st.session_state.filtered_df is not None:
                st.download_button("⬇ Filtered",
                                   st.session_state.filtered_df.to_csv(index=False).encode(),
                                   file_name="filtered_results.csv", mime="text/csv")
        with c3:
            if st.session_state.enriched_df is not None:
                st.download_button("⬇ Full Enriched",
                                   st.session_state.enriched_df.to_csv(index=False).encode(),
                                   file_name="enriched_full.csv", mime="text/csv")

# ── Footer ──
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;font-family:IBM Plex Mono;
font-size:.65rem;color:#1e3a5f;letter-spacing:.1em">
BIOKNOWLEDGE EXPLORER · PubTator3 · Open Targets · FAISS · Gemini 1.5 Pro
</div>""", unsafe_allow_html=True)
