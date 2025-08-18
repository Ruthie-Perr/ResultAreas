# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import pdfplumber
from typing import List, Dict
from rank_bm25 import BM25Okapi
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG (expects OPENAI_API_KEY and MODEL_ID in .streamlit/secrets.toml)
# ─────────────────────────────────────────────────────────────────────────────
EMBED_MODEL = "text-embedding-3-small"  # or text-embedding-3-large
GEN_MODEL = st.secrets.get("MODEL_ID", "gpt-4o-mini")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="RAG: Roles → Result Areas", layout="wide")
st.title("RAG for Role → Result Areas (Hybrid Search)")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def normalize(s: str) -> str:
    s = (s or "").strip()
    return re.sub(r"\s+", " ", s)

def row_to_example(row: pd.Series) -> Dict:
    """
    Map Excel columns → KB example.
    Adjust the column names below to match your sheet.
    """
    title = normalize(row.get("function_title", ""))
    desc  = normalize(row.get("function_description", ""))
    exp   = normalize(row.get("experience", ""))
    goal  = normalize(row.get("function_goal", ""))
    themes = [t.strip() for t in str(row.get("themes", "")).split(",") if t.strip()]

    # Expected: a JSON string cell with your curated outputs (edit if stored differently)
    ra_json = str(row.get("result_areas_json", "[]"))
    try:
        result_areas = json.loads(ra_json)
    except Exception:
        result_areas = []

    text = " ".join([
        f"Title: {title}",
        f"Description: {desc}",
        f"Experience: {exp}",
        f"Goal: {goal}",
        f"Themes: {', '.join(themes)}",
    ])

    return {
        "id": f"ex_{hash(title+desc+exp+goal) & 0xffffffff:x}",
        "title": title,
        "description": desc,
        "experience": exp,
        "goal": goal,
        "themes": themes,
        "result_areas": result_areas,
        "language": "en",
        "text": text
    }

def build_kb_from_excel(xlsx_bytes) -> List[Dict]:
    df = pd.read_excel(xlsx_bytes)
    kb = [row_to_example(row) for _, row in df.iterrows()]
    return kb

def build_hybrid_index(kb: List[Dict]):
    texts = [ex["text"] for ex in kb]
    tokens = [re.findall(r"\w+", t.lower()) for t in texts]
    bm25 = BM25Okapi(tokens)

    emb = client.embeddings.create(model=EMBED_MODEL, input=texts).data
    E = np.vstack([np.array(e.embedding, dtype=np.float32) for e in emb])  # (N, d)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)            # normalize for cosine
    return bm25, En, texts  # keep texts for debugging

def _grab_after(label: str, text: str):
    pat = re.compile(rf"(?im)^{re.escape(label)}\s*:\s*(.+)$")
    m = pat.search(text)
    return m.group(1).strip() if m else ""

def parse_job_pdf(file_like) -> dict:
    full = []
    with pdfplumber.open(file_like) as pdf:
        for p in pdf.pages:
            t = p.extract_text() or ""
            full.append(t)
    text = "\n".join(full)

    out = {
        "title": _grab_after("Function Title", text) or _grab_after("Functietitel", text),
        "description": _grab_after("Function Description", text) or _grab_after("Functieomschrijving", text),
        "experience": _grab_after("Experience", text) or _grab_after("Ervaring", text),
        "goal": _grab_after("Function Goal", text) or _grab_after("Doel van de functie", text),
        "themes": [s.strip() for s in (_grab_after("Themes", text) or _grab_after("Thema's", text)).split(",") if s.strip()],
        "language": "en",
    }
    out["query_text"] = " ".join([
        f"Title: {out['title']}",
        f"Description: {out['description']}",
        f"Experience: {out['experience']}",
        f"Goal: {out['goal']}",
        f"Themes: {', '.join(out['themes'])}",
    ]).strip()
    return out

def rrf_fuse(rank_lists, k=60):
    from collections import defaultdict
    scores = defaultdict(float)
    for ranks in rank_lists:
        for r, idx in enumerate(ranks):
            scores[idx] += 1.0 / (k + r + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

def hybrid_search(query_text: str, bm25: BM25Okapi, En: np.ndarray, kb: List[Dict], top_k=6):
    toks = re.findall(r"\w+", query_text.lower())
    bm_scores = bm25.get_scores(toks)
    bm_rank = np.argsort(-bm_scores).tolist()

    q = client.embeddings.create(model=EMBED_MODEL, input=[query_text]).data[0].embedding
    q = np.array(q, dtype=np.float32)
    qn = q / (np.linalg.norm(q) + 1e-8)
    cos = En @ qn
    sem_rank = np.argsort(-cos).tolist()

    fused = rrf_fuse([bm_rank, sem_rank])
    top = fused[:top_k]
    return [kb[i] for i in top], {"bm25_top": bm_rank[:top_k], "semantic_top": sem_rank[:top_k], "fused_top": top}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "result_areas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "target_bands": {
                        "type": "object",
                        "properties": {
                            "attachment": {"type": "string"},
                            "exploration": {"type": "string"},
                            "managing_complexity": {"type": "string"}
                        },
                        "required": ["attachment", "exploration", "managing_complexity"]
                    },
                    "rationale": {"type": "string"}
                },
                "required": ["name", "target_bands"]
            }
        }
    },
    "required": ["result_areas"]
}

def compose_prompt(job, retrieved_examples):
    examples_txt = []
    for ex in retrieved_examples:
        examples_txt.append(
            "- Example:\n"
            f"  Title: {ex['title']}\n"
            f"  Description: {ex['description']}\n"
            f"  Experience: {ex['experience']}\n"
            f"  Goal: {ex['goal']}\n"
            f"  Themes: {', '.join(ex['themes'])}\n"
            f"  Desired Output (excerpt): {json.dumps(ex['result_areas'][:2], ensure_ascii=False)}"
        )
    examples_block = "\n".join(examples_txt)

    job_block = (
        f"New Job:\n"
        f"  Title: {job['title']}\n"
        f"  Description: {job['description']}\n"
        f"  Experience: {job['experience']}\n"
        f"  Goal: {job['goal']}\n"
        f"  Themes: {', '.join(job['themes'])}\n"
    )

    instructions = (
        "You are an AEM-Cube specialist. Based on the new job and the retrieved examples, "
        "propose 4–6 result areas. For each, provide target AEM-Cube bandwidths "
        "(attachment, exploration, managing_complexity) as numeric ranges like '45-60'. "
        "Follow house style: concise names, one-sentence rationale. "
        "Return ONLY valid JSON matching this schema:\n"
        f"{json.dumps(OUTPUT_SCHEMA)}"
    )

    return f"{instructions}\n\n{job_block}\nRetrieved examples:\n{examples_block}"

def generate(job, retrieved_examples, model=GEN_MODEL):
    prompt = compose_prompt(job, retrieved_examples)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role":"system","content":"You are a precise, compliant JSON generator."},
            {"role":"user","content":prompt}
        ]
    )
    text = resp.choices[0].message.content.strip()
    # Attempt to parse/repair JSON if needed
    try:
        return json.loads(text)
    except Exception:
        stripped = text.strip().strip("`")
        return json.loads(stripped)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – KB upload and indexing
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Step 1 — Load Examples (Excel)")

    source = st.radio(
        "Select source for examples:",
        options=["GitHub URL", "Upload file"],
        index=0,
        help="Use a raw GitHub URL for public files, or set GITHUB_TOKEN in secrets for private repos."
    )

    github_url = st.text_input(
        "GitHub Raw URL (or regular file URL)",
        value="",
        placeholder="https://raw.githubusercontent.com/<org>/<repo>/<branch>/path/to/examples.xlsx",
    )

    ex_file = None
    if source == "Upload file":
        ex_file = st.file_uploader("Resultaatgebieden Excel", type=["xlsx", "xls"])

    # Optional: read token from secrets for private repos
    gh_token = st.secrets.get("GITHUB_TOKEN", None)

    @st.cache_data(show_spinner=False)
    def load_excel_from_github(url: str, token: str | None = None) -> pd.DataFrame:
        """
        Downloads an Excel file from GitHub and returns a DataFrame.
        Works with raw URLs; if you pass a normal GitHub URL, it tries to rewrite it to raw.
        """
        if not url:
            raise ValueError("No GitHub URL provided.")

        # Convert standard GitHub URL to raw if needed
        # e.g., https://github.com/org/repo/blob/branch/path/file.xlsx → https://raw.githubusercontent.com/org/repo/branch/path/file.xlsx
        raw_url = url.strip()
        if "github.com" in raw_url and "/blob/" in raw_url and "raw.githubusercontent.com" not in raw_url:
            raw_url = raw_url.replace("github.com/", "raw.githubusercontent.com/").replace("/blob/", "/")

        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
            headers["Accept"] = "application/vnd.github.v3.raw"

        resp = requests.get(raw_url, headers=headers, timeout=30)
        if not resp.ok:
            raise RuntimeError(f"GitHub download failed ({resp.status_code}): {resp.text[:200]}")

        # Quick sanity: avoid HTML error pages
        ctype = resp.headers.get("Content-Type", "").lower()
        if "text/html" in ctype and not raw_url.endswith((".xlsx", ".xls")):
            raise RuntimeError("Got HTML instead of an Excel file. Make sure you use the RAW file URL.")

        return pd.read_excel(io.BytesIO(resp.content))

    def build_kb_from_df(df: pd.DataFrame) -> list[dict]:
        # Reuse your existing row_to_example mapping
        return [row_to_example(row) for _, row in df.iterrows()]

    build_btn = st.button(
        "Build Knowledge Base & Index",
        use_container_width=True,
        disabled=(source == "GitHub URL" and not github_url) and (source == "Upload file" and ex_file is None)
    )

    if "kb" not in st.session_state:
        st.session_state.kb = None
        st.session_state.bm25 = None
        st.session_state.En = None
        st.session_state.texts = None

    if build_btn:
        try:
            if source == "GitHub URL":
                df = load_excel_from_github(github_url, token=gh_token)
            else:
                df = pd.read_excel(ex_file)

            kb = build_kb_from_df(df)
            bm25, En, texts = build_hybrid_index(kb)

            st.session_state.kb = kb
            st.session_state.bm25 = bm25
            st.session_state.En = En
            st.session_state.texts = texts

            st.success(f"Built KB with {len(kb)} examples from {'GitHub' if source=='GitHub URL' else 'uploaded file'}.")
        except Exception as e:
            st.error(f"Failed to build KB/index: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Main – PDF upload and generation
# ─────────────────────────────────────────────────────────────────────────────
st.header("Step 2 — Upload Job PDF")
job_pdf = st.file_uploader("Job PDF (title, description, experience, goal, themes)", type=["pdf"], key="jobpdf")

colA, colB = st.columns(2)
with colA:
    run = st.button("Retrieve & Generate", type="primary", disabled=job_pdf is None or st.session_state.kb is None)
with colB:
    top_k = st.number_input("Top-K examples", min_value=3, max_value=10, value=6, step=1)

if run:
    try:
        # Parse the job PDF to structured fields
        job = parse_job_pdf(job_pdf)

        # Hybrid retrieval
        retrieved, debug = hybrid_search(
            job["query_text"],
            st.session_state.bm25,
            st.session_state.En,
            st.session_state.kb,
            top_k=int(top_k)
        )

        # Show retrieval debug
        with st.expander("Retrieved examples (debug)"):
            st.write("BM25 top indices:", debug["bm25_top"])
            st.write("Semantic top indices:", debug["semantic_top"])
            st.write("Fused top indices:", debug["fused_top"])
            st.write(pd.DataFrame([{"rank": i+1, "title": ex["title"], "themes": ", ".join(ex["themes"])} for i, ex in enumerate(retrieved)]))

        # Generate final JSON
        output = generate(job, retrieved, model=GEN_MODEL)

        st.subheader("Generated Result Areas (JSON)")
        st.json(output)

    except Exception as e:
        st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.caption("Hybrid RAG: BM25 + embeddings (RRF). Upload Excel once → upload job PDF → retrieve → generate.")
