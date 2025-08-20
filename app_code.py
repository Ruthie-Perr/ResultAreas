# app.py — Result Areas Generator (Retriever + LLM)
# ---------------------------------------------------------------------
# NOTE: ensure requirements.txt includes:
# streamlit
# langchain-openai
# langchain-community
# chromadb
# pysqlite3-binary
# pandas
# python-dotenv (optional)
# ---------------------------------------------------------------------

# --- SQLite shim for Streamlit Cloud (Chromadb needs sqlite >= 3.35) ---
import sys
try:
    import pysqlite3  # ships modern sqlite
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import os, re, json
from typing import List, Dict

import streamlit as st
import pandas as pd

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PERSIST_DIR     = "chroma_db"            # folder you committed to GitHub
COLLECTION_NAME = "kb_result_areas"      # same as in build_kb.py
EMBED_MODEL     = "text-embedding-3-small"
GEN_MODEL       = "gpt-4o-mini"          # or "gpt-4.1-mini" / "gpt-4o"

# Streamlit secrets → env for langchain_openai
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Result Areas Generator", layout="wide")
st.title("Role → Result Areas (Generator)")
st.caption("Beschrijf de functie. Ik haal voorbeelden op en genereer thema’s & resultaatgebieden inclusief AEM‑Cube band-scores (0–100) en buckets.")

# ─────────────────────────────────────────────────────────────────────────────
# Load vectorstore (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_vs() -> Chroma:
    emb = OpenAIEmbeddings(model=EMBED_MODEL)
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=emb,
        collection_name=COLLECTION_NAME,
    )

try:
    vs = load_vs()
except Exception as e:
    st.error(
        "Failed to load Chroma collection. Ensure the folder chroma_db exists in this repo and collection name matches.\n\n"
        f"Error: {e}"
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval helper (new Chroma 0.5+ filter syntax)
# ─────────────────────────────────────────────────────────────────────────────
def retrieve_examples(
    vs: Chroma,
    query_text: str,
    k: int = 8,
    app_scope: str = "result_areas",
    content_type: str = "example",
) -> List[Dict]:
    filt = {
        "$and": [
            {"app_scope": {"$eq": app_scope}},
            {"content_type": {"$eq": content_type}},
        ]
    }
    retriever = vs.as_retriever(search_kwargs={"k": k, "filter": filt})
    docs = retriever.get_relevant_documents(query_text)

    # Deduplicate by (theme, result_area)
    seen, out = set(), []
    for d in docs:
        md = d.metadata or {}
        theme = md.get("theme", "")
        rarea = md.get("result_area", "")
        key = (theme, rarea)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "function_title": md.get("function_title", ""),
            "theme": theme,
            "result_area": rarea,
            "band_attachment": md.get("band_attachment", ""),
            "band_exploration": md.get("band_exploration", ""),
            "band_managing_complexity": md.get("band_managing_complexity", ""),
            "source": md.get("source", ""),
            "snippet": d.page_content,  # useful context for few-shot
        })
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Prompt pieces (theory & rules)
# ─────────────────────────────────────────────────────────────────────────────
THEORY = """AEM-Cube theory:
- Attachment: security via people (relationships) vs content (systems/ideas). Strategic: product–customer vs process–employee.
- Exploration: innovate vs optimise. Strategic: speed-to-market. Growth-Curve: early stages ↔ exploratory; later ↔ optimising.
- Managing Complexity: specialists (deep) vs generalists (broad). Strategic: business agility.
"""

HOW_TO_RA_NL = """Hoe definieer je resultaatgebieden:
- 3–6 per functie; essentieel, niet “nice-to-have”; door één individu uitvoerbaar.
- Beschrijf een proces met begin en eind; gebruik werkwoorden die iets opleveren/creëren (“opleveren”, “neerzetten”, “creëren”).
- Het resultaat moet concreet en begrijpelijk zijn (“loyale klantenbasis”, “degelijke sales pijplijn”, “efficiënte productketen”, “betrouwbare bedrijfscultuur”).
- Benoem de reden/het waarom van elk resultaatgebied.
Voorbeelden:
- Slecht: “rekeningen versturen” (activiteit).
- Goed: “transparante rekeningen leveren teneinde een tevreden klantenbasis te bouwen” (resultaatgebied).
- Slecht: “afspraken nakomen” (activiteit).
- Goed: “een betrouwbare bedrijfscultuur creëren zodat iedereen op elkaar kan bouwen” (resultaatgebied).
"""

# ─────────────────────────────────────────────────────────────────────────────
# LLM utilities
# ─────────────────────────────────────────────────────────────────────────────
def bucket_for(score: int) -> str:
    s = max(0, min(100, int(score)))
    if s <= 25: return "0-25"
    if s <= 50: return "25-50"
    if s <= 75: return "50-75"
    return "75-100"

def build_system_msg() -> str:
    return f"""You are an HR/Org design assistant. Use AEM‑Cube theory and the writing rules below to propose themes and resultaatgebieden for a given function. 
Return 3–6 resultaatgebieden. Each result area must be outcome‑oriented, concrete, and include a short 'why' rationale.
Also estimate AEM‑Cube band widths (Attachment, Exploration, Managing Complexity) that best fit each result area.
Prefer Dutch if language='nl'.

AEM‑Cube band rules:
- Each dimension (Attachment, Exploration, Managing Complexity) has a score range of 0–100.
- In addition to a numeric score (0–100), assign a band *bucket* exactly from this set: ["0-25","25-50","50-75","75-100"].
- Choose buckets that reflect the numeric score. Example: score 62 → bucket "50-75".
- If uncertainty is high, you may choose a mid bucket but still provide a concrete score.

THEORY
{THEORY}

WRITING RULES (NL)
{HOW_TO_RA_NL}

OUTPUT FORMAT (both):
1) Markdown for humans (in Dutch if language='nl').
2) A strict JSON object:
{{
  "themes": [
    {{
      "theme": "<string>",
      "result_areas": [
        {{
          "title": "<kort resultaatgebied>",
          "why": "<korte rationale>",
          "bands": {{
            "attachment": {{ "score": <0-100 integer>, "bucket": "0-25|25-50|50-75|75-100" }},
            "exploration": {{ "score": <0-100 integer>, "bucket": "0-25|25-50|50-75|75-100" }},
            "managing_complexity": {{ "score": <0-100 integer>, "bucket": "0-25|25-50|50-75|75-100" }}
          }}
        }}
      ]
    }}
  ]
}}
Validation rules:
- All scores must be integers between 0 and 100.
- Buckets must be exactly one of: "0-25", "25-50", "50-75", "75-100".
- The chosen bucket must match the score (0–25 → "0-25"; 26–50 → "25-50"; 51–75 → "50-75"; 76–100 → "75-100").
Only produce grounded, sensible bands; if uncertain, prefer middle buckets with scores near the center of the bucket.
"""

def build_examples_block(examples: List[Dict]) -> str:
    """Compact block that we pass to the LLM; we also display this to the user."""
    if not examples:
        return "(none found)"
    lines = []
    for ex in examples[:8]:
        line = (
            f"- Thema: {ex.get('theme','')} | "
            f"Resultaatgebied: {ex.get('result_area','')} | "
            f"A:{ex.get('band_attachment','')} "
            f"E:{ex.get('band_exploration','')} "
            f"M:{ex.get('band_managing_complexity','')}"
        )
        # You can add source or function title if you want:
        if ex.get("function_title"):
            line += f" | Functie: {ex['function_title']}"
        if ex.get("source"):
            line += f" | Bron: {ex['source']}"
        lines.append(line)
    return "Relevant examples (use as inspiration, adapt if relevant):\n" + "\n".join(lines)

def generate_result_areas(role_text: str, examples: List[Dict], language: str = "nl") -> Dict:
    examples_block = build_examples_block(examples)

    system_msg = build_system_msg()
    user_msg = f"""Language: {language}

User function / role context:
\"\"\"{role_text.strip()}\"\"\"

{examples_block}

Task:
- Base your proposal on BOTH the user role AND the retrieved examples above.
- Propose suitable themes and 3–6 resultaatgebieden for this function.
- For each result area include: a short why + A/E/M band scores (0–100) AND the correct bucket.
- Keep it concise and concrete, in Dutch."""

    llm = ChatOpenAI(model=GEN_MODEL, temperature=0.2)
    resp = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ])

    text = resp.content

    # Try to extract trailing JSON block if present
    json_obj = None
    json_match = re.search(r"\{[\s\S]*\}\s*$", text.strip())
    if json_match:
        try:
            json_obj = json.loads(json_match.group(0))
        except Exception:
            json_obj = None

    # Post-parse guard: snap buckets to score if needed
    if isinstance(json_obj, dict):
        for th in json_obj.get("themes", []):
            for ra in th.get("result_areas", []):
                for dim in ["attachment", "exploration", "managing_complexity"]:
                    b = ra.get("bands", {}).get(dim, {})
                    try:
                        sc = int(b.get("score", 0))
                    except Exception:
                        sc = 0
                    sc = max(0, min(100, sc))
                    b["score"] = sc
                    b["bucket"] = bucket_for(sc)

    return {
        "markdown": text,                  # full text from model (usually includes markdown + json)
        "json": json_obj,                  # parsed JSON (sanitized)
        "prompt": system_msg + "\n\n---\n\n" + user_msg,
        "examples_block": examples_block,  # EXACT examples text sent to model
    }

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
with st.form("ra_form"):
    col1, col2 = st.columns([3,1])
    with col1:
        role_text = st.text_area(
            "Beschrijf de functie (taken, scope, senioriteit, context):",
            height=220,
            placeholder="Bijv. Accountmanager B2B SaaS: new business & upsell, pipeline bouwen, demo's, onderhandelingen, samenwerking met marketing & CS, NL/EN markt, mid-market…"
        )
    with col2:
        k = st.number_input("Aantal voorbeelden (k)", min_value=2, max_value=20, value=8)
        language = st.selectbox("Taal", ["nl", "en"], index=0)
        show_prompt = st.checkbox("Toon prompt/debug", value=False)
        show_examples_block = st.checkbox("Toon voorbeelden die naar het model zijn gestuurd", value=True)
    submitted = st.form_submit_button("Genereer resultaatgebieden")

if submitted:
    if not role_text.strip():
        st.warning("Vul eerst de functie/rol in.")
        st.stop()

    with st.spinner("Retrieving voorbeelden en genereren…"):
        examples = retrieve_examples(vs, role_text, k=int(k))
        result = generate_result_areas(role_text, examples, language=language)

    st.markdown("### Resultaat")
    st.write(result["markdown"])

    if result["json"] is not None:
        st.download_button(
            "Download JSON",
            data=json.dumps(result["json"], ensure_ascii=False, indent=2),
            file_name="result_areas.json",
            mime="application/json",
        )

    # Show the retrieved examples as a table (what *was* retrieved)
    st.markdown("### Opgehaalde voorbeelden (tabel)")
    if examples:
        ex_df = pd.DataFrame(examples)[[
            "theme","result_area",
            "band_attachment","band_exploration","band_managing_complexity",
            "source","function_title"
        ]]
        st.dataframe(ex_df, use_container_width=True, hide_index=True)
    else:
        st.info("Geen voorbeelden gevonden voor deze query.")

    # Show the EXACT examples block sent to the model (what the model *saw*)
    if show_examples_block:
        with st.expander("📎 Voorbeelden naar model (exacte tekst)"):
            st.code(result["examples_block"], language="markdown")

    # Full prompt/debug
    if show_prompt:
        with st.expander("🛠 Prompt (debug)"):
            st.code(result["prompt"], language="markdown")
