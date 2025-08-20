# app.py — Result Areas Generator (Retriever + LLM)

# --- SQLite shim for Streamlit Cloud (Chromadb needs sqlite >= 3.35) ---
import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import os, re, json
from typing import List, Dict

import streamlit as st
import pandas as pd

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

# ── CONFIG ─────────────────────────────────────────────────────────────
PERSIST_DIR     = "chroma_db"
COLLECTION_NAME = "kb_result_areas"
EMBED_MODEL     = "text-embedding-3-small"
GEN_MODEL       = "gpt-4o-mini"

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Result Areas Generator", layout="wide")
st.title("Role → Result Areas (Generator)")
st.caption("Voer functietitel en -omschrijving in. Ik haal voorbeelden op en genereer thema’s & resultaatgebieden inclusief AEM-Cube band-scores (0–100).")

# ── Load vectorstore ─────────────────────────────────────────────────
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
    st.error(f"Failed to load Chroma collection. Error: {e}")
    st.stop()

# ── Retrieval helper ─────────────────────────────────────────────────
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
            "snippet": d.page_content,
        })
    return out

# ── Prompt pieces ────────────────────────────────────────────────────
THEORY = """AEM-Cube theory:
- Attachment: security via people vs content.
- Exploration: innovate vs optimise (growth-curve: early=explore, late=optimise).
- Managing Complexity: specialists vs generalists.
"""

HOW_TO_RA_NL = """Hoe definieer je resultaatgebieden:
- 3–6 per functie; essentieel, niet “nice-to-have”.
- Proces met begin en eind, werkwoorden die iets opleveren.
- Concreet en begrijpelijk resultaat, met waarom erbij.
"""

def bucket_for(score: int) -> str:
    s = max(0, min(100, int(score)))
    if s <= 25: return "0-25"
    if s <= 50: return "25-50"
    if s <= 75: return "50-75"
    return "75-100"

def build_system_msg() -> str:
    return f"""You are an HR/Org design assistant. Use AEM-Cube theory and the writing rules below to propose themes and resultaatgebieden.
Return 3–6 resultaatgebieden. Each result area must be outcome-oriented, concrete, and include a short 'why' rationale.
Also estimate AEM-Cube band widths (Attachment, Exploration, Managing Complexity) that best fit each result area.
Prefer Dutch if language='nl'.

AEM-Cube band rules:
- Each dimension has a score 0–100.
- Provide both a numeric score and a bucket ["0-25","25-50","50-75","75-100"].
- Bucket must match the score.

THEORY
{THEORY}

WRITING RULES (NL)
{HOW_TO_RA_NL}

OUTPUT FORMAT:
1) Markdown for humans (in Dutch if language='nl').
2) JSON with structure:
{{
  "themes": [
    {{
      "theme": "<string>",
      "result_areas": [
        {{
          "title": "<kort resultaatgebied>",
          "why": "<rationale>",
          "bands": {{
            "attachment": {{ "score": <0-100>, "bucket": "…" }},
            "exploration": {{ "score": <0-100>, "bucket": "…" }},
            "managing_complexity": {{ "score": <0-100>, "bucket": "…" }}
          }}
        }}
      ]
    }}
  ]
}}
"""

def build_examples_block(examples: List[Dict]) -> str:
    if not examples: return "(none found)"
    lines = []
    for ex in examples[:8]:
        line = (
            f"- Thema: {ex.get('theme','')} | "
            f"Resultaatgebied: {ex.get('result_area','')} | "
            f"A:{ex.get('band_attachment','')} "
            f"E:{ex.get('band_exploration','')} "
            f"M:{ex.get('band_managing_complexity','')}"
        )
        if ex.get("function_title"):
            line += f" | Functie: {ex['function_title']}"
        lines.append(line)
    return "Relevant examples (use as inspiration):\n" + "\n".join(lines)

def generate_result_areas(role_title: str, role_desc: str, examples: List[Dict], language: str = "nl") -> Dict:
    examples_block = build_examples_block(examples)
    role_text = f"Functietitel: {role_title}\nOmschrijving: {role_desc}"

    system_msg = build_system_msg()
    user_msg = f"""Language: {language}

User function / role context:
\"\"\"{role_text.strip()}\"\"\"

{examples_block}

Task:
- Base your proposal on BOTH the user role and the retrieved examples.
- Propose 3–6 resultaatgebieden.
- Include: short why + A/E/M score (0–100) and correct bucket.
- Output in Dutch (if language='nl')."""

    llm = ChatOpenAI(model=GEN_MODEL, temperature=0.2)
    resp = llm.invoke([{"role": "system", "content": system_msg},
                       {"role": "user", "content": user_msg}])

    text = resp.content

    json_obj = None
    json_match = re.search(r"\{[\s\S]*\}\s*$", text.strip())
    if json_match:
        try:
            json_obj = json.loads(json_match.group(0))
        except Exception:
            pass

    if isinstance(json_obj, dict):
        for th in json_obj.get("themes", []):
            for ra in th.get("result_areas", []):
                for dim in ["attachment","exploration","managing_complexity"]:
                    b = ra.get("bands", {}).get(dim, {})
                    try: sc = int(b.get("score", 0))
                    except: sc = 0
                    sc = max(0, min(100, sc))
                    b["score"] = sc
                    b["bucket"] = bucket_for(sc)

    return {
        "markdown": text,
        "json": json_obj,
        "prompt": system_msg + "\n\n---\n\n" + user_msg,
        "examples_block": examples_block,
    }

# ── UI with two fields (title + description) ─────────────────────────
with st.form("ra_form"):
    role_title = st.text_input("Functietitel", placeholder="Bijv. Accountmanager B2B SaaS")
    role_desc = st.text_area("Functieomschrijving", height=180,
                             placeholder="Beschrijf de scope, taken, verantwoordelijkheden…")
    col1, col2 = st.columns([1,1])
    with col1:
        k = st.number_input("Aantal voorbeelden (k)", min_value=2, max_value=20, value=8)
    with col2:
        language = st.selectbox("Taal", ["nl", "en"], index=0)
    show_prompt = st.checkbox("Toon prompt/debug", value=False)
    show_examples_block = st.checkbox("Toon voorbeelden die naar het model zijn gestuurd", value=True)
    submitted = st.form_submit_button("Genereer resultaatgebieden")

if submitted:
    if not role_title.strip() and not role_desc.strip():
        st.warning("Vul functietitel en/of omschrijving in.")
        st.stop()

    with st.spinner("Retrieving voorbeelden en genereren…"):
        query_text = f"{role_title} {role_desc}"
        examples = retrieve_examples(vs, query_text, k=int(k))
        result = generate_result_areas(role_title, role_desc, examples, language=language)

    st.markdown("### Resultaat")
    st.write(result["markdown"])

    if result["json"] is not None:
        st.download_button(
            "Download JSON",
            data=json.dumps(result["json"], ensure_ascii=False, indent=2),
            file_name="result_areas.json",
            mime="application/json",
        )

    st.markdown("### Opgehaalde voorbeelden (tabel)")
    if examples:
        ex_df = pd.DataFrame(examples)[[
            "theme","result_area",
            "band_attachment","band_exploration","band_managing_complexity",
            "source","function_title"
        ]]
        st.dataframe(ex_df, use_container_width=True, hide_index=True)
    else:
        st.info("Geen voorbeelden gevonden.")

    if show_examples_block:
        with st.expander("📎 Voorbeelden naar model (exacte tekst)"):
            st.code(result["examples_block"], language="markdown")

    if show_prompt:
        with st.expander("🛠 Prompt (debug)"):
            st.code(result["prompt"], language="markdown")

