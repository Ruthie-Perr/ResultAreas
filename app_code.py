# app.py
import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass  # fallback if package not available

import os
import re
import json
import streamlit as st
import pandas as pd
from typing import List, Dict, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PERSIST_DIR     = "chroma_db"            # folder you committed to GitHub
COLLECTION_NAME = "kb_result_areas"      # same as in build_kb.py
EMBED_MODEL     = "text-embedding-3-small"

# Streamlit secrets → env for langchain_openai
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Result Areas Retriever", layout="wide")
st.title("Generator Result Areas")

#st.caption(
#    "Generating..."
#)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def normalize(s: str) -> str:
    s = (s or "").strip()
    return re.sub(r"\s+", " ", s)

def build_query(function_title: str, description: str) -> str:
    title = normalize(function_title)
    desc  = normalize(description)
    parts = []
    if title:
        parts.append(f"Function: {title}")
    if desc:
        parts.append(f"Description: {desc}")
    return " | ".join(parts)

@st.cache_resource(show_spinner=True)
def load_vectorstore(persist_dir: str, collection_name: str):
    """Load the persisted Chroma collection with the same embedding model used to build it."""
    emb = OpenAIEmbeddings(model=EMBED_MODEL)  # uses OPENAI_API_KEY from env
    vs = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=emb,
    )
    return vs

def retrieve_examples(
    vs: Chroma,
    query_text: str,
    k: int = 8,
    app_scope: str = "result_areas",
    content_type: str = "example"
) -> List[Dict]:
    """Vector search with metadata filter; returns list of dicts with theme/result_area/bands."""

    # ✅ New filter syntax for Chroma 0.5+
    filt = {
        "$and": [
            {"app_scope": {"$eq": app_scope}},
            {"content_type": {"$eq": content_type}},
        ]
    }

    retriever = vs.as_retriever(
        search_kwargs={
            "k": k,
            "filter": filt,
        }
    )
    docs = retriever.get_relevant_documents(query_text)

    # Deduplicate by (theme, result_area) keeping first occurrence
    seen = set()
    out = []
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
            # "snippet": d.page_content,  # optional for debugging
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    k = st.number_input("Top‑K results", min_value=3, max_value=20, value=8, step=1)
    st.markdown(
        f"**KB folder:** `{PERSIST_DIR}`  \n"
        f"**Collection:** `{COLLECTION_NAME}`  \n"
        f"**Embed model:** `{EMBED_MODEL}`"
    )

# Load KB
try:
    vs = load_vectorstore(PERSIST_DIR, COLLECTION_NAME)
except Exception as e:
    st.error(
        f"Failed to load Chroma collection. Ensure the folder `{PERSIST_DIR}` "
        f"exists in this repo and collection name matches.\n\nError: {e}"
    )
    st.stop()

col1, col2 = st.columns([1, 2])
with col1:
    title_input = st.text_input("Function title", placeholder="e.g., Senior Data Analyst")
with col2:
    desc_input = st.text_input("Function description (1 sentence)", placeholder="e.g., Responsible for building product analytics and insights...")

go = st.button("Retrieve Themes & Result Areas", type="primary", use_container_width=True)

if go:
    if not title_input and not desc_input:
        st.warning("Please enter at least a function title or a description.")
    else:
        query = build_query(title_input, desc_input)
        results = retrieve_examples(vs, query, k=int(k))

        if not results:
            st.info("No matching examples found.")
        else:
            st.subheader("Retrieved Examples")
            for r in results:
                st.markdown(
                    f"**Theme:** {r['theme']}  \n"
                    f"**Result area:** {r['result_area']}  \n"
                    f"**A/E/M bands:** "
                    f"A={r['band_attachment']} | "
                    f"E={r['band_exploration']} | "
                    f"M={r['band_managing_complexity']}  \n"
                    f"<span style='color:gray'>Source: {r['source']} | Function: {r['function_title']}</span>",
                    unsafe_allow_html=True
                )
                st.write("---")

            st.subheader("Table view")
            df = pd.DataFrame(results)[
                ["theme", "result_area", "band_attachment", "band_exploration", "band_managing_complexity", "function_title", "source"]
            ]
            st.dataframe(df, use_container_width=True)

            st.download_button(
                "Download JSON",
                data=json.dumps(results, ensure_ascii=False, indent=2),
                file_name="retrieved_result_areas.json",
                mime="application/json",
            )

st.caption("Uses vector search over the persisted Chroma KB with metadata filter (no PDFs/Excels at runtime).")
