# app.py — Result Areas Generator (zonder JSON, met vaste voorbeelden-tabel)

# --- SQLite shim voor Streamlit Cloud (Chromadb vereist sqlite >= 3.35) ---
import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import os
from typing import List, Dict

import streamlit as st
import pandas as pd

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

# ── CONFIG ─────────────────────────────────────────────────────────────
PERSIST_DIR     = "chroma_db"
COLLECTION_NAME = "kb_result_areas"
EMBED_MODEL     = "text-embedding-3-small"
GEN_MODEL       = "gpt-4o-mini"  # or "gpt-4o" / "gpt-4.1-mini"

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Result Areas Generator", layout="wide")

# ---- Logo + Title row ----
from PIL import Image

def show_header():
    col1, col2 = st.columns([8, 2])  # give the logo a bit more space
    with col1:
        st.title("Resultaatgebieden (Generator)")
    with col2:
        try:
            logo = Image.open("AEM-Cube_Poster3_HI_Logo.png")
            st.image(logo, width=180)   # was 100 → now 180
        except Exception:
            pass

show_header()


# ── Vectorstore laden ─────────────────────────────────────────────────
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
    st.error(f"Kon Chroma-collectie niet laden. Controleer map '{PERSIST_DIR}' en collection name. Fout: {e}")
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

# ── Theorie & schrijfregels ──────────────────────────────────────────
THEORY = """AEM-Cube theory:
- Attachment: veiligheid via mensen (relaties) vs inhoud (systemen/ideeën).
- Exploration: innoveren vs optimaliseren.
- Managing Complexity: specialisten (diepte) vs generalisten (breedte)."""

HOW_TO_RA_NL = """Resultaatgebieden:
- 3–6 per functie; ingedeeld in thema's; essentieel, door één individu uitvoerbaar.
- Proces met begin en eind; gebruik werkwoorden die iets opleveren/creëren.
- Eén zin die het **wat** én het **waarom** combineert.
- Voorbeeld: “transparante rekeningen leveren **zodat** we een tevreden klantenbasis opbouwen.”"""

# ── Prompt bouw ──────────────────────────────────────────────────────
def build_system_msg() -> str:
    return f"""Je bent een HR/Org design assistent. Gebruik de AEM-Cube theorie en onderstaande schrijfregels om thema’s en resultaatgebieden te formuleren.
Geef de thema's die passen bij de functie, met de daarbij behorende **AEM-Cube positie** (alleen buckets).
Geef 2–4 resultaatgebieden per thema. Elk in **exact één zin** (wat + waarom).

THEORY
{THEORY}

SCHRIJFREGELS (NL)
{HOW_TO_RA_NL}

UITVOERFORMAAT:
Voor elk **thema**:
- Subtitel = themanaam.
  • **AEM-Cube positie**: A=…, E=…, M=… (alleen buckets)  
- Daaronder 2–4 bullets:
  • **Resultaatgebied**: één zin."""

def build_examples_block(examples: List[Dict]) -> str:
    if not examples:
        return "(geen voorbeelden gevonden)"
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
        if ex.get("source"):
            line += f" | Bron: {ex['source']}"
        lines.append(line)
    return "Relevante voorbeelden:\n" + "\n".join(lines)

def generate_result_areas(role_title: str, role_desc: str, examples: List[Dict], language: str = "nl") -> str:
    examples_block = build_examples_block(examples)
    role_text = f"Functietitel: {role_title}\nOmschrijving: {role_desc}"

    system_msg = build_system_msg()
    user_msg = f"""Language: {language}

Functiecontext:
\"\"\"{role_text.strip()}\"\"\"

Voorbeelden:
{examples_block}

Taak:
- Baseer je voorstel op de functiecontext én voorbeelden.
- Maak per thema 3–6 resultaatgebieden (één zin elk).
- Voeg per resultaatgebied de AEM-Cube positie toe (alleen buckets).
- Schrijf compact en concreet, in het Nederlands."""

    llm = ChatOpenAI(model=GEN_MODEL, temperature=0.2)
    resp = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ])
    return resp.content

# ── UI (titel + omschrijving) ────────────────────────────────────────
with st.form("ra_form"):
    role_title = st.text_input("Functietitel", placeholder="Bijv. Accountmanager B2B SaaS")
    role_desc = st.text_area(
        "Functieomschrijving",
        height=180,
        placeholder="Beschrijf kort de scope, taken en verantwoordelijkheden…"
    )
    k = st.number_input("Aantal voorbeelden (k)", min_value=2, max_value=10, value=4)
    submitted = st.form_submit_button("Formuleer resultaatgebieden")

if submitted:
    if not role_title.strip() and not role_desc.strip():
        st.warning("Vul functietitel en/of omschrijving in.")
        st.stop()

    with st.spinner("Voorbeelden ophalen en genereren…"):
        query_text = f"{role_title} {role_desc}"
        examples = retrieve_examples(vs, query_text, k=int(k))
        markdown = generate_result_areas(role_title, role_desc, examples, language="nl")

    st.markdown("### Resultaat")
    st.markdown(markdown, unsafe_allow_html=False)

    st.markdown("### Opgehaalde voorbeelden (tabel)")
    if examples:
        ex_df = pd.DataFrame(examples)[[
            "theme", "result_area",
            "band_attachment", "band_exploration", "band_managing_complexity",
            "source", "function_title"
        ]]
        st.dataframe(ex_df, use_container_width=True, hide_index=True)
    else:
        st.info("Geen voorbeelden gevonden voor deze query.")


