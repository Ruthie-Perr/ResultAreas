# app.py — Result Areas Generator (zonder JSON, met vaste voorbeelden-tabel)

# --- SQLite shim voor Streamlit Cloud (Chromadb vereist sqlite >= 3.35) ---
import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import os, re
from typing import List, Dict

import streamlit as st
import pandas as pd

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma


# ── CONFIG ─────────────────────────────────────────────────────────────
PERSIST_DIR     = "chroma_db"
COLLECTION_NAME = "kb_result_areas"
EMBED_MODEL     = "text-embedding-3-small"
GEN_MODEL       = "gpt-4o-mini"  # of "gpt-4o" / "gpt-4.1-mini"

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


st.set_page_config(page_title="Result Areas Generator", layout="wide")

st.markdown("""
<style>
/* === Fix page scrolling === */
html, body, .stApp {
  height: auto !important;
  overflow-y: auto !important;
}

/* === Teal background wash @15% === */
.stApp {
  position: relative;
}
.stApp::before {
  content: "";
  position: fixed;
  inset: 0;
  background: rgba(0, 117, 138, 0.15);  /* #00758A at 15% */
  z-index: -1;
  pointer-events: none;
}

/* === Typography (charcoal text) === */
html, body, .stApp, .stAppViewContainer, .main, .block-container,
h1, h2, h3, h4, h5, h6, p, span, div, label, textarea, input, button {
  color: #222222 !important;
  font-family: 'Museo Sans', 'Source Sans 3', sans-serif !important;
}

/* === Input boxes (white background) === */
.stTextInput > div > div > input,
.stTextArea textarea,
.stNumberInput input {
  background: #ffffff !important;
  color: #222222 !important;
  border: 1px solid rgba(0,0,0,0.1) !important;
  border-radius: 10px !important;
}

/* === Buttons (turquoise) === */
.stButton button,
[data-testid="stButton"] button,
button[kind],
[data-testid^="baseButton"] {
  all: unset !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  cursor: pointer !important;

  background: #2BA6B5 !important;
  color: #ffffff !important;
  border-radius: 10px !important;
  padding: 0.55rem 1rem !important;
  font-weight: 600 !important;
  font-size: 16px !important;
}
.stButton button:hover,
[data-testid="stButton"] button:hover,
button[kind]:hover,
[data-testid^="baseButton"]:hover {
  background: #2593A0 !important;
  color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)



# ---- Logo + Title row ----
from PIL import Image

def show_header():
    col1, col2 = st.columns([8, 1])
    with col1:
        st.markdown("<h1>Resultaatgebieden (Generator)</h1>", unsafe_allow_html=True)
    with col2:
        try:
            logo = Image.open("AEM-Cube_Poster3_HI_Logo.png")
            st.image(logo, width=100)
        except Exception:
            pass


show_header()






#st.caption("Voer functietitel en -omschrijving in. Ik haal voorbeelden op en genereer thema’s & resultaatgebieden inclusief AEM‑Cube positie (alleen buckets).")

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

# ── Retrieval helper (Chroma 0.5+ filters) ───────────────────────────
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
            "snippet": d.page_content,  # context voor few-shot
        })
    return out

# ── Theorie & schrijfregels ──────────────────────────────────────────
THEORY = """AEM-Cube theory:
- Attachment: veiligheid via mensen (relaties) vs inhoud (systemen/ideeën).
- Exploration: innoveren vs optimaliseren (growth-curve: vroeg = exploreren, later = optimaliseren).
- Managing Complexity: specialisten (diepte) vs generalisten (breedte)."""

HOW_TO_RA_NL = """Resultaatgebieden:
- 3–6 per functie; essentieel, door één individu uitvoerbaar.
- Proces met begin en eind; gebruik werkwoorden die iets opleveren/creëren.
- Eén zin die het **wat** én het **waarom** combineert (concreet en begrijpelijk).
- Voorbeeld (goed): “transparante rekeningen leveren **zodat** we een tevreden klantenbasis opbouwen.”"""

# ── Prompt bouw ──────────────────────────────────────────────────────
def build_system_msg() -> str:
    # Let op: alleen buckets, geen numerieke scores.
    return f"""Je bent een HR/Org design assistent. Gebruik de AEM‑Cube theorie en onderstaande schrijfregels om thema’s en resultaatgebieden voor een functie te formuleren.
Geef 3–6 resultaatgebieden. Elk resultaatgebied moet **exact één zin** zijn waarin het **wat** en het **waarom** geïntegreerd zijn.
Geef daarnaast per resultaatgebied de **AEM‑Cube positie** (alleen buckets) voor Attachment, Exploration en Managing Complexity.

BELANGRIJKE PRINCIPES
- Prioriteit bij onderbouwing: (1) Functiecontext van de gebruiker, (2) Opgehaalde voorbeelden, (3) Theorie.
- Gebruik de voorbeelden als leidraad/inspiratie; herformuleer passend bij de functiecontext. Kopieer geen zinnen letterlijk.
- Geef 3–6 resultaatgebieden. Elk resultaatgebied is **exact één zin** met wat + waarom.
- Geef per resultaatgebied de **AEM‑Cube positie** als buckets (A/E/M) met **exact één** uit: "0-25", "25-50", "50-75", "75-100". Geen numerieke scores.


THEORY
{THEORY}

SCHRIJFREGELS (NL)
{HOW_TO_RA_NL}

UITVOERFORMAAT (alleen Markdown, geen JSON):
Voor elk **thema**:
- Zet de themanaam als subtitel.
- Geef daaronder 3–6 bullets, ieder met:
  • **Resultaatgebied**: één zin met wat + waarom.  
  • **AEM‑Cube positie**: A=…, E=…, M=… (met alleen een bucket per dimensie).
Schrijf in het Nederlands als language='nl'."""

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
    return "Relevante voorbeelden (ter inspiratie, pas aan indien passend):\n" + "\n".join(lines)

def generate_result_areas(role_title: str, role_desc: str, examples: List[Dict], language: str = "nl") -> str:
    examples_block = build_examples_block(examples)
    role_text = f"Functietitel: {role_title}\nOmschrijving: {role_desc}"

    system_msg = build_system_msg()
    user_msg = f"""Language: {language}

Functiecontext:
\"\"\"{role_text.strip()}\"\"\"

OPGEHAALDE VOORBEELDEN — TE GEBRUIKEN ALS INSPIRATIE (herformuleer passend, niet letterlijk kopiëren):
{examples_block}

Taak:
- Baseer je voorstel op **zowel** de functiecontext **als** de opgehaalde voorbeelden.
- Maak per **thema** 3–6 resultaatgebieden (één zin elk; wat + waarom).
- Voeg per resultaatgebied de **AEM‑Cube positie** toe als buckets: A=0-25|25-50|50-75|75-100, E=…, M=….
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


    # Altijd de opgehaalde voorbeelden als tabel tonen (wat we hebben gebruikt)
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



