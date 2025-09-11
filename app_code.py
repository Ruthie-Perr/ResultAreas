# app.py — Result Areas Generator (LLM selects themes from allowed list; sector/profit-aware)

# --- SQLite shim for Streamlit Cloud (Chromadb requires sqlite >= 3.35) ---
import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import os, re, json
from typing import List, Dict, Optional

import numpy as np
import streamlit as st
import pandas as pd

from PIL import Image
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────
PERSIST_DIR     = "chroma_db"
COLLECTION_NAME = "kb_result_areas"
EMBED_MODEL     = "text-embedding-3-small"
GEN_MODEL       = "gpt-4o-mini"  # or "gpt-4o" / "gpt-4.1-mini"
THEMES_PATH     = Path(__file__).parent / "AEM_Cube_Themas.docx"

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Resultaatgebieden (Generator)", layout="wide")

# ---- Logo + Title row ----
def show_header():
    col1, col2 = st.columns([8, 2])
    with col1:
        st.title("Resultaatgebieden (Generator)")
    with col2:
        try:
            logo = Image.open("AEM-Cube_Poster3_HI_Logo.png")
            st.image(logo, width=180)
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

# ── Retrieval helper (soft bias now, hard filter when available) ─────
@st.cache_resource
def embedder():
    return OpenAIEmbeddings(model=EMBED_MODEL)

def retrieve_examples(
    vs: Chroma,
    query_text: str,
    k: int = 8,
    app_scope: str = "result_areas",
    content_type: str = "example",
    org_type: Optional[str] = None,   # "profit" | "nonprofit" | None
    sector: Optional[str] = None,     # single string | None
) -> List[Dict]:
    base_filter = {"$and": [
        {"app_scope": {"$eq": app_scope}},
        {"content_type": {"$eq": content_type}},
    ]}
    filt_and = list(base_filter["$and"])
    if org_type:
        filt_and.append({"org_type": {"$eq": org_type}})
    if sector:
        filt_and.append({"sector": {"$eq": sector}})

    # Soft bias: append signals to query (works even if metadata not present)
    bias_bits = []
    if org_type: bias_bits.append(f"organisatietype: {org_type}")
    if sector:   bias_bits.append(f"sector: {sector}")
    biased_query = query_text if not bias_bits else (query_text + " | " + " | ".join(bias_bits))

    # Try with hard filters first
    retriever = vs.as_retriever(search_kwargs={"k": k, "filter": {"$and": filt_and}})
    docs = retriever.get_relevant_documents(biased_query)

    # Fallback to base if nothing found
    if not docs and (org_type or sector):
        retriever = vs.as_retriever(search_kwargs={"k": k, "filter": base_filter})
        docs = retriever.get_relevant_documents(biased_query)

    # Optional rerank: boost docs that mention selected org_type/sector
    def score_doc(d):
        t = (d.page_content or "") + " " + " ".join(str(v) for v in (d.metadata or {}).values())
        s = 0
        if org_type and org_type.lower() in t.lower():
            s += 1
        if sector and sector.lower() in t.lower():
            s += 1
        return s
    if org_type or sector:
        docs.sort(key=score_doc, reverse=True)

    # Dedupe and package
    seen, out = set(), []
    for d in docs:
        md = d.metadata or {}
        key = (md.get("theme",""), md.get("result_area",""))
        if key in seen: continue
        seen.add(key)
        out.append({
            "function_title": md.get("function_title", ""),
            "theme": md.get("theme", ""),
            "result_area": md.get("result_area", ""),
            "band_attachment": md.get("band_attachment", ""),
            "band_exploration": md.get("band_exploration", ""),
            "band_managing_complexity": md.get("band_managing_complexity", ""),
            "org_type": md.get("org_type", ""),
            "sector": md.get("sector", ""),
            "source": md.get("source", ""),
            "snippet": d.page_content,
        })
    return out

# ── AEM-Cube theory + schrijfregels ──────────────────────────────────
THEORY = """AEM-Cube theory:
- Attachment: veiligheid via mensen (relaties) vs inhoud (systemen/ideeën).
- Exploration: innoveren vs optimaliseren.
- Managing Complexity: specialisten (diepte) vs generalisten (breedte)."""

HOW_TO_RA_NL = """Resultaatgebieden:
- 3–6 per functie; ingedeeld in thema's; essentieel, door één individu uitvoerbaar.
- Proces met begin en eind; gebruik werkwoorden die iets opleveren/creëren.
- Eén zin die het **wat** én het **waarom** combineert.
- Voorbeeld: “transparante rekeningen leveren **zodat** we een tevreden klantenbasis opbouwen.”"""

# ── Theme loading from Word ───────────────────────────────────────────
def _normalize_txt(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _parse_theme_line(txt: str):
    out = {"name": txt.strip()}
    m = re.search(r"\((.*?)\)$", out["name"])
    if m:
        inside = m.group(1)
        out["name"] = out["name"][: m.start()].strip()
        for part in inside.split(","):
            part = part.strip()
            if re.match(r"^A\s*:", part, flags=re.I):
                out["A"] = part.split(":", 1)[1].strip()
            if re.match(r"^E\s*:", part, flags=re.I):
                out["E"] = part.split(":", 1)[1].strip()
            if re.match(r"^M\s*:", part, flags=re.I):
                out["M"] = part.split(":", 1)[1].strip()
    return out

LEVEL_HEADERS = {
    "strategisch": "Strategisch",
    "tactisch": "Tactisch",
    "operationeel": "Operationeel",
    # voeg "specialistisch werk": "Specialistisch werk" toe als je die laag ook gebruikt
}

def _is_level_header(line: str) -> str | None:
    norm = _normalize_txt(line)
    for key, label in LEVEL_HEADERS.items():
        if norm == key or norm.startswith(key):
            return label
    return None

def _is_metric_line(line: str) -> Optional[tuple[str, str]]:
    m = re.match(r"^\s*([AEM])\s*:\s*(.+?)\s*$", line, flags=re.IGNORECASE)
    if not m: return None
    axis = m.group(1).upper()
    val  = m.group(2).strip().replace("–", "-").replace("—", "-")
    return axis, val

def _clean_theme_name(line: str) -> str:
    return line.lstrip("-*• ").strip()

@st.cache_resource
def load_themes_from_docx(path: str | Path = THEMES_PATH) -> List[Dict]:
    try:
        from docx import Document
    except ImportError:
        st.error("Voeg `python-docx` toe aan requirements.txt om thema’s uit Word te lezen.")
        return []
    path = Path(path)
    if not path.exists():
        st.error(f"Thema-bestand niet gevonden op {path}. Zet het .docx in je repo (bijv. ./).")
        return []

    doc = Document(str(path))
    lines: List[str] = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t: lines.append(t)
    for tbl in getattr(doc, "tables", []):
        for row in tbl.rows:
            for cell in row.cells:
                t = (cell.text or "").strip()
                if t: lines.append(t)

    themes: List[Dict] = []
    seen_keys: set[str] = set()
    current_level: Optional[str] = None
    i, n = 0, len(lines)

    while i < n:
        raw = lines[i].strip()
        if not raw: i += 1; continue

        lvl = _is_level_header(raw)
        if lvl:
            current_level = lvl
            i += 1
            continue

        if _is_metric_line(raw):
            i += 1
            continue

        candidate = _clean_theme_name(raw)
        if len(candidate.split()) > 8:
            i += 1
            continue

        theme_dict = _parse_theme_line(candidate)
        theme_dict.setdefault("name", candidate)
        if current_level:
            theme_dict["level"] = current_level

        j = i + 1
        axes = {"A": None, "E": None, "M": None}
        while j < n:
            nxt = lines[j].strip()
            if not nxt: j += 1; continue
            if _is_level_header(nxt): break
            mm = _is_metric_line(nxt)
            if mm:
                axis, val = mm
                axes[axis] = val
                j += 1
                continue
            looks_like_theme = (len(_clean_theme_name(nxt).split()) <= 8) and (not _is_metric_line(nxt))
            if looks_like_theme: break
            j += 1

        if axes["A"] and not theme_dict.get("A"): theme_dict["A"] = axes["A"]
        if axes["E"] and not theme_dict.get("E"): theme_dict["E"] = axes["E"]
        if axes["M"] and not theme_dict.get("M"): theme_dict["M"] = axes["M"]

        key = f"{_normalize_txt(theme_dict['name'])}||{theme_dict.get('level','')}"
        if theme_dict["name"] and key not in seen_keys:
            seen_keys.add(key)
            themes.append(theme_dict)

        i = j

    if not themes:
        seen = set()
        for ln in lines:
            raw = ln.lstrip("-*• ").strip()
            if not raw: continue
            if len(raw.split()) <= 6 or re.match(r"^[\-\*•]\s+", ln):
                t = _parse_theme_line(raw)
                key = _normalize_txt(t.get("name", ""))
                if key and key not in seen:
                    seen.add(key); themes.append(t)

    return themes

# ── Prompt build (LLM selects themes from allowed list) ───────────────
def build_system_msg(allowed_themes: List[Dict]) -> str:
    allowed_names = [t["name"] for t in allowed_themes] if allowed_themes else []
    allowed_str = ", ".join(allowed_names) if allowed_names else "(none)"

    return f"""Je bent een HR/Org design assistent. 
Je taak:
1) Kies het **best passende aantal thema's** (meestal 2–5, maximaal 6) uitsluitend uit ALLOWED_THEMES die het beste aansluiten op de functiecontext.
2) Formuleer per gekozen thema **2–4 resultaatgebieden**, elk in **exact één zin** (wat + waarom).
3) Geef per resultaatgebied de **AEM-buckets** op (A, E, M) op bucket-niveau (geen exacte percentages).

Selectiecriteria (in volgorde):
- Match met **functietitel** en **beschrijving** (taken/scope).
- Relevantie voor **sector** en **organisatietype** (profit/nonprofit) indien opgegeven.
- Dekkingsgraad in **opgehaalde voorbeelden** (indien aanwezig).
- Vermijd overlap; kies complementaire thema's die samen de functie dekken.

AEM THEORY (beknopt)
- Attachment (A): mens/relatie vs inhoud/systeem
- Exploration (E): vernieuwen vs optimaliseren
- Managing Complexity (M): specialistisch vs generalistisch

SCHRIJFREGELS (NL)
- Resultaatgebied = één zin met **wat** + **waarom** (concreet, compact).

ALLOWED_THEMES: {allowed_str}

UITVOERFORMAAT (STRICT JSON, alleen dit object retourneren):
{{
  "themes": [
    {{
      "name": "<één uit ALLOWED_THEMES>",
      "A": "<bucket (bijv. 0-25 | 25-50 | 50-75 | 75-100 | hoog/laag)>",
      "E": "<bucket>",
      "M": "<bucket>",
      "result_areas": [
        "<één zin wat+waarom>",
        "<…>"
      ]
    }}
  ]
}}
"""


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

def _render_markdown_from_struct(struct: dict, allowed_names: List[str]) -> str:
    """Render naar Markdown; alleen exact-allowed thema's worden getoond."""
    lines = []
    for item in struct.get("themes", []):
        name = (item.get("name") or "").strip()
        if name not in allowed_names:
            continue  # drop onbekend thema

        A = item.get("A", "")
        E = item.get("E", "")
        M = item.get("M", "")
        ras = [ra for ra in (item.get("result_areas") or []) if isinstance(ra, str) and ra.strip()]
        if not ras:
            continue

        lines.append(f"### {name}")
        aem_bits = []
        if A: aem_bits.append(f"A={A}")
        if E: aem_bits.append(f"E={E}")
        if M: aem_bits.append(f"M={M}")
        if aem_bits:
            lines.append("**AEM-Cube positie:** " + ", ".join(aem_bits))
        for ra in ras[:4]:
            lines.append(f"- **Resultaatgebied:** {ra.strip()}")
        lines.append("")
    return "\n".join(lines).strip() or "_Geen geldige thema’s met resultaatgebieden._"

def generate_result_areas(
    role_title: str,
    role_desc: str,
    examples: List[Dict],
    allowed_themes: List[Dict],
    language: str = "nl",
    org_type: Optional[str] = None,
    sector: Optional[str] = None,
) -> str:
    # 1) Context
    examples_block = build_examples_block(examples)
    ctx = [f"Functietitel: {role_title}", f"Omschrijving: {role_desc}"]
    if org_type: ctx.append(f"Organisatietype: {org_type}")
    if sector:   ctx.append(f"Sector: {sector}")
    role_text = "\n".join(ctx)

    # 2) Messages
    system_msg = build_system_msg(allowed_themes)
    user_msg = f"""Language: {language}

Functiecontext:
\"\"\"{role_text.strip()}\"\"\"

Voorbeelden (ter referentie, pas thema-keuze en formuleringen hierop aan):
{examples_block}

Taak:
- Kies het best passende aantal thema's (meestal 2–5, max 6) uit ALLOWED_THEMES die het best passen bij de functie + (optioneel) sector/orgtype.
- Per gekozen thema 2–4 resultaatgebieden, elk één zin, met A/E/M buckets (alleen buckets).
- Retourneer **enkel JSON** volgens het afgesproken schema.
"""

    # 3) JSON-only response
    llm = ChatOpenAI(
        model=GEN_MODEL,
        temperature=0.2,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    resp = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ])

    raw = resp.content or "{}"
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        data = json.loads(m.group(0)) if m else {"themes": []}

    allowed_names = [t["name"] for t in allowed_themes]
    markdown = _render_markdown_from_struct(data, allowed_names)
    return markdown


# ── UI (titel + omschrijving + filters) ──────────────────────────────
with st.form("ra_form"):
    role_title = st.text_input("Functietitel", placeholder="Bijv. Accountmanager B2B SaaS")
    role_desc = st.text_area(
        "Functieomschrijving",
        height=180,
        placeholder="Beschrijf kort de scope, taken en verantwoordelijkheden…"
    )
    k = st.number_input("Aantal voorbeelden (k)", min_value=2, max_value=10, value=4)

    # Single-select dropdowns
    ORG_TYPES = ["(alle)", "profit", "nonprofit"]
    SECTORS   = ["(alle)",
        "healthcare", "education", "government", "finance", "tech",
        "manufacturing", "retail", "logistics", "energy", "nonprofit/ngo"
    ]
    org_type_choice = st.selectbox("Organisatietype", ORG_TYPES, index=0)
    sector_choice   = st.selectbox("Sector", SECTORS, index=0)


    submitted = st.form_submit_button("Formuleer resultaatgebieden")

if submitted:
    if not role_title.strip() and not role_desc.strip():
        st.warning("Vul functietitel en/of omschrijving in.")
        st.stop()

    chosen_org    = org_type_choice if org_type_choice != "(alle)" else None
    chosen_sector = sector_choice   if sector_choice   != "(alle)" else None

   with st.spinner("Voorbeelden ophalen en genereren…"):
    query_text = f"{role_title} {role_desc}"
    examples = retrieve_examples(
        vs,
        query_text,
        k=int(k),
        org_type=chosen_org,
        sector=chosen_sector
    )

    # Allowed list = alle thema's uit Word
    all_themes = load_themes_from_docx()

    # LLM kiest ZELF het aantal best passende thema's uit allowed list
    markdown = generate_result_areas(
        role_title, role_desc, examples,
        allowed_themes=all_themes,
        language="nl",
        org_type=chosen_org,
        sector=chosen_sector
    )



    st.markdown("### Resultaat")
    st.markdown(markdown, unsafe_allow_html=False)

    st.markdown("### Opgehaalde voorbeelden (tabel)")
    if examples:
        ex_cols = [
            "theme", "result_area",
            "band_attachment", "band_exploration", "band_managing_complexity",
            "org_type", "sector", "source", "function_title"
        ]
        ex_cols = [c for c in ex_cols if any(c in e for e in examples)]
        ex_df = pd.DataFrame(examples)[ex_cols]
        st.dataframe(ex_df, use_container_width=True, hide_index=True)
    else:
        st.info("Geen voorbeelden gevonden voor deze selectie.")


