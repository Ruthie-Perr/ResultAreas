# app.py — Result Areas Generator (locked to AEM themes, soft-bias retrieval, single-select filters)

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
THEMES_PATH = Path(__file__).parent / "AEM_Cube_Themas.docx"

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

    # Try with hard filters first (if they exist in the KB, great)
    retriever = vs.as_retriever(search_kwargs={"k": k, "filter": {"$and": filt_and}})
    docs = retriever.get_relevant_documents(biased_query)

    # If filters returned nothing but user selected filters, fall back to base filter (soft bias only)
    if not docs and (org_type or sector):
        retriever = vs.as_retriever(search_kwargs={"k": k, "filter": base_filter})
        docs = retriever.get_relevant_documents(biased_query)

    # Optional rerank: boost docs that mention selected org_type/sector in text or metadata
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
        if key in seen:
            continue
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

# ── Theme loading from Word + selection logic ────────────────────────
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

@st.cache_resource
LEVEL_HEADERS = {
    "strategisch": "Strategisch",
    "tactisch": "Tactisch",
    "operationeel": "Operationeel",
    "specialistisch werk": "Specialistisch werk",
}

def _is_level_header(line: str) -> str | None:
    norm = _normalize_txt(line)
    for key, label in LEVEL_HEADERS.items():
        # exact of begint met (soms staan er haakjes of subtekst)
        if norm == key or norm.startswith(key):
            return label
    return None

def _is_metric_line(line: str) -> tuple[str, str] | None:
    """
    Herkent regels als:
      A: 75-100
      E: 25–50
      M: 0-25
    en retourneert ("A"|"E"|"M", "waarde-raw").
    """
    m = re.match(r"^\s*([AEM])\s*:\s*(.+?)\s*$", line, flags=re.IGNORECASE)
    if not m:
        return None
    axis = m.group(1).upper()
    val  = m.group(2).strip()
    # normalize en dash/ minus
    val = val.replace("–", "-").replace("—", "-")
    return axis, val

def _clean_theme_name(line: str) -> str:
    # strip bullets en whitespace
    s = line.lstrip("-*• ").strip()
    # soms staan er (A/E/M)-hints achter de naam → laat _parse_theme_line dat doen
    return s

def load_themes_from_docx(path: str | Path = THEMES_PATH) -> list[dict]:
    try:
        from docx import Document
    except ImportError:
        st.error("Voeg `python-docx` toe aan requirements.txt om thema’s uit Word te lezen.")
        return []

    path = Path(path)
    if not path.exists():
        st.error(f"Thema-bestand niet gevonden op {path}. Zet het .docx in je repo (bijv. ./data/).")
        return []

    doc = Document(str(path))

    # 1) alle lijnen uit paragrafen + tabellen verzamelen
    lines: list[str] = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            lines.append(t)
    for tbl in getattr(doc, "tables", []):
        for row in tbl.rows:
            for cell in row.cells:
                t = (cell.text or "").strip()
                if t:
                    lines.append(t)

    # 2) door de lijnen itereren en blokken maken
    themes: list[dict] = []
    seen_keys: set[str] = set()

    current_level: str | None = None
    i = 0
    n = len(lines)

    while i < n:
        raw = lines[i].strip()
        if not raw:
            i += 1
            continue

        # A) level header?
        lvl = _is_level_header(raw)
        if lvl:
            current_level = lvl
            i += 1
            continue

        # B) metric-line? (dan hoort dit bij voorafgaande thema – maar als we hier binnenkomen
        #    zonder thema, slaan we het veilig over)
        if _is_metric_line(raw):
            i += 1
            continue

        # C) anders: kandidaat-thema (korte regel die geen metric is en geen header)
        #    We laten _parse_theme_line ook eventuele "(A:..., E:..., M:...)" hints oppakken.
        candidate = _clean_theme_name(raw)
        # skip te lange alinea's, we willen thematitels (heuristiek)
        if len(candidate.split()) > 8:
            i += 1
            continue

        theme_dict = _parse_theme_line(candidate)  # haalt name en evt hints uit (...)
        theme_dict.setdefault("name", candidate)
        if current_level:
            theme_dict["level"] = current_level

        # D) verzamel A/E/M regels die direct volgen
        j = i + 1
        axes = {"A": None, "E": None, "M": None}
        while j < n:
            nxt = lines[j].strip()
            if not nxt:
                j += 1
                continue
            if _is_level_header(nxt):
                break  # nieuw blok
            mm = _is_metric_line(nxt)
            if mm:
                axis, val = mm
                axes[axis] = val
                j += 1
                continue
            # volgende thema start als het GEEN metric is en lijkt op een titel
            # stop dus bij de volgende "candidate-like" lijn
            # (korte regel, geen dubbele punt aan het begin, geen opsomming van zinnen)
            looks_like_theme = (len(_clean_theme_name(nxt).split()) <= 8) and (not _is_metric_line(nxt))
            if looks_like_theme:
                break
            j += 1

        # merge axes met eventuele hints uit (...) die _parse_theme_line al gezet kan hebben
        if axes["A"] and not theme_dict.get("A"): theme_dict["A"] = axes["A"]
        if axes["E"] and not theme_dict.get("E"): theme_dict["E"] = axes["E"]
        if axes["M"] and not theme_dict.get("M"): theme_dict["M"] = axes["M"]

        # dedupe op (name, level) genormaliseerd
        key = f"{_normalize_txt(theme_dict['name'])}||{theme_dict.get('level','')}"
        if theme_dict["name"] and key not in seen_keys:
            seen_keys.add(key)
            themes.append(theme_dict)

        i = j  # ga verder na het thema-blok

    # 3) fallback: als we niets gevonden hebben, gebruik je oude heuristiek nog 1x
    if not themes:
        seen = set()
        for ln in lines:
            raw = ln.lstrip("-*• ").strip()
            if not raw:
                continue
            if len(raw.split()) <= 6 or re.match(r"^[\-\*•]\s+", ln):
                t = _parse_theme_line(raw)
                key = _normalize_txt(t.get("name", ""))
                if key and key not in seen:
                    seen.add(key)
                    themes.append(t)

    return themes
def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def _match_to_known_theme(example_theme: str, known_theme_names_norm: List[str]) -> Optional[str]:
    ex = set(_normalize_txt(example_theme).split())
    if not ex:
        return None
    best, best_score = None, 0.0
    for k in known_theme_names_norm:
        kset = set(k.split())
        j = len(ex & kset) / (len(ex | kset) + 1e-8)
        if j > best_score:
            best, best_score = k, j
    return best if best_score >= 0.34 else None

def select_themes_for_role_and_examples(
    themes: List[Dict],
    role_title: str,
    role_desc: str,
    examples: List[Dict],
    top_n: int = 3,
    alpha: float = 0.7,
) -> List[Dict]:
    if not themes:
        return []
    names = [t["name"] for t in themes]
    names_norm = [_normalize_txt(n) for n in names]

    emb = embedder()
    role_vec = emb.embed_query(f"{role_title} {role_desc}".strip())
    theme_vecs = emb.embed_documents(names)

    sims = np.array([_cos(role_vec, v) for v in theme_vecs])
    if sims.max() > sims.min():
        sims = (sims - sims.min()) / (sims.max() - sims.min())
    else:
        sims = np.ones_like(sims) * 0.5

    counts = np.zeros(len(names))
    for ex in (examples or []):
        ex_theme = ex.get("theme") or ""
        mt = _match_to_known_theme(ex_theme, names_norm)
        if mt:
            idx = names_norm.index(mt)
            counts[idx] += 1

    cov = counts / counts.max() if counts.max() > 0 else counts
    score = alpha * sims + (1.0 - alpha) * cov
    order = np.argsort(-score)
    picked_idx = order[:max(1, top_n)]
    return [themes[i] for i in picked_idx]

# ── Prompt build (LOCK to selected themes) ────────────────────────────
def build_system_msg(selected_themes: List[Dict]) -> str:
    if selected_themes:
        fixed_lines = []
        for t in selected_themes:
            line = f"- {t['name']}"
            hints = []
            if t.get("A"): hints.append(f"A={t['A']}")
            if t.get("E"): hints.append(f"E={t['E']}")
            if t.get("M"): hints.append(f"M={t['M']}")
            if hints:
                line += "  (" + ", ".join(hints) + ")"
            fixed_lines.append(line)
        fixed_block = "Gebruik **uitsluitend** deze thema's (géén nieuwe toevoegen):\n" + "\n".join(fixed_lines)
    else:
        fixed_block = "Geen vaste thema’s gevonden; gebruik je beste inschatting (AEM-Cube)."

    return f"""Je bent een HR/Org design assistent. Gebruik AEM-Cube en onderstaande schrijfregels.
**Voeg géén nieuwe thema’s toe**; werk uitsluitend binnen de opgegeven lijst.

THEORY
{THEORY}

SCHRIJFREGELS (NL)
{HOW_TO_RA_NL}

THEMA'S (VAST):
{fixed_block}

UITVOERFORMAAT:
Voor elk thema:
- Subtitel = themanaam.
  • **AEM-Cube positie**: A=…, E=…, M=… (alleen buckets; volg hints indien aanwezig)
- 2–4 bullets:
  • **Resultaatgebied**: exact één zin (wat + waarom)."""

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

def generate_result_areas(
    role_title: str,
    role_desc: str,
    examples: List[Dict],
    selected_themes: List[Dict],
    language: str = "nl",
    org_type: Optional[str] = None,
    sector: Optional[str] = None,
) -> str:
    examples_block = build_examples_block(examples)
    ctx = [f"Functietitel: {role_title}", f"Omschrijving: {role_desc}"]
    if org_type: ctx.append(f"Organisatietype: {org_type}")
    if sector:   ctx.append(f"Sector: {sector}")
    role_text = "\n".join(ctx)

    system_msg = build_system_msg(selected_themes)
    user_msg = f"""Language: {language}

Functiecontext:
\"\"\"{role_text.strip()}\"\"\"

Voorbeelden:
{examples_block}

Taak:
- Baseer je voorstel op de functiecontext, voorbeelden **en** de vaste thema-lijst.
- Per thema 2–4 resultaatgebieden (één zin elk) met AEM-Cube buckets.
- Schrijf compact en concreet, in het Nederlands."""

    llm = ChatOpenAI(model=GEN_MODEL, temperature=0.2)
    resp = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ])
    return resp.content

# ── UI (titel + omschrijving + filters) ──────────────────────────────
with st.form("ra_form"):
    role_title = st.text_input("Functietitel", placeholder="Bijv. Accountmanager B2B SaaS")
    role_desc = st.text_area(
        "Functieomschrijving",
        height=180,
        placeholder="Beschrijf kort de scope, taken en verantwoordelijkheden…"
    )
    k = st.number_input("Aantal voorbeelden (k)", min_value=2, max_value=10, value=4)

    # Single-select dropdowns (work now with soft bias; later become hard filters automatically)
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

        # Load AEM themes from Word and auto-select best ones (role + examples), locked to your list
        all_themes = load_themes_from_docx()
        picked = select_themes_for_role_and_examples(
            themes=all_themes,
            role_title=role_title,
            role_desc=role_desc,
            examples=examples,
            top_n=3,        # tweak 1..6
            alpha=0.7       # 70% role similarity, 30% example coverage
        )

        markdown = generate_result_areas(
            role_title, role_desc, examples,
            selected_themes=picked,
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
        # keep only columns that exist in the dicts
        ex_cols = [c for c in ex_cols if any(c in e for e in examples)]
        ex_df = pd.DataFrame(examples)[ex_cols]
        st.dataframe(ex_df, use_container_width=True, hide_index=True)
    else:
        st.info("Geen voorbeelden gevonden voor deze selectie.")




