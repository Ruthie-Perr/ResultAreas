# app.py — Result Areas Generator (LLM generates result areas first; then assigns themes)

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
import hashlib

from PIL import Image
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from pathlib import Path

from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from datetime import datetime


# ── CONFIG ─────────────────────────────────────────────────────────────
PERSIST_DIR     = "chroma_db"
COLLECTION_NAME = "kb_result_areas"
EMBED_MODEL     = "text-embedding-3-small"
GEN_MODEL       = "gpt-4o-mini"  # or "gpt-4o" / "gpt-4.1-mini"

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
- 6–10 per functie; ingedeeld in thema's; essentieel, door één individu uitvoerbaar.
- Proces met begin en eind; gebruik werkwoorden die iets opleveren/creëren.
- Eén zin die het **wat** én het **waarom** combineert.
- Voorbeeld: “transparante rekeningen leveren **zodat** we een tevreden klantenbasis opbouwen.”"""

# ── Inline ALLOWED THEMES (replaces Word .docx loader) ────────────────
THEMES_ALLOWED: List[Dict[str, str]] = [
    # Strategisch (Beleidsmakers)
    {"name": "Stakeholder alignment", "A": "75-100", "E": "25-50", "M": "75-100", "level": "Strategisch"},
    {"name": "Governance & Cultuur", "A": "75-100", "E": "0-25", "M": "75-100", "level": "Strategisch"},
    {"name": "Netwerkvorming & Partnerships", "A": "75-100", "E": "75-100", "M": "75-100", "level": "Strategisch"},
    {"name": "Markt- en Trendontwikkeling", "A": "50-75", "E": "75-100", "M": "75-100", "level": "Strategisch"},
    {"name": "Beleidsplanning & Control", "A": "0-25", "E": "0-25", "M": "75-100", "level": "Strategisch"},
    {"name": "Compliancebeleid", "A": "0-25", "E": "0-25", "M": "75-100", "level": "Strategisch"},
    {"name": "Strategische innovatie & R&D", "A": "0-25", "E": "75-100", "M": "75-100", "level": "Strategisch"},
    {"name": "Visievorming nieuwe markten", "A": "25-50", "E": "75-100", "M": "75-100", "level": "Strategisch"},

    # Tactisch (Managers)
    {"name": "HR-processen", "A": "75-100", "E": "25-50", "M": "25-50", "level": "Tactisch"},
    {"name": "Organisatiecultuur", "A": "75-100", "E": "50-75", "M": "50-75", "level": "Tactisch"},
    {"name": "Schakel en Afstemming", "A": "75-100", "E": "25-50", "M": "50-75", "level": "Tactisch"},
    {"name": "Teamontwikkeling & Samenwerking", "A": "75-100", "E": "50-75", "M": "50-75", "level": "Tactisch"},
    {"name": "Resourceplanning & Efficiency", "A": "0-25", "E": "0-25", "M": "50-75", "level": "Tactisch"},
    {"name": "Kwaliteitsmanagement", "A": "0-25", "E": "0-25", "M": "25-50", "level": "Tactisch"},
    {"name": "Procesmanagement", "A": "0-25", "E": "0-25", "M": "25-50", "level": "Tactisch"},
    {"name": "Procesinnovatie", "A": "25-50", "E": "50-75", "M": "50-75", "level": "Tactisch"},

    # Operationeel (Uitvoerend)
    {"name": "Relatiebeheer", "A": "75-100", "E": "25-50", "M": "0-25", "level": "Operationeel"},
    {"name": "Dienstverlening & Ondersteuning", "A": "50-75", "E": "25-50", "M": "0-25", "level": "Operationeel"},
    {"name": "Nieuwe relaties & Doelgroepen ontwikkelen", "A": "75-100", "E": "75-100", "M": "25-50", "level": "Operationeel"},
    {"name": "Efficiënte uitvoering van processen", "A": "0-25", "E": "0-25", "M": "0-25", "level": "Operationeel"},
    {"name": "Veiligheid & Naleving", "A": "0-25", "E": "0-25", "M": "0-25", "level": "Operationeel"},
    {"name": "Specialistisch werk", "A": "0-25", "E": "0-25", "M": "0-25", "level": "Operationeel"},
    {"name": "Kennis en inzicht delen", "A": "25-50", "E": "25-50", "M": "25-50", "level": "Operationeel"},
    {"name": "Praktische innovaties", "A": "0-25", "E": "50-75", "M": "0-25", "level": "Operationeel"},
]

# ── Prompt build (LLM defines RAs first, then assigns themes) ─────────
def build_system_msg(allowed_themes: List[Dict]) -> str:
    def fmt_theme(t):
        name = t.get("name", "").strip()
        lvl  = (t.get("level") or "").strip()
        A = (t.get("A") or "").strip()
        E = (t.get("E") or "").strip()
        M = (t.get("M") or "").strip()
        parts = []
        if lvl: parts.append(lvl)
        if A: parts.append(f"A: {A}")
        if E: parts.append(f"E: {E}")
        if M: parts.append(f"M: {M}")
        return f"- {name}" + (f" ({', '.join(parts)})" if parts else "")

    allowed_block = "\n".join(fmt_theme(t) for t in allowed_themes if t.get("name"))

    return f"""Je bent een HR/Org design assistent.

DOEL EN SPELREGELS (ENKEL HIER, BRON VAN WAARHEID):
- Genereer eerst **6–10 resultaatgebieden** (elk **exact één zin**: wat + waarom) die samen de functie dekken.
- **Deel ALLE resultaatgebieden in** onder **3–4 best passende thema’s** uit ALLOWED_THEMES.
- Elk resultaatgebied hoort in **exact één** thema (geen dubbeltellingen, geen restcategorie).
- **Gebruik themanaam exact** zoals in ALLOWED_THEMES.
- **Gebruik per thema de A/E/M-buckets exact** zoals in ALLOWED_THEMES (geen alternatieve waarden).
- **Geen rationale** opnemen in de output.
- **Context meenemen**:
  - Baseer je primair op functietitel en -omschrijving.
  - Neem opgegeven **sector** en **organisatietype** expliciet mee in formulering en clustering.
  - Gebruik **voorbeelden** alleen ter inspiratie/bias; kopieer niet letterlijk.
- Output is **strikt JSON** conform het schema hieronder (geen extra tekst).
- Kies thema’s die de **kern van de functiecontext** weerspiegelen. 
  Vermijd thema’s die duidelijk niet relevant zijn (bijvoorbeeld: geen HR-processen kiezen 
  tenzij de functieomschrijving expliciet over HR gaat).
- Voor uitvoerende of technische functies met veel vakkennis of analytische taken 
  moet het thema **Specialistisch werk** expliciet overwogen worden.


ALLOWED_THEMES (met exact te gebruiken A/E/M):
{allowed_block}

UITVOERFORMAAT (STRICT JSON — alleen dit object):
{{
  "themes": [
    {{
      "name": "<exacte themanaam uit ALLOWED_THEMES>",
      "A": "<exact uit ALLOWED_THEMES>",
      "E": "<exact uit ALLOWED_THEMES>",
      "M": "<exact uit ALLOWED_THEMES>",
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


def _render_markdown_from_struct(struct: dict, allowed_themes: List[Dict]) -> str:
    # Map: normalized name -> canonical {name, A, E, M}
    def norm_name(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s

    canon_map = {}
    for t in allowed_themes:
        nm = t.get("name")
        if not nm: continue
        canon_map[norm_name(nm)] = {
            "name": nm,
            "A": (t.get("A") or "").strip(),
            "E": (t.get("E") or "").strip(),
            "M": (t.get("M") or "").strip(),
        }

    lines = []
    for item in struct.get("themes", []):
        raw = (item.get("name") or "").strip()
        canon = canon_map.get(norm_name(raw))
        if not canon:
            continue  # onbekend thema → negeren

        # OVERRULE: altijd A/E/M uit allowed list gebruiken (niet uit model)
        A = canon["A"]
        E = canon["E"]
        M = canon["M"]
        ras = [ra for ra in (item.get("result_areas") or []) if isinstance(ra, str) and ra.strip()]
        if not ras:
            continue

        lines.append(f"**Thema: {canon['name']}**")
        aem_bits = []
        if A: aem_bits.append(f"A={A}")
        if E: aem_bits.append(f"E={E}")
        if M: aem_bits.append(f"M={M}")
        if aem_bits:
            lines.append("**AEM-Cube Score:** " + ", ".join(aem_bits))
        for ra in ras[:4]:
            lines.append(f"- **Resultaatgebied:** {ra.strip()}")
        lines.append("")

    return "\n".join(lines).strip() or "_Geen geldige thema’s met resultaatgebieden._"


def _markdown_to_plain_lines(md: str) -> list[str]:
    """
    Very light markdown-to-plain rendering:
    - Strip leading ###/##/# (keep the text)
    - Keep bullet markers
    - Remove **bold** markers
    - Trim extra spaces
    """
    lines = []
    for raw in (md or "").splitlines():
        s = raw.rstrip()
        if not s:
            lines.append("")
            continue
        # headers -> keep text only
        s = re.sub(r"^\s*#{1,6}\s*", "", s)
        # bold/italics markers
        s = s.replace("**", "").replace("__", "").replace("*", "")
        # turn smart dashes into normal hyphen
        s = s.replace("–", "-").replace("—", "-")
        lines.append(s)
    return lines


def build_pdf_bytes(title: str, role_desc: str, md_content: str) -> bytes:
    """
    Render a simple, readable PDF with a title, optional role description,
    and the generated markdown content converted to plain lines.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Font setup
    font_name = "Helvetica"
    c.setTitle(title or "Resultaatgebieden")
    c.setAuthor("Resultaatgebieden (Generator)")
    c.setFont(font_name, 14)

    # Margins and layout
    left = 2.0 * cm
    right = width - 2.0 * cm
    top = height - 2.0 * cm
    bottom = 2.0 * cm
    line_height = 14 * 1.2  # 1.2 leading

    y = top

    def write_line(text: str, size: int = 11, indent: float = 0):
        nonlocal y
        c.setFont(font_name, size)
        max_width = right - (left + indent)
        # naive wrap
        words = text.split(" ")
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            if c.stringWidth(test, font_name, size) > max_width and line:
                c.drawString(left + indent, y, line)
                y -= line_height
                if y < bottom:
                    c.showPage()
                    y = top
                    c.setFont(font_name, size)
                line = w
            else:
                line = test
        # last bit
        if line or text == "":
            c.drawString(left + indent, y, line)
            y -= line_height
            if y < bottom:
                c.showPage()
                y = top
                c.setFont(font_name, size)

    # Header
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    write_line(title or "Resultaatgebieden", size=16)
    write_line(f"Aangemaakt: {now_str}", size=9)
    write_line("")

    # Optional context
    if role_desc and role_desc.strip():
        write_line("Functiecontext", size=12)
        for l in _markdown_to_plain_lines(role_desc):
            write_line(l, size=10)
        write_line("")

    # Body (from markdown)
    lines = _markdown_to_plain_lines(md_content)
    bullet_indent = 0.6 * cm
    for l in lines:
        if l.startswith("- "):
            write_line("• " + l[2:], size=11, indent=bullet_indent)
        else:
            if l and not l.startswith(("•", "- ", "Resultaatgebied:")) and l == l.upper():
                write_line(l, size=12)
            else:
                write_line(l, size=11)

    c.save()
    buffer.seek(0)
    return buffer.read()

def generate_result_areas(
    role_title: str,
    role_desc: str,
    examples: List[Dict],
    allowed_themes: List[Dict],
    language: str = "nl",
    org_type: Optional[str] = None,
    sector: Optional[str] = None,
) -> str:
    examples_block = build_examples_block(examples)
    ctx = [f"Functietitel: {role_title}", f"Omschrijving: {role_desc}"]
    if org_type: ctx.append(f"Organisatietype: {org_type}")
    if sector:   ctx.append(f"Sector: {sector}")
    role_text = "\n".join(ctx)

    system_msg = build_system_msg(allowed_themes)
    user_msg = f"""Language: {language}

FUNCTIECONTEXT
\"\"\"{role_text.strip()}\"\"\"

VOORBEELDEN (alleen ter inspiratie/bias; kies wat inhoudelijk past)
{examples_block}

OPDRACHT
Formuleer resultaatgebieden en cluster ze in thema’s **volgens de spelregels in de system prompt**.
Lever uitsluitend het JSON-object in het opgegeven formaat.
"""

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

    # Renderer houdt A/E/M altijd aan de allowed list (niet wat het model terugstuurt)
    markdown = _render_markdown_from_struct(data, allowed_themes)
    return markdown


# ── UI (titel + omschrijving + filters) ──────────────────────────────
with st.form("ra_form"):
    role_title = st.text_input("Functietitel", placeholder="Bijv. Accountmanager B2B SaaS")
    role_desc = st.text_area(
        "Functieomschrijving",
        height=180,
        placeholder="Beschrijf kort de scope, taken en verantwoordelijkheden…"
    )
    k = 3
    # Single-select dropdowns
    ORG_TYPES = ["(alle)", "profit", "nonprofit"]
    SECTORS   = ["(alle)",
        "Tech/IT", "Vastgoed", "Zorg & Welzijn", "Onderwijs", "Overheid", "Retail & Horeca", "Energie & Duurzaamheid", "Landbouw & Voeding",
        "Industrie & Productie", "Logistiek & Transport", "Financiele Dienstverlening", "Bouw & Installatie", "Zakelijke Dienstverlening"
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

        # Allowed list = inline themes (no Word docx)
        all_themes = THEMES_ALLOWED

        # LLM genereert eerst RAs, clustert daarna in thema's, en levert JSON
        markdown = generate_result_areas(
            role_title, role_desc, examples,
            allowed_themes=all_themes,
            language="nl",
            org_type=chosen_org,
            sector=chosen_sector
        )

        # Sla resultaat op in session_state zodat het niet verdwijnt bij een rerun
        st.session_state["ra_markdown"] = markdown
        st.session_state["ra_title"] = role_title
        st.session_state["ra_role_desc"] = role_desc

        # Init editor alleen als er nog niets staat of als de brontekst veranderd is
        if (
            "ra_markdown_source" not in st.session_state
            or st.session_state["ra_markdown_source"] != markdown
        ):
            st.session_state["ra_edited_md"] = markdown
            st.session_state["ra_markdown_source"] = markdown

    # Opgehaalde voorbeelden (tabel) – dit mag binnen 'if submitted:' blijven
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

# Toon resultaat + editor + PDF op basis van state (blijft staan bij klikken)
if "ra_markdown" in st.session_state:
    st.markdown("### Resultaat")
    st.markdown(st.session_state["ra_markdown"], unsafe_allow_html=False)

    st.markdown("### Bewerk hier resultaat indien gewenst")
    st.text_area(
        "Pas de tekst aan (dit is wat er in de PDF komt):",
        key="ra_edited_md",
        height=350
    )

    import hashlib
    pdf_input = st.session_state.get("ra_edited_md", "")
    pdf_bytes = build_pdf_bytes(
        title=f"Resultaatgebieden — {st.session_state.get('ra_title') or 'Onbekende functie'}",
        role_desc=(
            f"Functietitel: {st.session_state.get('ra_title','')}"
            f"\n\nOmschrijving: {st.session_state.get('ra_role_desc','')}"
        ),
        md_content=pdf_input,  # always the edited text
    )

    # Force a fresh button whenever text changes
    pdf_key = "download_pdf_" + hashlib.md5(pdf_input.encode("utf-8")).hexdigest()[:8]

    st.download_button(
        label="📄 Download PDF",
        data=pdf_bytes,
        file_name=f"resultaatgebieden_{re.sub(r'[^a-zA-Z0-9_-]+','_', st.session_state.get('ra_title') or 'functie')}.pdf",
        mime="application/pdf",
        use_container_width=True,
        key=pdf_key,
    )














