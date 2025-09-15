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

from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime


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
}

def _is_level_header(line: str) -> str | None:
    norm = _normalize_txt(line)
    for key, label in LEVEL_HEADERS.items():
        if norm == key or norm.startswith(key):
            return label
    return None

@st.cache_resource
def load_themes_from_docx(path: str | Path = THEMES_PATH) -> List[Dict]:
    """
    Inline-only parser for themes like:
      Relatiebeheer (A: 75-100, E: 25-50, M: 0-25)
    grouped under level headers:
      Strategisch / Tactisch / Operationeel
    """
    try:
        from docx import Document
    except ImportError:
        st.error("Voeg `python-docx` toe aan requirements.txt om thema’s uit Word te lezen.")
        return []

    path = Path(path)
    if not path.exists():
        st.error(f"Thema-bestand niet gevonden op {path}. Zet het .docx in je repo.")
        return []

    doc = Document(str(path))

    # Gather all lines (paragraphs + tables)
    lines: List[str] = []
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

    # Inline pattern: "<name> (A: ..., E: ..., M: ...)"
    inline_re = re.compile(
        r"^(?P<name>.+?)\s*\(\s*A\s*:\s*(?P<A>[^,]+?),\s*E\s*:\s*(?P<E>[^,]+?),\s*M\s*:\s*(?P<M>[^)]+?)\s*\)\s*$",
        flags=re.IGNORECASE
    )

    themes: List[Dict] = []
    seen: set[str] = set()
    current_level: Optional[str] = None

    for raw in lines:
        s = raw.strip()
        if not s:
            continue

        # Level header?
        lvl = _is_level_header(s)
        if lvl:
            current_level = lvl
            continue

        # Inline theme?
        m = inline_re.match(s)
        if not m:
            continue  # ignore everything that isn't an inline theme row

        name = m.group("name").strip()
        A = m.group("A").strip().replace("–", "-").replace("—", "-")
        E = m.group("E").strip().replace("–", "-").replace("—", "-")
        M = m.group("M").strip().replace("–", "-").replace("—", "-")

        rec = {"name": name, "A": A, "E": E, "M": M}
        if current_level:
            rec["level"] = current_level

        key = f"{_normalize_txt(name)}||{rec.get('level','')}"
        if key not in seen:
            seen.add(key)
            themes.append(rec)

    return themes


# ── Prompt build (LLM selects themes from allowed list) ───────────────
def build_system_msg(allowed_themes: List[Dict]) -> str:
    def fmt_theme(t):
        name = t.get("name","").strip()
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

Doel:
- Kies **2–6 thema's** die **het beste aansluiten** bij de functiecontext (functietitel + beschrijving), rekening houdend met sector en organisatietype (indien gegeven) en met de opgehaalde voorbeelden.
- **Gebruik de themanaam en A/E/M-buckets exact zoals hieronder weergegeven**. Geen alternatieve of geschatte waarden.
- Per gekozen thema: geef **2–4 resultaatgebieden** (elk in **exact één zin**, wat+waarom) en een **korte rationale** (één zinnetje) waarom dit thema past.

Selectierichtlijnen:
- Kies eerst thema's die de **kern van het werk** in de beschrijving weergeven.
- Gebruik voorbeelden als **ondersteuning** bij de keuze van deze thema's.
- Zorg voor **complementariteit**: de gekozen thema’s moeten samen de rol dekken, niet elkaars duplicaat zijn.


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
      "reason": "<korte rationale waarom dit thema past>",
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

        # OVERRULE: altijd A/E/M uit Word gebruiken (niet uit model)
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
            lines.append("**AEM-Cube positie:** " + ", ".join(aem_bits))
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
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setTitle(title or "Resultaatgebieden")
    c.setAuthor("Resultaatgebieden (Generator)")

    # Page geometry
    left = 2.0 * cm
    right = width - 2.0 * cm
    top = height - 2.0 * cm
    bottom = 2.0 * cm

    base_size = 11
    line_height = base_size * 1.2

    # --- Logo (top-right) ---
    logo_height = 0
    try:
        logo_path = "AEM-Cube_Poster3_HI_Logo.png"
        logo_width = 3.5 * cm
        logo_height = 3.5 * cm
        c.drawImage(
            logo_path,
            width - logo_width - 2*cm,     # right margin
            height - logo_height - 2*cm,   # top margin
            width=logo_width,
            height=logo_height,
            preserveAspectRatio=True,
            mask="auto",
        )
    except Exception:
        # If missing, just skip; y will start at top
        logo_height = 0

    # Start y below the logo to avoid overlap
    y = top - (logo_height + 1*cm if logo_height else 0)

    # --- Wrapped writer helper (uses enclosing y) ---
    def write_wrapped(text: str, size: int = base_size, indent: float = 0, font_name: str = None):
        nonlocal y
        font_name = font_name or PDF_FONT
        c.setFont(font_name, size)
        max_width = right - (left + indent)

        words = text.split(" ")
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            if c.stringWidth(test, font_name, size) > max_width and line:
                c.drawString(left + indent, y, line)
                y -= line_height
                if y < bottom:
                    c.showPage()
                    # Optionally draw logo again on new pages; comment out if not desired
                    try:
                        if logo_height:
                            c.drawImage(
                                logo_path,
                                width - logo_width - 2*cm,
                                height - logo_height - 2*cm,
                                width=logo_width,
                                height=logo_height,
                                preserveAspectRatio=True,
                                mask="auto",
                            )
                    except Exception:
                        pass
                    y = top - (logo_height + 1*cm if logo_height else 0)
                    c.setFont(font_name, size)
                line = w
            else:
                line = test

        # last segment
        if line or text == "":
            c.drawString(left + indent, y, line)
            y -= line_height
            if y < bottom:
                c.showPage()
                # repeat logo on new page if present
                try:
                    if logo_height:
                        c.drawImage(
                            logo_path,
                            width - logo_width - 2*cm,
                            height - logo_height - 2*cm,
                            width=logo_width,
                            height=logo_height,
                            preserveAspectRatio=True,
                            mask="auto",
                        )
                except Exception:
                    pass
                y = top - (logo_height + 1*cm if logo_height else 0)

    # --- Header ---
    c.setFont(PDF_FONT_BOLD, 16)
    c.drawString(left, y, (title or "Resultaatgebieden"))
    y -= line_height

    c.setFont(PDF_FONT, 9)
    c.drawString(left, y, f"Aangemaakt: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= line_height * 1.2

    # --- Role context (optional) ---
    if role_desc and role_desc.strip():
        c.setFont(PDF_FONT_BOLD, 12)
        c.drawString(left, y, "Functiecontext")
        y -= line_height
        for l in _markdown_to_plain_lines(role_desc):
            write_wrapped(l, size=10, font_name=PDF_FONT)
        y -= line_height * 0.5

    # --- Body from markdown ---
    lines = _markdown_to_plain_lines(md_content)
    bullet_indent = 0.6 * cm

    for l in lines:
        if l.startswith("Thema:"):
            # TRUE BOLD for theme lines
            write_wrapped(l, size=12, font_name=PDF_FONT_BOLD)
            continue
        if l.startswith("- "):
            write_wrapped("• " + l[2:], size=11, indent=bullet_indent, font_name=PDF_FONT)
        else:
            write_wrapped(l, size=11, font_name=PDF_FONT)

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

OPDRACHT (VOLG STRENG DEZE REGELS)
- Kies in totaal **3–4 thema’s** **uitsluitend** uit ALLOWED_THEMES.
- **Neem minstens 2 thema’s** op uit de lijst REQUIRED_THEMES (indien aanwezig).
- Gebruik de **themanaam exact** zoals in ALLOWED_THEMES.
- Gebruik per thema de **A/E/M-buckets exact** zoals in ALLOWED_THEMES (geen alternatieve waarden).
- Per gekozen thema: **2–4 resultaatgebieden**, elk in **exact één zin** (wat + waarom).
- Voeg per gekozen thema een **korte rationale** toe: waarom past dit thema bij deze functiecontext?
- **Introducéér geen nieuwe thema’s** en verander geen thema-namen.

ALLEEN DIT JSON-OBJECT RETOURNEREN (geen extra tekst):
{{
  "themes": [
    {{
      "name": "<exacte themanaam uit ALLOWED_THEMES>",
      "A": "<exact uit ALLOWED_THEMES>",
      "E": "<exact uit ALLOWED_THEMES>",
      "M": "<exact uit ALLOWED_THEMES>",
      "reason": "<korte rationale (1 zin)>",
      "result_areas": [
        "<één zin (wat + waarom)>",
        "<…>"
      ]
    }}
  ]
}}
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

    # NB: geef de VOLLEDIGE allowed dicts mee, zodat de renderer canonieke A/E/M kan afdwingen:
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

        # ⬇️ NEW: PDF download button
    try:
        pdf_bytes = build_pdf_bytes(
            title=f"Resultaatgebieden — {role_title.strip() or 'Onbekende functie'}",
            role_desc=f"Functietitel: {role_title}\n\nOmschrijving: {role_desc}",
            md_content=markdown,
        )
        st.download_button(
            label="📄 Download als PDF",
            data=pdf_bytes,
            file_name=f"resultaatgebieden_{re.sub(r'[^a-zA-Z0-9_-]+','_', role_title or 'functie')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"Kon PDF niet genereren: {e}")

    

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
















