# ======================================
# Motor de Análisis con IA (ai_analyzer.py)
# ======================================
import os
import re
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Callable, Set
from collections import Counter
import pandas as pd
from openai import OpenAI
from rapidfuzz import fuzz
from unidecode import unidecode

logger = logging.getLogger("ai_analyzer")

FORBIDDEN_TRAILING_WORDS = {
    "de", "del", "la", "el", "los", "las", "en", "para", "por", "con", "a", "al",
    "y", "o", "u", "e", "un", "una", "unos", "unas", "su", "sus", "sobre", "tras",
    "hacia", "desde", "sin", "que", "se"
}

MIN_SUBTEMA_WORDS = 4
MAX_SUBTEMA_WORDS = 7

# Longest first so "mencion del" wins over "mencion de" (which also matches "mencion del…").
FORBIDDEN_SUBTEMA_PREFIXES = [
    "mencion de la", "mencion del", "mencion de", "mencion a", "mencion en",
    "presencia de", "declaraciones de", "noticia sobre", "alusion a", "referencia a",
    "entrevista con", "entrevista al", "entrevista a",
]

# PKL tema labels that must not leak into Subtema_IA.
PKL_TEMA_BLEED_TOKENS = {
    "mencion", "entrevista", "opinion", "columna", "editorial", "noticia",
    "presencia", "alusion", "referencia", "declaraciones", "cobertura", "nota",
}

_LEAD_NARRATION_RE = re.compile(
    r"^(?:(?:la|el|los|las)\s+)?"
    r"(?:universidad|institucion|instituci[oó]n|clinica|cl[ií]nica|hospital|"
    r"fundacion|fundaci[oó]n|entidad|empresa|colegio|marca)\s+"
    r"(?:anunci[oó]|inaugur[oó]|present[oó]|inform[oó]|indic[oó]|dijo|"
    r"declar[oó]|revel[oó]|confirm[oó]|destac[oó]|explic[oó]|lanzo|lanz[oó])\s+",
    re.IGNORECASE,
)

DISCOURSE_STARTERS = {
    "de", "del", "ese", "esa", "esos", "esas", "este", "esta", "estos", "estas",
    "aquel", "aquella", "un", "una", "unos", "unas", "el", "la", "los", "las",
    "que", "se", "su", "sus",
}

LEAD_ACTION_VERBS = {
    "salio", "dijo", "hizo", "fue", "era", "eran", "hubo", "hay", "tiene",
    "llego", "paso", "conto", "vivia", "nacio", "crecio", "tuvo",
    "anuncio", "inauguro", "presento", "informo", "indico", "declaro",
    "revelo", "confirmo", "destaco", "explico", "lanzo",
}

INST_HEADS = {
    "universidad", "universidades", "fundacion", "clinica", "hospital",
    "colegio", "instituto", "institucion",
}

DEGREE_LEMMAS = {
    "abogado", "abogada", "medico", "medica", "ingeniero", "ingeniera",
    "licenciado", "licenciada", "profesional", "egresado", "egresada",
    "graduado", "graduada", "estudiante", "magister", "maestria", "doctorado",
    "doctorada", "contador", "contadora", "arquitecto", "arquitecta",
    "enfermero", "enfermera", "psicologo", "psicologa",
}

EVENT_FACT_TOKENS = {
    "feria", "congreso", "premio", "award", "awards", "patrimonio", "matricula",
    "inscripcion", "catedra", "beca", "becas", "diplomado", "foro", "simposio",
    "festival", "convocatoria", "inauguracion", "sede", "cine", "conversatorio",
    "alianza", "convenio",
}

FACT_ANCHORS = INST_HEADS | DEGREE_LEMMAS | {
    "formacion", "academica", "academico", "beca", "becas", "posgrado",
    "pregrado", "sede", "convenio", "alianza", "dialogo", "inauguracion",
    "nombramiento", "rector", "rectora", "egresado", "carrera",
} | EVENT_FACT_TOKENS

GENERIC_PARTICIPATION = {
    "participacion", "presencia", "apoyo", "apoya", "apoyan", "actividad",
    "mencion", "gestion", "acompanamiento", "nota", "cobertura",
}

INST_SPAN_BREAKERS = {
    "abre", "abren", "apoya", "apoyan", "participa", "participan", "participo",
    "organiza", "organizaron", "envia", "enviaron", "realiza", "convoca",
    "inaugura", "inauguran", "recibe", "reciben", "gana", "ganan",
    "sera", "comienza", "comienzan", "presenta", "presentan",
    "lanza", "lanzan", "reporta", "reportan",
}

_MEDIA_SUFFIX_RE = re.compile(
    r"\s*[-|–—:/]\s*(?:noticias?\s+\w+|video|en vivo|fotos?|imagenes?|portada)\s*$",
    re.IGNORECASE,
)

CITY_TAILS = {
    "barranquilla", "bogota", "cali", "medellin", "cartagena", "bucaramanga",
    "pereira", "manizales", "cucuta", "ibague", "neiva", "pasto", "armenia",
    "villavicencio", "valledupar", "monteria", "sincelejo", "popayan",
    "tunja", "riohacha", "quibdo", "quindio",
}

GENERIC_CLUSTER_STOP = INST_HEADS | CITY_TAILS | GENERIC_PARTICIPATION | {
    "noticia", "vital", "colombia", "nacional", "regional", "programa",
    "anuncia", "presente", "nuevo", "nueva", "hoy", "ano", "anos",
}

_DEGREE_RE = re.compile(
    r"\b(abogad[oa]s?|m[eé]dic[oa]s?|ingenier[oa]s?|licenciado[as]?|profesional(?:es)?|"
    r"egresad[oa]s?|graduad[oa]s?|estudiante[s]?|mag[ií]ster|maestr[ií]as?|"
    r"doctorad[oa]s?|contad[oa]r(?:es)?|arquitect[oa]s?|enfermer[oa]s?|"
    r"psic[oó]log[oa]s?)\b",
    re.I,
)
_EDU_CUE_RE = re.compile(
    r"\b(se\s+hizo|estudi[oó]|estudio|egres[oó]|se\s+gradu[oó]|se\s+form[oó]|"
    r"curs[oó]|formaci[oó]n|acad[eé]mic[oa]|pregrado|posgrado|carrera)\b",
    re.I,
)

STOPWORDS_ES = {
    "de", "del", "la", "el", "los", "las", "en", "para", "por", "con", "a", "al",
    "y", "o", "u", "e", "un", "una", "unos", "unas", "sobre", "tras", "este", "esta",
    "estos", "estas", "fue", "fueron", "era", "eran", "como", "mas", "pero", "sus",
    "que", "se", "ha", "han", "hay", "les", "nos", "son"
}

INSTITUTIONAL_PREFIXES = [
    "fundacion", "clinica", "hospital", "universidad", "instituto", "institucion",
    "colegio", "banco", "aerolinea", "empresa", "grupo", "corporacion", "alcaldia",
    "gobernacion", "ministerio", "centro", "complejo", "organizacion", "sociedad",
    "asociacion", "proyecto", "urbanizacion"
]

def clean_text_strictly_no_links(text: str) -> str:
    """Elimina URLs (http, https, www), diccionarios y la palabra 'Link'."""
    if not text:
        return ""
    if isinstance(text, dict):
        val = text.get("value", "")
        text = str(val)

    s = str(text).strip()
    if s.lower() in ("nan", "none", "null", "link", "link nota", "ver nota"):
        return ""

    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"www\.\S+", "", s)
    s = re.sub(r"\b(?:http|https)://\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    if s.lower() in ("link", "link nota", ""):
        return ""
    return s

def normalize_text_for_matching(text: str) -> str:
    if not text:
        return ""
    t = unidecode(str(text).lower().strip())
    t = re.sub(r"^(?:imagenes|en imagenes|fotos|en fotos|video|en video|en vivo)\s*\|\s*", "", t)
    words = re.findall(r"\b[a-z0-9]+\b", t)
    
    stemmed = []
    for w in words:
        if w in STOPWORDS_ES or len(w) < 2:
            continue
        if w.endswith("ces") and len(w) > 4:
            w = w[:-3] + "z"
        elif w.endswith("es") and len(w) > 4:
            w = w[:-2]
        elif w.endswith("s") and not w.endswith("is") and len(w) > 3:
            w = w[:-1]
        stemmed.append(w)
        
    return " ".join(stemmed)

def get_content_words_set(text_norm: str) -> Set[str]:
    return {w for w in text_norm.split() if len(w) > 2 and w not in STOPWORDS_ES}

def get_lead_content_words(text_norm: str, n_words: int = 3) -> Tuple[str, ...]:
    words = [w for w in text_norm.split() if len(w) > 2 and w not in STOPWORDS_ES]
    return tuple(words[:n_words])


def brand_content_tokens(brand: str, aliases: Optional[List[str]] = None) -> Set[str]:
    """Content tokens of the client name; must not count as 'same story' evidence."""
    toks: Set[str] = set()
    for item in [brand] + list(aliases or []):
        if not str(item).strip():
            continue
        toks |= get_content_words_set(normalize_text_for_matching(str(item)))
        compact = unidecode(str(item).lower().strip())
        compact = re.sub(r"\s+", "", compact)
        if 2 <= len(compact) <= 8:
            toks.add(compact)
        for acr in re.findall(r"\(([a-z0-9]{2,6})\)", unidecode(str(item).lower())):
            toks.add(acr)
    return toks


def collect_run_marca_names(
    brand: str,
    aliases: Optional[List[str]] = None,
    rows: Optional[List[dict]] = None,
    km: Optional[dict] = None,
) -> List[str]:
    """Marca de Empresas Consulta + alias + Menciones - Empresa del corrido (cualquier cliente)."""
    names: List[str] = []
    seen: Set[str] = set()

    def _add(val: object) -> None:
        s = str(val or "").strip()
        if not s or s.lower() in {"nan", "none", "-", "null"}:
            return
        key = unidecode(s.lower())
        if key in seen:
            return
        seen.add(key)
        names.append(s)

    _add(brand)
    for alias in aliases or []:
        _add(alias)
    menciones_key = (km or {}).get("menciones", "Menciones - Empresa")
    for row in rows or []:
        raw = row.get(menciones_key) or row.get("Menciones - Empresa") or ""
        for part in re.split(r"[;|/]", str(raw)):
            _add(part)
    return names


def collect_run_marca_tokens(
    brand: str,
    aliases: Optional[List[str]] = None,
    rows: Optional[List[dict]] = None,
    km: Optional[dict] = None,
) -> Set[str]:
    names = collect_run_marca_names(brand, aliases, rows, km)
    extra = names[1:] if names else list(aliases or [])
    return brand_content_tokens(names[0] if names else brand, extra)


def _without_brand_tokens(norm_text: str, brand_toks: Set[str]) -> str:
    if not brand_toks:
        return norm_text
    return " ".join(w for w in (norm_text or "").split() if w not in brand_toks)


def _strip_media_suffix(title: str) -> str:
    """Drop outlet tails ('- NOTICIAS VITAL') and ordinal marks that split otherwise-equal titles."""
    t = str(title or "").strip()
    t = t.replace("ª", "a").replace("º", "o")
    prev = None
    while prev != t:
        prev = t
        t = _MEDIA_SUFFIX_RE.sub("", t).strip()
    return t


def _title_compact(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", unidecode(_strip_media_suffix(title).lower()))


def _stem_content_token(w: str) -> str:
    w = unidecode(str(w).lower())
    if w.endswith("ces") and len(w) > 4:
        return w[:-3] + "z"
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("is") and len(w) > 3:
        return w[:-1]
    return w


def _has_echoed_content_pair(words: List[str]) -> bool:
    """True for 'Beneficios y beneficios…' — consecutive repeated content lemmas."""
    content: List[str] = []
    for w in words:
        low = unidecode(w.lower())
        if low in STOPWORDS_ES or len(low) < 3:
            continue
        content.append(_stem_content_token(w))
    for i in range(len(content) - 1):
        if content[i] == content[i + 1] and len(content[i]) > 3:
            return True
    return False


_VAGUE_TEMPLATE_RES = (
    re.compile(r"^como aliad", re.I),
    re.compile(
        r"^participa(?:cion)?(?: de)?(?: la)? en (?:un |una |el |la )?(?:evento|jornada|encuentro|foro)s?\b",
        re.I,
    ),
    re.compile(r"^realiza(?:cion)? (?:un |una |el |la )?(?:encuentro|evento|jornada)s?\b", re.I),
    re.compile(r"^analiza medidas\b", re.I),
    re.compile(r"^anuncia nuevos programas\b", re.I),
    re.compile(r"^apoya(?:n)? en \w+$", re.I),
    re.compile(r"^participa en\b", re.I),
)


def _alias_short_forms(marca_toks: Set[str]) -> List[str]:
    return sorted((t for t in marca_toks if 2 <= len(t) <= 6), key=len, reverse=True)


def _strip_concatenated_alias(token: str, shorts: List[str]) -> str:
    raw = unidecode(token.lower())
    for alias in shorts:
        if len(raw) <= len(alias) + 2:
            continue
        if raw.endswith(alias):
            rest = token[: len(token) - len(alias)]
            if rest and rest[-1] in "-_":
                rest = rest[:-1]
            if len(unidecode(rest)) >= 3:
                return rest
        if raw.startswith(alias):
            rest = token[len(alias) :]
            if rest and rest[0] in "-_":
                rest = rest[1:]
            if len(unidecode(rest)) >= 3:
                return rest
    return token


def _drop_marca_words(words: List[str], marca_toks: Set[str]) -> List[str]:
    """Remove client name / alias / short form tokens from a subtema (any client)."""
    if not words or not marca_toks:
        return words
    shorts = _alias_short_forms(marca_toks)
    kept: List[str] = []
    for w in words:
        parts = re.split(r"([-_/])", w)
        if len(parts) > 1:
            rebuilt: List[str] = []
            for part in parts:
                if part in "-_/":
                    if rebuilt and rebuilt[-1] not in "-_/":
                        rebuilt.append(part)
                    continue
                stem = _stem_content_token(part)
                raw = unidecode(part.lower())
                if stem in marca_toks or raw in marca_toks:
                    continue
                rebuilt.append(_strip_concatenated_alias(part, shorts))
            token = "".join(rebuilt).strip("-_/")
            if token:
                kept.append(token)
            continue
        stem = _stem_content_token(w)
        raw = unidecode(w.lower())
        if stem in marca_toks or raw in marca_toks:
            continue
        stripped = _strip_concatenated_alias(w, shorts)
        stem2 = _stem_content_token(stripped)
        if stem2 in marca_toks or unidecode(stripped.lower()) in marca_toks:
            continue
        kept.append(stripped)
    return kept


def _is_vague_brand_template(phrase: str) -> bool:
    """True for leftover scraps after stripping the client ('como aliada', 'participa en jornada')."""
    if not phrase or not str(phrase).strip():
        return True
    low = unidecode(phrase.strip().lower())
    for rx in _VAGUE_TEMPLATE_RES:
        if rx.search(low):
            return True
    words = [w for w in str(phrase).split() if w]
    content = [w for w in words if unidecode(w.lower()) not in STOPWORDS_ES]
    return len(content) < 2


def _is_brand_led_template(phrase: str, marca_toks: Set[str]) -> bool:
    """Reject 'X como aliada', 'X participa en jornada', 'X realiza encuentro'."""
    if not phrase:
        return False
    words = [w for w in str(phrase).split() if w]
    i = 0
    stripped_lead = False
    while i < len(words):
        stem = _stem_content_token(words[i])
        raw = unidecode(words[i].lower())
        if stem in marca_toks or raw in marca_toks or raw in INST_HEADS:
            stripped_lead = True
            i += 1
            while i < len(words) and unidecode(words[i].lower()) in STOPWORDS_ES | DISCOURSE_STARTERS:
                i += 1
            continue
        break
    rest = " ".join(words[i:])
    if stripped_lead and _is_vague_brand_template(rest):
        return True
    return False


def _distinctive_story_tokens(words) -> Set[str]:
    """Tokens that can evidence 'same story' — never marca/city/generic participation."""
    return {w for w in words if len(w) >= 4 and w not in GENERIC_CLUSTER_STOP}


def _subtema_fact_tokens(subtema: str, brand: str, aliases: Optional[List[str]] = None) -> Set[str]:
    toks = get_content_words_set(normalize_text_for_matching(subtema))
    toks -= brand_content_tokens(brand, aliases)
    toks -= CITY_TAILS
    toks -= GENERIC_PARTICIPATION
    return {t for t in toks if len(t) >= 4}


_EVENT_SYNONYMS = {
    "premio": {"award", "awards"},
    "award": {"premio"},
    "awards": {"premio"},
}


def _token_in_context(tok: str, ctx_toks: Set[str]) -> bool:
    if tok in ctx_toks:
        return True
    return bool(_EVENT_SYNONYMS.get(tok, set()) & ctx_toks)


def subtema_supported_by_context(
    subtema: str,
    ctx: str,
    brand: str,
    aliases: Optional[List[str]] = None,
) -> bool:
    """True when the subtema’s fact words are present in THIS contexto (not another story)."""
    if not subtema or not ctx or str(ctx).strip() in ("", "-", "nan", "None"):
        return False
    if _is_institution_name_chain(subtema, brand):
        return False
    facts = _subtema_fact_tokens(subtema, brand, aliases)
    if not facts:
        return False
    ctx_toks = get_content_words_set(normalize_text_for_matching(ctx))
    foreign_events = {
        t for t in (facts & EVENT_FACT_TOKENS) if not _token_in_context(t, ctx_toks)
    }
    if foreign_events:
        return False
    overlap = {t for t in facts if _token_in_context(t, ctx_toks)}
    if overlap:
        return True
    # Template educational noun phrases (formación académica…) are entailed by degree/egreso cues.
    return facts <= {"formacion", "academica", "academico", "actividad"}


def _subtema_compatible_with_story(
    subtema: str,
    story: str,
    brand: str,
    aliases: Optional[List[str]] = None,
) -> bool:
    """Per-row backup: keep cluster labels unless this note is a different event."""
    if _is_institution_name_chain(subtema, brand):
        return False
    if not subtema or not story or str(story).strip() in ("", "-", "nan", "None"):
        return False
    facts = _subtema_fact_tokens(subtema, brand, aliases)
    if not facts:
        return False
    story_toks = get_content_words_set(normalize_text_for_matching(story))
    sub_events = facts & EVENT_FACT_TOKENS
    if sub_events and (story_toks & EVENT_FACT_TOKENS) and not any(
        _token_in_context(t, story_toks) for t in sub_events
    ):
        return False
    if facts and not any(_token_in_context(t, story_toks) for t in facts):
        return False
    return True


def extract_event_anchor(title_raw: str) -> str:
    if not title_raw:
        return ""
    t = _strip_media_suffix(title_raw)
    parts = re.split(r"\s*[:|-]\s*", t, 1)
    if len(parts) > 1 and len(parts[0].strip()) >= 10:
        return normalize_text_for_matching(parts[0])
    return ""

def generate_brand_variants(brand: str, aliases: List[str]) -> List[str]:
    raw_inputs = [brand] + [a for a in aliases if a.strip()]
    variants_set = set()

    for item in raw_inputs:
        base = unidecode(item.lower().strip())
        if not base:
            continue
        variants_set.add(base)

        acronym_match = re.search(r"\(([a-z0-9]{2,6})\)", base)
        if acronym_match:
            acronym = acronym_match.group(1)
            variants_set.add(acronym)
            variants_set.add(r"\b" + r"\.?\s*".join(list(acronym)) + r"\.?\b")
            base = re.sub(r"\([a-z0-9]{2,6}\)", "", base).strip()
            variants_set.add(base)

        if len(base) <= 5 and base.isalpha():
            variants_set.add(r"\b" + r"\.?\s*".join(list(base)) + r"\.?\b")
            continue

        if "santa fe" in base:
            variants_set.add(base.replace("santa fe", "santafe"))
            variants_set.add("santa fe")
            variants_set.add("santafe")
            variants_set.add("clinica santa fe")
            variants_set.add("hospital santa fe")
            variants_set.add("fundacion santa fe")

        if "serena del mar" in base:
            variants_set.add("serena")
            variants_set.add("hospital serena")
            variants_set.add("hospital serena del mar")
            variants_set.add("clinica serena")
            variants_set.add("clinica serena del mar")

        for prefix in ["fundacion", "clinica", "hospital", "universidad", "instituto", "asociacion"]:
            if base.startswith(prefix + " "):
                core = base[len(prefix):].strip()
                if len(core) >= 4:
                    variants_set.add(core)
                    for alt_p in ["clinica", "hospital", "fundacion", "centro"]:
                        variants_set.add(f"{alt_p} {core}")

    sorted_variants = sorted(list(variants_set), key=lambda x: len(x), reverse=True)
    compiled_regexes = []
    for v in sorted_variants:
        if v.startswith(r"\b"):
            compiled_regexes.append(v)
        else:
            compiled_regexes.append(rf"\b{re.escape(v)}\b")
            
    return compiled_regexes

def extract_brand_context(resumen: str, titulo: str, brand_regexes: List[str]) -> str:
    """Extrae las oraciones del Resumen y Título sin links ni etiquetas."""
    t_clean = clean_text_strictly_no_links(titulo)
    r_clean = clean_text_strictly_no_links(resumen)
    
    r_norm = unidecode(r_clean.lower())
    t_norm = unidecode(t_clean.lower())
    
    matched_sentences = []
    
    if r_clean:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', r_clean) if s.strip()]
        for idx, s in enumerate(sentences):
            s_clean_sub = clean_text_strictly_no_links(s)
            if not s_clean_sub:
                continue
            s_norm = unidecode(s_clean_sub.lower())
            if any(re.search(rx, s_norm) for rx in brand_regexes):
                block = s_clean_sub
                if len(s_clean_sub.split()) < 10 and idx + 1 < len(sentences):
                    next_s = clean_text_strictly_no_links(sentences[idx + 1])
                    if next_s:
                        block = f"{s_clean_sub} {next_s}"
                if block not in matched_sentences:
                    matched_sentences.append(block)

        if not matched_sentences:
            for rx in brand_regexes:
                for m in re.finditer(rx, r_norm):
                    start = max(0, m.start() - 120)
                    end = min(len(r_clean), m.end() + 150)
                    snippet = clean_text_strictly_no_links(r_clean[start:end])
                    if snippet and snippet not in matched_sentences:
                        matched_sentences.append(f"...{snippet}..." if start > 0 else snippet)
                    if len(matched_sentences) >= 2:
                        break
                if matched_sentences:
                    break

    title_matches = any(re.search(rx, t_norm) for rx in brand_regexes)

    if matched_sentences:
        resumen_context = " ".join(matched_sentences).strip()
        if title_matches and t_clean and t_clean.lower() not in resumen_context.lower():
            res = f"{t_clean}. {resumen_context}"
        else:
            res = resumen_context
        return clean_text_strictly_no_links(res)[:800]

    if title_matches:
        if r_clean:
            res = f"{t_clean}. {r_clean[:380]}"
        else:
            res = t_clean
        return clean_text_strictly_no_links(res)[:800]

    if t_clean and r_clean:
        res = f"{t_clean}. {r_clean[:400]}"
    else:
        res = t_clean or r_clean[:500]
    return clean_text_strictly_no_links(res)[:800]

def check_exact_byline_rule(text: str, brand: str, aliases: List[str]) -> bool:
    """
    REGLA LITERAL SOLICITADA:
    Si el texto contiene exactamente las frases de autoría indicadas:
    - 'Editora web y periodista egresada de [marca]'
    - 'Editor web y periodista egresado de [marca]'
    - 'Estudiante en formación [marca]'
    - 'Periodista egresado/a de [marca]'
    Se retorna True para asignar directamente Neutro, Estudiantes, Redacción de artículo.
    """
    if not text:
        return False
        
    t_norm = unidecode(str(text).lower())
    
    # Términos de búsqueda (marca y todos los alias)
    targets = [unidecode(brand.lower().strip())] + [unidecode(a.lower().strip()) for a in aliases if a.strip()]
    
    for tgt in targets:
        if not tgt:
            continue
        tgt_esc = re.escape(tgt)
        
        # 1. Editora web y periodista egresada de [marca]
        if re.search(rf"\beditora\s+web\s+y\s+periodista\s+egresada\s+(?:de\s+(?:la\s+)?)?{tgt_esc}\b", t_norm):
            return True
            
        # 2. Editor web y periodista egresado de [marca]
        if re.search(rf"\beditor\s+web\s+y\s+periodista\s+egresado\s+(?:de\s+(?:la\s+)?)?{tgt_esc}\b", t_norm):
            return True
            
        # 3. Estudiante en formación [marca] (con o sin 'de' / 'de la')
        if re.search(rf"\bestudiante\s+en\s+formacion\s+(?:de\s+(?:la\s+)?)?{tgt_esc}\b", t_norm):
            return True
            
        # 4. Periodista egresado/a de [marca] / Editor(a) egresado/a de [marca]
        if re.search(rf"\b(?:periodista|editor[a]?|redactor[a]?)\s+egresad[oa]\s+(?:de\s+(?:la\s+)?)?{tgt_esc}\b", t_norm):
            return True

    return False

def _tokenize_phrase_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñÜü0-9]+", str(text or ""))


def _strip_forbidden_subtema_prefixes(text: str) -> str:
    res = (text or "").strip()
    changed = True
    while res and changed:
        changed = False
        res_norm = unidecode(res.lower())
        for fs in FORBIDDEN_SUBTEMA_PREFIXES:
            if re.match(rf"^{re.escape(fs)}\b", res_norm):
                # Map prefix length on the normalized string back to original split.
                prefix_n = len(fs.split())
                words = res.split()
                res = " ".join(words[prefix_n:]).strip()
                changed = True
                break
    return res


def _strip_tema_echo_prefix(text: str, tema: str) -> str:
    if not text or not tema:
        return text
    words = text.split()
    tema_words = _tokenize_phrase_words(tema)
    if not tema_words or len(words) <= len(tema_words):
        return text
    n = len(tema_words)
    head = unidecode(" ".join(words[:n]).lower())
    tema_norm = unidecode(" ".join(tema_words).lower())
    if head == tema_norm:
        return " ".join(words[n:]).strip()
    return text


def _is_brand_restatement(phrase: str, brand: str) -> bool:
    """True when the phrase names the client (or a generic 'participación de…') without a news fact."""
    if not phrase or not brand:
        return False
    frase_toks = _content_token_set(phrase)
    brand_toks = get_content_words_set(normalize_text_for_matching(brand))
    leftover = set(frase_toks) - brand_toks
    leftover -= GENERIC_PARTICIPATION
    leftover -= INST_HEADS
    leftover -= CITY_TAILS
    leftover -= {w for w in leftover if w in STOPWORDS_ES or len(w) < 3}
    fact = leftover & (FACT_ANCHORS | EVENT_FACT_TOKENS | DEGREE_LEMMAS)
    if fact:
        return False
    return len(leftover) <= 1


def _context_has_event_fact(text: str) -> bool:
    toks = _content_token_set(text)
    return bool(toks & EVENT_FACT_TOKENS)


def _normalize_subtema_phrase(
    text: str,
    brand: str,
    tema: str = "",
    aliases: Optional[List[str]] = None,
) -> str:
    if not text:
        return ""
    clean = re.sub(r'[,.;:!?¿¡"\'\(\)\[\]\{\}\-_/\\|]', " ", str(text))
    words = [w for w in clean.split() if w]
    if _has_echoed_content_pair(words):
        return ""
    marca_toks = brand_content_tokens(brand, aliases)
    if _is_brand_led_template(" ".join(words), marca_toks):
        return ""
    words = _drop_marca_words(words, marca_toks)
    words = _fit_to_max_words(words)
    res = _strip_forbidden_subtema_prefixes(" ".join(words).strip())
    res = _strip_tema_echo_prefix(res, tema)
    words = _drop_marca_words([w for w in res.split() if w], marca_toks)
    words = _fit_to_max_words(words)
    if _has_echoed_content_pair(words):
        return ""
    res = " ".join(words).strip()
    if not res or _is_vague_brand_template(res):
        return ""
    brand_words = set(re.findall(r"\b[a-z0-9]+\b", unidecode(brand.lower())))
    res_words = set(re.findall(r"\b[a-z0-9]+\b", unidecode(res.lower())))
    if res_words.issubset(brand_words | marca_toks):
        return ""
    if _is_brand_restatement(res, brand):
        return ""
    if _is_institution_name_chain(res, brand):
        return ""
    if unidecode(res.lower()) in {
        "universidad", "autonoma", "fundacion", "clinica", "hospital",
        "institucion", "asociacion", "mencion", "entrevista", "noticia",
    }:
        return ""
    if len(res.split()) < MIN_SUBTEMA_WORDS:
        return ""
    return res.capitalize()


def _fit_to_max_words(words: List[str]) -> List[str]:
    words = [w for w in words if w]
    while words and words[-1].lower() in FORBIDDEN_TRAILING_WORDS:
        words.pop()
    while len(words) > MAX_SUBTEMA_WORDS:
        if len(words) >= 2 and unidecode(words[-2].lower()) in {"de", "del", "en", "por"}:
            words = words[:-2]
            continue
        if unidecode(words[0].lower()) in STOPWORDS_ES or unidecode(words[0].lower()) in DISCOURSE_STARTERS:
            words = words[1:]
            continue
        break
    if len(words) > MAX_SUBTEMA_WORDS:
        words = words[:MAX_SUBTEMA_WORDS]
    while words and words[-1].lower() in FORBIDDEN_TRAILING_WORDS:
        words.pop()
    return words


def _is_name_continuation(token: str) -> bool:
    if not token:
        return False
    low = unidecode(token.lower())
    if low in STOPWORDS_ES or low in DISCOURSE_STARTERS or low in LEAD_ACTION_VERBS:
        return False
    if token[0].isupper():
        return True
    return token[0].isalpha() and low not in CITY_TAILS


def _complete_proper_names_from_context(phrase: str, ctx: str) -> str:
    """If a phrase ends mid proper name (Simón / Universidad Simón), finish it from contexto."""
    if not phrase or not ctx:
        return phrase
    pwords = phrase.split()
    if pwords and unidecode(pwords[-1].lower()) in CITY_TAILS:
        return phrase
    cwords = _tokenize_phrase_words(ctx)
    if not pwords or not cwords:
        return phrase
    last = unidecode(pwords[-1].lower())
    phrase_lows = {unidecode(w.lower()) for w in pwords}
    for i, src in enumerate(cwords):
        if unidecode(src.lower()) != last:
            continue
        j = i + 1
        extra: List[str] = []
        while j < len(cwords) and _is_name_continuation(cwords[j]):
            low = unidecode(cwords[j].lower())
            if low in phrase_lows:
                break
            extra.append(cwords[j])
            j += 1
        if extra:
            fitted = _fit_to_max_words(pwords + extra)
            # Prefer dropping a leading filler rather than chopping the added surname.
            while (
                len(fitted) == MAX_SUBTEMA_WORDS
                and extra
                and unidecode(fitted[-1].lower()) != unidecode(extra[-1].lower())
                and unidecode(fitted[0].lower()) in STOPWORDS_ES | DISCOURSE_STARTERS
            ):
                fitted = _fit_to_max_words(fitted[1:] + extra[-1:])
            return " ".join(fitted)
        break
    return phrase


def _content_token_set(text: str) -> Set[str]:
    return {w for w in normalize_text_for_matching(text).split() if w}


def _is_title_scrap(phrase: str, title: str) -> bool:
    """True when the phrase is just the titular cropped to N words, not a distinct fact."""
    if not phrase or not title:
        return False
    phrase_words = [unidecode(w.lower()) for w in phrase.split()]
    title_words = [unidecode(w.lower()) for w in _tokenize_phrase_words(title)]
    n = len(phrase_words)
    if n < MIN_SUBTEMA_WORDS or len(title_words) < n:
        return False
    if phrase_words == title_words[:n]:
        return True
    np = normalize_text_for_matching(phrase)
    nt_lead = normalize_text_for_matching(" ".join(_tokenize_phrase_words(title)[:n]))
    return bool(np) and np == nt_lead


def _has_fact_anchor(words: List[str]) -> bool:
    lows = [unidecode(w.lower()) for w in words]
    for lw in lows:
        if lw in FACT_ANCHORS:
            return True
        if any(lw.startswith(h) for h in INST_HEADS):
            return True
    return False


def _is_lead_clause_scrap(words: List[str]) -> bool:
    """Reject 'De ese barrio salió…' style openers without an institutional fact."""
    if not words:
        return True
    lows = [unidecode(w.lower()) for w in words]
    start = lows[0]
    has_anchor = _has_fact_anchor(words)
    has_lead_verb = any(v in lows for v in {unidecode(x) for x in LEAD_ACTION_VERBS})
    if start in DISCOURSE_STARTERS and not has_anchor:
        return True
    if start in DISCOURSE_STARTERS and has_lead_verb and not any(
        lw in INST_HEADS or lw in DEGREE_LEMMAS for lw in lows
    ):
        return True
    return False


def _story_text(title: str, ctx: str) -> str:
    """Title + contexto of THIS note; the fact often lives in the titular."""
    t = str(title or "").strip()
    c = str(ctx or "").strip()
    if c in ("", "-", "nan", "None"):
        return t
    if t and t.lower() not in c.lower():
        return f"{t}. {c}"
    return c or t


def _is_inst_marker_token(tok: str) -> bool:
    if tok in INST_HEADS:
        return True
    prefixes = {unidecode(p) for p in INSTITUTIONAL_PREFIXES}
    if tok in prefixes:
        return True
    return tok.startswith("universit")


def _is_institution_name_chain(phrase: str, brand: str) -> bool:
    """True for 'Libre seccional Cartagena la Fundación Universitaria Minuto' — org list, no event."""
    if not phrase:
        return False
    toks = _content_token_set(phrase)
    toks -= brand_content_tokens(brand)
    if toks & EVENT_FACT_TOKENS:
        return False
    inst_n = sum(1 for t in toks if _is_inst_marker_token(t))
    glue = toks & {"seccional", "filial", "seccion"}
    if inst_n >= 2:
        return True
    if inst_n >= 1 and glue:
        return True
    return False


def _is_strong_subtema(
    phrase: str,
    tema: str,
    title: str,
    brand: str,
    aliases: Optional[List[str]] = None,
) -> bool:
    if not phrase:
        return False
    words = phrase.split()
    if len(words) < MIN_SUBTEMA_WORDS or len(words) > MAX_SUBTEMA_WORDS:
        return False
    if _has_echoed_content_pair(words):
        return False
    if _is_vague_brand_template(phrase):
        return False
    if _is_lead_clause_scrap(words):
        return False
    if _is_brand_restatement(phrase, brand):
        return False
    if _is_institution_name_chain(phrase, brand):
        return False
    if _labels_too_close(tema, phrase):
        return False
    frase_toks = _content_token_set(phrase)
    tema_toks = _content_token_set(tema)
    if tema_toks and frase_toks and len(frase_toks & tema_toks) / len(frase_toks) >= 0.55:
        return False
    bleed = {unidecode(t) for t in PKL_TEMA_BLEED_TOKENS}
    if frase_toks and sum(1 for t in frase_toks if t in bleed) / len(frase_toks) >= 0.4:
        return False
    if unidecode(phrase.lower()) in {
        "hecho informativo", "gestion institucional", "gestión institucional",
    }:
        return False
    marca_toks = brand_content_tokens(brand, aliases)
    if frase_toks and (frase_toks & marca_toks):
        return False
    if frase_toks and frase_toks.issubset(marca_toks):
        return False
    # Event-named titles ("Feria Educativa Inspírate…") are the fact, not scrap.
    if _is_title_scrap(phrase, title) and not (_content_token_set(phrase) & EVENT_FACT_TOKENS):
        return False
    return True


def _looks_like_name_part(token: str) -> bool:
    if not token:
        return False
    low = unidecode(token.lower())
    if low in LEAD_ACTION_VERBS or low in INST_SPAN_BREAKERS:
        return False
    if low in DISCOURSE_STARTERS or low in STOPWORDS_ES:
        return False
    if token[0].isupper():
        return True
    return token[0].isalpha()


def _institution_span(words: List[str], start_idx: int) -> List[str]:
    if start_idx >= len(words):
        return []
    if start_idx + 1 < len(words):
        nxt = unidecode(words[start_idx + 1].lower())
        if nxt in LEAD_ACTION_VERBS or nxt in INST_SPAN_BREAKERS:
            return []
    span = [words[start_idx]]
    j = start_idx + 1
    while j < len(words):
        w = words[j]
        low = unidecode(w.lower())
        if low in {"de", "del", "la", "las", "los", "el"}:
            if j + 1 < len(words) and _looks_like_name_part(words[j + 1]):
                span.append(w)
                j += 1
                continue
            break
        if _looks_like_name_part(w):
            span.append(w)
            j += 1
            continue
        break
    return span if len(span) >= 2 else []


def _split_inst_core_loc(inst: List[str]) -> Tuple[List[str], List[str]]:
    lows = [unidecode(w.lower()) for w in inst]
    for i in range(1, len(inst) - 1):
        if lows[i] in {"de", "del"} and lows[i + 1] in CITY_TAILS:
            return inst[:i], inst[i:]
    return inst, []


def _join_prefix_and_inst(prefix: List[str], inst_core: List[str]) -> List[str]:
    prefix = list(prefix)
    inst_core = list(inst_core)
    while prefix and len(prefix) + len(inst_core) > MAX_SUBTEMA_WORDS:
        last_low = unidecode(prefix[-1].lower())
        if last_low in STOPWORDS_ES or last_low in DISCOURSE_STARTERS or last_low == "academica":
            prefix.pop()
            continue
        if unidecode(prefix[0].lower()) in STOPWORDS_ES or unidecode(prefix[0].lower()) in DISCOURSE_STARTERS:
            prefix.pop(0)
            continue
        break
    return _fit_to_max_words(prefix + inst_core)


def _education_fact_phrase(ctx: str, brand: str) -> str:
    words = _tokenize_phrase_words(ctx)
    lows = [unidecode(w.lower()) for w in words]
    inst_idx = next((i for i, lw in enumerate(lows) if lw in INST_HEADS), None)
    if inst_idx is None:
        return ""
    inst = _institution_span(words, inst_idx)
    if not inst:
        return ""
    inst_core, _loc = _split_inst_core_loc(inst)
    if not inst_core:
        return ""
    has_degree = _DEGREE_RE.search(ctx or "") is not None
    has_edu_cue = has_degree or _EDU_CUE_RE.search(ctx or "") is not None
    if _context_has_event_fact(ctx):
        return ""
    if has_edu_cue:
        degree_m = _DEGREE_RE.search(ctx or "")
        if degree_m:
            return _normalize_subtema_phrase(f"Formación académica de {degree_m.group(1)}", brand)
        return _normalize_subtema_phrase("Formación académica profesional aplicada", brand)
    return ""


def _score_subtema_window(words: List[str], tema: str, title: str, brand: str, at_sentence_start: bool = False) -> int:
    if _is_lead_clause_scrap(words):
        return -120
    if _has_echoed_content_pair(words):
        return -110
    phrase = " ".join(words)
    if _is_institution_name_chain(phrase, brand):
        return -100
    if _is_brand_restatement(phrase, brand):
        return -90
    low = unidecode(phrase.lower())
    if any(re.match(rf"^{re.escape(fs)}\b", low) for fs in FORBIDDEN_SUBTEMA_PREFIXES):
        return -100
    content = [
        w for w in words
        if unidecode(w.lower()) not in STOPWORDS_ES and len(unidecode(w.lower())) > 2
    ]
    if len(content) < 2:
        return -60
    if _labels_too_close(tema, phrase):
        return -80
    tema_toks = _content_token_set(tema)
    phrase_toks = _content_token_set(phrase)
    if tema_toks and phrase_toks and len(phrase_toks & tema_toks) / len(phrase_toks) >= 0.5:
        return -45
    bleed = {unidecode(t) for t in PKL_TEMA_BLEED_TOKENS}
    bleed_n = sum(1 for t in phrase_toks if t in bleed)
    event_in_phrase = bool(phrase_toks & EVENT_FACT_TOKENS)
    title_pen = 30 if _is_title_scrap(phrase, title) and not event_in_phrase else 0
    noun_bonus = 0
    title_toks = get_content_words_set(normalize_text_for_matching(title))
    title_events = title_toks & EVENT_FACT_TOKENS
    lead_events = set(get_lead_content_words(normalize_text_for_matching(title), n_words=4)) & EVENT_FACT_TOKENS
    if phrase_toks & title_events:
        noun_bonus += 28
    if lead_events and not (phrase_toks & lead_events):
        noun_bonus -= 55
    shared_dist = phrase_toks & _distinctive_story_tokens(title_toks)
    if shared_dist:
        noun_bonus += 8 * min(3, len(shared_dist))
    inst_bonus = 4 if lead_events else 18
    for w in content:
        wl = unidecode(w.lower())
        if wl.endswith(("cion", "sion", "miento", "dad", "aje", "ncia", "encia", "ura", "azgo")):
            noun_bonus += 8
        if wl in EVENT_FACT_TOKENS:
            noun_bonus += 22
        elif wl in FACT_ANCHORS or wl in DEGREE_LEMMAS:
            noun_bonus += 18
        elif wl in INST_HEADS:
            noun_bonus += inst_bonus
    start = unidecode(words[0].lower())
    start_pen = -25 if start in DISCOURSE_STARTERS else (-8 if start in STOPWORDS_ES else 6)
    lead_pen = 40 if at_sentence_start and start in DISCOURSE_STARTERS else 0
    verb_pen = 0
    for w in words:
        wl = unidecode(w.lower())
        if wl in LEAD_ACTION_VERBS or wl in INST_SPAN_BREAKERS:
            verb_pen += 50
    brand_words = set(re.findall(r"\b[a-z0-9]+\b", unidecode(brand.lower())))
    brand_toks = get_content_words_set(normalize_text_for_matching(brand)) | brand_words
    brand_pen = 20 if phrase_toks and phrase_toks.issubset(brand_toks) else 0
    if phrase_toks and brand_toks:
        brand_frac = len(phrase_toks & brand_toks) / max(1, len(phrase_toks))
        if brand_frac >= 0.5:
            brand_pen += 28
    return (
        12 * len(content)
        + noun_bonus
        + start_pen
        - title_pen
        - lead_pen
        - verb_pen
        - (12 * bleed_n)
        - brand_pen
    )


def _noun_phrase_from_context(
    ctx: str,
    tema: str,
    title: str,
    brand: str,
    aliases: Optional[List[str]] = None,
) -> str:
    """Build a fact noun phrase from THIS note; never a lead-clause scrap.

    If the contexto already names an event, stay inside it (cine ≠ title-only
    crop). If it is only a participant list, the fact often lives in the titular.
    """
    ctx_clean = str(ctx or "").strip()
    if _context_has_event_fact(ctx_clean):
        text = ctx_clean
    else:
        text = _story_text(title, ctx_clean)
    if not text or text == "-":
        return ""
    edu = _education_fact_phrase(text, brand)
    if edu and not _is_lead_clause_scrap(edu.split()) and not _labels_too_close(tema, edu):
        edu = _complete_proper_names_from_context(edu, text)
        return _normalize_subtema_phrase(edu, brand, tema, aliases)

    stripped = _LEAD_NARRATION_RE.sub("", text)
    stripped = re.sub(
        r"^(?:[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚáéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚáéíóúñü]+){0,5})\s+"
        r"(?:anunci[oó]|inaugur[oó]|present[oó]|inform[oó]|dijo|declar[oó]|lanzo|lanz[oó])\s+",
        "",
        stripped,
        flags=re.IGNORECASE,
    )
    words = _tokenize_phrase_words(stripped)
    if len(words) < MIN_SUBTEMA_WORDS:
        return ""
    best = ""
    best_score = -1
    orig_lead = [unidecode(w.lower()) for w in _tokenize_phrase_words(text)[:3]]
    for n in range(MAX_SUBTEMA_WORDS, MIN_SUBTEMA_WORDS - 1, -1):
        for i in range(0, len(words) - n + 1):
            window = words[i:i + n]
            if _is_institution_name_chain(" ".join(window), brand):
                continue
            at_start = [unidecode(w.lower()) for w in window[:2]] == orig_lead[:2]
            score = _score_subtema_window(window, tema, title, brand, at_sentence_start=at_start)
            if score > best_score:
                best_score = score
                best = " ".join(window)
    if best_score < 0 or not best:
        return edu
    completed = _complete_proper_names_from_context(best, text)
    return _normalize_subtema_phrase(completed, brand, tema, aliases)


def clean_subtema(
    text: str,
    brand: str,
    title_fallback: str,
    aliases: Optional[List[str]] = None,
) -> str:
    if not text:
        return _fallback_from_title(title_fallback)
    res = _normalize_subtema_phrase(text, brand, aliases=aliases)
    if not res:
        return _fallback_from_title(title_fallback)
    return res


def clean_subtema_specific(
    text: str,
    brand: str,
    tema: str = "",
    aliases: Optional[List[str]] = None,
) -> str:
    """PKL-tema path: never collapse to a title scrap or a 1–2 word leftover."""
    return _normalize_subtema_phrase(text, brand, tema, aliases)

def clean_tema(text: str) -> str:
    if not text:
        return "Gestión Institucional"
    clean = re.sub(r'[,.;:!?¿¡"\'\(\)\[\]\{\}\-_/\\|]', ' ', str(text)).strip()
    words = clean.split()[:4]
    res = " ".join(words).title()
    if res.lower() in ["otros", "otro", "general", "varios", "miscelanea", "sin clasificar", ""]:
        return "Gestión Institucional"
    return res

def ensure_different_tema_subtema(tema: str, subtema: str, ctx: str) -> str:
    t_clean = tema.strip().title()
    s_clean = subtema.strip().capitalize()
    
    if t_clean.lower() == s_clean.lower() or fuzz.ratio(t_clean.lower(), s_clean.lower()) >= 80:
        c_low = f"{s_clean} {ctx}".lower()
        if any(w in c_low for w in ["salud", "hospital", "clinica", "medico", "medicina", "paciente", "quirurg", "enfermedad"]):
            return "Sector Salud"
        if any(w in c_low for w in ["aduan", "dian", "fiscal", "tributar", "impuesto", "arancel"]):
            return "Gestión Tributaria"
        if any(w in c_low for w in ["universidad", "estudiante", "academ", "carrera", "educacion", "profesor", "beca", "feria"]):
            return "Educación Superior"
        if any(w in c_low for w in ["aniversario", "celebracion", "decadas", "anos", "reconocimiento", "homenaje"]):
            return "Hitos y Aniversarios"
        if any(w in c_low for w in ["rescate", "bombero", "emergencia", "siniestro", "accidente", "desastre"]):
            return "Gestión de Emergencias"
        if any(w in c_low for w in ["obra", "construccion", "via", "infraestructura", "puente", "sede"]):
            return "Infraestructura"
        if any(w in c_low for w in ["seguridad", "policia", "captura", "hurto", "delito", "fiscalia", "crimen"]):
            return "Seguridad Ciudadana"
        if any(w in c_low for w in ["convenio", "acuerdo", "alianza", "gremio", "liderazgo"]):
            return "Relaciones Gremiales"
        return "Gestión Institucional"
        
    return t_clean

def check_positive_institutional_override(
    ctx: str,
    brand: str = "",
    aliases: Optional[List[str]] = None,
) -> bool:
    """Positive tone only when this brand (not a third party) backs, celebrates or allies."""
    if not ctx:
        return False
    c_low = unidecode(ctx.lower())
    if brand:
        tokens = [unidecode(brand.lower().strip())] + [
            unidecode(a.lower().strip()) for a in (aliases or []) if a.strip()
        ]
        if tokens and not any(t and t in c_low for t in tokens):
            return False
    positive_actions = [
        "celebra y respalda", "respalda el nombramiento", "respaldan el nombramiento",
        "acompanamos desde", "acompanamiento desde", "asesoria gratuita", "apoyo gratuito",
        "pusieron en marcha", "pone en marcha", "felicita a", "felicitamos a",
        "rinde homenaje", "reconocimiento destaca el compromiso", "abren espacio"
    ]
    has_positive = any(p in c_low for p in positive_actions)
    has_negative_allegation = any(n in c_low for n in ["denuncia penal", "sancion fiscal", "investigacion por corrupcion", "plagio"])
    return has_positive and not has_negative_allegation


_LIST_MENTION_RE = re.compile(
    r"\b(participaron|participa(?:n|ron)?|asistieron|asistio|"
    r"entre ellas|entre las que|entre las instituciones|"
    r"junto a otras|otras universidades|otras instituciones|"
    r"invitad[oa]s|convocad[oa]s)\b",
    re.IGNORECASE,
)


def check_list_mention_neutral(
    ctx: str,
    brand: str,
    aliases: Optional[List[str]] = None,
) -> bool:
    """Marca named only in a participant list → Neutro unless praise/criticism of that marca."""
    if not ctx or not brand:
        return False
    c_low = unidecode(ctx.lower())
    if not _LIST_MENTION_RE.search(c_low):
        return False
    tokens = [unidecode(brand.lower().strip())] + [
        unidecode(a.lower().strip()) for a in (aliases or []) if a.strip()
    ]
    if not any(t and t in c_low for t in tokens):
        return False
    if c_low.count(",") < 1 and " y " not in c_low:
        return False
    if check_positive_institutional_override(ctx, brand, aliases):
        return False
    if any(n in c_low for n in ["denuncia", "escandalo", "corrupcion", "queja contra", "sancion a"]):
        return False
    return True


def brand_tone_instructions(brand: str, aliases: Optional[List[str]] = None, step: int = 1) -> str:
    """Tono = reputational impact on this client, never overall article mood."""
    alias_txt = ", ".join(a.strip() for a in (aliases or []) if a.strip()) or "ninguno"
    return (
        f'{step}. "tono": SOLO el impacto reputacional sobre el cliente "{brand}" '
        f'(alias: {alias_txt}). Valores: "Positivo", "Negativo" o "Neutro".\n'
        "   PROHIBIDO clasificar el sentimiento general de la noticia o el enojo hacia un tercero.\n"
        "   Si el artículo ataca a otro actor (gobierno, otra empresa, un particular) y la marca "
        "solo aparece como dato, sede, fuente o mención neutra/positiva, el tono de la MARCA "
        'es "Neutro" o "Positivo", nunca el ánimo del resto del texto.\n'
        "   Si la marca solo aparece en un LISTADO de participantes, invitados, universidades o "
        "instituciones, sin elogio ni crítica de ESA marca, el tono es estrictamente \"Neutro\".\n"
        "   REGLA DE ORO: si ESTE cliente expresa o recibe ACOMPAÑAMIENTO, RESPALDO, APOYO, "
        'FELICITACIONES, CELEBRACIÓN o ALIANZA dirigidos a la marca, el tono es "Positivo".'
    )


def brand_tone_examples(brand: str) -> str:
    b = brand or "la marca"
    return f"""
EJEMPLOS DE TONO (respecto a "{b}", no al clima de la noticia):
- Caso 1: "{b} acompaña a las familias afectadas por un sismo..."
  -> Tono: "Positivo" (solidaridad de la marca).
- Caso 2: "{b} celebra y respalda un nombramiento..."
  -> Tono: "Positivo" (respaldo institucional de la marca).
- Caso 3: "{b} y otra entidad abren asesoría gratuita..."
  -> Tono: "Positivo" (alianza de la marca).
- Caso 4: "Denuncian cobros excesivos de un TERCERO. {b} solo es la sede del evento."
  -> Tono: "Neutro" (el enojo no es reputacional para la marca).
- Caso 5: "Quejas por fallas atribuibles a {b}..."
  -> Tono: "Negativo" (afectación directa a la marca).
- Caso 6: "En el foro participaron la Universidad Nacional, {b} y otras instituciones."
  -> Tono: "Neutro" (la marca solo está en un listado de participantes).
- Caso 7: "Boletín de cifras donde {b} aporta un dato técnico..."
  -> Tono: "Neutro".
"""

def _fallback_from_title(title: str) -> str:
    if not title:
        return "Hecho Informativo"
    t = re.sub(r"^(?:imagenes|video|en fotos)\s*\|\s*", "", title, flags=re.IGNORECASE).strip()
    words = re.sub(r'[,.;:!?¿¡"\'\(\)\[\]\{\}\-_/\\|]', ' ', t).split()
    clean_words = words[:MAX_SUBTEMA_WORDS]
    while clean_words and clean_words[-1].lower() in FORBIDDEN_TRAILING_WORDS:
        clean_words.pop()
    return " ".join(clean_words).capitalize() if clean_words else "Hecho Informativo"

def cluster_similar_rows(
    rows: List[dict],
    km: dict,
    brand_regexes: List[str],
    brand: str = "",
    aliases: Optional[List[str]] = None,
) -> Dict[int, int]:
    """Group only near-duplicate titles of the same story. Prefer under-grouping.

    Brand/city tokens and shared body boilerplate must not glue different facts.
    """
    cluster_map = {}
    clusters_rep = {}
    current_cluster = 0
    brand_toks = brand_content_tokens(brand, aliases)

    active_indices = [i for i in range(len(rows)) if not rows[i].get("is_duplicate")]
    sorted_indices = sorted(
        active_indices,
        key=lambda idx: normalize_text_for_matching(
            _strip_media_suffix(str(rows[idx].get(km.get("titulo", "Título"), "")))
        )
    )

    for i in sorted_indices:
        t_raw = str(rows[i].get(km.get("titulo", "Título"), ""))
        t_stripped = _strip_media_suffix(t_raw)
        t_norm = normalize_text_for_matching(t_stripped)
        t_story = _without_brand_tokens(t_norm, brand_toks)
        c_words = get_content_words_set(t_story)
        lead_words = get_lead_content_words(t_story, n_words=3)
        anchor = extract_event_anchor(t_raw)
        if brand_toks and get_content_words_set(anchor).issubset(brand_toks):
            anchor = ""
        compact = _title_compact(t_raw)

        assigned = False
        incoming = {
            "title_compact": compact,
            "title_story": t_story,
            "content_words": c_words,
            "lead_words": lead_words,
            "anchor": anchor,
        }
        for cid, rep in clusters_rep.items():
            for mem in rep["members"]:
                if _titles_are_same_story(
                    compact,
                    mem["title_compact"],
                    t_story,
                    mem["title_story"],
                    c_words,
                    mem["content_words"],
                    lead_words,
                    mem["lead_words"],
                    anchor,
                    mem["anchor"],
                ):
                    cluster_map[i] = cid
                    rep["members"].append(incoming)
                    assigned = True
                    break
            if assigned:
                break

        if not assigned:
            cluster_map[i] = current_cluster
            clusters_rep[current_cluster] = {"members": [incoming]}
            current_cluster += 1

    return cluster_map


def _titles_are_same_story(
    compact: str,
    rep_compact: str,
    t_story: str,
    rep_story: str,
    c_words: Set[str],
    rep_words: Set[str],
    lead_words: Tuple[str, ...],
    rep_lead: Tuple[str, ...],
    anchor: str,
    rep_anchor: str,
) -> bool:
    """Near-duplicate / lightly rewritten titles of one event — not shared marca/city."""
    if compact and rep_compact:
        if compact == rep_compact:
            return True
        min_c = min(len(compact), len(rep_compact))
        if min_c >= 18 and (compact in rep_compact or rep_compact in compact):
            return True
        if min_c >= 18 and fuzz.ratio(compact, rep_compact) >= 92:
            return True

    if anchor and rep_anchor and len(anchor) >= 12 and anchor == rep_anchor:
        if _distinctive_story_tokens(get_content_words_set(anchor)):
            return True
    # One title uses a colon ("Feria Educativa Inspírate: …"); the other starts
    # with that same event name and a different suffix.
    for a, other in ((anchor, rep_story), (rep_anchor, t_story)):
        if (
            a
            and other
            and len(a) >= 12
            and _distinctive_story_tokens(get_content_words_set(a))
            and (other.startswith(a) or f" {a} " in f" {other} ")
        ):
            return True

    lead_dist = _distinctive_story_tokens(lead_words) | _distinctive_story_tokens(rep_lead)
    if len(lead_words) >= 3 and len(rep_lead) >= 3 and lead_words == rep_lead and lead_dist:
        return True
    # Shared distinctive 3-word stem anywhere in the other title (A→Z neighbors
    # with suffix drift: "…Inspírate 2026 impulsa…" vs "…Inspírate: 10 universidades…").
    def _lead_stem_in(story: str, lead: Tuple[str, ...]) -> bool:
        if len(lead) < 3 or not story:
            return False
        stem = " ".join(lead[:3])
        if len(stem) < 18:
            return False
        if not _distinctive_story_tokens(lead[:3]):
            return False
        return stem in story

    if _lead_stem_in(rep_story, lead_words) or _lead_stem_in(t_story, rep_lead):
        return True

    if len(lead_words) >= 2 and len(rep_lead) >= 2 and lead_words[:2] == rep_lead[:2]:
        combined_lead_len = len(" ".join(lead_words[:2]))
        lead2_dist = _distinctive_story_tokens(lead_words[:2])
        if combined_lead_len >= 13 and lead2_dist:
            return True

    overlap = c_words & rep_words
    distinctive = _distinctive_story_tokens(overlap)

    if t_story and rep_story:
        shorter, longer = (t_story, rep_story) if len(t_story) <= len(rep_story) else (rep_story, t_story)
        if (
            len(shorter) >= 18
            and shorter in longer
            and _distinctive_story_tokens(get_content_words_set(shorter))
        ):
            return True
        min_len = min(len(t_story), len(rep_story))
        if min_len >= 22 and t_story[:22] == rep_story[:22] and distinctive:
            return True

    if len(distinctive) >= 3:
        return True
    if len(overlap) >= 4 and len(distinctive) >= 2:
        return True
    rare = distinctive - DEGREE_LEMMAS - INST_HEADS
    if (
        len(distinctive) >= 2
        and rare
        and t_story
        and rep_story
        and min(len(t_story), len(rep_story)) >= 12
        and (
            fuzz.token_set_ratio(t_story, rep_story) >= 60
            or fuzz.partial_ratio(t_story, rep_story) >= 70
        )
    ):
        return True
    if (
        len(distinctive) >= 2
        and t_story
        and rep_story
        and min(len(t_story), len(rep_story)) >= 12
        and (
            fuzz.token_set_ratio(t_story, rep_story) >= 78
            or fuzz.partial_ratio(t_story, rep_story) >= 82
        )
    ):
        return True
    if (
        t_story
        and rep_story
        and min(len(t_story), len(rep_story)) >= 18
        and distinctive
        and fuzz.partial_ratio(t_story, rep_story) >= 90
        and fuzz.token_set_ratio(t_story, rep_story) >= 88
    ):
        return True
    return False

def canonicalize_subtopics(cluster_results: Dict[int, Tuple[str, str, str]]) -> Dict[int, Tuple[str, str, str]]:
    """Collapse near-identical subtema strings and share tema/tono inside that fact family."""
    subtemas_list = [sub for _, _, sub in cluster_results.values() if sub]
    counts = Counter(subtemas_list)
    unique_subs = list(counts.keys())

    mapping = {}
    for i in range(len(unique_subs)):
        s1 = unique_subs[i]
        for j in range(i + 1, len(unique_subs)):
            s2 = unique_subs[j]
            if _subtemas_near_duplicate(s1, s2):
                chosen = s1 if counts[s1] >= counts[s2] else s2
                mapping[s1] = chosen
                mapping[s2] = chosen

    staged: Dict[int, Tuple[str, str, str]] = {}
    by_sub: Dict[str, List[int]] = {}
    for cid, (tono, tema, sub) in cluster_results.items():
        canonical_sub = mapping.get(sub, sub)
        staged[cid] = (tono, tema, canonical_sub)
        by_sub.setdefault(canonical_sub, []).append(cid)

    final_results = {}
    for sub, cids in by_sub.items():
        chosen_tono = _majority_label([staged[c][0] for c in cids])
        chosen_tema = _majority_label([staged[c][1] for c in cids])
        for cid in cids:
            tono, tema, _ = staged[cid]
            final_results[cid] = (chosen_tono or tono, chosen_tema or tema, sub)
    return final_results


def _subtemas_near_duplicate(a: str, b: str) -> bool:
    if not a or not b:
        return False
    na = normalize_text_for_matching(a)
    nb = normalize_text_for_matching(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return fuzz.ratio(na, nb) >= 90 or (
        fuzz.token_set_ratio(na, nb) >= 90 and fuzz.token_sort_ratio(na, nb) >= 88
    )


def _majority_label(labels: List[str]) -> str:
    cleaned = [str(x).strip() for x in labels if str(x).strip()]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    generic = {"gestión institucional", "gestion institucional", "otros", "otro", "general", "-", ""}
    specific = [(lab, n) for lab, n in counts.items() if lab.lower() not in generic]
    pool = specific or list(counts.items())
    pool.sort(key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
    return pool[0][0]

def _labels_too_close(a: str, b: str) -> bool:
    if not a or not b:
        return False
    na = normalize_text_for_matching(a)
    nb = normalize_text_for_matching(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    return fuzz.ratio(na, nb) >= 80 or fuzz.token_set_ratio(na, nb) >= 85


def ensure_subtema_distinct_from_tema(
    tema: str,
    subtema: str,
    brand: str,
    title: str,
    ctx: str,
    aliases: Optional[List[str]] = None,
) -> str:
    """Keep a 4–7 word fact noun phrase from THIS note (title + contexto), distinct from the PKL tema."""
    story = _story_text(title, ctx)
    llm_clean = clean_subtema_specific(subtema or "", brand, tema, aliases)
    if llm_clean:
        llm_clean = _complete_proper_names_from_context(llm_clean, story)
        llm_clean = _normalize_subtema_phrase(llm_clean, brand, tema, aliases)
    if llm_clean and not subtema_supported_by_context(llm_clean, story, brand, aliases):
        llm_clean = ""

    ctx_phrase = _noun_phrase_from_context(ctx, tema, title, brand, aliases)
    if ctx_phrase:
        ctx_phrase = _complete_proper_names_from_context(ctx_phrase, story)
        ctx_phrase = _normalize_subtema_phrase(ctx_phrase, brand, tema, aliases)
    if ctx_phrase and not subtema_supported_by_context(ctx_phrase, story, brand, aliases):
        ctx_phrase = ""

    if llm_clean and _is_strong_subtema(llm_clean, tema, title, brand, aliases):
        return llm_clean
    if ctx_phrase and _is_strong_subtema(ctx_phrase, tema, title, brand, aliases):
        return ctx_phrase
    if ctx_phrase and not _is_lead_clause_scrap(ctx_phrase.split()) and not _labels_too_close(tema, ctx_phrase):
        if MIN_SUBTEMA_WORDS <= len(ctx_phrase.split()) <= MAX_SUBTEMA_WORDS:
            return ctx_phrase
    if llm_clean and not _is_lead_clause_scrap(llm_clean.split()) and not _labels_too_close(tema, llm_clean):
        if MIN_SUBTEMA_WORDS <= len(llm_clean.split()) <= MAX_SUBTEMA_WORDS:
            return llm_clean
    return (
        ctx_phrase
        or (
            llm_clean
            if llm_clean and not _is_lead_clause_scrap(llm_clean.split())
            else ""
        )
        or "Hecho informativo institucional"
    )


def _call_openai_cluster(
    client: OpenAI,
    model: str,
    brand: str,
    aliases: List[str],
    brand_regexes: List[str],
    ctx: str,
    title_ref: str,
    request_tone: bool = True,
    request_theme: bool = True,
    pkl_theme: Optional[str] = None,
) -> Tuple[str, str, str]:
    # REGLA EXACTA DE AUTORÍA/EGRESADOS (SI ESTÁN LAS PALABRAS NO SE ANALIZA CON IA)
    search_scope = f"{title_ref} {ctx}"
    if check_exact_byline_rule(search_scope, brand, aliases):
        return "Neutro", "Estudiantes", "Redacción de artículo"

    json_fields = []
    steps = []
    n = 1
    if request_tone:
        steps.append(brand_tone_instructions(brand, aliases, n))
        json_fields.append('"tono": "..."')
        n += 1
    if request_theme:
        steps.append(
            f'{n}. "tema": DOMINIO GENERAL (Nivel Macro, 1 a 3 palabras. Ej: "Educación Superior", "Gestión Tributaria", "Sector Salud"). PROHIBIDO "Otros".'
        )
        json_fields.append('"tema": "..."')
        n += 1

    subtema_rule = (
        f'{n}. "subtema": HECHO ESPECÍFICO: una sola frase nominal coherente en español colombiano, '
        "de 4 a 7 palabras (máximo 7), completa. "
        "Describe el hecho (feria, congreso, premio, matrícula, patrimonio, formación, rectoría), "
        "NUNCA el nombre, alias ni sigla del cliente. "
        'Sin comas ni puntos. PROHIBIDO "Mención", collage, recortar el titular o copiar el tema. '
        'PROHIBIDO repetir la misma palabra de contenido (MAL: "Beneficios y beneficios académicos"). '
        'PROHIBIDO plantillas con la marca (MAL: "X como aliada", "X participa en jornada", '
        '"X realiza encuentro", "X analiza medidas", "X anuncia nuevos programas"). '
        "PROHIBIDO encadenar nombres de instituciones participantes sin el hecho "
        "(feria, cátedra, alianza, becas); el subtema nombra el evento, no el listado."
    )
    steps.append(subtema_rule)
    json_fields.append('"subtema": "..."')

    if request_theme:
        differ_rule = 'REGLA OBLIGATORIA: "tema" y "subtema" DEBEN SER DIFERENTES.'
    elif pkl_theme:
        differ_rule = (
            f'TEMA YA CLASIFICADO POR EL MODELO DEL CLIENTE: "{pkl_theme}". '
            "NO inventes otro tema. NO copies ese tema ni lo parafrasees como subtema "
            '(MAL: tema "Entrevista" → subtema "Entrevista al rector"; '
            'MAL: "Mención"; MAL: recortar el inicio "Ese barrio salió primero un joven"). '
            "El subtema debe ser una frase nominal concreta de 4 a 7 palabras, "
            "distinta al tema y SIN el nombre del cliente "
            '(BIEN: "Formación académica de abogado"; MAL: incluir marca o sigla).'
        )
    else:
        differ_rule = "El subtema debe describir el hecho concreto, no un dominio general."

    tone_examples = ""
    if request_tone:
        tone_examples = brand_tone_examples(brand)

    prompt = f"""Analiza esta noticia para el cliente: "{brand}" (Alias: {', '.join(aliases) if aliases else 'Ninguno'}).
El tono se evalúa ÚNICAMENTE sobre este cliente y sus alias, no sobre el sentimiento general del artículo.

Titular de referencia: "{title_ref}"
Contexto analizado:
\"\"\"{ctx}\"\"\"
{tone_examples}
Instrucciones:
{chr(10).join(steps)}

{differ_rule}

Responde estrictamente en JSON:
{{{", ".join(json_fields)}}}"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Auditor reputacional de medios en Colombia. "
                        "El tono mide SOLO el impacto sobre el cliente y sus alias, "
                        "nunca el sentimiento general de la noticia. "
                        "El subtema es una frase nominal completa en español colombiano "
                        "anclada al hecho, no al nombre de la marca."
                    ),
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=180
        )
        data = json.loads(resp.choices[0].message.content)

        if request_tone:
            tono_raw = str(data.get("tono", "Neutro")).strip().capitalize()
            tono = tono_raw if tono_raw in ["Positivo", "Negativo", "Neutro"] else "Neutro"
            if check_positive_institutional_override(ctx, brand, aliases):
                tono = "Positivo"
            elif check_list_mention_neutral(ctx, brand, aliases):
                tono = "Neutro"
        else:
            tono = "Neutro"

        raw_sub = data.get("subtema", "")
        if pkl_theme and not request_theme:
            subtema = clean_subtema_specific(raw_sub, brand, pkl_theme, aliases)
        else:
            subtema = clean_subtema(raw_sub, brand, title_ref, aliases)

        if request_theme:
            tema = clean_tema(data.get("tema", ""))
            tema = ensure_different_tema_subtema(tema, subtema, ctx)
        else:
            tema = (pkl_theme or "").strip() or "Gestión Institucional"
            subtema = ensure_subtema_distinct_from_tema(tema, subtema, brand, title_ref, ctx, aliases)

        return tono, tema, subtema
    except Exception as e:
        logger.error(f"Error en llamada OpenAI: {e}")
        if request_theme:
            sub_fb = _fallback_from_title(title_ref)
            tema_fb = ensure_different_tema_subtema("Gestión Institucional", sub_fb, ctx)
        else:
            tema_fb = (pkl_theme or "").strip() or "Gestión Institucional"
            sub_fb = ensure_subtema_distinct_from_tema(tema_fb, "", brand, title_ref, ctx, aliases)
        tono_fb = "Positivo" if check_positive_institutional_override(ctx, brand, aliases) else "Neutro"
        if tono_fb != "Positivo" and check_list_mention_neutral(ctx, brand, aliases):
            tono_fb = "Neutro"
        return tono_fb, tema_fb, sub_fb


_AZ_NEIGHBOR_WINDOW = 12
_AZ_CTX_RATIO = 88


def _row_title(row: dict, km: dict) -> str:
    return str(row.get((km or {}).get("titulo", "Título"), "") or "")


def _row_ctx(row: dict) -> str:
    return str(row.get("Contexto analizado") or "")


def _az_sort_key(text: str) -> str:
    return unidecode(_strip_media_suffix(str(text or "")).strip().lower())


def _same_story_pair(
    row_a: dict,
    row_b: dict,
    km: dict,
    brand: str,
    aliases: Optional[List[str]] = None,
) -> bool:
    """Same news fact only — shared city/marca is not enough."""
    brand_toks = brand_content_tokens(brand, aliases)
    ta = _row_title(row_a, km)
    tb = _row_title(row_b, km)
    a_norm = _without_brand_tokens(normalize_text_for_matching(_strip_media_suffix(ta)), brand_toks)
    b_norm = _without_brand_tokens(normalize_text_for_matching(_strip_media_suffix(tb)), brand_toks)
    if _titles_are_same_story(
        _title_compact(ta),
        _title_compact(tb),
        a_norm,
        b_norm,
        get_content_words_set(a_norm),
        get_content_words_set(b_norm),
        get_lead_content_words(a_norm, 3),
        get_lead_content_words(b_norm, 3),
        extract_event_anchor(ta),
        extract_event_anchor(tb),
    ):
        return True
    ca = _without_brand_tokens(normalize_text_for_matching(_row_ctx(row_a)[:400]), brand_toks)
    cb = _without_brand_tokens(normalize_text_for_matching(_row_ctx(row_b)[:400]), brand_toks)
    if not ca or not cb:
        return False
    wa, wb = get_content_words_set(ca), get_content_words_set(cb)
    ev_a, ev_b = wa & EVENT_FACT_TOKENS, wb & EVENT_FACT_TOKENS
    if ev_a and ev_b and ev_a.isdisjoint(ev_b):
        return False
    rare = _distinctive_story_tokens(wa & wb) - DEGREE_LEMMAS - INST_HEADS
    return bool(rare) and len(rare) >= 2 and fuzz.token_set_ratio(ca, cb) >= _AZ_CTX_RATIO


def _best_component_labels(
    rows: List[dict],
    members: List[int],
    km: dict,
    brand: str,
    aliases: Optional[List[str]] = None,
) -> Optional[Tuple[str, str, str]]:
    scored = []
    marca = brand_content_tokens(brand, aliases)
    for i in members:
        row = rows[i]
        tono = str(row.get("Tono_IA") or "")
        tema = str(row.get("Tema_IA") or "")
        sub = str(row.get("Subtema_IA") or "")
        if not sub or sub == "-" or tono in {"Duplicada", "No aplica"}:
            continue
        title = _row_title(row, km)
        ctx = _row_ctx(row)
        ok = subtema_supported_by_context(sub, _story_text(title, ctx), brand, aliases)
        n = len(sub.split())
        marca_hit = bool(_content_token_set(sub) & marca)
        score = (2 if ok else 0) + (1 if MIN_SUBTEMA_WORDS <= n <= MAX_SUBTEMA_WORDS else 0)
        score += 0 if marca_hit else 2
        score += 0 if _has_echoed_content_pair(sub.split()) else 1
        scored.append((score, n, i, tono, tema, sub))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    _, _, _, tono, tema, sub = scored[0]
    return tono, tema, sub


def apply_marca_free_subtemas(
    rows: List[dict],
    km: dict,
    brand: str,
    aliases: Optional[List[str]] = None,
) -> List[dict]:
    """Final strip of marca/alias tokens; regenerate if the leftover is vague."""
    marca_toks = collect_run_marca_tokens(brand, aliases, rows, km)
    for row in rows:
        if row.get("is_duplicate") or str(row.get("Tono_IA") or "") in {"Duplicada", "No aplica"}:
            continue
        sub = str(row.get("Subtema_IA") or "")
        if not sub or sub == "-":
            continue
        if unidecode(sub.lower()) in {
            "redaccion de articulo",
            "hecho informativo institucional",
            "hecho informativo",
        }:
            continue
        title = _row_title(row, km)
        ctx = _row_ctx(row)
        tema = str(row.get("Tema_IA") or "")
        cleaned = _normalize_subtema_phrase(sub, brand, tema, aliases)
        words = _drop_marca_words((cleaned or sub).split(), marca_toks)
        phrase = " ".join(_fit_to_max_words(words)).strip()
        if phrase:
            phrase = phrase[:1].upper() + phrase[1:]
        weak = (
            not phrase
            or _is_vague_brand_template(phrase)
            or _has_echoed_content_pair(phrase.split())
            or len(phrase.split()) < MIN_SUBTEMA_WORDS
            or bool(_content_token_set(phrase) & marca_toks)
        )
        if weak:
            phrase = ensure_subtema_distinct_from_tema(tema, "", brand, title, ctx, aliases)
            words = _drop_marca_words((phrase or "").split(), marca_toks)
            phrase = " ".join(_fit_to_max_words(words)).strip()
            if phrase:
                phrase = phrase[:1].upper() + phrase[1:]
        if phrase:
            row["Subtema_IA"] = phrase
    return rows


def reconcile_az_same_story_labels(
    rows: List[dict],
    km: dict,
    brand: str,
    aliases: Optional[List[str]] = None,
) -> List[dict]:
    """Human-analyst pass: Título A→Z and Contexto A→Z. Does not touch Duplicada / IDs."""
    run_aliases = collect_run_marca_names(brand, aliases, rows, km)
    extra = [n for n in run_aliases if unidecode(n.lower()) != unidecode(str(brand or "").lower())]
    alias_all = list(aliases or []) + extra
    active = [
        i
        for i, row in enumerate(rows)
        if not row.get("is_duplicate") and str(row.get("Tono_IA") or "") not in {"Duplicada", "No aplica"}
    ]
    if len(active) >= 2:
        parent = {i: i for i in active}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        def walk(ordered: List[int]) -> None:
            n = len(ordered)
            for pos, i in enumerate(ordered):
                for jpos in range(pos + 1, min(n, pos + 1 + _AZ_NEIGHBOR_WINDOW)):
                    j = ordered[jpos]
                    if _same_story_pair(rows[i], rows[j], km, brand, alias_all):
                        union(i, j)

        walk(sorted(active, key=lambda i: _az_sort_key(_row_title(rows[i], km))))
        walk(sorted(active, key=lambda i: _az_sort_key(_row_ctx(rows[i]))))

        groups: Dict[int, List[int]] = {}
        for i in active:
            groups.setdefault(find(i), []).append(i)
        for members in groups.values():
            if len(members) < 2:
                continue
            best = _best_component_labels(rows, members, km, brand, alias_all)
            if not best:
                continue
            tono, tema, sub = best
            for i in members:
                rows[i]["Tono_IA"] = tono
                rows[i]["Tema_IA"] = tema
                rows[i]["Subtema_IA"] = sub

    return apply_marca_free_subtemas(rows, km, brand, alias_all)


def enrich_rows_with_ai(
    rows: List[dict],
    km: dict,
    brand: str,
    aliases: List[str],
    api_key: str,
    model: str = "gpt-4.1-nano-2025-04-14",
    progress_callback: Optional[Callable[[int, str], None]] = None,
    tone_model=None,
    theme_model=None,
) -> List[dict]:
    from pkl_classifier import classification_plan, format_theme_label, map_tone_label, _safe_predict

    client = OpenAI(api_key=api_key)
    plan = classification_plan(True, tone_model, theme_model)

    brand_regexes = generate_brand_variants(brand, aliases)
    
    if progress_callback:
        progress_callback(71, "Extrayendo contexto de la marca y sus variantes para auditoría…")
    for row in rows:
        if row.get("is_duplicate"):
            row["Contexto analizado"] = "-"
        else:
            resumen_val = row.get("Resumen - Aclaracion") or row.get("resumen corto") or row.get("Resumen") or ""
            titulo_val = row.get(km.get("titulo", "Título")) or ""
            
            ctx = extract_brand_context(
                str(resumen_val),
                str(titulo_val),
                brand_regexes
            )
            row["Contexto analizado"] = ctx

    if progress_callback:
        progress_callback(74, "Agrupando eventos y noticias similares (ordenamiento por titular)…")
    cluster_map = cluster_similar_rows(
        rows, km, brand_regexes, brand=brand, aliases=aliases
    )
    
    unique_clusters = sorted(set(cluster_map.values()))
    total_clusters = len(unique_clusters)
    
    cluster_to_sample_idx = {}
    for row_idx, cid in cluster_map.items():
        if cid not in cluster_to_sample_idx:
            cluster_to_sample_idx[cid] = row_idx
            
    cluster_results: Dict[int, Tuple[str, str, str]] = {}
    cluster_pkl: Dict[int, Tuple[Optional[str], Optional[str]]] = {}

    if progress_callback:
        progress_callback(77, f"Analizando {total_clusters} hechos únicos con {model}…")
        
    completed = 0
    with ThreadPoolExecutor(max_workers=14) as executor:
        future_to_cid = {}
        for cid, row_idx in cluster_to_sample_idx.items():
            ctx = rows[row_idx]["Contexto analizado"]
            t_ref = str(rows[row_idx].get(km.get("titulo", "Título"), ""))
            pkl_tone = None
            pkl_theme = None
            if tone_model is not None:
                pkl_tone = map_tone_label(_safe_predict(tone_model, [ctx or ""], "tono")[0])
            if theme_model is not None:
                pkl_theme = format_theme_label(_safe_predict(theme_model, [ctx or ""], "tema")[0])
            cluster_pkl[cid] = (pkl_tone, pkl_theme)
            fut = executor.submit(
                _call_openai_cluster,
                client,
                model,
                brand,
                aliases,
                brand_regexes,
                ctx,
                t_ref,
                request_tone=plan["use_llm_tone"],
                request_theme=plan["use_llm_theme"],
                pkl_theme=pkl_theme,
            )
            future_to_cid[fut] = cid
            
        for fut in as_completed(future_to_cid):
            cid = future_to_cid[fut]
            tono, tema, subtema = fut.result()
            pkl_tone, pkl_theme = cluster_pkl.get(cid, (None, None))
            sample_idx = cluster_to_sample_idx[cid]
            if pkl_tone:
                tono = pkl_tone
            if pkl_theme:
                tema = pkl_theme
            subtema = ensure_subtema_distinct_from_tema(
                tema,
                subtema,
                brand,
                str(rows[sample_idx].get(km.get("titulo", "Título"), "")),
                rows[sample_idx].get("Contexto analizado", ""),
                aliases,
            )
            cluster_results[cid] = (tono, tema, subtema)
            completed += 1
            if progress_callback and (completed % 15 == 0 or completed == total_clusters):
                pct = 77 + int((completed / total_clusters) * 16)
                progress_callback(pct, f"Analizando con IA… {completed}/{total_clusters} procesados")

    cluster_results = canonicalize_subtopics(cluster_results)

    for i, row in enumerate(rows):
        if row.get("is_duplicate"):
            row["Tono_IA"] = "Duplicada"
            row["Tema_IA"] = "-"
            row["Subtema_IA"] = "-"
            continue

        cid = cluster_map.get(i)
        if cid is not None and cid in cluster_results:
            tono, tema, subtema = cluster_results[cid]
        else:
            tono, tema, subtema = "Neutro", "Gestión Institucional", "Hecho Informativo"

        # CHEQUEO DIRECTO POR FILA: ejes sin PKL siguen la regla de autoría; el subtema no se pierde.
        row_full_text = f"{row.get(km.get('titulo', 'Título'), '')} {row.get('Contexto analizado', '')} {row.get('Resumen - Aclaracion', '')}"
        if check_exact_byline_rule(row_full_text, brand, aliases):
            if tone_model is None:
                tono = "Neutro"
            if theme_model is None:
                tema = "Estudiantes"
            subtema = "Redacción de artículo"

        if theme_model is None:
            tema = ensure_different_tema_subtema(tema, subtema, row.get("Contexto analizado", ""))

        row_ctx = row.get("Contexto analizado", "")
        row_title = str(row.get(km.get("titulo", "Título"), ""))
        row_story = _story_text(row_title, row_ctx)
        if not _subtema_compatible_with_story(subtema, row_story, brand, aliases):
            subtema = ensure_subtema_distinct_from_tema(
                tema, "", brand, row_title, row_ctx, aliases
            )
            if theme_model is None:
                tema = ensure_different_tema_subtema(tema, subtema, row_ctx)

        row["Tono_IA"] = tono
        row["Tema_IA"] = tema
        row["Subtema_IA"] = subtema

    if progress_callback:
        progress_callback(93, "Reconciliando etiquetas de un mismo hecho (Título y Contexto A→Z)…")
    reconcile_az_same_story_labels(rows, km, brand, aliases)
    return rows
