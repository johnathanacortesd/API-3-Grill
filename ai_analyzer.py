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

FACT_ANCHORS = INST_HEADS | DEGREE_LEMMAS | {
    "formacion", "academica", "academico", "beca", "becas", "posgrado",
    "pregrado", "sede", "convenio", "alianza", "dialogo", "inauguracion",
    "nombramiento", "rector", "rectora", "egresado", "carrera",
}

CITY_TAILS = {
    "barranquilla", "bogota", "cali", "medellin", "cartagena", "bucaramanga",
    "pereira", "manizales", "cucuta", "ibague", "neiva", "pasto", "armenia",
    "villavicencio", "valledupar", "monteria", "sincelejo", "popayan",
    "tunja", "riohacha", "quibdo",
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
    """Content tokens of the client name; must not be used as 'same story' evidence."""
    toks: Set[str] = set()
    for item in [brand] + list(aliases or []):
        if not str(item).strip():
            continue
        toks |= get_content_words_set(normalize_text_for_matching(str(item)))
    return toks


def _without_brand_tokens(norm_text: str, brand_toks: Set[str]) -> str:
    if not brand_toks:
        return norm_text
    return " ".join(w for w in (norm_text or "").split() if w not in brand_toks)


def extract_event_anchor(title_raw: str) -> str:
    if not title_raw:
        return ""
    parts = re.split(r"\s*[:|-]\s*", title_raw, 1)
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

        if " " in base:
            compact = base.replace(" ", "")
            if len(compact) >= 6:
                variants_set.add(compact)

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


_DE_WORDS = {"de", "del"}


def _collapse_repeated_prepositions(words: List[str]) -> List[str]:
    """Drop stacked 'de de' glue that produces keyword collages."""
    out: List[str] = []
    prev_de = False
    for w in words:
        is_de = unidecode(w.lower()) in _DE_WORDS
        if is_de and prev_de:
            continue
        out.append(w)
        prev_de = is_de
    return out


def _is_de_collage(words: List[str]) -> bool:
    """True when 'de' is stuffing keywords together rather than a noun phrase."""
    if not words:
        return False
    lows = [unidecode(w.lower()) for w in words]
    de_n = sum(1 for w in lows if w in _DE_WORDS)
    if any(lows[i] in _DE_WORDS and lows[i + 1] in _DE_WORDS for i in range(len(lows) - 1)):
        return True
    if de_n >= 3:
        return True
    return False


def _normalize_subtema_phrase(text: str, brand: str, tema: str = "") -> str:
    if not text:
        return ""
    clean = re.sub(r'[,.;:!?¿¡"\'\(\)\[\]\{\}\-_/\\|]', " ", str(text))
    words = [w for w in clean.split() if w]
    if _is_de_collage(words):
        return ""
    words = _collapse_repeated_prepositions(words)
    words = _fit_to_max_words(words)
    res = _strip_forbidden_subtema_prefixes(" ".join(words).strip())
    res = _strip_tema_echo_prefix(res, tema)
    words = _fit_to_max_words(_collapse_repeated_prepositions([w for w in res.split() if w]))
    if _is_de_collage(words):
        return ""
    res = " ".join(words).strip()
    if not res:
        return ""
    brand_words = set(re.findall(r"\b[a-z0-9]+\b", unidecode(brand.lower())))
    res_words = set(re.findall(r"\b[a-z0-9]+\b", unidecode(res.lower())))
    if res_words.issubset(brand_words):
        return ""
    if unidecode(res.lower()) in {
        "universidad", "autonoma", "fundacion", "clinica", "hospital",
        "institucion", "asociacion", "mencion", "entrevista", "noticia",
    }:
        return ""
    return res.capitalize()


def _fit_to_max_words(words: List[str]) -> List[str]:
    words = _collapse_repeated_prepositions([w for w in words if w])
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
    return _collapse_repeated_prepositions(words)


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


def _is_strong_subtema(phrase: str, tema: str, title: str, brand: str) -> bool:
    if not phrase:
        return False
    words = phrase.split()
    if len(words) < MIN_SUBTEMA_WORDS or len(words) > MAX_SUBTEMA_WORDS:
        return False
    if _is_lead_clause_scrap(words):
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
    brand_words = set(re.findall(r"\b[a-z0-9]+\b", unidecode(brand.lower())))
    if frase_toks and frase_toks.issubset(brand_words):
        return False
    if _is_title_scrap(phrase, title):
        return False
    if _is_de_collage(words):
        return False
    return True


def _looks_like_name_part(token: str) -> bool:
    if not token:
        return False
    low = unidecode(token.lower())
    if low in LEAD_ACTION_VERBS or low in DISCOURSE_STARTERS or low in STOPWORDS_ES:
        return False
    if token[0].isupper():
        return True
    return token[0].isalpha()


def _institution_span(words: List[str], start_idx: int) -> List[str]:
    if start_idx >= len(words):
        return []
    if start_idx + 1 < len(words):
        nxt = unidecode(words[start_idx + 1].lower())
        if nxt in LEAD_ACTION_VERBS:
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
    if has_edu_cue:
        fitted = _join_prefix_and_inst(["Formación", "académica", "en", "la"], inst_core)
        return _normalize_subtema_phrase(" ".join(fitted), brand)
    if len(inst_core) >= MIN_SUBTEMA_WORDS:
        return _normalize_subtema_phrase(" ".join(inst_core), brand)
    fitted = _join_prefix_and_inst(["Actividad", "en", "la"], inst_core)
    return _normalize_subtema_phrase(" ".join(fitted), brand)


def _score_subtema_window(words: List[str], tema: str, title: str, brand: str, at_sentence_start: bool = False) -> int:
    if _is_lead_clause_scrap(words):
        return -120
    phrase = " ".join(words)
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
    title_pen = 30 if _is_title_scrap(phrase, title) else 0
    noun_bonus = 0
    for w in content:
        wl = unidecode(w.lower())
        if wl.endswith(("cion", "sion", "miento", "dad", "aje", "ncia", "encia", "ura", "azgo")):
            noun_bonus += 8
        if wl in FACT_ANCHORS or wl in INST_HEADS or wl in DEGREE_LEMMAS:
            noun_bonus += 18
    start = unidecode(words[0].lower())
    start_pen = -25 if start in DISCOURSE_STARTERS else (-8 if start in STOPWORDS_ES else 6)
    lead_pen = 40 if at_sentence_start and start in DISCOURSE_STARTERS else 0
    brand_words = set(re.findall(r"\b[a-z0-9]+\b", unidecode(brand.lower())))
    brand_pen = 20 if phrase_toks and phrase_toks.issubset(brand_words) else 0
    de_n = sum(1 for w in words if unidecode(w.lower()) in _DE_WORDS)
    de_pen = 80 if _is_de_collage(words) else (18 * max(0, de_n - 1))
    return (
        12 * len(content)
        + noun_bonus
        + start_pen
        - title_pen
        - lead_pen
        - (12 * bleed_n)
        - brand_pen
        - de_pen
    )


def _noun_phrase_from_context(ctx: str, tema: str, title: str, brand: str) -> str:
    """Build a fact noun phrase from contexto; never return a lead-clause scrap."""
    text = str(ctx or "").strip()
    if not text or text == "-":
        return ""
    edu = _education_fact_phrase(text, brand)
    if edu and not _is_lead_clause_scrap(edu.split()) and not _labels_too_close(tema, edu):
        edu = _complete_proper_names_from_context(edu, text)
        return _normalize_subtema_phrase(edu, brand, tema)

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
            if _is_de_collage(window):
                continue
            at_start = [unidecode(w.lower()) for w in window[:2]] == orig_lead[:2]
            score = _score_subtema_window(window, tema, title, brand, at_sentence_start=at_start)
            if score > best_score:
                best_score = score
                best = " ".join(window)
    if best_score < 0 or not best:
        return edu
    completed = _complete_proper_names_from_context(best, text)
    return _normalize_subtema_phrase(completed, brand, tema)


def clean_subtema(text: str, brand: str, title_fallback: str) -> str:
    if not text:
        return _fallback_from_title(title_fallback)
    res = _normalize_subtema_phrase(text, brand)
    if not res:
        return _fallback_from_title(title_fallback)
    return res


def clean_subtema_specific(text: str, brand: str, tema: str = "") -> str:
    """PKL-tema path: never collapse to a title scrap or a 1–2 word leftover."""
    return _normalize_subtema_phrase(text, brand, tema)

MAX_TEMA_WORDS = 5

def clean_tema(text: str) -> str:
    if not text:
        return "Gestión Institucional"
    clean = re.sub(r'[,.;:!?¿¡"\'\(\)\[\]\{\}\-_/\\|]', ' ', str(text)).strip()
    words = clean.split()[:MAX_TEMA_WORDS]
    while words and words[-1].lower() in FORBIDDEN_TRAILING_WORDS:
        words.pop()
    res = " ".join(words).title()
    if res.lower() in ["otros", "otro", "general", "varios", "miscelanea", "sin clasificar", ""]:
        return "Gestión Institucional"
    return res

def ensure_different_tema_subtema(tema: str, subtema: str, ctx: str) -> str:
    t_clean = tema.strip().title()
    s_clean = subtema.strip().capitalize()
    
    if t_clean.lower() == s_clean.lower() or fuzz.ratio(t_clean.lower(), s_clean.lower()) >= 80:
        c_low = unidecode(f"{s_clean} {ctx}".lower())
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
    """Positive tone only when the brand (not a third party) backs, celebrates or allies."""
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


GENERIC_TEMA_LABELS = frozenset({
    "gestión institucional", "gestion institucional", "otros", "otro", "general",
    "-", "",
})


def _majority_label(labels: List[str]) -> str:
    cleaned = [str(x).strip() for x in labels if str(x).strip()]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    specific = [(lab, n) for lab, n in counts.items() if lab.lower() not in GENERIC_TEMA_LABELS]
    pool = specific or list(counts.items())
    pool.sort(key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
    return pool[0][0]


def _subtemas_related(a: str, b: str) -> bool:
    """Same fact family (sede norte / inauguración sede norte), not a shared opener like 'apertura de'."""
    if not a or not b:
        return False
    na = normalize_text_for_matching(a)
    nb = normalize_text_for_matching(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    sa = get_content_words_set(na)
    sb = get_content_words_set(nb)
    if not sa or not sb:
        return False
    overlap = sa & sb
    if len(overlap) < 2:
        return False
    denser = min(len(sa), len(sb))
    if denser and len(overlap) / denser >= 0.5:
        return True
    return fuzz.token_set_ratio(na, nb) >= 78


def synthesize_temas_from_subtemas(
    cluster_results: Dict[int, Tuple[str, str, str]],
) -> Dict[int, Tuple[str, str, str]]:
    """Related subtemas share one tema. Does not invent labels; majority among the family."""
    subs = [sub for _, _, sub in cluster_results.values() if sub]
    unique_subs = list(dict.fromkeys(subs))
    parent = {s: s for s in unique_subs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(unique_subs)):
        for j in range(i + 1, len(unique_subs)):
            if _subtemas_related(unique_subs[i], unique_subs[j]):
                ra, rb = find(unique_subs[i]), find(unique_subs[j])
                if ra != rb:
                    parent[rb] = ra

    families: Dict[str, List[str]] = {}
    for s in unique_subs:
        families.setdefault(find(s), []).append(s)

    sub_to_tema: Dict[str, str] = {}
    for members in families.values():
        member_set = set(members)
        temas = [tema for _, tema, sub in cluster_results.values() if sub in member_set]
        chosen = _majority_label(temas)
        if chosen:
            for m in members:
                sub_to_tema[m] = chosen

    return {
        cid: (tono, sub_to_tema.get(sub, tema), sub)
        for cid, (tono, tema, sub) in cluster_results.items()
    }

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
    progress_callback: Optional[Callable[[int, str], None]] = None,
    progress_pct: int = 74,
    brand: str = "",
    aliases: Optional[List[str]] = None,
) -> Dict[int, int]:
    """Group only true reprints / near-duplicate stories, never 'same brand'."""
    n = len(rows)
    cluster_map = {}
    clusters_rep = {}
    current_cluster = 0
    brand_toks = brand_content_tokens(brand, aliases)

    active_indices = [i for i in range(n) if not rows[i].get("is_duplicate")]
    total_active = len(active_indices)
    sorted_indices = sorted(
        active_indices,
        key=lambda idx: normalize_text_for_matching(str(rows[idx].get(km.get("titulo", "Título"), "")))
    )

    for k, i in enumerate(sorted_indices, start=1):
        t_raw = str(rows[i].get(km.get("titulo", "Título"), ""))
        r_raw = str(rows[i].get("Resumen - Aclaracion") or rows[i].get("resumen corto") or "")
        
        t_norm = normalize_text_for_matching(t_raw)
        t_story = _without_brand_tokens(t_norm, brand_toks)
        c_words = get_content_words_set(t_story)
        lead_words = get_lead_content_words(t_story, n_words=3)
        anchor = extract_event_anchor(t_raw)
        if brand_toks and get_content_words_set(anchor).issubset(brand_toks):
            anchor = ""
        r_norm = _without_brand_tokens(normalize_text_for_matching(r_raw[:350]), brand_toks)
        r_words = get_content_words_set(r_norm)
        
        assigned = False
        for cid, rep in clusters_rep.items():
            rep_t = rep["title_norm"]
            rep_story = rep["title_story"]
            rep_words = rep["content_words"]
            rep_lead = rep["lead_words"]
            rep_anchor = rep["anchor"]
            rep_r = rep["body_norm"]
            rep_r_words = rep["body_words"]
            
            if anchor and rep_anchor and anchor == rep_anchor:
                cluster_map[i] = cid
                assigned = True
                break
                
            if len(lead_words) >= 3 and len(rep_lead) >= 3 and lead_words == rep_lead:
                cluster_map[i] = cid
                assigned = True
                break

            if len(lead_words) >= 2 and len(rep_lead) >= 2 and lead_words[:2] == rep_lead[:2]:
                combined_lead_len = len(" ".join(lead_words[:2]))
                if combined_lead_len >= 13:
                    cluster_map[i] = cid
                    assigned = True
                    break

            if t_story and rep_story:
                if t_story in rep_story or rep_story in t_story:
                    cluster_map[i] = cid
                    assigned = True
                    break
                min_len = min(len(t_story), len(rep_story))
                if min_len >= 18 and t_story[:18] == rep_story[:18]:
                    cluster_map[i] = cid
                    assigned = True
                    break

            overlap = c_words & rep_words
            if len(overlap) >= 4:
                cluster_map[i] = cid
                assigned = True
                break

            if len(overlap) < 2:
                continue

            if t_story and rep_story and min(len(t_story), len(rep_story)) >= 12:
                if fuzz.partial_ratio(t_story, rep_story) >= 90:
                    cluster_map[i] = cid
                    assigned = True
                    break
                if fuzz.token_set_ratio(t_story, rep_story) >= 88:
                    cluster_map[i] = cid
                    assigned = True
                    break
            
            if r_norm and rep_r and len(r_norm) > 40 and len(rep_r) > 40:
                body_overlap = r_words & rep_r_words
                if len(body_overlap) >= 5 and fuzz.token_set_ratio(r_norm, rep_r) >= 90:
                    cluster_map[i] = cid
                    assigned = True
                    break
                    
        if not assigned:
            cluster_map[i] = current_cluster
            clusters_rep[current_cluster] = {
                "title_norm": t_norm,
                "title_story": t_story,
                "content_words": c_words,
                "lead_words": lead_words,
                "anchor": anchor,
                "body_norm": r_norm,
                "body_words": r_words,
            }
            current_cluster += 1

        if progress_callback and (k == 1 or k % 40 == 0 or k == total_active):
            progress_callback(
                progress_pct,
                f"Agrupando noticias similares… {k}/{total_active} notas, {current_cluster} hechos",
            )
            
    return cluster_map


def _subtemas_near_duplicate(a: str, b: str) -> bool:
    """Reprint of the same fact phrase — not a shared brand token or weak overlap."""
    if not a or not b:
        return False
    na = normalize_text_for_matching(a)
    nb = normalize_text_for_matching(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return fuzz.ratio(na, nb) >= 92 or (
        fuzz.token_set_ratio(na, nb) >= 92 and fuzz.token_sort_ratio(na, nb) >= 90
    )


def canonicalize_subtopics(
    cluster_results: Dict[int, Tuple[str, str, str]],
    unify_related_themes: bool = True,
) -> Dict[int, Tuple[str, str, str]]:
    """Only collapse near-duplicate subtema strings. Do not broadcast a dominant junk label."""
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

    final_results = {}
    for cid, (tono, tema, sub) in cluster_results.items():
        canonical_sub = mapping.get(sub, sub)
        final_results[cid] = (tono, tema, canonical_sub)

    if unify_related_themes:
        # Same canonical subtema only (reprints). Never merge unrelated facts into one tema family.
        by_sub: Dict[str, List[int]] = {}
        for cid, (tono, tema, sub) in final_results.items():
            by_sub.setdefault(sub, []).append(cid)
        aligned = dict(final_results)
        for sub, cids in by_sub.items():
            if len(cids) < 2:
                continue
            chosen_tema = _majority_label([aligned[c][1] for c in cids])
            chosen_tono = _majority_label([aligned[c][0] for c in cids])
            for cid in cids:
                tono, tema, _ = aligned[cid]
                aligned[cid] = (chosen_tono or tono, chosen_tema or tema, sub)
        return aligned
    return final_results

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
) -> str:
    """Keep a 4–7 word fact noun phrase, distinct from the PKL tema. Never use a lead-clause scrap."""
    llm_clean = clean_subtema_specific(subtema or "", brand, tema)
    if llm_clean:
        llm_clean = _complete_proper_names_from_context(llm_clean, ctx)
        llm_clean = _normalize_subtema_phrase(llm_clean, brand, tema)

    ctx_phrase = _noun_phrase_from_context(ctx, tema, title, brand)
    if ctx_phrase:
        ctx_phrase = _complete_proper_names_from_context(ctx_phrase, ctx)
        ctx_phrase = _normalize_subtema_phrase(ctx_phrase, brand, tema)

    if llm_clean and _is_strong_subtema(llm_clean, tema, title, brand):
        return llm_clean
    if ctx_phrase and _is_strong_subtema(ctx_phrase, tema, title, brand):
        return ctx_phrase
    if ctx_phrase and not _is_lead_clause_scrap(ctx_phrase.split()) and not _labels_too_close(tema, ctx_phrase):
        if MIN_SUBTEMA_WORDS <= len(ctx_phrase.split()) <= MAX_SUBTEMA_WORDS:
            return ctx_phrase
    if llm_clean and not _is_lead_clause_scrap(llm_clean.split()) and not _labels_too_close(tema, llm_clean):
        if MIN_SUBTEMA_WORDS <= len(llm_clean.split()) <= MAX_SUBTEMA_WORDS:
            return llm_clean
    return ctx_phrase or llm_clean or "Hecho informativo institucional"


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
        "   REGLA DE ORO: si el cliente expresa o recibe ACOMPAÑAMIENTO, RESPALDO, APOYO, "
        'FELICITACIONES, CELEBRACIÓN o ALIANZA, el tono es estrictamente "Positivo".'
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
- Caso 6: "Boletín de cifras donde {b} aporta un dato técnico..."
  -> Tono: "Neutro".
"""


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
            f'{n}. "tema": dominio que AGRUPA subtemas relacionados (español colombiano, 2 a 5 palabras). '
            'Sintetiza el hecho; no copies el subtema ni uses "Otros". '
            'Ejemplos de forma (no de cliente): "Educación Superior", "Gestión Tributaria", "Sector Salud".'
        )
        json_fields.append('"tema": "..."')
        n += 1

    subtema_rule = (
        f'{n}. "subtema": HECHO ESPECÍFICO: una sola frase nominal coherente en español colombiano, '
        "de 4 a 7 palabras (máximo 7), completa, sin cortar nombres propios. "
        "Describe el hecho (formación, grado, evento), no la cláusula inicial de la frase. "
        'Sin comas ni puntos. PROHIBIDO "Mención", collage, recortar el titular o copiar el tema. '
        'PROHIBIDO pegar palabras sueltas con "de" (MAL: "abogado de universidad de joven de barrio").'
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
            "con nombres propios completos, distinta al tema "
            '(BIEN: "Formación académica en la Universidad Simón Bolívar").'
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
                        "El tema agrupa subtemas relacionados. "
                        "El subtema es una frase nominal completa en español colombiano."
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
        else:
            tono = "Neutro"

        raw_sub = data.get("subtema", "")
        if pkl_theme and not request_theme:
            subtema = clean_subtema_specific(raw_sub, brand, pkl_theme)
        else:
            subtema = clean_subtema(raw_sub, brand, title_ref)

        if request_theme:
            tema = clean_tema(data.get("tema", ""))
            tema = ensure_different_tema_subtema(tema, subtema, ctx)
        else:
            tema = (pkl_theme or "").strip() or "Gestión Institucional"
            subtema = ensure_subtema_distinct_from_tema(tema, subtema, brand, title_ref, ctx)

        return tono, tema, subtema
    except Exception as e:
        logger.error(f"Error en llamada OpenAI: {e}")
        if request_theme:
            sub_fb = _fallback_from_title(title_ref)
            tema_fb = ensure_different_tema_subtema("Gestión Institucional", sub_fb, ctx)
        else:
            tema_fb = (pkl_theme or "").strip() or "Gestión Institucional"
            sub_fb = ensure_subtema_distinct_from_tema(tema_fb, "", brand, title_ref, ctx)
        tono_fb = "Positivo" if check_positive_institutional_override(ctx, brand, aliases) else "Neutro"
        return tono_fb, tema_fb, sub_fb


def _batch_pkl_labels_for_clusters(
    ordered_cids: List[int],
    sample_texts: List[str],
    tone_model,
    theme_model,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[int, Tuple[Optional[str], Optional[str]]]:
    """One sklearn predict per axis (not per cluster) so hundreds of notes don't stall."""
    from pkl_classifier import format_theme_label, map_tone_label, _safe_predict

    tones: List[Optional[str]] = [None] * len(ordered_cids)
    themes: List[Optional[str]] = [None] * len(ordered_cids)
    if tone_model is not None:
        if progress_callback:
            progress_callback(75, f"Clasificando tono PKL en lote ({len(sample_texts)} hechos)…")
        preds = _safe_predict(tone_model, sample_texts, "tono")
        tones = [map_tone_label(p) for p in preds]
    if theme_model is not None:
        if progress_callback:
            progress_callback(76, f"Clasificando tema PKL en lote ({len(sample_texts)} hechos)…")
        preds = _safe_predict(theme_model, sample_texts, "tema")
        themes = [format_theme_label(p) or None for p in preds]
    return {cid: (tones[i], themes[i]) for i, cid in enumerate(ordered_cids)}


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
    from pkl_classifier import classification_plan

    client = OpenAI(api_key=api_key)
    plan = classification_plan(True, tone_model, theme_model)

    brand_regexes = generate_brand_variants(brand, aliases)
    n_active = sum(1 for row in rows if not row.get("is_duplicate"))
    
    if progress_callback:
        progress_callback(71, f"Extrayendo contexto de la marca… 0/{n_active} notas")
    done_ctx = 0
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
            done_ctx += 1
            if progress_callback and (done_ctx == 1 or done_ctx % 80 == 0 or done_ctx == n_active):
                progress_callback(71, f"Extrayendo contexto de la marca… {done_ctx}/{n_active} notas")

    if progress_callback:
        progress_callback(74, f"Agrupando noticias similares… 0/{n_active} notas")
    cluster_map = cluster_similar_rows(
        rows,
        km,
        brand_regexes,
        progress_callback=progress_callback,
        progress_pct=74,
        brand=brand,
        aliases=aliases,
    )
    
    unique_clusters = sorted(set(cluster_map.values()))
    total_clusters = len(unique_clusters)
    
    cluster_to_sample_idx = {}
    for row_idx, cid in cluster_map.items():
        if cid not in cluster_to_sample_idx:
            cluster_to_sample_idx[cid] = row_idx

    ordered_cids = list(cluster_to_sample_idx.keys())
    sample_texts = [rows[cluster_to_sample_idx[cid]].get("Contexto analizado") or "" for cid in ordered_cids]
    cluster_pkl: Dict[int, Tuple[Optional[str], Optional[str]]] = {
        cid: (None, None) for cid in ordered_cids
    }
    if tone_model is not None or theme_model is not None:
        if progress_callback:
            progress_callback(
                75,
                f"Temas listos: {total_clusters} hechos únicos de {n_active} notas. Clasificando PKL en lote…",
            )
        cluster_pkl = _batch_pkl_labels_for_clusters(
            ordered_cids, sample_texts, tone_model, theme_model, progress_callback
        )
    elif progress_callback:
        progress_callback(
            76,
            f"Temas listos: {total_clusters} hechos únicos de {n_active} notas. Iniciando etiquetado…",
        )
            
    cluster_results: Dict[int, Tuple[str, str, str]] = {}

    if progress_callback:
        progress_callback(77, f"Etiquetando con {model}… 0/{total_clusters} hechos")
        
    completed = 0
    with ThreadPoolExecutor(max_workers=14) as executor:
        future_to_cid = {}
        for cid, row_idx in cluster_to_sample_idx.items():
            ctx = rows[row_idx]["Contexto analizado"]
            t_ref = str(rows[row_idx].get(km.get("titulo", "Título"), ""))
            pkl_tone, pkl_theme = cluster_pkl.get(cid, (None, None))
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
            if pkl_tone:
                tono = pkl_tone
            if pkl_theme:
                tema = pkl_theme
                sample_idx = cluster_to_sample_idx[cid]
                subtema = ensure_subtema_distinct_from_tema(
                    tema,
                    subtema,
                    brand,
                    str(rows[sample_idx].get(km.get("titulo", "Título"), "")),
                    rows[sample_idx].get("Contexto analizado", ""),
                )
            cluster_results[cid] = (tono, tema, subtema)
            completed += 1
            if progress_callback and (
                completed == 1 or completed % 5 == 0 or completed == total_clusters
            ):
                pct = 77 + int((completed / total_clusters) * 16) if total_clusters else 93
                progress_callback(pct, f"Etiquetando con IA… {completed}/{total_clusters} hechos")

    if theme_model is None:
        for cid, (tono, tema, subtema) in list(cluster_results.items()):
            sample_idx = cluster_to_sample_idx[cid]
            tema = ensure_different_tema_subtema(
                tema, subtema, rows[sample_idx].get("Contexto analizado", "")
            )
            cluster_results[cid] = (tono, tema, subtema)

    cluster_results = canonicalize_subtopics(
        cluster_results,
        unify_related_themes=theme_model is None,
    )

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

        row["Tono_IA"] = tono
        row["Tema_IA"] = tema
        row["Subtema_IA"] = subtema
            
    return rows
