# ======================================
# Motor de Análisis con IA (ai_analyzer.py)
# ======================================
import os
import re
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Callable, Set
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

MAX_SUBTEMA_WORDS = 7
MIN_SUBTEMA_WORDS = 4

DISCOURSE_STARTERS = {
    "de", "del", "ese", "esa", "esos", "esas", "este", "esta", "estos", "estas",
    "aquel", "aquella", "luego", "despues", "ademas", "tambien", "mientras",
    "aunque", "cuando", "donde", "primero", "primera", "entonces", "asi",
}

FACT_HINTS = {
    "beca", "becas", "feria", "catedra", "congreso", "abogado", "formacion",
    "inauguracion", "matricula", "premio", "convenio", "sede", "hospital",
    "captura", "obra", "acuerdo", "alianza", "egresado", "inscripcion",
    "oferta", "cine", "memoria", "patrimonio", "inspirate",
}

TITLE_EVENT_HINTS = (
    "feria", "inspirate", "catedra", "ficci", "siab", "women", "cine",
    "patrimonio",
)

_MEDIA_PREFIX_RE = re.compile(
    r"^(?:imagenes|en imagenes|fotos|en fotos|video|en video|en vivo)\s*[|:]\s*",
    re.IGNORECASE,
)
_OUTLET_SUFFIX_RE = re.compile(
    r"\s*[-|]\s*(?:noticias\s+vital|en\s+vivo)\s*$",
    re.IGNORECASE,
)
_LEAD_FUNCTION_RE = re.compile(
    r"^(?:el|la|los|las|un|una|unos|unas|de|del|en|al|a|y|para|por|con)\s+"
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


def _brand_token_set(brand: str, aliases: Optional[List[str]] = None) -> Set[str]:
    parts = [brand or ""] + list(aliases or [])
    toks: Set[str] = set()
    for part in parts:
        for w in re.findall(r"[a-z0-9]+", unidecode(part.lower())):
            if len(w) > 1 and w not in STOPWORDS_ES:
                toks.add(w)
    return toks


def _surface_tokens(text: str) -> List[str]:
    t = unidecode(str(text or "").lower())
    t = _MEDIA_PREFIX_RE.sub("", t)
    t = _OUTLET_SUFFIX_RE.sub("", t)
    return re.findall(r"[a-z0-9]+", t)


def _lead_tokens_for_grouping(text: str, brand_tokens: Set[str]) -> List[str]:
    """Opening tokens of título/resumen after dropping articles and the client marca.

    Matching raw first-3/4 words that are only the brand would re-introduce
    over-grouping (every 'Gobernadora Lucy García…' story glued together).
    """
    toks = _surface_tokens(text)
    while toks and (toks[0] in STOPWORDS_ES or toks[0] in brand_tokens):
        toks.pop(0)
    return toks


def _same_opening_lead(text_a: str, text_b: str, brand_tokens: Set[str]) -> bool:
    """True when both strings share the first 3 or 4 grouping words."""
    a = _lead_tokens_for_grouping(text_a, brand_tokens)
    b = _lead_tokens_for_grouping(text_b, brand_tokens)
    if len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4]:
        return True
    if len(a) >= 3 and len(b) >= 3 and a[:3] == b[:3]:
        return True
    raw_a, raw_b = _surface_tokens(text_a), _surface_tokens(text_b)
    if len(raw_a) >= 4 and len(raw_b) >= 4 and raw_a[:4] == raw_b[:4]:
        content = [w for w in raw_a[:4] if w not in STOPWORDS_ES]
        if content and not set(content).issubset(brand_tokens):
            return True
    if len(raw_a) >= 3 and len(raw_b) >= 3 and raw_a[:3] == raw_b[:3]:
        content = [w for w in raw_a[:3] if w not in STOPWORDS_ES]
        if content and not set(content).issubset(brand_tokens):
            return True
    return False


def _compact_title_key(text: str) -> str:
    t = unidecode(str(text or "").lower())
    t = _MEDIA_PREFIX_RE.sub("", t)
    t = _OUTLET_SUFFIX_RE.sub("", t)
    t = t.replace("-", "").replace("_", "")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def titles_clearly_similar(title_a: str, title_b: str, brand_tokens: Optional[Set[str]] = None) -> bool:
    """Same título, one is a prefix of the other, or very high order-aware similarity."""
    ka, kb = _compact_title_key(title_a), _compact_title_key(title_b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    brand_tokens = brand_tokens or set()
    shorter, longer = (ka, kb) if len(ka) <= len(kb) else (kb, ka)
    if len(shorter) >= 24 and longer.startswith(shorter):
        content = [w for w in shorter.split() if w not in STOPWORDS_ES]
        if content and not set(content).issubset(brand_tokens):
            return True
    if min(len(ka), len(kb)) >= 24 and fuzz.ratio(ka, kb) >= 88:
        return True
    return False


# Grouping only needs the opening of resumen/cuerpo (first 3–4 words).
# Sucre xlsx often stores the full article in CuerpoEs; all-pairs tokenization
# of that body is what hung Streamlit at ~74%.
_RESUMEN_GROUPING_CHARS = 280
_TITLE_PREFIX_MIN = 24
_FUZZY_RATIO_MIN = 88
_FUZZY_PREFIX_CHARS = 8
_FUZZY_FULL_BUCKET = 48


def _resumen_for_grouping(text: str) -> str:
    s = str(text or "")
    if len(s) > _RESUMEN_GROUPING_CHARS:
        return s[:_RESUMEN_GROUPING_CHARS]
    return s


def should_group_news_items(
    title_a: str,
    resumen_a: str,
    title_b: str,
    resumen_b: str,
    brand: str = "",
    aliases: Optional[List[str]] = None,
) -> bool:
    """Grouping is optional. Only same/similar título or shared first 3–4 words."""
    brand_tokens = _brand_token_set(brand, aliases)
    if titles_clearly_similar(title_a, title_b, brand_tokens):
        return True
    if _same_opening_lead(title_a, title_b, brand_tokens):
        return True
    ra, rb = _resumen_for_grouping(resumen_a), _resumen_for_grouping(resumen_b)
    if ra and rb and _same_opening_lead(ra, rb, brand_tokens):
        return True
    return False


def _row_title(row: dict, km: dict) -> str:
    return str(row.get(km.get("titulo", "Título"), "") or row.get("Título") or "")


def _row_resumen(row: dict) -> str:
    return str(
        row.get("Resumen - Aclaracion")
        or row.get("resumen corto")
        or row.get("Resumen")
        or row.get("CuerpoEs")
        or ""
    )


def _content_not_brand_only(words: List[str], brand_tokens: Set[str]) -> bool:
    content = [w for w in words if w not in STOPWORDS_ES]
    return bool(content) and not set(content).issubset(brand_tokens)


def _lead_bucket_keys(text: str, brand_tokens: Set[str]) -> List[Tuple]:
    """Hashable first-3 / first-4 word keys (marca-stripped and raw)."""
    if not text:
        return []
    keys: List[Tuple] = []
    stripped = _lead_tokens_for_grouping(text, brand_tokens)
    if len(stripped) >= 3:
        keys.append(("s3", tuple(stripped[:3])))
    raw = _surface_tokens(text)
    if len(raw) >= 4 and _content_not_brand_only(raw[:4], brand_tokens):
        keys.append(("r4", tuple(raw[:4])))
    if len(raw) >= 3 and _content_not_brand_only(raw[:3], brand_tokens):
        keys.append(("r3", tuple(raw[:3])))
    return keys


def _dsu_union_all(dsu: "_DSU", members: List[int]) -> None:
    if len(members) < 2:
        return
    head = members[0]
    for other in members[1:]:
        dsu.union(head, other)


def _has_echoed_content_pair(words: List[str]) -> bool:
    lows = [unidecode(w.lower()) for w in words if unidecode(w.lower()) not in STOPWORDS_ES]
    return any(lows[i] == lows[i + 1] for i in range(len(lows) - 1))


def _tokenize_phrase_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñÜü0-9]+", str(text or ""))


def _is_lead_clause_scrap(words: List[str]) -> bool:
    if not words:
        return True
    first = unidecode(words[0].lower())
    return first in DISCOURSE_STARTERS


def _is_title_scrap(phrase: str, title: str) -> bool:
    if not phrase or not title:
        return False
    phrase_words = [unidecode(w.lower()) for w in phrase.split()]
    title_words = [unidecode(w.lower()) for w in _tokenize_phrase_words(title)]
    n = len(phrase_words)
    if n < MIN_SUBTEMA_WORDS or len(title_words) < n:
        return False
    return phrase_words == title_words[:n]


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
        extra: List[str] = []
        j = i + 1
        while j < len(cwords):
            tok = cwords[j]
            low = unidecode(tok.lower())
            if low in STOPWORDS_ES or low in DISCOURSE_STARTERS or low in phrase_lows:
                break
            if not tok or not tok[0].isupper():
                break
            extra.append(tok)
            j += 1
            if len(pwords) + len(extra) >= MAX_SUBTEMA_WORDS:
                break
        if extra:
            fitted = pwords + extra
            while len(fitted) > MAX_SUBTEMA_WORDS and unidecode(fitted[0].lower()) in STOPWORDS_ES:
                fitted = fitted[1:]
            fitted = fitted[:MAX_SUBTEMA_WORDS]
            while fitted and fitted[-1].lower() in FORBIDDEN_TRAILING_WORDS:
                fitted.pop()
            return " ".join(fitted)
        break
    return phrase


def _window_from_context(ctx: str, tema: str, title: str, brand: str) -> str:
    """Noun-phrase window from this story's contexto; never a lead-clause scrap."""
    words = _tokenize_phrase_words(ctx)
    if len(words) < MIN_SUBTEMA_WORDS:
        return ""
    brand_toks = _brand_token_set(brand)
    best = ""
    best_score = -10**9
    for n in range(MAX_SUBTEMA_WORDS, MIN_SUBTEMA_WORDS - 1, -1):
        for i in range(0, len(words) - n + 1):
            window = list(words[i:i + n])
            if _is_lead_clause_scrap(window):
                continue
            while window and window[-1].lower() in FORBIDDEN_TRAILING_WORDS:
                window.pop()
            if len(window) < MIN_SUBTEMA_WORDS:
                continue
            phrase = " ".join(window)
            if _labels_too_close(tema, phrase) or _is_title_scrap(phrase, title):
                continue
            lows = [unidecode(w.lower()) for w in window]
            content = [w for w in lows if w not in STOPWORDS_ES and len(w) > 2]
            if len(content) < 2:
                continue
            if set(content).issubset(brand_toks):
                continue
            score = 10 * len(content)
            score += 16 * sum(1 for w in content if w in FACT_HINTS or any(w.startswith(h) for h in FACT_HINTS))
            if lows[0] in STOPWORDS_ES:
                score -= 8
            if _has_echoed_content_pair(window):
                score -= 40
            if score > best_score:
                best_score = score
                best = phrase
    if best_score < 8 or not best:
        return ""
    return best.capitalize()


def subtema_supported_by_context(
    subtema: str,
    ctx: str,
    brand: str = "",
    aliases: Optional[List[str]] = None,
) -> bool:
    """True when the subtema names this story, not a foreign fact glued on."""
    stoks = get_content_words_set(normalize_text_for_matching(subtema))
    ctoks = get_content_words_set(normalize_text_for_matching(ctx))
    brand_toks = _brand_token_set(brand, aliases)
    stoks -= brand_toks
    ctoks -= brand_toks
    if not stoks:
        return False
    overlap_ratio = len(stoks & ctoks) / len(stoks)
    if overlap_ratio >= 0.4:
        return True
    s_fam = _fact_families_present(stoks)
    c_fam = _fact_families_present(ctoks)
    if s_fam and c_fam and s_fam.isdisjoint(c_fam):
        return False
    return overlap_ratio >= 0.25 or not s_fam


_FACT_FAMILIES = (
    frozenset({"matricula", "beca", "inscrip", "feria", "inspirate", "oferta"}),
    frozenset({"cine", "catedra", "ficci", "festival", "memoria", "patrimonio"}),
    frozenset({"siab", "quindio", "ingenier"}),
    frozenset({"premio", "award", "women"}),
)


def _fact_families_present(tokens: Set[str]) -> Set[int]:
    hits: Set[int] = set()
    for i, fam in enumerate(_FACT_FAMILIES):
        for tok in tokens:
            if any(tok.startswith(f) or f in tok for f in fam):
                hits.add(i)
                break
    return hits


def clean_subtema(text: str, brand: str, title_fallback: str) -> str:
    if not text:
        return _fallback_from_title(title_fallback)
        
    clean = re.sub(r'[,.;:!?¿¡"\'\(\)\[\]\{\}\-_/\\|]', ' ', str(text))
    words = [w for w in clean.split() if w]
    
    if len(words) > MAX_SUBTEMA_WORDS:
        words = words[:MAX_SUBTEMA_WORDS]
        
    while words and words[-1].lower() in FORBIDDEN_TRAILING_WORDS:
        words.pop()
        
    res = " ".join(words).strip()
    res_lower = res.lower()
    
    forbidden_starts = [
        "mencion de", "mencion a", "mencion en", "mencion del", "presencia de",
        "declaraciones de", "noticia sobre", "alusion a", "referencia a"
    ]
    for fs in forbidden_starts:
        if res_lower.startswith(fs):
            res = res[len(fs):].strip()
            break
            
    brand_words = set(re.findall(r"\b[a-z0-9]+\b", unidecode(brand.lower())))
    res_words = set(re.findall(r"\b[a-z0-9]+\b", unidecode(res.lower())))
    
    if (
        not res
        or res_words.issubset(brand_words)
        or _is_lead_clause_scrap(res.split())
        or _has_echoed_content_pair(res.split())
        or res_lower in ["universidad", "autonoma", "fundacion", "clinica", "hospital", "institucion", "asociacion"]
    ):
        if res_words.issubset(brand_words) or res_lower in [
            "universidad", "autonoma", "fundacion", "clinica", "hospital",
            "institucion", "asociacion",
        ] or not res:
            return _fallback_from_title(title_fallback)
        return ""
        
    return res.capitalize()

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
        if any(w in c_low for w in ["salud", "hospital", "clinica", "medico", "medicina", "paciente", "quirurg", "enfermedad", "achc"]):
            return "Sector Salud"
        if any(w in c_low for w in ["aduan", "dian", "fiscal", "tributar", "impuesto", "arancel"]):
            return "Gestión Tributaria"
        if any(w in c_low for w in ["universidad", "estudiante", "academ", "carrera", "educacion", "profesor", "beca", "uao", "feria", "inspirate"]):
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
    """Detecta de forma infalible acompañamiento, respaldo y felicitaciones."""
    c_low = unidecode(ctx.lower())
    positive_actions = [
        "celebra y respalda", "respalda el nombramiento", "respaldan el nombramiento",
        "acompanamos desde", "acompanamiento desde", "asesoria gratuita", "apoyo gratuito",
        "pusieron en marcha", "pone en marcha", "felicita a", "felicitamos a",
        "rinde homenaje", "reconocimiento destaca el compromiso", "abren espacio"
    ]
    has_positive = any(p in c_low for p in positive_actions)
    has_negative_allegation = any(n in c_low for n in ["denuncia penal", "sancion fiscal", "investigacion por corrupcion", "plagio"])
    if not has_positive or has_negative_allegation:
        return False
    if brand:
        names = [brand] + list(aliases or [])
        if not any(unidecode(n.lower()).strip() and unidecode(n.lower()) in c_low for n in names):
            return False
    return True

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
    progress_callback: Optional[Callable[[int, str], None]] = None,
    progress_pct: int = 74,
) -> Dict[int, int]:
    """Assign a group only when items are the same story.

    Allowed signals (any one is enough):
    1. Same or clearly similar Título
    2. Título starts with the same first 3 or 4 words
    3. Resumen / aclaración starts with the same first 3 or 4 words

    Shared client, región, vague tema or keyword overlap is NOT enough.
    Grouping is optional: unrelated items keep their own cluster.

    Complexity: hash maps on compact titles and first-3/4-word keys (O(n)),
    plus a bounded fuzzy pass inside small prefix buckets. Never all-pairs
    on the full sheet (Sucre xlsx is often 700–1500 rows).
    """
    _ = brand_regexes
    active_indices = [i for i in range(len(rows)) if not rows[i].get("is_duplicate")]
    if not active_indices:
        return {}

    def _emit(pct: int, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

    _emit(progress_pct, "Agrupando eventos y noticias similares (ordenamiento por titular)…")

    brand_tokens = _brand_token_set(brand, aliases)
    n_active = len(active_indices)
    dsu = _DSU(active_indices)

    compact_to_idxs: Dict[str, List[int]] = defaultdict(list)
    lead_buckets: Dict[Tuple, List[int]] = defaultdict(list)

    for i in active_indices:
        title = _row_title(rows[i], km)
        resumen = _resumen_for_grouping(_row_resumen(rows[i]))
        compact = _compact_title_key(title)
        if compact:
            compact_to_idxs[compact].append(i)
        for key in _lead_bucket_keys(title, brand_tokens):
            lead_buckets[key].append(i)
        for key in _lead_bucket_keys(resumen, brand_tokens):
            lead_buckets[("res",) + key].append(i)

    _emit(
        progress_pct,
        f"Agrupando eventos y noticias similares ({n_active} notas, claves de titular)…",
    )

    for idxs in compact_to_idxs.values():
        _dsu_union_all(dsu, idxs)
    for idxs in lead_buckets.values():
        _dsu_union_all(dsu, idxs)

    # Longer compact title that starts with a shorter one (≥24 chars).
    sorted_keys = sorted(compact_to_idxs)
    for i, ka in enumerate(sorted_keys):
        if len(ka) < _TITLE_PREFIX_MIN:
            continue
        content = [w for w in ka.split() if w not in STOPWORDS_ES]
        if not content or set(content).issubset(brand_tokens):
            continue
        head = compact_to_idxs[ka][0]
        for kb in sorted_keys[i + 1:]:
            if not kb.startswith(ka):
                break
            dsu.union(head, compact_to_idxs[kb][0])

    # fuzz.ratio only inside small prefix buckets (hyphen / typo, e.g. FICCI-UTB
    # vs FICCIUTB). Large buckets are brand-prefixed Sucre sheets — skip fuzzy
    # rather than chaining near-neighbors into one mega-group.
    fuzzy_buckets: Dict[str, List[str]] = defaultdict(list)
    for ka in compact_to_idxs:
        if len(ka) >= _TITLE_PREFIX_MIN:
            fuzzy_buckets[ka[:_FUZZY_PREFIX_CHARS]].append(ka)

    for keys in fuzzy_buckets.values():
        if len(keys) < 2 or len(keys) > _FUZZY_FULL_BUCKET:
            continue
        keys.sort()
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                if fuzz.ratio(keys[a], keys[b]) >= _FUZZY_RATIO_MIN:
                    dsu.union(compact_to_idxs[keys[a]][0], compact_to_idxs[keys[b]][0])

    _emit(
        progress_pct,
        f"Agrupando eventos y noticias similares ({n_active} notas, grupos listos)…",
    )
    return {i: dsu.find(i) for i in active_indices}


class _DSU:
    """Union-Find simple para fusionar clústers de forma transitiva y
    determinista (A~B y B~C implica A~B~C, sin importar el orden de
    comparación — el código anterior sobrescribía el mapeo par a par y podía
    dejar una fusión a medias según el orden de iteración)."""

    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def canonicalize_subtopics(
    cluster_results: Dict[int, Tuple[str, str, str]],
    cluster_contexts: Optional[Dict[int, str]] = None,
) -> Dict[int, Tuple[str, str, str]]:
    """Do not merge clusters after classification.

    Previous versions glued clusters whose LLM subtema text or contexto
    overlapped (token_set_ratio), which invented a shared subtema for
    unrelated stories. Grouping is decided only by should_group_news_items.
    """
    del cluster_contexts
    return dict(cluster_results)

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


def _title_event_hints(title: str) -> Set[str]:
    low = unidecode(title or "").lower()
    return {h for h in TITLE_EVENT_HINTS if h in low}


def _phrase_covers_title_event(phrase: str, title: str) -> bool:
    hints = _title_event_hints(title)
    if not hints:
        return True
    plow = unidecode(phrase or "").lower()
    return any(h in plow for h in hints)


def _event_phrase_from_title(title: str, brand: str) -> str:
    if not _title_event_hints(title):
        return ""
    phrase = _fallback_from_title(title)
    if _phrase_covers_title_event(phrase, title):
        brand_toks = _brand_token_set(brand)
        content = {unidecode(w.lower()) for w in phrase.split() if unidecode(w.lower()) not in STOPWORDS_ES}
        if content and not content.issubset(brand_toks):
            return phrase
    return ""


def ensure_subtema_distinct_from_tema(
    tema: str,
    subtema: str,
    brand: str,
    title: str,
    ctx: str,
    aliases: Optional[List[str]] = None,
) -> str:
    """Keep a 4–7 word fact noun phrase from this story. Never a lead-clause scrap."""
    llm_clean = clean_subtema(subtema or "", brand, title)

    def _strong(phrase: str) -> bool:
        if not phrase:
            return False
        words = phrase.split()
        if len(words) < MIN_SUBTEMA_WORDS or len(words) > MAX_SUBTEMA_WORDS:
            return False
        if _is_lead_clause_scrap(words) or _has_echoed_content_pair(words):
            return False
        if _labels_too_close(tema, phrase) or _is_title_scrap(phrase, title):
            return False
        brand_toks = _brand_token_set(brand)
        content = {unidecode(w.lower()) for w in words if unidecode(w.lower()) not in STOPWORDS_ES}
        if content and content.issubset(brand_toks):
            return False
        if not _phrase_covers_title_event(phrase, title):
            return False
        return True

    ctx_ok = bool(ctx) and str(ctx).strip() not in ("", "-")
    if _strong(llm_clean) and (
        not ctx_ok or subtema_supported_by_context(llm_clean, ctx, brand, aliases)
    ):
        completed = _complete_proper_names_from_context(llm_clean, ctx)
        completed = clean_subtema(completed, brand, title) or completed
        if _strong(completed) and (
            not ctx_ok or subtema_supported_by_context(completed, ctx, brand, aliases)
        ):
            return completed
        return llm_clean

    title_event = _event_phrase_from_title(title, brand)
    if title_event and MIN_SUBTEMA_WORDS <= len(title_event.split()) <= MAX_SUBTEMA_WORDS:
        return title_event

    ctx_phrase = _window_from_context(ctx, tema, title, brand)
    if ctx_phrase:
        ctx_phrase = _complete_proper_names_from_context(ctx_phrase, ctx)
        ctx_phrase = ctx_phrase.capitalize()
    if _strong(ctx_phrase):
        return ctx_phrase
    if ctx_phrase and not _is_lead_clause_scrap(ctx_phrase.split()) and not _labels_too_close(tema, ctx_phrase):
        if MIN_SUBTEMA_WORDS <= len(ctx_phrase.split()) <= MAX_SUBTEMA_WORDS:
            return ctx_phrase
    if llm_clean and _strong(llm_clean):
        return llm_clean
    return ctx_phrase or llm_clean or _fallback_from_title(title)


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
        steps.append(
            f'{n}. "tono": Impacto reputacional en el cliente ("{brand}"): "Positivo", "Negativo" o "Neutro".\n'
            '   REGLA DE ORO: Si el cliente expresa o recibe ACOMPAÑAMIENTO, RESPALDO, APOYO, FELICITACIONES, CELEBRACIÓN o ALIANZA, el tono es estrictamente "Positivo".'
        )
        json_fields.append('"tono": "..."')
        n += 1
    if request_theme:
        steps.append(
            f'{n}. "tema": DOMINIO GENERAL (Nivel Macro, 1 a 3 palabras. Ej: "Educación Superior", "Gestión Tributaria", "Sector Salud"). PROHIBIDO "Otros".'
        )
        json_fields.append('"tema": "..."')
        n += 1

    subtema_rule = (
        f'{n}. "subtema": HECHO ESPECÍFICO de ESTA noticia (frase nominal coherente '
        "en español colombiano, 4 a 7 palabras, completa, sin cortar nombres propios). "
        "Tómalo del título + contexto analizado de esta fila. "
        "PROHIBIDO collage de keywords, recortar el titular, copiar el tema, "
        "usar la marca del cliente como subtema, o inventar un subtema compartido "
        "para forzar agrupación con otras notas."
    )
    steps.append(subtema_rule)
    json_fields.append('"subtema": "..."')

    if request_theme:
        differ_rule = 'REGLA OBLIGATORIA: "tema" y "subtema" DEBEN SER DIFERENTES.'
    elif pkl_theme:
        differ_rule = (
            f'TEMA YA CLASIFICADO POR EL MODELO DEL CLIENTE: "{pkl_theme}". '
            "NO inventes otro tema ni lo copies como subtema. "
            "El subtema debe ser un hecho más específico y distinto a ese tema."
        )
    else:
        differ_rule = "El subtema debe describir el hecho concreto, no un dominio general."

    tone_examples = ""
    if request_tone:
        tone_examples = """
EJEMPLOS DE TONO OBLIGATORIO:
- Caso 1: "Sismo en la región: Acompañamos desde la Universidad Autónoma de Occidente a las familias afectadas..."
  -> Tono: "Positivo" (solidaridad y acompañamiento institucional de la marca).
- Caso 2: "Designación ministerial: La Universidad Autónoma de Occidente celebra y respalda el nombramiento..."
  -> Tono: "Positivo" (respaldo y felicitación institucional de la marca).
- Caso 3: "UAO y DIAN abren espacio de asesoría gratuita en trámites aduaneros..."
  -> Tono: "Positivo" (alianza y beneficio para la comunidad).
- Caso 4: "Denuncian quejas por cobros excesivos o fallas en el servicio..."
  -> Tono: "Negativo" (afectación directa).
- Caso 5: "Boletín general de cifras donde la entidad aporta un dato técnico..."
  -> Tono: "Neutro" (informativo sin juicio de valor).
"""

    prompt = f"""Analiza esta noticia para el cliente: "{brand}" (Alias: {', '.join(aliases) if aliases else 'Ninguno'}).

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
                {"role": "system", "content": "Auditor senior de medios. Clasifica el tono institucional y los hechos con alta precisión."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=140
        )
        data = json.loads(resp.choices[0].message.content)

        if request_tone:
            tono_raw = str(data.get("tono", "Neutro")).strip().capitalize()
            tono = tono_raw if tono_raw in ["Positivo", "Negativo", "Neutro"] else "Neutro"
            if check_positive_institutional_override(ctx, brand, aliases):
                tono = "Positivo"
        else:
            tono = "Neutro"

        subtema = clean_subtema(data.get("subtema", ""), brand, title_ref)

        if request_theme:
            tema = clean_tema(data.get("tema", ""))
            tema = ensure_different_tema_subtema(tema, subtema, ctx)
        else:
            tema = (pkl_theme or "").strip() or "Gestión Institucional"
            subtema = ensure_subtema_distinct_from_tema(
                tema, subtema, brand, title_ref, ctx, aliases
            )

        return tono, tema, subtema
    except Exception as e:
        logger.error(f"Error en llamada OpenAI: {e}")
        sub_fb = _fallback_from_title(title_ref)
        if request_theme:
            tema_fb = ensure_different_tema_subtema("Gestión Institucional", sub_fb, ctx)
        else:
            tema_fb = (pkl_theme or "").strip() or "Gestión Institucional"
            sub_fb = ensure_subtema_distinct_from_tema(
                tema_fb, sub_fb, brand, title_ref, ctx, aliases
            )
        tono_fb = "Positivo" if check_positive_institutional_override(ctx, brand, aliases) else "Neutro"
        return tono_fb, tema_fb, sub_fb

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

    cluster_map = cluster_similar_rows(
        rows,
        km,
        brand_regexes,
        brand=brand,
        aliases=aliases,
        progress_callback=progress_callback,
        progress_pct=74,
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
                aliases,
            )
            cluster_results[cid] = (tono, tema, subtema)
            completed += 1
            if progress_callback and (completed % 15 == 0 or completed == total_clusters):
                pct = 77 + int((completed / total_clusters) * 16)
                progress_callback(pct, f"Analizando con IA… {completed}/{total_clusters} procesados")

    cluster_contexts = {
        cid: rows[row_idx].get("Contexto analizado", "")
        for cid, row_idx in cluster_to_sample_idx.items()
    }
    cluster_results = canonicalize_subtopics(cluster_results, cluster_contexts)

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
        elif sum(1 for v in cluster_map.values() if v == cid) <= 1:
            subtema = ensure_subtema_distinct_from_tema(
                tema, subtema, brand,
                str(row.get(km.get("titulo", "Título"), "")),
                row.get("Contexto analizado", ""),
                aliases,
            )

        row["Tono_IA"] = tono
        row["Tema_IA"] = tema
        row["Subtema_IA"] = subtema
            
    return rows
