# ======================================
# Análisis Sucre (Lucy / Gobernación) — aislado del Grill
# ======================================
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from unidecode import unidecode

from ai_analyzer import extract_brand_context, generate_brand_variants
from pipeline import clean_cuerpo, clean_text

logger = logging.getLogger("sucre_analyzer")

BRAND = "Gobernación de Sucre"
LUCY_CANONICAL_NAME = "Lucy Inés García Montes"
LUCY_CANONICAL_CARGO = "Gobernadora de Sucre"

DEFAULT_ALIASES = [
    "Lucy Inés García Montes",
    "Lucy Inés García",
    "Lucy García Montes",
    "Lucy García",
    "Lucy Ines Garcia Montes",
    "gobernadora Lucy",
    "gobernadora de Sucre",
    "Gobernación de Sucre",
    "Gobernación Departamental de Sucre",
    "gobierno departamental de Sucre",
    "administración departamental de Sucre",
    "Secretaría de Educación departamental",
    "Secretaría de Educación de Sucre",
    "despacho de la gobernadora",
]

COL_TONO = "Tono"
COL_PROPIOS = "Nombre y cargo — actores propios"
COL_INT_PROPIA = "Intervención actores propios (extracto)"
COL_EXTERNOS = "Nombre y cargo — actores externos"
COL_MENCION_EXT = "Mención / intervención externa (extracto)"

SUCRE_OUTPUT_COLUMNS = [
    COL_TONO,
    COL_PROPIOS,
    COL_INT_PROPIA,
    COL_EXTERNOS,
    COL_MENCION_EXT,
]

DEFAULT_MODEL = "gpt-4.1-nano-2025-04-14"

_FILLER_RE = re.compile(
    r"^\s*(?:"
    r"\(?(?:sin\s+actor(?:es)?(?:\s+externos?|\s+propios?)?|"
    r"sin\s+intervenci[oó]n|sin\s+menci[oó]n|"
    r"no\s+aplica|n/?a|ningun[oa]|vac[ií]o|"
    r"no\s+hay(?:\s+\w+){0,4}|"
    r"no\s+se\s+(?:identifica|encontr[oó]|evidencia).*)\)?"
    r"|-|—|–|\.{1,3}"
    r")\s*$",
    re.IGNORECASE,
)

_LUCY_RE = re.compile(
    r"\b(?:lucy\s+in[eé]s\s+garc[ií]a(?:\s+montes)?|"
    r"lucy\s+garc[ií]a(?:\s+montes)?|"
    r"gobernador[a]\s+lucy(?:\s+in[eé]s)?(?:\s+garc[ií]a)?(?:\s+montes)?|"
    r"gobernador[a]\s+de\s+sucre|"
    r"la\s+mandataria(?:\s+departamental)?|"
    r"la\s+gobernador[a])\b",
    re.IGNORECASE,
)

_GOBERNACION_RE = re.compile(
    r"\b(?:gobernaci[oó]n\s+(?:departamental\s+)?de\s+sucre|"
    r"gobierno\s+departamental(?:\s+de\s+sucre)?|"
    r"administraci[oó]n\s+departamental(?:\s+de\s+sucre)?|"
    r"despacho\s+de\s+la\s+gobernador[a]|"
    r"ejecutivo\s+departamental)\b",
    re.IGNORECASE,
)

_SECRETARIA_STOP = (
    r"(?!emit|expid|anunc|firm|dij|afirm|se[nñ]al|resolv|present|lanz|"
    r"inform|destin|aprob|sancion|adopt|orden|dijo|puso|entreg|"
    r"inaugur|habilit|comunic|report|suscrib|adjudic|asign)"
)

_SECRETARIA_RE = re.compile(
    r"\bsecretar[ií]a\s+de\s+" + _SECRETARIA_STOP + r"[a-záéíóúñü]+"
    r"(?:\s+(?:y\s+)?" + _SECRETARIA_STOP + r"[a-záéíóúñü]+){0,3}"
    r"(?:\s+departamental|\s+de\s+sucre)?",
    re.IGNORECASE,
)

_SECRETARIO_PERSONA_RE = re.compile(
    r"\bsecretari[oa]\s+de\s+" + _SECRETARIA_STOP + r"[a-záéíóúñü]+"
    r"(?:\s+(?:y\s+)?" + _SECRETARIA_STOP + r"[a-záéíóúñü]+){0,3}"
    r"(?:\s+departamental|\s+de\s+sucre)?",
    re.IGNORECASE,
)

_VOCERO_RE = re.compile(
    r"\bvocer[oa]s?\s+(?:de\s+(?:la\s+)?gobernaci[oó]n(?:\s+de\s+sucre)?|"
    r"departamental(?:es)?)\b",
    re.IGNORECASE,
)

_AGENCY_VERBS_RE = re.compile(
    r"\b(?:anunci[oó]|dijo|afirm[oó]|se[nñ]al[oó]|indic[oó]|asegur[oó]|"
    r"entreg[oó]|inaugur[oó]|sancion[oó]|firm[oó]|emiti[oó]|expidi[oó]|"
    r"resolvi[oó]|destac[oó]|present[oó]|lanz[oó]|explic[oó]|inform[oó]|"
    r"confirm[oó]|rechaz[oó]|pidi[oó]|solicit[oó]|advirti[oó]|resalt[oó]|"
    r"reiter[oó]|sostuvo|manifest[oó]|declar[oó]|instal[oó]|abri[oó]|"
    r"destin[oó]|invirti[oó]|gestion[oó]|lider[oó]|orden[oó]|dispuso|"
    r"reglament[oó]|adopt[oó]|aprob[oó]|puso\s+en\s+marcha|"
    r"dieron\s+a\s+conocer|dio\s+a\s+conocer|comunic[oó]|report[oó]|"
    r"habilit[oó]|destinaron|suscribi[oó]|adjudic[oó]|asign[oó]|"
    r"expide|emite|anuncia|entrega|inaugura)\b",
    re.IGNORECASE,
)

_SPEECH_VERBS_RE = re.compile(
    r"\b(?:dijo|afirm[oó]|se[nñ]al[oó]|indic[oó]|asegur[oó]|explic[oó]|"
    r"destac[oó]|denunci[oó]|cuestion[oó]|critic[oó]|felicit[oó]|"
    r"rechaz[oó]|pidi[oó]|solicit[oó]|advirti[oó]|sostuvo|manifest[oó]|"
    r"declar[oó]|reproch[oó]|exigi[oó]|agradec[ioó]|respald[oó]|"
    r"cuestionaron|criticaron|denunciaron|pidieron|señalaron)\b",
    re.IGNORECASE,
)

_NEG_RE = re.compile(
    r"\b(?:denuncia(?:n|ron)?|corrupci[oó]n|esc[aá]ndalo|investigaci[oó]n\s+penal|"
    r"incumplimiento|abandono|captura|demanda|cuestion(?:a|an|aron)?|"
    r"critic(?:a|an|aron)?|irregularidad(?:es)?|desfalco|peculado|"
    r"hallazgos?\s+fiscales|sanci[oó]n\s+fiscal|no\s+ha\s+ejecutado|"
    r"retras(?:o|os)|crisis|desgobierno|negligencia)\b",
    re.IGNORECASE,
)

_POS_RE = re.compile(
    r"\b(?:inaugur[oó]|entreg[oó]|inversi[oó]n|becas?|beneficio|"
    r"felicit(?:a|an|aron|ó)|logro|avances?|mejoras?|"
    r"obra(?:s)?|hospital|cobertura|calidad\s+educativa|"
    r"alianza|acompa[nñ]a(?:mos|miento)?|respaldo|homenaje|"
    r"puesta\s+en\s+marcha|programa\s+social)\b",
    re.IGNORECASE,
)

_ROLE_HINT_RE = re.compile(
    r"\b(?:senador(?:a)?|representante|alcalde(?:sa)?|concejal(?:a)?|"
    r"ministro|ministra|gobernador(?:a)?|presidente|presidenta|"
    r"contralor(?:a)?|procurador(?:a)?|fiscal|diputad[oa]|"
    r"director(?:a)?|rector(?:a)?|vocer[oa]|defensor(?:a)?|"
    r"l[ií]der|dirigente|congresista)\b",
    re.IGNORECASE,
)


def empty_result() -> Dict[str, str]:
    return {
        COL_TONO: "Neutro",
        COL_PROPIOS: "",
        COL_INT_PROPIA: "",
        COL_EXTERNOS: "",
        COL_MENCION_EXT: "",
    }


def is_filler(text: str) -> bool:
    if text is None:
        return True
    s = str(text).strip()
    if not s:
        return True
    return bool(_FILLER_RE.match(s))


def normalize_cell_text(val) -> str:
    if val is None:
        return ""
    if isinstance(val, dict):
        val = val.get("value", "")
    if isinstance(val, float):
        try:
            import math
            if math.isnan(val):
                return ""
        except Exception:
            pass
    s = str(val).strip()
    if s.lower() in ("nan", "none", "null"):
        return ""
    return s


def article_sources(titulo: str, cuerpo: str) -> Tuple[str, str, str]:
    title = clean_text(normalize_cell_text(titulo)) if titulo else ""
    body = clean_cuerpo(normalize_cell_text(cuerpo))
    combined = f"{title}. {body}".strip() if title else body
    return title, body, combined


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _is_filler_or_empty(text: str) -> bool:
    return is_filler(text)


def recover_verbatim(candidate: str, *sources: str) -> str:
    """Devuelve el tramo literal del origen si `candidate` coincide; si no, ''."""
    raw = (candidate or "").strip()
    if _is_filler_or_empty(raw):
        return ""

    for source in sources:
        if source and raw in source:
            return raw

    stripped = raw.strip(" «»\"'`")
    if stripped and stripped != raw:
        for source in sources:
            if source and stripped in source:
                return stripped

    for source in sources:
        span = _recover_ws_flexible(raw, source)
        if span:
            return span
        if stripped and stripped != raw:
            span = _recover_ws_flexible(stripped, source)
            if span:
                return span

    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", raw) if p.strip()]
    if len(parts) > 1:
        recovered = []
        for part in parts:
            got = ""
            for source in sources:
                if part in source:
                    got = part
                    break
                got = _recover_ws_flexible(part, source)
                if got:
                    break
            if got:
                recovered.append(got)
        if recovered:
            return " ".join(recovered)
    return ""


def _recover_ws_flexible(candidate: str, source: str) -> str:
    if not candidate or not source:
        return ""
    cand = collapse_ws(candidate)
    if not cand:
        return ""

    norm_chars: List[str] = []
    idx_map: List[int] = []
    prev_space = False
    for i, ch in enumerate(source):
        if ch.isspace():
            if not prev_space and norm_chars:
                norm_chars.append(" ")
                idx_map.append(i)
            prev_space = True
        else:
            norm_chars.append(ch)
            idx_map.append(i)
            prev_space = False
    norm = "".join(norm_chars)
    if not norm:
        return ""

    pos = norm.lower().find(cand.lower())
    if pos < 0:
        pos = unidecode(norm).lower().find(unidecode(cand).lower())
        if pos < 0:
            return ""
    end_idx = pos + len(cand) - 1
    if end_idx >= len(idx_map) or pos >= len(idx_map):
        # longitud distinta por unidecode; buscar de nuevo en unidecode map
        return _recover_unidecode_span(cand, source)
    start = idx_map[pos]
    end = idx_map[end_idx] + 1
    return source[start:end]


def _recover_unidecode_span(cand: str, source: str) -> str:
    src_u = unidecode(source)
    cand_u = unidecode(cand)
    pos = src_u.lower().find(cand_u.lower())
    if pos < 0:
        return ""
    # Aprox. 1:1 para textos en español con tildes (unidecode acorta poco)
    end = min(len(source), pos + len(cand))
    return source[pos:end]


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def mentions_lucy(text: str) -> bool:
    return bool(text and _LUCY_RE.search(text))


def mentions_gobernacion(text: str) -> bool:
    return bool(text and _GOBERNACION_RE.search(text))


def mentions_own_entity(text: str) -> bool:
    if not text:
        return False
    return bool(
        mentions_lucy(text)
        or mentions_gobernacion(text)
        or _SECRETARIA_RE.search(text)
        or _SECRETARIO_PERSONA_RE.search(text)
        or _VOCERO_RE.search(text)
    )


def _has_own_agency(sentence: str) -> bool:
    """La Gobernación / Lucy es sujeto de un decir o un hacer, no el objeto de un tercero."""
    if not mentions_own_entity(sentence):
        return False
    low = unidecode(sentence.lower())
    if re.search(r"\bsegun\s+la\s+gobernador", low) or re.search(r"\bsegun\s+la\s+gobernacion", low):
        return True
    if re.search(r"\b(?:emitida|expedida|firmada|sancionada|anunciada)\s+por\b", low):
        return True
    if "resolucion" in low and mentions_gobernacion(sentence) and _AGENCY_VERBS_RE.search(sentence):
        verb = _AGENCY_VERBS_RE.search(sentence)
        return mentions_own_entity(sentence[: verb.start()]) if verb else True

    verb = _AGENCY_VERBS_RE.search(sentence)
    if not verb:
        return False
    before = sentence[: verb.start()]
    if mentions_own_entity(before):
        return True
    # Inversión: «…», afirmó la gobernadora / la mandataria
    after = sentence[verb.end():].lstrip(" ,")
    after_u = unidecode(after).lower()
    return bool(re.match(
        r"^(?:la\s+)?(?:gobernador[a]|mandataria|gobernacion)\b",
        after_u,
    ))


def _external_speaker_in(sentence: str) -> bool:
    """Hay un actor que no es Lucy/Gobernación hablando en la oración."""
    if not _SPEECH_VERBS_RE.search(sentence) and not _ROLE_HINT_RE.search(sentence):
        return False
    if _has_own_agency(sentence) and not _ROLE_HINT_RE.search(sentence):
        return False
    # Rol institucional ajeno (alcalde, senador, contralor, etc.)
    for m in _ROLE_HINT_RE.finditer(sentence):
        role = m.group(0)
        if re.match(r"gobernador", role, re.I):
            continue
        window = sentence[max(0, m.start() - 12): min(len(sentence), m.end() + 8)]
        if mentions_lucy(window) or re.search(r"gobernador[a]\s+de\s+sucre", window, re.I):
            continue
        return True
    # Nombre propio + verbo de habla, y mención a la marca
    if mentions_own_entity(sentence) and _SPEECH_VERBS_RE.search(sentence):
        before_verb = sentence[: _SPEECH_VERBS_RE.search(sentence).start()]
        if mentions_own_entity(before_verb) and not _ROLE_HINT_RE.search(before_verb):
            return False
        if re.search(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+){1,3}\b", before_verb):
            return True
    return False


def _span_from_sentences(source: str, sentences: Sequence[str]) -> str:
    kept = [s for s in sentences if s and s in source]
    if not kept:
        # intentar recuperación flexible oración a oración
        recovered = [recover_verbatim(s, source) for s in sentences]
        recovered = [s for s in recovered if s]
        return " ".join(recovered)
    first, last = kept[0], kept[-1]
    start = source.find(first)
    end = source.find(last)
    if start < 0 or end < 0:
        return " ".join(kept)
    return source[start: end + len(last)]


def _format_actors(actors: List[Tuple[str, str]]) -> str:
    seen = set()
    parts = []
    for name, cargo in actors:
        name = collapse_ws(name or "")
        cargo = collapse_ws(cargo or "")
        if not name or is_filler(name):
            continue
        key = unidecode(f"{name}|{cargo}").lower()
        if key in seen:
            continue
        seen.add(key)
        if cargo and not is_filler(cargo):
            parts.append(f"{name}, {cargo}")
        else:
            parts.append(name)
    return "; ".join(parts)


def _own_actors_from_sentence(sentence: str) -> List[Tuple[str, str]]:
    actors: List[Tuple[str, str]] = []
    if mentions_lucy(sentence):
        actors.append((LUCY_CANONICAL_NAME, LUCY_CANONICAL_CARGO))
    for m in _SECRETARIA_RE.finditer(sentence):
        label = collapse_ws(m.group(0))
        actors.append((label[0].upper() + label[1:] if label else label, "Gobernación de Sucre"))
    for m in _SECRETARIO_PERSONA_RE.finditer(sentence):
        cargo = collapse_ws(m.group(0))
        name = _person_name_near(sentence, m.start())
        if name:
            actors.append((name, cargo[0].upper() + cargo[1:] if cargo else cargo))
        elif not any(a[0].lower().startswith("secretar") for a in actors):
            actors.append((cargo[0].upper() + cargo[1:], "Gobernación de Sucre"))
    for m in _VOCERO_RE.finditer(sentence):
        cargo = collapse_ws(m.group(0))
        name = _person_name_near(sentence, m.start())
        actors.append((name or cargo[0].upper() + cargo[1:], "Gobernación de Sucre" if name else ""))
    if mentions_gobernacion(sentence) and not actors:
        actors.append((BRAND, ""))
    return actors


def _person_name_near(sentence: str, idx: int) -> str:
    before = sentence[max(0, idx - 80): idx]
    # "Juan Pérez, secretario..."
    m = re.search(
        r"([A-ZÁÉÍÓÚÑ][a-záéíóúñü]+(?:\s+(?:de\s+)?[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+){0,3})\s*,\s*$",
        before,
    )
    if m:
        return collapse_ws(m.group(1))
    after = sentence[idx: min(len(sentence), idx + 80)]
    m = re.search(
        r",\s*([A-ZÁÉÍÓÚÑ][a-záéíóúñü]+(?:\s+(?:de\s+)?[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+){0,3})",
        after,
    )
    if m:
        return collapse_ws(m.group(1))
    return ""


def _external_actors_from_sentence(sentence: str) -> List[Tuple[str, str]]:
    actors: List[Tuple[str, str]] = []
    # "El senador X" / "la alcaldesa Y"
    for m in _ROLE_HINT_RE.finditer(sentence):
        role = collapse_ws(m.group(0))
        if re.match(r"gobernador", role, re.I):
            continue
        window_before = sentence[max(0, m.start() - 8): m.end() + 1]
        if mentions_lucy(window_before) or re.search(r"gobernador[a]\s+de\s+sucre", window_before, re.I):
            continue
        after = sentence[m.end(): m.end() + 90]
        name_m = re.search(
            r"^\s*(?:de\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+){0,2})\s*,?\s*)?"
            r"([A-ZÁÉÍÓÚÑ][a-záéíóúñü]+(?:\s+(?:de\s+)?[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+){0,3})?",
            after,
        )
        name = ""
        extra_role = ""
        if name_m:
            place = collapse_ws(name_m.group(1) or "")
            person = collapse_ws(name_m.group(2) or "")
            if place:
                extra_role = f"{role} de {place}"
            name = person
        cargo = extra_role or (role[0].upper() + role[1:] if role else "")
        if name and not mentions_own_entity(name) and not _ROLE_HINT_RE.match(name):
            actors.append((name, cargo))
        elif cargo and not mentions_lucy(cargo):
            actors.append((cargo, ""))
    return actors


def heuristic_tone(title: str, body: str) -> str:
    blob = f"{title} {body}"
    pos = bool(_POS_RE.search(blob))
    neg = bool(_NEG_RE.search(blob))
    if neg and not pos:
        return "Negativo"
    if pos and not neg:
        return "Positivo"
    if neg and pos:
        # acusación suele pesar más si no hay agencia propia positiva
        own = [s for s in split_sentences(blob) if _has_own_agency(s)]
        own_text = " ".join(own)
        if own_text and _POS_RE.search(own_text) and not _NEG_RE.search(own_text):
            return "Positivo"
        return "Negativo"
    return "Neutro"


def heuristic_analyze(titulo: str, cuerpo: str) -> Dict[str, str]:
    title, body, combined = article_sources(titulo, cuerpo)
    result = empty_result()
    if not combined:
        return result

    body_sents = split_sentences(body)
    title_sents = split_sentences(title) if title else []

    own_body = [s for s in body_sents if _has_own_agency(s)]
    own_title = [s for s in title_sents if _has_own_agency(s)]
    own_agency = own_body or own_title
    lucy_agency = [s for s in own_agency if mentions_lucy(s)]
    # Prioridad: lo que Lucy dijo/hizo; si no interviene, la Gobernación / secretarías
    own_sents = lucy_agency or own_agency
    extract_source = body if own_body else combined

    ext_sents = [
        s for s in body_sents
        if mentions_own_entity(s) and _external_speaker_in(s)
    ]
    mention_sents = [
        s for s in body_sents
        if mentions_own_entity(s) and s not in own_sents and s not in ext_sents
    ]

    own_actors: List[Tuple[str, str]] = []
    for s in own_sents:
        own_actors.extend(_own_actors_from_sentence(s))

    ext_actors: List[Tuple[str, str]] = []
    for s in ext_sents:
        ext_actors.extend(_external_actors_from_sentence(s))

    own_span = _span_from_sentences(extract_source, own_sents) if own_sents else ""
    if not own_span and own_sents:
        own_span = " ".join(own_sents)

    if ext_sents:
        ext_span = _span_from_sentences(body or combined, ext_sents)
    elif mention_sents and not own_sents:
        ext_span = _span_from_sentences(body or combined, mention_sents)
    else:
        ext_span = ""

    result[COL_TONO] = heuristic_tone(title, body)
    result[COL_PROPIOS] = _format_actors(own_actors)
    result[COL_INT_PROPIA] = recover_verbatim(own_span, combined, body, title)
    result[COL_EXTERNOS] = _format_actors(ext_actors)
    result[COL_MENCION_EXT] = recover_verbatim(ext_span, combined, body, title)
    return result


def _sanitize_names(raw: str) -> str:
    if is_filler(raw):
        return ""
    text = collapse_ws(str(raw or ""))
    # quitar notas editoriales entre paréntesis que no son cargo
    text = re.sub(r"\((?:sin\s+|no\s+|extracto|nota|columna)[^)]*\)", "", text, flags=re.I)
    return collapse_ws(text)


def merge_analysis(llm: Optional[Dict[str, str]], heuristic: Dict[str, str], titulo: str, cuerpo: str) -> Dict[str, str]:
    title, body, combined = article_sources(titulo, cuerpo)
    sources = (combined, body, title)
    base = dict(heuristic)
    if not llm:
        return base

    tono = str(llm.get(COL_TONO) or llm.get("tono") or "").strip().capitalize()
    if tono in ("Positivo", "Negativo", "Neutro"):
        base[COL_TONO] = tono

    propios = _sanitize_names(llm.get(COL_PROPIOS) or llm.get("nombre_cargo_propios") or "")
    if propios:
        base[COL_PROPIOS] = propios

    externos = _sanitize_names(llm.get(COL_EXTERNOS) or llm.get("nombre_cargo_externos") or "")
    if externos:
        base[COL_EXTERNOS] = externos
    elif COL_EXTERNOS in llm or "nombre_cargo_externos" in llm:
        # LLM vació explícitamente
        if not _sanitize_names(externos):
            base[COL_EXTERNOS] = ""

    int_propia_raw = llm.get(COL_INT_PROPIA) or llm.get("intervencion_propia") or ""
    int_propia = recover_verbatim(str(int_propia_raw), *sources)
    if int_propia:
        base[COL_INT_PROPIA] = int_propia
    elif int_propia_raw and not is_filler(str(int_propia_raw)):
        # paráfrasis: caer a heurística ya presente
        pass

    menc_raw = llm.get(COL_MENCION_EXT) or llm.get("mencion_externa") or ""
    menc = recover_verbatim(str(menc_raw), *sources)
    if menc:
        base[COL_MENCION_EXT] = menc
    elif menc_raw and not is_filler(str(menc_raw)):
        pass

    # Nunca rellenar con placeholders
    for col in (COL_PROPIOS, COL_INT_PROPIA, COL_EXTERNOS, COL_MENCION_EXT):
        if is_filler(base.get(col, "")):
            base[col] = ""
    return base


def _call_openai_sucre(client, model: str, brand: str, aliases: List[str], ctx: str, title: str, body: str) -> Dict[str, str]:
    alias_txt = ", ".join(aliases) if aliases else "Ninguno"
    prompt = f"""Analiza esta noticia para la marca principal: "{brand}" (Gobernadora {LUCY_CANONICAL_NAME}).
Alias a reconocer: {alias_txt}.

Titular:
\"\"\"{title}\"\"\"

Cuerpo:
\"\"\"{body}\"\"\"

Contexto anclado a la marca (oraciones donde aparece Lucy / Gobernación / secretarías):
\"\"\"{ctx}\"\"\"

Debes devolver JSON con:
1. "tono": impacto reputacional SOBRE Lucy Inés García Montes y/o la Gobernación de Sucre: "Positivo", "Negativo" o "Neutro".
   Si la mandataria o la Gobernación anuncian obras, entregas, inversión, respaldo o alianzas → Positivo.
   Si hay denuncias, investigaciones, incumplimiento o críticas dirigidas a ellas → Negativo.
   Hecho informativo sin juicio → Neutro.
2. "nombre_cargo_propios": nombre y cargo SOLO si el actor que habla o actúa es Lucy Inés García Montes (Gobernadora de Sucre) o personas/entidades de la Gobernación de Sucre (secretarías, voceros departamentales, despacho). Varios se separan con "; ".
3. "intervencion_propia": EXTRACTO LITERAL del Cuerpo (o del Título si hace falta) de lo que Lucy dijo/hizo; si ella no interviene, de lo que la Gobernación de Sucre dijo/hizo (resoluciones, acciones, comunicados).
4. "nombre_cargo_externos": nombre y cargo de personas/entidades ajenas que hablan de o se refieren a Lucy / Gobernación de Sucre. Si no hay, cadena vacía.
5. "mencion_externa": EXTRACTO LITERAL (A) del texto de la nota que menciona a Lucy/Gobernación cuando no es intervención propia, o (B) de la intervención del actor externo refiriéndose a ellas.

REGLAS ESTRICTAS DE EXTRACTOS:
- Copia un tramo LITERAL del Cuerpo o del Título. Prohibido parafrasear, resumir o reordenar.
- Prohibido intros ("Extracto:", "La nota dice:"), notas editoriales, comillas que no estén en el origen y paréntesis explicativos.
- Prohibido rellenar: no escribas "(sin actor externo)", "N/A", "ninguno", "no aplica". Si no hay dato, usa "".

Responde estrictamente en JSON:
{{"tono": "...", "nombre_cargo_propios": "...", "intervencion_propia": "...", "nombre_cargo_externos": "...", "mencion_externa": "..."}}"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Auditor de medios en Colombia. Extrae tono institucional e intervenciones "
                    "con citas literales; nunca parafrasees los extractos."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=700,
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    return {
        COL_TONO: str(data.get("tono", "")).strip(),
        COL_PROPIOS: str(data.get("nombre_cargo_propios", "")).strip(),
        COL_INT_PROPIA: str(data.get("intervencion_propia", "")).strip(),
        COL_EXTERNOS: str(data.get("nombre_cargo_externos", "")).strip(),
        COL_MENCION_EXT: str(data.get("mencion_externa", "")).strip(),
    }


def analyze_article(
    titulo: str,
    cuerpo: str,
    client=None,
    model: str = DEFAULT_MODEL,
    brand: str = BRAND,
    aliases: Optional[List[str]] = None,
    brand_regexes: Optional[List[str]] = None,
) -> Dict[str, str]:
    aliases = aliases if aliases is not None else list(DEFAULT_ALIASES)
    heuristic = heuristic_analyze(titulo, cuerpo)
    if client is None:
        return heuristic

    title, body, _ = article_sources(titulo, cuerpo)
    regexes = brand_regexes or generate_brand_variants(brand, aliases)
    ctx = extract_brand_context(body, title, regexes)
    try:
        llm = _call_openai_sucre(client, model, brand, aliases, ctx, title, body)
    except Exception:
        logger.exception("Fallo OpenAI en análisis Sucre; se usa heurística")
        return heuristic
    return merge_analysis(llm, heuristic, titulo, cuerpo)


def enrich_sucre_rows(
    rows: List[dict],
    titulo_key: str,
    cuerpo_key: str,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    max_workers: int = 10,
) -> List[dict]:
    aliases = list(DEFAULT_ALIASES)
    brand_regexes = generate_brand_variants(BRAND, aliases)
    client = None
    if api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

    n = len(rows)
    if progress_callback:
        progress_callback(72, f"Analizando {n} notas (tono e intervenciones)…")

    def _one(idx_row):
        idx, row = idx_row
        titulo = row.get(titulo_key, "")
        cuerpo = row.get(cuerpo_key, "")
        result = analyze_article(
            titulo,
            cuerpo,
            client=client,
            model=model,
            brand=BRAND,
            aliases=aliases,
            brand_regexes=brand_regexes,
        )
        return idx, result

    completed = 0
    if n == 0:
        return rows

    with ThreadPoolExecutor(max_workers=max_workers if client else 1) as executor:
        futures = [executor.submit(_one, (i, row)) for i, row in enumerate(rows)]
        for fut in as_completed(futures):
            idx, result = fut.result()
            rows[idx].update(result)
            completed += 1
            if progress_callback and (completed % 8 == 0 or completed == n):
                pct = 72 + int((completed / n) * 20)
                progress_callback(pct, f"Analizando notas Sucre… {completed}/{n}")

    return rows
