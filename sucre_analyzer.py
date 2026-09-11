# ======================================
# Análisis Sucre — actores (personas) + extractos literales
# Se suma al pipeline Grill; no reemplaza tono/tema/subtema.
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
LUCY_CANONICAL = f"{LUCY_CANONICAL_NAME}, {LUCY_CANONICAL_CARGO}"

DEFAULT_ALIASES = [
    "Lucy Inés García Montes",
    "Lucy Inés García",
    "Lucy García Montes",
    "Lucy Montes",
    "Lucy García",
    "Lucy Ines Garcia Montes",
    "gobernadora Lucy",
    "gobernadora de Sucre",
    "Gobernadora de Sucre",
    "Gobernación de Sucre",
    "Gobernación Departamental de Sucre",
    "gobierno departamental de Sucre",
    "administración departamental de Sucre",
    "despacho de la gobernadora",
]

COL_PROPIOS = "Nombre y cargo — actores propios"
COL_INT_PROPIA = "Intervención actores propios (extracto)"
COL_EXTERNOS = "Nombre y cargo — actores externos"
COL_MENCION_EXT = "Mención / intervención externa (extracto)"

# Alias histórico: el Grill ya exporta Tono_IA; no se añade otra columna Tono.
COL_TONO = "Tono"

SUCRE_ACTOR_COLUMNS = [
    COL_PROPIOS,
    COL_INT_PROPIA,
    COL_EXTERNOS,
    COL_MENCION_EXT,
]
SUCRE_OUTPUT_COLUMNS = list(SUCRE_ACTOR_COLUMNS)

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

# Persona Lucy / gobernadora (no la entidad Gobernación).
_LUCY_PERSON_RE = re.compile(
    r"\b(?:"
    r"lucy\s+in[eé]s\s+garc[ií]a(?:\s+montes)?|"
    r"lucy\s+garc[ií]a(?:\s+montes)?|"
    r"lucy\s+montes|"
    r"lucy\s+in[eé]s|"
    r"gobernador[a]\s+lucy(?:\s+in[eé]s)?(?:\s+garc[ií]a)?(?:\s+montes)?|"
    r"gobernador[a]\s+de\s+sucre"
    r")\b",
    re.IGNORECASE,
)

# «la gobernadora» / «la mandataria» solo si el artículo ancla a Sucre.
_LUCY_ROLE_LOOSE_RE = re.compile(
    r"\b(?:la\s+gobernador[a]|la\s+mandataria(?:\s+departamental)?)\b",
    re.IGNORECASE,
)

_GOBERNACION_RE = re.compile(
    r"\b(?:gobernaci[oó]n\s+(?:departamental\s+)?de\s+sucre|"
    r"gobierno\s+departamental(?:\s+de\s+sucre)?|"
    r"administraci[oó]n\s+departamental(?:\s+de\s+sucre)?|"
    r"despacho\s+de\s+la\s+gobernador[a])\b",
    re.IGNORECASE,
)

# Entidad «Secretaría de …» (no persona).
_SECRETARIA_ENTITY_RE = re.compile(
    r"\bsecretar[ií]a\s+de\s+[a-záéíóúñü]+"
    r"(?:\s+(?:y\s+)?[a-záéíóúñü]+){0,4}"
    r"(?:\s+departamental|\s+de\s+sucre)?",
    re.IGNORECASE,
)

_ENTITY_ONLY_HEADS = (
    r"secretar[ií]a(?:\s+de)?",
    r"gobernaci[oó]n(?:\s+de)?",
    r"ministerio(?:\s+de(?:l)?)?",
    r"presidencia(?:\s+de(?:\s+la)?)?",
    r"alcald[ií]a(?:\s+de)?",
    r"contralor[ií]a",
    r"procuradur[ií]a",
    r"fiscal[ií]a",
    r"congreso",
    r"gobierno(?:\s+(?:nacional|departamental|de))?",
    r"departamento(?:\s+de)?",
    r"instituto(?:\s+de)?",
    r"direcci[oó]n(?:\s+de)?",
)

_ENTITY_ONLY_RE = re.compile(
    r"^\s*(?:la\s+|el\s+|las\s+|los\s+)?"
    r"(?:" + "|".join(_ENTITY_ONLY_HEADS) + r")"
    r"\b",
    re.IGNORECASE,
)

_FORBIDDEN_ENTITY_EXACT = {
    "secretaria de educacion departamental",
    "secretaria de educacion de sucre",
    "gobernacion de sucre",
    "gobernacion departamental de sucre",
    "ministerio del interior",
    "presidencia de la republica",
    "gobierno nacional",
    "casa de nariño",
    "casa de narino",
}

_AGENCY_VERBS_RE = re.compile(
    r"\b(?:anunci[oó]|dijo|afirm[oó]|se[nñ]al[oó]|indic[oó]|asegur[oó]|"
    r"entreg[oó]|inaugur[oó]|sancion[oó]|firm[oó]|emiti[oó]|expidi[oó]|"
    r"resolvi[oó]|destac[oó]|present[oó]|lanz[oó]|explic[oó]|inform[oó]|"
    r"confirm[oó]|rechaz[oó]|pidi[oó]|solicit[oó]|advirti[oó]|resalt[oó]|"
    r"reiter[oó]|sostuvo|manifest[oó]|declar[oó]|instal[oó]|abri[oó]|"
    r"destin[oó]|invirti[oó]|gestion[oó]|lider[oó]|orden[oó]|dispuso|"
    r"reglament[oó]|adopt[oó]|aprob[oó]|puso\s+en\s+marcha|"
    r"dieron\s+a\s+conocer|dio\s+a\s+conocer|comunic[oó]|report[oó]|"
    r"habilit[oó]|suscribi[oó]|adjudic[oó]|asign[oó]|"
    r"expide|emite|anuncia|entrega|inaugura|lidera)\b",
    re.IGNORECASE,
)

_SPEECH_VERBS_RE = re.compile(
    r"\b(?:dijo|afirm[oó]|se[nñ]al[oó]|indic[oó]|asegur[oó]|explic[oó]|"
    r"destac[oó]|denunci[oó]|cuestion[oó]|critic[oó]|felicit[oó]|"
    r"rechaz[oó]|pidi[oó]|solicit[oó]|advirti[oó]|sostuvo|manifest[oó]|"
    r"declar[oó]|reproch[oó]|exigi[oó]|agradec[ioó]|respald[oó]|"
    r"opin[oó]|consider[oó]|calific[oó]|cuestionaron|criticaron|"
    r"denunciaron|pidieron|se[nñ]alaron)\b",
    re.IGNORECASE,
)

_OBJECT_PREP_RE = re.compile(
    r"\b(?:a|ante|contra|sobre|hacia|de|del)\s+(?:la\s+)?(?:gobernador[a]|mandataria|lucy)\b",
    re.IGNORECASE,
)

_ROLE_HINT_RE = re.compile(
    r"\b(?:senador(?:a)?|representante(?:\s+a\s+la\s+c[aá]mara)?|"
    r"alcalde(?:sa)?|concejal(?:a)?|ministro|ministra|"
    r"gobernador(?:a)?|presidente|presidenta|"
    r"contralor(?:a)?|procurador(?:a)?|fiscal|diputad[oa]|"
    r"director(?:a)?|rector(?:a)?|vocer[oa]|defensor(?:a)?|"
    r"l[ií]der|dirigente|congresista|magistrad[oa])\b",
    re.IGNORECASE,
)

# «Nombre Apellido, Cargo» o «El cargo Nombre Apellido».
_NAME_TOKEN = r"[A-ZÁÉÍÓÚÑ][a-záéíóúñü']+"
_NAME_RE = re.compile(
    rf"({_NAME_TOKEN}(?:\s+(?:de\s+l[oa]s?\s+|de\s+|del\s+)?{_NAME_TOKEN}){{1,4}})"
)

_SECRETARY_NAME_FIRST_RE = re.compile(
    rf"(?P<name>{_NAME_TOKEN}(?:\s+(?:de\s+l[oa]s?\s+|de\s+|del\s+)?{_NAME_TOKEN}){{1,4}})"
    r"\s*,\s*"
    r"(?P<role>secretari[oa]\s+de\s+[^.,;]{3,70})",
    re.IGNORECASE,
)

_SECRETARY_ROLE_FIRST_RE = re.compile(
    r"(?:el|la)\s+"
    r"(?P<role>secretari[oa]\s+de\s+[^.,;]{3,50}?)"
    r"\s*,\s*"
    rf"(?P<name>{_NAME_TOKEN}(?:\s+(?:de\s+l[oa]s?\s+|de\s+|del\s+)?{_NAME_TOKEN}){{1,4}})",
    re.IGNORECASE,
)

_SECRETARY_ROLE_THEN_NAME_RE = re.compile(
    r"(?:el|la)\s+"
    r"(?P<role>secretari[oa]\s+de\s+[a-záéíóúñü]+(?:\s+y\s+[a-záéíóúñü]+){0,2}"
    r"(?:\s+(?:de\s+sucre|departamental))?)"
    rf"\s+(?P<name>{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{1,3}})",
    re.IGNORECASE,
)

_OTHER_DEPT_RE = re.compile(
    r"\b(?:antioquia|bol[ií]var|c[oó]rdoba|atl[aá]ntico|magdalena|cesar|"
    r"guajira|santander|cundinamarca|bogot[aá]|valle|nari[nñ]o|huila|"
    r"tolima|meta|choco|choc[oó]|caldas|risaralda|quindio|quind[ií]o)\b",
    re.IGNORECASE,
)

_SUCRE_ANCHOR_RE = re.compile(
    r"\b(?:sucre|sincelejo|gobernaci[oó]n\s+de\s+sucre|lucy)\b",
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


def _norm(text: str) -> str:
    return unidecode(collapse_ws(text or "")).lower()


def recover_verbatim(candidate: str, *sources: str) -> str:
    """Devuelve el tramo literal del origen si `candidate` coincide; si no, ''."""
    raw = (candidate or "").strip()
    if is_filler(raw):
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
    end = min(len(source), pos + len(cand))
    return source[pos:end]


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def mentions_gobernacion(text: str) -> bool:
    return bool(text and _GOBERNACION_RE.search(text))


def article_anchors_sucre(title: str, body: str) -> bool:
    return bool(_SUCRE_ANCHOR_RE.search(f"{title} {body}"))


def mentions_lucy_person(text: str, *, sucre_context: bool = True) -> bool:
    if not text:
        return False
    if _LUCY_PERSON_RE.search(text):
        return True
    if sucre_context and _LUCY_ROLE_LOOSE_RE.search(text):
        return True
    return False


def mentions_brand_or_lucy(text: str, *, sucre_context: bool = True) -> bool:
    return mentions_lucy_person(text, sucre_context=sucre_context) or mentions_gobernacion(text)


def is_entity_only_label(text: str) -> bool:
    raw = collapse_ws(text or "")
    if not raw or is_filler(raw):
        return True
    n = _norm(raw)
    n = re.sub(r"\s*,\s*", " ", n)
    if n in _FORBIDDEN_ENTITY_EXACT:
        return True
    # «Nombre, cargo» se valida por la parte del nombre.
    name_part = raw.split(",", 1)[0].strip()
    name_n = _norm(name_part)
    if name_n in _FORBIDDEN_ENTITY_EXACT:
        return True
    if _ENTITY_ONLY_RE.match(name_part):
        return True
    return False


def looks_like_person_name(name: str) -> bool:
    name = collapse_ws(name or "")
    if not name or is_filler(name) or is_entity_only_label(name):
        return False
    if _ROLE_HINT_RE.fullmatch(name) or re.match(r"^secretari[oa]\b", name, re.I):
        return False
    tokens = name.split()
    if len(tokens) < 2:
        return False
    caps = sum(1 for t in tokens if t[:1].isupper() or t[:1] in "ÁÉÍÓÚÑ")
    return caps >= 2


def _title_case_role(role: str) -> str:
    role = collapse_ws(role or "")
    if not role:
        return ""
    small = {"de", "del", "la", "las", "los", "y", "e", "a"}
    parts = []
    for i, w in enumerate(role.split()):
        low = w.lower()
        if i > 0 and low in small:
            parts.append(low)
        else:
            parts.append(w[:1].upper() + w[1:] if w else w)
    return " ".join(parts)


def _format_person_role(name: str, cargo: str) -> str:
    name = collapse_ws(name)
    cargo = _title_case_role(cargo)
    if not name:
        return ""
    if cargo:
        return f"{name}, {cargo}"
    return name


def _is_lucy_name_or_role(name: str, cargo: str = "") -> bool:
    blob = f"{name} {cargo}"
    return mentions_lucy_person(blob, sucre_context=True) or _norm(name).startswith("lucy")


def _person_is_agent(sentence: str, matcher) -> bool:
    """True si la persona es sujeto de un verbo de agencia/habla, no el objeto."""
    if not sentence or not matcher(sentence):
        return False
    verb = _AGENCY_VERBS_RE.search(sentence) or _SPEECH_VERBS_RE.search(sentence)
    if not verb:
        return False
    before = sentence[: verb.start()]
    after = sentence[verb.end():]
    before_tail = before[-48:] if len(before) > 48 else before
    if matcher(before) and not _OBJECT_PREP_RE.search(before_tail):
        return True
    # Inversión periodística: «…», afirmó la gobernadora / dijo Carlos Méndez
    after_head = after.lstrip(" ,;:«»\"'")
    if matcher(after_head[:70]) and re.match(
        r"^(?:la\s+|el\s+)?(?:gobernador[a]|mandataria|lucy|secretari[oa]|[A-ZÁÉÍÓÚÑ])",
        after_head,
    ):
        return True
    before_u = unidecode(before.lower())
    if re.search(r"\bsegun\s+", before_u) and matcher(sentence):
        return True
    return False


def lucy_intervenes(title: str, body: str) -> bool:
    sucre = article_anchors_sucre(title, body)

    def _lucy(text: str) -> bool:
        return mentions_lucy_person(text, sucre_context=sucre)

    for sent in split_sentences(body):
        if _person_is_agent(sent, _lucy):
            return True
    # Título como apoyo: confirma identidad, pero la intervención debe estar en el cuerpo.
    # Si el cuerpo usa «la mandataria» y el título nombra a Lucy, ya cubierto por sucre_context.
    return False


def _secretary_belongs_to_sucre(role: str, sentence: str, title: str, body: str) -> bool:
    blob = f"{role} {sentence}"
    blob_n = _norm(blob)
    if "sucre" in blob_n or "departamental" in blob_n or "gobernacion" in blob_n:
        if _OTHER_DEPT_RE.search(role) and "sucre" not in _norm(role):
            return False
        return True
    # Secretario departamental en una nota anclada a Sucre, sin otro departamento.
    if article_anchors_sucre(title, body) and not _OTHER_DEPT_RE.search(blob):
        return True
    return False


def _iter_named_secretaries(sentence: str) -> List[Tuple[str, str]]:
    found: List[Tuple[str, str]] = []
    seen = set()
    for rx in (_SECRETARY_NAME_FIRST_RE, _SECRETARY_ROLE_FIRST_RE, _SECRETARY_ROLE_THEN_NAME_RE):
        for m in rx.finditer(sentence):
            name = collapse_ws(m.group("name"))
            role = collapse_ws(m.group("role"))
            role = re.sub(r"\s+", " ", role).rstrip(" ,.;")
            if not looks_like_person_name(name):
                continue
            if _SECRETARIA_ENTITY_RE.fullmatch(name):
                continue
            key = _norm(f"{name}|{role}")
            if key in seen:
                continue
            seen.add(key)
            found.append((name, role))
    return found


def named_secretaries_intervening(title: str, body: str) -> List[Tuple[str, str]]:
    actors: List[Tuple[str, str]] = []
    seen = set()
    for sent in split_sentences(body):
        for name, role in _iter_named_secretaries(sent):
            if not _secretary_belongs_to_sucre(role, sent, title, body):
                continue

            def _this_person(text, _name=name):
                return _name.lower() in (text or "").lower()

            if not _person_is_agent(sent, _this_person) and not _person_is_agent(
                sent, lambda t, r=role: r.lower() in (t or "").lower()
            ):
                continue
            key = _norm(f"{name}|{role}")
            if key in seen:
                continue
            seen.add(key)
            actors.append((name, role))
    return actors


def _span_from_sentences(source: str, sentences: Sequence[str]) -> str:
    kept = [s for s in sentences if s and source and s in source]
    if not kept:
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
        if not name or is_filler(name) or is_entity_only_label(name):
            continue
        if not looks_like_person_name(name) and not _is_lucy_name_or_role(name, cargo):
            continue
        if not cargo:
            continue
        key = _norm(f"{name}|{cargo}")
        if key in seen:
            continue
        seen.add(key)
        parts.append(_format_person_role(name, cargo))
    return "; ".join(parts)


def _external_person_role(sentence: str) -> List[Tuple[str, str]]:
    """Personas ajenas con nombre + cargo que hablan en la oración."""
    actors: List[Tuple[str, str]] = []
    # «Andrés Julián Rendón, Gobernador de Antioquia»
    for m in re.finditer(
        rf"(?P<name>{_NAME_TOKEN}(?:\s+(?:de\s+l[oa]s?\s+|de\s+|del\s+)?{_NAME_TOKEN}){{1,4}})"
        r"\s*,\s*"
        rf"(?P<role>{_ROLE_HINT_RE.pattern}(?:\s+de\s+[^.,;]{{2,40}})?)",
        sentence,
        flags=re.IGNORECASE,
    ):
        name = collapse_ws(m.group("name"))
        role = collapse_ws(m.group("role"))
        if looks_like_person_name(name) and not _is_lucy_name_or_role(name, role):
            if re.match(r"gobernador[a]\s+de\s+sucre", role, re.I):
                continue
            actors.append((name, role))

    # «El senador Andrés Pérez» / «el alcalde Ricardo Hernández»
    for m in _ROLE_HINT_RE.finditer(sentence):
        role = collapse_ws(m.group(0))
        if re.match(r"gobernador[a]\s+de\s+sucre", sentence[m.start(): m.start() + 40], re.I):
            continue
        if mentions_lucy_person(sentence[max(0, m.start() - 12): m.end() + 8], sucre_context=True):
            continue
        after = sentence[m.end(): m.end() + 100]
        place_m = re.match(
            rf"^\s*(?:de\s+({_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,2}})\s*,?\s*)?"
            rf"({_NAME_TOKEN}(?:\s+(?:de\s+)?{_NAME_TOKEN}){{1,3}})",
            after,
        )
        if not place_m:
            continue
        place = collapse_ws(place_m.group(1) or "")
        name = collapse_ws(place_m.group(2) or "")
        if not looks_like_person_name(name):
            continue
        if _is_lucy_name_or_role(name, role):
            continue
        cargo = f"{role} de {place}" if place else role
        actors.append((name, cargo))
    return actors


def _external_speaks_about_brand(sentence: str, title: str, body: str) -> bool:
    if not mentions_brand_or_lucy(sentence, sucre_context=article_anchors_sucre(title, body)):
        return False
    return bool(_SPEECH_VERBS_RE.search(sentence) or _AGENCY_VERBS_RE.search(sentence))


def heuristic_tone(title: str, body: str) -> str:
    blob = f"{title} {body}"
    pos = bool(_POS_RE.search(blob))
    neg = bool(_NEG_RE.search(blob))
    if neg and not pos:
        return "Negativo"
    if pos and not neg:
        return "Positivo"
    if neg and pos:
        return "Negativo"
    return "Neutro"


def heuristic_analyze(titulo: str, cuerpo: str) -> Dict[str, str]:
    title, body, _combined = article_sources(titulo, cuerpo)
    result = empty_result()
    if not body and not title:
        return result

    sucre = article_anchors_sucre(title, body)

    def _lucy(text: str) -> bool:
        return mentions_lucy_person(text, sucre_context=sucre)

    own_actors: List[Tuple[str, str]] = []
    own_sents: List[str] = []
    if lucy_intervenes(title, body):
        own_actors.append((LUCY_CANONICAL_NAME, LUCY_CANONICAL_CARGO))
        own_sents.extend(s for s in split_sentences(body) if _person_is_agent(s, _lucy))

    for name, role in named_secretaries_intervening(title, body):
        own_actors.append((name, role))
        for s in split_sentences(body):
            if name.lower() in s.lower() and s not in own_sents:
                if _person_is_agent(s, lambda t, n=name: n.lower() in (t or "").lower()):
                    own_sents.append(s)

    ext_actors: List[Tuple[str, str]] = []
    ext_sents: List[str] = []
    for s in split_sentences(body):
        if not _external_speaks_about_brand(s, title, body):
            continue
        people = _external_person_role(s)
        people = [
            (n, r) for n, r in people
            if looks_like_person_name(n)
            and not is_entity_only_label(n)
            and not _is_lucy_name_or_role(n, r)
            and not re.match(r"gobernador[a]\s+de\s+sucre", r or "", re.I)
        ]
        if not people:
            continue
        ext_actors.extend(people)
        ext_sents.append(s)

    own_span = recover_verbatim(_span_from_sentences(body, own_sents), body) if own_sents else ""
    ext_span = recover_verbatim(_span_from_sentences(body, ext_sents), body) if ext_sents else ""

    result[COL_TONO] = heuristic_tone(title, body)
    result[COL_PROPIOS] = _format_actors(own_actors)
    result[COL_INT_PROPIA] = own_span if result[COL_PROPIOS] else ""
    result[COL_EXTERNOS] = _format_actors(ext_actors)
    result[COL_MENCION_EXT] = ext_span if result[COL_EXTERNOS] else ""
    return enforce_people_only(result, title, body)


def parse_actor_list(raw: str) -> List[Tuple[str, str]]:
    if is_filler(raw):
        return []
    actors = []
    for chunk in re.split(r"\s*;\s*", str(raw)):
        chunk = collapse_ws(chunk)
        if not chunk or is_filler(chunk):
            continue
        if "," in chunk:
            name, cargo = chunk.split(",", 1)
            actors.append((collapse_ws(name), collapse_ws(cargo)))
        else:
            actors.append((chunk, ""))
    return actors


def _sanitize_names(raw: str) -> str:
    if is_filler(raw):
        return ""
    text = collapse_ws(str(raw or ""))
    text = re.sub(r"\((?:sin\s+|no\s+|extracto|nota|columna)[^)]*\)", "", text, flags=re.I)
    return collapse_ws(text)


def enforce_people_only(result: Dict[str, str], titulo: str, cuerpo: str) -> Dict[str, str]:
    """HARD rules: solo personas; Lucy solo si interviene; extractos literales de CuerpoEs."""
    title, body, _ = article_sources(titulo, cuerpo)
    out = dict(result)

    lucy_ok = lucy_intervenes(title, body)
    secretaries = {(_norm(n), _norm(r)) for n, r in named_secretaries_intervening(title, body)}
    secretary_names = {_norm(n) for n, _r in named_secretaries_intervening(title, body)}

    propios_ok: List[Tuple[str, str]] = []
    for name, cargo in parse_actor_list(out.get(COL_PROPIOS, "")):
        if is_entity_only_label(name) or is_entity_only_label(f"{name}, {cargo}".strip(", ")):
            continue
        if _is_lucy_name_or_role(name, cargo):
            if lucy_ok:
                propios_ok.append((LUCY_CANONICAL_NAME, LUCY_CANONICAL_CARGO))
            continue
        if not looks_like_person_name(name) or not cargo:
            continue
        if re.match(r"secretari[oa]\b", cargo, re.I):
            if (_norm(name), _norm(cargo)) in secretaries or _norm(name) in secretary_names:
                if _secretary_belongs_to_sucre(cargo, f"{name} {cargo}", title, body):
                    propios_ok.append((name, cargo))
            continue
        # Otros cargos propios no están permitidos (solo Lucy o secretarios nombrados).
        continue

    externos_ok: List[Tuple[str, str]] = []
    for name, cargo in parse_actor_list(out.get(COL_EXTERNOS, "")):
        if is_entity_only_label(name):
            continue
        if _is_lucy_name_or_role(name, cargo):
            continue
        if not looks_like_person_name(name) or not cargo:
            continue
        if re.match(r"gobernador[a]\s+de\s+sucre", cargo, re.I):
            continue
        externos_ok.append((name, cargo))

    own_extract = recover_verbatim(out.get(COL_INT_PROPIA, ""), body)
    ext_extract = recover_verbatim(out.get(COL_MENCION_EXT, ""), body)

    out[COL_PROPIOS] = _format_actors(propios_ok)
    out[COL_EXTERNOS] = _format_actors(externos_ok)
    out[COL_INT_PROPIA] = own_extract if out[COL_PROPIOS] else ""
    out[COL_MENCION_EXT] = ext_extract if out[COL_EXTERNOS] else ""

    for col in (COL_PROPIOS, COL_INT_PROPIA, COL_EXTERNOS, COL_MENCION_EXT):
        if is_filler(out.get(col, "")):
            out[col] = ""
    return out


def merge_analysis(
    llm: Optional[Dict[str, str]],
    heuristic: Dict[str, str],
    titulo: str,
    cuerpo: str,
) -> Dict[str, str]:
    title, body, _ = article_sources(titulo, cuerpo)
    base = dict(heuristic)
    if not llm:
        return enforce_people_only(base, title, body)

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
        if not _sanitize_names(externos):
            base[COL_EXTERNOS] = ""

    int_propia_raw = llm.get(COL_INT_PROPIA) or llm.get("intervencion_propia") or ""
    int_propia = recover_verbatim(str(int_propia_raw), body)
    if int_propia:
        base[COL_INT_PROPIA] = int_propia

    menc_raw = llm.get(COL_MENCION_EXT) or llm.get("mencion_externa") or ""
    menc = recover_verbatim(str(menc_raw), body)
    if menc:
        base[COL_MENCION_EXT] = menc

    merged = enforce_people_only(base, title, body)
    heur_ok = enforce_people_only(dict(heuristic), title, body)
    if not merged[COL_PROPIOS] and heur_ok[COL_PROPIOS]:
        merged[COL_PROPIOS] = heur_ok[COL_PROPIOS]
        merged[COL_INT_PROPIA] = heur_ok[COL_INT_PROPIA]
    if not merged[COL_EXTERNOS] and heur_ok[COL_EXTERNOS]:
        merged[COL_EXTERNOS] = heur_ok[COL_EXTERNOS]
        merged[COL_MENCION_EXT] = heur_ok[COL_MENCION_EXT]
    return merged


def _call_openai_sucre(client, model: str, brand: str, aliases: List[str], ctx: str, title: str, body: str) -> Dict[str, str]:
    alias_txt = ", ".join(aliases) if aliases else "Ninguno"
    prompt = f"""Analiza esta noticia para la marca principal: "{brand}" (Gobernadora {LUCY_CANONICAL_NAME}).
Alias a reconocer: {alias_txt}.

Titular:
\"\"\"{title}\"\"\"

Cuerpo (CuerpoEs):
\"\"\"{body}\"\"\"

Contexto anclado a la marca:
\"\"\"{ctx}\"\"\"

Devuelve JSON con:
1. "tono": impacto reputacional SOBRE Lucy Inés García Montes y/o la Gobernación de Sucre: "Positivo", "Negativo" o "Neutro".
2. "nombre_cargo_propios": SOLO personas (nombre + cargo) que INTERVIENEN en el Cuerpo.
   Permitido únicamente:
   - Lucy Inés García Montes / Lucy García Montes / Lucy Montes / Lucy García / Gobernadora de Sucre, SI ella habla o actúa.
   - Un secretario o secretaria NOMBRADO/A de la Gobernación de Sucre (persona + cargo).
   Si Lucy NO interviene, NO pongas su nombre.
   PROHIBIDO como nombre/cargo (son entidades, no personas): «Secretaría de Educación departamental», «Gobernación de Sucre», «Ministerio del Interior», «Presidencia de la República» u otra entidad sola.
3. "intervencion_propia": EXTRACTO LITERAL del CuerpoEs de lo que ESA PERSONA dijo o hizo. Sin intros, sin paráfrasis, sin notas editoriales.
4. "nombre_cargo_externos": SOLO personas con nombre + cargo que hablan/opinan y se refieren a la Gobernación de Sucre o a Lucy. Ejemplo: «Andrés Julián Rendón, Gobernador de Antioquia». Nunca una entidad sola.
5. "mencion_externa": EXTRACTO LITERAL del CuerpoEs de la intervención de esa persona refiriéndose a la Gobernación/Lucy.

REGLAS:
- Extractos = copia literal del Cuerpo. Prohibido parafrasear.
- Si no hay dato, usa "" (nunca «(sin actor externo)», N/A, ninguno).
- Varios actores se separan con "; " en formato «Nombre Apellido, Cargo».

Responde estrictamente en JSON:
{{"tono": "...", "nombre_cargo_propios": "...", "intervencion_propia": "...", "nombre_cargo_externos": "...", "mencion_externa": "..."}}"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Auditor de medios. Extrae solo personas (nombre + cargo) e intervenciones "
                    "con citas literales del cuerpo; nunca entidades sueltas ni paráfrasis."
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
    km: dict,
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

    n = sum(1 for r in rows if not r.get("is_duplicate"))
    if progress_callback:
        progress_callback(90, f"Extrayendo actores e intervenciones Sucre ({n} notas)…")

    titulo_key = km.get("titulo", "Título")

    def _cuerpo_of(row: dict) -> str:
        return (
            row.get("Resumen - Aclaracion")
            or row.get("resumen corto")
            or row.get("CuerpoEs")
            or row.get("Resumen")
            or ""
        )

    def _one(idx_row):
        idx, row = idx_row
        if row.get("is_duplicate"):
            empty = empty_result()
            return idx, {k: "" for k in SUCRE_ACTOR_COLUMNS} | {COL_TONO: empty[COL_TONO]}
        result = analyze_article(
            row.get(titulo_key, ""),
            _cuerpo_of(row),
            client=client,
            model=model,
            brand=BRAND,
            aliases=aliases,
            brand_regexes=brand_regexes,
        )
        return idx, result

    completed = 0
    if not rows:
        return rows

    workers = max_workers if client else 1
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_one, (i, row)) for i, row in enumerate(rows)]
        for fut in as_completed(futures):
            idx, result = fut.result()
            for col in SUCRE_ACTOR_COLUMNS:
                rows[idx][col] = result.get(col, "")
            completed += 1
            if progress_callback and (completed % 8 == 0 or completed == len(rows)):
                pct = 90 + int((completed / len(rows)) * 3)
                progress_callback(min(93, pct), f"Actores Sucre… {completed}/{len(rows)}")

    return rows
