# ======================================
# Análisis de Tono, Tema y Subtema con IA
# ======================================
"""
Módulo independiente de la limpieza. Se ejecuta DESPUÉS de que pipeline.py
ya limpió, normalizó y detectó duplicados. No modifica ninguna columna
existente: solo agrega Tono_IA, Tema_IA y Subtema_IA.

Estrategia (3 pasos, pensada para no pagar de más ni fragmentar el conteo
en el informe final):

1. Agrupar noticias iguales/similares SIN usar la API (TF-IDF + coseno
   sobre Título normalizado + inicio del Resumen). Esto detecta que varios
   medios distintos están contando la misma historia aunque el título o la
   URL no coincidan exactamente.
2. Analizar con gpt-4.1-nano solo UN representante por grupo, enfocando el
   prompt en el contexto de la marca/alias dentro de "Resumen - Aclaracion"
   (no la noticia completa). El resultado (Tono, Subtema) se copia a todas
   las filas del grupo, incluidas las duplicadas exactas del pipeline.
3. Canonizar subtemas parecidos entre grupos distintos (rapidfuzz) y pedirle
   al modelo, en una sola llamada extra, que agrupe los subtemas únicos en
   Temas. Los subtemas que no calzan en ningún grupo quedan con su propio
   Tema (no se fuerza la agrupación).
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

from unidecode import unidecode

logger = logging.getLogger("limpieza_grill.ia")

ProgressCb = Optional[Callable[[int, str], None]]

TONOS_VALIDOS = {"Positivo", "Neutro", "Negativo"}

STOPWORDS_FINALES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
    "en", "por", "para", "con", "sin", "sobre", "entre", "hacia", "hasta",
    "segun", "según", "y", "o", "que", "al", "a", "su", "sus",
}
VERBO_SUFIJOS = re.compile(
    r"(ar|er|ir|ando|iendo|ado|ada|ados|adas|ido|ida|idos|idas|amos|emos|imos)$",
    re.IGNORECASE,
)


def _norm(txt: str) -> str:
    return unidecode(str(txt or "")).lower().strip()


# ---------------------------------------------------------------
# 1) Alias / contexto de marca (sin API)
# ---------------------------------------------------------------
def parse_alias_list(marca: str, alias_text: str) -> List[str]:
    """Combina marca principal + alias (separados por coma o punto y coma)."""
    partes = re.split(r"[;,]", alias_text or "")
    aliases = [(marca or "").strip()] + [p.strip() for p in partes if p.strip()]
    vistos, out = set(), []
    for a in aliases:
        k = _norm(a)
        if a and k not in vistos:
            vistos.add(k)
            out.append(a)
    return out


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def extract_brand_context(resumen: str, titulo: str, aliases: List[str], max_chars: int = 2200) -> str:
    """
    Devuelve solo las oraciones del Resumen - Aclaracion donde aparece la
    marca/alias (+1 oración de contexto antes y después). Si no hay
    coincidencia clara, devuelve el resumen completo (truncado) para no
    dejar la noticia sin analizar, tal como se pidió.
    """
    resumen = resumen or ""
    if not aliases:
        return (str(titulo) + "\n" + resumen)[:max_chars]

    patrones = [re.compile(r"\b" + re.escape(_norm(a)) + r"\b") for a in aliases if a]
    oraciones = _split_sentences(resumen)
    if not oraciones:
        return (str(titulo) + "\n" + resumen)[:max_chars]

    idx_match = set()
    for i, frase in enumerate(oraciones):
        frase_norm = _norm(frase)
        if any(p.search(frase_norm) for p in patrones):
            idx_match.add(i)

    if not idx_match:
        # Sin mención clara en el resumen: se analiza igual, resumen completo.
        return (str(titulo) + "\n" + resumen)[:max_chars]

    idx_contexto = set()
    for i in idx_match:
        idx_contexto.update({i - 1, i, i + 1})
    idx_contexto = sorted(i for i in idx_contexto if 0 <= i < len(oraciones))
    contexto = " ".join(oraciones[i] for i in idx_contexto)
    return (str(titulo) + "\n" + contexto)[:max_chars]


# ---------------------------------------------------------------
# 2) Agrupar noticias iguales/similares (sin API) — TF-IDF + coseno
# ---------------------------------------------------------------
def _comparison_text(row: dict) -> str:
    from pipeline import normalize_title_for_comparison  # import diferido: evita ciclo

    titulo = normalize_title_for_comparison(row.get("Título", ""))
    resumen = _norm(str(row.get("Resumen - Aclaracion", "")))[:400]
    return f"{titulo} {resumen}".strip()


def cluster_similar_news(rows: List[dict], threshold: float = 0.72) -> List[List[int]]:
    """
    Agrupa índices de `rows` que corresponden a la misma noticia contada por
    distintos medios. No usa la API: TF-IDF de (Título normalizado + inicio
    del Resumen) + similitud coseno, agrupación voraz contra representantes
    ya creados. Las filas marcadas por el pipeline como duplicadas exactas
    (is_duplicate=True) se excluyen aquí y heredan el resultado más adelante.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    candidatos = [i for i, r in enumerate(rows) if not r.get("is_duplicate")]
    if not candidatos:
        return []

    textos = [_comparison_text(rows[i]) for i in candidatos]
    non_empty = [t for t in textos if t.strip()]
    if len(non_empty) < 2:
        return [[i] for i in candidatos]

    vect = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9)
    matriz = vect.fit_transform(textos)

    clusters: List[List[int]] = []
    reps: List[int] = []  # posiciones (dentro de `candidatos`) representantes de cada cluster

    for pos in range(len(candidatos)):
        if not reps:
            clusters.append([pos])
            reps.append(pos)
            continue
        sims = cosine_similarity(matriz[pos], matriz[reps])[0]
        best = int(sims.argmax())
        if sims[best] >= threshold:
            clusters[best].append(pos)
        else:
            clusters.append([pos])
            reps.append(pos)

    # traducir posiciones -> índices reales de `rows`
    return [[candidatos[p] for p in grupo] for grupo in clusters]


# ---------------------------------------------------------------
# 3) Llamadas a la API (gpt-4.1-nano)
# ---------------------------------------------------------------
SYSTEM_PROMPT = """Eres un analista senior de una agencia de monitoreo de medios en Colombia.
Analizas UNA noticia enfocándote exclusivamente en cómo queda retratada la marca
indicada (o sus alias), NO el resto de la noticia si no involucra a la marca.

Devuelve SOLO un JSON con esta forma exacta, sin texto adicional:
{"tono": "Positivo|Neutro|Negativo", "subtema": "..."}

Reglas para "tono" (siempre respecto a la marca, no al tema general):
- Positivo: la marca es elogiada, beneficiada o mostrada favorablemente.
- Negativo: la marca es criticada, señalada, o aparece en un hecho perjudicial.
- Neutro: mención informativa, sin carga positiva ni negativa clara.
- Si la marca no aparece o no hay contexto claro, evalúa el tono general de la noticia.

Reglas ESTRICTAS para "subtema" (se usará para agrupar cientos de noticias, sé consistente):
- Máximo 6 palabras.
- Debe ser una frase nominal completa y coherente (no un título de noticia).
- Sin comas ni puntos.
- No debe terminar en verbo, artículo ni preposición.
- Usa siempre el mismo estilo/orden de palabras para hechos equivalentes, para
  que noticias sobre el mismo hecho reciban el mismo subtema.
- Ejemplos válidos: "Paro de transportistas en Bogotá", "Resultados financieros trimestrales",
  "Lanzamiento de nueva sede universitaria".
- Ejemplos inválidos: "Se anuncia el paro de..." (verbo/incompleto), "Con relación a..." (termina en preposición).
"""

USER_TEMPLATE = """Marca a analizar: {marca}
Alias equivalentes: {alias}

Contexto relevante de la noticia (puede ser un fragmento centrado en la marca):
\"\"\"{contexto}\"\"\"

Responde solo con el JSON pedido."""


def _validar_subtema(texto: str) -> bool:
    t = (texto or "").strip()
    if not t or "," in t or "." in t:
        return False
    palabras = t.split()
    if len(palabras) > 6:
        return False
    ultima = _norm(palabras[-1])
    if ultima in STOPWORDS_FINALES:
        return False
    if VERBO_SUFIJOS.search(ultima) and len(ultima) > 4:
        return False
    return True


def _llamar_openai(client, model: str, marca: str, alias: List[str], contexto: str, reintentos: int = 2) -> dict:
    user_msg = USER_TEMPLATE.format(marca=marca, alias=", ".join(alias), contexto=contexto[:3000])
    last_exc = None
    for intento in range(reintentos + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            data = json.loads(resp.choices[0].message.content)
            tono = str(data.get("tono", "")).strip().title()
            subtema = str(data.get("subtema", "")).strip()
            if tono not in TONOS_VALIDOS:
                tono = "Neutro"
            if not _validar_subtema(subtema):
                user_msg = (
                    user_msg
                    + f"\n\nEl subtema '{subtema}' no cumple las reglas "
                    "(máx. 6 palabras, sin comas/puntos, no termina en verbo/artículo/preposición). "
                    "Corrígelo y responde de nuevo solo con el JSON."
                )
                continue
            return {"tono": tono, "subtema": subtema}
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("Fallo llamada IA (intento %s): %s", intento, exc)
    logger.error("No se pudo analizar con IA tras reintentos: %s", last_exc)
    return {"tono": "Neutro", "subtema": "Sin clasificar"}


# ---------------------------------------------------------------
# 4) Canonizar subtemas parecidos + asignar Temas
# ---------------------------------------------------------------
def canonicalize_subtemas(subtemas: List[str], threshold: int = 88) -> Dict[str, str]:
    """
    Unifica subtemas casi-idénticos generados en llamadas distintas
    (ej. 'Bloqueos viales por paro camionero' vs 'Afectación vial por paro
    de transporte') bajo el texto más frecuente, usando similitud difusa.
    """
    from rapidfuzz import fuzz, process

    conteo = Counter(subtemas)
    canonicos: List[str] = []
    mapa: Dict[str, str] = {}
    for s in sorted(set(subtemas), key=lambda x: -conteo[x]):
        match = None
        if canonicos:
            match = process.extractOne(s, canonicos, scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
        if match:
            mapa[s] = match[0]
        else:
            canonicos.append(s)
            mapa[s] = s
    return mapa


TEMA_SYSTEM_PROMPT = """Eres un analista de medios. Recibes una lista de subtemas ya
depurados de un dossier de noticias. Agrupa los que traten sobre el mismo asunto
amplio bajo un mismo "Tema" (máximo 4 palabras, sin comas ni puntos, coherente).
Si un subtema no encaja claramente con otros, NO lo fuerces a un grupo: su Tema
puede ser igual a su propio subtema o una versión corta de él.
Devuelve SOLO un JSON: {"asignaciones": {"<subtema>": "<tema>", ...}} con una
entrada por cada subtema recibido, sin omitir ninguno."""


def assign_temas(client, model: str, subtemas_unicos: List[str]) -> Dict[str, str]:
    if not subtemas_unicos:
        return {}
    lote = subtemas_unicos[:300]  # margen de seguridad de contexto
    user_msg = "Subtemas:\n" + "\n".join(f"- {s}" for s in lote)
    asignaciones = {}
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": TEMA_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        asignaciones = data.get("asignaciones", {}) or {}
    except Exception:
        logger.exception("Fallo al asignar Temas; se usa Subtema como Tema.")
    return {s: asignaciones.get(s, s) for s in subtemas_unicos}


# ---------------------------------------------------------------
# 5) Orquestador principal
# ---------------------------------------------------------------
def enrich_rows_with_ia(
    rows: List[dict],
    marca: str,
    alias_text: str,
    api_key: str,
    model: str = "gpt-4.1-nano-2025-04-14",
    progress: ProgressCb = None,
    max_workers: int = 6,
    similarity_threshold: float = 0.72,
    stats_out: Optional[dict] = None,
) -> List[dict]:
    """
    Agrega Tono_IA, Tema_IA y Subtema_IA a cada fila de `rows` (las mismas
    filas que produce pipeline.detectar_duplicados_avanzado, antes de
    exportar a Excel). No toca ninguna otra columna.
    """
    from openai import OpenAI

    def emit(pct, msg):
        if progress:
            progress(pct, msg)

    aliases = parse_alias_list(marca, alias_text)
    for r in rows:
        r.setdefault("Tono_IA", "")
        r.setdefault("Tema_IA", "")
        r.setdefault("Subtema_IA", "")

    emit(2, "Agrupando noticias similares (sin usar la API)…")
    grupos = cluster_similar_news(rows, threshold=similarity_threshold)
    duplicadas_exactas = [i for i, r in enumerate(rows) if r.get("is_duplicate")]

    client = OpenAI(api_key=api_key)

    def analizar_grupo(grupo: List[int]) -> Tuple[List[int], dict]:
        rep_idx = max(grupo, key=lambda i: len(str(rows[i].get("Resumen - Aclaracion", ""))))
        rep = rows[rep_idx]
        contexto = extract_brand_context(rep.get("Resumen - Aclaracion", ""), rep.get("Título", ""), aliases)
        resultado = _llamar_openai(client, model, marca, aliases, contexto)
        return grupo, resultado

    total = len(grupos)
    completados = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futuros = {pool.submit(analizar_grupo, g): g for g in grupos}
        for fut in as_completed(futuros):
            grupo, resultado = fut.result()
            for i in grupo:
                rows[i]["Tono_IA"] = resultado["tono"]
                rows[i]["Subtema_IA"] = resultado["subtema"]
            completados += 1
            if completados % 3 == 0 or completados == total:
                pct = 5 + int(completados / max(1, total) * 55)  # 5-60
                emit(pct, f"Analizando con IA… {completados}/{total} grupos de noticias")

    # Duplicadas exactas del pipeline: heredan el resultado de su noticia original.
    id_a_indice = {str(r.get("ID Noticia")): i for i, r in enumerate(rows) if not r.get("is_duplicate")}
    for i in duplicadas_exactas:
        original_id = str(rows[i].get("ID duplicada", ""))
        j = id_a_indice.get(original_id)
        if j is not None:
            rows[i]["Tono_IA"] = rows[j]["Tono_IA"]
            rows[i]["Subtema_IA"] = rows[j]["Subtema_IA"]

    emit(62, "Unificando subtemas parecidos entre grupos…")
    todos_subtemas = [r["Subtema_IA"] for r in rows if r.get("Subtema_IA")]
    mapa_canon = canonicalize_subtemas(todos_subtemas)
    for r in rows:
        if r.get("Subtema_IA"):
            r["Subtema_IA"] = mapa_canon.get(r["Subtema_IA"], r["Subtema_IA"])

    emit(75, "Agrupando subtemas en Temas…")
    subtemas_unicos = sorted(set(r["Subtema_IA"] for r in rows if r.get("Subtema_IA")))
    mapa_temas = assign_temas(client, model, subtemas_unicos)
    for r in rows:
        if r.get("Subtema_IA"):
            r["Tema_IA"] = mapa_temas.get(r["Subtema_IA"], r["Subtema_IA"])

    if stats_out is not None:
        stats_out["grupos_analizados"] = total
        stats_out["subtemas_unicos"] = len(subtemas_unicos)
        stats_out["temas_unicos"] = len(set(mapa_temas.values()))
        stats_out["por_tono"] = dict(Counter(r["Tono_IA"] for r in rows if r.get("Tono_IA")))
        stats_out["por_tema"] = dict(Counter(r["Tema_IA"] for r in rows if r.get("Tema_IA")))

    emit(100, "Análisis con IA completado")
    return rows
