# ======================================
# Pipeline Sucre — mismo flujo Grill + columnas de actores
# ======================================
import datetime
import gc
import io
import logging
import re
import time
from typing import Callable, List, Optional

import pandas as pd
from unidecode import unidecode

from pipeline import (
    BASE_OUTPUT_COLUMNS,
    KEY_MAP,
    _load_optional_pkl_models,
    detectar_duplicados_avanzado,
    emit_progress,
    expand_menciones,
    file_to_bytes,
    generate_output_excel,
    load_dossier_dataframe,
    normalize_dossier_dataframe,
)
from sucre_analyzer import (
    BRAND,
    DEFAULT_ALIASES,
    DEFAULT_MODEL,
    SUCRE_ACTOR_COLUMNS,
    SUCRE_OUTPUT_COLUMNS,
    enrich_sucre_rows,
)

logger = logging.getLogger("sucre_pipeline")

ProgressCb = Optional[Callable[[int, str], None]]


class SucreInputError(ValueError):
    """El xlsx no trajo filas utilizables."""


def merge_sucre_ai_config(ai_config: Optional[dict]) -> Optional[dict]:
    """Ancla marca y alias a Gobernación de Sucre / Lucy; conserva PKL y API key."""
    cfg = dict(ai_config or {})
    user_aliases = [a.strip() for a in (cfg.get("aliases") or []) if str(a).strip()]
    merged: List[str] = []
    seen = set()
    for item in DEFAULT_ALIASES + user_aliases:
        key = unidecode(item).lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item)
    cfg["brand"] = BRAND
    cfg["aliases"] = merged
    if cfg.get("enabled") or cfg.get("tone_pkl_bytes") or cfg.get("theme_pkl_bytes"):
        cfg.setdefault("model", DEFAULT_MODEL)
        return cfg
    # Sin IA ni PKL: igual anclamos marca por si el caller la lee; el Grill
    # no exporta columnas IA, y Sucre sí añade actores heurísticos.
    return cfg


def process_sucre_dossier(
    file_obj,
    region_map,
    internet_map,
    progress: ProgressCb = None,
    ai_config: Optional[dict] = None,
) -> dict:
    """Copia del flujo `process_dossier` (Grill) + columnas de actores Sucre.

    Misma lectura xlsx, normalización, menciones, duplicados, tono/tema/subtema,
    PKL, estilo Link y hojas. Después añade las 4 columnas de personas.
    """
    t0 = time.time()
    emit_progress(progress, 2, "Cargando archivo…")
    file_bytes = file_to_bytes(file_obj)

    df_normalized = load_dossier_dataframe(file_bytes, progress=progress)
    del file_bytes
    df_normalized = normalize_dossier_dataframe(
        df_normalized, region_map, internet_map, progress=progress
    )

    medios_sin_region = []
    if not df_normalized.empty and "Región" in df_normalized.columns and "Medio" in df_normalized.columns:
        medios_sin_region = sorted(set(
            df_normalized.loc[df_normalized["Región"] == "N/A", "Medio"]
            .astype(str).str.strip()
        ) - {"", "nan", "None"})

    emit_progress(progress, 55, "Expandiendo menciones…")
    rows_expanded = expand_menciones(df_normalized)
    del df_normalized
    gc.collect()

    emit_progress(progress, 62, "Detectando duplicados…")
    rows = detectar_duplicados_avanzado(rows_expanded, KEY_MAP)

    ai_config = merge_sucre_ai_config(ai_config)
    has_ai = bool(ai_config and ai_config.get("enabled"))
    tone_model, theme_model = _load_optional_pkl_models(ai_config)
    has_pkl = tone_model is not None or theme_model is not None

    if has_ai:
        from ai_analyzer import enrich_rows_with_ai

        emit_progress(progress, 70, "Iniciando análisis reputacional con IA…")
        rows = enrich_rows_with_ai(
            rows=rows,
            km=KEY_MAP,
            brand=ai_config["brand"],
            aliases=ai_config.get("aliases", []),
            api_key=ai_config["api_key"],
            model=ai_config.get("model", DEFAULT_MODEL),
            progress_callback=progress,
            tone_model=tone_model,
            theme_model=theme_model,
        )
    elif has_pkl:
        from pkl_classifier import apply_pkl_classifiers, fill_classification_context

        emit_progress(progress, 70, "Preparando textos para clasificadores PKL…")
        rows = fill_classification_context(
            rows,
            KEY_MAP,
            brand=(ai_config or {}).get("brand", BRAND),
            aliases=(ai_config or {}).get("aliases", []),
        )
        emit_progress(progress, 88, "Aplicando modelos PKL del cliente (tono/tema)…")
        rows = apply_pkl_classifiers(
            rows,
            KEY_MAP,
            tone_model=tone_model,
            theme_model=theme_model,
            progress_callback=progress,
            brand=(ai_config or {}).get("brand", BRAND),
            aliases=(ai_config or {}).get("aliases", []),
        )

    api_key = None
    if has_ai:
        api_key = (ai_config or {}).get("api_key")
    emit_progress(progress, 90, "Extrayendo intervenciones de actores (Sucre)…")
    rows = enrich_sucre_rows(
        rows,
        KEY_MAP,
        api_key=api_key,
        model=(ai_config or {}).get("model") or DEFAULT_MODEL,
        progress_callback=progress,
    )

    if has_ai or has_pkl:
        rev_idx = BASE_OUTPUT_COLUMNS.index("revalorización")
        ai_cols = ["Contexto analizado", "Tono_IA", "Tema_IA", "Subtema_IA"]
        cols_to_export = BASE_OUTPUT_COLUMNS[:rev_idx + 1] + ai_cols + BASE_OUTPUT_COLUMNS[rev_idx + 1:]
    else:
        cols_to_export = list(BASE_OUTPUT_COLUMNS)

    for col in SUCRE_ACTOR_COLUMNS:
        if col not in cols_to_export:
            cols_to_export.append(col)

    emit_progress(progress, 94, "✓ Estructuración finalizada. Generando archivo Excel…")

    unique_rows = sum(1 for r in rows if not r.get("is_duplicate"))
    total_rows = len(rows)

    def export_progress(pct, msg):
        overall = 94 + int(pct * 0.06)
        emit_progress(progress, overall, msg)

    output_data = generate_output_excel(rows, KEY_MAP, progress=export_progress, columns_to_use=cols_to_export)
    del rows, rows_expanded
    gc.collect()
    duration = time.time() - t0
    emit_progress(progress, 100, "Limpieza y análisis completados")

    brand_raw = BRAND
    clean_tag = re.sub(r"[^\w\s-]", "", unidecode(brand_raw)).strip()
    brand_tag = re.sub(r"[-\s]+", "_", clean_tag)
    filename_prefix = f"Dossier_{brand_tag}" if brand_tag else "Dossier_Limpio"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    final_filename = f"{filename_prefix}_{timestamp}.xlsx"

    return {
        "output_data": output_data,
        "output_filename": final_filename,
        "total_rows": total_rows,
        "unique_rows": unique_rows,
        "duplicates": total_rows - unique_rows,
        "process_duration": f"{duration:.2f}s",
        "medios_sin_mapear": medios_sin_region,
    }


def build_sample_xlsx() -> bytes:
    """Dossier de ejemplo (formato Grill) para pruebas manuales y de regresión."""
    df = pd.DataFrame(
        [
            {
                "NoticiaId": 1,
                "Fecha": "10/03/2026",
                "Hora": "08:00:00",
                "Medio": "El Meridiano",
                "Tipo de Medio": "internet",
                "Título": "Gobernadora Lucy García entregó 200 becas en Sincelejo",
                "CuerpoEs": (
                    "La gobernadora Lucy García Montes entregó 200 becas universitarias en Sincelejo. "
                    '"La educación es la prioridad de este gobierno", afirmó la mandataria. '
                    "El alcalde Ricardo Hernández felicitó a la gobernadora por la inversión social."
                ),
                "URL Nota": "https://example.com/becas",
                "Empresa rel.": "Gobernación de Sucre",
            },
            {
                "NoticiaId": 2,
                "Fecha": "11/03/2026",
                "Hora": "09:00:00",
                "Medio": "El Heraldo",
                "Tipo de Medio": "internet",
                "Título": "Senador cuestiona ejecución de vías en Sucre",
                "CuerpoEs": (
                    "El senador Andrés Pérez señaló que la Gobernación de Sucre no ha ejecutado "
                    "el presupuesto de vías rurales y pidió explicaciones a la mandataria."
                ),
                "URL Nota": "https://example.com/vias",
                "Empresa rel.": "Gobernación de Sucre",
            },
            {
                "NoticiaId": 3,
                "Fecha": "12/03/2026",
                "Hora": "10:00:00",
                "Medio": "RCN Radio",
                "Tipo de Medio": "radio",
                "Título": "Secretaría de Educación departamental expide resolución de cobertura",
                "CuerpoEs": (
                    "La Secretaría de Educación departamental emitió la Resolución 045 de 2026 "
                    "para ampliar la cobertura escolar en los municipios del Golfo de Morrosquillo."
                ),
                "URL Nota AV": "https://example.com/resolucion",
                "Menciones - Empresa": "Gobernación de Sucre",
            },
            {
                "NoticiaId": 4,
                "Fecha": "13/03/2026",
                "Hora": "11:00:00",
                "Medio": "Caracol",
                "Tipo de Medio": "internet",
                "Título": "Boletín climático del Ideam para la región Caribe",
                "CuerpoEs": (
                    "El Ideam publicó el pronóstico de lluvias para la región Caribe. "
                    "No se registran declaraciones de autoridades departamentales de Sucre."
                ),
                "URL Nota": "https://example.com/clima",
                "Empresa rel.": "Ideam",
            },
            {
                "NoticiaId": 5,
                "Fecha": "14/03/2026",
                "Hora": "12:00:00",
                "Medio": "El Universal",
                "Tipo de Medio": "internet",
                "Título": "Secretario de Educación de Sucre anuncia ampliación de cupos",
                "CuerpoEs": (
                    "El secretario de Educación de Sucre, Carlos Méndez, anunció la ampliación "
                    "de cupos escolares en Sincelejo y reiteró el compromiso de la Gobernación."
                ),
                "URL Nota": "https://example.com/cupos",
                "Empresa rel.": "Gobernación de Sucre",
            },
            {
                "NoticiaId": 6,
                "Fecha": "15/03/2026",
                "Hora": "13:00:00",
                "Medio": "La República",
                "Tipo de Medio": "internet",
                "Título": "Gobernación de Sucre emite comunicado presupuestal",
                "CuerpoEs": (
                    "La Gobernación de Sucre emitió un comunicado sobre el presupuesto de 2026 "
                    "sin declaraciones de la mandataria ni de secretarios."
                ),
                "URL Nota": "https://example.com/presupuesto",
                "Empresa rel.": "Gobernación de Sucre",
            },
            {
                "NoticiaId": 7,
                "Fecha": "16/03/2026",
                "Hora": "14:00:00",
                "Medio": "El Colombiano",
                "Tipo de Medio": "internet",
                "Título": "Gobernador de Antioquia se refiere a la gestión en Sucre",
                "CuerpoEs": (
                    "Andrés Julián Rendón, Gobernador de Antioquia, afirmó que la Gobernación de Sucre "
                    "puede replicar el modelo de vías terciarias del occidente antioqueño."
                ),
                "URL Nota": "https://example.com/rendon",
                "Empresa rel.": "Gobernación de Sucre",
            },
        ]
    )
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()
