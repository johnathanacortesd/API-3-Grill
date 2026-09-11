# ======================================
# Pipeline Sucre — I/O xlsx (reutiliza lectura Grill, columnas propias)
# ======================================
import datetime
import io
import logging
import time
from typing import Callable, List, Optional

import pandas as pd

from pipeline import (
    clean_cuerpo,
    emit_progress,
    file_to_bytes,
    generate_output_excel,
    get_column_robust,
    load_dossier_dataframe,
    norm_key,
)
from sucre_analyzer import (
    COL_EXTERNOS,
    COL_INT_PROPIA,
    COL_MENCION_EXT,
    COL_PROPIOS,
    COL_TONO,
    DEFAULT_MODEL,
    SUCRE_OUTPUT_COLUMNS,
    enrich_sucre_rows,
    normalize_cell_text,
)

logger = logging.getLogger("sucre_pipeline")

ProgressCb = Optional[Callable[[int, str], None]]

REQUIRED_TITLE_HINTS = ("Título", "Titulo", "titulo")
REQUIRED_BODY_HINTS = ("CuerpoEs", "Cuerpo ES", "Cuerpo")


class SucreInputError(ValueError):
    """El xlsx no trae las columnas mínimas de Sucre."""


def _find_column(df: pd.DataFrame, hints: tuple) -> Optional[str]:
    for hint in hints:
        for col in df.columns:
            if norm_key(col) == norm_key(hint):
                return col
    return None


def resolve_title_body_columns(df: pd.DataFrame) -> tuple:
    title_col = _find_column(df, REQUIRED_TITLE_HINTS)
    body_col = _find_column(df, ("CuerpoEs", "Cuerpo ES"))
    if body_col is None:
        # misma familia de insumos Grill: Resumen - Aclaracion / Resumen / Cuerpo
        for hint in ("CuerpoEs", "Resumen - Aclaracion", "Resumen", "Cuerpo"):
            body_col = _find_column(df, (hint,))
            if body_col:
                break
    if not title_col or not body_col:
        raise SucreInputError(
            "El archivo debe incluir las columnas «Título» y «CuerpoEs» "
            "(se acepta también Resumen - Aclaracion / Resumen / Cuerpo como cuerpo)."
        )
    return title_col, body_col


def _original_headers(df: pd.DataFrame) -> List[str]:
    return [str(c) for c in df.columns if str(c).strip()]


def dataframe_to_records(df: pd.DataFrame) -> List[dict]:
    records = []
    for rec in df.to_dict("records"):
        row = {}
        for k, v in rec.items():
            if isinstance(v, float) and pd.isna(v):
                row[k] = ""
            elif isinstance(v, float) and v.is_integer():
                row[k] = int(v)
            else:
                row[k] = v
        records.append(row)
    return records


def process_sucre_dossier(
    file_obj,
    progress: ProgressCb = None,
    ai_config: Optional[dict] = None,
) -> dict:
    t0 = time.time()
    emit_progress(progress, 2, "Cargando archivo…")
    file_bytes = file_to_bytes(file_obj)

    df = load_dossier_dataframe(file_bytes, progress=progress)
    if df is None or df.empty:
        raise SucreInputError("El Excel no contiene filas para analizar.")

    title_col, body_col = resolve_title_body_columns(df)
    emit_progress(progress, 45, f"Columnas detectadas: «{title_col}» y «{body_col}».")

    # Cuerpo usable para análisis (HTML limpio) sin borrar la columna original
    working_body_key = "_sucre_cuerpo_trabajo"
    df[working_body_key] = get_column_robust(df, body_col).map(
        lambda v: clean_cuerpo(normalize_cell_text(v))
    )
    df["_sucre_titulo_trabajo"] = get_column_robust(df, title_col).map(
        lambda v: normalize_cell_text(v)
    )

    rows = dataframe_to_records(df)
    ai_config = ai_config or {}
    api_key = ai_config.get("api_key") if ai_config.get("enabled", True) else None
    model = ai_config.get("model") or DEFAULT_MODEL

    emit_progress(progress, 70, "Clasificando tono y extrayendo intervenciones literales…")
    rows = enrich_sucre_rows(
        rows,
        titulo_key="_sucre_titulo_trabajo",
        cuerpo_key=working_body_key,
        api_key=api_key,
        model=model,
        progress_callback=progress,
    )

    input_cols = _original_headers(df)
    cols_to_export = [c for c in input_cols if c not in SUCRE_OUTPUT_COLUMNS and not str(c).startswith("_sucre")]
    cols_to_export.extend(SUCRE_OUTPUT_COLUMNS)

    # no aplicar corregir_texto de Grill sobre extractos
    km = {"titulo": title_col} if title_col in cols_to_export else {}

    emit_progress(progress, 94, "Generando archivo Excel…")

    def export_progress(pct, msg):
        emit_progress(progress, 94 + int(pct * 0.06), msg)

    output_data = generate_output_excel(rows, km, progress=export_progress, columns_to_use=cols_to_export)

    duration = time.time() - t0
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"Sucre_Lucy_Garcia_{timestamp}.xlsx"
    emit_progress(progress, 100, "Análisis Sucre completado")

    tono_counts = {"Positivo": 0, "Neutro": 0, "Negativo": 0}
    for row in rows:
        t = row.get(COL_TONO, "Neutro")
        if t in tono_counts:
            tono_counts[t] += 1

    return {
        "output_data": output_data,
        "output_filename": filename,
        "total_rows": len(rows),
        "process_duration": f"{duration:.2f}s",
        "tono_counts": tono_counts,
        "title_col": title_col,
        "body_col": body_col,
        "columns": cols_to_export,
        "preview_rows": [
            {
                title_col: r.get(title_col, r.get("_sucre_titulo_trabajo", "")),
                COL_TONO: r.get(COL_TONO, ""),
                COL_PROPIOS: r.get(COL_PROPIOS, ""),
                COL_INT_PROPIA: r.get(COL_INT_PROPIA, ""),
                COL_EXTERNOS: r.get(COL_EXTERNOS, ""),
                COL_MENCION_EXT: r.get(COL_MENCION_EXT, ""),
            }
            for r in rows[:12]
        ],
    }


def build_sample_xlsx() -> bytes:
    """Dossier de ejemplo para pruebas manuales y de regresión."""
    df = pd.DataFrame(
        [
            {
                "ID": 1,
                "Fecha": "10/03/2026",
                "Medio": "El Meridiano",
                "Título": "Gobernadora Lucy García entregó 200 becas en Sincelejo",
                "CuerpoEs": (
                    "La gobernadora Lucy García Montes entregó 200 becas universitarias en Sincelejo. "
                    '"La educación es la prioridad de este gobierno", afirmó la mandataria. '
                    "El alcalde Ricardo Hernández felicitó a la gobernadora por la inversión social."
                ),
            },
            {
                "ID": 2,
                "Fecha": "11/03/2026",
                "Medio": "El Heraldo",
                "Título": "Senador cuestiona ejecución de vías en Sucre",
                "CuerpoEs": (
                    "El senador Andrés Pérez señaló que la Gobernación de Sucre no ha ejecutado "
                    "el presupuesto de vías rurales y pidió explicaciones a la mandataria."
                ),
            },
            {
                "ID": 3,
                "Fecha": "12/03/2026",
                "Medio": "RCN Radio",
                "Título": "Secretaría de Educación departamental expide resolución de cobertura",
                "CuerpoEs": (
                    "La Secretaría de Educación departamental emitió la Resolución 045 de 2026 "
                    "para ampliar la cobertura escolar en los municipios del Golfo de Morrosquillo."
                ),
            },
            {
                "ID": 4,
                "Fecha": "13/03/2026",
                "Medio": "Caracol",
                "Título": "Boletín climático del Ideam para la región Caribe",
                "CuerpoEs": (
                    "El Ideam publicó el pronóstico de lluvias para la región Caribe. "
                    "No se registran declaraciones de autoridades departamentales de Sucre."
                ),
            },
        ]
    )
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()
