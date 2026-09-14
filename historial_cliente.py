# -*- coding: utf-8 -*-
"""Historial de resultados por cliente (para analizar más noticias del mismo
cliente en corridas sucesivas sin perder lo anterior).

Guarda, por cada corrida:
  - el XLSX resultante en  <raiz>/<cliente>/
  - un indice.json con metadatos (fecha, filas, taxonomia usada)

Usa el filesystem de Streamlit Cloud de forma best-effort: si el host es
efimero no persiste entre arranques, pero sirve para la ejecucion local y por
sesion. Nunca lanza: si no puede escribir, lo registra y sigue.
"""
from __future__ import annotations

import datetime
import io
import json
import logging
import os
import re
import unicodedata

logger = logging.getLogger("limpieza_grill")

_RAIZ_DEFECTO = os.path.join(os.path.expanduser("~"), "GrillAPI_historial")


def _slug(texto: str) -> str:
    s = unicodedata.normalize("NFKD", str(texto or "cliente"))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    return re.sub(r"[-\\s]+", "_", s) or "cliente"


def _raiz(extra: dict = None) -> str:
    ruta = (extra or {}).get("historial_dir") or _RAIZ_DEFECTO
    return os.path.expanduser(ruta)


def _dir_cliente(slug: str, extra: dict = None) -> str:
    return os.path.join(_raiz(extra), slug)


def guardar_resultado(brand: str, output_bytes: bytes, resultado: dict,
                      extra: dict = None):
    """Persiste el XLSX y el indice de la corrida para el cliente `brand`."""
    slug = _slug(brand)
    carpeta = _dir_cliente(slug, extra)
    try:
        os.makedirs(carpeta, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_xlsx = f"{stamp}.xlsx"
        with open(os.path.join(carpeta, nombre_xlsx), "wb") as fh:
            fh.write(output_bytes)

        indice_path = os.path.join(carpeta, "indice.json")
        entradas = []
        if os.path.exists(indice_path):
            try:
                with open(indice_path, "r", encoding="utf-8") as fh:
                    entradas = json.load(fh)
            except Exception:
                entradas = []
        if not isinstance(entradas, list):
            entradas = []
        analisis = (resultado or {}).get("analisis") or {}
        entradas.insert(0, {
            "archivo": nombre_xlsx,
            "fecha": stamp,
            "total_rows": (resultado or {}).get("total_rows"),
            "unique_rows": (resultado or {}).get("unique_rows"),
            "duplicates": (resultado or {}).get("duplicates"),
            "process_duration": (resultado or {}).get("process_duration"),
            "taxonomia": list(analisis.get("taxonomia") or []),
        })
        # conserva los ultimos 20 resultados por cliente
        entradas = entradas[:20]
        with open(indice_path, "w", encoding="utf-8") as fh:
            json.dump(entradas, fh, ensure_ascii=False, indent=1)
        return {"carpeta": carpeta, "archivo": nombre_xlsx, "n_previos": len(entradas)}
    except Exception as e:
        logger.warning("No se pudo guardar historial del cliente: %s", e)
        return None


def listar_historial(brand: str, extra: dict = None) -> list:
    slug = _slug(brand)
    indice_path = os.path.join(_dir_cliente(slug, extra), "indice.json")
    if not os.path.exists(indice_path):
        return []
    try:
        with open(indice_path, "r", encoding="utf-8") as fh:
            entradas = json.load(fh)
        return entradas if isinstance(entradas, list) else []
    except Exception:
        return []


def taxonomia_anterior(brand: str, extra: dict = None):
    """Devuelve la taxonomia (dict con 'temas' y 'reglas') de la corrida mas
    reciente del cliente, para reutilizarla y no romper el cruce entre periodos.
    Los reglas se derivan de los temas con derivar_reglas del motor."""
    entradas = listar_historial(brand, extra)
    if not entradas:
        return None
    temas = (entradas[0] or {}).get("taxonomia") or []
    if not temas:
        return None
    try:
        from analyzer_tono_tema import derivar_reglas
        return {"temas": list(temas), "reglas": derivar_reglas(temas)}
    except Exception:
        return {"temas": list(temas), "reglas": []}

