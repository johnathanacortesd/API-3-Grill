# -*- coding: utf-8 -*-
"""Auditoria de uso de la app por correo (SMTP).

Tras cada corrida exitosa se envia un correo a USAGE_NOTIFY_EMAIL con el
resumen: cliente (marca) analizado, filtro de alias/voceros, totales de filas,
grupos, distribucion de tono/tema/subtema, cubos usados, y avisos del motor.
Nunca lanza: si el correo falla o no hay SMTP configurado, lo registra y sigue.
"""
from __future__ import annotations

import os
import smtplib
import html
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Optional


def _cfg() -> Dict[str, str]:
    """Leo la config de st.secrets cuando hay Streamlit; si no, de env vars."""
    c = {}
    try:
        import streamlit as st
        src = st.secrets
        for k in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD",
                  "SMTP_FROM", "USAGE_NOTIFY_EMAIL"):
            try:
                if st.secrets.get(k):
                    c[k] = str(st.secrets.get(k))
            except Exception:
                pass
    except Exception:
        src = os.environ
    for k in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD",
              "SMTP_FROM", "USAGE_NOTIFY_EMAIL"):
        if k not in c:
            v = os.environ.get(k)
            if v:
                c[k] = v
    return c


def _cubos(analisis) -> str:
    t = (analisis or {}).get("taxonomia") or []
    return " · ".join(str(x) for x in t) if t else "(automatica no generada)"


def _armar_html(result, ai_config, analisis) -> str:
    brand = html.escape(str((ai_config or {}).get("brand") or "?"))
    alias = (ai_config or {}).get("aliases") or []
    voceros = (ai_config or {}).get("voceros") or []
    modelo = html.escape(str((ai_config or {}).get("model") or "-"))

    filas = (result or {}).get("_filas") or {}
    total = (result or {}).get("total_rows") or 0
    uniq = (result or {}).get("unique_rows") or 0
    dups = (result or {}).get("duplicates") or 0
    dur = (result or {}).get("process_duration") or "-"
    grupos = (analisis or {}).get("grupos") or 0
    tono_corregido = (analisis or {}).get("tono_corregido_por_guarda") or []
    cubos_nuevos = (analisis or {}).get("cubos_nuevos") or []
    modalidad = (analisis or {}).get("modo_taxonomia") or "-"

    filas_html = "".join(
        "<tr><td>%s</td><td><b>%s</b></td></tr>" % (html.escape(k), v)
        for k, v in sorted(filas.items()))
    if not filas_html:
        filas_html = "<tr><td>—</td><td>0</td></tr>"

    lista_alias = html.escape("; ".join(alias)) or "(sin alias)"
    lista_voceros = html.escape("; ".join(voceros)) or "(sin voceros)"
    lista_cubos = html.escape(", ".join(str(x) for x in cubos_nuevos)) or "(ninguno)"

    return (
        "<html><body style='font-family:sans-serif;font-size:14px'>"
        "<h2>Auditoria de uso · Grill-API</h2>"
        "<p>Se ejecuto un analisis de <b>%s</b> el <b>%s</b>.</p>"
        "<table border='0' cellpadding='6' style='border-collapse:collapse'>"
        "<tr><td>Cliente / marca</td><td><b>%s</b></td></tr>"
        "<tr><td>Modelo</td><td>%s</td></tr>"
        "<tr><td>Alias</td><td>%s</td></tr>"
        "<tr><td>Voceros</td><td>%s</td></tr>"
        "<tr><td>Filas totales</td><td>%s</td></tr>"
        "<tr><td>Filas unicas</td><td>%s</td></tr>"
        "<tr><td>Duplicadas</td><td>%s</td></tr>"
        "<tr><td>Tiempo</td><td>%s</td></tr>"
        "<tr><td>Grupos de hecho</td><td>%s</td></tr>"
        "<tr><td>Taxonomia</td><td>%s</td></tr>"
        "</table>"
        "<h3>Distribucion de tono (filas unicas)</h3><table border='0' cellpadding='6'>"
        "%s</table>"
        "<h3>Lista de Temas usada</h3><p>%s</p>"
        "<h3>Cubos nuevos propuestos</h3><p>%s</p>"
        "<h3>Guarda del tono</h3><p>%d correcciones (Negativos sin señalamiento "
        "pasaron a Neutro)</p>"
        "</body></html>") % (
            brand,
            __import__("datetime").datetime.now().strftime("%d/%m/%Y %H:%M"),
            brand, modelo, lista_alias, lista_voceros,
            total, uniq, dups, html.escape(str(dur)),
            grupos, html.escape(str(modalidad)),
            filas_html, html.escape(_cubos(analisis)),
            lista_cubos, len(tono_corregido))


def enviar_auditoria(result, ai_config, analisis) -> str:
    """Envia el correo. Devuelve 'enviado', 'desactivado' o 'error: <detalle>'."""
    c = _cfg()
    host = c.get("SMTP_HOST")
    to = c.get("USAGE_NOTIFY_EMAIL")
    user = c.get("SMTP_USER")
    pw = c.get("SMTP_PASSWORD")
    if not host or not to:
        return "desactivado"
    try:
        port = int(c.get("SMTP_PORT") or 587)
    except Exception:
        port = 587
    desde = c.get("SMTP_FROM") or user or "grill-api@local"
    if not user or not pw:
        return "desactivado: falta SMTP_USER/SMTP_PASSWORD"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "[Grill-API] Auditoria de uso · %s" % (
        (ai_config or {}).get("brand") or "cliente")
    msg["From"] = desde
    msg["To"] = to
    msg.attach(MIMEText(_armar_html(result, ai_config, analisis), "html", "utf-8"))
    try:
        srv = smtplib.SMTP(host, port, timeout=20)
        try:
            srv.ehlo()
            if srv.has_extn("starttls"):
                srv.starttls()
                srv.ehlo()
            srv.login(user, pw)
            srv.sendmail(desde, [to], msg.as_string())
        finally:
            try:
                srv.quit()
            except Exception:
                pass
        return "enviado"
    except Exception as e:
        return "error: %s" % str(e)[:200]


def enviar_auditoria_desde_resultado(result, ai_config) -> str:
    """Wrapper: usa el bloque 'analisis' ya dentro de `result` si existe."""
    analisis = (result or {}).get("analisis") or {}
    estado = enviar_auditoria(result, ai_config, analisis)
    try:
        import logging
        logging.getLogger("limpieza_grill").info("Auditoria por correo: %s", estado)
    except Exception:
        pass
    return estado
