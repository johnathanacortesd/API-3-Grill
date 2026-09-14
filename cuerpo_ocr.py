# -*- coding: utf-8 -*-
"""Recuperacion del cuerpo completo de una nota cuando la celda CuerpoEs quedo
truncada (tope de celda de Excel 32767) o incompleta.

Flujo:
  1. detecta si el texto recibido parece incompleto (len >= 32000, corta sin
     puntuacion final, o vacio con URL disponible);
  2. baja la URL (requests, UA de navegador);
  3. si es HTML -> extrae el texto visible (articulo principal);
  4. si es imagen (jpg/png/webp/gif) -> OCR con un modelo OpenAI de vision;
     si es PDF -> OCR por pagina con el mismo modelo de vision;
  5. devuelve el texto recuperado, mas largo que el truncado.

El OCR usa un modelo de VISION aparte (gpt-4.1-nano NO ve imagenes); por defecto
gpt-4o-mini, configurable desde la app. Si no hay red o falla, se conserva el
texto truncado (no se inventa nada).
"""
from __future__ import annotations

import base64
import io
import re
from html.parser import HTMLParser
from typing import Optional

import requests

MODELO_OCR_DEFECTO = "gpt-4o-mini"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
EXT_IMG = (".jpg", ".jpeg", ".png", ".webp", ".gif")
EXT_PDF = ".pdf"


def _parece_incompleto(texto: str) -> bool:
    """Heuristica de truncamiento: tope de celda, o corte sin puntuacion final.""" 
    t = (texto or "").strip()
    if not t:
        return False
    if len(t) >= 32000:
        return True
    # corta a mitad de frase (no termina en . ! ? … : ; o comillas de cierre)
    if len(t) > 400 and not re.search(r'[.!?\u2026:;»"\u201d]\s*$', t):
        return True
    return False


class _TextoParser(HTMLParser):
    """Extrae el texto visible de un HTML: ignora script/style, convierte
    etiquetas de bloque en nuevas lineas y decodifica entidades."""

    BLOQUE = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "article",
              "section", "figcaption", "blockquote", "tr", "td"}
    OMITE = {"script", "style", "noscript", "head", "title", "nav", "footer",
             "header", "aside", "iframe", "svg", "form", "button"}

    def __init__(self):
        super().__init__()
        self.piezas = []
        self.omitir = 0
        self.ultimo_cr = False

    def handle_starttag(self, tag, attrs):
        tl = tag.lower()
        if tl in self.OMITE:
            self.omitir += 1
        elif tl in self.BLOQUE:
            self._nl()

    def handle_endtag(self, tag):
        tl = tag.lower()
        if tl in self.OMITE and self.omitir:
            self.omitir -= 1
        elif tl in self.BLOQUE:
            self._nl()

    def handle_data(self, data):
        if self.omitir:
            return
        if data and data.strip():
            self.piezas.append(re.sub(r"\s+", " ", data))

    def _nl(self):
        # un salto por cada etiqueta de bloque, sin duplicar
        if self.piezas and not self.piezas[-1].endswith("\n"):
            self.piezas[-1] += "\n"
            self.ultimo_cr = True

    def texto(self) -> str:
        raw = " ".join(self.piezas)
        # quitar saltos solos y comprimir espacios
        lineas = [l.strip() for l in raw.split("\n")]
        lineas = [l for l in lineas if l]
        return "\n".join(lineas)


def _extraer_texto_html(bytes_html: bytes) -> str:
    try:
        texto = bytes_html.decode("utf-8", errors="replace")
    except Exception:
        texto = bytes_html.decode("latin-1", errors="replace")
    parser = _TextoParser()
    try:
        parser.feed(texto)
    except Exception:
        pass
    articulo = parser.texto()
    if articulo:
        return articulo[:12000]
    return ""


def _url_id(url: str) -> str:
    # extension en minuscula; parchea urls con query
    u = (url or "").split("?")[0].split("#")[0].lower().rstrip("/")
    for ext in EXT_IMG:
        if u.endswith(ext):
            return ext
    if u.endswith(EXT_PDF):
        return EXT_PDF
    return ""


def _bajar(url: str, timeout: int = 20) -> Optional[bytes]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout,
                         allow_redirects=True)
        if r.status_code != 200 or not r.content:
            return None
        ctype = (r.headers.get("Content-Type") or "").lower()
        # si el servidor dice que es imagen/pdf lo respetamos aunque la url no
        # termine en extension conocida
        _map = {"image/": EXT_IMG[0], "application/pdf": EXT_PDF}
        if not _url_id(url) and ctype:
            for prefijo, ext in _map.items():
                if prefijo in ctype:
                    return r.content
        return r.content
    except Exception:
        return None


def _leer_imagen(id_, bytes_, api_key, base_url, modelo) -> str:
    """OCR de una imagen o PDF con un modelo OpenAI de vision."""
    try:
        from openai import OpenAI
    except Exception:
        return ""
    b64 = base64.b64encode(bytes_).decode("ascii")
    mime = "application/pdf" if id_ == EXT_PDF else "image/jpeg"
    content = [
        {"type": "text", "text": (
            "Transcribe TODO el texto de esta {} en español, respetando "
            "párrafos y puntuación. Devuelve solo el texto, sin resumir, "
            "sin comentarios ni encabezados.".format(
                "PDF (todas las páginas)" if id_ == EXT_PDF else "imagen"))},
        {"type": "image_url", "image_url": {
            "url": "data:%s;base64,%s" % (mime, b64),
            "detail": "high"}},
    ]
    try:
        client = OpenAI(api_key=api_key, base_url=base_url or None)
        resp = client.chat.completions.create(
            model=modelo, messages=[{"role": "user", "content": content}],
            temperature=0.0, max_tokens=2000 * (6 if id_ == EXT_PDF else 1))
        return re.sub(r"\s+", " ", resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def recuperar_cuerpo(url: str, api_key: str, base_url: str = "",
                     modelo: str = MODELO_OCR_DEFECTO,
                     timeout: int = 20) -> str:
    """Devuelve el cuerpo completo recuperado de `url`, o '' si no se pudo."""
    if not url or not str(url).strip().lower().startswith("http"):
        return ""
    datos = _bajar(url, timeout)
    if not datos:
        return ""
    id_ = _url_id(url)
    if id_ in EXT_IMG or id_ == EXT_PDF:
        return _leer_imagen(id_, datos, api_key, base_url, modelo)
    return _extraer_texto_html(datos)


def fusionar_cuerpo(previo: str, recuperado: str) -> str:
    """Junta el cuerpo truncado con el recuperado sin duplicar el inicio."""
    previo = (previo or "").strip()
    rec = (recuperado or "").strip()
    if not rec:
        return previo
    if len(rec) <= len(previo):
        return previo
    # si el recuperado ya contiene casi todo el previo, usar el recuperado
    if rec[: len(previo)] == previo[: len(previo)] or previo in rec[:2000]:
        return rec
    return "%s %s" % (previo, rec)
