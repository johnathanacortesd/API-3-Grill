# ======================================
# App Sucre — Gobernación de Sucre / Lucy Inés García Montes
# Entrada aislada: streamlit run app_sucre.py
# ======================================
import html
import io
import logging
import time

import pandas as pd
import streamlit as st

from sucre_analyzer import (
    COL_EXTERNOS,
    COL_INT_PROPIA,
    COL_MENCION_EXT,
    COL_PROPIOS,
    COL_TONO,
    SUCRE_OUTPUT_COLUMNS,
)
from sucre_pipeline import SucreInputError, build_sample_xlsx, process_sucre_dossier

logger = logging.getLogger("sucre_app")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

THEME_LIGHT_VARS = """
:root,[data-testid="stApp"]{
    --bg:#f4f7f5;--s1:#ffffff;--s2:#eef4f0;--s3:#e1ebe5;
    --border:#c9d6ce;--border2:#9aafa4;--border-focus:#188a4e;
    --text:#1c2420;--text2:#3d4a44;--text3:#5c6b64;--text4:#8a9891;
    --accent:#188a4e;--accent2:#147a45;--accent3:#0f5c34;
    --accent-bg:#ecf8f1;--accent-bg2:#d8f0e2;--accent-bdr:#a7d4b8;
    --green:#059669;--green2:#047857;--green-bg:#ecfdf5;--green-bdr:#a7f3d0;
    --red:#dc2626;--amber:#d97706;--blue:#1a73e8;
    --success-bg:linear-gradient(135deg,#ecfdf5,#d1fae5);
    --success-title:#047857;
    --r:8px;--r2:12px;--r3:16px;
    --shadow-sm:0 1px 2px rgba(28,36,32,0.08),0 1px 3px rgba(28,36,32,0.06);
    --shadow-md:0 1px 3px rgba(28,36,32,0.1),0 4px 8px rgba(28,36,32,0.07);
    --shadow-lg:0 2px 6px rgba(28,36,32,0.08),0 8px 24px rgba(28,36,32,0.08);
    --transition:all 0.2s cubic-bezier(0.4,0,0.2,1);
}
"""

THEME_DARK_VARS = """
:root,[data-testid="stApp"]{
    --bg:#121816;--s1:#1b221f;--s2:#242c28;--s3:#2e3833;
    --border:#3d4a44;--border2:#5c6b64;--border-focus:#34d399;
    --text:#e8eeeb;--text2:#c5ceca;--text3:#9aa8a1;--text4:#6e7a74;
    --accent:#34d399;--accent2:#6ee7b7;--accent3:#a7f3d0;
    --accent-bg:#0f291e;--accent-bg2:#134e3a;--accent-bdr:#065f46;
    --green:#34d399;--green2:#6ee7b7;--green-bg:#0f291e;--green-bdr:#065f46;
    --red:#f87171;--amber:#fbbf24;--blue:#60a5fa;
    --success-bg:linear-gradient(135deg,#0f291e,#134e3a);
    --success-title:#6ee7b7;
    --r:8px;--r2:12px;--r3:16px;
    --shadow-sm:0 1px 2px rgba(0,0,0,0.4);
    --shadow-md:0 1px 3px rgba(0,0,0,0.45),0 4px 8px rgba(0,0,0,0.3);
    --shadow-lg:0 2px 6px rgba(0,0,0,0.4),0 8px 24px rgba(0,0,0,0.35);
    --transition:all 0.2s cubic-bezier(0.4,0,0.2,1);
}
"""


def _default_theme() -> str:
    try:
        theme_obj = getattr(getattr(st, "context", None), "theme", None)
        theme_type = getattr(theme_obj, "type", None)
        if theme_type in ("dark", "light"):
            return theme_type
    except Exception:
        pass
    return "light"


def current_ui_theme() -> str:
    theme = st.session_state.get("ui_theme")
    if theme in ("dark", "light"):
        return theme
    return _default_theme()


def load_custom_css():
    theme_vars = THEME_DARK_VARS if current_ui_theme() == "dark" else THEME_LIGHT_VARS
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Google+Sans+Text:wght@400;500;700&family=Roboto+Mono:wght@400;500&display=swap');
""" + theme_vars + """
html,body,[data-testid="stApp"]{
    background:var(--bg)!important;color:var(--text)!important;
    font-family:'Google Sans Text',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
    font-size:14px;-webkit-font-smoothing:antialiased;
}
#MainMenu,footer,header{visibility:hidden}.stDeployButton{display:none}
.block-container{padding-top:1rem!important;padding-bottom:0!important}
.app-header{background:var(--s1);border:1px solid var(--border);border-radius:var(--r3);padding:1rem 1.5rem;margin-bottom:1rem;display:flex;align-items:center;gap:1rem;box-shadow:var(--shadow-sm);position:relative;overflow:hidden;}
.app-header::after{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#147a45,#188a4e,#c41e3a);}
.app-header-icon{width:40px;height:40px;background:linear-gradient(135deg,#188a4e,#0f5c34);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.2rem;color:white;flex-shrink:0;}
.app-header-title{font-family:'Google Sans',sans-serif;font-size:1.25rem;font-weight:700;color:var(--text);line-height:1.3}
.app-header-version{font-family:'Roboto Mono',monospace;font-size:0.65rem;color:var(--text3);margin-top:0.15rem}
.app-header-badge{background:var(--accent-bg);border:1px solid var(--accent-bdr);color:var(--accent2);font-family:'Roboto Mono',monospace;font-size:0.6rem;font-weight:500;padding:0.25rem 0.75rem;border-radius:100px;text-transform:uppercase;}
.metrics-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:0.6rem;margin:0.8rem 0}
.metric-card{background:var(--s1);border:1px solid var(--border);border-radius:var(--r2);padding:0.8rem 0.6rem;text-align:center;box-shadow:var(--shadow-sm)}
.metric-val{font-family:'Google Sans',sans-serif;font-size:1.5rem;font-weight:700;line-height:1;margin-bottom:0.3rem}
.metric-lbl{font-family:'Roboto Mono',monospace;font-size:0.62rem;color:var(--text3);text-transform:uppercase;letter-spacing:0.08em}
[data-testid="stForm"]{background:var(--s1)!important;border:1px solid var(--border)!important;border-radius:var(--r3)!important;padding:1.2rem 1.5rem!important;box-shadow:var(--shadow-md)!important;}
.sec-label{font-family:'Google Sans',sans-serif;font-size:0.72rem;font-weight:700;color:var(--text2);letter-spacing:0.08em;text-transform:uppercase;padding-bottom:0.3rem;border-bottom:2px solid var(--s3);margin:0.8rem 0 0.5rem;display:flex;align-items:center;gap:0.5rem;}
.sec-label::before{content:'';display:inline-block;width:3px;height:12px;background:linear-gradient(180deg,#188a4e,#0f5c34);border-radius:2px}
[data-testid="stFileUploader"]{border:1.5px dashed var(--border)!important;border-radius:var(--r)!important;}
[data-testid="stFileUploader"]:hover{border-color:var(--accent)!important;background:var(--accent-bg)!important;}
.stButton>button,[data-testid="stDownloadButton"]>button{border-radius:100px!important;font-family:'Google Sans',sans-serif!important;}
.stButton>button[kind="primary"],[data-testid="stDownloadButton"]>button[kind="primary"],
[data-testid="stFormSubmitButton"] button,[data-testid="baseButton-primary"]{
    background:var(--accent)!important;border:none!important;color:#fff!important;
}
.success-banner{background:var(--success-bg);border:1px solid var(--green-bdr);border-left:4px solid var(--green);border-radius:var(--r2);padding:0.8rem 1.2rem;margin:0.5rem 0 0.8rem;display:flex;align-items:center;gap:0.8rem;}
.success-icon{width:34px;height:34px;background:linear-gradient(135deg,#059669,#047857);border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;flex-shrink:0;}
.success-title{font-family:'Google Sans',sans-serif;font-size:1rem;font-weight:700;color:var(--success-title)}
.success-sub{font-size:0.8rem;color:var(--text2)}
.auth-wrap{max-width:380px;margin:8vh auto 0;text-align:center}
.auth-icon{width:60px;height:60px;background:linear-gradient(135deg,#188a4e,#0f5c34);border-radius:16px;display:inline-flex;align-items:center;justify-content:center;font-size:1.6rem;color:white;margin-bottom:1rem;}
.auth-title{font-family:'Google Sans',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.3rem}
.auth-sub{font-size:0.85rem;color:var(--text3);margin-bottom:2rem}
[data-testid="stProgressBar"]>div>div{background:linear-gradient(90deg,#147a45,#34d399)!important;border-radius:100px!important;height:5px!important;}
.footer{font-family:'Roboto Mono',monospace;font-size:0.6rem;color:var(--text4);text-align:center;padding:0.8rem 0 0.5rem;border-top:1px solid var(--s3);margin-top:1rem;}
.live-panel{background:var(--s1);border:1px solid var(--border);border-radius:var(--r3);padding:1rem 1.2rem;margin:0.4rem 0 0.8rem;box-shadow:var(--shadow-md);position:relative;overflow:hidden;}
.live-panel::after{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#147a45,#c41e3a);}
.live-title{font-family:'Google Sans',sans-serif;font-size:1.02rem;font-weight:700}
.live-sub{font-size:0.78rem;color:var(--text3);margin-top:0.15rem}
.rules-box{background:var(--accent-bg);border:1px solid var(--accent-bdr);border-radius:var(--r2);padding:0.75rem 1rem;font-size:0.8rem;color:var(--text2);line-height:1.45;margin:0.4rem 0 0.8rem}
@media(max-width:768px){.metrics-grid{grid-template-columns:repeat(2,1fr)}}
</style>
""", unsafe_allow_html=True)


def _on_theme_toggle():
    st.session_state["ui_theme"] = "dark" if st.session_state.get("theme_toggle") else "light"


def render_theme_toggle():
    if "ui_theme" not in st.session_state:
        st.session_state["ui_theme"] = _default_theme()
    if "theme_toggle" not in st.session_state:
        st.session_state["theme_toggle"] = st.session_state["ui_theme"] == "dark"
    _, col_theme = st.columns([6, 1])
    with col_theme:
        st.toggle("Modo oscuro", key="theme_toggle", on_change=_on_theme_toggle)


def check_password():
    if st.session_state.get("password_correct", False):
        return True
    st.markdown("""
    <div class="auth-wrap">
        <div class="auth-icon">◈</div>
        <div class="auth-title">Gobernación de Sucre</div>
        <div class="auth-sub">Análisis de noticias · Lucy Inés García Montes</div>
    </div>""", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        with st.form("pw"):
            pw = st.text_input("Contraseña", type="password", placeholder="Ingresa tu contraseña")
            if st.form_submit_button("Ingresar", use_container_width=True, type="primary"):
                if pw == st.secrets.get("APP_PASSWORD", "INVALID"):
                    st.session_state["password_correct"] = True
                    st.rerun()
                else:
                    st.error("Contraseña incorrecta")
    return False


PIPELINE_STEPS = [
    ("read", "Leer el Excel"),
    ("norm", "Detectar Título y CuerpoEs"),
    ("ai", "Tono e intervenciones literales"),
    ("export", "Generar archivo de resultado"),
]


def _fmt_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} s"
    return f"{seconds // 60} min {seconds % 60:02d} s"


def _active_step(pct: int, msg: str) -> str:
    if pct >= 100 or "completad" in msg.lower():
        return "done"
    if pct >= 94 or "Generando" in msg:
        return "export"
    if pct >= 70 or "Analizando" in msg or "tono" in msg.lower():
        return "ai"
    if pct >= 40 or "Columnas" in msg:
        return "norm"
    return "read"


def _render_live_html(pct, msg, elapsed, file_label, active_key):
    steps_html = []
    reached_active = False
    for key, label in PIPELINE_STEPS:
        if active_key == "done":
            cls, mark = "is-done", "✓"
        elif key == active_key:
            cls, mark = "is-active", "●"
            reached_active = True
        elif not reached_active:
            cls, mark = "is-done", "✓"
        else:
            cls, mark = "", ""
        steps_html.append(f'<div class="step-item {cls}"><span class="dot">{mark}</span>{html.escape(label)}</div>')
    file_line = f" · {html.escape(file_label)}" if file_label else ""
    title = "Análisis completado" if active_key == "done" else "Procesando noticias de Sucre"
    return f"""
    <div class="live-panel">
      <div class="live-title">{title}</div>
      <div class="live-sub">El proceso sigue activo{file_line}. No cierres esta pestaña.</div>
      <p style="margin:0.6rem 0 0.2rem;font-family:Roboto Mono,monospace;font-size:0.75rem">{int(pct)}% · {html.escape(elapsed)}</p>
      {''.join(steps_html)}
      <div class="live-sub" style="margin-top:0.5rem">{html.escape(str(msg or ""))}</div>
    </div>
    """


def run_sucre_process(df_file, file_meta=None, ai_config=None):
    file_meta = file_meta or {}
    file_label = file_meta.get("name", "")
    t_start = time.time()
    panel = st.empty()
    progress_bar = st.progress(0, text="Iniciando…")

    def paint(pct, msg):
        elapsed = _fmt_elapsed(time.time() - t_start)
        active = _active_step(pct, msg)
        panel.markdown(_render_live_html(pct, msg, elapsed, file_label, active), unsafe_allow_html=True)
        progress_bar.progress(min(100, max(0, int(pct))), text=msg)

    paint(1, "Cargando archivo…")
    with st.status("Procesando dossier de Sucre…", expanded=True) as status_widget:
        def on_progress(pct, msg):
            paint(pct, msg)
            status_widget.update(label=f"{int(pct)}% · {msg}")

        try:
            result = process_sucre_dossier(df_file, progress=on_progress, ai_config=ai_config)
            paint(100, "Análisis completado")
            status_widget.update(label="✓ Análisis completado", state="complete")
        except SucreInputError as exc:
            status_widget.update(label="Archivo inválido", state="error")
            st.error(str(exc))
            raise
        except Exception as exc:
            logger.exception("Fallo en el proceso Sucre")
            status_widget.update(label="Error durante el procesamiento", state="error")
            st.error(f"El proceso se interrumpió: {exc}")
            raise

    st.session_state["sucre_output_data"] = result["output_data"]
    st.session_state["sucre_output_filename"] = result["output_filename"]
    st.session_state["sucre_complete"] = True
    st.session_state["sucre_total"] = result["total_rows"]
    st.session_state["sucre_duration"] = result["process_duration"]
    st.session_state["sucre_tono"] = result.get("tono_counts") or {}
    st.session_state["sucre_preview"] = result.get("preview_rows") or []


def main():
    st.set_page_config(
        page_title="Sucre · Análisis de noticias",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    load_custom_css()
    render_theme_toggle()
    if not check_password():
        return

    st.markdown("""
    <div class="app-header">
        <div class="app-header-icon">◈</div>
        <div class="app-header-text">
            <div class="app-header-title">Análisis de noticias · Gobernación de Sucre</div>
            <div class="app-header-version">Marca: Lucy Inés García Montes · extractos literales · v1.0</div>
        </div>
        <div class="app-header-badge">Variante Sucre</div>
    </div>""", unsafe_allow_html=True)

    if st.session_state.get("pending_sucre_dossier"):
        blob = st.session_state.pop("pending_sucre_dossier")
        meta = st.session_state.pop("pending_sucre_meta", {}) or {}
        ai_cfg = st.session_state.pop("pending_sucre_ai", None)
        run_sucre_process(io.BytesIO(blob), meta, ai_config=ai_cfg)
        st.rerun()

    if not st.session_state.get("sucre_complete", False):
        st.markdown(
            '<div class="rules-box">'
            "<b>Marca principal:</b> gobernadora Lucy Inés García Montes / Gobernación de Sucre. "
            "El tono (Positivo / Neutro / Negativo) se ancla a esa marca. "
            "Los extractos se copian <b>literalmente</b> de Título y CuerpoEs: sin paráfrasis, "
            "sin notas entre paréntesis y sin rellenos del tipo «(sin actor externo)»."
            "</div>",
            unsafe_allow_html=True,
        )

        with st.form("sucre_form"):
            st.markdown('<div class="sec-label">1. Sube el dossier (.xlsx)</div>', unsafe_allow_html=True)
            st.caption("Columnas críticas: Título y CuerpoEs. El resto de columnas de entrada se conservan.")
            f1 = st.file_uploader("Dossier Sucre", type=["xlsx"], label_visibility="collapsed", key="sucre_xlsx")

            st.markdown('<div class="sec-label">2. Análisis reputacional</div>', unsafe_allow_html=True)
            enable_ai = st.checkbox(
                "Activar análisis con IA (gpt-4.1-nano) para tono y actores",
                value=True,
                help="Si se desactiva, se usa solo la extracción heurística (útil para una pasada rápida).",
            )
            st.markdown(
                f"Columnas de salida: **{' · '.join(SUCRE_OUTPUT_COLUMNS)}**"
            )

            submitted = st.form_submit_button(
                "▶ Analizar noticias de Sucre",
                use_container_width=True,
                type="primary",
            )

        st.download_button(
            "⬇ Descargar xlsx de ejemplo (4 notas)",
            data=build_sample_xlsx(),
            file_name="sucre_dossier_ejemplo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        if submitted:
            if not f1:
                st.error("Por favor, sube un archivo Excel con Título y CuerpoEs.")
            else:
                api_key = st.secrets.get("OPENAI_API_KEY")
                if enable_ai and not api_key:
                    st.error("❌ Falta configurar OPENAI_API_KEY en los Secrets de Streamlit.")
                    st.stop()
                st.session_state["pending_sucre_dossier"] = f1.getvalue()
                st.session_state["pending_sucre_meta"] = {"name": f1.name}
                st.session_state["pending_sucre_ai"] = {
                    "enabled": bool(enable_ai),
                    "api_key": api_key if enable_ai else None,
                    "model": "gpt-4.1-nano-2025-04-14",
                }
                st.rerun()
    else:
        total = st.session_state.get("sucre_total", 0)
        dur = st.session_state.get("sucre_duration", "")
        tonos = st.session_state.get("sucre_tono") or {}
        st.markdown(
            '<div class="success-banner"><div class="success-icon">✓</div>'
            "<div><div class=\"success-title\">Proceso completado</div>"
            "<div class=\"success-sub\">El archivo con tono y extractos literales está listo para descargar</div></div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"""
        <div class="metrics-grid">
          <div class="metric-card"><div class="metric-val" style="color:var(--text)">{total}</div><div class="metric-lbl">Notas</div></div>
          <div class="metric-card"><div class="metric-val" style="color:var(--green)">{tonos.get("Positivo", 0)}</div><div class="metric-lbl">Positivo</div></div>
          <div class="metric-card"><div class="metric-val" style="color:var(--text3)">{tonos.get("Neutro", 0)}</div><div class="metric-lbl">Neutro</div></div>
          <div class="metric-card"><div class="metric-val" style="color:var(--red)">{tonos.get("Negativo", 0)}</div><div class="metric-lbl">Negativo</div></div>
        </div>""", unsafe_allow_html=True)
        st.caption(f"Tiempo de ejecución: {dur}")

        preview = st.session_state.get("sucre_preview") or []
        if preview:
            st.markdown('<div class="sec-label">Vista previa</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        c1.download_button(
            "⬇ Descargar xlsx Sucre",
            data=st.session_state.sucre_output_data,
            file_name=st.session_state.sucre_output_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary",
        )
        if c2.button("Nuevo análisis", use_container_width=True):
            pwd = st.session_state.get("password_correct")
            theme = st.session_state.get("ui_theme")
            st.session_state.clear()
            st.session_state.password_correct = pwd
            if theme in ("dark", "light"):
                st.session_state.ui_theme = theme
            st.rerun()

    st.markdown(
        '<div class="footer">Variante Sucre · no modifica el producto Grill (tono / tema / subtema) · Johnathan Cortés</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
