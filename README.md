# API-3-Grill

Apps Streamlit para limpiar y analizar dossiers de noticias en Excel.

Hay **dos entradas independientes** (dos apps en Streamlit Cloud). El producto Grill (`app.py`: tono / tema / subtema, 22 columnas, PKL, mapas de región, estilo Link) no se modifica al usar la variante Sucre.

| App | Archivo | Marca | Qué hace |
| --- | --- | --- | --- |
| Grill (principal) | `app.py` | La que indiques en el formulario | Pipeline completo: xlsx, región/internet, duplicados, tono/tema/subtema, PKL, Link, descarga |
| Sucre (alterna) | `app_sucre.py` | Gobernación de Sucre / Lucy Inés García Montes | **El mismo pipeline Grill** + 4 columnas de actores (solo personas) |

## App Grill (producto principal)

```bash
streamlit run app.py
```

En Streamlit Cloud, deja **Main file path** en `app.py`. Secrets: `APP_PASSWORD`, `OPENAI_API_KEY`, `REGIONES_CSV_URL`, `INTERNET_CSV_URL`.

## Variante Sucre (Gobernación de Sucre / Lucy Inés García Montes)

Entrada: `app_sucre.py`. Es una **copia completa** de Grill (mismo formulario: dossier xlsx, marca/alias, PKL opcionales, Sheets de región/internet, métricas, descarga, estilo Link, hojas) con la marca anclada a **Gobernación de Sucre** y la gobernadora **Lucy Inés García Montes** (y variantes: Lucy García Montes, Lucy Montes, Lucy García, gobernadora de Sucre).

Al resultado Grill se **añaden** estas columnas (no se quita ninguna columna Grill):

1. `Nombre y cargo — actores propios` — solo **personas**. Lucy (si ella interviene) o un **secretario/secretaria nombrado/a** de la Gobernación de Sucre.
2. `Intervención actores propios (extracto)` — tramo **literal** de `CuerpoEs` de lo que esa persona dijo/hizo.
3. `Nombre y cargo — actores externos` — persona **con nombre + cargo** que habla/opina y se refiere a la Gobernación o a Lucy (ej. `Andrés Julián Rendón, Gobernador de Antioquia`).
4. `Mención / intervención externa (extracto)` — tramo **literal** de `CuerpoEs` de esa intervención.

El tono reputacional sigue siendo el de Grill (`Tono_IA` Positivo / Neutro / Negativo) anclado a Gobernación de Sucre / Lucy, usando Título + CuerpoEs.

### Reglas duras de actores

- Nunca entidades sueltas como «Nombre y cargo»: `Secretaría de Educación departamental`, `Gobernación de Sucre`, `Ministerio del Interior`, `Presidencia de la República`.
- Si Lucy **no** interviene en el cuerpo, **no** va su nombre en propios.
- Extractos = copia literal de `CuerpoEs`. Sin intros, sin paráfrasis, sin notas editoriales.
- Si no hay actor, cadena vacía (nunca «(sin actor externo)» ni N/A).

### Cómo ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app_sucre.py
```

Crea `.streamlit/secrets.toml` (no se versiona):

```toml
APP_PASSWORD = "tu-clave"
OPENAI_API_KEY = "sk-..."
REGIONES_CSV_URL = "https://..."
INTERNET_CSV_URL = "https://..."
```

En la interfaz puedes descargar `sucre_dossier_ejemplo.xlsx`.

### Cómo desplegar en Streamlit Cloud

Crea una **app nueva** (no reutilices la de Grill si quieres aislar URLs).

1. Repositorio: este mismo (`API-3-Grill`).
2. **Main file path:** `app_sucre.py` (no `app.py`). Es otra app Streamlit.
3. **Python version:** 3.11 o 3.12.
4. **Secrets** (☰ → Settings → Secrets) — los mismos que Grill:

```toml
APP_PASSWORD = "..."
OPENAI_API_KEY = "..."
REGIONES_CSV_URL = "..."
INTERNET_CSV_URL = "..."
```

5. Deploy. La URL queda como `https://<app>.streamlit.app`.
6. Prueba: entra con la contraseña, sube un xlsx Grill (Título + CuerpoEs / Resumen), activa IA, opcionalmente PKL, descarga el resultado y verifica:
   - columnas Grill (incl. Tono_IA / Tema_IA / Subtema_IA, Link);
   - las 4 columnas Sucre (personas + extractos literales del cuerpo).

Para actualizar, un push a la rama conectada redespliega la app.

## Desarrollo

```bash
python -m unittest discover -s tests -v
```

Las pruebas de Sucre (`tests/test_sucre.py`) no reescriben el clasificador Grill. El camino principal sigue en `pipeline.py` + `ai_analyzer.py` + `app.py`.
