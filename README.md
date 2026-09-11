# API-3-Grill

Apps Streamlit para limpiar y analizar dossiers de noticias en Excel.

Hay **dos entradas independientes**. El producto Grill (tono / tema / subtema, 22 columnas, PKL, mapas de región) no se modifica al usar la variante Sucre.

| App | Archivo | Marca | Clasificación |
| --- | --- | --- | --- |
| Grill (principal) | `app.py` | La que indiques en el formulario | Tono, Tema, Subtema |
| Sucre (alterna) | `app_sucre.py` | Lucy Inés García Montes / Gobernación de Sucre | Tono + actores e intervenciones literales |

## App Grill (producto principal)

```bash
streamlit run app.py
```

En Streamlit Cloud, deja **Main file path** en `app.py`. Secrets típicos: `APP_PASSWORD`, `OPENAI_API_KEY`, `REGIONES_CSV_URL`, `INTERNET_CSV_URL`.

## Variante Sucre (Gobernación de Sucre / Lucy Inés García Montes)

Entrada: `app_sucre.py`. No pide marca ni PKL: la marca está fija. Conserva las columnas del xlsx de entrada y **añade** al final:

- `Tono` — Positivo / Neutro / Negativo respecto a Lucy / Gobernación de Sucre (título + cuerpo como contexto).
- `Nombre y cargo — actores propios` — Lucy Inés García Montes (Gobernadora de Sucre) o secretarías / voceros de la Gobernación cuando intervienen.
- `Intervención actores propios (extracto)` — tramo **literal** de `CuerpoEs` (o `Título`) de lo que Lucy dijo/hizo; si ella no interviene, de lo que hizo la Gobernación (resoluciones, acciones, comunicados).
- `Nombre y cargo — actores externos` — quien habla de ellas. Vacío si no hay (nunca textos tipo «(sin actor externo)»).
- `Mención / intervención externa (extracto)` — tramo **literal** de la mención o de la intervención externa.

Los extractos se validan contra el origen: si el modelo parafrasea, se descarta y se usa un tramo literal. Cadena vacía cuando no hay extracto.

Alias reconocidos (entre otros): Lucy Inés García Montes, Lucy García Montes, Lucy García, gobernadora de Sucre, Gobernación de Sucre, Secretaría de Educación departamental y secretarías análogas.

Columnas de entrada críticas: `Título`, `CuerpoEs`. Si el cuerpo viene como `Resumen - Aclaracion` / `Resumen` / `Cuerpo`, también se acepta. El resto de columnas se conserva.

### Cómo ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app_sucre.py
```

Crea `.streamlit/secrets.toml` (no se versiona):

```toml
APP_PASSWORD = "tu-clave"
OPENAI_API_KEY = "sk-..."
```

En la interfaz puedes descargar `sucre_dossier_ejemplo.xlsx` (4 notas de prueba).

### Cómo desplegar en Streamlit Cloud

1. Repositorio: este mismo (`API-3-Grill`). Crea una **app nueva** (no reutilices la de Grill si quieres aislar URLs).
2. **Main file path:** `app_sucre.py` (no `app.py`).
3. **Python version:** 3.11 o 3.12.
4. **Secrets** (☰ → Settings → Secrets):

```toml
APP_PASSWORD = "..."
OPENAI_API_KEY = "..."
```

La variante Sucre **no** usa `REGIONES_CSV_URL` ni `INTERNET_CSV_URL` (esos son del Grill).

5. Deploy. La URL queda como `https://<app>.streamlit.app`.
6. Prueba: entra con la contraseña, sube un xlsx con `Título` y `CuerpoEs`, activa IA, descarga el resultado y verifica tono + extractos literales (deben aparecer como texto del cuerpo, sin notas editoriales).

Para actualizar, un push a la rama conectada redespliega la app.

## Desarrollo

```bash
python -m unittest discover -s tests -v
```

Las pruebas de Sucre (`tests/test_sucre.py`) no reescriben el clasificador Grill. El camino principal sigue en `pipeline.py` + `ai_analyzer.py` + `app.py`.
