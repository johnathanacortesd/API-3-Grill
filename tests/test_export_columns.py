# ======================================
# Contrato de columnas Excel: sin resumen corto / revalorización,
# Resumen - Aclaracion termina en un punto, Contexto analizado al final.
# ======================================
import io
import os
import sys
import unittest

from openpyxl import load_workbook

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline import (  # noqa: E402
    AI_OUTPUT_COLUMNS,
    BASE_OUTPUT_COLUMNS,
    KEY_MAP,
    build_export_columns,
    corregir_texto,
    generate_output_excel,
)
from sucre_analyzer import SUCRE_ACTOR_COLUMNS  # noqa: E402


class BaseOutputColumnsTests(unittest.TestCase):
    def test_dropped_columns_are_not_exported(self):
        self.assertNotIn("resumen corto", BASE_OUTPUT_COLUMNS)
        self.assertNotIn("revalorización", BASE_OUTPUT_COLUMNS)

    def test_contexto_analizado_is_last_ai_column(self):
        self.assertEqual(AI_OUTPUT_COLUMNS[-1], "Contexto analizado")
        self.assertEqual(
            AI_OUTPUT_COLUMNS,
            ["Tono_IA", "Tema_IA", "Subtema_IA", "Contexto analizado"],
        )

    def test_generate_output_excel_omits_dropped_columns(self):
        rows = [
            {
                "ID Noticia": 1,
                "Título": "Nota",
                "resumen corto": "no debe salir",
                "revalorización": 123,
                "Resumen - Aclaracion": "Cuerpo de la nota.",
                "Contexto analizado": "marca en el párrafo",
                "Tono_IA": "Neutro",
                "Tema_IA": "Tema",
                "Subtema_IA": "Subtema",
            }
        ]
        data = generate_output_excel(rows, KEY_MAP)
        wb = load_workbook(io.BytesIO(data))
        headers = [c.value for c in wb["Resultado"][1]]
        self.assertNotIn("resumen corto", headers)
        self.assertNotIn("revalorización", headers)
        self.assertEqual(headers, list(BASE_OUTPUT_COLUMNS))
        wb.close()

        data_ai = generate_output_excel(
            rows, KEY_MAP, columns_to_use=build_export_columns(True)
        )
        wb = load_workbook(io.BytesIO(data_ai))
        headers = [c.value for c in wb["Resultado"][1]]
        self.assertEqual(headers, build_export_columns(True))
        self.assertEqual(headers[-4:], list(AI_OUTPUT_COLUMNS))
        self.assertEqual(headers[-1], "Contexto analizado")
        self.assertNotIn("resumen corto", headers)
        self.assertNotIn("revalorización", headers)
        wb.close()

    def test_build_export_columns_puts_contexto_last(self):
        self.assertEqual(build_export_columns(False), list(BASE_OUTPUT_COLUMNS))
        with_ai = build_export_columns(True)
        self.assertEqual(with_ai[-4:], list(AI_OUTPUT_COLUMNS))
        self.assertEqual(with_ai[-1], "Contexto analizado")

        sucre = build_export_columns(True, SUCRE_ACTOR_COLUMNS)
        self.assertEqual(sucre[-1], "Contexto analizado")
        for col in SUCRE_ACTOR_COLUMNS:
            self.assertLess(sucre.index(col), sucre.index("Contexto analizado"))
        self.assertEqual(
            sucre[-(len(SUCRE_ACTOR_COLUMNS) + 1):-1],
            list(SUCRE_ACTOR_COLUMNS),
        )
        self.assertNotIn("resumen corto", sucre)
        self.assertNotIn("revalorización", sucre)


class CorregirTextoTests(unittest.TestCase):
    def test_ends_with_single_period_not_ellipsis(self):
        self.assertEqual(corregir_texto("La marca anunció un plan"), "La marca anunció un plan.")
        self.assertEqual(corregir_texto("La marca anunció un plan."), "La marca anunció un plan.")
        self.assertEqual(corregir_texto("La marca anunció un plan..."), "La marca anunció un plan.")

    def test_empty_stays_empty(self):
        self.assertEqual(corregir_texto(""), "")
        self.assertEqual(corregir_texto("nan"), "")


if __name__ == "__main__":
    unittest.main()
