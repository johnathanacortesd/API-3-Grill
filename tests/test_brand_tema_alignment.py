# ======================================
# Brand-anchored tono, similar-news labels, tema-from-subtema
# ======================================
import unittest
from unittest.mock import patch

from ai_analyzer import (
    brand_tone_examples,
    brand_tone_instructions,
    canonicalize_subtopics,
    check_positive_institutional_override,
    clean_tema,
    cluster_similar_rows,
    enrich_rows_with_ai,
    generate_brand_variants,
    synthesize_temas_from_subtemas,
    _subtemas_related,
)
from pipeline import KEY_MAP
from test_pkl_subtema_grouping import _PredictByKeyword, _news_row


class BrandAnchoredTonoTests(unittest.TestCase):
    def test_instructions_score_brand_not_article_mood(self):
        brand = "Ecopetrol"
        aliases = ["ECO"]
        text = brand_tone_instructions(brand, aliases)
        examples = brand_tone_examples(brand)
        blob = f"{text}\n{examples}".lower()
        self.assertIn("ecopetrol", blob)
        self.assertIn("eco", blob)
        self.assertIn("tercero", blob)
        self.assertIn("sentimiento general", blob)
        self.assertNotIn("universidad autónoma de occidente", blob)
        self.assertNotIn("uao y dian", blob)
        self.assertIn("positivo", blob)
        self.assertIn("negativo", blob)
        self.assertIn("neutro", blob)

    def test_positive_override_requires_the_brand(self):
        ctx_brand = "Ecopetrol celebra y respalda el nombramiento del nuevo ministro."
        ctx_other = "Otra empresa celebra y respalda el nombramiento del nuevo ministro."
        self.assertTrue(check_positive_institutional_override(ctx_brand, "Ecopetrol", ["ECO"]))
        self.assertFalse(check_positive_institutional_override(ctx_other, "Ecopetrol", ["ECO"]))

    def test_variants_are_generic_not_hardcoded_clients(self):
        import re

        rx = generate_brand_variants("Banco Popular", ["BP"])
        hay = "el banco popular anunció resultados"
        self.assertTrue(any(re.search(r, hay) for r in rx))
        self.assertTrue(any("bancopopular" in r for r in rx))
        self.assertFalse(any("santa fe" in r for r in rx))
        self.assertFalse(any("serena del mar" in r for r in rx))


class TemaFromSubtemaTests(unittest.TestCase):
    def test_same_subtema_unifies_tono_and_tema(self):
        aligned = canonicalize_subtopics(
            {
                0: ("Positivo", "Infraestructura", "Apertura de sede norte"),
                1: ("Neutro", "Educación Superior", "Apertura de sede norte"),
                2: ("Negativo", "Seguridad Ciudadana", "Captura de banda criminal"),
            }
        )
        self.assertEqual(aligned[0][2], aligned[1][2])
        self.assertEqual(aligned[0][1], aligned[1][1])
        self.assertEqual(aligned[0][0], aligned[1][0])
        self.assertNotEqual(aligned[2][2], aligned[0][2])
        self.assertNotEqual(aligned[2][1], aligned[0][1])

    def test_related_subtemas_share_synthesized_tema(self):
        self.assertTrue(_subtemas_related("Apertura de sede norte", "Inauguración de sede norte"))
        self.assertFalse(_subtemas_related("Apertura de sede norte", "Apertura de convocatoria de becas"))
        out = synthesize_temas_from_subtemas(
            {
                0: ("Positivo", "Infraestructura", "Apertura de sede norte"),
                1: ("Neutro", "Educación Superior", "Inauguración de sede norte"),
                2: ("Negativo", "Seguridad Ciudadana", "Captura de banda criminal"),
            }
        )
        self.assertEqual(out[0][1], out[1][1])
        self.assertIn(out[0][1], {"Infraestructura", "Educación Superior"})
        self.assertEqual(out[2][1], "Seguridad Ciudadana")

    def test_pkl_path_does_not_rewrite_unrelated_pkl_temas(self):
        aligned = canonicalize_subtopics(
            {
                0: ("Neutro", "Entrevista", "Diálogo con el rector"),
                1: ("Neutro", "Mención", "Apertura de sede norte"),
            },
            unify_related_themes=False,
        )
        self.assertEqual(aligned[0][1], "Entrevista")
        self.assertEqual(aligned[1][1], "Mención")

    def test_clean_tema_allows_five_words_and_rejects_otros(self):
        self.assertEqual(clean_tema("Gestión de Emergencias Públicas Hoy"), "Gestión De Emergencias Públicas Hoy")
        self.assertEqual(clean_tema("Otros"), "Gestión Institucional")

    def test_enrich_keeps_shared_subtema_and_synthesized_tema(self):
        rows = [
            _news_row(
                "Inauguración sede norte en Cali",
                "La universidad inauguró la sede norte en Cali.",
            ),
            _news_row(
                "Inauguración sede norte en Cali hoy",
                "Inauguró la sede norte para ampliar cobertura.",
            ),
            _news_row(
                "Capturan banda en el centro",
                "La policía capturó una banda criminal en el centro.",
            ),
        ]
        with patch("ai_analyzer.OpenAI"):
            with patch("ai_analyzer._call_openai_cluster") as mock_llm:
                def _fake_llm(*args, **kwargs):
                    title = str(kwargs.get("title_ref") or (args[6] if len(args) > 6 else ""))
                    if "captur" in title.lower():
                        return ("Negativo", "Seguridad Ciudadana", "Captura de banda criminal")
                    return ("Positivo", "Infraestructura", "Apertura de sede norte")

                mock_llm.side_effect = _fake_llm
                out = enrich_rows_with_ai(rows, KEY_MAP, "UdeA", ["Alma Mater"], "sk-test")
        self.assertEqual(out[0]["Subtema_IA"], out[1]["Subtema_IA"])
        self.assertEqual(out[0]["Tema_IA"], out[1]["Tema_IA"])
        self.assertEqual(out[0]["Tono_IA"], out[1]["Tono_IA"])
        self.assertNotEqual(out[2]["Subtema_IA"], out[0]["Subtema_IA"])
        self.assertGreaterEqual(len(out[0]["Subtema_IA"].split()), 4)

    def test_similar_news_cluster_via_brand_context(self):
        rows = [
            {
                "Título": "Cali estrena campus",
                "Resumen - Aclaracion": "texto largo irrelevante de otra sección",
                "Contexto analizado": "La universidad inauguró la sede norte en Cali para ampliar cobertura.",
                "is_duplicate": False,
            },
            {
                "Título": "Nueva sede universitaria en el norte",
                "Resumen - Aclaracion": "otro cuerpo distinto",
                "Contexto analizado": "La universidad inauguró la sede norte en Cali para ampliar cobertura educativa.",
                "is_duplicate": False,
            },
            {
                "Título": "Denuncian cobros en peajes del valle",
                "Resumen - Aclaracion": "usuarios reportan tarifas",
                "Contexto analizado": "Usuarios denuncian cobros excesivos en peajes del valle.",
                "is_duplicate": False,
            },
        ]
        cm = cluster_similar_rows(rows, KEY_MAP, [])
        self.assertEqual(cm[0], cm[1])
        self.assertNotEqual(cm[0], cm[2])


if __name__ == "__main__":
    unittest.main()
