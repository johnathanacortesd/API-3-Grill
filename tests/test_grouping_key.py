# ======================================
# Grouping is optional: only same/similar título or shared first 3–4 words
# ======================================
import os
import sys
import unittest
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ai_analyzer import (
    canonicalize_subtopics,
    cluster_similar_rows,
    enrich_rows_with_ai,
    generate_brand_variants,
    should_group_news_items,
)
from pipeline import KEY_MAP


BRAND = "Universidad Tecnológica de Bolívar"
ALIASES = ["UTB"]
SUCRE_BRAND = "Gobernación de Sucre"
SUCRE_ALIASES = [
    "Lucy Inés García Montes",
    "Lucy García",
    "gobernadora Lucy",
    "Gobernadora de Sucre",
]
KM = KEY_MAP


def _row(title, body, duplicate=False):
    return {
        "Título": title,
        "Resumen - Aclaracion": body,
        "is_duplicate": duplicate,
    }


class ShouldGroupPredicateTests(unittest.TestCase):
    def test_same_first_three_title_words_group(self):
        self.assertTrue(
            should_group_news_items(
                "Feria Educativa Inspírate: 10 universidades en Cartagena",
                "Participan varias instituciones.",
                "Feria Educativa Inspírate: 3 días con becas",
                "Otro recuento de la feria.",
                BRAND,
                ALIASES,
            )
        )

    def test_same_first_four_title_words_group(self):
        self.assertTrue(
            should_group_news_items(
                "Women in Tech Latam Awards 2026 abren convocatoria",
                "La UTB participa.",
                "Women in Tech Latam Awards 2026 - NOTICIAS VITAL",
                "Cobertura del premio.",
                BRAND,
                ALIASES,
            )
        )

    def test_identical_titles_group(self):
        t = "Inauguración sede norte en Cali"
        self.assertTrue(should_group_news_items(t, "a", t, "b", "UdeA", []))

    def test_unrelated_titles_do_not_group(self):
        self.assertFalse(
            should_group_news_items(
                "Feria Educativa Inspírate abre becas UTB",
                "La UTB abre becas en la feria.",
                "Congreso de patrimonio cultural en Cartagena",
                "La UTB organiza un congreso de patrimonio.",
                BRAND,
                ALIASES,
            )
        )

    def test_shared_client_city_does_not_force_group(self):
        self.assertFalse(
            should_group_news_items(
                "Cátedra FICCI-UTB de cine y memoria en Cartagena",
                "Arranca la cátedra FICCI de cine.",
                "UTB registra aumento de matrícula en Cartagena",
                "Sube la matrícula universitaria en Cartagena.",
                BRAND,
                ALIASES,
            )
        )

    def test_brand_prefix_alone_does_not_group(self):
        self.assertFalse(
            should_group_news_items(
                "Gobernadora Lucy García entregó 200 becas en Sincelejo",
                "Entrega de becas en Sincelejo.",
                "Gobernadora Lucy García inauguró el hospital de Sampués",
                "Inauguración hospitalaria en Sampués.",
                SUCRE_BRAND,
                SUCRE_ALIASES,
            )
        )

    def test_same_resumen_opening_groups_different_titles(self):
        lead = (
            "Ingenieros SIAB de la UTB apoyan comunidades del Quindío "
            "en el Eje Cafetero con asistencia técnica."
        )
        self.assertTrue(
            should_group_news_items(
                "Cobertura especial del Eje Cafetero",
                lead,
                "Otra mirada a la emergencia cafetera",
                "Ingenieros SIAB de la UTB apoyan comunidades del Quindío tras las lluvias.",
                BRAND,
                ALIASES,
            )
        )


class ClusterSimilarRowsTests(unittest.TestCase):
    def _cm(self, rows, brand=BRAND, aliases=ALIASES):
        rx = generate_brand_variants(brand, aliases)
        return cluster_similar_rows(rows, KM, rx, brand=brand, aliases=aliases)

    def test_inspirate_first_words_share_cluster(self):
        rows = [
            _row("Feria Educativa Inspírate: 10 universidades presentarán su oferta", "lista"),
            _row("Feria Educativa Inspírate: 3 días con becas e inscripciones", "lista"),
            _row("Feria Educativa Inspírate 2026 impulsa el acceso a la educación", "lista"),
        ]
        cm = self._cm(rows)
        self.assertEqual(len(set(cm.values())), 1)

    def test_unrelated_titles_get_distinct_clusters(self):
        rows = [
            _row("Feria Educativa Inspírate abre becas UTB", "feria"),
            _row("Congreso de patrimonio cultural en Cartagena", "congreso"),
            _row("Incremento en matrícula universitaria en Cartagena", "matricula"),
        ]
        cm = self._cm(rows)
        self.assertEqual(len(set(cm.values())), 3)

    def test_no_forced_grouping_on_keyword_overlap(self):
        rows = [
            _row(
                "Cemento País: ingenieros SIAB al Eje Cafetero",
                "El programa SIAB envió ingenieros de la UTB al Eje Cafetero.",
            ),
            _row(
                "Universidad tecnológica de bolívar apoya en quindío",
                "La UTB apoya en el Quindío a través de otra agenda académica.",
            ),
        ]
        cm = self._cm(rows)
        self.assertNotEqual(cm[0], cm[1])

    def test_sucre_brand_prefix_does_not_glue_distinct_acts(self):
        rows = [
            _row(
                "Gobernadora Lucy García entregó 200 becas en Sincelejo",
                "La gobernadora Lucy García Montes entregó 200 becas universitarias.",
            ),
            _row(
                "Gobernadora Lucy García inauguró el hospital de Sampués",
                "La gobernadora inauguró el hospital de Sampués.",
            ),
        ]
        cm = self._cm(rows, SUCRE_BRAND, SUCRE_ALIASES)
        self.assertNotEqual(cm[0], cm[1])


class CanonicalizeDoesNotGlueTests(unittest.TestCase):
    def test_similar_llm_subtemas_do_not_merge_clusters(self):
        results = {
            0: ("Neutro", "Educación", "Agenda académica en Cartagena"),
            1: ("Positivo", "Cultura", "Agenda cultural en Cartagena"),
        }
        ctx = {
            0: "La UTB anunció matrícula y becas en Cartagena.",
            1: "La cátedra FICCI-UTB de cine se lanza en Cartagena.",
        }
        out = canonicalize_subtopics(results, ctx)
        self.assertEqual(out[0][2], "Agenda académica en Cartagena")
        self.assertEqual(out[1][2], "Agenda cultural en Cartagena")
        self.assertNotEqual(out[0][1], out[1][1])


class EnrichBroadcastTests(unittest.TestCase):
    def test_similar_first_words_share_group_and_subtema(self):
        rows = [
            _row(
                "Feria Educativa Inspírate: 10 universidades presentarán su oferta",
                "La UTB participa en la Feria Educativa Inspírate con becas.",
            ),
            _row(
                "Feria Educativa Inspírate: 3 días con becas e inscripciones",
                "Cobertura de la Feria Educativa Inspírate en Cartagena.",
            ),
        ]
        with patch("ai_analyzer.OpenAI"):
            with patch("ai_analyzer._call_openai_cluster") as mock_llm:
                mock_llm.return_value = (
                    "Neutro",
                    "Educación Superior",
                    "Feria educativa inspírate con becas",
                )
                out = enrich_rows_with_ai(rows, KM, BRAND, ALIASES, "sk-test")
        self.assertEqual(mock_llm.call_count, 1)
        self.assertEqual(out[0]["Subtema_IA"], out[1]["Subtema_IA"])
        self.assertEqual(out[0]["Tema_IA"], out[1]["Tema_IA"])
        self.assertEqual(out[0]["Tono_IA"], out[1]["Tono_IA"])
        self.assertIn("feria", out[0]["Subtema_IA"].lower())

    def test_unrelated_titles_keep_own_subtema(self):
        rows = [
            _row(
                "Feria Educativa Inspírate abre becas UTB",
                "La Universidad Tecnológica de Bolívar abre becas en la Feria Educativa Inspírate.",
            ),
            _row(
                "Congreso de patrimonio cultural en Cartagena",
                "La Universidad Tecnológica de Bolívar organiza un congreso de patrimonio cultural.",
            ),
        ]
        with patch("ai_analyzer.OpenAI"):
            with patch("ai_analyzer._call_openai_cluster") as mock_llm:
                def _fake(*args, **kwargs):
                    title = str(kwargs.get("title_ref") or (args[6] if len(args) > 6 else ""))
                    if "patrimonio" in title.lower() or "congreso" in title.lower():
                        return ("Neutro", "Cultura", "Congreso de patrimonio cultural")
                    return ("Neutro", "Educación", "Feria educativa inspírate con becas")

                mock_llm.side_effect = _fake
                out = enrich_rows_with_ai(rows, KM, BRAND, ALIASES, "sk-test")
        self.assertEqual(mock_llm.call_count, 2)
        self.assertNotEqual(out[0]["Subtema_IA"], out[1]["Subtema_IA"])
        self.assertNotEqual(out[0]["Tema_IA"], out[1]["Tema_IA"])

    def test_no_forced_grouping_shared_city_and_brand(self):
        ficci = (
            "Cartagena, cine y memoria: así comienza la nueva cátedra FICCI-UTB. "
            "La Universidad Tecnológica de Bolívar y el FICCI presentan la Cátedra."
        )
        matricula = (
            "La Universidad Tecnológica de Bolívar en Cartagena presenta un incremento "
            "en matrícula universitaria."
        )
        rows = [
            _row("Cartagena, cine y memoria: así comienza la nueva cátedra FICCI-UTB", ficci),
            _row("Incremento en matrícula universitaria en Cartagena", matricula),
        ]
        with patch("ai_analyzer.OpenAI"):
            with patch("ai_analyzer._call_openai_cluster") as mock_llm:
                def _fake(*args, **kwargs):
                    title = str(kwargs.get("title_ref") or (args[6] if len(args) > 6 else ""))
                    if "ficci" in title.lower() or "cine" in title.lower():
                        return ("Neutro", "Cultura", "Cátedra FICCI cine y memoria")
                    return ("Positivo", "Educación Superior", "Incremento en matrícula universitaria")

                mock_llm.side_effect = _fake
                out = enrich_rows_with_ai(rows, KM, BRAND, ALIASES, "sk-test")
        self.assertGreaterEqual(mock_llm.call_count, 2)
        self.assertNotEqual(out[0]["Subtema_IA"], out[1]["Subtema_IA"])
        self.assertNotIn("matrícula", out[0]["Subtema_IA"].lower())
        self.assertNotIn("matricula", out[0]["Subtema_IA"].lower())


if __name__ == "__main__":
    unittest.main()
