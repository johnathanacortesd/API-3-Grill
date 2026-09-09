# ======================================
# Echo subtema, near-duplicate grouping, marca-centric tono
# ======================================
import os
import sys
import unittest
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
TESTS = os.path.dirname(os.path.abspath(__file__))
if TESTS not in sys.path:
    sys.path.insert(0, TESTS)

from ai_analyzer import (
    brand_tone_examples,
    brand_tone_instructions,
    check_list_mention_neutral,
    check_positive_institutional_override,
    cluster_similar_rows,
    enrich_rows_with_ai,
    ensure_subtema_distinct_from_tema,
    generate_brand_variants,
    subtema_supported_by_context,
    _has_echoed_content_pair,
)
from pipeline import KEY_MAP
from test_pkl_subtema_grouping import _PredictByKeyword, _news_row


BRAND = "Universidad Tecnológica de Bolívar"
ALIASES = ["UTB"]
KM = KEY_MAP


def _row(title, body, duplicate=False):
    return {
        "Título": title,
        "Resumen - Aclaracion": body,
        "is_duplicate": duplicate,
    }


class EchoSubtemaTests(unittest.TestCase):
    def test_detects_consecutive_repeated_content_words(self):
        self.assertTrue(_has_echoed_content_pair("Beneficios y beneficios académicos".split()))
        self.assertFalse(_has_echoed_content_pair("Becas e inscripciones en feria educativa".split()))

    def test_rejects_echo_and_grounds_in_feria_contexto(self):
        ctx = (
            "La Universidad Tecnológica de Bolívar abre beneficios de becas e "
            "inscripciones en la Feria Educativa Inspírate para nuevos estudiantes."
        )
        sub = ensure_subtema_distinct_from_tema(
            "Beneficios Académicos",
            "Beneficios y beneficios académicos",
            BRAND,
            "Feria Educativa Inspírate UTB",
            ctx,
        )
        low = sub.strip().lower()
        self.assertNotEqual(low, "beneficios y beneficios académicos")
        self.assertFalse(_has_echoed_content_pair(sub.split()))
        self.assertGreaterEqual(len(sub.split()), 4)
        self.assertLessEqual(len(sub.split()), 7)
        self.assertTrue(
            "beca" in low or "feria" in low or "inscrip" in low,
            f"subtema should name the news fact, got {sub!r}",
        )

    def test_rejects_brand_only_participation_phrase(self):
        ctx = (
            "La Universidad Tecnológica de Bolívar participa en el Congreso de "
            "patrimonio cultural de Cartagena con una ponencia sobre archivos."
        )
        sub = ensure_subtema_distinct_from_tema(
            "Eventos",
            "Participación de la universidad tecnológica de bolívar",
            BRAND,
            "Congreso de patrimonio cultural",
            ctx,
        )
        low = sub.strip().lower()
        self.assertNotEqual(low, "participación de la universidad tecnológica de bolívar")
        self.assertGreaterEqual(len(sub.split()), 4)
        self.assertLessEqual(len(sub.split()), 7)
        self.assertTrue(
            "congreso" in low or "patrimonio" in low or "cartagena" in low,
            f"subtema should prefer the event, got {sub!r}",
        )


class NearSimilarClusteringTests(unittest.TestCase):
    def test_siab_quindio_variants_share_cluster_unrelated_do_not(self):
        rows = [
            _row(
                "Cemento País: ingenieros SIAB al Eje Cafetero",
                "El programa SIAB envió ingenieros de la Universidad Tecnológica de Bolívar "
                "al Eje Cafetero para apoyar obras de Cemento País.",
            ),
            _row(
                "Ingenieros SIAB de la UTB apoyan en el Quindío",
                "Ingenieros SIAB de la UTB apoyan comunidades del Quindío en el Eje Cafetero "
                "con asistencia técnica.",
            ),
            _row(
                "Universidad tecnológica de bolívar apoya en quindío",
                "La Universidad Tecnológica de Bolívar apoya en el Quindío a través del "
                "programa SIAB de ingenieros.",
            ),
            _row(
                "SIAB: ingenieros al Eje Cafetero - NOTICIAS VITAL",
                "SIAB lleva ingenieros al Eje Cafetero. Participa la Universidad Tecnológica de Bolívar.",
            ),
            _row(
                "Feria Educativa Inspírate abre becas UTB",
                "La Universidad Tecnológica de Bolívar abre becas e inscripciones en la "
                "Feria Educativa Inspírate.",
            ),
        ]
        rx = generate_brand_variants(BRAND, ALIASES)
        cm = cluster_similar_rows(rows, KM, rx, brand=BRAND, aliases=ALIASES)
        # Same SIAB/Quindío *story* titles share a grupo. A brand-only Quindío
        # title is not a near-duplicate and must not steal another fact's label.
        self.assertEqual(cm[0], cm[1])
        self.assertEqual(cm[0], cm[3])
        self.assertNotEqual(cm[0], cm[4])

    def test_women_in_tech_and_ficci_variants_share_cluster(self):
        women = [
            _row(
                "Women in Tech Latam Awards 2026",
                "La Universidad Tecnológica de Bolívar participa en Women in Tech Latam Awards 2026.",
            ),
            _row(
                "Women in Tech Latam Awards 2026 - NOTICIAS VITAL",
                "Cobertura de Women in Tech Latam Awards 2026. La UTB fue reconocida.",
            ),
            _row(
                "Ganadoras del Women in Tech Latam Awards",
                "Estudiantes de la Universidad Tecnológica de Bolívar fueron ganadoras del "
                "Women in Tech Latam Awards.",
            ),
        ]
        ficci = [
            _row(
                "Cátedra FICCI-UTB inaugura temporada",
                "La Cátedra FICCI-UTB inaugura temporada académica con conferencistas.",
            ),
            _row(
                "Cátedra FICCIUTB inaugura temporada",
                "Arranca la Cátedra FICCIUTB con la Universidad Tecnológica de Bolívar.",
            ),
            _row(
                "1ª Cátedra FICCI-UTB",
                "Se realizó la 1ª Cátedra FICCI-UTB en el campus de Cartagena.",
            ),
        ]
        rx = generate_brand_variants(BRAND, ALIASES)
        cm_w = cluster_similar_rows(women, KM, rx, brand=BRAND, aliases=ALIASES)
        self.assertEqual(cm_w[0], cm_w[1])
        self.assertEqual(cm_w[0], cm_w[2])
        cm_f = cluster_similar_rows(ficci, KM, rx, brand=BRAND, aliases=ALIASES)
        self.assertEqual(cm_f[0], cm_f[1])
        self.assertEqual(cm_f[0], cm_f[2])

    def test_enrich_broadcasts_one_subtema_to_near_matches_not_duplicates(self):
        rows = [
            _row(
                "Women in Tech Latam Awards 2026",
                "La Universidad Tecnológica de Bolívar participa en Women in Tech Latam Awards 2026.",
            ),
            _row(
                "Women in Tech Latam Awards 2026 - NOTICIAS VITAL",
                "Cobertura de Women in Tech Latam Awards 2026 con presencia de la UTB.",
            ),
            _row(
                "Women in Tech Latam Awards 2026",
                "mismo url duplicado",
            ),
            _row(
                "Congreso de patrimonio cultural en Cartagena",
                "La Universidad Tecnológica de Bolívar organiza un congreso de patrimonio cultural.",
            ),
        ]
        rows[2]["is_duplicate"] = True
        with patch("ai_analyzer.OpenAI"):
            with patch("ai_analyzer._call_openai_cluster") as mock_llm:
                def _fake(*args, **kwargs):
                    title = str(kwargs.get("title_ref") or (args[6] if len(args) > 6 else ""))
                    if "patrimonio" in title.lower() or "congreso" in title.lower():
                        return ("Neutro", "Cultura", "Congreso de patrimonio cultural")
                    return ("Neutro", "Premios", "Premios women in tech latam")

                mock_llm.side_effect = _fake
                out = enrich_rows_with_ai(rows, KM, BRAND, ALIASES, "sk-test")
        self.assertEqual(out[0]["Subtema_IA"], out[1]["Subtema_IA"])
        self.assertEqual(out[0]["Tema_IA"], out[1]["Tema_IA"])
        self.assertEqual(out[0]["Tono_IA"], out[1]["Tono_IA"])
        self.assertEqual(out[2]["Tono_IA"], "Duplicada")
        self.assertEqual(out[2]["Subtema_IA"], "-")
        self.assertNotEqual(out[3]["Subtema_IA"], out[0]["Subtema_IA"])
        self.assertEqual(mock_llm.call_count, 2)

    def test_pkl_theme_path_still_keeps_llm_subtema_on_cluster(self):
        rows = [
            _news_row(
                "Women in Tech Latam Awards 2026",
                "La universidad participa en Women in Tech Latam Awards 2026.",
                subtema="Premios women in tech latam",
            ),
            _news_row(
                "Women in Tech Latam Awards 2026 - NOTICIAS VITAL",
                "Cobertura de Women in Tech Latam Awards 2026.",
                subtema="Otro",
            ),
        ]
        theme_model = _PredictByKeyword([], "Mención")
        with patch("ai_analyzer.OpenAI"):
            with patch("ai_analyzer._call_openai_cluster") as mock_llm:
                mock_llm.return_value = (
                    "Neutro",
                    "Educación Superior",
                    "Premios women in tech latam",
                )
                out = enrich_rows_with_ai(
                    rows, KM, BRAND, ALIASES, "sk-test", theme_model=theme_model
                )
        self.assertFalse(mock_llm.call_args.kwargs["request_theme"])
        self.assertEqual(out[0]["Tema_IA"], "Mención")
        self.assertEqual(out[1]["Tema_IA"], "Mención")
        self.assertEqual(out[0]["Subtema_IA"], out[1]["Subtema_IA"])
        self.assertIn("women", out[0]["Subtema_IA"].lower())
        self.assertIn("tech", out[0]["Subtema_IA"].lower())
        self.assertGreaterEqual(len(out[0]["Subtema_IA"].split()), 4)
        self.assertLessEqual(len(out[0]["Subtema_IA"].split()), 7)


class BrandCentricTonoTests(unittest.TestCase):
    def test_instructions_use_marca_not_article_mood(self):
        text = brand_tone_instructions("Ecopetrol", ["ECO"])
        examples = brand_tone_examples("Ecopetrol")
        blob = f"{text}\n{examples}".lower()
        self.assertIn("ecopetrol", blob)
        self.assertIn("eco", blob)
        self.assertIn("listado", blob)
        self.assertIn("participantes", blob)
        self.assertIn("sentimiento general", blob)
        self.assertNotIn("universidad autónoma de occidente", blob)
        self.assertNotIn("uao y dian", blob)

    def test_positive_override_requires_the_brand(self):
        ctx_brand = "Ecopetrol celebra y respalda el nombramiento del nuevo ministro."
        ctx_other = "Otra empresa celebra y respalda el nombramiento del nuevo ministro."
        self.assertTrue(check_positive_institutional_override(ctx_brand, "Ecopetrol", ["ECO"]))
        self.assertFalse(check_positive_institutional_override(ctx_other, "Ecopetrol", ["ECO"]))

    def test_list_of_participants_is_neutro(self):
        ctx = (
            "En el foro participaron la Universidad Nacional, la Universidad de los Andes, "
            "la Universidad Tecnológica de Bolívar y la Javeriana."
        )
        self.assertTrue(check_list_mention_neutral(ctx, BRAND, ALIASES))
        praise = (
            f"{BRAND} celebra y respalda el nombramiento y participaron otras universidades."
        )
        self.assertFalse(check_list_mention_neutral(praise, BRAND, ALIASES))


FICCI_CINE_CTX = (
    "Cartagena, cine y memoria: así comienza la nueva cátedra FICCI-UTB. "
    "El lanzamiento será este lunes 7 de septiembre con el conversatorio "
    '"La Cartagena de Quemada: cine, memoria y ciudad" La Universidad Tecnológica de Bolívar '
    "y el Festival Internacional de Cine de Cartagena de Indias (FICCI) presentan la Cátedra FICCI-UTB."
)
MATRICULA_CTX = (
    "La Universidad Tecnológica de Bolívar en Cartagena presenta un incremento en matrícula "
    "universitaria. El nuevo semestre en la ciudad recibe más estudiantes de la institución "
    "este 7 de septiembre con un lanzamiento de cifras académicas en pregrado."
)


class ContextGroundingTests(unittest.TestCase):
    def test_ficci_cine_never_gets_matricula_subtema(self):
        title = "Cartagena, cine y memoria: así comienza la nueva cátedra FICCI-UTB"
        sub = ensure_subtema_distinct_from_tema(
            "Educación Superior",
            "Incremento en matrícula universitaria en cartagena",
            BRAND,
            title,
            FICCI_CINE_CTX,
            ALIASES,
        )
        low = sub.strip().lower()
        self.assertNotIn("matrícula", low)
        self.assertNotIn("matricula", low)
        self.assertNotIn("beca", low)
        self.assertNotIn("inscrip", low)
        self.assertGreaterEqual(len(sub.split()), 4)
        self.assertLessEqual(len(sub.split()), 7)
        self.assertTrue(
            "ficci" in low or "cátedra" in low or "catedra" in low or "cine" in low,
            f"FICCI cine contexto must keep a cine/cátedra subtema, got {sub!r}",
        )
        self.assertTrue(subtema_supported_by_context(sub, FICCI_CINE_CTX, BRAND, ALIASES))

    def test_shared_brand_city_does_not_share_subtema(self):
        rows = [
            _row(
                "Cartagena, cine y memoria: así comienza la nueva cátedra FICCI-UTB",
                FICCI_CINE_CTX,
            ),
            _row("Incremento en matrícula universitaria en Cartagena", MATRICULA_CTX),
            _row("UTB registra aumento de matrícula en Cartagena", MATRICULA_CTX),
        ]
        rx = generate_brand_variants(BRAND, ALIASES)
        cm = cluster_similar_rows(rows, KM, rx, brand=BRAND, aliases=ALIASES)
        self.assertNotEqual(cm[0], cm[1], "FICCI cine must not share grupo with matrícula")
        self.assertNotEqual(cm[0], cm[2])

        with patch("ai_analyzer.OpenAI"):
            with patch("ai_analyzer._call_openai_cluster") as mock_llm:
                def _fake(*args, **kwargs):
                    title = str(kwargs.get("title_ref") or (args[6] if len(args) > 6 else ""))
                    ctx = str(kwargs.get("ctx") or (args[5] if len(args) > 5 else ""))
                    blob = f"{title} {ctx}".lower()
                    if "ficci" in blob or "cine" in blob or "cátedra" in blob or "catedra" in blob:
                        return ("Neutro", "Cultura", "Cátedra FICCI cine y memoria")
                    return ("Positivo", "Educación Superior", "Incremento en matrícula universitaria en cartagena")

                mock_llm.side_effect = _fake
                out = enrich_rows_with_ai(rows, KM, BRAND, ALIASES, "sk-test")
        ficci_sub = out[0]["Subtema_IA"].lower()
        mat_sub = out[1]["Subtema_IA"].lower()
        self.assertNotEqual(out[0]["Subtema_IA"], out[1]["Subtema_IA"])
        self.assertNotIn("matrícula", ficci_sub)
        self.assertNotIn("matricula", ficci_sub)
        self.assertTrue("ficci" in ficci_sub or "catedra" in ficci_sub or "cátedra" in ficci_sub or "cine" in ficci_sub)
        self.assertTrue(
            subtema_supported_by_context(out[0]["Subtema_IA"], FICCI_CINE_CTX, BRAND, ALIASES)
        )
        self.assertFalse(
            subtema_supported_by_context(
                "Incremento en matrícula universitaria en cartagena",
                FICCI_CINE_CTX,
                BRAND,
                ALIASES,
            )
        )
        # Wrong cluster-rep subtema must be rejected against this contexto.
        self.assertNotEqual(ficci_sub, mat_sub)

    def test_broadcast_rejects_foreign_subtema_for_this_contexto(self):
        rows = [
            _row(
                "Cartagena, cine y memoria: así comienza la nueva cátedra FICCI-UTB",
                FICCI_CINE_CTX,
            ),
        ]
        with patch("ai_analyzer.OpenAI"):
            with patch("ai_analyzer._call_openai_cluster") as mock_llm:
                mock_llm.return_value = (
                    "Positivo",
                    "Educación Superior",
                    "Incremento en matrícula universitaria en cartagena",
                )
                out = enrich_rows_with_ai(rows, KM, BRAND, ALIASES, "sk-test")
        low = out[0]["Subtema_IA"].lower()
        self.assertNotIn("matrícula", low)
        self.assertNotIn("matricula", low)
        self.assertTrue("ficci" in low or "catedra" in low or "cátedra" in low or "cine" in low)


if __name__ == "__main__":
    unittest.main()
