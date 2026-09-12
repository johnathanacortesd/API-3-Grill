# -*- coding: utf-8 -*-
# ======================================
# Bateria de casos reales para el tono (guarda positiva)
# Casos del dominio del cliente: universidad, gobernacion, gremio.
# Cada caso trae TITULAR y CUERPO como llegan del dossier (bloques separados).
# Sin API: se prueba la guarda determinista sobre un tono Neutro de partida.
# ======================================
import unittest

import analyzer_tono_tema as A

USB = 'Universidad Simón Bolívar'
SUCRE = 'Gobernación de Sucre'
FENAVI = 'FENAVI'

# (titular, cuerpo, marca, alias, tono de partida, tono esperado final)
CASOS = [
    # --- la marca es la autora: programas, obras, eventos y logros propios ---
    ("El programa de Medicina de la Universidad Simón Bolívar realizará los días 23, 24 y 25 de "
     "septiembre la I Conferencia Internacional Caribe de Educación Médica",
     "La Universidad Simón Bolívar informó que el encuentro reunirá a especialistas de la región.",
     USB, [], 'Neutro', 'Positivo'),
    ("La Universidad Simón Bolívar organizará el congreso nacional de ingeniería en Barranquilla",
     "La Universidad Simón Bolívar informó que el congreso reunirá a 500 asistentes.",
     USB, [], 'Neutro', 'Positivo'),
    ("Estudiantes de la Universidad Simón Bolívar ganaron el concurso nacional de robótica",
     "La Universidad Simón Bolívar acompañó al equipo ganador de la competencia nacional.",
     USB, [], 'Neutro', 'Positivo'),
    ("La Universidad Simón Bolívar recibió la acreditación de alta calidad por ocho años",
     "El organismo acreditador reconoció los programas académicos de la Universidad Simón Bolívar.",
     USB, [], 'Neutro', 'Positivo'),
    ("La Universidad Simón Bolívar recibió el premio nacional a la investigación",
     "El galardón destaca los proyectos de los grupos de investigación de la Universidad.",
     USB, [], 'Neutro', 'Positivo'),
    ("La Universidad Simón Bolívar ocupó el primer lugar entre las universidades del Caribe en el ranking",
     "El ranking nacional evalúa docencia, investigación y proyección social.",
     USB, [], 'Neutro', 'Positivo'),
    ("La Universidad Simón Bolívar abrirá dos nuevos programas de ingeniería en 2027",
     "La Universidad Simón Bolívar ampliará su oferta académica en la sede norte.",
     USB, [], 'Neutro', 'Positivo'),
    ("La Universidad Simón Bolívar firmó un convenio con la embajada de Canadá para intercambios académicos",
     "La Universidad Simón Bolívar gestionará las movilidades estudiantiles del convenio.",
     USB, [], 'Neutro', 'Positivo'),
    ("La Universidad Simón Bolívar graduó a 1.200 nuevos profesionales en su ceremonia de grado",
     "La Universidad Simón Bolívar entregó los títulos en la sede principal.",
     USB, [], 'Neutro', 'Positivo'),
    ("El Hospital de la Universidad Simón Bolívar atenderá gratis a 2.000 pacientes en jornada",
     "La Universidad Simón Bolívar desarrollará la jornada con sus estudiantes de salud.",
     USB, [], 'Neutro', 'Positivo'),
    ("La Universidad Simón Bolívar entregó el nuevo bloque de aulas en la sede norte",
     "La Universidad Simón Bolívar informó que la obra beneficia a 1.200 estudiantes.",
     USB, [], 'Neutro', 'Positivo'),
    ("La Universidad Simón Bolívar anunció una inversión de $20 mil millones para vías terciarias",
     "La Universidad Simón Bolívar detalló que los recursos saldrán de su presupuesto.",
     USB, [], 'Neutro', 'Positivo'),
    ("La Gobernación de Sucre entregará 500 kits escolares la próxima semana",
     "La Gobernación de Sucre informó que la entrega será en los municipios.",
     SUCRE, ['Sucre'], 'Neutro', 'Positivo'),
    ("La Gobernación de Sucre aprobó $39 mil millones para la variante Sampués - Segovia",
     "La Gobernación de Sucre informó que la obra mejorará la conectividad.",
     SUCRE, ['Sucre'], 'Neutro', 'Positivo'),
    ("La Gobernación de Sucre puso en marcha el programa de becas para 3.000 jóvenes",
     "La Gobernación de Sucre informó que las becas cubrirán la matrícula.",
     SUCRE, ['Sucre'], 'Neutro', 'Positivo'),
    ("FENAVI realizará su congreso de 2028 en Barranquilla", "El gremio confirmó la sede del evento.",
     FENAVI, ['Fenavi'], 'Neutro', 'Positivo'),

    # --- la marca solo aparece, asiste, participa o reporta: Neutro ---
    ("La Universidad Simón Bolívar participará en el foro nacional de educación superior",
     "La Universidad Simón Bolívar asistirá con una delegación de docentes.",
     USB, [], 'Neutro', 'Neutro'),
    ("La Universidad Simón Bolívar asistió a la reunión de rectores del Caribe",
     "La Universidad Simón Bolívar participó en la agenda de la reunión.",
     USB, [], 'Neutro', 'Neutro'),
    ("La Universidad Simón Bolívar advirtió sobre el aumento de la deserción escolar",
     "La Universidad Simón Bolívar presentó datos del sistema de seguimiento.",
     USB, [], 'Neutro', 'Neutro'),
    ("La Universidad Simón Bolívar presentó el informe Panorama de la Juventud: desempleo en alerta",
     "La Universidad Simón Bolívar entregó el informe al departamento.",
     USB, [], 'Neutro', 'Neutro'),
    ("El estudio de la Universidad Simón Bolívar revela brechas de salud mental en los jóvenes",
     "El estudio de la Universidad Simón Bolívar analizó 4.000 casos.",
     USB, [], 'Neutro', 'Neutro'),
    ("La Universidad Simón Bolívar analiza el impacto de la reforma tributaria",
     "La Universidad Simón Bolívar revisa el efecto en sus finanzas.",
     USB, [], 'Neutro', 'Neutro'),
    ("La Universidad Simón Bolívar anunció que el desempleo juvenil sigue creciendo",
     "La Universidad Simón Bolívar advirtió sobre la cifra del último trimestre.",
     USB, [], 'Neutro', 'Neutro'),
    ("La Universidad Simón Bolívar fue mencionada en el informe de la Contraloría",
     "El documento de la Contraloría cita varias universidades de la región.",
     USB, [], 'Neutro', 'Neutro'),
    ("El alcalde de Sincelejo pidió a la Universidad Simón Bolívar entregar los recursos del convenio",
     "El mandatario insistió en la ejecución del convenio firmado el año pasado.",
     USB, [], 'Neutro', 'Neutro'),
    ("Gobernadores del Caribe y la ANI evalúan el proyecto del canal del Dique",
     "La Gobernación de Sucre estuvo representada en la reunión de gobernadores.",
     SUCRE, ['Sucre'], 'Neutro', 'Neutro'),
    # contagio entre oraciones: la accion del Gobierno no es de la marca
    ("Anuncian la construcción de un colegio en Sincelejo",
     "El Gobierno nacional anunció la construcción de un colegio en Sincelejo. La Universidad "
     "Simón Bolívar acompaña el proceso en la región.",
     USB, [], 'Neutro', 'Neutro'),
    ("En Sucre destruyen más de 250 mil productos de contrabando",
     "Las autoridades reportaron la operación en el sur del departamento.",
     SUCRE, ['Sucre'], 'Neutro', 'Neutro'),
    ("Vecinos denuncian que la Universidad Simón Bolívar no ha terminado la obra del bloque",
     "La comunidad aseguró que los retrasos llevan dos años.",
     USB, [], 'Neutro', 'Neutro'),

    # --- lo que ya venia etiquetado no se toca ---
    ("La Contraloría sancionó a la Universidad Simón Bolívar por los sobrecostos en la obra",
     "El órgano de control cuestionó la demora de la institución.",
     USB, [], 'Negativo', 'Negativo'),
    ("La Universidad Simón Bolívar entregó el nuevo bloque de aulas",
     "La Universidad Simón Bolívar informó que la obra beneficia a 1.200 estudiantes.",
     USB, [], 'Positivo', 'Positivo'),
]


class TestCasosRealesTono(unittest.TestCase):

    def test_guarda_positiva_sobre_casos_reales(self):
        fallos = []
        for titulo, cuerpo, brand, aliases, inicial, esperado in CASOS:
            grupos = [{'grupo': 1, 'titulo': titulo, 'texto': cuerpo}]
            et = {1: {'tono': inicial, 'sub_tema': 'x'}}
            A.aplicar_guarda_positiva(grupos, et, brand, aliases)
            if et[1]['tono'] != esperado:
                fallos.append("%s -> %s (esperado %s)\n      %s"
                              % (inicial, et[1]['tono'], esperado, titulo[:78]))
        self.assertEqual(fallos, [], "\n  " + "\n  ".join(fallos))

    def test_todo_tono_resultante_es_valido(self):
        for titulo, cuerpo, brand, aliases, inicial, _ in CASOS:
            grupos = [{'grupo': 1, 'titulo': titulo, 'texto': cuerpo}]
            et = {1: {'tono': inicial, 'sub_tema': 'x'}}
            A.aplicar_guarda_positiva(grupos, et, brand, aliases)
            self.assertIn(et[1]['tono'], A.TONOS)

    def test_los_ejemplos_del_catalogo_no_se_mueven(self):
        """Sobre-disparo: ningun ejemplo ya etiquetado del catalogo debe cambiar de tono."""
        from catalogo_tono_tema import EJEMPLOS, EJEMPLOS_SECTOR, EJEMPLOS_TEMA
        pares = [(EJEMPLOS, SUCRE, ['Sucre']), (EJEMPLOS_TEMA, 'La entidad', ['la entidad']),
                 (EJEMPLOS_SECTOR, FENAVI, ['el gremio', 'Fenavi'])]
        movidos = []
        for ejemplos, brand, aliases in pares:
            for e in ejemplos:
                grupos = [{'grupo': 1, 'titulo': e['titulo'], 'texto': e['titulo']}]
                et = {1: {'tono': e['tono'], 'sub_tema': e['sub_tema']}}
                A.aplicar_guarda_positiva(grupos, et, brand, aliases)
                if et[1]['tono'] != e['tono']:
                    movidos.append("%s -> %s | %s" % (e['tono'], et[1]['tono'], e['titulo'][:70]))
        self.assertEqual(movidos, [], "la guarda movio ejemplos ya etiquetados:\n  " + "\n  ".join(movidos))

    def test_caso_reportado_por_el_cliente(self):
        texto = ("El programa de Medicina de la Universidad Simón Bolívar realizará los días 23, 24 y 25 "
                 "de septiembre la I Conferencia Internacional Caribe de Educación Médica")
        grupos = [{'grupo': 7, 'titulo': texto, 'texto': ''}]
        et = {7: {'tono': 'Neutro', 'sub_tema': 'Conferencia internacional de educación médica'}}
        subidos = A.aplicar_guarda_positiva(grupos, et, USB, [])
        self.assertEqual(et[7]['tono'], 'Positivo')
        self.assertEqual(subidos, [7])


if __name__ == '__main__':
    unittest.main()
