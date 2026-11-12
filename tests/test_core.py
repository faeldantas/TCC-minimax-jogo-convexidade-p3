"""
Testes Unitários Essenciais do CORE.
Valida o Grafo Caterpillar Uniforme, o Fecho Convexo P_3 e o Minimax Puro.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from p3game import Caterpillar, JogoFechoConvexoP3, Minimax, MAX, MIN, fecho_p3


class TestCore(unittest.TestCase):
    def test_caterpillar_uniforme_estrutura(self):
        # m = 2 nós internos com 2 pernas cada -> L = (0, 2, 2, 0)
        cat = Caterpillar((0, 2, 2, 0))
        self.assertEqual(cat.k, 4)
        self.assertEqual(cat.n, 8)
        self.assertEqual(len(cat.adj), 8)

    def test_fecho_p3_propagacao(self):
        # Em L = (0, 2, 0): v2 é centro (adjacente a v1, v3, v4, v5)
        cat = Caterpillar((0, 2, 0))
        # Se v1 e v3 forem infectados (máscara bits 0 e 2: 1 | 4 = 5), v2 tem 2 vizinhos infectados e propaga!
        S = (1 << 0) | (1 << 2)
        novo_S, props = fecho_p3(cat.adj, S)
        self.assertIn(1, props)  # v2 (índice 1) propagou
        self.assertTrue((novo_S >> 1) & 1)

    def test_minimax_uniforme_resultados_teoricos(self):
        # m = 1 -> n = 5 (Alice vence)
        cat1 = Caterpillar((0, 2, 0))
        s1 = Minimax(JogoFechoConvexoP3(cat1.adj, cat1.rotulos))
        self.assertEqual(s1.resolver(), MAX)

        # m = 2 -> n = 8 (Alice vence)
        cat2 = Caterpillar((0, 2, 2, 0))
        s2 = Minimax(JogoFechoConvexoP3(cat2.adj, cat2.rotulos))
        self.assertEqual(s2.resolver(), MAX)

        # m = 3 -> n = 11 (Alice vence)
        cat3 = Caterpillar((0, 2, 2, 2, 0))
        s3 = Minimax(JogoFechoConvexoP3(cat3.adj, cat3.rotulos))
        self.assertEqual(s3.resolver(), MAX)

        # m = 4 -> n = 14 (Bob vence)
        cat4 = Caterpillar((0, 2, 2, 2, 2, 0))
        s4 = Minimax(JogoFechoConvexoP3(cat4.adj, cat4.rotulos))
        self.assertEqual(s4.resolver(), MIN)

        # Paridade: turnos + propagados = n
        pv4 = s4.variacao_principal()
        props4 = sum(len(p["propagados"]) for p in pv4)
        self.assertEqual(len(pv4) + props4, cat4.n)


if __name__ == "__main__":
    unittest.main()
