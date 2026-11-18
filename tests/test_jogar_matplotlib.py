"""
Testes unitários para o módulo jogar_matplotlib.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import matplotlib
matplotlib.use("Agg")  # Backend não-interativo para testes automatizados

from p3game import Caterpillar
from p3game.jogar_matplotlib import calcular_layout


class TestJogarMatplotlib(unittest.TestCase):
    def test_calcular_layout_caterpillar(self):
        # L = (0, 2, 0) -> k=3 (espinha 0, 1, 2) e pernas 3, 4 no nó 1
        cat = Caterpillar((0, 2, 0))
        pos = calcular_layout(cat)

        self.assertEqual(len(pos), 5)
        # Espinha no eixo y = 0
        self.assertEqual(pos[0][1], 0.0)
        self.assertEqual(pos[1][1], 0.0)
        self.assertEqual(pos[2][1], 0.0)

        # Duas pernas no nó 1: uma acima (y > 0) e uma abaixo (y < 0)
        self.assertEqual(pos[3][1], 1.2)
        self.assertEqual(pos[4][1], -1.2)
        self.assertEqual(pos[3][0], pos[1][0])
        self.assertEqual(pos[4][0], pos[1][0])


if __name__ == "__main__":
    unittest.main()
