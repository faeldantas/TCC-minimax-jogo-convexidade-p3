"""Testes da Camada 2: operadores da convexidade P_3 e propriedades provadas no TCC."""

import random
import unittest

import _caminho  # noqa: F401
from p3game import (
    Caterpillar, enumerar_caterpillars, fecho_p3, fecho_p3_iterativo,
    eh_p3_convexo, para_mascara,
)


class TestFecho(unittest.TestCase):
    def test_exemplo_figura_2(self):
        """S = {v2, v6, v9, v10} propaga v5, v4, v3 nessa ordem (Figura 2)."""
        c = Caterpillar((0, 2, 0, 1, 1, 0))
        idx = {r: i for i, r in enumerate(c.rotulos)}
        S = para_mascara(idx[r] for r in ["v2", "v6", "v9", "v10"])
        F, prop = fecho_p3(c.adj, S)
        self.assertEqual([c.rotulos[v] for v in prop], ["v5", "v4", "v3"])
        self.assertEqual(sorted(c.rotulos[v] for v in range(c.n) if (F >> v) & 1),
                         sorted(["v2", "v3", "v4", "v5", "v6", "v9", "v10"]))

    def test_bfs_coincide_com_ponto_fixo(self):
        rng = random.Random(2026)
        for n in range(3, 11):
            for L in enumerar_caterpillars(n):
                c = Caterpillar(L)
                for _ in range(30):
                    S = rng.getrandbits(c.n)
                    F, _ = fecho_p3(c.adj, S)
                    self.assertEqual(F, fecho_p3_iterativo(c.adj, S))
                    self.assertTrue(eh_p3_convexo(c.adj, F))
                    self.assertEqual(F & S, S)

    def test_lema_folhas_nunca_propagam(self):
        """Lema: um vértice de grau 1 fora de S nunca pertence a [S]."""
        rng = random.Random(7)
        for n in range(3, 11):
            for L in enumerar_caterpillars(n):
                c = Caterpillar(L)
                for _ in range(30):
                    S = rng.getrandbits(c.n)
                    F, _ = fecho_p3(c.adj, S)
                    for f in c.folhas:
                        if not (S >> f) & 1:
                            self.assertFalse((F >> f) & 1)

    def test_lema_perna_com_pai_infectado_nao_propaga(self):
        """Lema: se S é convexo, v ∈ S e u é folha de v, então [S ∪ {u}] = S ∪ {u}."""
        rng = random.Random(11)
        for n in range(3, 11):
            for L in enumerar_caterpillars(n):
                c = Caterpillar(L)
                for _ in range(30):
                    S, _ = fecho_p3(c.adj, rng.getrandbits(c.n))
                    for u in c.folhas:
                        pai = c.adj[u][0]
                        if (S >> pai) & 1 and not (S >> u) & 1:
                            F, prop = fecho_p3(c.adj, S | (1 << u))
                            self.assertEqual(F, S | (1 << u))
                            self.assertEqual(prop, [])


if __name__ == "__main__":
    unittest.main()
