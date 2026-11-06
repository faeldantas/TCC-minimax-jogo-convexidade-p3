"""Testes da Camada 1: enumeração canônica de caterpillars."""

import unittest

import networkx as nx

import _caminho  # noqa: F401
from p3game import (
    Caterpillar, enumerar_caterpillars, numero_caterpillars, eh_caterpillar,
    assinatura_de_grafo, eh_assinatura_canonica,
)

N_MAX = 12


class TestEnumeracao(unittest.TestCase):
    def test_contagem_harary_schwenk(self):
        """A enumeração produz exatamente c(n) = 2^(n-4) + 2^floor((n-4)/2) grafos."""
        for n in range(3, N_MAX + 1):
            self.assertEqual(len(enumerar_caterpillars(n)), numero_caterpillars(n), n)

    def test_valores_conhecidos(self):
        esperado = {3: 1, 4: 2, 5: 3, 6: 6, 7: 10, 8: 20, 9: 36, 10: 72, 11: 136, 12: 272}
        for n, c in esperado.items():
            self.assertEqual(numero_caterpillars(n), c)

    def test_completude_e_unicidade_contra_networkx(self):
        """
        Validação independente: entre TODAS as árvores não-isomorfas de ordem n
        (nx.nonisomorphic_trees), as que são caterpillars correspondem
        bijetivamente às assinaturas enumeradas.
        """
        for n in range(3, N_MAX + 1):
            gerados = set(enumerar_caterpillars(n))
            referencia = [assinatura_de_grafo(T) for T in nx.nonisomorphic_trees(n) if eh_caterpillar(T)]
            self.assertEqual(len(referencia), len(set(referencia)), f"duplicata na referência n={n}")
            self.assertEqual(set(referencia), gerados, n)

    def test_grafos_gerados_sao_caterpillars_dois_a_dois_nao_isomorfos(self):
        for n in range(3, 10):
            grafos = [Caterpillar(L).para_networkx() for L in enumerar_caterpillars(n)]
            for G in grafos:
                self.assertTrue(eh_caterpillar(G))
                self.assertEqual(G.number_of_nodes(), n)
            for i in range(len(grafos)):
                for j in range(i + 1, len(grafos)):
                    self.assertFalse(nx.is_isomorphic(grafos[i], grafos[j]))

    def test_assinatura_ida_e_volta(self):
        for n in range(3, N_MAX + 1):
            for L in enumerar_caterpillars(n):
                self.assertTrue(eh_assinatura_canonica(L))
                self.assertEqual(assinatura_de_grafo(Caterpillar(L).para_networkx()), L)

    def test_rotulacao_do_grafo_do_tcc(self):
        c = Caterpillar((0, 2, 0, 1, 1, 0))
        self.assertEqual(c.n, 10)
        nomes = lambda vs: sorted(c.rotulos[u] for u in vs)
        self.assertEqual(nomes(c.adj[1]), ["v1", "v3", "v7", "v8"])
        self.assertEqual(nomes(c.adj[3]), ["v3", "v5", "v9"])
        self.assertEqual(nomes(c.adj[4]), ["v10", "v4", "v6"])
        self.assertEqual(sorted(c.rotulos[v] for v in c.folhas), ["v1", "v10", "v6", "v7", "v8", "v9"])


if __name__ == "__main__":
    unittest.main()
