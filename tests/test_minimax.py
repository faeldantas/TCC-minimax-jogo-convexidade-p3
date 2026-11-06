"""
Testes das Camadas 3 e 4: o solucionador é validado contra um ORÁCULO
independente (busca exaustiva ingênua em conjuntos de rótulos, sem bitmask,
sem memoização, sem simetria e sem reutilizar nenhum código do pacote).
"""

import unittest
from functools import lru_cache

import networkx as nx

import _caminho  # noqa: F401
from p3game import (
    Caterpillar, enumerar_caterpillars, JogoFechoConvexoP3, JogoFechoConvexoP3Caterpillar,
    Minimax, MAX, resolver_caterpillar,
)


# ---------------------------------------------------------------------------
# Oráculo independente
# ---------------------------------------------------------------------------

def _grafo_independente(L):
    """Constrói o caterpillar diretamente no NetworkX, com rótulos 'v1'...'vn'."""
    G = nx.Graph()
    k = len(L)
    nx.add_path(G, [f"v{i}" for i in range(1, k + 1)])
    prox = k + 1
    for i, q in enumerate(L, start=1):
        for _ in range(q):
            G.add_edge(f"v{i}", f"v{prox}")
            prox += 1
    return G


def _fecho_ingenuo(G, S):
    S = set(S)
    mudou = True
    while mudou:
        mudou = False
        for v in G:
            if v not in S and sum(1 for u in G[v] if u in S) >= 2:
                S.add(v)
                mudou = True
    return frozenset(S)


def oraculo(L):
    """Retorna (Alice vence?, conjunto de primeiras jogadas vencedoras)."""
    G = _grafo_independente(L)
    V = frozenset(G)

    @lru_cache(maxsize=None)
    def quem_joga_vence(S):
        return any(not quem_joga_vence(_fecho_ingenuo(G, S | {v})) for v in V - S)

    vencedoras = {v for v in V if not quem_joga_vence(_fecho_ingenuo(G, {v}))}
    return quem_joga_vence(frozenset()), vencedoras


CONFIGURACOES = [
    dict(corte=False, memoizacao=False, simetria=False),
    dict(corte=True, memoizacao=False, simetria=False),
    dict(corte=True, memoizacao=False, simetria=True),
    dict(corte=True, memoizacao=True, simetria=False),
    dict(corte=True, memoizacao=True, simetria=True),
]


class TestMinimax(unittest.TestCase):
    def test_concordancia_com_oraculo(self):
        """Valor e jogadas iniciais vencedoras coincidem com o oráculo, n <= 9."""
        for n in range(3, 10):
            for L in enumerate_safe(n):
                alice, venc = oraculo(L)
                v, solver = resolver_caterpillar(L)
                self.assertEqual(v == +1, alice, L)
                rot = {solver.jogo.rotulo(j) for j in solver.jogadas_vencedoras()}
                self.assertEqual(rot, venc, L)

    def test_todas_as_configuracoes_concordam(self):
        """As cinco configurações do estudo de ablação produzem o mesmo valor (n <= 8)."""
        for n in range(3, 9):
            for L in enumerar_caterpillars(n):
                valores = {resolver_caterpillar(L, **cfg)[0] for cfg in CONFIGURACOES}
                self.assertEqual(len(valores), 1, L)

    def test_jogo_generico_sem_simetria_em_grafo_arbitrario(self):
        """O solucionador é agnóstico: funciona com o CHG em um ciclo C_5."""
        adj = [[(i - 1) % 5, (i + 1) % 5] for i in range(5)]
        a = Minimax(JogoFechoConvexoP3(adj), simetria=False).resolver()
        b = Minimax(JogoFechoConvexoP3(adj), corte=False, memoizacao=False, simetria=False).resolver()
        self.assertEqual(a, b)

    def test_p3_alice_vence_apenas_pelo_centro(self):
        v, solver = resolver_caterpillar((0, 0, 0))
        self.assertEqual(v, +1)
        self.assertEqual([solver.jogo.rotulo(j) for j in solver.jogadas_vencedoras()], ["v2"])

    def test_variacao_principal_e_valida(self):
        """Paridade: a partida ótima tem T turnos e T + |propagados| = n."""
        for L in [(0, 2, 0, 1, 1, 0), (0, 0, 0, 0, 0), (0, 3, 0)]:
            v, solver = resolver_caterpillar(L)
            pv = solver.variacao_principal()
            T = len(pv)
            P = sum(len(p["propagados"]) for p in pv)
            self.assertEqual(T + P, Caterpillar(L).n)
            self.assertEqual(pv[-1]["jogador"], "Alice" if v == +1 else "Bob")


def enumerate_safe(n):
    return enumerar_caterpillars(n)


if __name__ == "__main__":
    unittest.main()
