"""
Camada 3 -- Regras do Closed Hull Game (CHG) na convexidade P_3 (CORE).
"""

from __future__ import annotations
from .convexidade import fecho_p3


class JogoFechoConvexoP3:
    """
    Closed Hull Game sob convenção normal.
    - Estado inicial: conjunto vazio S = 0.
    - Jogada legal: escolher qualquer vértice v não infectado.
    - Transição: novo estado é o fecho [S ∪ {v}]_P3.
    - Vitória: quem faz a última jogada válida (completando o grafo) vence.
    """

    def __init__(self, adj: list[list[int]], rotulos: list[str]):
        self.adj = adj
        self.rotulos = rotulos
        self.n = len(adj)

    def estado_inicial(self) -> int:
        return 0

    def jogadas(self, S: int) -> list[tuple[int, int]]:
        """Retorna lista de pares (vértice escolhido, máscara do estado sucessor)."""
        res = []
        for v in range(self.n):
            if not (S >> v) & 1:
                prox_S, _ = fecho_p3(self.adj, S | (1 << v))
                res.append((v, prox_S))
        return res

    def rotulo(self, v: int) -> str:
        return self.rotulos[v]
