"""
Camada 3 -- Regras do Closed Hull Game (CHG) na convexidade P_3 (CORE).
Define o estado inicial, as transições válidas por jogada e os rótulos dos vértices.
"""

from __future__ import annotations
import numpy as np
from .convexidade import fecho_p3  # Importa o operador de fecho convexo P3


class JogoFechoConvexoP3:
    """
    Closed Hull Game sob convenção normal.
    - Estado inicial: conjunto vazio S = 0 (nenhum vértice infectado).
    - Jogada legal: escolher qualquer vértice v não infectado (bit v == 0).
    - Transição: novo estado é o fecho convexo [S ∪ {v}]_P3.
    - Vitória: quem faz a última jogada válida (completando o grafo) vence.
    """

    def __init__(self, adj: list[list[int]], rotulos: list[str]):
        """Inicializa o jogo com a lista de adjacência do grafo e suas estruturas NumPy (CSR + bitmask)."""
        self.adj = adj          # Lista de adjacência: adj[i] contém os vizinhos do vértice i
        self.rotulos = rotulos  # Rótulos legíveis dos vértices (ex: "v1", "v2", ...)
        self.n = len(adj)       # Número total de vértices no grafo

        # Pré-calcula a representação CSR (Compressed Sparse Row) + máscaras de adjacência
        # para consumo direto em código de máquina pelo Numba (suporta até n <= 63 vértices).
        if self.n <= 63:
            offsets = np.zeros(self.n + 1, dtype=np.int32)
            flat: list[int] = []
            adj_mask = np.zeros(self.n, dtype=np.uint64)
            for i, viz in enumerate(adj):
                offsets[i + 1] = offsets[i] + len(viz)
                mascara = 0
                for w in viz:
                    flat.append(w)
                    mascara |= (1 << w)
                adj_mask[i] = np.uint64(mascara)
            self.offsets = offsets
            self.neighbors = np.array(flat, dtype=np.int32)
            self.adj_mask = adj_mask
        else:
            self.offsets = None
            self.neighbors = None
            self.adj_mask = None

    def estado_inicial(self) -> int:
        """Retorna o estado inicial do jogo: máscara 0 (conjunto vazio)."""
        return 0

    def jogadas(self, S: int) -> list[tuple[int, int]]:
        """
        Retorna a lista de todas as jogadas legais a partir da máscara de estado S.
        Cada jogada é representada por um par: (vértice_escolhido, máscara_do_estado_sucessor).
        """
        res = []  # Lista que acumulará os pares (v, prox_S)
        for v in range(self.n):  # Percorre todos os vértices de 0 a n-1
            if not (S >> v) & 1:  # Verifica se o vértice v NÃO está infectado (bit v é 0)
                # Calcula o novo estado fechado após infectar v: [S ∪ {v}]_P3
                prox_S, _ = fecho_p3(self.adj, S | (1 << v))
                res.append((v, prox_S))  # Adiciona a jogada e o estado sucessor resultante
        return res

    def rotulo(self, v: int) -> str:
        """Retorna o rótulo amigável (ex: 'v1', 'v2') a partir do índice numérico do vértice."""
        return self.rotulos[v]
