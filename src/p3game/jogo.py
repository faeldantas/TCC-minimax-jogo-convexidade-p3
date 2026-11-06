"""
Camada 3 -- Modelagem do jogo como objeto abstrato.

O solucionador Minimax (``minimax.py``) conhece apenas a interface
``JogoImparcialNormal``; ele não sabe o que é um grafo, uma convexidade ou um
caterpillar. Qualquer jogo imparcial finito sob convenção normal (o último a
jogar vence) pode ser resolvido implementando os três métodos abstratos.

Implementações concretas:
* ``JogoFechoConvexoP3``             -- Closed Hull Game (CHG) em grafo qualquer;
* ``JogoFechoConvexoP3Caterpillar``  -- mesmo jogo, acrescido da chave canônica
                                         por automorfismos do caterpillar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Hashable

from .convexidade import fecho_p3
from .grafos import Caterpillar


class JogoImparcialNormal(ABC):
    """Interface mínima de um jogo combinatório imparcial sob convenção normal."""

    @abstractmethod
    def estado_inicial(self) -> Hashable: ...

    @abstractmethod
    def jogadas(self, estado) -> list[tuple[int, Hashable]]:
        """Todas as jogadas legais: lista de pares (jogada, estado sucessor)."""

    def jogadas_reduzidas(self, estado) -> list[tuple[int, Hashable]]:
        """
        Um representante por classe de sucessores equivalentes por simetria.
        Padrão: nenhuma redução.
        """
        return self.jogadas(estado)

    def chave_canonica(self, estado) -> Hashable:
        """
        Invariante de estado sob os automorfismos do jogo. Estados com a mesma
        chave devem ter o mesmo valor. Padrão: o próprio estado.
        """
        return estado

    def rotulo(self, jogada: int) -> str:
        return str(jogada)


class JogoFechoConvexoP3(JogoImparcialNormal):
    """
    Closed Hull Game (CHG) na convexidade P_3, convenção normal.

    Estado: máscara do conjunto P_3-convexo S (inicialmente vazio).
    Jogada: escolher v ∉ S; o sucessor é [S ∪ {v}]_{P_3}.
    Terminal: S = V(G) (não há jogadas); quem deveria jogar perde.
    """

    def __init__(self, adj: list[list[int]], rotulos: list[str] | None = None):
        self.adj = adj
        self.n = len(adj)
        self.completo = (1 << self.n) - 1
        self.rotulos = rotulos or [f"v{i + 1}" for i in range(self.n)]

    def estado_inicial(self) -> int:
        return 0

    def sucessor(self, S: int, v: int) -> tuple[int, list[int]]:
        return fecho_p3(self.adj, S | (1 << v))

    def jogadas(self, S: int) -> list[tuple[int, int]]:
        return [(v, self.sucessor(S, v)[0]) for v in range(self.n) if not (S >> v) & 1]

    def rotulo(self, jogada: int) -> str:
        return self.rotulos[jogada]


class JogoFechoConvexoP3Caterpillar(JogoFechoConvexoP3):
    """
    CHG em caterpillar com redução por simetria.

    Os automorfismos de um caterpillar (n >= 3) são gerados por (i) permutações
    das folhas adjacentes a um mesmo vértice interno e (ii) a reflexão do
    caminho interno, quando a sequência (f_1, ..., f_m) de folhas por vértice
    interno é palíndroma. Assim, a chave
        ( (pertence(u_i), folhas_infectadas(u_i)) para i = 1..m )
    -- tomada como a menor entre ela e sua reversa, quando há reflexão -- é
    constante em cada órbita e distingue órbitas diferentes.
    Note que os extremos da espinha diametral (v_1 e v_k) também são folhas e
    são tratados como tais (são equivalentes às pernas do mesmo vértice).
    """

    def __init__(self, cat: Caterpillar):
        super().__init__(cat.adj, cat.rotulos)
        self.cat = cat
        self.internos = cat.internos
        grau = [len(a) for a in cat.adj]
        self.folhas_de = [sorted(u for u in cat.adj[v] if grau[u] == 1) for v in self.internos]
        f = [len(x) for x in self.folhas_de]
        self.reflexao = len(self.internos) >= 2 and f == f[::-1]

    def chave_canonica(self, S: int):
        chave = tuple(
            ((S >> v) & 1, sum((S >> u) & 1 for u in folhas))
            for v, folhas in zip(self.internos, self.folhas_de)
        )
        if self.reflexao:
            chave = min(chave, chave[::-1])
        return chave

    def jogadas_reduzidas(self, S: int) -> list[tuple[int, int]]:
        candidatos = []
        for v, folhas in zip(self.internos, self.folhas_de):
            if not (S >> v) & 1:
                candidatos.append(v)
            livre = next((u for u in folhas if not (S >> u) & 1), None)
            if livre is not None:
                candidatos.append(livre)
        vistos = set()
        resultado = []
        for v in sorted(candidatos):
            T = self.sucessor(S, v)[0]
            c = self.chave_canonica(T)
            if c not in vistos:
                vistos.add(c)
                resultado.append((v, T))
        return resultado
