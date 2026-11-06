"""
Camada 4 -- Algoritmo Minimax Puro sem poda Alfa-Beta (CORE).
"""

from __future__ import annotations
import sys
import time

MAX, MIN = +1, -1


class Minimax:
    """
    Minimax Puro para jogos combinatórios imparciais sob convenção normal.
    - Alice = MAX (+1)
    - Bob = MIN (-1)
    - Sem corte alfa-beta: avalia todos os ramos exaustivamente via max/min.
    """

    def __init__(self, jogo, memoizacao: bool = True):
        self.jogo = jogo
        self.memoizacao = memoizacao
        self.tabela: dict[int, int] = {}
        self.nos_visitados = 0
        self.tempo_s = 0.0
        sys.setrecursionlimit(max(10_000, sys.getrecursionlimit()))

    def valor(self, S: int, jogador: int) -> int:
        self.nos_visitados += 1

        if self.memoizacao and S in self.tabela:
            return self.tabela[S] * jogador

        sucessores = self.jogo.jogadas(S)
        if not sucessores:
            v = -jogador  # Sem lances legais: quem deveria jogar perde
        elif jogador == MAX:
            v = max(self.valor(filho, MIN) for _, filho in sucessores)
        else:
            v = min(self.valor(filho, MAX) for _, filho in sucessores)

        if self.memoizacao:
            self.tabela[S] = v * jogador

        return v

    def resolver(self) -> int:
        t0 = time.perf_counter()
        try:
            return self.valor(self.jogo.estado_inicial(), MAX)
        finally:
            self.tempo_s = time.perf_counter() - t0

    def variacao_principal(self) -> list[dict]:
        """Reconstrói a partida passo a passo sob jogo ótimo."""
        from .convexidade import fecho_p3
        S = self.jogo.estado_inicial()
        jogador = MAX
        partida = []

        while True:
            sucessores = self.jogo.jogadas(S)
            if not sucessores:
                return partida

            # Seleciona o lance ótimo
            escolhida = next(
                ((v, prox) for v, prox in sucessores if self.valor(prox, -jogador) == jogador),
                sucessores[0],
            )
            v, prox_S = escolhida
            _, props = fecho_p3(self.jogo.adj, S | (1 << v))

            partida.append({
                "turno": len(partida) + 1,
                "jogador": "Alice" if jogador == MAX else "Bob",
                "jogada": self.jogo.rotulo(v),
                "propagados": [self.jogo.rotulo(p) for p in props],
            })
            S = prox_S
            jogador = -jogador
