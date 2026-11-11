"""
Camada 4 -- Algoritmo Minimax com Poda Alfa-Beta (CORE).
"""

from __future__ import annotations
import sys
import time

MAX, MIN = +1, -1


class Minimax:
    """
    Minimax com Poda Alfa-Beta para jogos combinatórios imparciais sob convenção normal.
    - Alice = MAX (+1)
    - Bob = MIN (-1)
    - Com corte alfa-beta: interrompe a busca no nó assim que alfa >= beta.
    """

    def __init__(self, jogo, memoizacao: bool = True):
        self.jogo = jogo
        self.memoizacao = memoizacao
        self.tabela: dict[int, int] = {} # Tabela de memoização para armazenar os valores dos estados já avaliados
        self.nos_visitados = 0
        self.tempo_s = 0.0
        sys.setrecursionlimit(max(10_000, sys.getrecursionlimit()))

    def valor(self, S: int, jogador: int, alfa: int = MIN, beta: int = MAX) -> int: # Retorna o valor do estado S para o jogador atual (MAX ou MIN) com poda Alfa-Beta
        self.nos_visitados += 1 # Incrementa o contador de nós visitados a cada chamada recursiva do método valor

        if self.memoizacao and S in self.tabela: # Se a memoização estiver ativada e o estado S já tiver sido avaliado, retorna o valor armazenado multiplicado pelo jogador atual
            return self.tabela[S] * jogador

        sucessores = self.jogo.jogadas(S)
        if not sucessores:
            v = -jogador  # Sem lances legais: quem deveria jogar perde
        elif jogador == MAX:
            v = MIN
            for _, filho in sucessores:
                v = max(v, self.valor(filho, MIN, alfa, beta))
                alfa = max(alfa, v)
                if alfa >= beta:
                    break  # Poda Beta: MAX já garantiu vitória (+1)
        else:
            v = MAX
            for _, filho in sucessores:
                v = min(v, self.valor(filho, MAX, alfa, beta))
                beta = min(beta, v)
                if beta <= alfa:
                    break  # Poda Alfa: MIN já garantiu vitória (-1)

        if self.memoizacao:
            self.tabela[S] = v * jogador

        return v

    def resolver(self) -> int:
        t0 = time.perf_counter() #Marca o tempo de início da resolução do jogo
        try:
            return self.valor(self.jogo.estado_inicial(), MAX, MIN, MAX) # Inicia a resolução do jogo a partir do estado inicial com janela [MIN, MAX]
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
