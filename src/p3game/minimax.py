"""
Camada 4 -- Solucionador Minimax Puro (Agnóstico ao Jogo).

Implementação essencial do algoritmo Minimax:
- Sem poda Alfa-Beta (avalia todos os ramos via max/min exaustivo).
- Alice = MAX (+1), Bob = MIN (-1).
- Convenção Normal: quem não possui jogadas válidas no seu turno perde.
  Portanto, um estado terminal tem valor -1 para Alice (MAX) e +1 para Bob (MIN).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, asdict

from .jogo import JogoImparcialNormal

MAX, MIN = +1, -1


@dataclass
class Estatisticas:
    nos_visitados: int = 0
    nos_terminais: int = 0
    acertos_tabela: int = 0
    entradas_tabela: int = 0
    tempo_s: float = 0.0

    def como_dict(self) -> dict:
        return asdict(self)


class Minimax:
    """
    Motor Minimax essencial para jogos combinatórios imparciais sob convenção normal.
    Opera exclusivamente sobre a interface JogoImparcialNormal.
    """

    def __init__(
        self,
        jogo: JogoImparcialNormal,
        memoizacao: bool = True,
        **kwargs,  # Aceita e ignora parâmetros de podas antigas para compatibilidade
    ):
        self.jogo = jogo
        self.memoizacao = memoizacao
        self.tabela: dict = {}
        self.stats = Estatisticas()
        sys.setrecursionlimit(max(10_000, sys.getrecursionlimit()))

    def valor(self, estado, jogador: int) -> int:
        """
        Calcula o valor Minimax PURO (+1: vitória de Alice, -1: vitória de Bob).
        Não utiliza poda Alfa-Beta: avalia exaustivamente todos os sucessores.
        """
        st = self.stats
        st.nos_visitados += 1

        if self.memoizacao:
            r = self.tabela.get(estado)
            if r is not None:
                st.acertos_tabela += 1
                return r * jogador

        sucessores = self.jogo.jogadas(estado)
        if not sucessores:
            st.nos_terminais += 1
            v = -jogador  # O jogador da vez não tem lances e perde
        elif jogador == MAX:
            # Alice escolhe a jogada que maximiza o valor (sem corte alfa-beta)
            v = max(self.valor(filho, MIN) for _, filho in sucessores)
        else:
            # Bob escolhe a jogada que minimiza o valor (sem corte alfa-beta)
            v = min(self.valor(filho, MAX) for _, filho in sucessores)

        if self.memoizacao:
            self.tabela[estado] = v * jogador
            st.entradas_tabela = len(self.tabela)

        return v

    def resolver(self) -> int:
        """Determina o valor minimax da posição inicial com Alice na vez."""
        t0 = time.perf_counter()
        try:
            return self.valor(self.jogo.estado_inicial(), MAX)
        finally:
            self.stats.tempo_s += time.perf_counter() - t0

    def jogadas_vencedoras(self, estado=None, jogador: int = MAX) -> list[int]:
        """Retorna todas as jogadas legais que garantem a vitória de `jogador`."""
        if estado is None:
            estado = self.jogo.estado_inicial()
        outro = -jogador
        return [
            j for j, filho in self.jogo.jogadas(estado)
            if self.valor(filho, outro) == jogador
        ]

    def variacao_principal(self) -> list[dict]:
        """Reconstrói uma partida sob jogo ótimo até a infecção total do grafo."""
        estado = self.jogo.estado_inicial()
        jogador = MAX
        partida = []
        while True:
            jogadas = self.jogo.jogadas(estado)
            if not jogadas:
                return partida
            # Jogador vencedor joga lance vencedor; o perdedor joga o primeiro lance legal
            escolhida = next(
                ((j, f) for j, f in jogadas if self.valor(f, -jogador) == jogador),
                jogadas[0],
            )
            j, filho = escolhida
            _, propagados = self.jogo.sucessor(estado, j) if hasattr(self.jogo, "sucessor") else (None, [])
            partida.append({
                "turno": len(partida) + 1,
                "jogador": "Alice" if jogador == MAX else "Bob",
                "jogada": self.jogo.rotulo(j),
                "propagados": [self.jogo.rotulo(p) for p in propagados],
            })
            estado, jogador = filho, -jogador
