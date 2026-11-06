"""
Camada 4 -- Solucionador Minimax genérico (agnóstico ao jogo).

Convenções: Alice = MAX (+1), Bob = MIN (-1). A utilidade de um estado
terminal (sem jogadas) é -1 se for a vez de MAX (Bob fez a última jogada) e
+1 se for a vez de MIN. O valor minimax de um estado é exatamente +1 ou -1.

Otimizações, todas opcionais para permitir o estudo de ablação:
* ``corte``      : interrompe a varredura dos filhos de um nó MAX (MIN) ao
                   encontrar um filho de valor +1 (-1). Como os valores
                   possíveis são apenas {-1, +1}, isso equivale à poda
                   alfa-beta com janela inicial [-1, +1].
* ``memoizacao`` : tabela de transposição. Como o jogo é imparcial, o
                   resultado depende apenas do estado, não de quem joga;
                   armazena-se v * jogador (o valor do ponto de vista de quem
                   está na vez), sob a chave do estado.
* ``simetria``   : explora um único sucessor por classe de equivalência
                   (``jogadas_reduzidas``) e usa ``chave_canonica`` na tabela.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, asdict

from .jogo import JogoImparcialNormal

MAX, MIN = +1, -1


class LimiteDeNosExcedido(RuntimeError):
    pass


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
    def __init__(
        self,
        jogo: JogoImparcialNormal,
        corte: bool = True,
        memoizacao: bool = True,
        simetria: bool = True,
        limite_nos: int | None = None,
    ):
        self.jogo = jogo
        self.corte = corte
        self.memoizacao = memoizacao
        self.simetria = simetria
        self.limite_nos = limite_nos
        self.tabela: dict = {}
        self.stats = Estatisticas()
        sys.setrecursionlimit(max(10_000, sys.getrecursionlimit()))

    # ------------------------------------------------------------------
    def _sucessores(self, estado):
        return self.jogo.jogadas_reduzidas(estado) if self.simetria else self.jogo.jogadas(estado)

    def _chave(self, estado):
        return self.jogo.chave_canonica(estado) if self.simetria else estado

    def valor(self, estado, jogador: int) -> int:
        """Valor minimax (+1: Alice vence; -1: Bob vence) com `jogador` na vez."""
        st = self.stats
        st.nos_visitados += 1
        if self.limite_nos is not None and st.nos_visitados > self.limite_nos:
            raise LimiteDeNosExcedido(self.limite_nos)

        if self.memoizacao:
            chave = self._chave(estado)
            r = self.tabela.get(chave)
            if r is not None:
                st.acertos_tabela += 1
                return r * jogador

        sucessores = self._sucessores(estado)
        if not sucessores:
            st.nos_terminais += 1
            v = -jogador
        elif jogador == MAX:
            v = -1
            for _, filho in sucessores:
                if self.valor(filho, MIN) == +1:
                    v = +1
                    if self.corte:
                        break
        else:
            v = +1
            for _, filho in sucessores:
                if self.valor(filho, MAX) == -1:
                    v = -1
                    if self.corte:
                        break

        if self.memoizacao:
            self.tabela[chave] = v * jogador
            st.entradas_tabela = len(self.tabela)
        return v

    # ------------------------------------------------------------------
    def resolver(self) -> int:
        """Valor da posição inicial com Alice na vez."""
        t0 = time.perf_counter()
        try:
            return self.valor(self.jogo.estado_inicial(), MAX)
        finally:
            self.stats.tempo_s += time.perf_counter() - t0

    def jogadas_vencedoras(self, estado=None, jogador: int = MAX) -> list[int]:
        """Todas as jogadas legais (sem redução) que preservam a vitória de `jogador`."""
        if estado is None:
            estado = self.jogo.estado_inicial()
        outro = -jogador
        return [j for j, filho in self.jogo.jogadas(estado) if self.valor(filho, outro) == jogador]

    def variacao_principal(self) -> list[dict]:
        """
        Uma partida sob jogo ótimo, escolhida deterministicamente: o jogador
        vencedor joga a primeira jogada vencedora (menor índice); o perdedor
        joga a primeira jogada legal. Não é a única partida ótima possível.
        """
        estado = self.jogo.estado_inicial()
        jogador = MAX
        partida = []
        while True:
            jogadas = self.jogo.jogadas(estado)
            if not jogadas:
                return partida
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
