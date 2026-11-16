"""
Camada 4 -- Algoritmo Minimax com Poda Alfa-Beta e Aceleração JIT via Numba (CORE).
Implementa o solucionador exaustivo com poda alfa-beta por curto-circuito binário,
tabela de transposição por endereçamento aberto e compilação nativa LLVM (@njit).
"""

from __future__ import annotations
import sys   # Módulo do sistema para alterar o limite de profundidade de recursão
import time  # Módulo para cronometragem de alta resolução (perf_counter)
from .convexidade import TEM_NUMBA, fecho_p3_numba

# Constantes dos jogadores:
# MAX (+1) = Alice (jogadora inicial que tenta maximizar para +1)
# MIN (-1) = Bob   (segundo jogador que tenta minimizar para -1)
MAX, MIN = +1, -1

_NUMBA_AQUECIDO = False

if TEM_NUMBA:
    import numpy as np
    from numba import njit, int8, int32, int64, uint64

    _VAZIO_HT = np.uint64(0xFFFFFFFFFFFFFFFF)

    @njit(
        int64(
            uint64, int64, int64, int64, int64, int64, uint64,
            int32[:], int32[:], uint64[:], int32[:],
            uint64[:, :], int32[:],
            uint64[:], int8[:], uint64,
            int64[:], int64,
        )
    )
    def minimax_ab_numba(
        S: int,
        jogador: int,
        alfa: int,
        beta: int,
        prof: int,
        n: int,
        mascara_internos: int,
        offsets: np.ndarray,
        neighbors: np.ndarray,
        adj_mask: np.ndarray,
        pred_folha: np.ndarray,
        buf_suc: np.ndarray,
        fila: np.ndarray,
        ht_keys: np.ndarray,
        ht_vals: np.ndarray,
        ht_mask: int,
        stats: np.ndarray,
        usar_memo: int,
    ) -> int:
        """
        Núcleo recursivo do Minimax com Poda Alfa-Beta, Corte de Paridade e Quebra de Simetria
        de Folhas Gêmeas compilado via Numba.
        """
        stats[0] += 1  # Incrementa contador de nós visitados

        # 0. Corte de Paridade em O(1): quando todos os nós de grau >= 2 já estão infectados,
        # nenhuma propagação P_3 adicional ocorrerá; os nós restantes são apenas folhas (grau <= 1).
        if (S & mascara_internos) == mascara_internos:
            infectados = 0
            x = S
            while x:
                x &= x - uint64(1)  # Truque de Kernighan: apaga o bit 1 menos significativo por iteração
                infectados += 1
            restantes = n - infectados
            return jogador if (restantes & 1) else -jogador

        # 1. Consulta à Tabela de Transposição (endereçamento aberto com hash multiplicativo de Fibonacci)
        if usar_memo:
            idx = (S * uint64(0x9E3779B97F4A7C15)) & ht_mask
            probes = 0
            while probes < 64:
                k = ht_keys[idx]
                if k == S:
                    return int64(ht_vals[idx]) * jogador
                if k == uint64(0xFFFFFFFFFFFFFFFF):
                    break
                idx = (idx + uint64(1)) & ht_mask
                probes += 1

        # 2. Gera sucessores com quebra de simetria de folhas gêmeas + fecho P_3 incremental
        qtd = 0
        for v in range(n):
            if not ((S >> uint64(v)) & uint64(1)):
                pf = pred_folha[v]
                if pf != -1 and not ((S >> uint64(pf)) & uint64(1)):
                    continue  # Só libera a folha gêmea v quando sua irmã anterior pf já estiver infectada
                buf_suc[prof, qtd] = fecho_p3_numba(offsets, neighbors, adj_mask, S, v, fila)
                qtd += 1

        # 3. Avaliação Terminal ou Recursão Minimax com Poda Alfa-Beta
        if qtd == 0:
            val = -jogador  # Sem lances legais pela Convenção Normal: quem está na vez perde
        elif jogador == 1:
            val = -1
            for i in range(qtd):
                filho = buf_suc[prof, i]
                res = minimax_ab_numba(
                    filho, -1, alfa, beta, prof + 1, n, mascara_internos,
                    offsets, neighbors, adj_mask, pred_folha, buf_suc, fila,
                    ht_keys, ht_vals, ht_mask, stats, usar_memo,
                )
                if res > val:
                    val = res
                if val > alfa:
                    alfa = val
                if alfa >= beta:
                    break  # Poda Beta: MAX já garantiu vitória (+1)
        else:
            val = 1
            for i in range(qtd):
                filho = buf_suc[prof, i]
                res = minimax_ab_numba(
                    filho, 1, alfa, beta, prof + 1, n, mascara_internos,
                    offsets, neighbors, adj_mask, pred_folha, buf_suc, fila,
                    ht_keys, ht_vals, ht_mask, stats, usar_memo,
                )
                if res < val:
                    val = res
                if val < beta:
                    beta = val
                if beta <= alfa:
                    break  # Poda Alfa: MIN já garantiu vitória (-1)

        # 4. Armazenamento na Tabela de Transposição
        if usar_memo:
            idx = (S * uint64(0x9E3779B97F4A7C15)) & ht_mask
            probes = 0
            while probes < 64:
                k = ht_keys[idx]
                if k == S or k == uint64(0xFFFFFFFFFFFFFFFF):
                    ht_keys[idx] = S
                    ht_vals[idx] = int8(val * jogador)
                    break
                idx = (idx + uint64(1)) & ht_mask
                probes += 1

        return val

    def _aquecer_numba() -> None:
        """Executa uma micro-chamada inicial para garantir a compilação JIT fora da medição."""
        global _NUMBA_AQUECIDO
        if _NUMBA_AQUECIDO:
            return
        off = np.array([0, 0], dtype=np.int32)
        nei = np.zeros(0, dtype=np.int32)
        am = np.zeros(1, dtype=np.uint64)
        pf = np.full(1, -1, dtype=np.int32)
        bs = np.empty((2, 1), dtype=np.uint64)
        fl = np.empty(1, dtype=np.int32)
        hk = np.full(16, _VAZIO_HT, dtype=np.uint64)
        hv = np.zeros(16, dtype=np.int8)
        st = np.zeros(1, dtype=np.int64)
        minimax_ab_numba(
            np.uint64(0), 1, -1, 1, 0, 1, np.uint64(0),
            off, nei, am, pf, bs, fl, hk, hv, np.uint64(15), st, 1,
        )
        _NUMBA_AQUECIDO = True
else:
    minimax_ab_numba = None

    def _aquecer_numba() -> None:
        pass


class Minimax:
    """
    Minimax com Poda Alfa-Beta para jogos combinatórios imparciais sob convenção normal.
    - Alice = MAX (+1)
    - Bob = MIN (-1)
    - Com corte alfa-beta: interrompe a busca no nó assim que alfa >= beta.
    - Memoização: tabela de transposição para não recalcular estados já visitados.
    - Aceleração Numba (@njit): executa o núcleo de busca em código nativo LLVM quando n <= 63.
    """

    def __init__(self, jogo, memoizacao: bool = True):
        """Inicializa o resolvedor Minimax associado a uma instância do jogo."""
        self.jogo = jogo                              # Referência à camada de regras do jogo
        self.memoizacao = memoizacao                  # Booleano que ativa ou desativa o cache de transposição
        self.tabela: dict[int, int] = {}              # Dicionário Python (fallback para n > 63 ou jogos customizados)
        self.nos_visitados = 0                        # Contador cumulativo de nós visitados na árvore de busca
        self.tempo_s = 0.0                            # Tempo total consumido pela resolução em segundos
        sys.setrecursionlimit(max(10_000, sys.getrecursionlimit()))

        # Verifica se o Numba está disponível e o jogo fornece representação NumPy (CSR + bitmask, n <= 63)
        self._usa_numba = TEM_NUMBA and getattr(jogo, "adj_mask", None) is not None and jogo.n <= 63
        if self._usa_numba:
            _aquecer_numba()
            n = jogo.n
            cap = 1 << min(23, max(12, n + 1))
            self._mascara_internos = np.uint64(getattr(jogo, "mascara_internos", 0))
            pf_arr = getattr(jogo, "pred_folha_arr", None)
            self._pred_folha = pf_arr if pf_arr is not None else np.full(n, -1, dtype=np.int32)
            self._ht_keys = np.full(cap, _VAZIO_HT, dtype=np.uint64)
            self._ht_vals = np.zeros(cap, dtype=np.int8)
            self._ht_mask = np.uint64(cap - 1)
            self._buf_suc = np.empty((n + 1, n), dtype=np.uint64)
            self._fila = np.empty(n, dtype=np.int32)
            self._stats = np.zeros(1, dtype=np.int64)

    def valor(self, S: int, jogador: int, alfa: int = MIN, beta: int = MAX) -> int:
        """
        Calcula recursivamente o valor Minimax do estado S para o jogador da vez com poda Alfa-Beta.
        Retorna:
          +1: Alice possui estratégia vencedora forçada a partir de S.
          -1: Bob possui estratégia vencedora forçada a partir de S.
        """
        if self._usa_numba:
            self._stats[0] = 0
            res = int(
                minimax_ab_numba(
                    np.uint64(S),
                    jogador,
                    alfa,
                    beta,
                    0,
                    self.jogo.n,
                    self._mascara_internos,
                    self.jogo.offsets,
                    self.jogo.neighbors,
                    self.jogo.adj_mask,
                    self._pred_folha,
                    self._buf_suc,
                    self._fila,
                    self._ht_keys,
                    self._ht_vals,
                    self._ht_mask,
                    self._stats,
                    1 if self.memoizacao else 0,
                )
            )
            self.nos_visitados += int(self._stats[0])
            return res

        self.nos_visitados += 1

        # Corte de paridade também no fallback em Python puro
        mi = getattr(self.jogo, "mascara_internos", None)
        if mi is not None and (S & mi) == mi:
            restantes = self.jogo.n - S.bit_count()
            return jogador if (restantes & 1) else -jogador

        if self.memoizacao and S in self.tabela:
            return self.tabela[S] * jogador

        sucessores = self.jogo.jogadas(S)
        if not sucessores:
            v = -jogador
        elif jogador == MAX:
            v = MIN
            for _, filho in sucessores:
                v = max(v, self.valor(filho, MIN, alfa, beta))
                alfa = max(alfa, v)
                if alfa >= beta:
                    break
        else:
            v = MAX
            for _, filho in sucessores:
                v = min(v, self.valor(filho, MAX, alfa, beta))
                beta = min(beta, v)
                if beta <= alfa:
                    break

        if self.memoizacao:
            self.tabela[S] = v * jogador

        return v

    def resolver(self) -> int:
        """
        Dispara a resolução do jogo a partir do estado inicial (S = 0), com Alice (MAX) iniciando.
        Cronometra o tempo de execução e armazena em self.tempo_s.
        """
        t0 = time.perf_counter()
        try:
            return self.valor(self.jogo.estado_inicial(), MAX, MIN, MAX)
        finally:
            self.tempo_s = time.perf_counter() - t0

    def variacao_principal(self) -> list[dict]: 
        """
        Reconstrói a sequência ótima de lances (partida perfeita) jogada por ambos os jogadores.
        Retorna uma lista de dicionários com turno, jogador, vértice escolhido e vértices propagados.
        """
        from .convexidade import fecho_p3  # Importação local para obter a lista exata de vértices propagados
        S = self.jogo.estado_inicial()     # Começa no estado inicial S = 0
        jogador = MAX                      # Alice é sempre a primeira a jogar
        partida = []                       # Lista que registrará o histórico dos turnos

        while True:
            sucessores = self.jogo.jogadas(S)  # Gera os lances possíveis a partir de S
            if not sucessores:
                return partida  # Fim da partida: nenhum lance restante

            # Seleciona o lance ótimo: escolhe um sucessor que mantém a condição de vitória do jogador atual
            escolhida = next(
                ((v, prox) for v, prox in sucessores if self.valor(prox, -jogador) == jogador),
                sucessores[0],  # Se estiver em posição perdedora forçada, escolhe o primeiro lance disponível
            )
            v, prox_S = escolhida  # Desempacota o vértice jogado v e o novo estado prox_S
            _, props = fecho_p3(self.jogo.adj, S | (1 << v))  # Identifica os vértices infectados em cascata

            # Registra as informações do turno na lista da partida
            partida.append({
                "turno": len(partida) + 1,
                "jogador": "Alice" if jogador == MAX else "Bob",
                "jogada": self.jogo.rotulo(v),
                "propagados": [self.jogo.rotulo(p) for p in props],
            })
            S = prox_S            # Atualiza o estado atual do grafo
            jogador = -jogador    # Alterna o jogador da vez (MAX -> MIN -> MAX ...)
