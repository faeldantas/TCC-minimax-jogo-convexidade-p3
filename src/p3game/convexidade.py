"""
Camada 2 -- Operador de Fecho Convexo P_3 em bitmask (CORE + Numba JIT).
Implementa o cálculo exato do fecho P3 por propagação em largura (BFS linear) em O(|V| + |E|),
incluindo a versão incremental compilada para código de máquina via Numba (@njit).
"""

from __future__ import annotations
from collections import deque  # Fila de duas pontas de alta performance para a BFS

try:
    import numpy as np
    from numba import njit, int32, int64, uint64
    TEM_NUMBA = True
except ImportError:
    TEM_NUMBA = False


def fecho_p3(adj: list[list[int]], S: int) -> tuple[int, list[int]]:
    """
    Calcula o Fecho Convexo P_3 do conjunto S por propagação em largura (BFS) em tempo O(|V| + |E|).
    
    Regra de infecção P_3:
      Qualquer vértice v externo (v ∉ S) que possua pelo menos 2 vizinhos já infectados
      é incorporado ao conjunto em cascata contínua até a estabilização (ponto fixo).

    Parâmetros:
      adj: Lista de adjacência do grafo (adj[v] lista todos os vizinhos de v).
      S: Máscara inteira de bits representando o subconjunto de vértices infectados.

    Retorno:
      Uma tupla (S_final, propagados):
        - S_final: Máscara inteira com todos os vértices após a propagação.
        - propagados: Lista ordenada dos vértices incorporados por contágio P_3.
    """
    n = len(adj)                   # Número total de vértices no grafo
    contador = [0] * n             # Vetor contador: armazena a quantidade de vizinhos infectados de cada vértice
    fila: deque[int] = deque()     # Fila FIFO para controlar a ordem de propagação dos nós que atingem o limiar >= 2
    enfileirado = [False] * n      # Vetor booleano para evitar enfileiramento duplicado de um mesmo vértice

    # 1. Fase de Inicialização: conta vizinhos infectados para todos os vértices não infectados
    for v in range(n):
        if not (S >> v) & 1:  # Se o vértice v NÃO está infectado (bit v == 0)
            # Conta quantos vizinhos u de v pertencem ao conjunto S atual (bit u == 1)
            c = sum(1 for u in adj[v] if (S >> u) & 1)
            contador[v] = c  # Armazena a contagem inicial
            if c >= 2:       # Se já possui pelo menos 2 vizinhos infectados, qualifica para propagar
                fila.append(v)
                enfileirado[v] = True  # Marca como já colocado na fila

    # 2. Fase de Propagação em Cascata (Efeito Dominó da BFS)
    propagados: list[int] = []  # Lista que registrará a ordem cronológica de cada infecção por cascata
    while fila:
        u = fila.popleft()          # Retira o próximo vértice a ser ativado
        S |= (1 << u)               # Infecta o vértice u (liga o bit u na máscara S)
        propagados.append(u)        # Registra u na lista de vértices propagados

        # Notifica todos os vizinhos w de u sobre a nova infecção
        for w in adj[u]:
            if not (S >> w) & 1 and not enfileirado[w]:  # Se w ainda está livre e não está na fila
                contador[w] += 1                         # Incrementa o número de vizinhos infectados de w
                if contador[w] >= 2:                     # Se atingiu o limiar de 2 vizinhos, propaga
                    fila.append(w)
                    enfileirado[w] = True                # Marca w para não ser re-adicionado

    return S, propagados  # Retorna o estado fechado final e a sequência de propagações


if TEM_NUMBA:
    @njit(uint64(int32[:], int32[:], uint64[:], uint64, int64, int32[:]))
    def fecho_p3_numba(
        offsets: np.ndarray,
        neighbors: np.ndarray,
        adj_mask: np.ndarray,
        S: int,
        v: int,
        fila: np.ndarray,
    ) -> int:
        """
        Versão incremental e compilada via Numba (@njit) do operador de Fecho Convexo P_3.
        Parte do pressuposto de que S já é fechado e infecta v:
          1. Verifica apenas os vizinhos de v e dos propagados em cascata (sem varrer o grafo).
          2. Usa o truque bitwise (viz_inf & (viz_inf - 1)) != 0 para testar >= 2 vizinhos em O(1).
        """
        S = S | (uint64(1) << uint64(v))  # Infecta o vértice escolhido v
        head = 0
        tail = 0

        # Verifica apenas os vizinhos diretos de v
        for idx in range(offsets[v], offsets[v + 1]):
            u = neighbors[idx]
            if not ((S >> uint64(u)) & uint64(1)):  # Se u ainda não está infectado
                viz_inf = S & adj_mask[u]           # Máscara dos vizinhos infectados de u
                # Expressão bitwise verdadeira se e só se u possui >= 2 vizinhos infectados
                if (viz_inf & (viz_inf - uint64(1))) != uint64(0):
                    S = S | (uint64(1) << uint64(u))
                    fila[tail] = u
                    tail += 1

        # Propaga em cascata (BFS) sobre o buffer pré-alocado 'fila'
        while head < tail:
            u = fila[head]
            head += 1
            for idx in range(offsets[u], offsets[u + 1]):
                w = neighbors[idx]
                if not ((S >> uint64(w)) & uint64(1)):
                    viz_inf = S & adj_mask[w]
                    if (viz_inf & (viz_inf - uint64(1))) != uint64(0):
                        S = S | (uint64(1) << uint64(w))
                        fila[tail] = w
                        tail += 1

        return S
else:
    fecho_p3_numba = None
