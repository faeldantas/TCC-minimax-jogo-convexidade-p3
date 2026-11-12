"""
Camada 2 -- Operador de Fecho Convexo P_3 em bitmask (CORE).
"""

from __future__ import annotations
from collections import deque


def fecho_p3(adj: list[list[int]], S: int) -> tuple[int, list[int]]:
    """
    Fecho convexo P_3 por propagação em largura (BFS) em O(|V| + |E|).
    Qualquer vértice externo com >= 2 vizinhos infectados é incorporado em cascata.
    Retorna (máscara final do fecho, lista de vértices propagados na ordem).
    """
    n = len(adj)
    contador = [0] * n
    fila: deque[int] = deque()
    enfileirado = [False] * n

    for v in range(n):
        if not (S >> v) & 1: # Se o vértice v não está infectado (não faz parte do conjunto S)
            c = sum(1 for u in adj[v] if (S >> u) & 1) # Conta quantos vizinhos de v estão infectados (fazem parte do conjunto S)
            contador[v] = c
            if c >= 2:
                fila.append(v)
                enfileirado[v] = True

    propagados: list[int] = []
    while fila:
        u = fila.popleft()
        S |= (1 << u)
        propagados.append(u)
        for w in adj[u]:
            if not (S >> w) & 1 and not enfileirado[w]:
                contador[w] += 1
                if contador[w] >= 2:
                    fila.append(w)
                    enfileirado[w] = True

    return S, propagados
