"""
Camada 2 -- Operadores da convexidade P_3 (independentes de jogo e de algoritmo).

Estados são representados como máscaras de bits (``int``): o bit v vale 1 se,
e somente se, o vértice v pertence ao conjunto. O grafo é dado por listas de
adjacência ``adj`` com vértices 0..n-1. As funções valem para grafos
quaisquer, não apenas caterpillars.

Definições (Centeno et al., 2013):
    I(S)  = S ∪ { v : |N(v) ∩ S| >= 2 }          (intervalo P_3)
    S é P_3-convexo  <=>  I(S) = S
    [S]   = menor conjunto P_3-convexo que contém S (fecho convexo P_3)
"""

from __future__ import annotations

from collections import deque


def _contar(adj: list[list[int]], v: int, S: int) -> int:
    return sum(1 for u in adj[v] if (S >> u) & 1)


def intervalo_p3(adj: list[list[int]], S: int) -> int:
    """Aplica uma única vez o operador de intervalo I(S)."""
    R = S
    for v in range(len(adj)):
        if not (S >> v) & 1 and _contar(adj, v, S) >= 2:
            R |= 1 << v
    return R


def eh_p3_convexo(adj: list[list[int]], S: int) -> bool:
    return intervalo_p3(adj, S) == S


def fecho_p3_iterativo(adj: list[list[int]], S: int) -> int:
    """Fecho por iteração de ponto fixo de I(.). Usado como oráculo nos testes."""
    while True:
        R = intervalo_p3(adj, S)
        if R == S:
            return S
        S = R


def fecho_p3(adj: list[list[int]], S: int) -> tuple[int, list[int]]:
    """
    Fecho convexo P_3 por propagação em largura (Algoritmo 1 do TCC).

    Mantém, para cada vértice fora do conjunto, o número de vizinhos já
    pertencentes a ele; um vértice entra na fila no exato momento em que esse
    contador atinge 2. Cada vértice é enfileirado no máximo uma vez e cada
    aresta é examinada O(1) vezes: tempo e espaço O(|V| + |E|).

    Retorna (máscara do fecho, lista dos vértices incorporados por propagação,
    na ordem de incorporação).
    """
    n = len(adj)
    contador = [0] * n
    fila: deque[int] = deque()
    enfileirado = [False] * n
    for v in range(n):
        if not (S >> v) & 1:
            c = _contar(adj, v, S)
            contador[v] = c
            if c >= 2:
                fila.append(v)
                enfileirado[v] = True
    propagados: list[int] = []
    while fila:
        u = fila.popleft()
        S |= 1 << u
        propagados.append(u)
        for w in adj[u]:
            if not (S >> w) & 1 and not enfileirado[w]:
                contador[w] += 1
                if contador[w] >= 2:
                    fila.append(w)
                    enfileirado[w] = True
    return S, propagados


def para_mascara(vertices) -> int:
    m = 0
    for v in vertices:
        m |= 1 << v
    return m


def de_mascara(S: int) -> list[int]:
    return [v for v in range(S.bit_length()) if (S >> v) & 1]
