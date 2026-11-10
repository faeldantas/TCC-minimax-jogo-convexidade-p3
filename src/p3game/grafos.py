"""
Camada 1 -- Representação topológica de grafos caterpillar (CORE).
"""

from __future__ import annotations
from dataclasses import dataclass, field

Assinatura = tuple[int, ...]


@dataclass
class Caterpillar:
    """
    Caterpillar construído a partir de uma assinatura L = (l_1, ..., l_k).
    Espinha: vértices 0..k-1 (v1..vk).
    Pernas: numeradas sequencialmente a partir de k.
    """
    assinatura: Assinatura
    adj: list[list[int]] = field(init=False) # Lista de adjacência do grafo caterpillar
    rotulos: list[str] = field(init=False) # Lista de rótulos dos vértices do grafo caterpillar
    espinha: list[int] = field(init=False) # Lista de índices dos vértices da espinha do grafo caterpillar

    def __post_init__(self) -> None:
        L = tuple(int(x) for x in self.assinatura) # Garante que a assinatura seja uma tupla de inteiros
        self.assinatura = L
        k = len(L) # Número de vértices na espinha do caterpillar
        n = k + sum(L) # Número total de vértices no caterpillar (espinha + pernas)
        self.adj = [[] for _ in range(n)] # Inicializa a lista de adjacência com listas vazias para cada vértice
        self.rotulos = [f"v{i + 1}" for i in range(n)]
        self.espinha = list(range(k)) # Lista de índices dos vértices da espinha do caterpillar (0..k-1)

        for i in range(k - 1): # Conecta os vértices da espinha em uma linha
            self.adj[i].append(i + 1) # Adiciona a aresta do vértice i para o vértice i+1
            self.adj[i + 1].append(i)

        prox = k # Índice do próximo vértice a ser adicionado (começa após a espinha)
        for i, qtd in enumerate(L): # Para cada vértice da espinha, adiciona as pernas correspondentes
            for _ in range(qtd): # Adiciona uma perna ao vértice i da espinha 
                self.adj[i].append(prox) # Adiciona a aresta do vértice i da espinha para a perna prox
                self.adj[prox].append(i) # Adiciona a aresta da perna prox para o vértice i da espinha
                prox += 1 # Incrementa o índice do próximo vértice a ser adicionado (próxima perna)

    @property
    def n(self) -> int: # Retorna o número total de vértices no caterpillar
        return len(self.adj)

    @property
    def k(self) -> int: # Retorna o número de vértices na espinha do caterpillar
        return len(self.assinatura)
