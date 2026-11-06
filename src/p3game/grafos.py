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
    adj: list[list[int]] = field(init=False)
    rotulos: list[str] = field(init=False)
    espinha: list[int] = field(init=False)

    def __post_init__(self) -> None:
        L = tuple(int(x) for x in self.assinatura)
        self.assinatura = L
        k = len(L)
        n = k + sum(L)
        self.adj = [[] for _ in range(n)]
        self.rotulos = [f"v{i + 1}" for i in range(n)]
        self.espinha = list(range(k))

        for i in range(k - 1):
            self.adj[i].append(i + 1)
            self.adj[i + 1].append(i)

        prox = k
        for i, qtd in enumerate(L):
            for _ in range(qtd):
                self.adj[i].append(prox)
                self.adj[prox].append(i)
                prox += 1

    @property
    def n(self) -> int:
        return len(self.adj)

    @property
    def k(self) -> int:
        return len(self.assinatura)
