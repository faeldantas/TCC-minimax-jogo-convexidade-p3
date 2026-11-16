"""
Camada 1 -- Representação topológica de grafos caterpillar (CORE).
Constrói a estrutura topológica, lista de adjacências e rótulos a partir da assinatura L.
"""

from __future__ import annotations
from dataclasses import dataclass, field

# Define o alias de tipo para assinaturas de caterpillar como tuplas de inteiros
Assinatura = tuple[int, ...]


@dataclass
class Caterpillar:
    """
    Representação topológica de um grafo Caterpillar construído a partir de uma assinatura L = (l_1, ..., l_k).
    - Espinha: Caminho central formado pelos vértices de índices 0 até k-1 (rótulos v1 até vk).
    - Pernas: Folhas penduradas nos nós da espinha, numeradas sequencialmente a partir do índice k.
    """
    assinatura: Assinatura
    adj: list[list[int]] = field(init=False)      # Lista de adjacência do grafo (adj[v] lista os vizinhos de v)
    rotulos: list[str] = field(init=False)        # Lista de rótulos amigáveis de cada vértice ("v1", "v2", ...)
    espinha: list[int] = field(init=False)        # Lista dos índices que compõem o caminho da espinha [0, ..., k-1]

    def __post_init__(self) -> None:
        """Executado automaticamente após a instanciação para montar toda a topologia do grafo."""
        # Normaliza a assinatura para garantir que seja uma tupla imutável de números inteiros
        L = tuple(int(x) for x in self.assinatura)
        self.assinatura = L
        k = len(L)              # Comprimento do caminho da espinha (quantidade de nós da espinha)
        n = k + sum(L)          # Número total de vértices do caterpillar: nós da espinha + total de pernas
        self.adj = [[] for _ in range(n)]      # Inicializa a lista de adjacência com listas vazias para os n vértices
        self.rotulos = [f"v{i + 1}" for i in range(n)]  # Gera os nomes dos vértices indexados a partir de 1
        self.espinha = list(range(k))          # Registra os índices da espinha: [0, 1, ..., k-1]

        # 1. Conecta os vértices da espinha em um caminho simples linear: v0 - v1 - ... - v(k-1)
        for i in range(k - 1):
            self.adj[i].append(i + 1)  # Adiciona aresta não-direcionada de i para i+1
            self.adj[i + 1].append(i)  # Adiciona aresta não-direcionada de i+1 para i

        # 2. Conecta as pernas (folhas) a cada respectivo nó da espinha
        prox = k  # Próximo índice livre para uma nova perna (começa logo após o último nó da espinha)
        for i, qtd in enumerate(L):   # Percorre cada nó i da espinha e sua respectiva quantidade de pernas
            for _ in range(qtd):      # Adiciona cada uma das pernas configuradas para o nó i
                self.adj[i].append(prox)  # Aresta do nó da espinha i para a folha prox
                self.adj[prox].append(i)  # Aresta da folha prox de volta para o nó da espinha i
                prox += 1                 # Avança para o próximo índice de vértice folha

    @property
    def n(self) -> int:
        """Retorna o número total de vértices |V(G)| no grafo caterpillar."""
        return len(self.adj)

    @property
    def k(self) -> int:
        """Retorna o número de vértices no caminho da espinha |V(P_k)|."""
        return len(self.assinatura)
