"""
Camada 1 -- Representação, enumeração canônica e famílias de grafos *caterpillar*.

Convenções (adotadas também no texto do TCC, Seções 3.1 e 6.1):

* Um *caterpillar* é uma árvore T tal que a remoção de todas as suas folhas
  (vértices de grau 1) resulta em um caminho (Harary; Schwenk, 1973).
* Para |V(T)| >= 3, a **espinha** é um caminho diametral (de comprimento máximo)
  P_k = (v_1, ..., v_k). Como os extremos de um caminho diametral são folhas,
  k >= 3 e os extremos não possuem pernas.
* As **pernas** são os vértices fora da espinha (todas são folhas de T).
* A **assinatura** L = (l_1, ..., l_k) registra quantas pernas são adjacentes a
  cada v_i. Pela convenção diametral, l_1 = l_k = 0.
* A **assinatura canônica** é a lexicograficamente menor entre L e sua reversa.

Com essa convenção, dois caterpillars são isomorfos se, e somente se, possuem a
mesma assinatura canônica (ver docstring de ``assinatura_de_grafo``); isso é
verificado empiricamente em ``tests/test_grafos.py`` contra o enumerador de
árvores não-isomorfas do NetworkX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterator

import networkx as nx

Assinatura = tuple[int, ...]


# ---------------------------------------------------------------------------
# Assinaturas
# ---------------------------------------------------------------------------

def eh_assinatura_canonica(L: Assinatura) -> bool:
    """Verifica se L está na forma canônica diametral (k>=3, l_1=l_k=0, L <= L^R)."""
    L = tuple(L)
    return (
        len(L) >= 3
        and all(isinstance(x, int) and x >= 0 for x in L)
        and L[0] == 0
        and L[-1] == 0
        and L <= L[::-1]
    )


def canonizar(L: Assinatura) -> Assinatura:
    """Retorna min(L, L^R). Pressupõe l_1 = l_k = 0 (espinha diametral)."""
    L = tuple(L)
    return min(L, L[::-1])


def _composicoes_fracas(total: int, partes: int) -> Iterator[tuple[int, ...]]:
    """Todas as tuplas de `partes` inteiros não negativos com soma `total` (stars and bars)."""
    if partes == 0:
        if total == 0:
            yield ()
        return
    for barras in combinations(range(total + partes - 1), partes - 1):
        anterior = -1
        valores = []
        for b in barras:
            valores.append(b - anterior - 1)
            anterior = b
        valores.append(total + partes - 1 - anterior - 1)
        yield tuple(valores)


def enumerar_caterpillars(n: int) -> list[Assinatura]:
    """
    Enumera, sem repetição, as assinaturas canônicas de todos os caterpillars
    não-isomorfos de ordem n >= 3.

    Para cada comprimento de espinha 3 <= k <= n, os k-2 vértices internos da
    espinha recebem n-k pernas (composições fracas); mantém-se apenas L <= L^R.
    """
    if n < 3:
        raise ValueError("A enumeração canônica é definida para n >= 3.")
    resultado = []
    for k in range(3, n + 1):
        for miolo in _composicoes_fracas(n - k, k - 2):
            L = (0,) + miolo + (0,)
            if L <= L[::-1]:
                resultado.append(L)
    return resultado


def numero_caterpillars(n: int) -> int:
    """
    Número de caterpillars não-isomorfos com n >= 3 vértices
    (Harary; Schwenk, 1973): c(n) = 2^(n-4) + 2^floor((n-4)/2).
    """
    if n < 3:
        raise ValueError("Fórmula válida para n >= 3.")
    if n == 3:
        return 1
    return 2 ** (n - 4) + 2 ** ((n - 4) // 2)


# ---------------------------------------------------------------------------
# Estrutura de dados do caterpillar
# ---------------------------------------------------------------------------

@dataclass
class Caterpillar:
    """
    Caterpillar construído a partir de uma assinatura L.

    Rotulação (idêntica à das Figuras do TCC): espinha v_1..v_k nos índices
    0..k-1; pernas numeradas em seguida (v_{k+1}, ...) na ordem dos vértices da
    espinha. Para L = (0, 2, 0, 1, 1, 0): v7, v8 em v2; v9 em v4; v10 em v5.
    """

    assinatura: Assinatura
    adj: list[list[int]] = field(init=False)
    rotulos: list[str] = field(init=False)
    espinha: list[int] = field(init=False)
    pai_da_perna: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        L = tuple(int(x) for x in self.assinatura)
        if len(L) < 1 or any(x < 0 for x in L):
            raise ValueError(f"Assinatura inválida: {L}")
        self.assinatura = L
        k = len(L)
        n = k + sum(L)
        self.adj = [[] for _ in range(n)]
        self.rotulos = [f"v{i + 1}" for i in range(n)]
        self.espinha = list(range(k))
        self.pai_da_perna = {}
        for i in range(k - 1):
            self._aresta(i, i + 1)
        prox = k
        for i, qtd in enumerate(L):
            for _ in range(qtd):
                self._aresta(i, prox)
                self.pai_da_perna[prox] = i
                prox += 1

    def _aresta(self, u: int, v: int) -> None:
        self.adj[u].append(v)
        self.adj[v].append(u)

    # --- parâmetros estruturais -------------------------------------------
    @property
    def n(self) -> int:
        return len(self.adj)

    @property
    def k(self) -> int:
        return len(self.assinatura)

    @property
    def num_pernas(self) -> int:
        return sum(self.assinatura)

    @property
    def folhas(self) -> list[int]:
        """Vértices de grau 1 da árvore (pernas + extremos da espinha, se folhas)."""
        return [v for v in range(self.n) if len(self.adj[v]) == 1]

    @property
    def internos(self) -> list[int]:
        """Vértices de grau >= 2, na ordem do caminho que eles induzem."""
        return [v for v in self.espinha if len(self.adj[v]) >= 2]

    def folhas_por_interno(self) -> list[int]:
        """f_i: número de folhas adjacentes a cada vértice interno (ordem da espinha)."""
        return [sum(1 for u in self.adj[v] if len(self.adj[u]) == 1) for v in self.internos]

    def para_networkx(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        for u in range(self.n):
            for v in self.adj[u]:
                if u < v:
                    G.add_edge(u, v)
        return G


# ---------------------------------------------------------------------------
# Reconhecimento e assinatura a partir de um grafo arbitrário
# ---------------------------------------------------------------------------

def eh_caterpillar(G: nx.Graph) -> bool:
    """Árvore cuja remoção de folhas produz um caminho (ou grafo vazio, se n <= 2)."""
    if G.number_of_nodes() == 0 or not nx.is_tree(G):
        return False
    internos = [v for v in G if G.degree(v) >= 2]
    if not internos:
        return True
    H = G.subgraph(internos)
    return nx.is_connected(H) and max(d for _, d in H.degree()) <= 2


def assinatura_de_grafo(G: nx.Graph) -> Assinatura:
    """
    Assinatura canônica diametral de um caterpillar qualquer com n >= 3.

    Os vértices internos (grau >= 2) induzem um caminho u_1..u_m. Todo caminho
    diametral é formado por esse caminho acrescido de uma folha adjacente a
    u_1 e outra adjacente a u_m. Logo, sendo f_i o número de folhas adjacentes a
    u_i, a assinatura é (0, f_1 - 1, f_2, ..., f_{m-1}, f_m - 1, 0) para m >= 2,
    e (0, f_1 - 2, 0) para m = 1 (estrela). Como ela depende apenas de (f_i) e
    da orientação do caminho, ela é um invariante completo de isomorfismo.
    """
    if not eh_caterpillar(G) or G.number_of_nodes() < 3:
        raise ValueError("O grafo não é um caterpillar com pelo menos 3 vértices.")
    internos = {v for v in G if G.degree(v) >= 2}
    H = G.subgraph(internos)
    extremos = [v for v in H if H.degree(v) <= 1]
    ordem = [extremos[0]] if len(internos) == 1 else list(nx.shortest_path(H, extremos[0], extremos[1]))
    f = [sum(1 for u in G[v] if G.degree(u) == 1) for v in ordem]
    if len(f) == 1:
        L = (0, f[0] - 2, 0)
    else:
        L = (0, f[0] - 1, *f[1:-1], f[-1] - 1, 0)
    return canonizar(L)


# ---------------------------------------------------------------------------
# Famílias paramétricas (Bateria II)
# ---------------------------------------------------------------------------

def caminho(n: int) -> Assinatura:
    """P_n, n >= 3."""
    return (0,) * n


def estrela(f: int) -> Assinatura:
    """K_{1,f}, f >= 2 folhas."""
    return (0, f - 2, 0)


def dupla_estrela(a: int, b: int) -> Assinatura:
    """S(a,b): dois centros adjacentes com a e b folhas, a, b >= 1."""
    return canonizar((0, a - 1, b - 1, 0))


def uniforme(m: int, p: int) -> Assinatura:
    """C(m,p): m vértices internos da espinha diametral, cada um com p pernas."""
    return (0,) + (p,) * m + (0,)


def vassoura_dupla(m: int, a: int, b: int) -> Assinatura:
    """D(m,a,b): espinha com m vértices internos; pernas só no 2º (a) e no penúltimo (b), m >= 2."""
    return canonizar((0, a) + (0,) * (m - 2) + (b, 0))
