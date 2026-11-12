"""
Camada 3 -- Regras do Closed Hull Game (CHG) na convexidade P_3 (CORE).
"""

from __future__ import annotations
from .convexidade import fecho_p3_incremental


class JogoFechoConvexoP3:
    """
    Closed Hull Game sob convenção normal com redução por simetria.
    - Estado inicial: conjunto vazio S = 0.
    - Jogada legal: escolher qualquer vértice v não infectado (respeitando a ordem
      canônica de folhas gêmeas e deduplicação por espelhamento).
    - Transição: novo estado é o fecho [S ∪ {v}]_P3.
    - Vitória: quem faz a última jogada válida (completando o grafo) vence.
    """

    def __init__(self, adj: list[list[int]], rotulos: list[str]):
        self.adj = adj
        self.rotulos = rotulos
        self.n = len(adj)
        self.adj_mask = [sum(1 << u for u in viz) for viz in adj]

        # 1. Simetria de folhas gêmeas e máscaras por nó interno da espinha
        grau = [len(viz) for viz in adj]
        self.internos = [v for v in range(self.n) if grau[v] > 1]
        self.mascara_internos = sum(1 << v for v in self.internos)
        self.pai_folha = [adj[v][0] if grau[v] == 1 else -1 for v in range(self.n)]
        folhas_de = [sorted(u for u in adj[v] if grau[u] == 1) for v in self.internos]
        self.folhas_mask_internos = [sum(1 << u for u in f) for f in folhas_de]

        self.pred_folha = [-1] * self.n
        for folhas in folhas_de:
            for idx in range(1, len(folhas)):
                self.pred_folha[folhas[idx]] = folhas[idx - 1]

    def estado_inicial(self) -> int:
        return 0

    def chave_canonica(self, S: int) -> tuple[tuple[int, ...], int]:
        """
        Decompõe o estado S pelos vértices internos já infectados (pontos de articulação):
        1. Cada segmento contíguo de nós internos livres é codificado em um inteiro
           canônico min(fwd, rev) invariante por reflexão local.
        2. Pares de segmentos ativos idênticos se anulam por simetria (C + C = 0).
        3. As folhas inertes livres são reduzidas a 1 bit de paridade (inertes & 1).
        """
        inertes = 0
        segs: list[int] = []
        fwd = 0
        rev = 0
        shift = 0
        am = self.adj_mask

        for u, fmask in zip(self.internos, self.folhas_mask_internos):
            if (S >> u) & 1:
                inertes += (fmask & ~S).bit_count()
                if shift:
                    seg = (fwd if fwd <= rev else rev) | (1 << shift)
                    if seg in segs:
                        segs.remove(seg)
                    else:
                        segs.append(seg)
                    fwd = rev = shift = 0
            else:
                c_inf = 1 if (S & am[u]) else 0
                f_liv = (fmask & ~S).bit_count()
                code = (c_inf << 2) | f_liv
                fwd = (fwd << 3) | code
                rev |= (code << shift)
                shift += 3

        if shift:
            seg = (fwd if fwd <= rev else rev) | (1 << shift)
            if seg in segs:
                segs.remove(seg)
            else:
                segs.append(seg)

        if len(segs) > 1:
            segs.sort()
        return (tuple(segs), inertes & 1)

    def jogadas(self, S: int) -> list[tuple[int, int]]:
        """Retorna lista de pares (vértice escolhido, máscara do estado sucessor) sem simetrias redundantes."""
        res = []
        vistos = set()
        adj = self.adj
        adj_mask = self.adj_mask
        pred = self.pred_folha
        pai_f = self.pai_folha
        ja_tem_inerte = False
        for v in range(self.n):
            if not (S >> v) & 1:
                pf = pred[v]
                if pf != -1 and not ((S >> pf) & 1):
                    continue
                pai = pai_f[v]
                if pai != -1 and ((S >> pai) & 1):
                    if ja_tem_inerte:
                        continue
                    ja_tem_inerte = True
                prox_S = fecho_p3_incremental(adj, adj_mask, S, v)
                ch = self.chave_canonica(prox_S)
                if ch not in vistos:
                    vistos.add(ch)
                    res.append((v, prox_S))
        return res

    def rotulo(self, v: int) -> str:
        return self.rotulos[v]
