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

        # 1. Simetria de folhas gêmeas (grau 1 adjacentes ao mesmo nó interno)
        grau = [len(viz) for viz in adj]
        internos = [v for v in range(self.n) if grau[v] > 1]
        self.mascara_internos = sum(1 << v for v in internos)
        self.pai_folha = [adj[v][0] if grau[v] == 1 else -1 for v in range(self.n)]
        folhas_de = [sorted(u for u in adj[v] if grau[u] == 1) for v in internos]
        folhas_mask = [sum(1 << u for u in adj[v] if grau[u] == 1) if grau[v] > 1 else 0 for v in range(self.n)]

        self.pred_folha = [-1] * self.n
        for folhas in folhas_de:
            for idx in range(1, len(folhas)):
                self.pred_folha[folhas[idx]] = folhas[idx - 1]

        # 2. Simetria de reflexão (espelhamento da espinha quando palíndroma)
        espelho_v = list(range(self.n))
        f_lens = [len(f) for f in folhas_de]
        if len(internos) >= 2 and f_lens == f_lens[::-1]:
            m_int = len(internos)
            for i, u in enumerate(internos):
                u_rev = internos[m_int - 1 - i]
                espelho_v[u] = u_rev
                for f1, f2 in zip(folhas_de[i], folhas_de[m_int - 1 - i]):
                    espelho_v[f1] = f2

        # Tabelas de consulta (LUTs) de 8 bits para espelhamento e máscara de folhas inertes em O(1)
        self.num_blocos = (self.n + 7) // 8
        self.luts: list[list[int]] = []
        self.luts_inertes: list[list[int]] = []
        for b in range(self.num_blocos):
            offset = b * 8
            lut = [0] * 256
            lut_in = [0] * 256
            for val in range(256):
                acc = 0
                acc_in = 0
                for bit in range(8):
                    v = offset + bit
                    if (val >> bit) & 1 and v < self.n:
                        acc |= (1 << espelho_v[v])
                        acc_in |= folhas_mask[v]
                lut[val] = acc
                lut_in[val] = acc_in
            self.luts.append(lut)
            self.luts_inertes.append(lut_in)

    def estado_inicial(self) -> int:
        return 0

    def chave_canonica(self, S: int) -> int:
        """
        Retorna a chave canônica combinando:
        1. O estado ativo espelhado min(S_ativo, espelho(S_ativo)) (excluindo folhas inertes).
        2. A contagem total de folhas inertes livres (folhas cujo pai interno já está infectado).
        """
        mask_inertes = 0
        for b in range(self.num_blocos):
            mask_inertes |= self.luts_inertes[b][(S >> (b * 8)) & 0xFF]
        inertes_livres = (mask_inertes & ~S).bit_count()
        S_ativo = S & ~mask_inertes
        S_rev = 0
        for b in range(self.num_blocos):
            S_rev |= self.luts[b][(S_ativo >> (b * 8)) & 0xFF]
        ch_ativo = S_ativo if S_ativo <= S_rev else S_rev
        return (ch_ativo << 6) | inertes_livres

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
