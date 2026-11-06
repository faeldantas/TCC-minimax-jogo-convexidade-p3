"""
p3game -- Closed Hull Game na convexidade P_3 em grafos caterpillar (Versão CORE).

Camadas essenciais:
    grafos.py       representação estrutural e enumeração de caterpillars
    convexidade.py  operadores de fecho convexo P_3 em bitmask
    jogo.py         regras do Closed Hull Game (CHG)
    minimax.py      algoritmo Minimax puro (sem poda Alfa-Beta)
"""

from .grafos import (
    Caterpillar,
    enumerar_caterpillars,
    numero_caterpillars,
    eh_assinatura_canonica,
    canonizar,
    eh_caterpillar,
    assinatura_de_grafo,
    caminho,
    estrela,
    dupla_estrela,
    uniforme,
    vassoura_dupla,
)
from .convexidade import (
    fecho_p3,
    fecho_p3_iterativo,
    intervalo_p3,
    eh_p3_convexo,
    para_mascara,
    de_mascara,
)
from .jogo import (
    JogoImparcialNormal,
    JogoFechoConvexoP3,
    JogoFechoConvexoP3Caterpillar,
)
from .minimax import Minimax, MAX, MIN


def resolver_caterpillar(L, **opcoes) -> tuple[int, Minimax]:
    """Atalho: resolve o CHG-P3 puro no caterpillar de assinatura L."""
    cat = Caterpillar(tuple(L))
    jogo = JogoFechoConvexoP3(cat.adj, cat.rotulos)
    solver = Minimax(jogo, **opcoes)
    return solver.resolver(), solver
