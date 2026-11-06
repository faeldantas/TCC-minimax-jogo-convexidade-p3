"""
p3game -- Closed Hull Game na convexidade P_3 em grafos caterpillar.

Arquitetura em camadas (cada camada depende apenas das anteriores):
    grafos.py       representação, enumeração canônica e famílias de caterpillars
    convexidade.py  operadores I(S) e [S] da convexidade P_3 (bitmask)
    jogo.py         interface abstrata de jogo imparcial + CHG em P_3
    minimax.py      solucionador Minimax genérico (corte, memoização, simetria)
"""

from .grafos import (
    Caterpillar, enumerar_caterpillars, numero_caterpillars, eh_assinatura_canonica,
    canonizar, eh_caterpillar, assinatura_de_grafo,
    caminho, estrela, dupla_estrela, uniforme, vassoura_dupla,
)
from .convexidade import fecho_p3, fecho_p3_iterativo, intervalo_p3, eh_p3_convexo, para_mascara, de_mascara
from .jogo import JogoImparcialNormal, JogoFechoConvexoP3, JogoFechoConvexoP3Caterpillar
from .minimax import Minimax, MAX, MIN, LimiteDeNosExcedido


def resolver_caterpillar(L, **opcoes) -> tuple[int, Minimax]:
    """Atalho: resolve o CHG-P3 no caterpillar de assinatura L."""
    jogo = JogoFechoConvexoP3Caterpillar(Caterpillar(tuple(L)))
    solver = Minimax(jogo, **opcoes)
    return solver.resolver(), solver


from .experimentos import (
    analisar_uniforme_expansao,
    executar_experimento_uniformes_espinha,
    assinatura_caterpillar_cheio,
    analisar_caterpillar_cheio_2_pernas,
    salvar_resultados_caterpillar_cheio,
)
from .visualizacao import (
    calcular_layout_caterpillar,
    reconstruir_historico_partida,
    desenhar_grafo_turno,
    gerar_painel_estatico,
    abrir_visualizador_interativo,
    gerar_html_interativo,
)

