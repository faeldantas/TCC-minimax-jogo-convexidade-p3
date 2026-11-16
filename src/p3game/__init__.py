"""
p3game -- Pacote Core: Convexidade P_3 em Caterpillars e Minimax com Alfa-Beta + Numba.
"""

from .grafos import Caterpillar
from .convexidade import fecho_p3, fecho_p3_numba
from .jogo import JogoFechoConvexoP3
from .minimax import Minimax, MAX, MIN
