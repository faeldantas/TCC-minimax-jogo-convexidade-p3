"""
Ponto de entrada principal (CLI) do pacote `p3game`.
Permite validar os exemplos do TCC e rodar as 3 baterias experimentais.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p3game import Caterpillar, fecho_p3, para_mascara, resolver_caterpillar
from p3game.experimentos import executar_tudo


def validar_exemplos_do_tcc():
    print("=" * 75)
    print("VALIDAÇÃO DOS EXEMPLOS DO TCC (Figuras 1, 2, 3, 4 e 7)")
    print("=" * 75)

    # 1. Figura 2: Fecho convexo de S = {v2, v6, v9, v10} em L = (0, 2, 0, 1, 1, 0)
    cat = Caterpillar((0, 2, 0, 1, 1, 0))
    idx = {r: i for i, r in enumerate(cat.rotulos)}
    S_ini = para_mascara(idx[r] for r in ["v2", "v6", "v9", "v10"])
    _, prop = fecho_p3(cat.adj, S_ini)
    print(f"[Figura 2] Propagação para S={{v2,v6,v9,v10}}: {[cat.rotulos[u] for u in prop]}")

    # 2. Figura 3: Caminho P_3 = (0, 0, 0)
    v_p3, s_p3 = resolver_caterpillar((0, 0, 0))
    venc_p3 = [s_p3.jogo.rotulo(j) for j in s_p3.jogadas_vencedoras()]
    print(f"[Figura 3] Caminho P_3=(0,0,0): utilidade={v_p3:+d}, jogadas ótimas de Alice={venc_p3}")

    # 3. Figuras 4 e 7: Estudo de caso L = (0, 2, 0, 1, 1, 0)
    v_tcc, s_tcc = resolver_caterpillar((0, 2, 0, 1, 1, 0))
    pv_tcc = s_tcc.variacao_principal()
    print(f"[Figura 7] Grafo TCC L=(0,2,0,1,1,0): utilidade={v_tcc:+d} (Bob vence em {len(pv_tcc)} turnos)")
    print("  Variação principal:", " -> ".join(
        f"{p['jogador'][0]}({p['jogada']})" + (f"+{p['propagados']}" if p["propagados"] else "")
        for p in pv_tcc
    ))
    print()


if __name__ == "__main__":
    validar_exemplos_do_tcc()
    executar_tudo()
