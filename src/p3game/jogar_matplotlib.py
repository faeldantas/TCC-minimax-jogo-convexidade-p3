"""
Interface Gráfica Minimalista com Matplotlib para Jogar contra a IA.

O usuário joga como Alice (1º lance, Vermelho) e a IA joga como Bob (Azul).
O fecho convexo P_3 propaga em cascata (Dourado/Amarelo).
"""

from __future__ import annotations

import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .grafos import Caterpillar, Assinatura
from .jogo import JogoFechoConvexoP3
from .minimax import Minimax, MAX, MIN
from .convexidade import fecho_p3


def calcular_layout(cat: Caterpillar) -> dict[int, tuple[float, float]]:
    """Calcula as coordenadas 2D (x, y) do caterpillar: espinha em y=0 e pernas em y=±1.2."""
    pos: dict[int, tuple[float, float]] = {}
    espacamento = 1.8

    for i in range(cat.k):
        pos[i] = (i * espacamento, 0.0)  # Nós da espinha no centro
        pernas = [u for u in cat.adj[i] if u >= cat.k]
        for j, p in enumerate(pernas):
            dy = 1.2 if j % 2 == 0 else -1.2
            dx = (j // 2) * 0.4
            pos[p] = (i * espacamento + dx, dy)

    return pos


def jogar_caterpillar_matplotlib(assinatura: Assinatura = (0, 2, 2, 0)) -> None:
    """Abre uma janela limpa do Matplotlib para o usuário jogar contra a IA."""
    cat = Caterpillar(assinatura)
    jogo = JogoFechoConvexoP3(cat.adj, cat.rotulos)
    solver = Minimax(jogo, memoizacao=True)
    pos = calcular_layout(cat)

    # Cores
    COR_LIVRE = "#F8FAFC"
    COR_ALICE = "#EF4444"      # Vermelho (Você)
    COR_BOB = "#0EA5E9"        # Azul (IA)
    COR_PROPAGADO = "#F59E0B"  # Dourado (Fecho P3)

    # Estado da partida
    estado = {
        "S": 0,
        "fim": False,
        "jogadas_alice": set(),
        "jogadas_bob": set(),
        "propagados": set(),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.canvas.manager.set_window_title("Jogo da Convexidade P3: Você (Alice) vs IA (Bob)")

    def aplicar_lance(v: int, jogador: str) -> bool:
        """Executa a infecção de um vértice, calcula o fecho P3 e retorna se o jogo acabou."""
        novo_S, props = fecho_p3(cat.adj, estado["S"] | (1 << v))
        estado["S"] = novo_S
        (estado["jogadas_alice"] if jogador == "Alice" else estado["jogadas_bob"]).add(v)
        estado["propagados"].update(props)
        return estado["S"] == (1 << cat.n) - 1

    def redesenhar(mensagem: str = "Sua vez (Alice)! Clique em um nó livre."):
        ax.clear()
        ax.axis("off")

        S = estado["S"]
        total_inf = bin(S).count("1")

        # 1. Arestas
        for u in range(cat.n):
            for v in cat.adj[u]:
                if u < v:
                    ativa = ((S >> u) & 1) and ((S >> v) & 1)
                    ax.plot(
                        [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                        color="#1E293B" if ativa else "#CBD5E1",
                        linewidth=3.0 if ativa else 1.8,
                        zorder=1,
                    )

        # 2. Nós
        for v in range(cat.n):
            x, y = pos[v]
            inf = (S >> v) & 1

            if not inf:
                cor_fill, cor_borda, cor_txt = COR_LIVRE, "#64748B", "#1E293B"
            elif v in estado["jogadas_alice"]:
                cor_fill, cor_borda, cor_txt = COR_ALICE, "#7F1D1D", "#FFFFFF"
            elif v in estado["jogadas_bob"]:
                cor_fill, cor_borda, cor_txt = COR_BOB, "#0C4A6E", "#FFFFFF"
            else:
                cor_fill, cor_borda, cor_txt = COR_PROPAGADO, "#78350F", "#FFFFFF"

            circulo = plt.Circle(
                (x, y), 0.26,
                facecolor=cor_fill, edgecolor=cor_borda, linewidth=2, zorder=2
            )
            ax.add_patch(circulo)
            ax.text(
                x, y, cat.rotulos[v],
                ha="center", va="center", fontsize=10, fontweight="bold",
                color=cor_txt, zorder=3
            )

        # 3. Limites e Título
        xs = [x for x, y in pos.values()]
        ys = [y for x, y in pos.values()]
        ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
        ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
        ax.set_aspect("equal", adjustable="datalim")

        ax.set_title(
            f"{mensagem}\n[Infectados: {total_inf}/{cat.n}]",
            fontsize=12, fontweight="bold", pad=12, color="#0F172A"
        )

        # 4. Legenda
        patches = [
            mpatches.Patch(facecolor=COR_LIVRE, edgecolor="#64748B", label="Livre"),
            mpatches.Patch(facecolor=COR_ALICE, label="Você (Alice)"),
            mpatches.Patch(facecolor=COR_BOB, label="IA (Bob)"),
            mpatches.Patch(facecolor=COR_PROPAGADO, label="Propagação P3"),
        ]
        ax.legend(
            handles=patches, loc="lower center", ncol=4,
            bbox_to_anchor=(0.5, -0.15), frameon=True, fontsize=9
        )

        fig.canvas.draw_idle()

    def ao_clicar(event):
        if estado["fim"] or event.xdata is None or event.ydata is None:
            return

        # 1. Encontra nó clicado
        clicado = None
        for v, (x, y) in pos.items():
            if math.hypot(event.xdata - x, event.ydata - y) <= 0.35:
                if not (estado["S"] >> v) & 1:
                    clicado = v
                    break

        if clicado is None:
            return

        # 2. Turno de Alice (Você)
        if aplicar_lance(clicado, "Alice"):
            estado["fim"] = True
            redesenhar(f"🏆 FIM DE JOGO: Você jogou em {cat.rotulos[clicado]} e VENCEU!")
            return

        # 3. Turno de Bob (IA Minimax)
        jogadas_ia = jogo.jogadas(estado["S"])
        lance_ia = next(
            (v for v, prox in jogadas_ia if solver.valor(prox, MAX) == MIN),
            jogadas_ia[0][0]
        )

        if aplicar_lance(lance_ia, "Bob"):
            estado["fim"] = True
            redesenhar(f"🤖 FIM DE JOGO: A IA jogou em {cat.rotulos[lance_ia]} e VENCEU!")
            return

        redesenhar(f"Você jogou em {cat.rotulos[clicado]} | IA jogou em {cat.rotulos[lance_ia]}. Sua vez!")

    fig.canvas.mpl_connect("button_press_event", ao_clicar)
    redesenhar()
    plt.tight_layout()
    plt.show()
