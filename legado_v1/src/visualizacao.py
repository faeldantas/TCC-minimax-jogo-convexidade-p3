import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def calcular_posicoes_caterpillar(G):
    """
    Calcula coordenadas (x, y) no plano para desenhar o grafo Caterpillar
    exatamente no estilo das Figuras 1, 2 e 4 do TCC:
    - Espinha alinhada horizontalmente no eixo y = 0;
    - Pernas distribuídas simetricamente acima (y > 0) e abaixo (y < 0)
      do respectivo vértice da espinha.
    """
    pos = {}
    espinha = G.graph.get("espinha", [])
    mapa_pernas = G.graph.get("mapa_pernas", {})
    dist_x = 2.0
    dist_y = 1.5

    for i, v_esp in enumerate(espinha):
        x_base = i * dist_x
        pos[v_esp] = (x_base, 0.0)

        folhas = mapa_pernas.get(v_esp, [])
        qtd = len(folhas)

        # Distribui folhas alternadamente em cima e embaixo
        superiores = folhas[0::2]
        inferiores = folhas[1::2]

        for idx, f_sup in enumerate(superiores):
            desloc_x = (idx - (len(superiores) - 1) / 2.0) * 0.8
            pos[f_sup] = (x_base + desloc_x, dist_y)

        for idx, f_inf in enumerate(inferiores):
            desloc_x = (idx - (len(inferiores) - 1) / 2.0) * 0.8
            pos[f_inf] = (x_base + desloc_x, -dist_y)

    # Caso algum grafo genérico seja passado sem metadados de caterpillar
    if not pos:
        pos = nx.spring_layout(G, seed=42)

    return pos


def mostrar_grafo(G, titulo="Estado Atual do Jogo da Convexidade P3", salvar_em=None, exibir=True):
    """
    Renderiza o grafo Caterpillar destacando:
    - Cinza claro: Vértices ainda não infectados (disponíveis para jogada)
    - Vermelho claro: Vértices escolhidos diretamente por Alice (turnos ímpares)
    - Azul claro: Vértices escolhidos diretamente por Bob (turnos pares)
    - Dourado: Vértices infectados automaticamente pelo fechamento convexo P3
    """
    pos = calcular_posicoes_caterpillar(G)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(titulo, fontsize=12, fontweight="bold", pad=15)

    lista_cores = [G.nodes[v].get("color", "lightgray") for v in G.nodes]

    # Rótulos internos e anotações de ordem da jogada
    labels = {}
    for v in G.nodes:
        jogada = G.nodes[v].get("jogada")
        auto = G.nodes[v].get("auto_infectado", False)
        if jogada is not None:
            marcador = f"{jogada}*" if auto else f"{jogada}"
            labels[v] = f"{v}\n(T{marcador})"
        else:
            labels[v] = v

    nx.draw_networkx_edges(G, pos, ax=ax, width=1.8, edge_color="dimgray")
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=lista_cores,
        edgecolors="black",
        linewidths=1.5,
        node_size=1300,
    )
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=9, font_weight="bold")

    # Legenda explicativa
    legenda = [
        mpatches.Patch(color="lightgray", label="Não infectado"),
        mpatches.Patch(color="lightcoral", label="Jogada de Alice"),
        mpatches.Patch(color="cornflowerblue", label="Jogada de Bob"),
        mpatches.Patch(color="gold", label="Fechamento P3 (Automático)"),
    ]
    ax.legend(handles=legenda, loc="upper right", framealpha=0.9, fontsize=8)
    ax.margins(0.2)
    ax.axis("off")
    plt.tight_layout()

    if salvar_em:
        plt.savefig(salvar_em, dpi=300, bbox_inches="tight")

    if exibir:
        plt.show()
    else:
        plt.close(fig)
