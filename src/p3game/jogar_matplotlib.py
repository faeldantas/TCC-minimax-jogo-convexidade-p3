"""
Interface Gráfica Interativa com Matplotlib para Jogar contra a IA (Minimax).

Permite ao usuário jogar o Jogo da Convexidade P_3 em Grafos Caterpillar
diretamente em uma janela interativa do Matplotlib, clicando nos nós com o mouse.
Recursos:
- Jogar como Alice (1º lance, Vermelho) ou Bob (2º lance, Azul)
- Resposta instantânea da IA Minimax com Poda Alfa-Beta
- Destaque visual para jogadas de Alice, Bob e nós propagados pelo Fecho P_3
- Atalhos de teclado: [R] Reiniciar, [Z] Desfazer lance, [D] Pedir Dica
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
    """
    Calcula as coordenadas 2D (x, y) estéticas para desenhar o caterpillar:
    - Espinha: alinhada horizontalmente no eixo y = 0.
    - Pernas: distribuídas simetricamente acima (y > 0) e abaixo (y < 0).
    """
    pos: dict[int, tuple[float, float]] = {}
    k = cat.k
    espacamento_x = 1.8

    # 1. Posiciona nós da espinha
    for i in range(k):
        pos[i] = (i * espacamento_x, 0.0)

    # 2. Agrupa pernas por nó pai da espinha
    pernas_por_pai: dict[int, list[int]] = {i: [] for i in range(k)}
    for i in range(k):
        # Pernas são vértices de índice >= k conectados ao nó da espinha i
        for u in cat.adj[i]:
            if u >= k:
                pernas_por_pai[i].append(u)

    # 3. Posiciona as pernas
    for pai, pernas in pernas_por_pai.items():
        qtd = len(pernas)
        if qtd == 0:
            continue
        px, _ = pos[pai]

        if qtd == 1:
            pos[pernas[0]] = (px, 1.2)
        elif qtd == 2:
            pos[pernas[0]] = (px, 1.2)
            pos[pernas[1]] = (px, -1.2)
        else:
            metade_cima = math.ceil(qtd / 2)
            metade_baixo = qtd - metade_cima
            largura = 0.6

            # Acima
            dx_c = largura / max(1, metade_cima - 1) if metade_cima > 1 else 0.0
            x0_c = px - (largura / 2 if metade_cima > 1 else 0.0)
            for idx in range(metade_cima):
                pos[pernas[idx]] = (x0_c + idx * dx_c, 1.2)

            # Abaixo
            dx_b = largura / max(1, metade_baixo - 1) if metade_baixo > 1 else 0.0
            x0_b = px - (largura / 2 if metade_baixo > 1 else 0.0)
            for idx in range(metade_baixo):
                pos[pernas[metade_cima + idx]] = (x0_b + idx * dx_b, -1.2)

    return pos


def jogar_caterpillar_matplotlib(
    assinatura: Assinatura = (0, 2, 2, 0),
    humano_eh_alice: bool = True,
) -> None:
    """
    Inicia a partida interativa do Jogo do Fecho Convexo P_3 em uma janela Matplotlib.
    
    Parâmetros:
      assinatura: Tupla de pernas por nó da espinha (ex: (0, 2, 2, 0)).
      humano_eh_alice: Se True, humano joga como Alice (MAX, 1º a jogar).
                       Se False, humano joga como Bob (MIN, IA inicia).
    """
    cat = Caterpillar(assinatura)
    jogo = JogoFechoConvexoP3(cat.adj, cat.rotulos)
    solver = Minimax(jogo, memoizacao=True)
    pos = calcular_layout(cat)

    # Configuração dos jogadores
    humano = MAX if humano_eh_alice else MIN
    ia = MIN if humano_eh_alice else MAX
    humano_nome = "Alice" if humano == MAX else "Bob"
    ia_nome = "Bob" if ia == MIN else "Alice"

    # Paleta de Cores
    COR_LIVRE = "#F8FAFC"
    COR_LIVRE_BORDA = "#64748B"
    COR_ALICE = "#EF4444"       # Vermelho
    COR_BOB = "#0EA5E9"         # Azul
    COR_PROPAGADO = "#F59E0B"   # Amarelo/Ouro
    COR_ARESTA_INATIVA = "#CBD5E1"
    COR_ARESTA_ATIVA = "#1E293B"

    # Estado da partida
    dados = {
        "S": 0,
        "turno": 1,
        "fim": False,
        "mensagem": "",
        "dica": None,
        "jogadas_alice": set(),
        "jogadas_bob": set(),
        "propagados": set(),
        "historico": [],  # Pilha para funcionalidade de desfazer (Undo)
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.canvas.manager.set_window_title(f"P3-Game: {humano_nome} (Você) vs {ia_nome} (IA Minimax)")

    def salvar_snapshot():
        """Salva o estado atual na pilha de histórico."""
        dados["historico"].append({
            "S": dados["S"],
            "turno": dados["turno"],
            "fim": dados["fim"],
            "mensagem": dados["mensagem"],
            "dica": dados["dica"],
            "jogadas_alice": set(dados["jogadas_alice"]),
            "jogadas_bob": set(dados["jogadas_bob"]),
            "propagados": set(dados["propagados"]),
        })

    def desfazer_lance():
        """Desfaz o último turno jogado (humano + resposta da IA)."""
        if len(dados["historico"]) > 1:
            # Se a IA começou, mantém pelo menos o primeiro lance da IA
            limite = 2 if not humano_eh_alice else 1
            if len(dados["historico"]) >= limite:
                anterior = dados["historico"].pop()
                # Restaura estado
                for k, v in anterior.items():
                    dados[k] = v
                dados["mensagem"] = "↩ Lance desfeito! Sua vez."
                dados["fim"] = False
                redesenhar()

    def obter_melhor_lance_ia(S_atual: int) -> int:
        """Calcula o melhor lance para a IA usando o Minimax."""
        sucessores = jogo.jogadas(S_atual)
        if not sucessores:
            return -1

        # Procura um lance que garanta a vitória da IA
        for v, prox in sucessores:
            if solver.valor(prox, humano) == ia:
                return v

        # Se todas levarem à derrota, escolhe o primeiro lance disponível
        return sucessores[0][0]

    def redesenhar():
        """Desenha todo o cenário gráfico no eixo Matplotlib."""
        ax.clear()
        ax.axis("off")

        S = dados["S"]
        total_infectados = bin(S).count("1")

        # 1. Desenha arestas
        for u in range(cat.n):
            for v in cat.adj[u]:
                if u < v:
                    ativa = ((S >> u) & 1) and ((S >> v) & 1)
                    cor_aresta = COR_ARESTA_ATIVA if ativa else COR_ARESTA_INATIVA
                    lw = 3.2 if ativa else 1.8
                    ax.plot(
                        [pos[u][0], pos[v][0]],
                        [pos[u][1], pos[v][1]],
                        color=cor_aresta,
                        linewidth=lw,
                        zorder=1,
                    )

        # 2. Desenha nós (círculos)
        for v in range(cat.n):
            x, y = pos[v]
            infectado = (S >> v) & 1

            if not infectado:
                cor_fill = COR_LIVRE
                cor_borda = COR_LIVRE_BORDA
                lw = 2.0
                texto_cor = "#1E293B"
            elif v in dados["jogadas_alice"]:
                cor_fill = COR_ALICE
                cor_borda = "#7F1D1D"
                lw = 2.5
                texto_cor = "#FFFFFF"
            elif v in dados["jogadas_bob"]:
                cor_fill = COR_BOB
                cor_borda = "#0C4A6E"
                lw = 2.5
                texto_cor = "#FFFFFF"
            elif v in dados["propagados"]:
                cor_fill = COR_PROPAGADO
                cor_borda = "#78350F"
                lw = 2.5
                texto_cor = "#FFFFFF"
            else:
                cor_fill = COR_PROPAGADO
                cor_borda = "#78350F"
                lw = 2.0
                texto_cor = "#FFFFFF"

            # Se for o vértice da dica ativa, destaca com borda dourada pulsante
            if dados["dica"] == v:
                cor_borda = "#10B981"  # Verde esmeralda de recomendação
                lw = 4.0

            raio = 0.28 if v in cat.espinha else 0.24
            circulo = plt.Circle(
                (x, y), raio,
                facecolor=cor_fill,
                edgecolor=cor_borda,
                linewidth=lw,
                zorder=2,
            )
            ax.add_patch(circulo)

            ax.text(
                x, y, cat.rotulos[v],
                ha="center", va="center",
                color=texto_cor,
                fontsize=10,
                fontweight="bold",
                zorder=3,
            )

        # 3. Limites e Proporção
        xs = [x for x, y in pos.values()]
        ys = [y for x, y in pos.values()]
        ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
        ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
        ax.set_aspect("equal", adjustable="datalim")

        # 4. Título e Barra Superior
        status_txt = dados["mensagem"] if dados["mensagem"] else f"Sua vez ({humano_nome})! Clique em um nó livre."
        progresso_txt = f"Infectados: {total_infectados}/{cat.n} ({int(total_infectados / cat.n * 100)}%)"
        
        ax.set_title(
            f"{status_txt}\n[Turno {dados['turno']} | {progresso_txt}]",
            fontsize=12,
            fontweight="bold",
            pad=14,
            color="#0F172A",
        )

        # 5. Legenda na parte inferior
        patches = [
            mpatches.Patch(facecolor=COR_LIVRE, edgecolor=COR_LIVRE_BORDA, label="Livre"),
            mpatches.Patch(facecolor=COR_ALICE, label="Alice"),
            mpatches.Patch(facecolor=COR_BOB, label="Bob"),
            mpatches.Patch(facecolor=COR_PROPAGADO, label="Fecho P3 (Propagado)"),
        ]
        leg = ax.legend(
            handles=patches,
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, -0.15),
            frameon=True,
            facecolor="#F8FAFC",
            edgecolor="#CBD5E1",
            fontsize=9,
        )

        fig.canvas.draw_idle()

    def ao_clicar(event):
        """Gerencia o clique do mouse em um vértice do grafo."""
        if dados["fim"] or event.xdata is None or event.ydata is None:
            return

        # 1. Encontra qual vértice foi clicado
        clicado = None
        for v, (x, y) in pos.items():
            if math.hypot(event.xdata - x, event.ydata - y) <= 0.35:
                if not (dados["S"] >> v) & 1:  # Vértice precisa estar livre
                    clicado = v
                    break

        if clicado is None:
            return  # Clique fora de nós livres

        # Salva snapshot antes do lance
        salvar_snapshot()
        dados["dica"] = None

        # 2. Executa a jogada do HUMANO
        novo_S, props_humano = fecho_p3(cat.adj, dados["S"] | (1 << clicado))
        dados["S"] = novo_S
        if humano == MAX:
            dados["jogadas_alice"].add(clicado)
        else:
            dados["jogadas_bob"].add(clicado)
        for p in props_humano:
            dados["propagados"].add(p)

        props_str = f" (+{len(props_humano)} propagados)" if props_humano else ""
        
        # 3. Verifica se o Humano venceu (completou a infecção)
        if dados["S"] == (1 << cat.n) - 1:
            dados["fim"] = True
            dados["mensagem"] = f"🏆 PARABÉNS! Você ({humano_nome}) jogou em {cat.rotulos[clicado]} e VENCEU a partida!"
            redesenhar()
            return

        # 4. Executa a jogada da IA MINIMAX
        lance_ia = obter_melhor_lance_ia(dados["S"])
        novo_S_ia, props_ia = fecho_p3(cat.adj, dados["S"] | (1 << lance_ia))
        dados["S"] = novo_S_ia
        if ia == MAX:
            dados["jogadas_alice"].add(lance_ia)
        else:
            dados["jogadas_bob"].add(lance_ia)
        for p in props_ia:
            dados["propagados"].add(p)

        dados["turno"] += 1
        props_ia_str = f" (+{len(props_ia)} propagados)" if props_ia else ""

        # 5. Verifica se a IA venceu
        if dados["S"] == (1 << cat.n) - 1:
            dados["fim"] = True
            dados["mensagem"] = f"💀 FIM DE JOGO: A IA ({ia_nome}) jogou em {cat.rotulos[lance_ia]} e VENCEU a partida!"
            redesenhar()
            return

        dados["mensagem"] = f"Você jogou em {cat.rotulos[clicado]}{props_str} | IA jogou em {cat.rotulos[lance_ia]}{props_ia_str}."
        redesenhar()

    def ao_pressionar_tecla(event):
        """Gerencia atalhos de teclado: R (Reiniciar), Z (Desfazer), D (Dica)."""
        tecla = event.key.lower() if event.key else ""

        if tecla == "r":
            # Reiniciar partida
            reiniciar_partida()
        elif tecla in ("z", "u"):
            # Desfazer lance
            desfazer_lance()
        elif tecla == "d":
            # Pedir Dica
            if not dados["fim"]:
                sucessores = jogo.jogadas(dados["S"])
                vencedores = [v for v, prox in sucessores if solver.valor(prox, ia) == humano]
                if vencedores:
                    dados["dica"] = vencedores[0]
                    rot = cat.rotulos[vencedores[0]]
                    dados["mensagem"] = f"💡 DICA: Jogada ótima recomendada no vértice [{rot}] (destacado em verde)."
                else:
                    dados["mensagem"] = "💡 DICA: Sob jogo perfeito da IA a posição é difícil, jogue com cautela!"
                redesenhar()

    def reiniciar_partida():
        """Reseta o tabuleiro e inicia uma nova partida."""
        dados["S"] = 0
        dados["turno"] = 1
        dados["fim"] = False
        dados["mensagem"] = ""
        dados["dica"] = None
        dados["jogadas_alice"].clear()
        dados["jogadas_bob"].clear()
        dados["propagados"].clear()
        dados["historico"].clear()

        # Se o Humano for Bob, a IA começa jogando como Alice
        if not humano_eh_alice:
            salvar_snapshot()
            lance_ia = obter_melhor_lance_ia(0)
            novo_S, props = fecho_p3(cat.adj, 1 << lance_ia)
            dados["S"] = novo_S
            dados["jogadas_alice"].add(lance_ia)
            for p in props:
                dados["propagados"].add(p)
            dados["mensagem"] = f"🤖 IA iniciou como Alice em {cat.rotulos[lance_ia]}. Sua vez (Bob)!"
        else:
            dados["mensagem"] = "Sua vez! Você joga como Alice (1º lance). Clique em um nó livre."

        salvar_snapshot()
        redesenhar()

    # Registra os manipuladores de eventos do Matplotlib
    fig.canvas.mpl_connect("button_press_event", ao_clicar)
    fig.canvas.mpl_connect("key_press_event", ao_pressionar_tecla)

    # Inicia a partida
    reiniciar_partida()
    plt.tight_layout()
    plt.show()
