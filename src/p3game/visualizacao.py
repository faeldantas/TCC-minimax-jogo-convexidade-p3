"""
Camada de Visualização Gráfica do Jogo da Convexidade P_3 em Caterpillars.

Permite visualizar graficamente:
1. A topologia do caterpillar (espinha no eixo horizontal, pernas acima e abaixo).
2. A evolução da partida lance a lance (quem jogou o quê).
3. As propagações em cascata do fecho P_3 destacadas visualmente.
4. Exportação de painéis estáticos PNG (para relatórios do TCC).
5. Visualizador interativo em Matplotlib com botões e sliders.
6. Visualizador interativo autônomo em HTML5/SVG que roda no navegador.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, Slider
import networkx as nx

from .grafos import Caterpillar


def calcular_layout_caterpillar(cat: Caterpillar) -> dict[int, tuple[float, float]]:
    """
    Gera coordenadas 2D (x, y) planas e estéticas para desenhar o caterpillar:
    - Vértices da espinha ficam no eixo y = 0, alinhados horizontalmente.
    - Pernas (folhas) ficam distribuídas acima (y > 0) e abaixo (y < 0).
    """
    pos: dict[int, tuple[float, float]] = {}
    k = cat.k

    # Posiciona a espinha
    espacamento_x = 1.8
    for i in range(k):
        pos[i] = (i * espacamento_x, 0.0)

    # Agrupa pernas por pai
    pernas_por_pai: dict[int, list[int]] = {i: [] for i in range(k)}
    for perna, pai in cat.pai_da_perna.items():
        pernas_por_pai[pai].append(perna)

    # Posiciona as pernas
    for pai, pernas in pernas_por_pai.items():
        qtd = len(pernas)
        if qtd == 0:
            continue
        px, py = pos[pai]

        if qtd == 1:
            pos[pernas[0]] = (px, 1.2)
        elif qtd == 2:
            pos[pernas[0]] = (px, 1.2)
            pos[pernas[1]] = (px, -1.2)
        else:
            # Distribui metade acima e metade abaixo
            metade_cima = math.ceil(qtd / 2)
            metade_baixo = qtd - metade_cima

            # Cima
            largura = 0.6
            dx_cima = largura / max(1, metade_cima - 1) if metade_cima > 1 else 0.0
            x_ini_cima = px - (largura / 2 if metade_cima > 1 else 0.0)
            for idx in range(metade_cima):
                u = pernas[idx]
                pos[u] = (x_ini_cima + idx * dx_cima, 1.2)

            # Baixo
            dx_baixo = largura / max(1, metade_baixo - 1) if metade_baixo > 1 else 0.0
            x_ini_baixo = px - (largura / 2 if metade_baixo > 1 else 0.0)
            for idx in range(metade_baixo):
                u = pernas[metade_cima + idx]
                pos[u] = (x_ini_baixo + idx * dx_baixo, -1.2)

    return pos


def reconstruir_historico_partida(cat: Caterpillar, variacao_principal: list[dict]) -> list[dict]:
    """
    Reconstrói o estado acumulado de cada vértice a cada turno da variação principal.
    Turno 0: Estado inicial vazio.
    Turno 1..T: Vértices jogados por Alice/Bob e vértices propagados.
    """
    rot_para_idx = {r: i for i, r in enumerate(cat.rotulos)}

    historico: list[dict] = []

    # Turno 0: Inicial
    estado_atual = {
        "turno": 0,
        "jogador": "Início",
        "jogada_rotulo": None,
        "jogada_idx": None,
        "propagados_rotulos": [],
        "propagados_indices": [],
        "jogadas_alice": set(),
        "jogadas_bob": set(),
        "todos_propagados": set(),
        "todos_infectados": set(),
    }
    historico.append(estado_atual)

    alice_set = set()
    bob_set = set()
    prop_set = set()
    inf_set = set()

    for p in variacao_principal:
        t = p["turno"]
        jog = p["jogador"]
        j_rot = p["jogada"]
        j_idx = rot_para_idx[j_rot]

        props_rot = p.get("propagados", [])
        props_idx = [rot_para_idx[r] for r in props_rot]

        if jog == "Alice":
            alice_set.add(j_idx)
        else:
            bob_set.add(j_idx)

        inf_set.add(j_idx)
        for u in props_idx:
            prop_set.add(u)
            inf_set.add(u)

        historico.append({
            "turno": t,
            "jogador": jog,
            "jogada_rotulo": j_rot,
            "jogada_idx": j_idx,
            "propagados_rotulos": list(props_rot),
            "propagados_indices": list(props_idx),
            "jogadas_alice": set(alice_set),
            "jogadas_bob": set(bob_set),
            "todos_propagados": set(prop_set),
            "todos_infectados": set(inf_set),
        })

    return historico


# Paleta de Cores Científica
CORES = {
    "livre_face": "#F8F9FA",
    "livre_borda": "#6C757D",
    "alice_atual": "#D90429",       # Vermelho vibrante
    "alice_passado": "#EF233C",     # Vermelho suave
    "bob_atual": "#0077B6",         # Azul vibrante
    "bob_passado": "#0096C7",       # Azul suave
    "propagado_atual": "#FFB703",   # Amarelo / Ouro vivo
    "propagado_passado": "#FB8500", # Laranja
    "aresta_inativa": "#ADB5BD",
    "aresta_ativa": "#212529",
    "texto_escuro": "#212529",
    "texto_claro": "#FFFFFF",
}


def desenhar_grafo_turno(
    cat: Caterpillar,
    pos: dict[int, tuple[float, float]],
    estado: dict,
    ax: plt.Axes | None = None,
    destacar_atual: bool = True,
    titulo: str | None = None,
) -> plt.Axes:
    """Desenha o grafo caterpillar no estado correspondente a um turno específico."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.clear()
    ax.axis("off")

    G = cat.para_networkx()

    inf_set = estado["todos_infectados"]
    alice_set = estado["jogadas_alice"]
    bob_set = estado["jogadas_bob"]
    prop_set = estado["todos_propagados"]
    j_atual = estado["jogada_idx"] if destacar_atual else None
    props_atuais = set(estado["propagados_indices"]) if destacar_atual else set()

    # 1. Desenha arestas
    arestas_ativas = []
    arestas_inativas = []
    for u, v in G.edges():
        if u in inf_set and v in inf_set:
            arestas_ativas.append((u, v))
        else:
            arestas_inativas.append((u, v))

    nx.draw_networkx_edges(
        G, pos, edgelist=arestas_inativas,
        ax=ax, edge_color=CORES["aresta_inativa"], width=1.5, style="solid"
    )
    if arestas_ativas:
        nx.draw_networkx_edges(
            G, pos, edgelist=arestas_ativas,
            ax=ax, edge_color=CORES["aresta_ativa"], width=2.8, style="solid"
        )

    # 2. Cores dos nós
    cores_nos = []
    cores_borda = []
    larguras_borda = []
    tamanhos_nos = []

    for v in range(cat.n):
        if v not in inf_set:
            cores_nos.append(CORES["livre_face"])
            cores_borda.append(CORES["livre_borda"])
            larguras_borda.append(1.5)
            tamanhos_nos.append(650)
        elif v == j_atual:
            cor = CORES["alice_atual"] if estado["jogador"] == "Alice" else CORES["bob_atual"]
            cores_nos.append(cor)
            cores_borda.append("#000000")
            larguras_borda.append(3.0)
            tamanhos_nos.append(850)
        elif v in props_atuais:
            cores_nos.append(CORES["propagado_atual"])
            cores_borda.append("#D00000")
            larguras_borda.append(2.8)
            tamanhos_nos.append(800)
        elif v in alice_set:
            cores_nos.append(CORES["alice_passado"])
            cores_borda.append("#590d22")
            larguras_borda.append(1.5)
            tamanhos_nos.append(650)
        elif v in bob_set:
            cores_nos.append(CORES["bob_passado"])
            cores_borda.append("#03045e")
            larguras_borda.append(1.5)
            tamanhos_nos.append(650)
        elif v in prop_set:
            cores_nos.append(CORES["propagado_passado"])
            cores_borda.append("#6a040f")
            larguras_borda.append(1.5)
            tamanhos_nos.append(650)
        else:
            cores_nos.append(CORES["livre_face"])
            cores_borda.append(CORES["livre_borda"])
            larguras_borda.append(1.5)
            tamanhos_nos.append(650)

    # Desenha nós
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=cores_nos,
        edgecolors=cores_borda,
        linewidths=larguras_borda,
        node_size=tamanhos_nos,
    )

    # 3. Rótulos dos nós (v1, v2, ...)
    rotulos = {i: cat.rotulos[i] for i in range(cat.n)}
    cores_texto = {
        i: CORES["texto_claro"] if i in inf_set else CORES["texto_escuro"]
        for i in range(cat.n)
    }
    for i, (x, y) in pos.items():
        ax.text(
            x, y, rotulos[i],
            color=cores_texto[i],
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
        )

    # Título do estado
    t = estado["turno"]
    if titulo is not None:
        ax.set_title(titulo, fontsize=12, fontweight="bold", pad=12)
    elif t == 0:
        ax.set_title("Estado Inicial: Grafo Vazio (Nenhum vértice infectado)", fontsize=11, fontweight="bold", pad=8)
    else:
        jog = estado["jogador"]
        lance = estado["jogada_rotulo"]
        props = estado["propagados_rotulos"]
        props_txt = f" | Propagados P3: {', '.join(props)}" if props else " | Sem propagação"
        ax.set_title(
            f"Turno {t}: Jogada de {jog} em {lance}{props_txt} (Infectados: {len(inf_set)}/{cat.n})",
            fontsize=11, fontweight="bold", pad=8
        )

    # Ajusta margens
    xs = [x for x, y in pos.values()]
    ys = [y for x, y in pos.values()]
    margem_x = 0.8
    margem_y = 0.8
    ax.set_xlim(min(xs) - margem_x, max(xs) + margem_x)
    ax.set_ylim(min(ys) - margem_y, max(ys) + margem_y)

    return ax


def gerar_painel_estatico(
    cat: Caterpillar,
    variacao_principal: list[dict],
    caminho_saida: str,
    colunas: int = 2,
) -> str:
    """
    Gera uma imagem de alta resolução com a sequência passo a passo de turnos
    disposta em grade, perfeita para inserção no texto do TCC.
    """
    historico = reconstruir_historico_partida(cat, variacao_principal)
    total_quadros = len(historico)
    linhas = math.ceil(total_quadros / colunas)

    fig, axes = plt.subplots(
        linhas, colunas,
        figsize=(colunas * 6.5, linhas * 3.6),
        squeeze=False,
    )

    pos = calcular_layout_caterpillar(cat)

    for idx, estado in enumerate(historico):
        r = idx // colunas
        c = idx % colunas
        desenhar_grafo_turno(cat, pos, estado, ax=axes[r][c])

    # Oculta subplots vazios no final da grade
    for idx in range(total_quadros, linhas * colunas):
        r = idx // colunas
        c = idx % colunas
        axes[r][c].axis("off")

    # Legenda global
    patches = [
        mpatches.Patch(color=CORES["livre_face"], ec=CORES["livre_borda"], label="Vértice Livre"),
        mpatches.Patch(color=CORES["alice_atual"], label="Jogada de Alice"),
        mpatches.Patch(color=CORES["bob_atual"], label="Jogada de Bob"),
        mpatches.Patch(color=CORES["propagado_atual"], label="Propagação Fecho P3"),
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=4,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=(0.0, 0.04, 1.0, 0.98))
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    fig.savefig(caminho_saida, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return caminho_saida


def abrir_visualizador_interativo(
    cat: Caterpillar,
    variacao_principal: list[dict],
) -> None:
    """
    Abre uma janela gráfica interativa do Matplotlib com controles de navegação
    (Botão Anterior, Próximo e Slider) para avançar e voltar pelos turnos.
    """
    historico = reconstruir_historico_partida(cat, variacao_principal)
    pos = calcular_layout_caterpillar(cat)
    max_t = len(historico) - 1

    fig, ax = plt.subplots(figsize=(11, 6))
    plt.subplots_adjust(bottom=0.22)

    # Estado inicial
    idx_atual = [0]

    desenhar_grafo_turno(cat, pos, historico[0], ax=ax)

    # Widgets
    ax_slider = plt.axes([0.20, 0.08, 0.60, 0.03])
    slider = Slider(ax_slider, "Turno", 0, max_t, valinit=0, valstep=1, valfmt="%0.0f")

    ax_ant = plt.axes([0.20, 0.02, 0.12, 0.04])
    btn_ant = Button(ax_ant, "◀ Anterior")

    ax_prox = plt.axes([0.68, 0.02, 0.12, 0.04])
    btn_prox = Button(ax_prox, "Próximo ▶")

    def atualizar(val):
        t = int(round(slider.val))
        idx_atual[0] = t
        desenhar_grafo_turno(cat, pos, historico[t], ax=ax)
        fig.canvas.draw_idle()

    def anterior(event):
        if idx_atual[0] > 0:
            slider.set_val(idx_atual[0] - 1)

    def proximo(event):
        if idx_atual[0] < max_t:
            slider.set_val(idx_atual[0] + 1)

    slider.on_changed(atualizar)
    btn_ant.on_clicked(anterior)
    btn_prox.on_clicked(proximo)

    # Legenda
    patches = [
        mpatches.Patch(color=CORES["livre_face"], ec=CORES["livre_borda"], label="Livre"),
        mpatches.Patch(color=CORES["alice_atual"], label="Alice"),
        mpatches.Patch(color=CORES["bob_atual"], label="Bob"),
        mpatches.Patch(color=CORES["propagado_atual"], label="Propagado P3"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9)

    plt.show()


def gerar_html_interativo(
    cat: Caterpillar,
    variacao_principal: list[dict],
    caminho_html: str,
    titulo: str = "Visualizador Interativo: Jogo da Convexidade P3 em Caterpillars",
) -> str:
    """
    Gera uma aplicação web autônoma (HTML5 + SVG interativo) que roda
    diretamente em qualquer navegador sem necessidade de servidor ou internet.
    """
    historico = reconstruir_historico_partida(cat, variacao_principal)
    pos = calcular_layout_caterpillar(cat)

    # Normaliza coordenadas para o SVG
    xs = [x for x, y in pos.values()]
    ys = [y for x, y in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    largura_svg = 900
    altura_svg = 400
    margem = 60

    dx = max_x - min_x if max_x > min_x else 1.0
    dy = max_y - min_y if max_y > min_y else 1.0

    nos_svg = []
    for v in range(cat.n):
        x, y = pos[v]
        svg_x = margem + ((x - min_x) / dx) * (largura_svg - 2 * margem)
        svg_y = altura_svg - (margem + ((y - min_y) / dy) * (altura_svg - 2 * margem))
        nos_svg.append({
            "id": v,
            "rotulo": cat.rotulos[v],
            "x": round(svg_x, 1),
            "y": round(svg_y, 1),
            "eh_espinha": v in cat.espinha,
        })

    G = cat.para_networkx()
    arestas_svg = []
    for u, v in G.edges():
        arestas_svg.append({"u": u, "v": v})

    # Prepara JSON com o histórico da partida
    dados_partida = {
        "assinatura": list(cat.assinatura),
        "n": cat.n,
        "k": cat.k,
        "nos": nos_svg,
        "arestas": arestas_svg,
        "historico": [
            {
                "turno": h["turno"],
                "jogador": h["jogador"],
                "jogada_rotulo": h["jogada_rotulo"],
                "jogada_idx": h["jogada_idx"],
                "propagados_rotulos": h["propagados_rotulos"],
                "propagados_indices": h["propagados_indices"],
                "jogadas_alice": list(h["jogadas_alice"]),
                "jogadas_bob": list(h["jogadas_bob"]),
                "todos_propagados": list(h["todos_propagados"]),
                "todos_infectados": list(h["todos_infectados"]),
            }
            for h in historico
        ],
    }

    html_template = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{titulo}</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --alice: #ef4444;
      --bob: #0ea5e9;
      --prop: #f59e0b;
      --livre: #475569;
      --livre-fill: #1e293b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 24px 16px;
      min-height: 100vh;
    }}
    .container {{
      max-width: 1000px;
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }}
    header {{
      background: var(--card-bg);
      padding: 20px 24px;
      border-radius: 12px;
      border: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    h1 {{ font-size: 1.25rem; font-weight: 700; color: #fff; }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 600;
      background: #334155;
    }}
    .stage {{
      background: var(--card-bg);
      border-radius: 12px;
      border: 1px solid var(--border);
      padding: 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
      position: relative;
    }}
    svg {{
      width: 100%;
      height: 400px;
      background: #0f172a;
      border-radius: 8px;
      border: 1px solid #1e293b;
    }}
    .edge {{ stroke: #334155; stroke-width: 2.5; transition: stroke 0.3s; }}
    .edge.active {{ stroke: #f8fafc; stroke-width: 3.5; }}
    .node-circle {{
      cursor: pointer;
      transition: fill 0.3s, stroke 0.3s, r 0.2s;
    }}
    .node-text {{
      font-size: 11px;
      font-weight: bold;
      text-anchor: middle;
      dominant-baseline: central;
      pointer-events: none;
      user-select: none;
    }}
    .controls {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 14px;
      margin-top: 16px;
      width: 100%;
      flex-wrap: wrap;
    }}
    button {{
      background: #334155;
      color: #fff;
      border: 1px solid var(--border);
      padding: 8px 16px;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    button:hover:not(:disabled) {{ background: #475569; }}
    button:disabled {{ opacity: 0.4; cursor: not-allowed; }}
    .btn-play {{ background: #2563eb; border-color: #3b82f6; }}
    .btn-play:hover {{ background: #1d4ed8; }}
    .slider-wrap {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex: 1;
      min-width: 200px;
      max-width: 400px;
    }}
    input[type=range] {{
      flex: 1;
      height: 6px;
      background: #334155;
      border-radius: 3px;
      outline: none;
      cursor: pointer;
    }}
    .info-card {{
      background: var(--card-bg);
      border-radius: 12px;
      border: 1px solid var(--border);
      padding: 20px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
    }}
    .info-item h3 {{ font-size: 0.8rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px; }}
    .info-item p {{ font-size: 1.1rem; font-weight: 700; }}
    .legend {{
      display: flex;
      justify-content: center;
      gap: 20px;
      margin-top: 12px;
      flex-wrap: wrap;
      font-size: 0.85rem;
    }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; }}
    .legend-box {{ width: 14px; height: 14px; border-radius: 50%; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>{titulo}</h1>
        <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 4px;">
          Assinatura L = {dados_partida['assinatura']} | Vértices n = {dados_partida['n']} | Espinha k = {dados_partida['k']}
        </p>
      </div>
      <div id="vencedor-badge" class="badge">Variação Principal</div>
    </header>

    <div class="stage">
      <svg id="svg-canvas" viewBox="0 0 {largura_svg} {altura_svg}"></svg>

      <div class="legend">
        <div class="legend-item"><div class="legend-box" style="background: var(--livre-fill); border: 2px solid var(--livre);"></div> Livre</div>
        <div class="legend-item"><div class="legend-box" style="background: var(--alice);"></div> Jogada de Alice</div>
        <div class="legend-item"><div class="legend-box" style="background: var(--bob);"></div> Jogada de Bob</div>
        <div class="legend-item"><div class="legend-box" style="background: var(--prop);"></div> Fecho P3 (Propagado)</div>
      </div>

      <div class="controls">
        <button id="btn-prev" onclick="mudarTurno(-1)">◀ Anterior</button>
        <button id="btn-play" class="btn-play" onclick="alternarPlay()">▶ Reproduzir</button>
        <button id="btn-next" onclick="mudarTurno(1)">Próximo ▶</button>
        <div class="slider-wrap">
          <span style="font-size: 0.85rem; font-weight: bold;">Turno:</span>
          <input type="range" id="slider" min="0" max="{len(historico) - 1}" value="0" oninput="irParaTurno(parseInt(this.value))">
          <span id="label-turno" style="font-size: 0.85rem; font-weight: bold; width: 30px;">0</span>
        </div>
      </div>
    </div>

    <div class="info-card">
      <div class="info-item">
        <h3>Turno Atual</h3>
        <p id="info-turno">0 / {len(historico) - 1}</p>
      </div>
      <div class="info-item">
        <h3>Jogador da Vez</h3>
        <p id="info-jogador">---</p>
      </div>
      <div class="info-item">
        <h3>Vértice Escolhido</h3>
        <p id="info-lance">---</p>
      </div>
      <div class="info-item">
        <h3>Propagações P3 no Lance</h3>
        <p id="info-props">---</p>
      </div>
      <div class="info-item">
        <h3>Grafo Infectado</h3>
        <p id="info-progresso">0% (0/{dados_partida['n']})</p>
      </div>
    </div>
  </div>

  <script>
    const DADOS = {json.dumps(dados_partida, ensure_ascii=False)};
    let turnoAtual = 0;
    let timerPlay = null;

    const svg = document.getElementById("svg-canvas");

    function renderizarGrafo() {{
      svg.innerHTML = "";
      const estado = DADOS.historico[turnoAtual];

      const infSet = new Set(estado.todos_infectados);
      const aliceSet = new Set(estado.jogadas_alice);
      const bobSet = new Set(estado.jogadas_bob);
      const propSet = new Set(estado.todos_propagados);
      const jAtual = estado.jogada_idx;
      const propsAtuais = new Set(estado.propagados_indices);

      // 1. Arestas
      DADOS.arestas.forEach(a => {{
        const u = DADOS.nos[a.u];
        const v = DADOS.nos[a.v];
        const ativa = infSet.has(a.u) && infSet.has(a.v);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", u.x);
        line.setAttribute("y1", u.y);
        line.setAttribute("x2", v.x);
        line.setAttribute("y2", v.y);
        line.setAttribute("class", "edge" + (ativa ? " active" : ""));
        svg.appendChild(line);
      }});

      // 2. Nós
      DADOS.nos.forEach(n => {{
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");

        let corFill = "var(--livre-fill)";
        let corBorda = "var(--livre)";
        let raio = n.eh_espinha ? 18 : 15;
        let larguraBorda = 2;

        if (n.id === jAtual) {{
          corFill = estado.jogador === "Alice" ? "var(--alice)" : "var(--bob)";
          corBorda = "#ffffff";
          larguraBorda = 3.5;
          raio += 4;
        }} else if (propsAtuais.has(n.id)) {{
          corFill = "var(--prop)";
          corBorda = "#fef08a";
          larguraBorda = 3.5;
          raio += 3;
        }} else if (aliceSet.has(n.id)) {{
          corFill = "var(--alice)";
          corBorda = "#7f1d1d";
        }} else if (bobSet.has(n.id)) {{
          corFill = "var(--bob)";
          corBorda = "#0c4a6e";
        }} else if (propSet.has(n.id)) {{
          corFill = "var(--prop)";
          corBorda = "#78350f";
        }}

        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", n.x);
        circle.setAttribute("cy", n.y);
        circle.setAttribute("r", raio);
        circle.setAttribute("fill", corFill);
        circle.setAttribute("stroke", corBorda);
        circle.setAttribute("stroke-width", larguraBorda);
        circle.setAttribute("class", "node-circle");

        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", n.x);
        text.setAttribute("y", n.y);
        text.setAttribute("fill", infSet.has(n.id) ? "#ffffff" : "#94a3b8");
        text.setAttribute("class", "node-text");
        text.textContent = n.rotulo;

        g.appendChild(circle);
        g.appendChild(text);
        svg.appendChild(g);
      }});

      // Atualiza card de informações
      document.getElementById("slider").value = turnoAtual;
      document.getElementById("label-turno").textContent = turnoAtual;
      document.getElementById("info-turno").textContent = `${{turnoAtual}} / ${{DADOS.historico.length - 1}}`;

      if (turnoAtual === 0) {{
        document.getElementById("info-jogador").textContent = "Alice (Início)";
        document.getElementById("info-lance").textContent = "---";
        document.getElementById("info-props").textContent = "Nenhuma";
      }} else {{
        document.getElementById("info-jogador").textContent = estado.jogador;
        document.getElementById("info-lance").textContent = estado.jogada_rotulo;
        document.getElementById("info-props").textContent = estado.propagados_rotulos.length > 0
          ? estado.propagados_rotulos.join(", ")
          : "Nenhuma";
      }}

      const pct = Math.round((estado.todos_infectados.length / DADOS.n) * 100);
      document.getElementById("info-progresso").textContent = `${{pct}}% (${{estado.todos_infectados.length}}/${{DADOS.n}})`;

      document.getElementById("btn-prev").disabled = (turnoAtual === 0);
      document.getElementById("btn-next").disabled = (turnoAtual === DADOS.historico.length - 1);
    }}

    function irParaTurno(t) {{
      turnoAtual = Math.max(0, Math.min(DADOS.historico.length - 1, t));
      renderizarGrafo();
    }}

    function mudarTurno(delta) {{
      irParaTurno(turnoAtual + delta);
    }}

    function alternarPlay() {{
      const btn = document.getElementById("btn-play");
      if (timerPlay) {{
        clearInterval(timerPlay);
        timerPlay = null;
        btn.textContent = "▶ Reproduzir";
      }} else {{
        btn.textContent = "⏸ Pausar";
        if (turnoAtual === DADOS.historico.length - 1) {{
          irParaTurno(0);
        }}
        timerPlay = setInterval(() => {{
          if (turnoAtual < DADOS.historico.length - 1) {{
            mudarTurno(1);
          }} else {{
            alternarPlay();
          }}
        }}, 900);
      }}
    }}

    renderizarGrafo();
  </script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(caminho_html), exist_ok=True)
    with open(caminho_html, "w", encoding="utf-8") as f:
        f.write(html_template)

    return caminho_html


def jogar_contra_ia_terminal(cat: Caterpillar, humano_eh_alice: bool = True) -> None:
    """
    Permite ao usuário jogar interativamente contra o algoritmo Minimax no terminal.
    Alice sempre começa jogando.
    """
    from .convexidade import fecho_p3, para_mascara, de_mascara
    from .jogo import JogoFechoConvexoP3Caterpillar
    from .minimax import Minimax, MAX, MIN

    print("\n" + "=" * 80)
    print("JOGO DA CONVEXIDADE P_3 -- JOGADOR VS IA (MINIMAX)")
    print("=" * 80)
    print(f"Grafo Caterpillar: Assinatura L = {cat.assinatura}")
    print(f"Total de vértices: n = {cat.n} (Rótulos: {cat.rotulos[0]} até {cat.rotulos[-1]})")
    print(f"Convenção Normal: Vence quem fizer o ÚLTIMO lance (infectando todo o grafo).")

    humano_nome = "Alice (MAX, +1)" if humano_eh_alice else "Bob (MIN, -1)"
    ia_nome = "Bob (MIN, -1)" if humano_eh_alice else "Alice (MAX, +1)"
    print(f"Você está jogando como: {humano_nome}")
    print(f"A máquina (IA) joga como: {ia_nome}")
    print("Comandos especiais: digite 'dica' para sugestão ótima, ou 'sair' para encerrar.")
    print("=" * 80 + "\n")

    S = 0
    turno = 1
    jogador_atual = MAX  # Alice sempre inicia

    rot_para_idx = {r: i for i, r in enumerate(cat.rotulos)}
    completo = (1 << cat.n) - 1

    historico_lances = []

    while S != completo:
        vez_do_humano = (jogador_atual == MAX and humano_eh_alice) or (jogador_atual == MIN and not humano_eh_alice)
        nome_vez = "Alice" if jogador_atual == MAX else "Bob"
        papel = "(Você)" if vez_do_humano else "(IA Minimax)"

        livres = [cat.rotulos[v] for v in range(cat.n) if not (S >> v) & 1]
        infectados = [cat.rotulos[v] for v in range(cat.n) if (S >> v) & 1]

        print(f"--- TURNO {turno}: Vez de {nome_vez} {papel} ---")
        print(f"Vértices infectados ({len(infectados)}/{cat.n}): {', '.join(infectados) if infectados else 'Nenhum'}")
        print(f"Vértices livres ({len(livres)}): {', '.join(livres)}")

        if vez_do_humano:
            # Entrada do usuário
            while True:
                entrada = input(f"\n[{nome_vez}] Escolha um vértice livre para infectar: ").strip()
                if entrada.lower() == "sair":
                    print("\nPartida cancelada pelo usuário.")
                    return
                if entrada.lower() == "dica":
                    # Calcula jogadas ótimas
                    jogo = JogoFechoConvexoP3Caterpillar(cat)
                    solver = Minimax(jogo, corte=True, memoizacao=True, simetria=True)
                    outro = -jogador_atual
                    vencedoras = [
                        cat.rotulos[j] for j, filho in jogo.jogadas(S)
                        if solver.valor(filho, outro) == jogador_atual
                    ]
                    if vencedoras:
                        print(f"💡 DICA: As jogadas vencedoras da posição atual são: {', '.join(vencedoras)}")
                    else:
                        print("💡 DICA: Sob jogo perfeito do oponente, a posição é difícil, mas tente manter a espinha equilibrada!")
                    continue

                # Normaliza rótulo (ex: 'v1' ou '1')
                escolhido_rot = entrada if entrada.startswith("v") else f"v{entrada}"
                if escolhido_rot in rot_para_idx and not (S >> rot_para_idx[escolhido_rot]) & 1:
                    j_idx = rot_para_idx[escolhido_rot]
                    break
                print(f"❌ Vértice inválido ou já infectado. Escolha entre: {', '.join(livres)}")
        else:
            # Lance da IA Minimax
            print("\n🤖 IA calculando jogada ótima...")
            jogo = JogoFechoConvexoP3Caterpillar(cat)
            solver = Minimax(jogo, corte=True, memoizacao=True, simetria=True)
            jogadas_legais = jogo.jogadas(S)

            # Procura jogada que mantém o valor do jogador IA
            melhor_j = None
            for j, filho in jogadas_legais:
                if solver.valor(filho, -jogador_atual) == jogador_atual:
                    melhor_j = j
                    break
            if melhor_j is None:
                melhor_j = jogadas_legais[0][0]

            j_idx = melhor_j
            escolhido_rot = cat.rotulos[j_idx]
            print(f"🤖 IA jogou em: {escolhido_rot}")

        # Aplica fecho P_3
        novo_S, propagados = fecho_p3(cat.adj, S | (1 << j_idx))
        props_rot = [cat.rotulos[p] for p in propagados]

        if props_rot:
            print(f"⚡ Efeito Cascata Fecho P_3: {escolhido_rot} infectou automaticamente por propagação: {', '.join(props_rot)}")
        else:
            print(f"Sem propagação adicional neste lance.")

        historico_lances.append({
            "turno": turno,
            "jogador": f"{nome_vez} {papel}",
            "jogada": escolhido_rot,
            "propagados": props_rot,
        })

        S = novo_S
        print("-" * 80 + "\n")

        if S == completo:
            print("=" * 80)
            if vez_do_humano:
                print(f"🎉 PARABÉNS! Você realizou o último lance válido em {escolhido_rot} e VENCEU A MÁQUINA!")
            else:
                print(f"💀 FIM DE JOGO! A IA realizou o último lance válido em {escolhido_rot} e venceu a partida.")
            print("=" * 80)
            print("\nTrilha completa da partida:")
            for p in historico_lances:
                prop_str = f" (+ {', '.join(p['propagados'])})" if p['propagados'] else ""
                print(f"  Turno {p['turno']:2d} | {p['jogador']:<20}: jogou {p['jogada']}{prop_str}")
            print()
            return

        jogador_atual = -jogador_atual
        turno += 1


def gerar_interface_jogavel_html(
    caminho_html: str,
    cat_inicial: Caterpillar | None = None,
) -> str:
    """
    Gera uma aplicação web autônoma completa (HTML5 + SVG interativo) que permite
    ao usuário jogar contra o algoritmo Minimax diretamente no navegador.
    Inclui motor Minimax embutido em JavaScript com fecho P3, dicas e múltiplos grafos.
    """
    if cat_inicial is None:
        cat_inicial = Caterpillar((2, 2, 2, 2))  # Caterpillar cheio m=4 padrão

    pos = calcular_layout_caterpillar(cat_inicial)
    xs = [x for x, y in pos.values()]
    ys = [y for x, y in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    largura_svg = 950
    altura_svg = 420
    margem = 60

    dx = max_x - min_x if max_x > min_x else 1.0
    dy = max_y - min_y if max_y > min_y else 1.0

    nos_svg = []
    for v in range(cat_inicial.n):
        x, y = pos[v]
        svg_x = margem + ((x - min_x) / dx) * (largura_svg - 2 * margem)
        svg_y = altura_svg - (margem + ((y - min_y) / dy) * (altura_svg - 2 * margem))
        nos_svg.append({
            "id": v,
            "rotulo": cat_inicial.rotulos[v],
            "x": round(svg_x, 1),
            "y": round(svg_y, 1),
            "eh_espinha": v in cat_inicial.espinha,
        })

    G = cat_inicial.para_networkx()
    arestas_svg = [{"u": u, "v": v} for u, v in G.edges()]

    adj_json = [cat_inicial.adj[i] for i in range(cat_inicial.n)]
    rotulos_json = list(cat_inicial.rotulos)

    html_template = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Jogar Contra a IA -- Jogo da Convexidade P3 em Caterpillars</title>
  <style>
    :root {{
      --bg: #0b132b;
      --card-bg: #1c2541;
      --border: #3a506b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --alice: #e63946;
      --bob: #00b4d8;
      --prop: #ffb703;
      --prop-glow: rgba(255, 183, 3, 0.45);
      --dica: #10b981;
      --livre: #475569;
      --livre-fill: #1e293b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px 16px;
      min-height: 100vh;
    }}
    .container {{
      max-width: 1100px;
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    header {{
      background: var(--card-bg);
      padding: 18px 24px;
      border-radius: 12px;
      border: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    h1 {{ font-size: 1.3rem; font-weight: 700; color: #fff; }}
    .badge {{
      display: inline-block;
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 700;
      background: #334155;
    }}
    .badge-alice {{ background: var(--alice); color: white; }}
    .badge-bob {{ background: var(--bob); color: #0b132b; }}
    .panel-config {{
      background: var(--card-bg);
      border-radius: 12px;
      border: 1px solid var(--border);
      padding: 14px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
      font-size: 0.9rem;
    }}
    select, input[type=text] {{
      background: #0f172a;
      color: #fff;
      border: 1px solid var(--border);
      padding: 6px 12px;
      border-radius: 6px;
      outline: none;
      font-size: 0.85rem;
    }}
    .stage {{
      background: var(--card-bg);
      border-radius: 12px;
      border: 1px solid var(--border);
      padding: 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
      position: relative;
    }}
    svg {{
      width: 100%;
      height: 420px;
      background: #0b132b;
      border-radius: 8px;
      border: 1px solid #1c2541;
      user-select: none;
    }}
    .edge {{ stroke: #334155; stroke-width: 2.5; transition: stroke 0.3s; }}
    .edge.active {{ stroke: #f8fafc; stroke-width: 3.5; }}
    .node-circle {{
      cursor: pointer;
      transition: fill 0.25s, stroke 0.25s, r 0.2s, filter 0.2s;
    }}
    .node-circle:hover:not(.infected) {{
      filter: drop-shadow(0 0 8px #38bdf8);
      stroke: #38bdf8;
      stroke-width: 3;
    }}
    .node-circle.dica {{
      stroke: var(--dica) !important;
      stroke-width: 4 !important;
      filter: drop-shadow(0 0 10px #10b981) !important;
    }}
    .node-text {{
      font-size: 11px;
      font-weight: bold;
      text-anchor: middle;
      dominant-baseline: central;
      pointer-events: none;
      user-select: none;
    }}
    .banner-status {{
      width: 100%;
      padding: 12px 18px;
      border-radius: 8px;
      margin-top: 14px;
      font-size: 1rem;
      font-weight: 600;
      text-align: center;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      background: #1e293b;
      border: 1px solid var(--border);
    }}
    .banner-status.vitoria {{ background: rgba(16, 185, 129, 0.2); border-color: #10b981; color: #34d399; }}
    .banner-status.derrota {{ background: rgba(239, 68, 68, 0.2); border-color: #ef4444; color: #f87171; }}
    .controls {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 12px;
      margin-top: 16px;
      width: 100%;
      flex-wrap: wrap;
    }}
    button {{
      background: #334155;
      color: #fff;
      border: 1px solid var(--border);
      padding: 8px 16px;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    button:hover:not(:disabled) {{ background: #475569; }}
    button:disabled {{ opacity: 0.4; cursor: not-allowed; }}
    .btn-action {{ background: #2563eb; border-color: #3b82f6; }}
    .btn-action:hover:not(:disabled) {{ background: #1d4ed8; }}
    .btn-dica {{ background: #059669; border-color: #10b981; }}
    .btn-dica:hover:not(:disabled) {{ background: #047857; }}
    .grid-info {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 16px;
    }}
    @media (max-width: 800px) {{
      .grid-info {{ grid-template-columns: 1fr; }}
    }}
    .info-card {{
      background: var(--card-bg);
      border-radius: 12px;
      border: 1px solid var(--border);
      padding: 18px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 12px;
    }}
    .info-item h3 {{ font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px; }}
    .info-item p {{ font-size: 1.05rem; font-weight: 700; }}
    .log-card {{
      background: var(--card-bg);
      border-radius: 12px;
      border: 1px solid var(--border);
      padding: 16px;
      max-height: 220px;
      overflow-y: auto;
      font-size: 0.85rem;
    }}
    .log-card h3 {{ font-size: 0.8rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px; }}
    .log-item {{ padding: 4px 0; border-bottom: 1px solid #2d3748; display: flex; justify-content: space-between; }}
    .legend {{
      display: flex;
      justify-content: center;
      gap: 16px;
      margin-top: 10px;
      flex-wrap: wrap;
      font-size: 0.85rem;
    }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; }}
    .legend-box {{ width: 14px; height: 14px; border-radius: 50%; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Jogo da Convexidade P3: Humano vs IA Minimax</h1>
        <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 4px;">
          Convenção Normal: Vence quem realizar a ÚLTIMA jogada válida infectando a totalidade do grafo.
        </p>
      </div>
      <div id="turno-badge" class="badge badge-alice">Vez de Alice (MAX)</div>
    </header>

    <div class="panel-config">
      <div style="display: flex; align-items: center; gap: 8px;">
        <label for="select-grafo"><strong>Grafo:</strong></label>
        <select id="select-grafo" onchange="trocarGrafo(this.value)">
          <option value="2,2">C(2, 2) Cheio (n=6) -- Alice vence</option>
          <option value="2,2,2">C(3, 2) Cheio (n=9) -- Alice vence</option>
          <option value="2,2,2,2" selected>C(4, 2) Cheio (n=12) -- BOB VENCE (Desafio!)</option>
          <option value="2,2,2,2,2">C(5, 2) Cheio (n=15) -- Alice vence</option>
          <option value="0,2,0,1,1,0">Grafo TCC L=(0,2,0,1,1,0) (n=10) -- Bob vence</option>
          <option value="0,0,0,0,0">Caminho P5 (n=5) -- Alice vence</option>
          <option value="0,0,0,0,0,0">Caminho P6 (n=6) -- Alice vence</option>
          <option value="0,2,0">Estrela K1,4 (n=5) -- Alice vence</option>
        </select>
      </div>

      <div style="display: flex; align-items: center; gap: 8px;">
        <label for="select-lado"><strong>Você joga como:</strong></label>
        <select id="select-lado" onchange="reiniciarJogo()">
          <option value="alice" selected>Alice (1º lance, Peças Vermelhas)</option>
          <option value="bob">Bob (2º lance, Peças Azuis)</option>
        </select>
      </div>

      <div style="display: flex; align-items: center; gap: 8px;">
        <label for="select-ia"><strong>Inteligência da IA:</strong></label>
        <select id="select-ia">
          <option value="minimax" selected>Minimax Perfeito (Ótimo / Implacável)</option>
          <option value="casual">Casual (Treinamento / Humano)</option>
        </select>
      </div>
    </div>

    <div class="stage">
      <svg id="svg-canvas" viewBox="0 0 {largura_svg} {altura_svg}"></svg>

      <div class="legend">
        <div class="legend-item"><div class="legend-box" style="background: var(--livre-fill); border: 2px solid var(--livre);"></div> Livre (Clique para jogar)</div>
        <div class="legend-item"><div class="legend-box" style="background: var(--alice);"></div> Jogada de Alice</div>
        <div class="legend-item"><div class="legend-box" style="background: var(--bob);"></div> Jogada de Bob</div>
        <div class="legend-item"><div class="legend-box" style="background: var(--prop);"></div> Fecho P3 (Propagado)</div>
        <div class="legend-item"><div class="legend-box" style="background: transparent; border: 2px dashed var(--dica);"></div> Dica Minimax</div>
      </div>

      <div id="banner-status" class="banner-status">
        Clique em um vértice livre do grafo para realizar sua jogada inicial!
      </div>

      <div class="controls">
        <button class="btn-action" onclick="reiniciarJogo()">🔄 Reiniciar Partida</button>
        <button onclick="desfazerLance()" id="btn-undo">↩️ Desfazer Lance</button>
        <button class="btn-dica" onclick="mostrarDicaMinimax()" id="btn-dica">💡 Pedir Dica do Minimax</button>
      </div>
    </div>

    <div class="grid-info">
      <div class="info-card">
        <div class="info-item">
          <h3>Turno Atual</h3>
          <p id="info-turno">Turno 1</p>
        </div>
        <div class="info-item">
          <h3>Jogador na Vez</h3>
          <p id="info-vez">Alice (Você)</p>
        </div>
        <div class="info-item">
          <h3>Último Lance</h3>
          <p id="info-ultimo-lance">Nenhum</p>
        </div>
        <div class="info-item">
          <h3>Propagados P3</h3>
          <p id="info-props">0 nós</p>
        </div>
        <div class="info-item">
          <h3>Infectados</h3>
          <p id="info-progresso">0 / {cat_inicial.n} (0%)</p>
        </div>
      </div>

      <div class="log-card">
        <h3>Histórico da Partida</h3>
        <div id="log-content">
          <div style="color: var(--text-muted); font-style: italic;">Aguardando lance inicial...</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    // --- MOTOR MATEMÁTICO EM JAVASCRIPT ---
    let grafoAtual = {{
      L: {list(cat_inicial.assinatura)},
      n: {cat_inicial.n},
      k: {cat_inicial.k},
      adj: {json.dumps(adj_json)},
      rotulos: {json.dumps(rotulos_json)},
      nos: {json.dumps(nos_svg)},
      arestas: {json.dumps(arestas_svg)}
    }};

    let S_atual = 0;
    let jogadorAtual = 1; // +1 = Alice, -1 = Bob
    let historicoPartida = [];
    let dicasAtivas = new Set();
    let bloqueado = false;

    // Operador Fecho Convexo P3 em JS (BFS O(|V|+|E|) com contadores)
    function fechoP3(adj, S_ini) {{
      let n = adj.length;
      let contador = new Int32Array(n);
      let fila = [];
      let enfileirado = new Uint8Array(n);

      for (let v = 0; v < n; v++) {{
        if (!((S_ini >> v) & 1)) {{
          let c = 0;
          for (let u of adj[v]) {{
            if ((S_ini >> u) & 1) c++;
          }}
          contador[v] = c;
          if (c >= 2) {{
            fila.push(v);
            enfileirado[v] = 1;
          }}
        }}
      }}

      let S = S_ini;
      let propagados = [];
      while (fila.length > 0) {{
        let u = fila.shift();
        S |= (1 << u);
        propagados.push(u);
        for (let w of adj[u]) {{
          if (!((S >> w) & 1) && !enfileirado[w]) {{
            contador[w]++;
            if (contador[w] >= 2) {{
              fila.push(w);
              enfileirado[w] = 1;
            }}
          }}
        }}
      }}
      return {{ fecho: S, propagados: propagados }};
    }}

    // Solucionador Minimax com Memoização e Poda Booleana em JS
    let cacheMinimax = new Map();

    function valorMinimax(adj, S, jogador) {{
      let n = adj.length;
      let completo = (1 << n) - 1;
      if (S === completo) return -jogador;

      let chave = S;
      if (cacheMinimax.has(chave)) {{
        return cacheMinimax.get(chave) * jogador;
      }}

      let v = -jogador;
      for (let u = 0; u < n; u++) {{
        if (!((S >> u) & 1)) {{
          let prox = fechoP3(adj, S | (1 << u)).fecho;
          let val = valorMinimax(adj, prox, -jogador);
          if (jogador === 1) {{
            if (val === 1) {{ v = 1; break; }}
          }} else {{
            if (val === -1) {{ v = -1; break; }}
          }}
        }}
      }}
      cacheMinimax.set(chave, v * jogador);
      return v;
    }}

    function obterMelhorJogadaIA(adj, S, jogadorIA, casual = false) {{
      let n = adj.length;
      let candidatas = [];
      let vencedoras = [];

      for (let u = 0; u < n; u++) {{
        if (!((S >> u) & 1)) {{
          let prox = fechoP3(adj, S | (1 << u)).fecho;
          let val = valorMinimax(adj, prox, -jogadorIA);
          candidatas.push({{ jogada: u, val: val }});
          if (val === jogadorIA) {{
            vencedoras.push(u);
          }}
        }}
      }}

      if (casual && Math.random() < 0.35 && candidatas.length > 1) {{
        return candidatas[Math.floor(Math.random() * candidatas.length)].jogada;
      }}

      if (vencedoras.length > 0) return vencedoras[0];
      return candidatas[0].jogada;
    }}

    // --- RENDERIZAÇÃO E INTERAÇÃO ---
    const svg = document.getElementById("svg-canvas");

    function renderizarTabuleiro() {{
      svg.innerHTML = "";
      const completo = (1 << grafoAtual.n) - 1;

      // Desenha Arestas
      grafoAtual.arestas.forEach(a => {{
        const u = grafoAtual.nos[a.u];
        const v = grafoAtual.nos[a.v];
        const ativa = ((S_atual >> a.u) & 1) && ((S_atual >> a.v) & 1);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", u.x);
        line.setAttribute("y1", u.y);
        line.setAttribute("x2", v.x);
        line.setAttribute("y2", v.y);
        line.setAttribute("class", "edge" + (ativa ? " active" : ""));
        svg.appendChild(line);
      }});

      // Desenha Nós
      grafoAtual.nos.forEach(n => {{
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        const infectado = (S_atual >> n.id) & 1;

        let corFill = "var(--livre-fill)";
        let corBorda = "var(--livre)";
        let raio = n.eh_espinha ? 20 : 16;
        let larguraBorda = 2;

        // Histórico de cores
        for (let h of historicoPartida) {{
          if (h.lance === n.id) {{
            corFill = h.jogador === 1 ? "var(--alice)" : "var(--bob)";
            corBorda = "#ffffff";
            larguraBorda = 3;
          }} else if (h.propagados.includes(n.id)) {{
            corFill = "var(--prop)";
            corBorda = "#fef08a";
            larguraBorda = 2.5;
          }}
        }}

        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", n.x);
        circle.setAttribute("cy", n.y);
        circle.setAttribute("r", raio);
        circle.setAttribute("fill", corFill);
        circle.setAttribute("stroke", corBorda);
        circle.setAttribute("stroke-width", larguraBorda);

        let classes = "node-circle";
        if (infectado) classes += " infected";
        if (dicasAtivas.has(n.id)) classes += " dica";
        circle.setAttribute("class", classes);

        if (!infectado && !bloqueado) {{
          circle.onclick = () => realizarJogadaHumano(n.id);
        }}

        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", n.x);
        text.setAttribute("y", n.y);
        text.setAttribute("fill", infectado ? "#ffffff" : "#94a3b8");
        text.setAttribute("class", "node-text");
        text.textContent = n.rotulo;

        g.appendChild(circle);
        g.appendChild(text);
        svg.appendChild(g);
      }});

      atualizarPainelInfo();
    }}

    function atualizarPainelInfo() {{
      const humanoEhAlice = document.getElementById("select-lado").value === "alice";
      const vezDoHumano = (jogadorAtual === 1 && humanoEhAlice) || (jogadorAtual === -1 && !humanoEhAlice);
      const completo = (1 << grafoAtual.n) - 1;

      // Badge de turno
      const badge = document.getElementById("turno-badge");
      if (S_atual === completo) {{
        badge.className = "badge";
        badge.textContent = "Partida Concluída";
        badge.style.background = "#10b981";
      }} else if (jogadorAtual === 1) {{
        badge.className = "badge badge-alice";
        badge.textContent = `Vez de Alice (MAX) ${{vezDoHumano ? '-- Você' : '-- IA'}}`;
      }} else {{
        badge.className = "badge badge-bob";
        badge.textContent = `Vez de Bob (MIN) ${{vezDoHumano ? '-- Você' : '-- IA'}}`;
      }}

      // Informações dos cards
      document.getElementById("info-turno").textContent = `Turno ${{historicoPartida.length + 1}}`;
      document.getElementById("info-vez").textContent = `${{jogadorAtual === 1 ? 'Alice' : 'Bob'}} ${{vezDoHumano ? '(Você)' : '(IA)'}}`;

      let contagemInf = 0;
      for (let i = 0; i < grafoAtual.n; i++) if ((S_atual >> i) & 1) contagemInf++;
      let pct = Math.round((contagemInf / grafoAtual.n) * 100);
      document.getElementById("info-progresso").textContent = `${{contagemInf}} / ${{grafoAtual.n}} (${{pct}}%)`;

      document.getElementById("btn-undo").disabled = (historicoPartida.length === 0 || bloqueado);
      document.getElementById("btn-dica").disabled = (S_atual === completo || !vezDoHumano || bloqueado);
    }}

    function realizarJogadaHumano(v) {{
      const humanoEhAlice = document.getElementById("select-lado").value === "alice";
      const vezDoHumano = (jogadorAtual === 1 && humanoEhAlice) || (jogadorAtual === -1 && !humanoEhAlice);

      if (!vezDoHumano || bloqueado || ((S_atual >> v) & 1)) return;

      dicasAtivas.clear();
      executarLance(v, true);
    }}

    function executarLance(v, ehHumano) {{
      const res = fechoP3(grafoAtual.adj, S_atual | (1 << v));
      const completo = (1 << grafoAtual.n) - 1;

      historicoPartida.push({{
        S_anterior: S_atual,
        jogador: jogadorAtual,
        lance: v,
        propagados: res.propagados,
        ehHumano: ehHumano
      }});

      S_atual = res.fecho;

      // Atualiza log
      adicionarAoLog(jogadorAtual, v, res.propagados, ehHumano);

      document.getElementById("info-ultimo-lance").textContent = grafoAtual.rotulos[v];
      document.getElementById("info-props").textContent = res.propagados.length > 0
        ? res.propagados.map(p => grafoAtual.rotulos[p]).join(", ")
        : "Nenhuma";

      renderizarTabuleiro();

      // Checa condição de vitória (Convenção Normal: último a jogar vence)
      if (S_atual === completo) {{
        finalizarPartida(ehHumano);
        return;
      }}

      jogadorAtual = -jogadorAtual;
      renderizarTabuleiro();

      // Se for a vez da IA, agenda jogada
      const humanoEhAlice = document.getElementById("select-lado").value === "alice";
      const vezDoHumano = (jogadorAtual === 1 && humanoEhAlice) || (jogadorAtual === -1 && !humanoEhAlice);

      if (!vezDoHumano) {{
        bloqueado = true;
        document.getElementById("banner-status").className = "banner-status";
        document.getElementById("banner-status").innerHTML = "🤖 IA Minimax pensando...";

        setTimeout(() => {{
          const modoCasual = document.getElementById("select-ia").value === "casual";
          const lanceIA = obterMelhorJogadaIA(grafoAtual.adj, S_atual, jogadorAtual, modoCasual);
          bloqueado = false;
          executarLance(lanceIA, false);
        }}, 450);
      }} else {{
        document.getElementById("banner-status").className = "banner-status";
        document.getElementById("banner-status").innerHTML = "Sua vez! Escolha um vértice livre no grafo.";
      }}
    }}

    function finalizarPartida(humanoVenceu) {{
      bloqueado = true;
      const banner = document.getElementById("banner-status");
      if (humanoVenceu) {{
        banner.className = "banner-status vitoria";
        banner.innerHTML = "🏆 PARABÉNS! Você realizou o último lance válido e VENCEU A IA!";
      }} else {{
        banner.className = "banner-status derrota";
        banner.innerHTML = "💀 FIM DE JOGO! A IA realizou o último lance válido e venceu a partida.";
      }}
    }}

    function adicionarAoLog(jog, lance, props, ehHumano) {{
      const logBox = document.getElementById("log-content");
      if (historicoPartida.length === 1) logBox.innerHTML = "";

      const item = document.createElement("div");
      item.className = "log-item";

      const nomeJog = jog === 1 ? "Alice" : "Bob";
      const quem = ehHumano ? "(Você)" : "(IA)";
      const propsTxt = props.length > 0 ? ` [propagou: ${{props.map(p => grafoAtual.rotulos[p]).join(", ")}}]` : "";

      item.innerHTML = `
        <span><strong>T${{historicoPartida.length}}</strong> ${{nomeJog}} ${{quem}}: <strong>${{grafoAtual.rotulos[lance]}}</strong>${{propsTxt}}</span>
        <span style="color: var(--text-muted); font-size: 0.8rem;">${{new Date().toLocaleTimeString().slice(3)}}</span>
      `;
      logBox.prepend(item);
    }}

    function mostrarDicaMinimax() {{
      const n = grafoAtual.n;
      let vencedoras = [];
      for (let u = 0; u < n; u++) {{
        if (!((S_atual >> u) & 1)) {{
          let prox = fechoP3(grafoAtual.adj, S_atual | (1 << u)).fecho;
          let val = valorMinimax(grafoAtual.adj, prox, -jogadorAtual);
          if (val === jogadorAtual) vencedoras.push(u);
        }}
      }}

      dicasAtivas.clear();
      const banner = document.getElementById("banner-status");
      if (vencedoras.length > 0) {{
        vencedoras.forEach(v => dicasAtivas.add(v));
        banner.className = "banner-status";
        banner.innerHTML = `💡 <strong>Dica Minimax:</strong> Jogada(s) com vitória garantida: <strong>${{vencedoras.map(v => grafoAtual.rotulos[v]).join(", ")}}</strong> (em verde)`;
      }} else {{
        banner.className = "banner-status";
        banner.innerHTML = "💡 <strong>Dica Minimax:</strong> A IA tem vitória forçada sob jogo perfeito, mas jogue mantendo o controle da espinha!";
      }}
      renderizarTabuleiro();
    }}

    function desfazerLance() {{
      if (historicoPartida.length === 0 || bloqueado) return;

      // Desfaz até a última jogada do humano
      if (historicoPartida.length >= 2 && !historicoPartida[historicoPartida.length - 1].ehHumano) {{
        historicoPartida.pop(); // desfaz IA
        const h = historicoPartida.pop(); // desfaz humano
        S_atual = h.S_anterior;
        jogadorAtual = h.jogador;
      }} else {{
        const h = historicoPartida.pop();
        S_atual = h.S_anterior;
        jogadorAtual = h.jogador;
      }}

      dicasAtivas.clear();
      bloqueado = false;
      document.getElementById("banner-status").className = "banner-status";
      document.getElementById("banner-status").innerHTML = "Jogada desfeita. Escolha seu próximo lance.";
      renderizarTabuleiro();
    }}

    function construirCaterpillarPorAssinatura(L) {{
      const k = L.length;
      let totalPernas = L.reduce((a, b) => a + b, 0);
      let n = k + totalPernas;

      let adj = Array.from({{ length: n }}, () => []);
      let rotulos = Array.from({{ length: n }}, (_, i) => `v${{i + 1}}`);
      let paiDaPerna = {{}};

      for (let i = 0; i < k - 1; i++) {{
        adj[i].push(i + 1);
        adj[i + 1].push(i);
      }}

      let prox = k;
      for (let i = 0; i < k; i++) {{
        for (let p = 0; p < L[i]; p++) {{
          adj[i].push(prox);
          adj[prox].push(i);
          paiDaPerna[prox] = i;
          prox++;
        }}
      }}

      // Coordenadas
      let pos = {{}};
      let espacamento = 1.8;
      for (let i = 0; i < k; i++) pos[i] = {{ x: i * espacamento, y: 0.0 }};

      let pernasPorPai = Array.from({{ length: k }}, () => []);
      for (let [perna, pai] of Object.entries(paiDaPerna)) {{
        pernasPorPai[pai].push(parseInt(perna));
      }}

      for (let i = 0; i < k; i++) {{
        let pernas = pernasPorPai[i];
        let qtd = pernas.length;
        if (qtd === 1) {{
          pos[pernas[0]] = {{ x: pos[i].x, y: 1.2 }};
        }} else if (qtd === 2) {{
          pos[pernas[0]] = {{ x: pos[i].x, y: 1.2 }};
          pos[pernas[1]] = {{ x: pos[i].x, y: -1.2 }};
        }} else if (qtd > 2) {{
          let mc = Math.ceil(qtd / 2);
          let mb = qtd - mc;
          for (let idx = 0; idx < mc; idx++) pos[pernas[idx]] = {{ x: pos[i].x - 0.3 + idx * 0.6, y: 1.2 }};
          for (let idx = 0; idx < mb; idx++) pos[pernas[mc + idx]] = {{ x: pos[i].x - 0.3 + idx * 0.6, y: -1.2 }};
        }}
      }}

      let xs = Object.values(pos).map(p => p.x);
      let ys = Object.values(pos).map(p => p.y);
      let min_x = Math.min(...xs), max_x = Math.max(...xs);
      let min_y = Math.min(...ys), max_y = Math.max(...ys);
      let dx = max_x > min_x ? max_x - min_x : 1;
      let dy = max_y > min_y ? max_y - min_y : 1;

      let nos_svg = [];
      let largura_svg = 950, altura_svg = 420, margem = 60;
      for (let v = 0; v < n; v++) {{
        let p = pos[v];
        nos_svg.push({{
          id: v,
          rotulo: rotulos[v],
          x: Math.round(margem + ((p.x - min_x) / dx) * (largura_svg - 2 * margem)),
          y: Math.round(altura_svg - (margem + ((p.y - min_y) / dy) * (altura_svg - 2 * margem))),
          eh_espinha: v < k
        }});
      }}

      let arestas_svg = [];
      for (let u = 0; u < n; u++) {{
        for (let v of adj[u]) {{
          if (u < v) arestas_svg.push({{ u, v }});
        }}
      }}

      return {{ L, n, k, adj, rotulos, nos: nos_svg, arestas: arestas_svg }};
    }}

    function trocarGrafo(strAssinatura) {{
      const L = strAssinatura.split(",").map(x => parseInt(x.trim()));
      grafoAtual = construirCaterpillarPorAssinatura(L);
      cacheMinimax.clear();
      reiniciarJogo();
    }}

    function reiniciarJogo() {{
      S_atual = 0;
      jogadorAtual = 1;
      historicoPartida = [];
      dicasAtivas.clear();
      bloqueado = false;
      cacheMinimax.clear();

      document.getElementById("log-content").innerHTML = '<div style="color: var(--text-muted); font-style: italic;">Aguardando lance inicial...</div>';
      document.getElementById("info-ultimo-lance").textContent = "Nenhum";
      document.getElementById("info-props").textContent = "0 nós";

      const humanoEhAlice = document.getElementById("select-lado").value === "alice";
      if (!humanoEhAlice) {{
        // IA começa jogando como Alice
        bloqueado = true;
        document.getElementById("banner-status").className = "banner-status";
        document.getElementById("banner-status").innerHTML = "🤖 A IA joga como Alice (1º lance)... Pensando...";
        renderizarTabuleiro();

        setTimeout(() => {{
          const modoCasual = document.getElementById("select-ia").value === "casual";
          const lanceIA = obterMelhorJogadaIA(grafoAtual.adj, S_atual, 1, modoCasual);
          bloqueado = false;
          executarLance(lanceIA, false);
        }}, 600);
      }} else {{
        document.getElementById("banner-status").className = "banner-status";
        document.getElementById("banner-status").innerHTML = "Sua vez! Você joga como Alice (1º lance). Clique em um nó para infectá-lo.";
        renderizarTabuleiro();
      }}
    }}

    // Inicia jogo ao carregar
    renderizarTabuleiro();
  </script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(caminho_html), exist_ok=True)
    with open(caminho_html, "w", encoding="utf-8") as f:
        f.write(html_template)

    return caminho_html

