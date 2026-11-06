import networkx as nx
from collections import deque
from itertools import product


def criar_caterpillar(pernas_por_espinha):
    """
    Cria um grafo Caterpillar G = (V, E) a partir de uma lista indicando a
    quantidade de pernas (folhas pendentes) em cada vértice da espinha.

    Exemplo:
        pernas_por_espinha = [0, 2, 0, 1, 1, 0]
        Cria o caterpillar das Figuras 1, 2 e 4 do TCC:
        - Espinha: v1, v2, v3, v4, v5, v6
        - Pernas de v2: v7, v8
        - Perna de v4: v9
        - Perna de v5: v10

    Retorna:
        G (nx.Graph): Grafo NetworkX com atributos nos nós ('tipo', 'espinha_idx',
                      'estado', 'jogada', 'auto_infectado') e no grafo ('espinha',
                      'pernas', 'assinatura').
    """
    G = nx.Graph()
    k = len(pernas_por_espinha)

    espinha = []
    # 1. Adiciona os vértices da espinha: v1 .. vk
    for i in range(1, k + 1):
        v = f"v{i}"
        espinha.append(v)
        G.add_node(
            v,
            tipo="espinha",
            espinha_idx=i - 1,
            color="lightgray",
            jogada=None,
            auto_infectado=False,
        )
        if i > 1:
            G.add_edge(f"v{i-1}", v)

    # 2. Adiciona os vértices das pernas (folhas ligadas à espinha): vk+1 .. vn
    prox_id = k + 1
    mapa_pernas = {}
    todas_pernas = []

    for i, qtd_pernas in enumerate(pernas_por_espinha):
        v_esp = f"v{i+1}"
        mapa_pernas[v_esp] = []
        for _ in range(qtd_pernas):
            v_folha = f"v{prox_id}"
            prox_id += 1
            G.add_node(
                v_folha,
                tipo="perna",
                pai_espinha=v_esp,
                espinha_idx=i,
                color="lightgray",
                jogada=None,
                auto_infectado=False,
            )
            G.add_edge(v_esp, v_folha)
            mapa_pernas[v_esp].append(v_folha)
            todas_pernas.append(v_folha)

    G.graph["espinha"] = espinha
    G.graph["mapa_pernas"] = mapa_pernas
    G.graph["pernas"] = todas_pernas
    G.graph["assinatura"] = tuple(pernas_por_espinha)

    return G


def fechamento_convexo_p3(G, conjunto_inicial):
    """
    Calcula o fechamento convexo P3 de um conjunto de vértices S em G.
    Todo vértice externo a S que possui pelo menos 2 vizinhos em S é incorporado
    sucessivamente até que nenhum novo vértice possa ser adicionado.

    Argumentos:
        G (nx.Graph): O grafo analisado.
        conjunto_inicial (iterable): Conjunto atual de vértices infectados S.

    Retorna:
        fechamento (frozenset): Conjunto P3-convexo resultante [S]_P3.
        ordem_propagacao (list): Lista ordenada dos vértices que foram incorporados
                                 automaticamente durante a propagação.
    """
    fechamento = set(conjunto_inicial)
    ordem_propagacao = []

    # Contagem de vizinhos já pertencentes a S para cada vértice fora de S
    vizinhos_infectados = {}
    fila = deque()

    for v in G.nodes:
        if v not in fechamento:
            qtd = sum(1 for u in G.neighbors(v) if u in fechamento)
            vizinhos_infectados[v] = qtd
            if qtd >= 2:
                fila.append(v)

    # Propagação em largura (BFS)
    enfileirados = set(fila)
    while fila:
        u = fila.popleft()
        fechamento.add(u)
        ordem_propagacao.append(u)

        for w in G.neighbors(u):
            if w not in fechamento and w not in enfileirados:
                vizinhos_infectados[w] += 1
                if vizinhos_infectados[w] >= 2:
                    enfileirados.add(w)
                    fila.append(w)

    return frozenset(fechamento), ordem_propagacao


def aplicar_jogada_no_grafo(G, vertice, numero_turno, jogador_nome):
    """
    Aplica uma jogada escolhida por Alice ou Bob diretamente nos atributos do grafo G,
    executando em seguida o fechamento convexo P3.

    Retorna:
        propagados (list): Lista de vértices infectados automaticamente pelo fechamento P3.
    """
    infectados_atuais = {
        v for v in G.nodes if G.nodes[v]["color"] != "lightgray"
    }

    # Marca o vértice selecionado diretamente pelo jogador
    cor_direta = "lightcoral" if jogador_nome == "Alice" else "cornflowerblue"
    G.nodes[vertice]["color"] = cor_direta
    G.nodes[vertice]["jogada"] = numero_turno
    G.nodes[vertice]["auto_infectado"] = False

    novo_conjunto = infectados_atuais | {vertice}
    fechamento, propagados = fechamento_convexo_p3(G, novo_conjunto)

    # Marca os vértices incorporados automaticamente pelo fechamento convexo P3
    for v_prop in propagados:
        G.nodes[v_prop]["color"] = "gold"
        G.nodes[v_prop]["jogada"] = numero_turno
        G.nodes[v_prop]["auto_infectado"] = True

    return propagados


def resetar_grafo(G):
    """Reseta as cores e marcações de partida de um grafo G."""
    for v in G.nodes:
        G.nodes[v]["color"] = "lightgray"
        G.nodes[v]["jogada"] = None
        G.nodes[v]["auto_infectado"] = False


def gerar_familias_caterpillar(max_espinha=5, max_pernas_por_vertice=2, max_total_vertices=12):
    """
    Gera assinaturas não-isomorfas de grafos caterpillar (removendo espelhamentos
    reversos da espinha) para experimentos sistemáticos.

    Retorna:
        lista de tuplas (pernas_por_espinha)
    """
    instancias = []
    vistos = set()

    for k in range(1, max_espinha + 1):
        for config in product(range(max_pernas_por_vertice + 1), repeat=k):
            total_v = k + sum(config)
            if total_v > max_total_vertices or total_v < 2:
                continue

            # Forma canônica quanto à inversão da espinha
            canonica = min(config, config[::-1])
            if canonica not in vistos:
                vistos.add(canonica)
                instancias.append(canonica)

    return instancias
