import networkx as nx
from caterpillar_p3 import (
    fechamento_convexo_p3,
    aplicar_jogada_no_grafo,
    resetar_grafo,
)
from visualizacao import mostrar_grafo

jogadas = {
    "Alice": 0,  # Primeiro jogador (MAX: busca utilidade +1)
    "Bob": 1,    # Segundo jogador  (MIN: busca utilidade -1)
}


def obter_vertices_disponiveis(G, estado_convexo):
    """Retorna a lista de vértices de V(G) que ainda não pertencem ao conjunto convexo S."""
    return [v for v in G.nodes if v not in estado_convexo]


def obter_jogadas_nao_simetricas(G, estado_convexo):
    """
    Otimização por Redução de Simetria de Folhas (Pernas):
    Em um grafo Caterpillar, todas as folhas não infectadas ligadas a um mesmo
    vértice da espinha são estruturalmente indistinguíveis. Portanto, basta
    testar apenas 1 folha disponível por vértice da espinha na busca Minimax.
    """
    disponiveis = obter_vertices_disponiveis(G, estado_convexo)
    pais_folhas_ja_incluidos = set()
    jogadas_filtradas = []

    for v in disponiveis:
        if G.nodes[v].get("tipo") == "perna":
            pai = G.nodes[v].get("pai_espinha")
            if pai in pais_folhas_ja_incluidos:
                continue
            pais_folhas_ja_incluidos.add(pai)
        jogadas_filtradas.append(v)

    return jogadas_filtradas


def minimax(
    G,
    estado_convexo=frozenset(),
    jogada=jogadas["Alice"],
    usar_memoizacao=True,
    reduzir_simetria=True,
    tabela_memo=None,
    estatisticas=None,
):
    """
    Algoritmo Minimax adaptado para o Jogo da Convexidade P3 (Closed Hull Game - CHG)
    na convenção normal (vence quem realiza a última jogada válida que infecta todo o grafo).

     Convenção de retorno (seguindo a estrutura do projeto base):
        (vitoria_alice: bool, utilidade: int, tag_estado: str, melhor_vertice: str | None)
        - utilidade = +1 se Alice (1º jogador / MAX) possui estratégia vencedora;
        - utilidade = -1 se Bob   (2º jogador / MIN) possui estratégia vencedora.
    """
    if tabela_memo is None:
        tabela_memo = {}
    if estatisticas is not None:
        estatisticas["nos_visitados"] = estatisticas.get("nos_visitados", 0) + 1

    total_vertices = G.number_of_nodes()

    # Caso base: Estado Terminal (todo o grafo já foi infectado: S == V(G))
    # Na convenção normal, quem fez a jogada imediatamente anterior venceu!
    # Portanto, o jogador que deveria jogar agora perdeu (não há mais jogadas válidas).
    if len(estado_convexo) == total_vertices:
        if estatisticas is not None:
            estatisticas["estados_terminais"] = estatisticas.get("estados_terminais", 0) + 1
        if jogada == jogadas["Alice"]:
            # É a vez de Alice mas o grafo já está cheio -> Bob fez a última jogada e venceu!
            return False, -1, "BEND", None
        else:
            # É a vez de Bob mas o grafo já está cheio -> Alice fez a última jogada e venceu!
            return True, +1, "AEND", None

    # Consulta à Tabela de Transposição (Memoização)
    chave_memo = (estado_convexo, jogada)
    if usar_memoizacao and chave_memo in tabela_memo:
        if estatisticas is not None:
            estatisticas["hits_memo"] = estatisticas.get("hits_memo", 0) + 1
        return tabela_memo[chave_memo]

    # Lista de candidatos a jogada
    if reduzir_simetria:
        candidatos = obter_jogadas_nao_simetricas(G, estado_convexo)
    else:
        candidatos = obter_vertices_disponiveis(G, estado_convexo)

    # Turno de Alice (Jogador MAX -> busca utilidade +1 / True)
    if jogada == jogadas["Alice"]:
        melhor_vertice = candidatos[0] if candidatos else None
        for vertice in candidatos:
            novo_estado, _ = fechamento_convexo_p3(G, estado_convexo | {vertice})
            vence_alice, utilidade, _, _ = minimax(
                G,
                novo_estado,
                jogadas["Bob"],
                usar_memoizacao,
                reduzir_simetria,
                tabela_memo,
                estatisticas,
            )
            if vence_alice:  # Encontrou jogada vencedora para Alice (+1)
                resultado = (True, +1, "A", vertice)
                if usar_memoizacao:
                    tabela_memo[chave_memo] = resultado
                return resultado

        # Se nenhuma jogada garante vitória a Alice, Bob vence (-1)
        resultado = (False, -1, "A", melhor_vertice)
        if usar_memoizacao:
            tabela_memo[chave_memo] = resultado
        return resultado

    # Turno de Bob (Jogador MIN -> busca utilidade -1 / False)
    else:
        melhor_vertice = candidatos[0] if candidatos else None
        for vertice in candidatos:
            novo_estado, _ = fechamento_convexo_p3(G, estado_convexo | {vertice})
            vence_alice, utilidade, _, _ = minimax(
                G,
                novo_estado,
                jogadas["Alice"],
                usar_memoizacao,
                reduzir_simetria,
                tabela_memo,
                estatisticas,
            )
            if not vence_alice:  # Encontrou jogada vencedora para Bob (-1)
                resultado = (False, -1, "B", vertice)
                if usar_memoizacao:
                    tabela_memo[chave_memo] = resultado
                return resultado

        # Se nenhuma jogada garante vitória a Bob, Alice vence (+1)
        resultado = (True, +1, "B", melhor_vertice)
        if usar_memoizacao:
            tabela_memo[chave_memo] = resultado
        return resultado


def listar_todas_jogadas_vencedoras_iniciais(G):
    """
    Avalia todas as jogadas iniciais v ∈ V(G) para Alice no estado inicial S = ∅
    e retorna quais vértices conduzem a uma posição vencedora para Alice (+1).
    """
    jogadas_vencedoras = []
    detalhes_por_vertice = {}
    tabela_memo = {}

    for vertice in G.nodes:
        novo_estado, propagados = fechamento_convexo_p3(G, {vertice})
        vence_alice, utilidade, _, resposta_bob = minimax(
            G,
            novo_estado,
            jogadas["Bob"],
            usar_memoizacao=True,
            reduzir_simetria=True,
            tabela_memo=tabela_memo,
        )
        detalhes_por_vertice[vertice] = {
            "vence_alice": vence_alice,
            "utilidade": utilidade,
            "fechamento_inicial": sorted(list(novo_estado)),
            "propagados": propagados,
            "melhor_resposta_bob": resposta_bob,
        }
        if vence_alice:
            jogadas_vencedoras.append(vertice)

    return jogadas_vencedoras, detalhes_por_vertice


def gerar_arvore_de_jogo(G, limite_nos=500):
    """
    Constrói explicitamente o grafo direcionado (DAG / Árvore de Estados) das
    configurações alcançáveis a partir de S = ∅, conforme a Seção 5.3 e 6.3 do TCC.

    Retorna:
        arvore (nx.DiGraph): Grafo direcionado onde cada nó é (estado_convexo, jogador_da_vez)
                             com seu valor Minimax (+1 ou -1).
    """
    arvore = nx.DiGraph()
    tabela_memo = {}

    estado_inicial = frozenset()
    vence_ini, util_ini, _, melhor_ini = minimax(
        G, estado_inicial, jogadas["Alice"], tabela_memo=tabela_memo
    )

    no_raiz = (estado_inicial, "Alice")
    arvore.add_node(
        no_raiz,
        estado=set(estado_inicial),
        jogador="Alice",
        utilidade=util_ini,
        melhor_jogada=melhor_ini,
        terminal=(len(estado_inicial) == G.number_of_nodes()),
    )

    fila = [no_raiz]
    visitados = {no_raiz}

    while fila and arvore.number_of_nodes() < limite_nos:
        estado_atual, nome_jog = fila.pop(0)
        if len(estado_atual) == G.number_of_nodes():
            continue

        prox_jog_nome = "Bob" if nome_jog == "Alice" else "Alice"
        prox_jog_id = jogadas[prox_jog_nome]

        for v in obter_jogadas_nao_simetricas(G, estado_atual):
            novo_estado, propagados = fechamento_convexo_p3(G, estado_atual | {v})
            _, util_filho, _, melhor_filho = minimax(
                G, novo_estado, prox_jog_id, tabela_memo=tabela_memo
            )

            no_filho = (novo_estado, prox_jog_nome)
            if no_filho not in visitados:
                visitados.add(no_filho)
                arvore.add_node(
                    no_filho,
                    estado=set(novo_estado),
                    jogador=prox_jog_nome,
                    utilidade=util_filho,
                    melhor_jogada=melhor_filho,
                    terminal=(len(novo_estado) == G.number_of_nodes()),
                )
                fila.append(no_filho)

            arvore.add_edge(
                (estado_atual, nome_jog),
                no_filho,
                jogada=v,
                propagados=propagados,
            )

    return arvore


def jogo_convexidade_p3(G, modo_interativo=True, exibir_graficos=True):
    """
    Executa uma partida completa do Jogo da Convexidade P3 sobre o grafo Caterpillar G.
    Adaptação direta de `jogo_coloracao_gulosa` do projeto de referência para o
    contexto da Convexidade P3.
    """
    resetar_grafo(G)
    jogador_escolhido = -1  # Padrão: IA vs IA

    if modo_interativo:
        escolha = input(
            "Escolha como deseja jogar ('Alice', 'Bob' ou 'IA' para assistir IA vs IA): "
        ).strip()
        if escolha in jogadas:
            jogador_escolhido = jogadas[escolha]
        else:
            print("Modo selecionado: IA (Alice) vs IA (Bob) via Minimax.")

    estado_convexo = frozenset()
    jogador_atual = jogadas["Alice"]
    turno = 1
    historico = []
    tabela_memo = {}

    if exibir_graficos:
        mostrar_grafo(G, titulo="Estado Inicial (Conjunto Convexo Vazio)")

    while len(estado_convexo) < G.number_of_nodes():
        nome_jogador = "Alice" if jogador_atual == jogadas["Alice"] else "Bob"
        disponiveis = obter_vertices_disponiveis(G, estado_convexo)

        print(f"\n--- Turno {turno} (Vez de {nome_jogador}) ---")
        print(f"Conjunto P3-convexo atual ({len(estado_convexo)}/{G.number_of_nodes()}): {sorted(list(estado_convexo))}")
        print(f"Vértices disponíveis: {disponiveis}")

        if jogador_atual == jogador_escolhido:
            vertice = input(f"{nome_jogador.upper()} - Digite o vértice que deseja infectar: ").strip()
            if vertice not in disponiveis:
                print("Jogada inválida! O vértice já pertence ao conjunto convexo ou não existe.")
                continue
        else:
            vence_alice, utilidade, _, melhor_v = minimax(
                G,
                estado_convexo,
                jogador_atual,
                usar_memoizacao=True,
                reduzir_simetria=True,
                tabela_memo=tabela_memo,
            )
            vertice = melhor_v
            status_ia = (
                "Alice possui estratégia vencedora (+1)"
                if vence_alice
                else "Bob possui estratégia vencedora (-1)"
            )
            print(f"[Minimax] Avaliação: {status_ia} | Jogada escolhida por {nome_jogador}: {vertice}")

        propagados = aplicar_jogada_no_grafo(G, vertice, turno, nome_jogador)
        estado_convexo, _ = fechamento_convexo_p3(G, estado_convexo | {vertice})

        if propagados:
            print(f"-> Fechamento convexo P3 incorporou automaticamente: {propagados} ( nessa ordem )")
        else:
            print("-> Nenhuma propagação automática ocorreu neste turno.")

        historico.append(
            {
                "turno": turno,
                "jogador": nome_jogador,
                "vertice_escolhido": vertice,
                "propagados": propagados,
                "conjunto_convexo": sorted(list(estado_convexo)),
            }
        )

        if exibir_graficos:
            titulo_fig = (
                f"Turno {turno}: {nome_jogador} escolheu {vertice}"
                + (f" | Propagou: {propagados}" if propagados else "")
            )
            mostrar_grafo(G, titulo=titulo_fig)

        # Verifica término da partida
        if len(estado_convexo) == G.number_of_nodes():
            print("\n=======================================================")
            print(f"FIM DE JOGO no Turno {turno}! Todo o grafo foi infectado.")
            print(f"VENCEDOR(A): {nome_jogador} (realizou a última jogada válida!)")
            print("=======================================================")
            return nome_jogador, historico

        jogador_atual = jogadas["Bob"] if jogador_atual == jogadas["Alice"] else jogadas["Alice"]
        turno += 1

    return None, historico
