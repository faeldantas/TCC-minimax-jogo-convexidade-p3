import time
from caterpillar_p3 import (
    criar_caterpillar,
    fechamento_convexo_p3,
    aplicar_jogada_no_grafo,
    gerar_familias_caterpillar,
)
from minimax import (
    minimax,
    jogadas,
    listar_todas_jogadas_vencedoras_iniciais,
    gerar_arvore_de_jogo,
    jogo_convexidade_p3,
)
from visualizacao import mostrar_grafo


def validar_exemplo_tcc():
    """
    Valida a implementação reproduzindo exatamente o grafo e a partida
    apresentados nas Figuras 1, 2 e 4 (Seção 5.2) do TCC:
    - Espinha: v1, v2, v3, v4, v5, v6
    - Pernas: v7, v8 (ligadas a v2), v9 (ligada a v4), v10 (ligada a v5)
    - Assinatura de pernas: [0, 2, 0, 1, 1, 0]
    """
    print("=" * 70)
    print("1. VALIDAÇÃO COM O GRAFO DO TCC (Figuras 1, 2 e 4 - Seção 5.2)")
    print("=" * 70)

    assinatura_tcc = [0, 2, 0, 1, 1, 0]
    G = criar_caterpillar(assinatura_tcc)

    print(f"Vértices da espinha : {G.graph['espinha']}")
    print(f"Mapa de pernas      : {G.graph['mapa_pernas']}")

    # Teste do Fechamento Convexo da Figura 2 (Seção 3.2): S = {v2, v6, v9, v10}
    conjunto_fig2 = {"v2", "v6", "v9", "v10"}
    fechamento_fig2, prop_fig2 = fechamento_convexo_p3(G, conjunto_fig2)
    print(f"\n[Teste Figura 2] Conjunto inicial : {sorted(list(conjunto_fig2))}")
    print(f"[Teste Figura 2] Ordem propagação : {prop_fig2} (Esperado: ['v5', 'v4', 'v3'])")
    print(f"[Teste Figura 2] Conjunto final   : {sorted(list(fechamento_fig2))}")

    # Teste da Partida da Figura 4 (Seção 5.2):
    # A(v9) - B(v10) - A(v2) - B(v6) [propaga v5, v4, v3] - A(v7) - B(v8) - A(v1)
    sequencia_fig4 = [
        ("Alice", "v9"),
        ("Bob", "v10"),
        ("Alice", "v2"),
        ("Bob", "v6"),
        ("Alice", "v7"),
        ("Bob", "v8"),
        ("Alice", "v1"),
    ]
    print("\n[Teste Figura 4] Reproduzindo partida passo a passo:")
    for turno, (jog_nome, vert) in enumerate(sequencia_fig4, start=1):
        prop = aplicar_jogada_no_grafo(G, vert, turno, jog_nome)
        msg_prop = f" -> Fechamento P3 incorporou: {prop}" if prop else ""
        print(f"  Jogada {turno}: {jog_nome}({vert}){msg_prop}")

    # Análise Minimax completa para o grafo do TCC
    G_limpo = criar_caterpillar(assinatura_tcc)
    estatisticas = {}
    t0 = time.perf_counter()
    vence_alice, utilidade, _, melhor_jogada = minimax(
        G_limpo,
        frozenset(),
        jogadas["Alice"],
        usar_memoizacao=True,
        reduzir_simetria=True,
        estatisticas=estatisticas,
    )
    dt = (time.perf_counter() - t0) * 1000

    vencedoras_iniciais, _ = listar_todas_jogadas_vencedoras_iniciais(G_limpo)
    vencedor_otimo = "Alice (1º Jogador)" if vence_alice else "Bob (2º Jogador)"

    print("\n[Análise Minimax do Grafo da Figura 1/4 do TCC]")
    print(f"  Vencedor com jogo ótimo : {vencedor_otimo} (Utilidade = {utilidade:+d})")
    print(f"  Melhor 1ª jogada        : {melhor_jogada}")
    print(f"  Todas 1ªs jogadas ótimas: {vencedoras_iniciais}")
    print(f"  Nós visitados (c/ memo) : {estatisticas.get('nos_visitados', 0)}")
    print(f"  Hits na tabela memo     : {estatisticas.get('hits_memo', 0)}")
    print(f"  Tempo de execução       : {dt:.2f} ms")


def validar_figura_3_caminho_p3():
    """
    Valida a Figura 3 do TCC (Seção 3.3.1): Caminho P3 com 3 vértices [0, 0, 0].
    Na Figura 3, se Alice joga no centro (v2), restam {v1} e {v3}; qualquer jogada
    de Bob infecta os 2 vizinhos de algum vértice? Não, se v2 já está infectado e Bob
    escolhe v1, então {v1, v2} estão infectados (não fecha v3 pois v3 só é vizinho de v2!).
    Se Alice escolhe v1 (ponta) e Bob escolhe v3 (outra ponta), v2 tem 2 vizinhos (v1 e v3)
    e fecha o grafo todo no turno 2 (Bob vence!).
    """
    print("\n" + "=" * 70)
    print("2. VALIDAÇÃO DA FIGURA 3 DO TCC (Caminho P3 = [0, 0, 0])")
    print("=" * 70)
    G_p3 = criar_caterpillar([0, 0, 0])
    vencedoras, detalhes = listar_todas_jogadas_vencedoras_iniciais(G_p3)
    for v, info in detalhes.items():
        status = "+1 (Alice vence)" if info["vence_alice"] else "-1 (Bob vence)"
        print(
            f"  Se Alice inicia em {v}: {status} | Resposta ótima de Bob: {info['melhor_resposta_bob']}"
        )
    arvore_p3 = gerar_arvore_de_jogo(G_p3)
    print(
        f"  Árvore de estados gerada para P3: {arvore_p3.number_of_nodes()} estados distintos e {arvore_p3.number_of_edges()} transições."
    )


def executar_amostra_experimentos():
    """
    Executa uma bateria rápida sobre famílias de grafos Caterpillar para demonstrar
    a coleta de dados que será utilizada no Passo 2 do TCC.
    """
    print("\n" + "=" * 70)
    print("3. AMOSTRA DE EXPERIMENTOS EM GRAFOS CATERPILLAR")
    print("=" * 70)
    print(
        f"{'Assinatura (Pernas)':<24} | {'|V|':>3} | {'Esp.':>4} | {'Folhas':>6} | {'Vencedor Ótimo':<16} | {'Jogadas Vencedoras (Alice)'}"
    )
    print("-" * 95)

    instancias = gerar_familias_caterpillar(
        max_espinha=4, max_pernas_por_vertice=2, max_total_vertices=8
    )

    for ass in instancias[:18]:
        G = criar_caterpillar(ass)
        vence_alice, util, _, _ = minimax(G)
        vencedoras, _ = listar_todas_jogadas_vencedoras_iniciais(G)
        vencedor_str = "Alice (+1)" if vence_alice else "Bob (-1)"
        k = len(ass)
        f = sum(ass)
        n = k + f
        print(
            f"{str(list(ass)):<24} | {n:>3} | {k:>4} | {f:>6} | {vencedor_str:<16} | {vencedoras}"
        )


def main():
    validar_exemplo_tcc()
    validar_figura_3_caminho_p3()
    executar_amostra_experimentos()

    print("\n" + "=" * 70)
    print("4. SIMULAÇÃO DE PARTIDA COMPLETA IA vs IA (Minimax) NO GRAFO DO TCC")
    print("=" * 70)
    G_partida = criar_caterpillar([0, 2, 0, 1, 1, 0])
    jogo_convexidade_p3(G_partida, modo_interativo=False, exibir_graficos=False)


if __name__ == "__main__":
    main()
