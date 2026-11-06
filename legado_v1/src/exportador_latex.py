def exportar_arvore_tikz(arvore, titulo_legenda="Árvore de jogo gerada para a instância analisada."):
    """
    Exporta uma árvore/DAG de jogo (gerada por `gerar_arvore_de_jogo`) para um
    bloco de código LaTeX/TikZ no estilo das Figuras 3 e 5 do TCC.
    """
    linhas = [
        "\\begin{figure}[htb]",
        "\\centering",
        f"\\caption{{{titulo_legenda}}}",
        "\\begin{tikzpicture}[",
        "    every node/.style={draw, rounded corners=2pt, minimum height=0.6cm, font=\\small},",
        "    edge from parent/.style={draw, -latex}",
        "]",
    ]

    # Agrupa os nós por nível (tamanho do conjunto ou distância a partir da raiz)
    niveis = {}
    for (estado, jogador), dados in arvore.nodes(data=True):
        tam = len(estado)
        niveis.setdefault(tam, []).append(((estado, jogador), dados))

    mapa_id = {}
    contador = 0
    espacamento_y = -1.8
    espacamento_x = 3.2

    for idx_nivel, tam in enumerate(sorted(niveis.keys())):
        nos_nivel = niveis[tam]
        largura = len(nos_nivel)
        for i, (chave, dados) in enumerate(nos_nivel):
            contador += 1
            node_id = f"N{contador}"
            mapa_id[chave] = node_id

            x = (i - (largura - 1) / 2.0) * espacamento_x
            y = idx_nivel * espacamento_y

            estado_lista = sorted(list(chave[0]))
            texto_estado = "\\emptyset" if not estado_lista else "\\{" + ", ".join(estado_lista) + "\\}"
            util = dados.get("utilidade", 0)
            sinal_util = f"+{util}" if util > 0 else f"{util}"

            linhas.append(
                f"    \\node ({node_id}) at ({x:.2f}, {y:.2f}) {{${texto_estado}$ \\; ({sinal_util})}};"
            )

    linhas.append("")
    for u, v, dados_aresta in arvore.edges(data=True):
        id_u = mapa_id[u]
        id_v = mapa_id[v]
        jog = dados_aresta.get("jogada", "")
        linhas.append(
            f"    \\draw[-latex] ({id_u}) -- node[draw=none, midway, fill=white, inner sep=1pt, font=\\scriptsize] {{{jog}}} ({id_v});"
        )

    linhas.extend(
        [
            "\\end{tikzpicture}",
            "\\\\ \\footnotesize{Fonte: elaborado pelo autor.}",
            "\\end{figure}",
        ]
    )
    return "\n".join(linhas)


def exportar_tabela_latex(resultados_experimentos):
    """
    Exporta uma lista de resultados experimentais em grafos Caterpillar para uma
    tabela LaTeX formatada no padrão ABNT/UFC usado no TCC.
    """
    linhas = [
        "\\begin{table}[htb]",
        "\\centering",
        "\\caption{Resultados do algoritmo Minimax para o Jogo da Convexidade $P_3$ em grafos \\textit{caterpillar}.}",
        "\\label{tab:resultados_minimax_caterpillar}",
        "\\begin{tabular}{lccccp{4.5cm}}",
        "\\toprule",
        "\\textbf{Assinatura $L$} & $|V(G)|$ & \\textbf{Espinha ($k$)} & \\textbf{Folhas ($f$)} & \\textbf{Vencedor} & \\textbf{Jogadas Iniciais Ótimas} \\\\",
        "\\midrule",
    ]

    for item in resultados_experimentos:
        ass_str = "(" + ", ".join(str(x) for x in item["assinatura"]) + ")"
        n = item["n"]
        k = item["k"]
        f = item["folhas"]
        venc = "Alice ($+1$)" if item["vence_alice"] else "Bob ($-1$)"
        jogadas_str = ", ".join(item["jogadas_vencedoras"]) if item["jogadas_vencedoras"] else "---"
        linhas.append(f"${ass_str}$ & {n} & {k} & {f} & {venc} & {jogadas_str} \\\\")

    linhas.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\\\ \\footnotesize{Fonte: elaborado pelo autor.}",
            "\\end{table}",
        ]
    )
    return "\n".join(linhas)
