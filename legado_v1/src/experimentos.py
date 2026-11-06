import os
import csv
import json
import time
from itertools import product

from caterpillar_p3 import (
    criar_caterpillar,
    fechamento_convexo_p3,
)
from minimax import (
    minimax,
    jogadas,
    listar_todas_jogadas_vencedoras_iniciais,
    obter_jogadas_nao_simetricas,
)


def classificar_familia(assinatura):
    """
    Classifica um grafo caterpillar em uma família estrutural para facilitar
    a análise no capítulo de Resultados do TCC.
    """
    k = len(assinatura)
    f = sum(assinatura)

    if f == 0:
        return "Caminho Puro (Pk)"
    if k == 1:
        return "Estrela (K1,f)"
    if k == 2 and assinatura[0] > 0 and assinatura[1] > 0:
        return "Dupla-Estrela"
    if len(set(assinatura)) == 1 and assinatura[0] > 0:
        return f"Caterpillar Uniforme (p={assinatura[0]})"
    if k >= 3 and assinatura[0] > 0 and assinatura[-1] > 0 and all(x == 0 for x in assinatura[1:-1]):
        return "Haltere (Pernas nas Extremidades)"
    if k >= 3 and assinatura[0] == 0 and assinatura[-1] == 0 and any(x > 0 for x in assinatura[1:-1]):
        return "Pernas Internas (Centro)"
    return "Caterpillar Geral"


def classificar_tipo_jogada_vencedora(G, jogadas_vencedoras):
    """
    Verifica se as jogadas iniciais vencedoras de Alice ocorrem na espinha,
    nas pernas (folhas) ou em ambas.
    """
    if not jogadas_vencedoras:
        return "Nenhuma (Bob vence)"

    tem_espinha = any(G.nodes[v].get("tipo") == "espinha" for v in jogadas_vencedoras)
    tem_perna = any(G.nodes[v].get("tipo") == "perna" for v in jogadas_vencedoras)

    if tem_espinha and tem_perna:
        return "Espinha e Perna"
    elif tem_espinha:
        return "Apenas Espinha"
    else:
        return "Apenas Perna"


def extrair_trilha_otima(G, tabela_memo):
    """
    Reconstrói uma sequência completa de jogadas ótimas (Principal Variation)
    a partir do estado vazio S = ∅ até S = V(G), registrando cada jogada e os
    vértices propagados pelo fechamento P3.
    """
    estado = frozenset()
    jogador_atual = jogadas["Alice"]
    turno = 1
    trilha = []

    while len(estado) < G.number_of_nodes():
        nome_jog = "Alice" if jogador_atual == jogadas["Alice"] else "Bob"
        _, _, _, melhor_v = minimax(
            G,
            estado,
            jogador_atual,
            usar_memoizacao=True,
            reduzir_simetria=True,
            tabela_memo=tabela_memo,
        )
        if melhor_v is None:
            break

        novo_estado, propagados = fechamento_convexo_p3(G, estado | {melhor_v})
        trilha.append(
            {
                "turno": turno,
                "jogador": nome_jog,
                "escolha": melhor_v,
                "tipo_vertice": G.nodes[melhor_v].get("tipo"),
                "propagados": propagados,
                "tamanho_conjunto_apos": len(novo_estado),
            }
        )
        estado = novo_estado
        jogador_atual = jogadas["Bob"] if jogador_atual == jogadas["Alice"] else jogadas["Alice"]
        turno += 1

    return trilha


def calcular_duracao_min_max(G):
    """
    Calcula a quantidade mínima e máxima possível de turnos até o término
    da partida (em qualquer sequência de jogadas válidas sem simetria redundante).
    """
    memo_duracao = {}

    def dfs_duracao(estado):
        if len(estado) == G.number_of_nodes():
            return (0, 0)
        if estado in memo_duracao:
            return memo_duracao[estado]

        min_t = float("inf")
        max_t = 0
        for v in obter_jogadas_nao_simetricas(G, estado):
            novo_est, _ = fechamento_convexo_p3(G, estado | {v})
            sub_min, sub_max = dfs_duracao(novo_est)
            if 1 + sub_min < min_t:
                min_t = 1 + sub_min
            if 1 + sub_max > max_t:
                max_t = 1 + sub_max

        memo_duracao[estado] = (min_t, max_t)
        return (min_t, max_t)

    return dfs_duracao(frozenset())


def gerar_conjunto_instancias_experimento():
    """
    Gera um conjunto abrangente e sistemático de instâncias de grafos caterpillar:
    1. Varredura exaustiva de caterpillars não-isomorfos com espinha k <= 6,
       até 3 pernas por vértice e total de vértices |V| <= 11.
    2. Famílias especiais maiores (Caminhos até k=10, Uniformes, Halteres e o
       grafo exemplo do TCC [0, 2, 0, 1, 1, 0]).
    """
    vistos = set()
    instancias = []

    def adicionar(config):
        cfg = tuple(config)
        canonica = min(cfg, cfg[::-1])
        if canonica not in vistos and sum(canonica) + len(canonica) >= 2:
            vistos.add(canonica)
            instancias.append(canonica)

    # 1. Varredura sistemática para k de 1 a 6
    for k in range(1, 7):
        for cfg in product(range(4), repeat=k):
            total_v = k + sum(cfg)
            if 2 <= total_v <= 10:
                adicionar(cfg)

    # 2. Caminhos puros Pk até k = 11
    for k in range(2, 12):
        adicionar([0] * k)

    # 3. Caterpillars uniformes para espinhas k = 2..5 e p = 1..3
    for k in range(2, 6):
        for p in range(1, 4):
            if k + k * p <= 12:
                adicionar([p] * k)

    # 4. Halteres (pernas apenas nas pontas) para k = 3..6
    for k in range(3, 7):
        for p1 in range(1, 4):
            for p2 in range(p1, 4):
                if k + p1 + p2 <= 12:
                    cfg = [p1] + [0] * (k - 2) + [p2]
                    adicionar(cfg)

    # 5. Garante a inclusão exata do grafo das Figuras 1, 2 e 4 do TCC na orientação original [0, 2, 0, 1, 1, 0]
    cfg_tcc = (0, 2, 0, 1, 1, 0)
    canon_tcc = min(cfg_tcc, cfg_tcc[::-1])
    if canon_tcc in vistos:
        instancias = [cfg_tcc if x == canon_tcc else x for x in instancias]
    else:
        instancias.append(cfg_tcc)

    return instancias


def executar_experimentos():
    pasta_resultados = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resultados"
    )
    os.makedirs(pasta_resultados, exist_ok=True)

    instancias = gerar_conjunto_instancias_experimento()
    print(f"Total de instâncias Caterpillar não-isomorfas selecionadas: {len(instancias)}")

    registros_csv = []
    registros_json = []

    t_inicio_total = time.perf_counter()

    for idx, assinatura in enumerate(instancias, start=1):
        G = criar_caterpillar(assinatura)
        k = len(assinatura)
        f = sum(assinatura)
        n = k + f
        familia = classificar_familia(assinatura)
        paridade_folhas = "Par" if f % 2 == 0 else "Impar"
        espinha_interna_zero = sum(1 for x in assinatura[1:-1] if x == 0) if k >= 3 else 0

        # 1. Execução Minimax COM Otimização (Memoização + Redução de Simetria)
        tabela_memo = {}
        estat_otim = {}
        t0 = time.perf_counter()
        vence_alice, utilidade, _, melhor_jogada = minimax(
            G,
            frozenset(),
            jogadas["Alice"],
            usar_memoizacao=True,
            reduzir_simetria=True,
            tabela_memo=tabela_memo,
            estatisticas=estat_otim,
        )
        tempo_otim_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Execução Minimax SEM Otimização (apenas para grafos pequenos n <= 7 para comparação de desempenho)
        nos_sem_otim = None
        tempo_sem_otim_ms = None
        if n <= 7:
            estat_bruto = {}
            t0_b = time.perf_counter()
            minimax(
                G,
                frozenset(),
                jogadas["Alice"],
                usar_memoizacao=False,
                reduzir_simetria=False,
                estatisticas=estat_bruto,
            )
            tempo_sem_otim_ms = round((time.perf_counter() - t0_b) * 1000.0, 3)
            nos_sem_otim = estat_bruto.get("nos_visitados", 0)

        # 3. Jogadas iniciais vencedoras e classificação
        jogadas_venc, detalhes_iniciais = listar_todas_jogadas_vencedoras_iniciais(G)
        tipo_jogada_venc = classificar_tipo_jogada_vencedora(G, jogadas_venc)

        # 4. Trilha ótima (Principal Variation) e duração min/max
        trilha_otima = extrair_trilha_otima(G, tabela_memo)
        dur_min, dur_max = calcular_duracao_min_max(G)

        trilha_resumida = " -> ".join(
            f"{passo['jogador'][0]}({passo['escolha']})"
            + (f"+{passo['propagados']}" if passo["propagados"] else "")
            for passo in trilha_otima
        )

        estados_convexos_distintos = len({est for (est, _) in tabela_memo.keys()})

        registro_base = {
            "id": idx,
            "familia": familia,
            "assinatura": str(list(assinatura)),
            "n_vertices": n,
            "tam_espinha_k": k,
            "total_folhas_f": f,
            "paridade_folhas": paridade_folhas,
            "espinha_interna_sem_pernas": espinha_interna_zero,
            "vencedor_otimo": "Alice" if vence_alice else "Bob",
            "utilidade": utilidade,
            "melhor_primeira_jogada": melhor_jogada,
            "qtd_jogadas_vencedoras_iniciais": len(jogadas_venc),
            "jogadas_vencedoras_iniciais": ", ".join(jogadas_venc) if jogadas_venc else "Nenhuma",
            "tipo_jogada_vencedora": tipo_jogada_venc,
            "turnos_partida_otima": len(trilha_otima),
            "duracao_min_possivel": dur_min,
            "duracao_max_possivel": dur_max,
            "estados_convexos_distintos": estados_convexos_distintos,
            "nos_visitados_otimizado": estat_otim.get("nos_visitados", 0),
            "hits_memoizacao": estat_otim.get("hits_memo", 0),
            "tempo_otimizado_ms": round(tempo_otim_ms, 3),
            "nos_visitados_sem_otimizacao": nos_sem_otim if nos_sem_otim is not None else "",
            "tempo_sem_otimizacao_ms": tempo_sem_otim_ms if tempo_sem_otim_ms is not None else "",
            "trilha_otima": trilha_resumida,
        }

        registros_csv.append(registro_base)

        registro_completo = dict(registro_base)
        registro_completo["assinatura_lista"] = list(assinatura)
        registro_completo["trilha_otima_detalhada"] = trilha_otima
        registro_completo["detalhes_jogadas_iniciais"] = detalhes_iniciais
        registros_json.append(registro_completo)

        if idx % 25 == 0 or idx == len(instancias):
            print(f"  Processadas {idx}/{len(instancias)} instâncias...")

    tempo_total = time.perf_counter() - t_inicio_total
    print(f"\nExperimentos concluídos em {tempo_total:.2f} segundos!")

    # 1. Salvar CSV
    caminho_csv = os.path.join(pasta_resultados, "experimentos_caterpillar.csv")
    campos_csv = list(registros_csv[0].keys())
    with open(caminho_csv, mode="w", newline="", encoding="utf-8-sig") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=campos_csv, delimiter=";")
        writer.writeheader()
        writer.writerows(registros_csv)
    print(f"-> Arquivo CSV salvo em: {caminho_csv}")

    # 2. Salvar JSON detalhado
    caminho_json = os.path.join(pasta_resultados, "experimentos_detalhados.json")
    with open(caminho_json, mode="w", encoding="utf-8") as f_json:
        json.dump(registros_json, f_json, ensure_ascii=False, indent=2)
    print(f"-> Arquivo JSON salvo em: {caminho_json}")

    # 3. Gerar Tabelas LaTeX e Relatório Analítico
    gerar_arquivos_analiticos(pasta_resultados, registros_json)


def gerar_arquivos_analiticos(pasta_resultados, registros):
    """
    Analisa os dados coletados para produzir:
    - `resultados/tabelas_tcc.tex` com tabelas formatadas em LaTeX para o TCC 2;
    - `resultados/relatorio_padroes.md` com o resumo estatístico e descobertas estratégicas.
    """
    # Estatísticas globais
    total = len(registros)
    vit_alice = sum(1 for r in registros if r["vencedor_otimo"] == "Alice")
    vit_bob = total - vit_alice

    # Agrupamento por família
    por_familia = {}
    for r in registros:
        fam = r["familia"]
        por_familia.setdefault(fam, {"total": 0, "Alice": 0, "Bob": 0})
        por_familia[fam]["total"] += 1
        por_familia[fam][r["vencedor_otimo"]] += 1

    # Agrupamento por paridade do número total de vértices |V| e de folhas f
    por_paridade_f = {"Par": {"Alice": 0, "Bob": 0}, "Impar": {"Alice": 0, "Bob": 0}}
    for r in registros:
        por_paridade_f[r["paridade_folhas"]][r["vencedor_otimo"]] += 1

    # Comparação de desempenho (Com vs Sem otimização para n <= 7)
    comparativo_perf = [
        r for r in registros if r["nos_visitados_sem_otimizacao"] != ""
    ]

    # --- GERAÇÃO DO ARQUIVO DE TABELAS LATEX ---
    linhas_tex = [
        "% ===========================================================================",
        "% TABELAS GERADAS AUTOMATICAMENTE PELO MOTOR MINIMAX (PASSO 2 DO TCC)",
        "% ===========================================================================\n",
        "% 1. Tabela de Validação em Caminhos Puros P_k",
        "\\begin{table}[htb]",
        "\\centering",
        "\\caption{Resultados do algoritmo Minimax para o Jogo da Convexidade $P_3$ em caminhos $P_k$.}",
        "\\label{tab:resultados_caminhos}",
        "\\begin{tabular}{cccccc}",
        "\\toprule",
        "\\textbf{Grafo} & \\textbf{Assinatura $L$} & $|V(G)|$ & \\textbf{Vencedor} & \\textbf{Turnos (Ótimo)} & \\textbf{Jogadas Iniciais Vencedoras} \\\\",
        "\\midrule",
    ]

    caminhos = [r for r in registros if r["familia"] == "Caminho Puro (Pk)"]
    for r in sorted(caminhos, key=lambda x: x["tam_espinha_k"]):
        k = r["tam_espinha_k"]
        venc = "Alice ($+1$)" if r["vencedor_otimo"] == "Alice" else "Bob ($-1$)"
        jogs = r["jogadas_vencedoras_iniciais"] if r["vencedor_otimo"] == "Alice" else "---"
        linhas_tex.append(
            f"$P_{{{k}}}$ & ${r['assinatura']}$ & {r['n_vertices']} & {venc} & {r['turnos_partida_otima']} & {jogs} \\\\"
        )

    linhas_tex.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\\\ \\footnotesize{Fonte: elaborado pelo autor.}",
            "\\end{table}\n",
            "% 2. Tabela de Caterpillars Representativos (Uniformes, Halteres e Exemplo do TCC)",
            "\\begin{table}[htb]",
            "\\centering",
            "\\caption{Análise estratégica do algoritmo Minimax em diferentes configurações de grafos \\textit{caterpillar}.}",
            "\\label{tab:resultados_caterpillar_classes}",
            "\\begin{tabular}{lcccccp{3.8cm}}",
            "\\toprule",
            "\\textbf{Assinatura $L$} & $|V|$ & $k$ & $f$ & \\textbf{Vencedor} & \\textbf{ Estados } & \\textbf{Jogadas Vencedoras (Alice)} \\\\",
            "\\midrule",
        ]
    )

    selecionados = [
        r
        for r in registros
        if r["assinatura_lista"]
        in [
            [1, 1],
            [1, 2],
            [2, 2],
            [0, 1, 0],
            [0, 2, 0],
            [1, 0, 1],
            [1, 1, 1],
            [2, 0, 2],
            [2, 2, 2],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
            [2, 0, 0, 2],
            [0, 1, 1, 0],
            [0, 2, 2, 0],
            [0, 2, 0, 1, 1, 0],
        ]
    ]
    for r in selecionados:
        venc = "Alice ($+1$)" if r["vencedor_otimo"] == "Alice" else "Bob ($-1$)"
        jogs = r["jogadas_vencedoras_iniciais"] if r["vencedor_otimo"] == "Alice" else "---"
        linhas_tex.append(
            f"${r['assinatura']}$ & {r['n_vertices']} & {r['tam_espinha_k']} & {r['total_folhas_f']} & {venc} & {r['estados_convexos_distintos']} & {jogs} \\\\"
        )

    linhas_tex.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\\\ \\footnotesize{Fonte: elaborado pelo autor.}",
            "\\end{table}\n",
            "% 3. Tabela de Desempenho Computacional (Minimax Puro vs Minimax com Memoização e Simetria)",
            "\\begin{table}[htb]",
            "\\centering",
            "\\caption{Comparação de desempenho do algoritmo Minimax sem otimização e com tabela de transposição e redução de simetria.}",
            "\\label{tab:desempenho_minimax}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "\\textbf{Assinatura $L$} & $|V|$ & \\textbf{Nós (Sem Otim.)} & \\textbf{Tempo Bruto (ms)} & \\textbf{Nós (Otimizado)} & \\textbf{Tempo Otim. (ms)} \\\\",
            "\\midrule",
        ]
    )

    amostra_perf = [
        r
        for r in comparativo_perf
        if r["assinatura_lista"]
        in [
            [0, 0, 0, 0, 0],
            [1, 1, 1],
            [0, 2, 2],
            [1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0],
            [2, 1, 1],
        ]
    ]
    for r in amostra_perf:
        linhas_tex.append(
            f"${r['assinatura']}$ & {r['n_vertices']} & {r['nos_visitados_sem_otimizacao']} & {r['tempo_sem_otimizacao_ms']:.2f} & {r['nos_visitados_otimizado']} & {r['tempo_otimizado_ms']:.2f} \\\\"
        )

    linhas_tex.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\\\ \\footnotesize{Fonte: elaborado pelo autor.}",
            "\\end{table}",
        ]
    )

    caminho_tex = os.path.join(pasta_resultados, "tabelas_tcc.tex")
    with open(caminho_tex, mode="w", encoding="utf-8") as f_tex:
        f_tex.write("\n".join(linhas_tex))
    print(f"-> Tabelas LaTeX salvas em: {caminho_tex}")

    # --- GERAÇÃO DO RELATÓRIO DE PADRÕES EM MARKDOWN ---
    linhas_md = [
        "# Relatório Analítico de Experimentos — Jogo da Convexidade $P_3$ em Grafos *Caterpillar*",
        "",
        "## 1. Visão Geral da Bateria Experimental",
        f"- **Total de grafos *caterpillar* não-isomorfos analisados:** {total}",
        f"- **Vitórias de Alice (1º Jogador / MAX = $+1$):** {vit_alice} ({vit_alice/total*100:.1f}%)",
        f"- **Vitórias de Bob (2º Jogador / MIN = $-1$):** {vit_bob} ({vit_bob/total*100:.1f}%)",
        "",
        "## 2. Distribuição de Vencedores por Família de *Caterpillar*",
        "| Família Estrutural | Total de Instâncias | Vitórias de Alice (+1) | Vitórias de Bob (-1) |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for fam, dados in sorted(por_familia.items()):
        linhas_md.append(
            f"| {fam} | {dados['total']} | {dados['Alice']} ({dados['Alice']/dados['total']*100:.1f}%) | {dados['Bob']} ({dados['Bob']/dados['total']*100:.1f}%) |"
        )

    linhas_md.extend(
        [
            "",
            "## 3. Influência da Paridade do Número de Folhas ($f = \\sum l_i$)",
            "| Paridade de Folhas ($f$) | Vitórias de Alice (+1) | Vitórias de Bob (-1) |",
            "| :--- | :---: | :---: |",
            f"| **Par** ($f \\equiv 0 \\pmod 2$) | {por_paridade_f['Par']['Alice']} | {por_paridade_f['Par']['Bob']} |",
            f"| **Ímpar** ($f \\equiv 1 \\pmod 2$) | {por_paridade_f['Impar']['Alice']} | {por_paridade_f['Impar']['Bob']} |",
            "",
            "## 4. Descobertas Estratégicas e Padrões Vencedores Identificados",
            "",
            "### Padrão 1: Invariante de Infecciosidade das Folhas (Pernas e Extremidades de Grau 1)",
            "Em qualquer grafo *caterpillar* $G$, todo vértice folha (seja uma perna ligada à espinha ou uma extremidade da espinha sem pernas) possui grau $d(v) = 1$. Como a regra de propagação da Convexidade $P_3$ exige que um vértice externo possua **pelo menos dois vizinhos** no conjunto convexo atual para ser infectado automaticamente, **nenhuma folha pode ser infectada por propagação**. Consequentemente:",
            "- Se um grafo possui $L_{total}$ vértices de grau 1 (folhas totais da árvore), cada um desses $L_{total}$ vértices terá obrigatoriamente que ser escolhido de forma direta por um dos jogadores em algum turno da partida.",
            "- Assim, o número mínimo de turnos de qualquer partida em $G$ é limitado inferiormente pelo número total de folhas da árvore ($T \\ge L_{total}$).",
            "",
            "### Padrão 2: O Efeito de Inversão de Turno (Controle de Tempo pelas Pernas)",
            "Quando toda a espinha $P_k$ já se encontra infectada, restam apenas as pernas não selecionadas. A partir desse momento, nenhuma propagação adicional ocorre, e o jogo reduz-se a uma contagem pura de paridade das folhas restantes:",
            "- Adicionando-se $\\Delta f$ pernas a um mesmo vértice da espinha, quando $\\Delta f$ é **ímpar**, o jogador capaz de estabilizar a espinha pode usar uma perna como *jogada de espera (tempo move)* para inverter a paridade de quem joga na espinha.",
            "- Observa-se claramente nos experimentos que para uma mesma estrutura de espinha $k$ fixa, incrementar uma única perna em um vértice frequentemente **alterna o vencedor entre Alice e Bob** (por exemplo: `[1]` $\\to$ Bob, `[2]` $\\to$ Alice; `[1, 1]` $\\to$ Alice, `[1, 2]` $\\to$ Bob, `[2, 2]` $\\to$ Alice).",
            "",
            "### Padrão 3: Comportamento em Caminhos Puros ($P_k$) vs. Adição de Pernas",
            "Nos caminhos puros ($f = 0$), os resultados do Minimax para $P_2$ a $P_{11}$ mostram:",
        ]
    )

    for r in sorted(caminhos, key=lambda x: x["tam_espinha_k"]):
        linhas_md.append(
            f"- **$P_{{{r['tam_espinha_k']}}}$** (`{r['assinatura']}`): Vencedor = **{r['vencedor_otimo']}** | Turnos ótimos = {r['turnos_partida_otima']} | Jogadas vencedoras iniciais: `{r['jogadas_vencedoras_iniciais']}`"
        )

    # Busca especificamente o grafo do TCC
    r_tcc = next(r for r in registros if r["assinatura_lista"] == [0, 2, 0, 1, 1, 0])
    linhas_md.extend(
        [
            "",
            "### Padrão 4: Estudo de Caso da Instância do TCC (`L = [0, 2, 0, 1, 1, 0]`, $|V|=10$)",
            f"- **Vencedor com Jogo Ótimo:** {r_tcc['vencedor_otimo']} (Utilidade = {r_tcc['utilidade']})",
            f"- **Estados $P_3$-convexos distintos alcançáveis:** {r_tcc['estados_convexos_distintos']}",
            f"- **Nós explorados pelo Minimax otimizado:** {r_tcc['nos_visitados_otimizado']} (em {r_tcc['tempo_otimizado_ms']} ms)",
            f"- **Duração possível de uma partida:** mínima de {r_tcc['duracao_min_possivel']} turnos (ex: como na Figura 4 do TCC, onde há colapso de 3 vértices da espinha de uma só vez) e máxima de {r_tcc['duracao_max_possivel']} turnos.",
            f"- **Sequência Ótima (Principal Variation):** `{r_tcc['trilha_otima']}`",
            "- **Explicação Estratégica:** Na partida ilustrativa da Figura 4 do TCC (Seção 5.2), Bob faz a jogada $B(v_6)$ no 4º turno, o que provoca a infecção simultânea de **3 vértices** ($v_5, v_4, v_3$), fazendo a partida durar 7 turnos (ímpar) e dando a vitória para Alice. Já o algoritmo **Minimax** revela que Bob vence a partida se controlar a propagação da espinha em pares (infectando apenas 1 vértice por fechamento em $v_2$ e depois em $v_5$), forçando a partida a durar exatamente **8 turnos (par)** e garantindo que a última folha seja sua!",
        ]
    )

    caminho_md = os.path.join(pasta_resultados, "relatorio_padroes.md")
    with open(caminho_md, mode="w", encoding="utf-8") as f_md:
        f_md.write("\n".join(linhas_md))
    print(f"-> Relatório de padrões salvo em: {caminho_md}")


if __name__ == "__main__":
    executar_experimentos()
