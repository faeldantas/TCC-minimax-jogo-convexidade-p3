"""
Ponto de entrada principal (CLI) do pacote `p3game`.

Funcionalidades:
1. Experimento Iterativo: Executa a análise para caterpillars cheios com 2 pernas
   por nó da espinha (L=(2,)*m), expandindo m até 100, com interrupção por Ctrl+C.
2. Visualização Gráfica (--visualizar): Permite ver graficamente a partida e as
   propagações P_3 em painéis PNG de alta resolução, visualizador interativo em
   janela Matplotlib com sliders/botões, e aplicação web autônoma HTML5/SVG.
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser

# Garante importação do pacote p3game
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p3game import Caterpillar, resolver_caterpillar
from p3game.minimax import LimiteDeNosExcedido
from p3game.experimentos import (
    analisar_caterpillar_cheio_2_pernas,
    salvar_resultados_caterpillar_cheio,
    assinatura_caterpillar_cheio,
)
from p3game.visualizacao import (
    gerar_painel_estatico,
    abrir_visualizador_interativo,
    gerar_html_interativo,
    gerar_interface_jogavel_html,
    jogar_contra_ia_terminal,
)


def formatar_cabecalho_tabela() -> str:
    cabecalho = (
        f"{'m':>3} | {'n':>3} | {'Folhas':>6} | "
        f"{'Solução Ótima':^13} | {'Alice':^5} | {'Bob':^5} | {'Turnos':^6} | "
        f"{'Propagados':^10} | {'Tempo':^11} | {'Nós Minimax':^11} | {'Hits (%)':^8} | "
        f"{'1ª Jogada Vencedora (Alice)':<26}"
    )
    linha_div = "-" * len(cabecalho)
    return f"{linha_div}\n{cabecalho}\n{linha_div}"


def formatar_linha_tabela(r: dict) -> str:
    venc_str = f"{r['vencedor_otimo']} ({r['utilidade']:+d})"
    primeira_jogada = (
        r["jogadas_vencedoras_iniciais_alice"]
        if r["vencedor_otimo"] == "Alice"
        else "--- (Bob vence forçado)"
    )

    if r["tempo_ms"] < 1000.0:
        tempo_str = f"{r['tempo_ms']:7.2f} ms"
    else:
        tempo_str = f"{r['tempo_s']:7.2f} s "

    return (
        f"{r['m_espinha']:3d} | {r['n_vertices']:3d} | "
        f"{r['folhas_totais']:2d} ({r['paridade_folhas'][0]}) | "
        f"{venc_str:^13} | {r['jogadas_alice']:5d} | {r['jogadas_bob']:5d} | {r['total_turnos']:6d} | "
        f"{r['vertices_propagados']:2d} ({r['taxa_propagacao_perc']:4.1f}%) | "
        f"{tempo_str:^11} | {r['nos_visitados']:11d} | {r['taxa_hits_perc']:7.1f}% | "
        f"{primeira_jogada:<26}"
    )


def imprimir_detalhe_instancia(r: dict) -> None:
    print(f"    Assinatura: {r['assinatura']}")
    print(f"    Trilha Ótima: {r['trilha_otima']}")


def imprimir_sintese_final(registros: list[dict], interrompido: bool = False) -> None:
    total = len(registros)
    if total == 0:
        return

    vitorias_alice = sum(1 for r in registros if r["vencedor_otimo"] == "Alice")
    vitorias_bob = sum(1 for r in registros if r["vencedor_otimo"] == "Bob")
    tempo_total_s = sum(r["tempo_s"] for r in registros)
    total_nos = sum(r["nos_visitados"] for r in registros)
    total_prop = sum(r["vertices_propagados"] for r in registros)
    total_vert = sum(r["n_vertices"] for r in registros)

    print("\n" + "=" * 95)
    print("SÍNTESE ANALÍTICA DO EXPERIMENTO" + (" (INTERROMPIDO PELO USUÁRIO)" if interrompido else " (CONCLUÍDO)"))
    print("=" * 95)
    print(f"1. Solubilidade Ótima e Resultados:")
    print(f"   - Tamanhos de espinha avaliados: m = {registros[0]['m_espinha']} até m = {registros[-1]['m_espinha']} (Total: {total} instâncias)")
    print(f"   - Soluções ótimas provadas pelo Minimax: 100% ({total}/{total})")
    print(f"   - Vitórias de Alice (inicia): {vitorias_alice} ({100 * vitorias_alice / total:.1f}%)")
    print(f"   - Vitórias de Bob (2º a jogar): {vitorias_bob} ({100 * vitorias_bob / total:.1f}%)")

    print(f"\n2. Paridade e Quantidade de Jogadas (Alice vs Bob):")
    print(f"   - Vitórias de Alice: Total de turnos sempre ÍMPAR (J_Alice = J_Bob + 1).")
    print(f"   - Vitórias de Bob:   Total de turnos sempre PAR   (J_Alice = J_Bob).")

    print(f"\n3. Fecho Convexo P_3 (Efeito Cascata):")
    perc_prop = (total_prop / total_vert) * 100.0 if total_vert > 0 else 0
    print(f"   - Total de vértices nas árvores: {total_vert}")
    print(f"   - Vértices propagados por fecho P_3: {total_prop} ({perc_prop:.1f}% dos vértices infectados sem consumir lances)")

    print(f"\n4. Métricas de Desempenho e Eficiência:")
    print(f"   - Tempo total acumulado de processamento: {tempo_total_s:.3f} s")
    print(f"   - Média de nós Minimax visitados por instância: {total_nos / total:.1f} nós")
    print(f"   - Poda Alfa-Beta por curto-circuito + Memoização canônica garantem solução exata.")
    print("=" * 95)


def executar_modo_visualizacao(args) -> None:
    """Executa a visualização gráfica (PNG, Janela Matplotlib e HTML Interativo)."""
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pasta_vis = os.path.join(raiz, "resultados", "visualizacao")
    os.makedirs(pasta_vis, exist_ok=True)

    if args.assinatura:
        L = tuple(int(x.strip()) for x in args.assinatura.split(","))
        nome_inst = f"caterpillar_L_{'_'.join(str(x) for x in L)}"
        titulo_vis = f"Caterpillar com Assinatura L = {L}"
    else:
        m = args.m_visualizar
        L = assinatura_caterpillar_cheio(m, modo=args.modo)
        nome_inst = f"caterpillar_cheio_m{m}_{args.modo}"
        titulo_vis = f"Caterpillar Cheio (m={m}, L={L})"

    cat = Caterpillar(L)
    print("=" * 95)
    print("VISUALIZAÇÃO GRÁFICA DO JOGO DA CONVEXIDADE P_3")
    print("=" * 95)
    print(f"Grafo selecionado: {titulo_vis}")
    print(f"Vértices totais: n = {cat.n} | Espinha: k = {cat.k} | Folhas: {len(cat.folhas)}")
    print("Resolvendo jogo ótimo com Minimax...")

    v, solver = resolver_caterpillar(L, corte=True, memoizacao=True, simetria=True)
    pv = solver.variacao_principal()

    venc = "Alice (+1)" if v == 1 else "Bob (-1)"
    print(f"Resultado: Vencedor = {venc} em {len(pv)} turnos (Propagados: {cat.n - len(pv)})")

    # 1. Gera Imagem PNG Estática com a Grade de Todos os Turnos
    caminho_png = os.path.join(pasta_vis, f"{nome_inst}.png")
    gerar_painel_estatico(cat, pv, caminho_png)
    print(f"[Painel PNG Gerado]: {caminho_png}")

    # 2. Gera Aplicação Web HTML5/SVG Interativa
    caminho_html = os.path.join(pasta_vis, f"{nome_inst}.html")
    gerar_html_interativo(cat, pv, caminho_html, titulo=titulo_vis)
    print(f"[Visualizador HTML Interativo Gerado]: {caminho_html}")

    # 3. Abre no Navegador se solicitado
    if args.navegador:
        print("Abrindo visualizador interativo no navegador padrão...")
        webbrowser.open(f"file://{os.path.abspath(caminho_html)}")

    # 4. Abre Janela Interativa Matplotlib se solicitado
    if args.janela:
        print("Abrindo janela interativa Matplotlib com controles de turno...")
        abrir_visualizador_interativo(cat, pv)


def main():
    parser = argparse.ArgumentParser(
        description="P3-Game: Experimento de Caterpillars Cheios e Visualização Gráfica das Propagações."
    )

    # Modo de Execução
    parser.add_argument(
        "--jogar",
        action="store_true",
        help="Inicia o jogo interativo contra a IA (Minimax) abrindo a interface gráfica no navegador.",
    )
    parser.add_argument(
        "--jogar-terminal",
        action="store_true",
        help="Inicia o jogo interativo contra a IA (Minimax) diretamente no terminal de texto.",
    )
    parser.add_argument(
        "--lado",
        type=str,
        choices=["alice", "bob"],
        default="alice",
        help="Escolha seu lado para jogar: 'alice' (inicia, peças vermelhas) ou 'bob' (2º lance, peças azuis). Padrão: 'alice'",
    )
    parser.add_argument(
        "--visualizar", "--vis",
        action="store_true",
        help="Ativa o modo de visualização gráfica para um grafo específico.",
    )
    parser.add_argument(
        "--m-visualizar", "--m",
        type=int,
        default=4,
        help="Tamanho da espinha para visualização gráfica (padrão: 4).",
    )
    parser.add_argument(
        "--assinatura",
        type=str,
        default=None,
        help="Assinatura arbitrária para visualizar (ex: '0,2,0,1,1,0' ou '2,2,2,2').",
    )
    parser.add_argument(
        "--janela",
        action="store_true",
        help="Abre a janela gráfica interativa do Matplotlib com sliders/botões.",
    )
    parser.add_argument(
        "--navegador",
        action="store_true",
        help="Abre a aplicação HTML5 interativa diretamente no seu navegador padrão.",
    )

    # Parâmetros do Loop de Expansão (m <= 100)
    parser.add_argument(
        "--m-min",
        type=int,
        default=1,
        help="Tamanho inicial da espinha m (padrão: 1).",
    )
    parser.add_argument(
        "--m-max",
        type=int,
        default=100,
        help="Tamanho limite máximo da espinha m (padrão: 100).",
    )
    parser.add_argument(
        "--modo",
        type=str,
        choices=["todos", "diametral"],
        default="todos",
        help=(
            "Modo de fixação das pernas: "
            "'todos' = exatamente 2 pernas em cada nó da espinha (L=(2,)*m, n=3*m); "
            "'diametral' = espinha diametral canônica com m nós internos (L=(0,)+(2,)*m+(0,), n=3*m+2). "
            "Padrão: 'todos'"
        ),
    )
    parser.add_argument(
        "--limite-nos",
        type=int,
        default=None,
        help="Limite opcional de nós Minimax por instância para evitar esperas excessivas em m elevado.",
    )
    parser.add_argument(
        "--detalhes",
        action="store_true",
        help="Exibe a trilha ótima passo a passo (variação principal) de cada instância iterativamente.",
    )
    parser.add_argument(
        "--gerar-vis",
        action="store_true",
        help="Durante o loop, gera automaticamente PNG e HTML de cada instância testada.",
    )
    parser.add_argument(
        "--sem-salvar",
        action="store_true",
        help="Não grava os arquivos CSV e JSON ao final da execução.",
    )

    args = parser.parse_args()

    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pasta_res = os.path.join(raiz, "resultados")
    os.makedirs(pasta_res, exist_ok=True)

    # 1. Modo Jogar no Navegador (HTML5 interativo)
    if args.jogar:
        if args.assinatura:
            L = tuple(int(x.strip()) for x in args.assinatura.split(","))
        else:
            L = assinatura_caterpillar_cheio(args.m_visualizar, modo=args.modo)
        cat = Caterpillar(L)
        caminho_jogo_html = os.path.join(pasta_res, "jogar_contra_ia.html")
        gerar_interface_jogavel_html(caminho_jogo_html, cat_inicial=cat)

        print("=" * 85)
        print("JOGAR CONTRA A IA MINIMAX (APLICAÇÃO WEB INTERATIVA)")
        print("=" * 85)
        print(f"Grafo inicial: Assinatura L = {cat.assinatura} (n={cat.n} vértices)")
        print(f"Aplicação gerada com sucesso em:\n -> {caminho_jogo_html}")
        print("Abrindo no seu navegador padrão...")
        webbrowser.open(f"file://{os.path.abspath(caminho_jogo_html)}")
        return

    # 2. Modo Jogar no Terminal (CLI)
    if args.jogar_terminal:
        if args.assinatura:
            L = tuple(int(x.strip()) for x in args.assinatura.split(","))
        else:
            L = assinatura_caterpillar_cheio(args.m_visualizar, modo=args.modo)
        cat = Caterpillar(L)
        humano_alice = (args.lado.lower() == "alice")
        jogar_contra_ia_terminal(cat, humano_eh_alice=humano_alice)
        return

    # 3. Modo Visualização Gráfica (PNG / Janela Matplotlib / HTML de partida pronta)
    if args.visualizar or args.janela or args.navegador or args.assinatura:
        executar_modo_visualizacao(args)
        return

    # Caso contrário, executa o loop iterativo com expansão da espinha até m <= 100
    desc_modo = (
        f"Todos os nós da espinha têm 2 pernas (n = 3*m, L = (2,)*m)"
        if args.modo == "todos"
        else f"Espinha diametral com m internos tendo 2 pernas (n = 3*m+2, L = (0, 2..2, 0))"
    )

    print("=" * 95)
    print("EXPERIMENTO ITERATIVO: CATERPILLARS CHEIOS COM 2 PERNAS POR NÓ DA ESPINHA")
    print("=" * 95)
    print(f"Modo: {desc_modo}")
    print(f"Faixa de espinha configurada: m = {args.m_min} até {args.m_max} (limite máximo: 100)")
    print("Convenção: Alice sempre inicia jogando (MAX = +1) | Convenção Normal (último a jogar vence)")
    print("Controle de Execução: Você decide quando parar a execução pressionando [Ctrl + C] a qualquer momento.")
    print("=" * 95)
    print(formatar_cabecalho_tabela(), flush=True)

    registros: list[dict] = []
    interrompido = False
    m_atual = args.m_min

    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pasta_vis = os.path.join(raiz, "resultados", "visualizacao")

    try:
        for m in range(args.m_min, args.m_max + 1):
            m_atual = m
            try:
                reg = analisar_caterpillar_cheio_2_pernas(
                    m=m,
                    modo=args.modo,
                    id_inst=len(registros) + 1,
                    limite_nos=args.limite_nos,
                )
                registros.append(reg)
                print(formatar_linha_tabela(reg), flush=True)

                if args.detalhes:
                    imprimir_detalhe_instancia(reg)
                    sys.stdout.flush()

                if args.gerar_vis:
                    cat = Caterpillar(assinatura_caterpillar_cheio(m, modo=args.modo))
                    gerar_painel_estatico(cat, reg["variacao_principal"], os.path.join(pasta_vis, f"caterpillar_m{m}.png"))
                    gerar_html_interativo(cat, reg["variacao_principal"], os.path.join(pasta_vis, f"caterpillar_m{m}.html"))

            except LimiteDeNosExcedido as err:
                print(f"{m:3d} | LIMITE DE NÓS EXCEDIDO ({err}). Avançando...", flush=True)

    except KeyboardInterrupt:
        interrompido = True
        print("\n\n" + "!" * 95)
        print(f"[INTERRUPÇÃO SOLICITADA] Execução parada pelo usuário (Ctrl+C) no tamanho de espinha m = {m_atual}.")
        print("!" * 95)

    finally:
        if registros:
            imprimir_sintese_final(registros, interrompido=interrompido)

            if not args.sem_salvar:
                caminho_csv, caminho_json = salvar_resultados_caterpillar_cheio(registros)
                print(f"\n[Resultados Salvos com Sucesso]")
                print(f" -> Arquivo CSV:  {caminho_csv}")
                print(f" -> Arquivo JSON: {caminho_json}")


if __name__ == "__main__":
    main()
