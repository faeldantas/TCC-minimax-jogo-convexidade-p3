"""
Ponto de entrada principal (CLI) do pacote `p3game`.

Executa a análise interativa e iterativa para caterpillars cheios com
duas pernas em todos os nós da espinha, expandindo o tamanho da espinha
em loop (limite máximo padrão: m <= 100).

O usuário decide quando parar a execução pressionando [Ctrl + C] a qualquer momento.
Ao interromper, o script resume todos os tamanhos testados e salva automaticamente
os dados gerados em arquivos CSV e JSON.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Garante importação do pacote p3game
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p3game.minimax import LimiteDeNosExcedido
from p3game.experimentos import (
    analisar_caterpillar_cheio_2_pernas,
    salvar_resultados_caterpillar_cheio,
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

    # Formata tempo de forma amigável (ms se < 1000, ou s se >= 1000)
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


def main():
    parser = argparse.ArgumentParser(
        description="Experimento Minimax Iterativo: Caterpillar cheio com 2 pernas em todos os nós da espinha (m <= 100)."
    )
    parser.add_argument(
        "--m-min",
        type=int,
        default=1,
        help="Tamanho inicial da espinha m (padrão: 1)",
    )
    parser.add_argument(
        "--m-max",
        type=int,
        default=100,
        help="Tamanho limite máximo da espinha m (padrão: 100)",
    )
    parser.add_argument(
        "--modo",
        type=str,
        choices=["todos", "diametral"],
        default="todos",
        help=(
            "Modo de fixação das pernas na espinha: "
            "'todos' = exatamente 2 pernas em cada nó da espinha (L=(2,)*m, n=3*m); "
            "'diametral' = espinha diametral canônica com m nós internos (L=(0,)+(2,)*m+(0,), n=3*m+2). "
            "Padrão: 'todos'"
        ),
    )
    parser.add_argument(
        "--limite-nos",
        type=int,
        default=None,
        help="Limite opcional de nós Minimax por instância para evitar esperas excessivas em m elevado (ex: 2000000).",
    )
    parser.add_argument(
        "--detalhes",
        action="store_true",
        help="Exibe a trilha ótima passo a passo (variação principal) de cada instância iterativamente.",
    )
    parser.add_argument(
        "--sem-salvar",
        action="store_true",
        help="Não grava os arquivos CSV e JSON ao final da execução.",
    )

    args = parser.parse_args()

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
