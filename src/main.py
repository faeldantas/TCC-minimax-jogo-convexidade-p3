"""
Aplicação Principal -- Versão CORE Essencial.

Closed Hull Game (CHG) na Convexidade P_3 em Grafos Caterpillar.
Utiliza exclusivamente o Algoritmo Minimax Puro (sem poda Alfa-Beta).
"""

from __future__ import annotations

import argparse
import os
import sys

# Garante importação do pacote p3game
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p3game import Caterpillar, JogoFechoConvexoP3, Minimax, MAX, resolver_caterpillar


def executar_partida_exemplo(assinatura: tuple[int, ...], nome: str = "") -> None:
    """Executa o solucionador Minimax Puro para um grafo caterpillar e exibe a análise essencial."""
    cat = Caterpillar(assinatura)
    jogo = JogoFechoConvexoP3(cat.adj, cat.rotulos)
    solver = Minimax(jogo, memoizacao=True)

    v = solver.resolver()
    venc = "Alice (+1)" if v == MAX else "Bob (-1)"
    pv = solver.variacao_principal()
    vencedoras_alice = [cat.rotulos[j] for j in solver.jogadas_vencedoras()] if v == MAX else []

    titulo = f"{nome} -- Assinatura L = {assinatura}" if nome else f"Assinatura L = {assinatura}"
    print("=" * 75)
    print(titulo)
    print("=" * 75)
    print(f"Vértices totais: n = {cat.n} (Espinha: k = {cat.k}, Folhas: {len(cat.folhas)})")
    print(f"Resultado do Minimax Puro: {venc}")
    if vencedoras_alice:
        print(f"Lances ótimos de Alice no 1º turno: {', '.join(vencedoras_alice)}")
    else:
        print("Lances ótimos de Alice no 1º turno: Nenhum (Bob força a vitória)")

    print(f"\nPartida sob Jogo Ótimo ({len(pv)} turnos):")
    for passo in pv:
        jog = passo["jogador"]
        lance = passo["jogada"]
        props = passo["propagados"]
        prop_str = f" [propagou por P3: {', '.join(props)}]" if props else ""
        print(f"  Turno {passo['turno']:2d} | {jog:<5}: jogou {lance}{prop_str}")

    stats = solver.stats
    print(f"\nEstatísticas da Busca Minimax:")
    print(f"  - Nós visitados: {stats.nos_visitados}")
    print(f"  - Estados em memória: {stats.entradas_tabela} (acertos: {stats.acertos_tabela})")
    print(f"  - Tempo de execução: {stats.tempo_s * 1000:.2f} ms")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="CORE: Jogo da Convexidade P_3 em Caterpillars com Minimax Puro."
    )
    parser.add_argument(
        "--assinatura",
        type=str,
        default=None,
        help="Assinatura canônica arbitrária (ex: '0,0,0' ou '0,2,0,1,1,0').",
    )
    args = parser.parse_args()

    print("\n" + "#" * 75)
    print("# VERSÃO CORE: JOGO DA CONVEXIDADE P_3 E MINIMAX PURO (SEM ALFA-BETA)")
    print("#" * 75 + "\n")

    if args.assinatura:
        L = tuple(int(x.strip()) for x in args.assinatura.split(","))
        executar_partida_exemplo(L, "Instância Customizada")
    else:
        # Casos clássicos e estudo de caso do TCC
        exemplos = [
            ((0, 0, 0), "Caminho P_3"),
            ((0, 2, 0), "Estrela K_{1,4}"),
            ((0, 1, 1, 0), "Caterpillar Uniforme C(2,1)"),
            ((0, 2, 0, 1, 1, 0), "Grafo Estudo de Caso do TCC"),
        ]
        for L, nome in exemplos:
            executar_partida_exemplo(L, nome)


if __name__ == "__main__":
    main()
