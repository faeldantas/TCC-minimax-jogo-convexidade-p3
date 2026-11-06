"""
Ponto de Entrada (Versão CORE).
Testa caterpillars uniformes no formato canônico L = (0, 2..2, 0),
expandindo iterativamente a espinha com suas 2 pernas por nó interno.

Executa o algoritmo Minimax Puro (sem poda Alfa-Beta) e exibe de forma
concisa: tamanho da espinha, quem vence, número de jogadas e poucas informações.
Interrompa a qualquer momento com [Ctrl + C].
"""

from __future__ import annotations

import argparse
import os
import sys

# Garante importação do pacote p3game
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p3game import Caterpillar, JogoFechoConvexoP3, Minimax, MAX


def assinatura_uniforme_2pernas(m: int) -> tuple[int, ...]:
    """Retorna a assinatura canônica uniforme (0, 2, ..., 2, 0) com m nós internos."""
    return (0,) + (2,) * m + (0,)


def formatar_cabecalho() -> str:
    cab = (
        f"{'m':>2} | {'k':>2} | {'n':>3} | "
        f"{'Vencedor':^8} | {'Jogadas':^7} | {'Alice':^5} | {'Bob':^5} | "
        f"{'Propagados':^10} | {'Tempo':^11} | {'Nós Minimax':^11}"
    )
    div = "-" * len(cab)
    return f"{div}\n{cab}\n{div}"


def formatar_linha(m: int, k: int, n: int, v: int, turnos: int, ja: int, jb: int, prop: int, tempo_s: float, nos: int) -> str:
    venc_str = "Alice" if v == MAX else "Bob"
    if tempo_s < 1.0:
        t_str = f"{tempo_s * 1000:7.2f} ms"
    else:
        t_str = f"{tempo_s:7.2f} s "

    return (
        f"{m:2d} | {k:2d} | {n:3d} | "
        f"{venc_str:^8} | {turnos:7d} | {ja:5d} | {jb:5d} | "
        f"{prop:10d} | {t_str:^11} | {nos:11d}"
    )


def resolver_passo(m: int) -> dict:
    L = assinatura_uniforme_2pernas(m)
    cat = Caterpillar(L)
    jogo = JogoFechoConvexoP3(cat.adj, cat.rotulos)
    solver = Minimax(jogo, memoizacao=True)

    v = solver.resolver()
    pv = solver.variacao_principal()
    ja = sum(1 for p in pv if p["jogador"] == "Alice")
    jb = sum(1 for p in pv if p["jogador"] == "Bob")
    prop = cat.n - len(pv)

    return {
        "m": m,
        "k": cat.k,
        "n": cat.n,
        "v": v,
        "turnos": len(pv),
        "ja": ja,
        "jb": jb,
        "prop": prop,
        "tempo_s": solver.stats.tempo_s,
        "nos": solver.stats.nos_visitados,
        "trilha": pv,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Expansão da espinha para caterpillars uniformes (0, 2..2, 0) com Minimax Puro."
    )
    parser.add_argument(
        "--m-min",
        type=int,
        default=1,
        help="Tamanho inicial da espinha interna m (padrão: 1).",
    )
    parser.add_argument(
        "--m-max",
        type=int,
        default=100,
        help="Limite máximo de espinha interna m (padrão: 100).",
    )
    parser.add_argument(
        "--trilha",
        action="store_true",
        help="Exibe os lances da partida ótima sob cada tamanho testado.",
    )
    args = parser.parse_args()

    print("=" * 87)
    print("CATERPILLARS UNIFORMES L = (0, 2..2, 0) -- MINIMAX PURO (CORE)")
    print("=" * 87)
    print("Formato: 2 pernas em cada nó interno da espinha | Alice sempre inicia (MAX)")
    print("Pressione [Ctrl + C] para interromper a execução a qualquer momento.")
    print("=" * 87)
    print(formatar_cabecalho(), flush=True)

    m = args.m_min
    try:
        for m in range(args.m_min, args.m_max + 1):
            r = resolver_passo(m)
            linha = formatar_linha(
                r["m"], r["k"], r["n"], r["v"],
                r["turnos"], r["ja"], r["jb"],
                r["prop"], r["tempo_s"], r["nos"]
            )
            print(linha, flush=True)

            if args.trilha:
                seq = " -> ".join(
                    f"{p['jogador'][0]}({p['jogada']})" + (f"+{p['propagados']}" if p["propagados"] else "")
                    for p in r["trilha"]
                )
                print(f"   Trilha: {seq}", flush=True)

    except KeyboardInterrupt:
        print("\n" + "-" * 87)
        print(f"[PARADA] Execução interrompida pelo usuário no tamanho m = {m}.")
        print("-" * 87)


if __name__ == "__main__":
    main()
