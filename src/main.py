"""
Ponto de Entrada Principal (CLI) - Versão CORE com Poda Alfa-Beta.
Executa a expansão contínua da espinha interna para caterpillars regulares C(m, p)
no formato canônico L = (0, p..p, 0), resolvendo cada instância com Minimax e Poda Alfa-Beta.
Interrompa a qualquer momento com [Ctrl + C] para registrar o limite computacional alcançado.
"""

from __future__ import annotations

import argparse  # Módulo para análise de argumentos passados via linha de comando
import os        # Módulo para interação com caminhos e sistema operacional
import sys       # Módulo para manipulação de variáveis do interpretador Python

# Adiciona o diretório atual do script ao sys.path para garantir que o pacote 'p3game' seja encontrado
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importa as classes e constantes fundamentais do pacote p3game
from p3game import Caterpillar, JogoFechoConvexoP3, Minimax, MAX, jogar_caterpillar_matplotlib


def assinatura_uniforme(m: int, p: int = 2) -> tuple[int, ...]:
    """
    Gera a assinatura diametral canônica L = (0, p, ..., p, 0).
    - As extremidades v1 e vk recebem 0 pernas extras (são folhas de grau 1).
    - Os m nós internos da espinha recebem exatamente p pernas cada.
    """
    return (0,) + (p,) * m + (0,)


def formatar_cabecalho() -> str:
    """Gera o cabeçalho tabular alinhado para a exibição dos resultados no terminal."""
    cab = (
        f"{'m':>2} | {'k':>2} | {'n':>3} | "
        f"{'Vencedor':^8} | {'Jogadas':^7} | {'Alice':^5} | {'Bob':^5} | "
        f"{'Propagados':^10} | {'Tempo':^11} | {'Nós Minimax':^11}"
    )
    div = "-" * len(cab)  # Linha divisória com o comprimento exato do cabeçalho
    return f"{div}\n{cab}\n{div}"


def formatar_tempo(tempo_s: float) -> str:
    """Converte o tempo em segundos para unidade legível: milissegundos (ms), segundos (s) ou minutos (min)."""
    if tempo_s < 1.0:
        return f"{tempo_s * 1000:7.2f} ms"  # Menor que 1 segundo: exibe em ms
    if tempo_s < 60.0:
        return f"{tempo_s:7.2f} s "         # Menor que 1 minuto: exibe em segundos
    return f"{tempo_s / 60.0:7.2f} min"     # Maior ou igual a 1 minuto: exibe em minutos


def formatar_linha(m: int, k: int, n: int, v: int, turnos: int, ja: int, jb: int, prop: int, tempo_s: float, nos: int) -> str:
    """Formata uma linha de resultado mantendo o alinhamento das colunas numéricas."""
    venc_str = "Alice" if v == MAX else "Bob"  # Traduz o valor numérico (+1/-1) para o nome do jogador
    t_str = formatar_tempo(tempo_s)             # Formata a string de tempo com unidade adequada

    return (
        f"{m:2d} | {k:2d} | {n:3d} | "
        f"{venc_str:^8} | {turnos:7d} | {ja:5d} | {jb:5d} | "
        f"{prop:10d} | {t_str:^11} | {nos:11d}"
    )


def resolver_passo(m: int, p: int = 2) -> dict:
    """
    Constrói e resolve uma instância do caterpillar de espinha interna m e p pernas por nó.
    Retorna um dicionário com todas as métricas analíticas calculadas.
    """
    L = assinatura_uniforme(m, p)                   # Gera a assinatura canônica (0, p..p, 0)
    cat = Caterpillar(L)                            # Constrói o grafo caterpillar e sua lista de adjacência
    jogo = JogoFechoConvexoP3(cat.adj, cat.rotulos) # Inicializa as regras do jogo e mapeamento de vértices
    solver = Minimax(jogo, memoizacao=True)         # Instancia o algoritmo Minimax com Poda Alfa-Beta e cache

    v = solver.resolver()                            # Executa a busca Minimax e obtém o vencedor (+1 ou -1)
    pv = solver.variacao_principal()                 # Reconstrói a sequência ótima de lances (partida ideal)
    ja = sum(1 for lance in pv if lance["jogador"] == "Alice")  # Conta jogadas realizadas por Alice
    jb = sum(1 for lance in pv if lance["jogador"] == "Bob")    # Conta jogadas realizadas por Bob
    prop = cat.n - len(pv)                           # Vértices infectados por propagação P3 (sem gastar turno)

    return {
        "m": m,
        "k": cat.k,
        "n": cat.n,
        "v": v,
        "turnos": len(pv),
        "ja": ja,
        "jb": jb,
        "prop": prop,
        "tempo_s": solver.tempo_s,
        "nos": solver.nos_visitados,
        "trilha": pv,
    }


def main():
    """Função principal que analisa argumentos de CLI e gerencia o loop de execução contínua."""
    parser = argparse.ArgumentParser(
        description="Expansão contínua da espinha para caterpillars regulares C(m, p) para mapear o limite da solução ótima."
    )
    # Define os parâmetros de linha de comando
    parser.add_argument(
        "--jogar",
        action="store_true",
        help="Inicia a partida gráfica interativa no Matplotlib contra a IA Minimax.",
    )
    parser.add_argument(
        "--lado",
        type=str,
        choices=["alice", "bob"],
        default="alice",
        help="Escolha seu lado para jogar: 'alice' (inicia, Vermelho) ou 'bob' (2º lance, Azul). Padrão: 'alice'",
    )
    parser.add_argument(
        "--assinatura", "-a",
        type=str,
        default=None,
        help="Executa uma assinatura arbitrária específica (ex: '0,2,2,0,2,2,0' ou '0,2,0,2,0,2,0').",
    )
    parser.add_argument(
        "--pernas", "-p",
        type=int,
        default=2,
        help="Quantidade de pernas em cada nó interno da espinha (padrão: 2).",
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
        help="Limite máximo de espinha interna m (padrão: 100, execute até onde a máquina suportar).",
    )
    parser.add_argument(
        "--trilha",
        action="store_true",
        help="Exibe os lances da partida ótima sob cada tamanho testado.",
    )
    args = parser.parse_args()  # Processa os argumentos passados pelo usuário

    # Se a flag --jogar foi passada, abre a interface gráfica interativa do Matplotlib
    if args.jogar:
        if args.assinatura:
            L = tuple(int(x.strip()) for x in args.assinatura.split(","))
        else:
            # Por padrão utiliza m=2 (L = (0, 2, 2, 0)) ou o m_min informado
            m_usado = args.m_min if args.m_min > 1 else 2
            L = assinatura_uniforme(m_usado, args.pernas)

        humano_alice = (args.lado.lower() == "alice")
        print("=" * 87)
        print("JOGO DA CONVEXIDADE P_3 -- INTERFACE INTERATIVA MATPLOTLIB")
        print("=" * 87)
        print(f"Grafo: Assinatura L = {L} | Total de vértices: {len(L) + sum(L)}")
        print(f"Você joga como: {'Alice (1º a jogar, Vermelho)' if humano_alice else 'Bob (2º a jogar, Azul)'}")
        print("Atalhos na janela:")
        print("  - Clique com o botão esquerdo em qualquer nó livre para jogar")
        print("  - Pressione [R] para reiniciar a partida")
        print("  - Pressione [Z] para desfazer o último lance")
        print("  - Pressione [D] para receber uma dica da IA")
        print("=" * 87)
        print("Abrindo janela gráfica do Matplotlib...")
        jogar_caterpillar_matplotlib(L, humano_eh_alice=humano_alice)
        return

    # Se o usuário passou uma assinatura específica (ex: -a "0,2,2,0,2,2,0")
    if args.assinatura:
        L = tuple(int(x.strip()) for x in args.assinatura.split(","))
        cat = Caterpillar(L)
        jogo = JogoFechoConvexoP3(cat.adj, cat.rotulos)
        solver = Minimax(jogo, memoizacao=True)
        v = solver.resolver()
        pv = solver.variacao_principal()
        ja = sum(1 for lance in pv if lance["jogador"] == "Alice")
        jb = sum(1 for lance in pv if lance["jogador"] == "Bob")
        prop = cat.n - len(pv)
        m_internos = len(L) - 2

        print("=" * 87)
        print(f"CATERPILLAR ESPECÍFICO -- ASSINATURA L = {L}")
        print("=" * 87)
        print(f"Ordem: n = {cat.n} vértices | Espinha: k = {cat.k} nós | Alice inicia (MAX)")
        print("=" * 87)
        print(formatar_cabecalho(), flush=True)
        linha = formatar_linha(m_internos, cat.k, cat.n, v, len(pv), ja, jb, prop, solver.tempo_s, solver.nos_visitados)
        print(linha, flush=True)
        if args.trilha:
            seq = " -> ".join(
                f"{p['jogador'][0]}({p['jogada']})" + (f"+{p['propagados']}" if p["propagados"] else "")
                for p in pv
            )
            print(f"   Trilha: {seq}", flush=True)
        return

    # Imprime informações iniciais do experimento
    print("=" * 87)
    print(f"CATERPILLARS REGULARES C(m, {args.pernas}) -- BUSCA DE LIMITE DA SOLUÇÃO ÓTIMA")
    print("=" * 87)
    print(f"Formato: {args.pernas} perna(s) em cada nó interno da espinha | Alice sempre inicia (MAX)")
    print("A busca expandirá a espinha continuamente até encontrar o limite computacional.")
    print("Pressione [Ctrl + C] para interromper a execução a qualquer momento.")
    print("=" * 87)
    print(formatar_cabecalho(), flush=True)  # Exibe o cabeçalho da tabela de resultados

    m = args.m_min  # Começa no valor mínimo solicitado
    try:
        # Itera incrementalmente pelo comprimento da espinha m
        for m in range(args.m_min, args.m_max + 1):
            r = resolver_passo(m, args.pernas)  # Resolve a instância com m nós internos e p pernas
            linha = formatar_linha(
                r["m"], r["k"], r["n"], r["v"],
                r["turnos"], r["ja"], r["jb"],
                r["prop"], r["tempo_s"], r["nos"]
            )
            print(linha, flush=True)  # Imprime a linha com os dados da instância

            # Se a flag --trilha foi passada, imprime a sequência jogada a jogada
            if args.trilha:
                seq = " -> ".join(
                    f"{p['jogador'][0]}({p['jogada']})" + (f"+{p['propagados']}" if p["propagados"] else "")
                    for p in r["trilha"]
                )
                print(f"   Trilha: {seq}", flush=True)

    except KeyboardInterrupt:
        # Tratamento gracioso para quando o usuário pressionar Ctrl + C para encerrar o teste
        print("\n" + "-" * 87)
        print(f"[LIMITE ATINGIDO / INTERROMPIDO] Parada do teste no tamanho de espinha m = {m}.")
        print("Anote os dados da tabela acima para registrar a fronteira alcançada no TCC.")
        print("-" * 87)


if __name__ == "__main__":
    main()  # Dispara a execução principal se o script for chamado diretamente
