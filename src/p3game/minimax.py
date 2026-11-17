"""
Camada 4 -- Algoritmo Minimax com Poda Alfa-Beta (CORE).
Implementa o solucionador exaustivo com poda alfa-beta por curto-circuito binário e memoização.
"""

from __future__ import annotations
import sys   # Módulo do sistema para alterar o limite de profundidade de recursão
import time  # Módulo para cronometragem de alta resolução (perf_counter)

# Constantes dos jogadores:
# MAX (+1) = Alice (jogadora inicial que tenta maximizar para +1)
# MIN (-1) = Bob   (segundo jogador que tenta minimizar para -1)
MAX, MIN = +1, -1


class Minimax:
    """
    Minimax com Poda Alfa-Beta para jogos combinatórios imparciais sob convenção normal.
    - Alice = MAX (+1)
    - Bob = MIN (-1)
    - Com corte alfa-beta: interrompe a busca no nó assim que alfa >= beta.
    - Memoização: tabela de transposição para não recalcular estados isomórficos já visitados.
    """

    def __init__(self, jogo, memoizacao: bool = True):
        """Inicializa o resolvedor Minimax associado a uma instância do jogo."""
        self.jogo = jogo                              # Referência à camada de regras do jogo
        self.memoizacao = memoizacao                  # Booleano que ativa ou desativa o cache de transposição
        self.tabela: dict[int, int] = {}              # Dicionário de memoização: mapeia estado S para seu valor normalizado
        self.nos_visitados = 0                        # Contador cumulativo de nós visitados na árvore de busca
        self.tempo_s = 0.0                            # Tempo total consumido pela resolução em segundos
        # Aumenta o limite de recursão do Python para suportar árvores de busca profundas (m elevado)
        sys.setrecursionlimit(max(10_000, sys.getrecursionlimit()))

    def valor(self, S: int, jogador: int, alfa: int = MIN, beta: int = MAX) -> int:
        """
        Calcula recursivamente o valor Minimax do estado S para o jogador da vez com poda Alfa-Beta.
        Retorna:
          +1: Alice possui estratégia vencedora forçada a partir de S.
          -1: Bob possui estratégia vencedora forçada a partir de S.
        """
        self.nos_visitados += 1  # Incrementa o número de chamadas da função de avaliação

        # Consulta a Tabela de Transposição: se o estado já foi resolvido, retorna o valor ajustado
        if self.memoizacao and S in self.tabela:
            # Multiplica por 'jogador' para converter a perspectiva relativa (quem joga) para absoluta (+1/-1)
            return self.tabela[S] * jogador

        # Obtém todos os movimentos válidos a partir do estado atual S
        sucessores = self.jogo.jogadas(S)

        if not sucessores:
            # Caso Base Terminal: Não há mais lances legais (todo o grafo foi infectado).
            # Pela Convenção Normal, quem não tem jogada perde; logo, o jogador anterior venceu:
            v = -jogador  # Se era vez de Alice (+1), Alice perde (-1); se era Bob (-1), Bob perde (+1)
        elif jogador == MAX:
            # Turno de Alice (MAX): quer o maior valor possível (+1)
            v = MIN  # Inicializa com o pior valor para MAX (-1)
            for _, filho in sucessores:  # Avalia cada estado sucessor
                v = max(v, self.valor(filho, MIN, alfa, beta))  # Chama recursivamente para a vez de Bob (MIN)
                alfa = max(alfa, v)                             # Atualiza a melhor garantia de Alice (alfa)
                if alfa >= beta:
                    break  # Poda Beta: MAX já garantiu vitória (+1), ramos irmãos são cortados
        else:
            # Turno de Bob (MIN): quer o menor valor possível (-1)
            v = MAX  # Inicializa com o pior valor para MIN (+1)
            for _, filho in sucessores:  # Avalia cada estado sucessor
                v = min(v, self.valor(filho, MAX, alfa, beta))  # Chama recursivamente para a vez de Alice (MAX)
                beta = min(beta, v)                             # Atualiza a melhor garantia de Bob (beta)
                if beta <= alfa:
                    break  # Poda Alfa: MIN já garantiu vitória (-1), ramos irmãos são cortados

        # Armazena o resultado na tabela de memoização (normalizado do ponto de vista do jogador da vez)
        if self.memoizacao:
            self.tabela[S] = v * jogador

        return v  # Retorna o valor minimax do estado

    def resolver(self) -> int:
        """
        Dispara a resolução do jogo a partir do estado inicial (S = 0), com Alice (MAX) iniciando.
        Cronometra o tempo de execução e armazena em self.tempo_s.
        """
        t0 = time.perf_counter()  # Marca o timestamp inicial com precisão de nanossegundos
        try:
            # Inicia a busca a partir do estado inicial vazio S = 0 com a janela clássica [-1, +1]
            return self.valor(self.jogo.estado_inicial(), MAX, MIN, MAX)
        finally:
            # Garante que o tempo decorrido seja gravado mesmo se ocorrer interrupção (Ctrl + C)
            self.tempo_s = time.perf_counter() - t0

    def variacao_principal(self) -> list[dict]: 
        """
        Reconstrói a sequência ótima de lances (partida perfeita) jogada por ambos os jogadores.
        Retorna uma lista de dicionários com turno, jogador, vértice escolhido e vértices propagados.
        """
        from .convexidade import fecho_p3  # Importação local para obter a lista exata de vértices propagados
        S = self.jogo.estado_inicial()     # Começa no estado inicial S = 0
        jogador = MAX                      # Alice é sempre a primeira a jogar
        partida = []                       # Lista que registrará o histórico dos turnos

        while True:
            sucessores = self.jogo.jogadas(S)  # Gera os lances possíveis a partir de S
            if not sucessores:
                return partida  # Fim da partida: nenhum lance restante

            # Seleciona o lance ótimo: escolhe um sucessor que mantém a condição de vitória do jogador atual
            escolhida = next(
                ((v, prox) for v, prox in sucessores if self.valor(prox, -jogador) == jogador),
                sucessores[0],  # Se estiver em posição perdedora forçada, escolhe o primeiro lance disponível
            )
            v, prox_S = escolhida  # Desempacota o vértice jogado v e o novo estado prox_S
            _, props = fecho_p3(self.jogo.adj, S | (1 << v))  # Identifica os vértices infectados em cascata

            # Registra as informações do turno na lista da partida
            partida.append({
                "turno": len(partida) + 1,
                "jogador": "Alice" if jogador == MAX else "Bob",
                "jogada": self.jogo.rotulo(v),
                "propagados": [self.jogo.rotulo(p) for p in props],
            })
            S = prox_S            # Atualiza o estado atual do grafo
            jogador = -jogador    # Alterna o jogador da vez (MAX -> MIN -> MAX ...)
