# Jogo da Convexidade $P_3$ em Grafos *Caterpillar* e Algoritmo Minimax

Este repositório contém a implementação computacional desenvolvida para o **Trabalho de Conclusão de Curso (TCC)** intitulado *"Análise de Estratégias Vencedoras no Jogo da Convexidade $P_3$ em Grafos Caterpillar Utilizando o Algoritmo Minimax"* (Universidade Federal do Ceará — Campus de Russas).

## Descrição do Jogo

No **Jogo do Fecho Convexo $P_3$** (*Closed Hull Game* — CHG sob a convenção de jogo normal), dois jogadores (**Alice** e **Bob**) alternam turnos escolhendo vértices ainda não infectados de um grafo *caterpillar* $G = (V, E)$, iniciando a partir do estado vazio $S_0 = \emptyset$:

- A cada vértice $v \notin S$ escolhido pelo jogador da vez, o conjunto de vértices infectados evolui para o seu **fecho convexo $P_3$**, denotado por $S' = [S \cup \{v\}]_{P_3}$: todo vértice externo que possua pelo menos **dois vizinhos** já pertencentes ao conjunto é incorporado iterativamente em cascata até a estabilização.
- O jogo encerra quando $S = V(G)$ (todo o grafo foi infectado).
- **Condição de vitória (convenção normal):** Vence o jogador que realizar a **última jogada válida** (infectando a totalidade do grafo).

## Arquitetura Modular em 4 Camadas (`src/p3game/`)

O código foi projetado seguindo princípios de engenharia de software científica, isolando completamente o motor de busca **Minimax** da definição das regras do jogo, dos operadores algébricos da convexidade $P_3$ e da representação topológica dos grafos *caterpillar*:

1. **[`src/p3game/grafos.py`](src/p3game/grafos.py)** — *Camada Topológica e Enumeração Canônica (Harary & Schwenk, 1973)*:
   - Representa qualquer grafo *caterpillar* $G$ de ordem $n \ge 3$ por sua **assinatura diametral canônica** $L = (0, l_2, \dots, l_{k-1}, 0)$, onde $P_k = (v_1, \dots, v_k)$ é um caminho diametral (espinha máxima) e $L \le_{\text{lex}} L^R$.
   - Implementa o gerador exaustivo livre de isomorfismos `enumerar_caterpillars(n)` e valida a bijeção exata contra o teorema fechado de **Harary & Schwenk (1973)**:
     $$c(n) = 2^{n-4} + 2^{\lfloor (n-4)/2 \rfloor} \quad (n \ge 4), \quad c(3) = 1.$$
   - Fornece construtores para famílias paramétricas clássicas: caminhos $P_n$, estrelas $K_{1,f}$, duplas-estrelas $S(a,b)$, *caterpillars* $p$-uniformes $C(m,p)$ e halteres/vassouras duplas $D(m,a,b)$.

2. **[`src/p3game/convexidade.py`](src/p3game/convexidade.py)** — *Camada de Operadores de Convexidade $P_3$*:
   - Codifica subconjuntos $S \subseteq V(G)$ em máscaras de bits (`int`) para operações de conjunto em $O(1)$.
   - Implementa o operador de intervalo $I_{P_3}(S)$, o verificador de convexidade `eh_p3_convexo`, o fecho iterativo por ponto fixo `fecho_p3_iterativo` (oráculo de validação) e o algoritmo em largura (BFS incremental com contadores de vizinhos) `fecho_p3` em tempo linear $O(|V| + |E|)$.

3. **[`src/p3game/jogo.py`](src/p3game/jogo.py)** — *Camada de Abstração de Jogos Combinatórios Imparciais*:
   - Define o protocolo abstrato `JogoImparcialNormal` (`estado_inicial`, `eh_terminal`, `jogadas_legais`, `aplicar_jogada`, `chave_canonica`).
   - Implementa `JogoFechoConvexoP3` (para qualquer grafo arbitrário) e a especialização `JogoFechoConvexoP3Caterpillar`, que explora o grupo de automorfismos $S_{l_i} \wr \mathbb{Z}_2$ dos grafos *caterpillar* para podar movimentos simétricos equivalentes e gerar invariantes canônicos de órbita.

4. **[`src/p3game/minimax.py`](src/p3game/minimax.py)** — *Camada do Algoritmo Minimax Agnóstico*:
   - Opera exclusivamente sobre a interface `JogoImparcialNormal`, sem conhecer grafos ou convexidade.
   - Suporta chaves independentes (`corte`, `memoizacao`, `simetria`) para estudos de ablação rigorosos:
     - **Poda Alfa-Beta por curto-circuito binário** ($\alpha/\beta \in \{-1, +1\}$);
     - **Tabela de Transposição Imparcial** indexada apenas pela chave canônica do conjunto $S$ (independente do jogador $J$, dada a natureza imparcial de soma zero do jogo);
     - **Reconstrução de Variação Principal** (`variacao_principal`) e identificação de todas as jogadas iniciais vencedoras (`jogadas_vencedoras`).

## Baterias Experimentais (`src/p3game/experimentos.py`)

O módulo experimental executa três baterias sistemáticas e reprodutíveis:

- **Bateria I — Censo Exaustivo Canônico (Harary & Schwenk, 1973):** Avalia **100% dos 286 grafos *caterpillar* não-isomorfos** de ordem $3 \le n \le 11$ ($1 + 2 + 3 + 6 + 10 + 20 + 36 + 72 + 136 = 286$ grafos), eliminando qualquer viés de seleção amostral.
- **Bateria II — Famílias Paramétricas Clássicas (até $n = 14$):** Analisa sistematicamente caminhos $P_n$, estrelas $K_{1,f}$, duplas-estrelas $S(a,b)$, *caterpillars* uniformes $C(m,p)$, halteres $D(m,a,b)$ e o estudo de caso do TCC $L = (0, 2, 0, 1, 1, 0)$.
- **Bateria III — Estudo de Ablação das Otimizações (5 configurações):** Compara o número de nós expandidos e o tempo de execução entre: (1) Minimax Puro, (2) Só Corte Alfa-Beta, (3) Corte + Simetria, (4) Corte + Memoização e (5) Solver Completo (Corte + Memoização + Simetria).

## Como Executar

### 1. Executar a Suíte de Testes Unitários (15 testes)
Valida o teorema de enumeração de Harary & Schwenk contra o gerador não-isomorfo do `NetworkX`, a equivalência entre o BFS e a definição matemática de ponto fixo do fecho $P_3$, o Lema da Não-Propagação de Folhas e a concordância exata das 5 configurações do Minimax contra um oráculo ingênuo independente:

```bash
py -m unittest discover -s tests -v
```

### 2. Executar Validações do TCC e as 3 Baterias Experimentais
```bash
py src/main.py
```

Os arquivos gerados são salvos automaticamente em `resultados/`:
- `resultados/bateria1_censo_harary_schwenk.csv` (286 grafos canônicos);
- `resultados/bateria2_familias_parametricas.csv` (44 instâncias paramétricas);
- `resultados/experimentos_consolidados.json` (dados completos das Baterias I, II e III).
