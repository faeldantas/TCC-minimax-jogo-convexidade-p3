# Relatório Analítico de Experimentos — Jogo da Convexidade $P_3$ em Grafos *Caterpillar*

## 1. Visão Geral da Bateria Experimental
- **Total de grafos *caterpillar* não-isomorfos analisados:** 367
- **Vitórias de Alice (1º Jogador / MAX = $+1$):** 225 (61.3%)
- **Vitórias de Bob (2º Jogador / MIN = $-1$):** 142 (38.7%)

## 2. Distribuição de Vencedores por Família de *Caterpillar*
| Família Estrutural | Total de Instâncias | Vitórias de Alice (+1) | Vitórias de Bob (-1) |
| :--- | :---: | :---: | :---: |
| Caminho Puro (Pk) | 10 | 8 (80.0%) | 2 (20.0%) |
| Caterpillar Geral | 245 | 148 (60.4%) | 97 (39.6%) |
| Caterpillar Uniforme (p=1) | 3 | 2 (66.7%) | 1 (33.3%) |
| Caterpillar Uniforme (p=2) | 2 | 1 (50.0%) | 1 (50.0%) |
| Caterpillar Uniforme (p=3) | 1 | 1 (100.0%) | 0 (0.0%) |
| Dupla-Estrela | 6 | 4 (66.7%) | 2 (33.3%) |
| Estrela (K1,f) | 3 | 1 (33.3%) | 2 (66.7%) |
| Haltere (Pernas nas Extremidades) | 24 | 19 (79.2%) | 5 (20.8%) |
| Pernas Internas (Centro) | 73 | 41 (56.2%) | 32 (43.8%) |

## 3. Influência da Paridade do Número de Folhas ($f = \sum l_i$)
| Paridade de Folhas ($f$) | Vitórias de Alice (+1) | Vitórias de Bob (-1) |
| :--- | :---: | :---: |
| **Par** ($f \equiv 0 \pmod 2$) | 123 | 79 |
| **Ímpar** ($f \equiv 1 \pmod 2$) | 102 | 63 |

## 4. Descobertas Estratégicas e Padrões Vencedores Identificados

### Padrão 1: Invariante de Infecciosidade das Folhas (Pernas e Extremidades de Grau 1)
Em qualquer grafo *caterpillar* $G$, todo vértice folha (seja uma perna ligada à espinha ou uma extremidade da espinha sem pernas) possui grau $d(v) = 1$. Como a regra de propagação da Convexidade $P_3$ exige que um vértice externo possua **pelo menos dois vizinhos** no conjunto convexo atual para ser infectado automaticamente, **nenhuma folha pode ser infectada por propagação**. Consequentemente:
- Se um grafo possui $L_{total}$ vértices de grau 1 (folhas totais da árvore), cada um desses $L_{total}$ vértices terá obrigatoriamente que ser escolhido de forma direta por um dos jogadores em algum turno da partida.
- Assim, o número mínimo de turnos de qualquer partida em $G$ é limitado inferiormente pelo número total de folhas da árvore ($T \ge L_{total}$).

### Padrão 2: O Efeito de Inversão de Turno (Controle de Tempo pelas Pernas)
Quando toda a espinha $P_k$ já se encontra infectada, restam apenas as pernas não selecionadas. A partir desse momento, nenhuma propagação adicional ocorre, e o jogo reduz-se a uma contagem pura de paridade das folhas restantes:
- Adicionando-se $\Delta f$ pernas a um mesmo vértice da espinha, quando $\Delta f$ é **ímpar**, o jogador capaz de estabilizar a espinha pode usar uma perna como *jogada de espera (tempo move)* para inverter a paridade de quem joga na espinha.
- Observa-se claramente nos experimentos que para uma mesma estrutura de espinha $k$ fixa, incrementar uma única perna em um vértice frequentemente **alterna o vencedor entre Alice e Bob** (por exemplo: `[1]` $\to$ Bob, `[2]` $\to$ Alice; `[1, 1]` $\to$ Alice, `[1, 2]` $\to$ Bob, `[2, 2]` $\to$ Alice).

### Padrão 3: Comportamento em Caminhos Puros ($P_k$) vs. Adição de Pernas
Nos caminhos puros ($f = 0$), os resultados do Minimax para $P_2$ a $P_{11}$ mostram:
- **$P_{2}$** (`[0, 0]`): Vencedor = **Bob** | Turnos ótimos = 2 | Jogadas vencedoras iniciais: `Nenhuma`
- **$P_{3}$** (`[0, 0, 0]`): Vencedor = **Alice** | Turnos ótimos = 3 | Jogadas vencedoras iniciais: `v2`
- **$P_{4}$** (`[0, 0, 0, 0]`): Vencedor = **Alice** | Turnos ótimos = 3 | Jogadas vencedoras iniciais: `v1, v4`
- **$P_{5}$** (`[0, 0, 0, 0, 0]`): Vencedor = **Alice** | Turnos ótimos = 3 | Jogadas vencedoras iniciais: `v3`
- **$P_{6}$** (`[0, 0, 0, 0, 0, 0]`): Vencedor = **Alice** | Turnos ótimos = 5 | Jogadas vencedoras iniciais: `v2, v5`
- **$P_{7}$** (`[0, 0, 0, 0, 0, 0, 0]`): Vencedor = **Alice** | Turnos ótimos = 5 | Jogadas vencedoras iniciais: `v4`
- **$P_{8}$** (`[0, 0, 0, 0, 0, 0, 0, 0]`): Vencedor = **Alice** | Turnos ótimos = 5 | Jogadas vencedoras iniciais: `v3, v6`
- **$P_{9}$** (`[0, 0, 0, 0, 0, 0, 0, 0, 0]`): Vencedor = **Alice** | Turnos ótimos = 7 | Jogadas vencedoras iniciais: `v2, v5, v8`
- **$P_{10}$** (`[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`): Vencedor = **Bob** | Turnos ótimos = 8 | Jogadas vencedoras iniciais: `Nenhuma`
- **$P_{11}$** (`[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`): Vencedor = **Alice** | Turnos ótimos = 7 | Jogadas vencedoras iniciais: `v3, v6, v9`

### Padrão 4: Estudo de Caso da Instância do TCC (`L = [0, 2, 0, 1, 1, 0]`, $|V|=10$)
- **Vencedor com Jogo Ótimo:** Bob (Utilidade = -1)
- **Estados $P_3$-convexos distintos alcançáveis:** 146
- **Nós explorados pelo Minimax otimizado:** 546 (em 3.224 ms)
- **Duração possível de uma partida:** mínima de 6 turnos (ex: como na Figura 4 do TCC, onde há colapso de 3 vértices da espinha de uma só vez) e máxima de 10 turnos.
- **Sequência Ótima (Principal Variation):** `A(v1) -> B(v3)+['v2'] -> A(v4) -> B(v6)+['v5'] -> A(v7) -> B(v8) -> A(v9) -> B(v10)`
- **Explicação Estratégica:** Na partida ilustrativa da Figura 4 do TCC (Seção 5.2), Bob faz a jogada $B(v_6)$ no 4º turno, o que provoca a infecção simultânea de **3 vértices** ($v_5, v_4, v_3$), fazendo a partida durar 7 turnos (ímpar) e dando a vitória para Alice. Já o algoritmo **Minimax** revela que Bob vence a partida se controlar a propagação da espinha em pares (infectando apenas 1 vértice por fechamento em $v_2$ e depois em $v_5$), forçando a partida a durar exatamente **8 turnos (par)** e garantindo que a última folha seja sua!