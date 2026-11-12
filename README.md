# Jogo da Convexidade $P_3$ em Caterpillars Uniformes (Versão CORE)

Esta branch (`core-essencial`) contém a **implementação minimalista e essencial** do Jogo do Fecho Convexo $P_3$ (*Closed Hull Game*) em grafos *caterpillar* uniformes, utilizando o algoritmo **Minimax Puro** (sem podas Alfa-Beta).

Toda dependência externa, módulo acessório e código de visualização foi removido. A aplicação roda exclusivamente com a biblioteca padrão do Python 3 (`deque`, `dataclasses`, `time`, `sys`).

---

## 1. O Cenário Essencial: Caterpillars Uniformes $L = (0, 2, \dots, 2, 0)$

O cenário avalia a expansão contínua da espinha interna para a família canônica com 2 pernas em cada nó interno:

$$L = (0, \underbrace{2, 2, \dots, 2}_{m \text{ nós internos}}, 0)$$

- **Comprimento da espinha:** $k = m + 2$
- **Total de vértices:** $n = 3m + 2$
- **Regras do Jogo (Convenção Normal):**
  - Alice (MAX, $+1$) e Bob (MIN, $-1$) alternam turnos escolhendo um vértice não infectado.
  - O estado evolui pelo **fecho convexo $P_3$**: qualquer vértice com $\ge 2$ vizinhos infectados propaga em cascata.
  - Vence quem realizar a **última jogada válida** (infectando todo o grafo).

---

## 2. Estrutura Enxuta em 4 Módulos (`src/p3game/`)

O código foi reduzido ao seu núcleo puro:

1. **[`src/p3game/grafos.py`](src/p3game/grafos.py):** Classe `Caterpillar` que monta as listas de adjacência e rótulos a partir da assinatura $L$.
2. **[`src/p3game/convexidade.py`](src/p3game/convexidade.py):** Função `fecho_p3` em bitmask (`int`) com propagação linear em largura $O(|V| + |E|)$.
3. **[`src/p3game/jogo.py`](src/p3game/jogo.py):** Classe `JogoFechoConvexoP3` definindo o estado inicial ($S = 0$) e os sucessores legais.
4. **[`src/p3game/minimax.py`](src/p3game/minimax.py):** Algoritmo Minimax Puro clássico de livro (max/min exaustivo sem podas alfa-beta).

---

## 3. Como Executar

### Executar a expansão da espinha:
```bash
py src/main.py
```
*O loop roda iterativamente expandindo $m = 1, 2, 3, \dots, 100$. Pressione `Ctrl + C` para parar a qualquer momento.*

### Executar vendo a trilha de cada partida:
```bash
py src/main.py --trilha
```

### Executar os testes unitários do CORE:
```bash
py tests/test_core.py
```
