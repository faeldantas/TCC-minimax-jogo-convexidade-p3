# Jogo da Convexidade $P_3$ em Caterpillars — Modo Interativo com Matplotlib

Esta branch (`jogar-matplotlib`) estende a branch base `poda-alfa-beta` adicionando uma **interface gráfica interativa nativa em Matplotlib** para jogar contra a IA (Minimax com Poda Alfa-Beta) clicando diretamente nos nós do grafo com o mouse.

Tudo roda em **Python puro**, sem necessidade de servidores, navegador web, HTML ou JavaScript.

---

## 1. Como Jogar contra a IA

### Iniciar o jogo interativo padrão:
```bash
python src/main.py --jogar
```
*Abre a janela gráfica com o caterpillar padrão $L = (0, 2, 2, 0)$. Você joga como Alice (Vermelho) e a IA joga como Bob (Azul).*

### Jogar com uma assinatura arbitrária:
```bash
python src/main.py --jogar -a "0,2,0,1,1,0"
```

### Escolher o lado (jogar como Bob, fazendo a IA iniciar como Alice):
```bash
python src/main.py --jogar --lado bob
```

---

## 2. Controles e Atalhos na Janela Gráfica

| Ação | Como acionar | Descrição |
| :--- | :--- | :--- |
| **Fazer Lance** | **Clique do mouse** | Clique com o botão esquerdo em qualquer vértice livre (cinza) para infectá-lo. |
| **Pedir Dica** | Tecla **`D`** | Destaca em **verde** o vértice recomendado pelo Minimax para vencer ou equilibrar a posição. |
| **Desfazer Lance** | Tecla **`Z`** ou **`U`** | Desfaz a última rodada de jogadas (sua jogada e a resposta da IA). |
| **Reiniciar Jogo** | Tecla **`R`** | Reseta o grafo para o estado inicial $S = 0$ para começar uma nova partida. |

---

## 3. Cores Semânticas do Tabuleiro

- ⚪ **Cinza Claro:** Vértice livre (não infectado).
- 🔴 **Vermelho:** Vértices jogados por **Alice** (MAX).
- 🔵 **Azul:** Vértices jogados por **Bob** (MIN).
- 🟡 **Dourado / Amarelo:** Vértices infectados automaticamente por **propagação em cascata $P_3$** (efeito dominó).
- 🟢 **Borda Verde:** Indicação de lance ótimo quando a tecla **`D`** (dica) é pressionada.

---

## 4. Como Executar os Outros Modos

### Expansão contínua da espinha (terminal):
```bash
python src/main.py --trilha
```

### Testes unitários:
```bash
python tests/test_core.py
python tests/test_jogar_matplotlib.py
```
