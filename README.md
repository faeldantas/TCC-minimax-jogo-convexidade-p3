# Jogo da Convexidade $P_3$ em Caterpillars — Modo Interativo com Matplotlib

Esta branch (`jogar-matplotlib`) estende a branch base `poda-alfa-beta` adicionando uma **interface gráfica interativa nativa e minimalista em Matplotlib** para jogar contra a IA (Minimax com Poda Alfa-Beta).

Você joga como **Alice** (1º lance, Vermelho) e a máquina joga como **Bob** (Azul).
Tudo roda em **Python puro**, sem necessidade de servidores, navegador web, HTML ou JavaScript.

---

## 1. Como Jogar contra a IA

### Iniciar o jogo interativo padrão:
```bash
python src/main.py --jogar
```
*Abre a janela gráfica do Matplotlib com o caterpillar padrão $L = (0, 2, 2, 0)$. Basta clicar com o mouse nos nós livres (cinzas) para infectá-los!*

### Jogar com uma assinatura arbitrária:
```bash
python src/main.py --jogar -a "0,2,0,1,1,0"
```

---

## 2. Cores Semânticas do Grafo

- ⚪ **Cinza Claro:** Vértices livres (não infectados).
- 🔴 **Vermelho:** Vértices jogados por **Você (Alice)**.
- 🔵 **Azul:** Vértices jogados pela **IA (Bob)**.
- 🟡 **Dourado / Amarelo:** Vértices infectados automaticamente pelo **Fecho $P_3$** (efeito cascata).

---

## 3. Como Executar os Outros Modos

### Expansão contínua da espinha (terminal):
```bash
python src/main.py --trilha
```

### Testes unitários:
```bash
python tests/test_core.py
python tests/test_jogar_matplotlib.py
```
