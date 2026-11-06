"""
Módulo de Experimentos Reprodutíveis (3 Baterias Científicas):

* Bateria I   -- Censo Exaustivo Canônico (Harary & Schwenk, 1973) para todos os
                 caterpillars não-isomorfos de ordem 3 <= n <= 11 (total: 286 grafos).
* Bateria II  -- Famílias Paramétricas Clássicas (Caminhos P_n, Estrelas K_{1,f},
                 Duplas-Estrelas S(a,b), Caterpillars p-Uniformes C(m,p) e
                 Vassouras Duplas / Halteres D(m,a,b)).
* Bateria III -- Estudo de Ablação das Otimizações (Minimax Puro vs. Corte vs.
                 Simetria vs. Memoização vs. Completo) incluindo o grafo de
                 referência do TCC L = (0, 2, 0, 1, 1, 0) e instâncias maiores.
"""

from __future__ import annotations

import csv
import json
import os
import time

from p3game import (
    Caterpillar,
    enumerar_caterpillars,
    numero_caterpillars,
    caminho,
    estrela,
    dupla_estrela,
    uniforme,
    vassoura_dupla,
    resolver_caterpillar,
)


def classificar_subfamilia_canonica(L: tuple[int, ...]) -> str:
    """Classifica uma assinatura canônica diametral L = (0, l_2, ..., l_{k-1}, 0)."""
    k = len(L)
    p = sum(L)
    internos = L[1:-1]

    if p == 0:
        return "Caminho Puro (Pn)"
    if k == 3:
        return "Estrela (K1,f)"
    if k == 4:
        return "Dupla-Estrela (S_a,b)"
    if len(set(internos)) == 1 and internos[0] > 0:
        return f"Caterpillar Uniforme (p={internos[0]})"
    if len(internos) >= 3 and internos[0] > 0 and internos[-1] > 0 and all(x == 0 for x in internos[1:-1]):
        return "Vassoura Dupla / Haltere"
    return "Caterpillar Assimétrico / Geral"


def analisar_instancia(L: tuple[int, ...], id_inst: int, bateria: str, familia_rotulo: str | None = None) -> dict:
    cat = Caterpillar(L)
    v, solver = resolver_caterpillar(L, corte=True, memoizacao=True, simetria=True)
    nos_busca_inicial = solver.stats.nos_visitados
    tab_busca_inicial = solver.stats.entradas_tabela
    hits_busca_inicial = solver.stats.acertos_tabela
    tempo_busca_ms = round(solver.stats.tempo_s * 1000.0, 3)

    venc_idx = solver.jogadas_vencedoras()
    venc_rotulos = [cat.rotulos[j] for j in venc_idx]
    pv = solver.variacao_principal()

    trilha_str = " -> ".join(
        f"{passo['jogador'][0]}({passo['jogada']})"
        + (f"+{passo['propagados']}" if passo["propagados"] else "")
        for passo in pv
    )

    folhas_totais = len(cat.folhas)
    internos_qtd = len(cat.internos)

    return {
        "id": id_inst,
        "bateria": bateria,
        "familia": familia_rotulo or classificar_subfamilia_canonica(L),
        "assinatura": str(list(L)),
        "assinatura_tupla": list(L),
        "n_vertices": cat.n,
        "tam_espinha_diametral_k": cat.k,
        "vertices_internos_m": internos_qtd,
        "pernas_fora_espinha_p": cat.num_pernas,
        "folhas_totais_arvore_f": folhas_totais,
        "paridade_folhas_totais": "Par" if folhas_totais % 2 == 0 else "Impar",
        "vencedor_otimo": "Alice" if v == +1 else "Bob",
        "utilidade": v,
        "qtd_jogadas_vencedoras_iniciais": len(venc_rotulos),
        "jogadas_vencedoras_iniciais": ", ".join(venc_rotulos) if venc_rotulos else "---",
        "turnos_variacao_principal": len(pv),
        "propagados_variacao_principal": cat.n - len(pv),
        "nos_visitados": nos_busca_inicial,
        "estados_canonicos_tabela": tab_busca_inicial,
        "hits_tabela": hits_busca_inicial,
        "tempo_ms": tempo_busca_ms,
        "trilha_otima": trilha_str,
    }


def executar_bateria_1_censo(n_min: int = 3, n_max: int = 11) -> tuple[list[dict], list[dict]]:
    """
    Bateria I: Censo Exaustivo de todos os caterpillars não-isomorfos de ordem 3 <= n <= 11.
    Total exato: 286 grafos (Harary & Schwenk, 1973).
    """
    registros = []
    resumo_por_n = []
    idx = 1
    for n in range(n_min, n_max + 1):
        assinaturas = enumerar_caterpillars(n)
        esperado = numero_caterpillars(n)
        assert len(assinaturas) == esperado, f"Erro em n={n}: {len(assinaturas)} != {esperado}"
        v_alice = 0
        v_bob = 0
        tempo_n = 0.0
        nos_n = 0
        for L in assinaturas:
            reg = analisar_instancia(L, idx, "Bateria I (Censo Exaustivo)")
            registros.append(reg)
            idx += 1
            if reg["vencedor_otimo"] == "Alice":
                v_alice += 1
            else:
                v_bob += 1
            tempo_n += reg["tempo_ms"]
            nos_n += reg["nos_visitados"]
        resumo_por_n.append({
            "n": n,
            "c_n_harary_schwenk": esperado,
            "vitorias_alice": v_alice,
            "perc_alice": round(100.0 * v_alice / esperado, 1),
            "vitorias_bob": v_bob,
            "perc_bob": round(100.0 * v_bob / esperado, 1),
            "media_nos": round(nos_n / esperado, 1),
            "tempo_total_ms": round(tempo_n, 2),
        })
    return registros, resumo_por_n


def executar_bateria_2_familias() -> list[dict]:
    """
    Bateria II: Famílias paramétricas clássicas até n = 14 vértices.
    """
    casos = []
    # 1. Caminhos P_n (n = 3..12)
    for n in range(3, 13):
        casos.append((caminho(n), f"Caminho P_{n}"))
    # 2. Estrelas K_{1,f} (f = 2..8)
    for f in range(2, 9):
        casos.append((estrela(f), f"Estrela K_{{1,{f}}}"))
    # 3. Duplas-Estrelas S(a,b)
    for a, b in [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3), (2, 4), (3, 4)]:
        casos.append((dupla_estrela(a, b), f"Dupla-Estrela S({a},{b})"))
    # 4. Caterpillars Uniformes C(m, p) com m vértices internos e p pernas por vértice interno
    for m in range(1, 6):
        for p in (1, 2):
            if m + 2 + m * p <= 14:
                casos.append((uniforme(m, p), f"Uniforme C(m={m}, p={p})"))
    # 5. Vassouras Duplas / Halteres D(m, a, b)
    for m in (3, 4, 5):
        for a, b in [(1, 1), (2, 2), (1, 2)]:
            casos.append((vassoura_dupla(m, a, b), f"Haltere D(m={m}, a={a}, b={b})"))
    # 6. Grafo Exemplo do TCC L = (0, 2, 0, 1, 1, 0)
    casos.append(((0, 2, 0, 1, 1, 0), "Estudo de Caso TCC"))

    registros = []
    for idx, (L, rot) in enumerate(casos, start=1):
        registros.append(analisar_instancia(L, idx, "Bateria II (Famílias Paramétricas)", familia_rotulo=rot))
    return registros


def executar_bateria_3_ablacao() -> list[dict]:
    """
    Bateria III: Estudo de ablação das otimizações do algoritmo Minimax.
    Compara:
      1. Puro (sem corte, sem memoização, sem simetria) -- até n <= 8
      2. Só Corte Alfa-Beta (corte=True, memo=False, sim=False)
      3. Corte + Simetria (corte=True, memo=False, sim=True)
      4. Corte + Memoização (corte=True, memo=True, sim=False)
      5. Completo: Corte + Memoização + Simetria (corte=True, memo=True, sim=True)
    """
    instancias_ablacao = [
        ((0, 0, 0, 0, 0), "P_5"),
        ((0, 1, 1, 0), "C(2,1)"),
        ((0, 0, 0, 0, 0, 0, 0), "P_7"),
        ((0, 2, 2, 0), "S(3,3)"),
        ((0, 1, 1, 1, 1, 0), "C(4,1)"),
        ((0, 2, 0, 1, 1, 0), "Grafo TCC"),
        ((0, 2, 1, 1, 2, 0), "Simétrico n=12"),
        ((0, 2, 2, 2, 2, 0), "C(4,2) n=14"),
    ]

    resultados = []
    for L, nome in instancias_ablacao:
        cat = Caterpillar(L)
        n = cat.n
        linha = {"nome": nome, "assinatura": str(list(L)), "n": n}

        configs = [
            ("puro", dict(corte=False, memoizacao=False, simetria=False), n <= 8),
            ("corte", dict(corte=True, memoizacao=False, simetria=False), True),
            ("corte_sim", dict(corte=True, memoizacao=False, simetria=True), True),
            ("corte_memo", dict(corte=True, memoizacao=True, simetria=False), True),
            ("completo", dict(corte=True, memoizacao=True, simetria=True), True),
        ]

        for chave, opcoes, habilitar in configs:
            if habilitar:
                v, s = resolver_caterpillar(L, **opcoes)
                linha[f"nos_{chave}"] = s.stats.nos_visitados
                linha[f"tempo_{chave}_ms"] = round(s.stats.tempo_s * 1000.0, 2)
                linha["vencedor"] = "Alice (+1)" if v == +1 else "Bob (-1)"
                if chave == "completo":
                    linha["estados_canonicos"] = s.stats.entradas_tabela
                if chave == "corte_memo":
                    linha["estados_sem_sim"] = s.stats.entradas_tabela
            else:
                linha[f"nos_{chave}"] = None
                linha[f"tempo_{chave}_ms"] = None

        resultados.append(linha)
    return resultados


def executar_tudo():
    raiz = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pasta_res = os.path.join(raiz, "resultados")
    os.makedirs(pasta_res, exist_ok=True)

    t0 = time.perf_counter()
    print("Executando Bateria I (Censo Exaustivo Harary & Schwenk, 3 <= n <= 11: 286 grafos)...")
    bat1, resumo_n = executar_bateria_1_censo(3, 11)

    print("Executando Bateria II (Famílias Paramétricas até n = 14)...")
    bat2 = executar_bateria_2_familias()

    print("Executando Bateria III (Estudo de Ablação das Otimizações)...")
    bat3 = executar_bateria_3_ablacao()
    dt = time.perf_counter() - t0
    print(f"Todas as 3 baterias concluídas em {dt:.2f} s!")

    # Salvar CSV da Bateria I (Censo Canônico)
    caminho_csv1 = os.path.join(pasta_res, "bateria1_censo_harary_schwenk.csv")
    campos = [c for c in bat1[0].keys() if c != "assinatura_tupla"]
    with open(caminho_csv1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=campos, delimiter=";")
        w.writeheader()
        for r in bat1:
            w.writerow({k: r[k] for k in campos})

    # Salvar CSV da Bateria II (Famílias Paramétricas)
    caminho_csv2 = os.path.join(pasta_res, "bateria2_familias_parametricas.csv")
    with open(caminho_csv2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=campos, delimiter=";")
        w.writeheader()
        for r in bat2:
            w.writerow({k: r[k] for k in campos})

    # Salvar JSON consolidado
    caminho_json = os.path.join(pasta_res, "experimentos_consolidados.json")
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "resumo_censo_por_n": resumo_n,
                "bateria1_censo": bat1,
                "bateria2_familias": bat2,
                "bateria3_ablacao": bat3,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Imprimir sumário no console
    print("\n--- RESUMO BATERIA I (CENSO EXAUSTIVO HARARY & SCHWENK, 3 <= n <= 11) ---")
    tot_c = sum(x["c_n_harary_schwenk"] for x in resumo_n)
    tot_a = sum(x["vitorias_alice"] for x in resumo_n)
    tot_b = sum(x["vitorias_bob"] for x in resumo_n)
    for x in resumo_n:
        print(
            f"  n={x['n']:2d} | c(n)={x['c_n_harary_schwenk']:3d} | "
            f"Alice={x['vitorias_alice']:3d} ({x['perc_alice']:5.1f}%) | "
            f"Bob={x['vitorias_bob']:3d} ({x['perc_bob']:5.1f}%) | "
            f"Média nós={x['media_nos']:6.1f} | Tempo={x['tempo_total_ms']:6.2f} ms"
        )
    print(
        f"  TOTAL (3..11): {tot_c} grafos | Alice: {tot_a} ({100*tot_a/tot_c:.1f}%) | Bob: {tot_b} ({100*tot_b/tot_c:.1f}%)"
    )

    print("\n--- RESUMO BATERIA III (ABLAÇÃO DE OTIMIZAÇÕES) ---")
    for r in bat3:
        print(
            f"  {r['nome']:<16} {r['assinatura']:<22} n={r['n']:2d} | "
            f"Puro={str(r['nos_puro']):>8} | Corte={str(r['nos_corte']):>7} | "
            f"Corte+Sim={str(r['nos_corte_sim']):>6} | Corte+Memo={str(r['nos_corte_memo']):>5} | "
            f"Completo={r['nos_completo']:>4} ({r['tempo_completo_ms']:.2f} ms)"
        )


if __name__ == "__main__":
    executar_tudo()
