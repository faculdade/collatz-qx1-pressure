"""
E-109 — H-130, passo 2: a esterilidade extra (2 nao raiz primitiva mod q)
reescala W_u por tipo do mesmo jeito que a familia de escala ja
confirmada para q=5 (E-103 Estagio 4), entre os tipos NAO-esterceis?

Contexto: quando ord_q(2) < phi(q) (2 nao raiz primitiva), so os
residuos no subgrupo <2> sao nao-esterceis (tem filhos); os demais
(e u=0, e qualquer u nao coprimo a q) sao esterceis (achado da 3a
rodada de auditoria do paper, corrigido na Sec.2). A familia de escala
ja confirmada para q=5 (2 raiz primitiva, todos os residuos nao-nulos
nao-esterceis) prediz W_i =d 2^(-a0(i)*theta) * W*, theta=alpha_-(q).
Este experimento testa se a MESMA relacao vale entre os tipos
nao-esterceis, apesar da esterilidade extra alhures na arvore, para
DOIS casos com estruturas diferentes:

  - q=7 (primo, ord_7(2)=3<phi(7)=6, 3 de 6 residuos nao-nulos
    nao-esterceis, <2>={1,2,4})
  - q=15 (composto, ord_15(2)=4<phi(15)=8, 4 de 8 residuos coprimos
    nao-esterceis, <2>={1,2,4,8}) -- testa se o achado e' robusto a
    q composto, nao so primo com fatoracao trivial.

q=31 (o caso mais extremo da tabela de H-130, so 5/30 residuos
nao-nulos nao-esterceis) foi considerado mas descartado: theta_31 ~
0.0552 e tao pequeno que o crescimento populacional H^0.0552 exige H
astronomico para dar amostra razoavel (H^0.0552=100 exigiria H~10^36).
"""

import math
import random
import time

from tree_lib_sterility import count_tree, CYCLES  # mesma pasta

CONFIGS = {
    7: {
        "theta": 0.37350103443077054,
        "a0": {1: 3, 2: 2, 4: 1},
        "n_per_type": 5000,
        "v_range": (10_001, 2_000_001),
        "headroom": 10 ** 7,
    },
    15: {
        "theta": 0.131006,
        "a0": {1: 4, 2: 3, 4: 2, 8: 1},
        "n_per_type": 3000,
        "v_range": (20_001, 4_000_001),
        "headroom": 10 ** 8,
    },
}


def sample_roots(q, residue, n, rng, exclude, v_range):
    roots = []
    tried = 0
    while len(roots) < n and tried < n * 80:
        tried += 1
        v = rng.randrange(v_range[0], v_range[1], 2)
        if math.gcd(v, q) != 1:
            continue
        if v % q != residue:
            continue
        if v in exclude:
            continue
        roots.append(v)
    return roots


def geomean(xs):
    logs = [math.log(x) for x in xs if x > 0]
    return math.exp(sum(logs) / len(logs))


def median(xs):
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def run_for_q(q, cfg, seed):
    theta = cfg["theta"]
    a0 = cfg["a0"]
    n_per_type = cfg["n_per_type"]
    v_range = cfg["v_range"]
    headroom = cfg["headroom"]

    rng = random.Random(seed)
    exclude = CYCLES.get(q, set())
    results = {}
    t0 = time.time()
    print(f"\n{'='*70}\nq={q}  (theta={theta:.6f}, tipos nao-esterceis={sorted(a0)})\n{'='*70}")
    for residue in sorted(a0):
        roots = sample_roots(q, residue, n_per_type, rng, exclude, v_range)
        Ws = []
        for v in roots:
            n_max = v * headroom
            total, counts = count_tree(q, v, n_max, checkpoints=[n_max])
            w = counts[0] / (float(headroom) ** theta)
            Ws.append(w)
        gm_w = geomean(Ws)
        med_w = median(Ws)
        results[residue] = {"geomean": gm_w, "median": med_w, "Ws": Ws}
        print(f"  tipo u={residue} (a0={a0[residue]}): {len(Ws)} raizes "
              f"({time.time()-t0:.1f}s acumulado) -- "
              f"media_geom={gm_w:.4f} mediana={med_w:.4f} "
              f"min={min(Ws):.4f} max={max(Ws):.4f}")

    print()
    print(f"  === Razoes observadas (media geometrica) vs. previstas ===")
    types = sorted(a0)
    for idx_i in range(len(types)):
        for idx_j in range(idx_i + 1, len(types)):
            i, j = types[idx_i], types[idx_j]
            obs = results[i]["geomean"] / results[j]["geomean"]
            pred = 2.0 ** (-(a0[i] - a0[j]) * theta)
            print(f"    W_{i}/W_{j}: observado={obs:.4f}  previsto={pred:.4f}  "
                  f"razao obs/previsto={obs/pred:.4f}")


def main():
    for q, cfg in CONFIGS.items():
        run_for_q(q, cfg, seed=20260720)


if __name__ == "__main__":
    main()
