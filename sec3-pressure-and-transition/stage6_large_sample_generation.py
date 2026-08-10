"""
Estagio 6 -- amostra muito maior de W_v (q=5): 100.000 raizes (20x a
Rodada 1/2), mesmos 4 niveis de headroom (1e5..1e8), paralelizado (a
arvore reversa de cada raiz e independente das demais -- embaracosamente
paralelo). Runtime esperado: ~70-75 min com 12 processos.

V_RANGE expandido 10x (1001..2.000.001) em relacao a Rodada 1/2
(1001..200.001): o range original so tem ~79.600 valores validos
(impares, nao mult. de 5, fora dos ciclos), insuficiente para 100.000
amostras UNICAS sem repeticao excessiva. O range novo da ~800.000
valores validos, repeticao desprezivel para uma amostra de 100.000.

Salva (raiz, tipo=raiz mod 5, W_por_headroom) -- os scripts seguintes
(stage6_large_sample_battery.py, stage6_calibration_checks.py) leem o
arquivo local `tail_index_q5_large_sample.json` gerado aqui.
"""
import json, time, random
from multiprocessing import Pool

Q = 5
CYCLES_Q = {1, 3, 13, 33, 83, 17, 27, 43}
N_ROOTS = 100_000
V_RANGE = (1001, 2_000_001)
SEED = 20260719
H_LEVELS = [10**5, 10**6, 10**7, 10**8]
THETA = 0.650918639898
N_WORKERS = 12
OUT_FILE = "tail_index_q5_large_sample.json"


def count_tree(q, root, n_max, checkpoints):
    ordq = 1
    x = 2 % q
    while x != 1:
        x = (2 * x) % q
        ordq += 1
    cps = sorted(checkpoints)
    counts = [0] * len(cps)
    stack = [root]
    while stack:
        v = stack.pop()
        for i, cp in enumerate(cps):
            if v <= cp:
                counts[i] += 1
        a0 = None
        p = 2 % q
        for a in range(1, ordq + 1):
            if (p * v) % q == 1:
                a0 = a
                break
            p = (p * 2) % q
        if a0 is None:
            continue
        a = a0
        while True:
            w = ((1 << a) * v - 1) // q
            if w > n_max:
                break
            if w != root:
                stack.append(w)
            a += ordq
    return counts


def sample_roots(n, rng):
    roots = []
    tried = 0
    while len(roots) < n and tried < n * 3:
        tried += 1
        v = rng.randrange(V_RANGE[0], V_RANGE[1], 2)
        if v % Q == 0 or v in CYCLES_Q:
            continue
        roots.append(v)
    return roots


def process_root(v):
    checkpoints = [v * H for H in H_LEVELS]
    counts = count_tree(Q, v, v * H_LEVELS[-1], checkpoints)
    W = [c / (H ** THETA) for c, H in zip(counts, H_LEVELS)]
    return v, W


def main():
    rng = random.Random(SEED)
    roots = sample_roots(N_ROOTS, rng)
    print(f"Amostrando {len(roots)} raizes (V_RANGE={V_RANGE}), "
          f"{N_WORKERS} workers, headrooms {H_LEVELS}", flush=True)
    t0 = time.time()

    results = []
    with Pool(N_WORKERS) as pool:
        for idx, (v, W) in enumerate(pool.imap_unordered(process_root, roots, chunksize=50)):
            results.append((v, W))
            if (idx + 1) % 5000 == 0:
                dt = time.time() - t0
                print(f"  {idx+1}/{len(roots)} raizes processadas ({dt:.1f}s, "
                      f"{dt/(idx+1)*1000:.2f}ms/raiz media)", flush=True)

    dt = time.time() - t0
    print(f"Total: {dt:.1f}s para {len(results)} raizes", flush=True)

    out = {
        "roots": [r[0] for r in results],
        "types": [r[0] % Q for r in results],
        "W_by_level": {str(H): [r[1][i] for r in results] for i, H in enumerate(H_LEVELS)},
    }
    with open(OUT_FILE, "w") as f:
        json.dump(out, f)
    print(f"Amostra grande salva em {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
