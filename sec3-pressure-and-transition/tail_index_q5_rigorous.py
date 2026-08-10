#!/usr/bin/env python3
"""
Teste rigoroso do indice de cauda de W_v para q=5 (Conjectura 3.7 do
paper). Reusa count_tree de empirical_qx1_tree.py (mesma pasta).

Design:
- amostra grande de raizes (milhares), range amplo.
- MULTIPLOS niveis de headroom H numa unica passada de DFS por raiz
  (via checkpoints), para checar estabilidade da estimativa conforme
  H cresce.
- Hill estimator em varias fracoes de cauda, com bootstrap CI
  (reamostragem sobre RAIZES).
- estimador complementar via regressao log-log rank-size (Zipf plot),
  independente do Hill.
"""
import math, random, time, json

from empirical_qx1_tree import count_tree, CYCLES

ALPHA1 = 0.650918639898
PREDICTED_TAIL_INDEX = 1 / ALPHA1  # 1.536290...

N_ROOTS = 5000
H_LEVELS = [10**5, 10**6, 10**7, 10**8]
V_RANGE = (1001, 200001)
SEED = 20260718


def sample_roots(n, rng):
    roots = []
    tried = 0
    while len(roots) < n and tried < n * 3:
        tried += 1
        v = rng.randrange(V_RANGE[0], V_RANGE[1], 2)
        if v % 5 == 0 or v in CYCLES[5]:
            continue
        roots.append(v)
    return roots


def hill_estimator(sample_desc, frac):
    """sample_desc: lista ordenada decrescente. Hill classico no top frac."""
    n = len(sample_desc)
    k = max(5, int(n * frac))
    xk = sample_desc[k]
    if xk <= 0:
        return None
    s = sum(math.log(sample_desc[i] / xk) for i in range(k) if sample_desc[i] > 0)
    if s <= 0:
        return None
    return k / s


def bootstrap_hill_ci(sample, frac, n_boot=500, rng=None):
    rng = rng or random.Random(0)
    n = len(sample)
    estimates = []
    for _ in range(n_boot):
        resampled = [sample[rng.randrange(n)] for _ in range(n)]
        resampled.sort(reverse=True)
        h = hill_estimator(resampled, frac)
        if h is not None:
            estimates.append(h)
    estimates.sort()
    if not estimates:
        return None, None, None
    lo = estimates[int(0.025 * len(estimates))]
    hi = estimates[int(0.975 * len(estimates))]
    return sum(estimates) / len(estimates), lo, hi


def zipf_regression_index(sample_desc, frac):
    """Regressao log(rank) ~ a + b*log(W) no top frac; indice = -b."""
    n = len(sample_desc)
    k = max(10, int(n * frac))
    xs = []
    ys = []
    for i in range(k):
        w = sample_desc[i]
        if w <= 0:
            continue
        xs.append(math.log(w))
        ys.append(math.log(i + 1))
    m = len(xs)
    if m < 5:
        return None
    mean_x = sum(xs) / m
    mean_y = sum(ys) / m
    sxy = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(m))
    sxx = sum((xs[i] - mean_x) ** 2 for i in range(m))
    if sxx == 0:
        return None
    b = sxy / sxx
    return -b


def main():
    rng = random.Random(SEED)
    roots = sample_roots(N_ROOTS, rng)
    print(f"Amostrando {len(roots)} raizes, headrooms {H_LEVELS}, range v em {V_RANGE}")
    t0 = time.time()

    W_by_level = {H: [] for H in H_LEVELS}

    for idx, v in enumerate(roots):
        checkpoints = [v * H for H in H_LEVELS]
        tot, counts = count_tree(5, v, v * H_LEVELS[-1], checkpoints)
        for H, c in zip(H_LEVELS, counts):
            W = c / (H ** ALPHA1)
            W_by_level[H].append(W)
        if (idx + 1) % 500 == 0:
            dt = time.time() - t0
            print(f"  {idx+1}/{len(roots)} raizes processadas ({dt:.1f}s)", flush=True)

    dt = time.time() - t0
    print(f"\nTotal: {dt:.1f}s para {len(roots)} raizes x {len(H_LEVELS)} niveis de headroom\n")

    results = {}
    for H in H_LEVELS:
        sample = W_by_level[H]
        sample_desc = sorted(sample, reverse=True)
        n = len(sample_desc)
        print(f"=== H={H:.0e} (n={n}) ===")
        print(f"  W: min={sample_desc[-1]:.4f} mediana={sample_desc[n//2]:.4f} max={sample_desc[0]:.4f}")
        level_result = {"n": n, "hill": {}, "zipf": {}}
        for frac in [0.01, 0.02, 0.05, 0.10]:
            h = hill_estimator(sample_desc, frac)
            mean_boot, lo, hi = bootstrap_hill_ci(sample, frac, n_boot=300, rng=random.Random(42))
            z = zipf_regression_index(sample_desc, frac)
            print(f"  frac={frac:.2f}: Hill={h:.4f}  bootstrap_mean={mean_boot:.4f}  "
                  f"IC95%=[{lo:.4f},{hi:.4f}]  Zipf-regressao={z:.4f}  "
                  f"(previsto {PREDICTED_TAIL_INDEX:.4f})")
            level_result["hill"][frac] = {"point": h, "boot_mean": mean_boot, "ci_lo": lo, "ci_hi": hi}
            level_result["zipf"][frac] = z
        results[H] = level_result
        print()

    with open("tail_index_q5_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print("Resultados salvos em tail_index_q5_results.json")


if __name__ == "__main__":
    main()
