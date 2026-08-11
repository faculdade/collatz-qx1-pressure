"""
Extreme-value-theory (block-maxima) test of the tail index at q=3,
backing the second leg ("extreme-value theory on block maxima") of the
abstract's and Sec. 5's q=3 evidence for Conjecture~real-tree-tail,
alongside the Hill estimator in real_tree_tail_q3.py.

Self-contained: imports only count_tree from empirical_qx1_tree.py
(no other module-level side effects, since that file's own demo code
is guarded behind `if __name__ == "__main__":`).

Method (Fisher-Tippett-Gnedenko): if Pr(W>x) ~ C*x^{-alpha} (regularly
varying tail, alpha=2 predicted at q=3), the maximum of a block of size
n, suitably normalized, converges to a Frechet law with shape
xi=1/alpha=0.5. Consequence, simpler to test than fitting the full GEV:
the typical (median) block maximum should grow like n^{1/alpha}=n^0.5,
testable via a log-log regression of median block maximum against
block size. A GEV fit (scipy) on one block size is a secondary check.

Reproduzir: python3 evt_frechet_q3.py
"""
import math
import random
import statistics

from scipy.stats import genextreme

from empirical_qx1_tree import count_tree

H = 10 ** 5
V_RANGE = (1001, 50001)
N_TOTAL = 2500
SEED = 20260811
PREDICTED_XI = 0.5  # = 1/alpha, alpha=2


def sample_W(n, rng):
    vals = []
    tried = 0
    while len(vals) < n and tried < n * 20:
        tried += 1
        v = rng.randrange(V_RANGE[0], V_RANGE[1], 2)
        if v % 3 == 0 or v == 1:
            continue
        total, counts = count_tree(3, v, v * H, checkpoints=[v * H])
        w = counts[0] / H
        if w > 0:
            vals.append(w)
    return vals


def block_maxima(vals, block_size, rng):
    shuffled = vals[:]
    rng.shuffle(shuffled)
    n_blocks = len(shuffled) // block_size
    return [max(shuffled[i * block_size:(i + 1) * block_size]) for i in range(n_blocks)]


def main():
    print(f"Sampling {N_TOTAL} values of W_v(H), q=3, H={H:.0e}")
    rng = random.Random(SEED)
    vals = sample_W(N_TOTAL, rng)
    print(f"{len(vals)} valid values collected.\n")

    block_sizes = [10, 20, 50, 100, 150]
    log_n, log_median = [], []
    for n in block_sizes:
        maxima = block_maxima(vals, n, random.Random(SEED + n))
        med = statistics.median(maxima)
        print(f"n={n:>4}  n_blocks={len(maxima):>4}  median(max)={med:.3f}")
        log_n.append(math.log(n))
        log_median.append(math.log(med))

    nn = len(log_n)
    mx = sum(log_n) / nn
    my = sum(log_median) / nn
    cov = sum((x - mx) * (y - my) for x, y in zip(log_n, log_median)) / nn
    varx = sum((x - mx) ** 2 for x in log_n) / nn
    slope = cov / varx
    print(f"\nSlope of log(median block max) vs log(n): {slope:.4f}")
    print(f"(predicted for alpha=2: 1/alpha = {PREDICTED_XI:.4f})")

    print("\nGEV fit (scipy) on block maxima at n=100:")
    maxima_100 = block_maxima(vals, 100, random.Random(SEED + 100))
    print(f"n_blocks={len(maxima_100)}")
    c, loc, scale = genextreme.fit(maxima_100)
    xi_scipy = -c  # scipy uses c = -xi in the standard EVT convention
    print(f"scipy genextreme.fit: c={c:.4f}  loc={loc:.3f}  scale={scale:.3f}")
    print(f"xi (standard EVT convention, xi=-c) = {xi_scipy:.4f}")
    print(f"(predicted for alpha=2: xi = {PREDICTED_XI:.4f})")


if __name__ == "__main__":
    main()
