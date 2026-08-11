"""
Hill estimator on the real-tree growth scale factor W_v(H) at q=3,
backing the q=3 leg of the abstract's claim ("At q=3 the predicted
index alpha*=2 matches independent empirical measurements of that
scale factor") and the corresponding sentence in Sec. 5 of the paper.

Self-contained: reuses count_tree from empirical_qx1_tree.py (same
folder), the same DFS already verified for q=5,7 elsewhere in this
repository. At q=3, alpha_-(3)=1 exactly (the trivial root itself, per
Theorem~thm:pressure / the root table in this folder's README), so
W_v(H) := N_v(vH)/H^{alpha_-(3)} reduces to N_v(vH)/H, no fractional
exponent needed. Conjecture~real-tree-tail predicts
Pr(W>x) ~ x^{-alpha_+(3)/alpha_-(3)} = x^{-2}, since alpha_+(3)=2.

Root exclusion: 1 is the only q=3 trivial-cycle node (w_2(1)=1, a fixed
point of the smallest-branch map), and 1 is also a fixed point of the
forward map T_3 itself (T_3(1)=1), so no odd integer other than 1 ever
maps forward to 1; 1 can therefore never appear as a non-root
descendant in the reverse tree rooted at any v != 1. Excluding v=1 and
multiples of 3 from the sampled roots is sufficient (no deeper cycle
exists at q=3, unlike q=5,7).

Reproduzir: python3 real_tree_tail_q3.py
"""
import math
import random

from empirical_qx1_tree import count_tree

H = 10 ** 5
N_ROOTS = 600
V_RANGE = (1001, 50001)
SEED = 20260811
PREDICTED_ALPHA = 2.0


def sample_roots(n, rng):
    roots = []
    tried = 0
    while len(roots) < n and tried < n * 20:
        tried += 1
        v = rng.randrange(V_RANGE[0], V_RANGE[1], 2)
        if v % 3 == 0 or v == 1:
            continue
        roots.append(v)
    return roots


def hill_estimator(values, k):
    s = sorted(values, reverse=True)
    if k >= len(s):
        k = len(s) - 1
    xk1 = s[k]
    if xk1 <= 0:
        return None
    logs = [math.log(s[i] / xk1) for i in range(k)]
    mean_log = sum(logs) / k
    if mean_log <= 0:
        return None
    return 1.0 / mean_log


def main():
    rng = random.Random(SEED)
    roots = sample_roots(N_ROOTS, rng)
    print(f"Sampling {len(roots)} roots, q=3, headroom H={H:.0e}")
    W = []
    for v in roots:
        total, counts = count_tree(3, v, v * H, checkpoints=[v * H])
        w = counts[0] / H
        if w > 0:
            W.append(w)
    W.sort(reverse=True)
    n = len(W)
    print(f"n={n}, W: min={W[-1]:.4f} median={W[n // 2]:.4f} max={W[0]:.4f}")
    for frac in [0.02, 0.05, 0.10]:
        k = max(5, int(n * frac))
        alpha = hill_estimator(W, k)
        se = alpha / math.sqrt(k) if alpha else None
        print(f"Hill top {frac * 100:.0f}% (k={k}): alpha_tail = {alpha:.3f}"
              f"  (predicted {PREDICTED_ALPHA}, SE~{se:.3f})")


if __name__ == "__main__":
    main()
