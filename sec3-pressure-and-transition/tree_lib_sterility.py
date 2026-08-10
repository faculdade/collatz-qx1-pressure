"""Copia isolada (sem efeitos colaterais de import) de count_tree/CYCLES,
extraida de E-097/empirical_qx1_tree.py."""


def count_tree(q, root, n_max, checkpoints=None):
    """DFS; retorna contagem total e contagens em checkpoints (<=x)."""
    ordq = 1
    x = 2 % q
    while x != 1:
        x = (2 * x) % q
        ordq += 1
    cps = sorted(checkpoints) if checkpoints else []
    counts = [0] * len(cps)
    total = 0
    stack = [root]
    while stack:
        v = stack.pop()
        total += 1
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
    return total, counts


CYCLES = {5: {1, 3, 13, 33, 83, 17, 27, 43}, 7: {1}}
