"""
Calcula alpha_c(q), a raiz da entropia s(alpha) := P(alpha) - alpha*P'(alpha)
(Sec. 4 do paper, tabela apos a Proposicao "The larger root is always
frozen"), onde P(alpha) := log(rho_ann(alpha)) = (alpha-1)*log(q) - log(2^alpha-1)
e a log-pressao anelada.

Confere os quatro valores citados no paper para q em {3,5,7,9}
(1.355, 0.794, 0.564, 0.437) e o entrelacamento
alpha_-(q) < alpha_c(q) < alpha_+(q) que a Proposicao exige.

Reproduzir: python3 alpha_c_table.py
"""
from math import log
from scipy.optimize import brentq


def P(alpha, q):
    return (alpha - 1) * log(q) - log(2 ** alpha - 1)


def Pprime(alpha, q):
    return log(q) - log(2) * 2 ** alpha / (2 ** alpha - 1)


def s(alpha, q):
    return P(alpha, q) - alpha * Pprime(alpha, q)


def pressure_roots(q):
    """alpha=1 e sempre raiz; acha a outra raiz positiva de P(alpha)=0."""
    roots = [1.0]
    for lo, hi in [(1e-6, 0.999999), (1.000001, 8.0)]:
        f_lo, f_hi = P(lo, q), P(hi, q)
        if f_lo * f_hi < 0:
            roots.append(brentq(P, lo, hi, args=(q,), xtol=1e-15))
    return sorted(roots)


def alpha_c(q, alpha_minus, alpha_plus):
    return brentq(s, alpha_minus + 1e-9, alpha_plus - 1e-9, args=(q,), xtol=1e-15)


def main():
    print("=== alpha_c(q): raiz da entropia s(alpha)=P(alpha)-alpha*P'(alpha) ===\n")
    print(f"{'q':>3} {'alpha_-':>14} {'alpha_c':>14} {'alpha_+':>14}  entrelacamento")
    for q in [3, 5, 7, 9]:
        roots = pressure_roots(q)
        a_minus, a_plus = roots[0], roots[-1]
        ac = alpha_c(q, a_minus, a_plus)
        ok = a_minus < ac < a_plus
        print(f"{q:>3} {a_minus:>14.9f} {ac:>14.9f} {a_plus:>14.9f}  "
              f"{'OK' if ok else 'FALHOU'}")
        print(f"    alpha_c(q) arredondado a 3 casas: {ac:.3f}")


if __name__ == "__main__":
    main()
