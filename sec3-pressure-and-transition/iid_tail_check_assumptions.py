#!/usr/bin/env python3
"""Check the closed-form hypotheses for the iid tail theorem in paper 01."""

from __future__ import annotations

import math

import mpmath as mp


def pressure(q: int, alpha: mp.mpf) -> mp.mpf:
    return q ** (alpha - 1) / (2**alpha - 1)


def roots(q: int) -> tuple[mp.mpf, mp.mpf]:
    if q == 3:
        return mp.mpf(1), mp.mpf(2)
    f = lambda x: pressure(q, x) - 1
    left = mp.mpf("1e-8")
    f_left = f(left)
    bracket = None
    for step in range(1, 10001):
        right = mp.mpf(step) / 10001
        f_right = f(right)
        if f_left * f_right < 0:
            bracket = (left, right)
            break
        left, f_left = right, f_right
    assert bracket is not None
    left, right = bracket
    for _ in range(200):
        middle = (left + right) / 2
        if f(left) * f(middle) <= 0:
            right = middle
        else:
            left = middle
    smaller = (left + right) / 2
    return smaller, mp.mpf(1)


def log_pressure_derivative(q: int, alpha: mp.mpf) -> mp.mpf:
    return mp.log(q) - mp.log(2) * 2**alpha / (2**alpha - 1)


def main() -> None:
    mp.mp.dps = 50
    print("q alpha_minus alpha_plus kappa psi_prime_1 irrationality_check")
    for q in (3, 5, 7, 9, 15):
        alpha_minus, alpha_plus = roots(q)
        kappa = alpha_plus / alpha_minus
        psi_prime_1 = alpha_minus * log_pressure_derivative(q, alpha_minus)
        irrationality_check = math.gcd(q, 2) == 1 and q > 1
        assert abs(pressure(q, alpha_minus) - 1) < mp.mpf("1e-40")
        assert abs(pressure(q, alpha_minus * kappa) - 1) < mp.mpf("1e-40")
        assert psi_prime_1 < 0
        assert kappa > 1
        print(q, mp.nstr(alpha_minus, 13), mp.nstr(alpha_plus, 13),
              mp.nstr(kappa, 13), mp.nstr(psi_prime_1, 13),
              irrationality_check)


if __name__ == "__main__":
    main()
