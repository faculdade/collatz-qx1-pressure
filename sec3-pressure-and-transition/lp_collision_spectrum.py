#!/usr/bin/env python3
"""Finite-level L^p spectrum of Tao's Syracuse density martingale.

For the law mu_ell of the Syracuse sum modulo 3^ell, compute

    L_p(ell) = 3^(ell*(p-1)) * sum_u mu_ell(u)^p
             = ||d mu_ell / d Haar_ell||_p^p.

For integer p this is 3^(ell*(p-1)) times the p-fold collision
probability.  The recursion is the audited E-100 recursion.  The output
is finite-level evidence only and makes no asymptotic decision.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np


DEFAULT_P = (1.10, 1.25, 1.50, 1.75, 1.90, 2.00, 2.10, 2.50)


def build_nu(p_prev, ell):
    modulus = 3**ell
    nu = np.zeros(modulus)
    nu[1 + 3 * np.arange(len(p_prev))] = p_prev
    return nu


def solve_level(p_prev, ell, s_max=100):
    modulus = 3**ell
    orbit_size = 2 * 3 ** (ell - 1)
    nu = build_nu(p_prev, ell)

    orbit = np.empty(orbit_size, dtype=np.int64)
    orbit[0] = 1
    for k in range(1, orbit_size):
        orbit[k] = (2 * orbit[k - 1]) % modulus
    forcing = nu[orbit]

    density = np.zeros(orbit_size)
    for s in range(1, s_max + 1):
        density += 0.5**s * np.roll(forcing, -s)

    law = np.zeros(modulus)
    law[orbit] = density
    return law


def normalized_moment(law, ell, exponent):
    return float(3.0 ** (ell * (exponent - 1.0)) * np.sum(law**exponent))


def compute(max_ell, exponents):
    law = np.zeros(3)
    law[1] = 1.0 / 3.0
    law[2] = 2.0 / 3.0
    rows = []

    for ell in range(1, max_ell + 1):
        start = time.time()
        if ell > 1:
            law = solve_level(law, ell)
        if abs(float(law.sum()) - 1.0) > 2e-10:
            raise RuntimeError(f"mass check failed at ell={ell}: {law.sum()}")
        moments = [normalized_moment(law, ell, exponent) for exponent in exponents]
        rows.append((ell, moments))
        values = "  ".join(f"p={p:g}:{m:.8g}" for p, m in zip(exponents, moments))
        print(f"ell={ell:2d}  {values}  seconds={time.time()-start:.3f}", flush=True)

    return rows


def write_csv(path, rows, exponents):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ell", *(f"Lp_power_p={p:g}" for p in exponents)])
        for ell, moments in rows:
            writer.writerow([ell, *(f"{moment:.17g}" for moment in moments)])


def summarize(rows, exponents):
    print("\nFinite-level diagnostics (last step and last five levels):")
    for index, exponent in enumerate(exponents):
        series = np.array([moments[index] for _, moments in rows])
        ratio = series[-1] / series[-2] if len(series) > 1 else float("nan")
        tail = min(5, len(series))
        x = np.array([ell for ell, _ in rows[-tail:]], dtype=float)
        slope = np.polyfit(x, np.log(series[-tail:]), 1)[0] if tail > 1 else float("nan")
        print(f"p={exponent:g}  last_ratio={ratio:.8f}  log_slope_per_level={slope:.8f}")
    print("These diagnostics do not determine boundedness or divergence.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-ell", type=int, default=14)
    parser.add_argument("--p", type=float, nargs="*", default=DEFAULT_P)
    parser.add_argument("--output", type=Path, default=Path("lp_collision_spectrum.csv"))
    args = parser.parse_args()
    if args.max_ell < 1:
        parser.error("--max-ell must be positive")
    if any(exponent <= 1.0 for exponent in args.p):
        parser.error("every exponent must be greater than 1")

    rows = compute(args.max_ell, args.p)
    write_csv(args.output, rows, args.p)
    summarize(rows, args.p)


if __name__ == "__main__":
    main()
