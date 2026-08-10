#!/usr/bin/env python3
"""
Bateria completa de estimadores do indice de cauda de W_v (q=5),
receita do Fable (consulta 2026-07-18):
  1. Gabaix-Ibragimov (rank-1/2), checagem de robustez do Zipf.
  2. Hill com correcao de vies (Huisman et al. 2001, regressao em k).
  3. MLE de GPD com grafico de estabilidade de limiar.
  4. Clauset-Shalizi-Newman (x_min via KS) + teste de Vuong vs lognormal
     truncada.

Convencao: alpha = indice de cauda de sobrevivencia, P(W>x)~x^-alpha.
Previsto: alpha = 1/0.650918639898 = 1.536290. xi_GPD = 1/alpha = 0.650919.
CUIDADO: alpha_CSN (expoente de densidade) = alpha+1 = 2.536290, NAO 1.536.

Autocontido: gera as proprias amostras via count_tree (mesmo mecanismo
de tail_index_q5_rigorous.py, nesta mesma pasta) em vez de ler um
arquivo externo.
"""
import json, math, random, time
import numpy as np
from scipy.optimize import minimize

from empirical_qx1_tree import count_tree, CYCLES

ALPHA1 = 0.650918639898
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


def generate_raw_samples():
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
            print(f"  {idx+1}/{len(roots)} raizes ({time.time()-t0:.1f}s)", flush=True)
    print(f"Total: {time.time()-t0:.1f}s\n")
    return {str(H): W for H, W in W_by_level.items()}

PREDICTED_ALPHA = 1 / 0.650918639898  # 1.536290...
PREDICTED_XI = 0.650918639898

Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ---------- 1. Gabaix-Ibragimov (rank-1/2) ----------

def gabaix_ibragimov(x, k):
    xs = np.sort(x)[::-1][:k]
    lr = np.log(np.arange(1, k + 1) - 0.5)
    lx = np.log(xs)
    b = -np.polyfit(lx, lr, 1)[0]
    se = b * math.sqrt(2.0 / k)
    return b, se


# ---------- 2. Hill classico + Huisman (bias-corrected) ----------

def hill_estimator(sample_desc, k):
    xk = sample_desc[k]
    if xk <= 0:
        return None
    s = sum(math.log(sample_desc[i] / xk) for i in range(k) if sample_desc[i] > 0)
    if s <= 0:
        return None
    return k / s


def hill_curve(x):
    xs = np.sort(x)[::-1]
    lx = np.log(xs)
    n = len(xs)
    cums = np.cumsum(lx)
    ks = np.arange(1, n)
    gamma = cums[:-1] / ks - lx[1:]
    return ks, gamma


def huisman_fit(x, kappa=None):
    ks, g = hill_curve(x)
    kappa = kappa or len(x) // 2
    k = ks[:kappa].astype(float)
    g = g[:kappa]
    w = np.sqrt(k)
    X = np.column_stack([np.ones_like(k), k]) * w[:, None]
    beta, *_ = np.linalg.lstsq(X, g * w, rcond=None)
    gamma0 = beta[0]
    return 1.0 / gamma0, ks, g


def huisman_bootstrap_ci(x, kappa, n_boot=500, rng=None):
    rng = rng or np.random.default_rng(42)
    n = len(x)
    x = np.asarray(x)
    ests = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        resampled = x[idx]
        try:
            a, _, _ = huisman_fit(resampled, kappa)
            if np.isfinite(a) and 0 < a < 20:
                ests.append(a)
        except Exception:
            continue
    ests.sort()
    if not ests:
        return None, None, None
    lo = ests[int(0.025 * len(ests))]
    hi = ests[int(0.975 * len(ests))]
    return float(np.mean(ests)), lo, hi


# ---------- 3. GPD MLE + threshold stability ----------

def gpd_fit(y):
    y = np.asarray(y)
    m, v = y.mean(), y.var()
    if v > 0 and 0 < 0.5 * (1 - m * m / v) < 1:
        xi0 = 0.5 * (1 - m * m / v)
        sig0 = 0.5 * m * (m * m / v + 1)
    else:
        xi0, sig0 = 0.5, m / 2

    def nll(p):
        sig, xi = math.exp(p[0]), p[1]
        z = 1 + xi * y / sig
        if np.any(z <= 0):
            return 1e10
        return len(y) * p[0] + (1 + 1 / xi) * np.log(z).sum()

    r = minimize(nll, [math.log(sig0), xi0], method="Nelder-Mead",
                 options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})
    sig, xi = math.exp(r.x[0]), r.x[1]
    se_xi = (1 + xi) / math.sqrt(len(y))
    return xi, se_xi, sig


def gpd_threshold_stability(x, quantiles):
    x = np.sort(np.asarray(x))
    out = []
    for q in quantiles:
        u = np.quantile(x, q)
        y = x[x > u] - u
        if len(y) < 30:
            continue
        xi, se, sig = gpd_fit(y)
        out.append({"quantile": q, "u": float(u), "n_exc": len(y),
                     "xi": xi, "se_xi": se, "alpha": 1 / xi if xi > 0 else None})
    return out


# ---------- 4. CSN (x_min via KS) + Vuong vs lognormal truncada ----------

def csn_fit(x):
    xs = np.sort(np.asarray(x))
    xs = xs[xs > 0]
    n = len(xs)
    candidates = np.unique(xs)
    # restringe candidatos para manter n_tail razoavel e custo controlado
    if len(candidates) > 300:
        candidates = candidates[np.linspace(0, len(candidates) - 1, 300).astype(int)]
    best = None
    for u in candidates:
        tail = xs[xs >= u]
        n_u = len(tail)
        if n_u < 50:
            continue
        s = np.log(tail / u).sum()
        if s <= 0:
            continue
        alpha_csn = 1 + n_u / s
        # KS contra CDF de Pareto continua
        emp_rank = np.arange(1, n_u + 1) / n_u
        pred_cdf = 1 - (u / tail) ** (alpha_csn - 1)
        D = np.max(np.abs(emp_rank - pred_cdf))
        if best is None or D < best["D"]:
            best = {"u": float(u), "alpha_csn": alpha_csn, "D": float(D), "n_tail": n_u}
    return best


def lognorm_trunc_nll(p, xt, xmin):
    mu, ls = p
    s = math.exp(ls)
    lx = np.log(xt)
    tail = 1 - Phi((math.log(xmin) - mu) / s)
    if tail <= 1e-300:
        return 1e10
    ll = -np.log(xt * s * math.sqrt(2 * math.pi)) - (lx - mu) ** 2 / (2 * s * s) - math.log(tail)
    return -ll.sum()


def lognorm_trunc_fit(xt, xmin):
    lx = np.log(xt)
    r = minimize(lognorm_trunc_nll, [lx.mean(), math.log(max(lx.std(), 1e-3))],
                 args=(xt, xmin), method="Nelder-Mead",
                 options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})
    mu, s = r.x[0], math.exp(r.x[1])
    return mu, s


def vuong_test(xt, xmin, alpha_csn, mu_ln, s_ln):
    xt = np.asarray(xt)
    logp_pl = np.log(alpha_csn - 1) - np.log(xmin) - alpha_csn * np.log(xt / xmin)
    tail = 1 - Phi((math.log(xmin) - mu_ln) / s_ln)
    logp_ln = (-np.log(xt * s_ln * math.sqrt(2 * math.pi))
               - (np.log(xt) - mu_ln) ** 2 / (2 * s_ln ** 2) - math.log(tail))
    ell = logp_pl - logp_ln
    n_u = len(xt)
    R = ell.sum()
    ellbar = R / n_u
    sigma2 = np.mean((ell - ellbar) ** 2)
    if sigma2 <= 0:
        return R, None, None
    z = R / (math.sqrt(sigma2) * math.sqrt(n_u))
    p = 2 * (1 - Phi(abs(z)))
    return R, z, p


def run_all(x, label):
    x = np.array([w for w in x if w > 0])
    n = len(x)
    print(f"\n{'='*70}\n{label} (n={n})\n{'='*70}")
    results = {}

    print("\n--- 1. Gabaix-Ibragimov (rank-1/2) ---")
    gi_res = {}
    for frac in [0.01, 0.02, 0.05, 0.10]:
        k = max(10, int(n * frac))
        b, se = gabaix_ibragimov(x, k)
        lo, hi = b - 1.96 * se, b + 1.96 * se
        print(f"  frac={frac:.2f} (k={k}): alpha_GI={b:.4f}  SE={se:.4f}  IC95%=[{lo:.4f},{hi:.4f}]")
        gi_res[frac] = {"alpha": b, "se": se, "ci_lo": lo, "ci_hi": hi}
    results["gabaix_ibragimov"] = gi_res

    print("\n--- 2. Huisman (bias-corrected Hill) ---")
    huis_res = {}
    for kappa_frac, label_k in [(0.5, "kappa=n/2"), (0.25, "kappa=n/4")]:
        kappa = int(n * kappa_frac)
        alpha_h, ks, gamma = huisman_fit(x, kappa)
        mean_boot, lo, hi = huisman_bootstrap_ci(x, kappa, n_boot=300)
        # diagnostico de linearidade: correlacao entre k e gamma
        corr = np.corrcoef(ks[:kappa], gamma[:kappa])[0, 1]
        print(f"  {label_k}: alpha_Huisman={alpha_h:.4f}  bootstrap_mean={mean_boot:.4f}  "
              f"IC95%=[{lo:.4f},{hi:.4f}]  corr(k,gamma)={corr:.3f}")
        huis_res[label_k] = {"alpha": alpha_h, "boot_mean": mean_boot, "ci_lo": lo, "ci_hi": hi,
                              "corr_k_gamma": float(corr)}
    results["huisman"] = huis_res

    print("\n--- 3. GPD MLE + estabilidade de limiar ---")
    quantiles = [0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 0.99]
    gpd_res = gpd_threshold_stability(x, quantiles)
    for r in gpd_res:
        cover = "SIM" if (r["xi"] - 1.96 * r["se_xi"] <= PREDICTED_XI <= r["xi"] + 1.96 * r["se_xi"]) else "nao"
        print(f"  q={r['quantile']:.3f} u={r['u']:.3f} n_exc={r['n_exc']:4d}  "
              f"xi={r['xi']:.4f}±{1.96*r['se_xi']:.4f}  alpha={r['alpha']:.4f}  "
              f"cobre xi_previsto={PREDICTED_XI:.4f}? {cover}")
    results["gpd"] = gpd_res

    print("\n--- 4. CSN (x_min via KS) + Vuong vs lognormal truncada ---")
    csn = csn_fit(x)
    if csn:
        xmin, alpha_csn, D, n_tail = csn["u"], csn["alpha_csn"], csn["D"], csn["n_tail"]
        alpha_survival = alpha_csn - 1
        print(f"  x_min={xmin:.4f}  n_tail={n_tail}  D_KS={D:.4f}")
        print(f"  alpha_CSN(densidade)={alpha_csn:.4f}  =>  indice de sobrevivencia={alpha_survival:.4f}  "
              f"(previsto {PREDICTED_ALPHA:.4f})")
        xt = x[x >= xmin]
        mu_ln, s_ln = lognorm_trunc_fit(xt, xmin)
        R, z, p = vuong_test(xt, xmin, alpha_csn, mu_ln, s_ln)
        veredito = ("power law favorecida" if (z is not None and p < 0.05 and R > 0) else
                     "lognormal favorecida" if (z is not None and p < 0.05 and R < 0) else
                     "indistinguiveis nestes dados")
        print(f"  lognormal truncada: mu={mu_ln:.4f} sigma={s_ln:.4f}")
        print(f"  Vuong: R={R:.4f}  z={z:.4f}  p={p:.4f}  =>  {veredito}")
        results["csn_vuong"] = {"xmin": xmin, "alpha_csn": alpha_csn, "alpha_survival": alpha_survival,
                                 "D_ks": D, "n_tail": n_tail, "mu_ln": mu_ln, "s_ln": s_ln,
                                 "vuong_R": R, "vuong_z": z, "vuong_p": p, "veredito": veredito}
    else:
        print("  CSN nao convergiu (amostra insuficiente)")
        results["csn_vuong"] = None

    return results


def main():
    raw = generate_raw_samples()

    all_results = {}
    for H_str, W in raw.items():
        all_results[H_str] = run_all(W, f"H={float(H_str):.0e}")

    with open("full_battery_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\n\nResultados completos salvos em full_battery_results.json")


if __name__ == "__main__":
    main()
