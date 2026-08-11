# §3 — The closed-form pressure equation

Verifies the Theorem from paper Section 3: the tail exponent $\alpha$
satisfies $q^{\alpha-1}=2^\alpha-1$, with a structural transition at
$q=5$.

## Files

- **`pressure_qx1.py`** — builds the transfer operator
  $M_{q,k}(\alpha)$ explicitly (as a numeric matrix) on residues mod
  $q^k$, for several $k$, and verifies its Perron eigenvalue matches
  exactly the closed form $c_q(\alpha)=q^{\alpha-1}/(2^\alpha-1)$ —
  including checking that **the column sums are constant and
  independent of $k$** (the central part of the argument). At the end,
  solves by bisection (via `scipy.optimize.brentq`) the two roots of
  $q^{\alpha-1}=2^\alpha-1$ for $q=3,5,7,9,11,13,15$.
- **`empirical_qx1_tree.py`** — enumerates the REAL reverse tree (not
  the idealized matrix) for $q=5$ and $q=7$ via depth-first search,
  measuring the count slope per decade and comparing against the
  theoretical prediction. Also exposes `count_tree` and `CYCLES`, reused
  by the script below.
- **`tail_index_q5_rigorous.py`** — the first version of the $W_v$
  tail-index test for $q=5$: 5000 roots, 4 headroom levels
  ($10^5$–$10^8$), Hill estimator with bootstrap CI at 4 tail
  fractions, plus an independent rank-size (Zipf) regression.
  Superseded by `full_battery.py` below (kept for history). Reference
  result in `tail_index_q5_results_reference.json`.
- **`full_battery.py`** — a full battery of 4 estimators on the same
  sample (5000 roots, 4 headrooms): Gabaix-Ibragimov regression
  (rank−1/2 correction), bias-corrected Hill (Huisman et al.), GPD MLE
  with threshold-stability sweep, and Clauset-Shalizi-Newman + Vuong
  test against truncated lognormal. Self-contained (generates its own
  samples via `count_tree`). See "Result" below.
- **`exact_moment_test.py`** — an EXACT (non-statistical) tail-index
  test: reuses the DP for `Z_k(\theta;u)` (the quenched recursion's
  partition function) over the COMPLETE population of residues mod
  $5^k$ (not a sample), computing the population moment $M_k(p)$ for
  $k=5,\ldots,11$ (safe memory ceiling — $k=12$ overflows). The point
  where $M_k(p)$ stops saturating and starts diverging with $k$ is the
  tail index measured with no estimator noise. Reference result in
  `exact_moment_results_reference.json`.
- **`experiment_gap_check.py`** — exact numerical verification of the
  spectral gap of the dual linear transfer operator $M_\alpha$ (the
  correct formalization, distinct from the Koopman $L_\alpha$ that the
  paper's finite-state impossibility already forbids): confirms that
  $M_\alpha$'s spectrum restricted to any truncation level $K$ is
  exactly $\{\Lambda,0\}$, with no isolated subdominant eigenvalue —
  the transient $k^{-0.222}$ mentioned below does NOT come from this
  operator's spectrum (see the terminology correction further below).
- **`stage2_periodogram.py`** — tests the log-periodic hypothesis for
  the transient (branch weights are powers of 2): derives the
  theoretical period via the arithmetic/non-arithmetic dichotomy of
  implicit renewal theory (log₂5 is irrational ⟹ no asymptotic
  log-periodicity expected), then tests this against a periodogram
  computed on the data. Self-contained (reuses `csn_fit`/`generate_raw_samples`
  from `full_battery.py`).
- **`stage4_type_constants_check.py`** — tests the prediction of an
  exact scale family by the root's residue type ($u_0\bmod5$):
  confirms $W_i\stackrel{d}{=}2^{-a_0(i)\theta}\cdot W^*$ (the same
  distribution for every type, only rescaled), but shows this quantile
  ratio is invariant to the tail index $\kappa$ by construction — it
  doesn't test it. Self-contained (regenerates roots+type+counts from
  scratch).
- **`stage6_large_sample_generation.py`** / **`stage6_large_sample_battery.py`**
  / **`stage6_calibration_checks.py`** — a 20× larger sample (100,000
  roots, parallelized, ~70-75 min with 12 processes) and the same
  battery of 4 estimators, plus two sanity calibrations (an exact
  synthetic Pareto null; invariance to a deliberately wrong
  normalization exponent). See "Result" below — this is the strongest
  evidence gathered for the Tail-Index Conjecture.

- **`experiment_type_rescaling_sterility.py`** — tests whether the
  type-rescaling family ($W_i\stackrel{d}{=}2^{-a_0(i)\theta}\cdot
  W^*$, above) survives when $2$ is not a primitive root mod $q$ (the
  reverse tree then has extra sterile residue classes outside
  $\langle2\rangle$, beyond $u\equiv0$; see the paper's §2). Tested for
  $q=7$ (prime, $3$ of $6$ nonzero residues non-sterile) and $q=15$
  (composite, $4$ of $8$ coprime residues non-sterile). Self-contained
  (`tree_lib_sterility.py` is an isolated copy of
  `count_tree`/`CYCLES` from `empirical_qx1_tree.py`, no import
  side-effects).

- **`iid_tail_check_assumptions.py`** — for the additive martingale at
  the smaller pressure root, in the matching i.i.d. branching model,
  checks numerically the four hypotheses of Liu's (2000) implicit
  renewal theorem (Theorem 2.2): non-degeneracy
  ($\psi'(1)<0$), existence of the second zero $\kappa>1$, finite
  moments in a compact neighborhood, and the non-lattice condition
  (irrationality of $\log q/\log2$). Backs
  Theorem (`thm:iid-tail` in the paper) (paper §3).
- **`lp_collision_spectrum.py`** — evaluates the $L^p$ collision
  statistic $\|M_\ell\|_p^p=3^{\ell(p-1)}\sum_x\mu_\ell(x)^p$ for
  several $p$, reusing the exact recursion of
  `sec10-l2-refutation-and-jensen/experiment_k_ell.py`. Backs
  Theorem (`thm:lp-collision` in the paper) (paper §3) and, at the finite level,
  Empirical Result (`thm:lp-spectrum` in the paper) (paper §10, mirrored again in
  `sec10-l2-refutation-and-jensen/`).

## How to run

```
python3 pressure_qx1.py                 # ~1s, prints the validation table + roots
python3 empirical_qx1_tree.py           # slower (real-tree enumeration up to 1e12-1e13)
python3 tail_index_q5_rigorous.py       # ~20 min (old version, superseded by full_battery.py)
python3 full_battery.py                 # ~25 min, generates samples + runs the 4 estimators
python3 exact_moment_test.py            # ~15 min (k up to 11), uses ~10-15GB RAM at peak
python3 experiment_gap_check.py          # seconds, only numpy
python3 stage2_periodogram.py            # ~20-25 min (generates its own samples)
python3 stage4_type_constants_check.py   # ~20-25 min (same)
python3 stage6_large_sample_generation.py  # ~70-75 min, 12 processes, writes
                                            # tail_index_q5_large_sample.json locally
python3 stage6_large_sample_battery.py      # ~2 min, requires the file above
python3 stage6_calibration_checks.py        # ~2 min, same
python3 experiment_type_rescaling_sterility.py  # ~11s, both q=7 and q=15
python3 iid_tail_check_assumptions.py           # <1s
python3 lp_collision_spectrum.py                # a few seconds up to ell=14
```

## Expected result (`pressure_qx1.py`)

For each tested `q,k,alpha`, `rho` (the numerically computed
eigenvalue) should match `c` (the closed form) up to floating-point
error (~1e-15), with `colsum[min,max]` being a single repeated value
(constant column sum). The final root table should reproduce:

```
  q    smaller root a1   larger root a2
  3   1.000000000000   2.000000000000
  5   0.650918639898   1.000000000000
  7   0.373501034431   1.000000000000
  9   0.258108023834   1.000000000000
```

## Expected result (`tail_index_q5_rigorous.py`, old version)

Stable across the 4 headroom levels (e.g. Hill at fraction=5%: ~1.58 in
all of them). At the 5% fraction (the most balanced), Hill ≈ 1.58 (95%
CI ≈ [1.41; 1.80]), close to the predicted 1.536 — but unstable across
different fractions (from ~1.39 at 10% to ~2.10 at 1%).

## Result (`full_battery.py`) — a mixed, non-confirmatory picture

Bias-corrected Hill (Huisman) is stable across headrooms and lands
close to the predicted 1.536 — but the threshold-stability sweep (GPD)
shows no clean plateau, and the Vuong test favors the **lognormal**
alternative over the power law, with significance, in 3 of the 4
headroom levels. Reading the 4 estimators by the tail depth each one
summarizes, the apparent local index rises smoothly from ~1.3 (wide
window) to ~2.2 (narrow window) — without stabilizing near a single
value. Consistent with slow pre-asymptotic convergence, neither
confirmation nor refutation.

## Result (`exact_moment_test.py`) — inconclusive, with the reason identified

Sanity check: $M_k(1.0)=1.0$ exactly at every $k$ (forced by the
annealed pressure identity, the Theorem from §3 — confirms the
implementation). For the tail index itself: $M_k(p)$ saturates
(decreasing increments) for $p\le1.6$ and diverges (increasing
increments) for $p\ge1.7$ — which would place the real index above the
predicted 1.536 if taken at face value. But the RATIO between
successive increments still hasn't stabilized for $p\le1.6$ at $k=11$
(unlike $p\ge1.7$, already stable) — the classic signature of a system
still relaxing, not one that has already converged. Given the known
q=5 transient ($k^{-0.222}$ decay — see the terminology correction
below), halving this transient would require $k\approx250$ —
unreachable by exhaustive enumeration ($5^k$ residues). **Inconclusive,
not disconfirming**: this method cannot distinguish between the
predicted value sitting just below the real index or well below it.

## Terminology correction: the $k^{-0.222}$ transient is NOT spectral

A note from an earlier session attributed this transient to "a
subdominant complex root of the transfer operator" — that attribution
was **wrong**. `experiment_gap_check.py` confirms exactly the correct
formalization (the dual operator $M_\alpha$ has spectrum
$\{\Lambda,0\}$, a perfect gap, no isolated subdominant eigenvalue).
`stage2_periodogram.py` tests and refutes the alternative hypothesis
that the transient was log-periodic. The exact origin of the exponent
$0.222$ remains unlocated — see
`ResearchOS/projects/collatz/hypotheses/H-129-*.md` for the full
record of this investigation.

## Result (`stage4_type_constants_check.py`) — scale family confirmed, doesn't test κ

Quantile ratios by residue type match the prediction
$W_i\stackrel{d}{=}2^{-a_0(i)\theta}\cdot W^*$ to within 1–19%
deviation (36 type-pair/quantile/headroom comparisons; most under 10%,
the loosest is the top-20% quantile for the type with smallest
predicted scale), across all 4 headrooms and 3 tested tail levels — stable
across 4 orders of magnitude. But the tail index $\kappa$ cancels
algebraically in this ratio (verified): this test confirms $\theta$ and
the multi-type decomposition, not $\kappa$.

## Result (Stage 6: 20× larger sample) — evidence moves from inconclusive to strongly favorable

With 100,000 roots (vs. 5,000 in earlier rounds): GPD shows a clean
threshold plateau for the first time (ξ stable ≈0.63–0.68 across all 9
tested threshold levels, predicted 0.6509); Huisman very stable
(~1.545, 95% CI covering 1.536290, identical across the 4 headrooms);
Vuong stops favoring lognormal (it was 3 of the 4 cases before; now
"indistinguishable" in all 4). Two sanity calibrations
(`stage6_calibration_checks.py`) reveal no artifact: an exact synthetic
Pareto of index 1.536290 reproduces the same estimator-bias pattern
seen in the real data (confirms calibration, not bias); recomputing
with a deliberately wrong normalization exponent ($\theta'=0.60$)
reproduces the SAME numbers — rules out circularity.

**Neither confirmation nor closure**: the Vuong test gives
non-rejection, not "power law wins"; and the martingale $W$ provably
still hasn't converged at the reached headroom (the median drops
monotonically with headroom even with the tail index already stable).
But this is the strongest evidence gathered to date in favor of the
Tail-Index Conjecture for $q=5$ — exactly the pattern the paper itself
proposed as necessary to decide the question.

## Result (`experiment_type_rescaling_sterility.py`) — scale family survives extra sterility

For both $q=7$ ($3$ non-sterile types, $a_0=3,2,1$) and $q=15$ ($4$
non-sterile types, $a_0=4,3,2,1$), all pairwise ratios
$W_i/W_j$ match the predicted $2^{-(a_0(i)-a_0(j))\theta}$ to within
$0.2$–$4.2\%$ (tighter than the original $q=5$ finding above, "1–19%").
This is the empirical half of the paper's footnote (§3, tail-index
conjecture discussion) on why extra sterility (when $2$ is not a
primitive root mod $q$) doesn't disturb the scale family: analytically,
the multitype pressure matrix restricted to the surviving types
$\langle2\rangle$ is rank $1$, and its Perron eigenvalue collapses, by
a $d$-independent cancellation, to exactly the same pressure equation
$q^{s-1}/(2^s-1)$ --- see
`ResearchOS/projects/collatz/hypotheses/H-130-*.md` for the full
derivation.

## Result (`iid_tail_check_assumptions.py`) — Liu's implicit renewal theorem applies

For $q=3,5,7,9,15$, all four hypotheses hold: $\psi'(1)<0$ in every
case, the second zero $\kappa=\alpha_+(q)/\alpha_-(q)$ exists and
exceeds 1, and the irrationality check (needed for the non-lattice
condition) passes. This is the numerical half of
Theorem (`thm:iid-tail` in the paper); the citation itself (Q. Liu, *On
generalized multiplicative cascades*, SPA 86 (2000), 263–286, Theorem
2.2) was checked directly against the primary source — see
`ResearchOS/projects/collatz/hypotheses/H-132-*.md`.

## Result (`lp_collision_spectrum.py`) — an exact one-parameter family around the $L^2$ statistic

For $\ell$ up to 14, moments below $p=2$ grow more slowly than $p=2$ on
the tested range, and moments above $p=2$ grow faster — consistent
with, but not proof of, a genuine critical index at $p=2$. This finite
computation is shared with `sec10-l2-refutation-and-jensen/`, where it
backs the paper's Empirical Result on the finite-level $L^p$ spectrum.

## Note on the paper's prose (history)

An external review pointed out that an earlier version of Section 3's
prose described the mechanism imprecisely, as a finite automaton on
residues mod $q$ (instead of mod $q^k$). This has already been
corrected in the published paper: Section 3 now states and proves an
exact ANNEALED pressure identity (via a fiber-bijection lemma), with a
separate Proposition about the quenched/annealed freezing transition,
and downgrades the tail index from theorem to conjecture for $q\ge5$
— see `ResearchOS/projects/collatz/hypotheses/H-109-*.md` for the full
history of the correction.
