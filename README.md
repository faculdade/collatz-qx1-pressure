# collatz-qx1-pressure — verification material

This repository holds the code and data behind the paper "A Closed-Form
Pressure Equation for the Accelerated $qx+1$ Branching Process"
(project's main repository,
`ResearchOS/projects/collatz/papers/06-pressao-qx1-ramificacao/`).
Every numerical claim in the paper is reproducible from a script in
this repository.

The paper (LaTeX/PDF) is **not** here — it lives only in the project's
main repository. This repository is purely for **verification**.

## How to use this table

Enter `sec3-pressure-and-transition/`, read the local `README.md` (more
detailed, covers every file), and run the indicated commands.

| Folder | Paper sections | What it verifies | Expected result |
|---|---|---|---|
| `sec3-pressure-and-transition/` | §3–§7 | The pressure identity (transfer-operator eigenvalue vs.\ closed form), the two roots of $q^{\alpha-1}=2^\alpha-1$ and the structural transition at $q=5$, the freezing computation, the $L^2/L^p$ collision spectrum, the tail-index battery of estimators at $q=5$ (multiple rounds, including a large follow-up at $20\times$ the sample size), and empirical counting-slope confirmation on real reverse trees | root table ($\alpha^\ast(3)=2$ exact, $\alpha^\ast(5)=0.650919$, etc.); transfer-operator eigenvalue matches closed form to machine precision at every $(q,k,\alpha)$ tested; tail-index battery mixed at $n=5000$, confirmatory at the $n=10^5$ follow-up |

## Requirements

Python 3 with NumPy, SciPy, and (for the large-sample tail-index
battery) enough memory/time budget noted in the local README.

## Provenance

Mirrored from this project's pre-existing
`collatz-endogeny/sec3-pressure-equation/` (created before paper 01
was split) on 2026-08-10, when paper 06 was split off
`01-syracuse-qx1-endogenia`. Every script was re-run in this
repository's own copy before being committed, not just copied.
