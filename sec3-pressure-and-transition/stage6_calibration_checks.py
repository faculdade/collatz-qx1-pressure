"""
Estagio 6 (parte 3) -- duas calibracoes antes de promover o resultado
do Estagio 6 de "parece confirmar" para qualquer coisa mais forte:

1. NULO: roda a MESMA bateria de 4 estimadores num Pareto sintetico
   puro de indice kappa=1,536290, n=100.000 -- se GI/CSN derem os
   MESMOS valores baixos (~1,40) que os dados reais, isso e vies
   conhecido do estimador neste n/estrutura de limiar, nao desvio real.
   Se o sintetico desse 1,536 "limpo", o desvio nos dados reais seria
   real.

2. INVARIANCIA A THETA (descarta preocupacao de circularidade): kappa
   previsto = 1/theta, e W = count/H^theta -- parece autoreferente.
   Recalcula W usando theta'=0,60 (errado de proposito) e reroda a
   bateria; o indice de cauda estimado NAO deve mudar (e escalonar por
   H^(theta-theta') e uma renormalizacao que preserva o formato da
   cauda, so muda a escala) -- se mudar, ha um bug real no pipeline.

Requer ter rodado stage6_large_sample_generation.py antes.
"""
import json
import numpy as np
from full_battery import run_all, PREDICTED_ALPHA

THETA = 0.650918639898
KAPPA = PREDICTED_ALPHA  # 1.536290...
IN_FILE = "tail_index_q5_large_sample.json"

with open(IN_FILE) as f:
    data = json.load(f)

# ---------- 1. Nulo: Pareto sintetico puro, indice kappa, n=100000 ----------
print("="*70)
print("CALIBRACAO 1: Pareto sintetico puro, kappa=1,536290, n=100000")
print("="*70)
rng = np.random.default_rng(20260719)
n = 100_000
synthetic = rng.pareto(KAPPA, size=n) + 1.0
result_synth = run_all(synthetic, "PARETO SINTETICO (kappa=1,536290 exato)")

# ---------- 2. Invariancia a theta: recalcula W com theta'=0,60 ----------
print("\n" + "="*70)
print("CALIBRACAO 2: invariancia a theta (theta'=0,60 errado de proposito)")
print("="*70)
THETA_WRONG = 0.60
for H_str, W in data["W_by_level"].items():
    H = float(H_str)
    W = np.array(W)
    W_rescaled = W * (H ** (THETA - THETA_WRONG))
    print(f"\n--- H={H:.0e}, reescalado com theta'={THETA_WRONG} (vs theta real={THETA}) ---")
    run_all(W_rescaled, f"H={H:.0e} (theta'={THETA_WRONG})")

print("\nCalibracoes concluidas.")
