"""
Estagio 4 -- teste da previsao colateral de constantes de cauda por
TIPO de residuo do raiz (u0 mod 5).

Derivacao: a matriz media de posto 1 do processo de ramificacao
multi-tipo, M(kappa)_ij=(1/5)c_i(kappa) com c_i(kappa)=Soma_a (5*2^-a)^
(theta*kappa) sobre a progressao admissivel do tipo i, tem autovetor
DIREITO de Perron C_i proporcional a c_i(kappa) = 2^(-a0(i)*theta*kappa),
com kappa=alpha_+/alpha_-=1/0.650918639898=1.536290 (theta*kappa=1
exatamente, raiz alpha_+=1). Tabela a0(i) (menor a admissivel p/ tipo
i=u0 mod 5): a0(1)=4, a0(2)=3, a0(3)=1, a0(4)=2.

Previsao FALSIFICAVEL: condicionado ao tipo de raiz i, a cauda de W_v
tem constante de escala proporcional a C_i. Para uma probabilidade de
cauda fixa p, o quantil x_i(p) (tal que P(W_i>x_i)=p) deveria satisfazer
x_i/x_j = (C_i/C_j)^(1/kappa).

NOTA (achado desta investigacao): (C_i/C_j)^(1/kappa) =
2^((a0(j)-a0(i))*theta) -- kappa se CANCELA algebricamente. Este teste
confirma a familia de escala exata W_i =_d 2^(-a0(i)*theta)*W* (mesma
distribuicao para todo tipo, so reescalada) e rejeita a alternativa
"C_i entra linearmente" (que preveria x_i/x_j=C_i/C_j), mas NAO testa
o indice de cauda kappa em si.

Pre-registro: comparamos razoes de quantis entre tipos, em 3 niveis de
probabilidade de cauda (top 30%, 20%, 10% dentro de cada tipo), nos 4
niveis de headroom. Reportamos razao observada, razao prevista, e o
tamanho de amostra por tipo.

Autocontido: gera as proprias amostras (raizes + tipo + W por headroom)
via count_tree (empirical_qx1_tree.py, mesma pasta), mesma
semente/parametros de full_battery.py.
"""
import numpy as np

from empirical_qx1_tree import count_tree, CYCLES

N_ROOTS = 5000
V_RANGE = (1001, 200001)
SEED = 20260718
H_LEVELS = [10**5, 10**6, 10**7, 10**8]
THETA = 0.650918639898
KAPPA = 1.0 / THETA  # = alpha_+/alpha_- = 1.536290, ja que alpha_+=1
A0 = {1: 4, 2: 3, 3: 1, 4: 2}
C = {i: 2.0 ** (-A0[i] * THETA * KAPPA) for i in A0}  # theta*kappa=1 exatamente


def sample_roots(n, rng):
    import random
    roots = []
    tried = 0
    while len(roots) < n and tried < n * 3:
        tried += 1
        v = rng.randrange(V_RANGE[0], V_RANGE[1], 2)
        if v % 5 == 0 or v in CYCLES[5]:
            continue
        roots.append(v)
    return roots


def main():
    import random, time
    rng = random.Random(SEED)
    roots = sample_roots(N_ROOTS, rng)
    types_ = np.array([v % 5 for v in roots])

    print(f"Amostrando {len(roots)} raizes, headrooms {H_LEVELS}", flush=True)
    t0 = time.time()
    W_by_level = {H: [] for H in H_LEVELS}
    for idx, v in enumerate(roots):
        checkpoints = [v * H for H in H_LEVELS]
        tot, counts = count_tree(5, v, v * H_LEVELS[-1], checkpoints)
        for H, c in zip(H_LEVELS, counts):
            W_by_level[H].append(c / (H ** THETA))
        if (idx + 1) % 1000 == 0:
            print(f"  {idx+1}/{len(roots)} raizes ({time.time()-t0:.1f}s)", flush=True)
    print(f"Total: {time.time()-t0:.1f}s\n")

    print(f"Constantes previstas C_i (i=1..4): {C}")
    print(f"Razoes previstas C_i/C_1: {{i: C[i]/C[1] for i in C}}")
    print(f"kappa = {KAPPA:.6f}\n")

    for H in H_LEVELS:
        W = np.array(W_by_level[H])
        print(f"=== H={H:.0e} ===")
        by_type = {i: W[types_ == i] for i in [1, 2, 3, 4]}
        for i in [1, 2, 3, 4]:
            print(f"  tipo {i} (a0={A0[i]}): n={len(by_type[i])}")

        for tailp, label in [(0.30, "top30%"), (0.20, "top20%"), (0.10, "top10%")]:
            quantiles = {i: float(np.quantile(by_type[i], 1 - tailp)) for i in [1, 2, 3, 4]}
            print(f"  --- {label} (quantil 1-{tailp}) ---")
            for i in [2, 3, 4]:
                obs_ratio = quantiles[i] / quantiles[1]
                pred_ratio = (C[i] / C[1]) ** (1.0 / KAPPA)
                print(f"    x_{i}/x_1: observado={obs_ratio:.3f}  previsto={pred_ratio:.3f}  "
                      f"razao(obs/prev)={obs_ratio/pred_ratio:.3f}")
        print()


if __name__ == "__main__":
    main()
