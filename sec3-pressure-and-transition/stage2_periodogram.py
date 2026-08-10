"""
Estagio 2 -- teste da hipotese log-periodica na cauda de W_v (q=5).

Motivacao: o transiente k^-0,222 do teste de momento (Estagio 1) foi
inicialmente atribuido, em notas de sessao, a uma "raiz complexa
subdominante" -- suspeita corrigida depois (ver experiment_gap_check.py:
o operador de transferencia linear correto tem gap espectral exato
{Lambda,0}, sem autovalor subdominante). Levantamos entao a hipotese
alternativa de que o transiente fosse um efeito de reticulado
log-periodico (os pesos dos ramos sao potencias de 2).

Antes de testar, derivamos o periodo teorico (nao ajustado aos dados):
os multiplicadores A_a=(5*2^-a)^theta formam um reticulado deslocado
por tipo (u0 mod 5), com deslocamento b_i/s=(log_2(5)-a_0(i))/4 --
IRRACIONAL, pois log_2(5) e irracional (5 nao e potencia de 2). Pela
dicotomia aritmetico/nao-aritmetico da teoria de renovacao implicita
(Goldie 1991), este e o caso NAO-ARITMETICO: nao deve haver correcao
log-periodica assintotica (a fase gira log_2(5) por nivel, irracional,
e "lava" com k crescente -- teorema de Blackwell). Dois periodos
candidatos seriam visiveis so como artefato de profundidade finita, com
amplitude esperada DECRESCENTE em k: theta*log(2)=0,4512 ("uniao",
todos os a) e 4*theta*log(2)=1,8047 ("por-tipo", espacamento
d=ord_5(2)=4), em log natural de x.

Teste: ajuste de lei de potencia pura via CSN (full_battery.py, mesma
pasta) nas amostras de W_v, residuo log(S_emp/S_pred) em t=log(x/xmin),
potencia de Lomb-Scargle medida EXATAMENTE nesses dois periodos
pre-registrados (nao escolhidos a posteriori), comparada contra o
nivel de ruido de fundo (grade ampla de periodos).

Autocontido: reusa generate_raw_samples() e csn_fit() de full_battery.py
(mesma pasta) em vez de ler um arquivo externo.
"""
import numpy as np
from scipy.signal import lombscargle

from full_battery import csn_fit, generate_raw_samples

THETA = 0.650918639898
PERIOD_UNIAO = THETA * np.log(2)
PERIOD_TIPO = 4 * THETA * np.log(2)


def analyze(H_str, W):
    x = np.array(sorted(w for w in W if w > 0))
    csn = csn_fit(x)
    if not csn:
        return None
    xmin, alpha_csn = csn["u"], csn["alpha_csn"]
    alpha_surv = alpha_csn - 1
    xt = x[x >= xmin]
    n_u = len(xt)

    rank_desc = np.arange(n_u, 0, -1)
    S_emp = rank_desc / n_u
    S_pred = (xt / xmin) ** (-alpha_surv)
    resid = np.log(S_emp) - np.log(S_pred)
    t = np.log(xt / xmin)

    mask = (t > 1e-9) & (rank_desc > 1)
    t_use, r_use = t[mask], resid[mask]
    if len(t_use) < 20:
        return None

    ang_freqs = 2 * np.pi / np.array([PERIOD_UNIAO, PERIOD_TIPO])
    power_pred = lombscargle(t_use, r_use - r_use.mean(), ang_freqs, normalize=True)

    bg_periods = np.linspace(0.05, 5.0, 500)
    bg_power = lombscargle(t_use, r_use - r_use.mean(), 2 * np.pi / bg_periods, normalize=True)

    return {
        "H": float(H_str), "n_tail": int(n_u), "xmin": float(xmin),
        "alpha_surv": float(alpha_surv), "t_span": float(t_use.max()),
        "ciclos_uniao": float(t_use.max() / PERIOD_UNIAO),
        "ciclos_tipo": float(t_use.max() / PERIOD_TIPO),
        "power_uniao": float(power_pred[0]), "power_tipo": float(power_pred[1]),
        "bg_mean": float(bg_power.mean()), "bg_p95": float(np.percentile(bg_power, 95)),
        "bg_max": float(bg_power.max()),
    }


def main():
    raw = generate_raw_samples()

    print(f"Periodos teoricos (derivados, nao ajustados aos dados): "
          f"uniao={PERIOD_UNIAO:.4f}, por-tipo={PERIOD_TIPO:.4f} (log natural)\n")

    for H_str, W in sorted(raw.items(), key=lambda kv: float(kv[0])):
        r = analyze(H_str, W)
        if r is None:
            continue
        sig_uniao = "SIM" if r["power_uniao"] > r["bg_p95"] else "nao"
        sig_tipo = "SIM" if r["power_tipo"] > r["bg_p95"] else "nao"
        print(f"H={r['H']:.0e} (n_tail={r['n_tail']}, ciclos uniao={r['ciclos_uniao']:.1f}, "
              f"ciclos tipo={r['ciclos_tipo']:.1f}):")
        print(f"  potencia @ uniao = {r['power_uniao']:.4f}  (> p95 ruido={r['bg_p95']:.4f}? {sig_uniao})")
        print(f"  potencia @ tipo  = {r['power_tipo']:.4f}  (> p95 ruido={r['bg_p95']:.4f}? {sig_tipo})")
        print()


if __name__ == "__main__":
    main()
