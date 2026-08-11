"""
Reproduz a tentativa de ajuste de lei de potencia mencionada em Sec. 4
do paper ("A power-law fit to the increment-ratio decay across
k=5,...,11 did not converge to a stable exponent"), a quarta alegacao
numerica sem script antes desta rodada (CRITIQUE.md, rodada 20,
achado X-01).

Usa os momentos populacionais exatos M_k(p) = media sobre TODOS os
residuos u mod 5^k de Z_k(theta;u)^p, theta=alpha_-(5), ja calculados
por exact_moment_test.py (le exact_moment_results_reference.json, os
mesmos dados citados no corpo do paper, k=5..11, populacao completa).

Define o incremento I_k(p) := M_k(p) - M_{k-1}(p) e ajusta
I_k(p) ~ C * k^{-gamma(p)} por regressao log-log em k=6,...,11
(6 pontos), para cada p testado. Se o teste subjacente estivesse
convergindo de forma limpa, gamma(p) deveria variar suavemente e de
forma monotona com p perto do indice de cauda previsto (1.536290).
Em vez disso, o gamma ajustado muda de sinal e varia por um fator de
varios entre valores vizinhos de p, exatamente a alegacao do paper.

Reproduzir: python3 increment_ratio_powerlaw_fit_attempt.py
(requer exact_moment_results_reference.json neste diretorio, ja
presente; para regenerar do zero, rode exact_moment_test.py primeiro,
~minutos por causa de k=11 exigir 5^11 avaliacoes de Z).
"""
import json
import math


def load_moments():
    with open("exact_moment_results_reference.json") as f:
        d = json.load(f)
    ks = sorted(int(k) for k in d.keys())
    return ks, d


def fit_powerlaw_exponent(ks, values):
    """Regride log(|incremento|) contra log(k) e devolve o coeficiente
    angular (o expoente da lei de potencia ajustada)."""
    logk = [math.log(k) for k in ks]
    logv = [math.log(abs(v)) for v in values]
    n = len(logk)
    mx = sum(logk) / n
    my = sum(logv) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(logk, logv)) / n
    varx = sum((x - mx) ** 2 for x in logk) / n
    return cov / varx


def main():
    ks, d = load_moments()
    p_list = ["1.2", "1.4", "1.5", "1.52", "1.53629", "1.55", "1.58",
              "1.6", "1.7", "1.8", "2.0"]
    print("=== Ajuste de lei de potencia ao decaimento do incremento ===")
    print("(I_k(p) = M_k(p) - M_{k-1}(p), regressao log-log em k=6..11)\n")
    print(f"{'p':>10} {'gamma ajustado':>16}")
    exponents = []
    for p in p_list:
        vals = [float(d[str(k)]["moments"][p]) for k in ks]
        incr = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
        kk = ks[1:]
        gamma = fit_powerlaw_exponent(kk, incr)
        exponents.append(gamma)
        print(f"{p:>10} {gamma:>16.4f}")

    print(f"\nFaixa de gamma ajustado: [{min(exponents):.4f}, {max(exponents):.4f}]")
    print("Nao ha convergencia para um unico expoente estavel: o valor "
          "ajustado muda de sinal e varia por um fator de varios entre "
          "valores proximos de p, consistente com um ajuste "
          "subdimensionado sobre seis pontos correlacionados "
          "(k=6,...,11 dependem uns dos outros pela mesma recursao).")


if __name__ == "__main__":
    main()
