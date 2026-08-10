#!/usr/bin/env python3
"""
Estagio 1 (receita do Fable, 2026-07-18): teste do indice de cauda de
W_v via MOMENTO CRITICO da distribuicao POPULACIONAL exata de
Z_k(theta; u) sobre residuos u mod 5^k, em vez de estimadores
estatisticos (Hill/GPD/Vuong) sobre uma amostra de raizes.

Ideia: Z_k(theta,u) com theta=alpha_-(5)=0.650918639898 (a raiz da
equacao de pressao que ja provamos estar SEMPRE nao-congelada,
Proposicao prop:always-frozen) e a aproximacao de profundidade-k finita
do martingale W_u = lim Z_k(theta,u). Definindo o momento populacional

    M_k(p) = media sobre TODOS os residuos u mod 5^k de Z_k(theta,u)^p

a teoria de cauda regularmente variavel diz: para p < indice_de_cauda,
M_k(p) fica LIMITADO (satura) conforme k cresce; para p >
indice_de_cauda, M_k(p) CRESCE SEM LIMITE. O ponto de transicao e o
indice de cauda medido de forma exata (populacao completa, nao amostra
--  sem ruido de estimador).

Checagem de sanidade GRATUITA: por construcao da identidade de pressao
anelada exata (ja provada, Teorema no paper: soma sobre TODAS as raizes
de Z_k(alpha;raiz) = (q^alpha/(2^alpha-1))^k), em theta=alpha_- vale
q^theta/(2^theta-1)=q exatamente (definicao da raiz da equacao de
pressao), entao a soma anelada = q^k, e a MEDIA populacional M_k(1)
deve ser EXATAMENTE 1.0 para todo k. Se M_k(1) nao bater com 1.0 (a
menos de erro de truncamento de a_max), ha um bug na implementacao.

Para k pequeno (<=11): enumeracao exata de TODOS os 5^k residuos.
Para k maior: amostra aleatoria grande (a media populacional converge
a taxa classica de CLT sobre a amostra de residuos, MUITO mais benigno
que estimar indice de cauda via order statistics tipo Hill).
"""
import sys, math, random, time, json

sys.setrecursionlimit(200000)

Q = 5
THETA = 0.650918639898  # alpha_-(5), raiz da equacao de pressao q^(a-1)=2^a-1
A_MAX = 60
P_LIST = [1.0, 1.2, 1.3, 1.4, 1.45, 1.48, 1.5, 1.52, 1.536290, 1.55, 1.58, 1.6, 1.7, 1.8, 2.0]


def make_Z(q, theta, a_max=A_MAX):
    memo = {}

    def Z(depth, u0):
        if depth == 0:
            return 1.0
        key = (depth, u0)
        cached = memo.get(key)
        if cached is not None:
            return cached
        mod_next = q ** (depth - 1)
        u0_modq = u0 % q
        total = 0.0
        if u0_modq != 0:
            for a in range(1, a_max + 1):
                if (pow(2, a, q) * u0_modq) % q != 1:
                    continue
                weight = (q * 2.0 ** (-a)) ** theta
                if weight < 1e-16:
                    break
                num = (2 ** a) * u0 - 1
                w = (num // q) % mod_next
                total += weight * Z(depth - 1, w)
        memo[key] = total
        return total

    return Z, memo


def run_full(k):
    Z, memo = make_Z(Q, THETA)
    qk = Q ** k
    sums = {p: 0.0 for p in P_LIST}
    t0 = time.time()
    for u in range(qk):
        z = Z(k, u)
        for p in P_LIST:
            sums[p] += z ** p if z > 0 else 0.0
    dt = time.time() - t0
    moments = {p: sums[p] / qk for p in P_LIST}
    return moments, len(memo), dt


def run_sampled(k, n_sample, rng):
    Z, memo = make_Z(Q, THETA)
    sums = {p: 0.0 for p in P_LIST}
    t0 = time.time()
    for _ in range(n_sample):
        u = rng.randrange(Q ** k)
        z = Z(k, u)
        for p in P_LIST:
            sums[p] += z ** p if z > 0 else 0.0
    dt = time.time() - t0
    moments = {p: sums[p] / n_sample for p in P_LIST}
    return moments, len(memo), dt


def save(results):
    with open("exact_moment_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2, default=str)


def main():
    results = {}

    # k pequenos: enumeracao exata (populacao completa)
    for k in [5, 6, 7, 8, 9, 10, 11]:
        moments, memo_size, dt = run_full(k)
        print(f"k={k:2d} (populacao completa 5^{k}={5**k:>12d})  memo={memo_size:>10d}  tempo={dt:.1f}s", flush=True)
        for p in P_LIST:
            print(f"    M_k(p={p:.6f}) = {moments[p]:.6e}", flush=True)
        results[k] = {"type": "full", "n": 5 ** k, "moments": moments, "memo_size": memo_size, "time_s": dt}
        save(results)
        print(flush=True)

    # NOTA (2026-07-18): a abordagem "amostrada" para k>=12 planejada
    # originalmente foi abandonada -- verificamos que MESMO UMA UNICA
    # chamada Z(k,u0) para k grande precisa explorar essencialmente todo
    # o espaco de residuos nas profundidades rasas (mesmo problema de
    # explosao de estado que nos fez interromper verify_freezing_clean.py
    # em k=20 por seguranca de memoria), entao "amostrar" nao economiza
    # nada -- k=11 exato e o teto seguro desta abordagem. Ir mais fundo
    # exigiria uma formulacao nova (nao um ajuste de codigo), fica como
    # pergunta em aberto se a linha for retomada.
    print("Resultados salvos em exact_moment_results.json")


if __name__ == "__main__":
    main()
