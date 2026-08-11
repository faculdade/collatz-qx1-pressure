"""
Verificacao por enumeracao direta da identidade de pressao anelada
(eq:annealed-identity no paper): para cada q e k, soma-se
Z_k(alpha;u_0) sobre TODOS os q^k residuos-raiz u_0 mod q^k, e
confere-se contra a forma fechada (q^alpha/(2^alpha-1))^k.

Isto reproduz exatamente a alegacao numerica do corpo do paper
("We verified eq:annealed-identity independently by direct
enumeration for q in {3,5,7,9}, k in {1,2,3}, and alpha in
{0.26,0.5,1,1.37,2}").

Z_k(alpha;u_0) e calculado pela recursao exata
    Z_0(alpha;u) = 1
    Z_k(alpha;u) = sum_{a admissivel para u} (q*2^-a)^alpha * Z_{k-1}(alpha; w_a(u))
onde w_a(u) = (2^a u - 1)/q, e "a admissivel" significa 2^a u == 1 (mod q).
A soma sobre a e truncada quando o peso cai abaixo de 1e-18 (a serie
geometrica em a converge rapido).

A cada nivel de recursao apenas u mod q^(profundidade restante) importa
(exatamente o fato provado no paper via a bijecao de fibra,
lem:fibre-bijection): por isso a recursao usa u reduzido modulo
q^profundidade em vez do inteiro completo, o que faz a memoizacao
colapsar o ramo exponencial num DAG de tamanho tratavel.

Reproduzir: python3 annealed_identity_direct_enumeration.py
"""


def ordq(q):
    d, x = 1, 2 % q
    while x != 1:
        x = (2 * x) % q
        d += 1
    return d


def admissible_as(u_mod_q, q, a_max, d):
    a0 = None
    p = 2 % q
    for a in range(1, d + 1):
        if (p * u_mod_q) % q == 1:
            a0 = a
            break
        p = (p * 2) % q
    if a0 is None:
        return []
    return list(range(a0, a_max + 1, d))


def Zk(q, alpha, u0, k, a_max, d, memo):
    def rec(u_mod, depth):
        if depth == 0:
            return 1.0
        key = (u_mod, depth)
        if key in memo:
            return memo[key]
        mod_next = q ** (depth - 1)
        total = 0.0
        for a in admissible_as(u_mod % q, q, a_max, d):
            w = (q * 2.0 ** (-a)) ** alpha
            if w < 1e-18:
                break
            num_exact = (2 ** a) * u_mod - 1
            assert num_exact % q == 0
            w_val = (num_exact // q) % mod_next
            total += w * rec(w_val, depth - 1)
        memo[key] = total
        return total

    return rec(u0 % (q ** k), k)


def main():
    print("=== Verificacao por enumeracao direta de eq:annealed-identity ===\n")
    max_diff = 0.0
    for q in [3, 5, 7, 9]:
        d = ordq(q)
        for k in [1, 2, 3]:
            for alpha in [0.26, 0.5, 1, 1.37, 2]:
                memo = {}
                total = sum(Zk(q, alpha, u0, k, 200, d, memo) for u0 in range(q ** k))
                closed = (q ** alpha / (2 ** alpha - 1)) ** k
                diff = abs(total - closed)
                max_diff = max(max_diff, diff)
                print(f"q={q} k={k} alpha={alpha}: "
                      f"enum={total:.10f}  closed={closed:.10f}  diff={diff:.2e}")
        print()
    print(f"Maior desvio absoluto observado: {max_diff:.2e} "
          f"(esperado: ruido de ponto flutuante, nao truncamento da soma em a)")


if __name__ == "__main__":
    main()
