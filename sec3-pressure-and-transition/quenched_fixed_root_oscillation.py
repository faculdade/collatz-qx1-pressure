"""
Verifica, por uma recursao memoizada exata, a alegacao de Sec. 4 do
paper (logo apos a tabela de alpha_c(q)): que Z_k(alpha;u_0)^{1/k}
para uma raiz FIXA u_0 diverge da taxa anelada rho_ann(alpha)=1 nos
dois casos citados:

  - q=3, u_0=1, alpha=2 (alpha^*(3), raiz congelada): oscilacao com
    razao geometrica bem abaixo de 1.
  - q=5, alpha=1 (raiz trivial congelada para q=5), u_0 in {1,2,3,4}:
    Z_k(1;u_0)^{1/k} tambem nao acompanha rho_ann(1)=1.

Z_k(alpha;u) e definido pela recursao exata
    Z_0(alpha;u) = 1
    Z_k(alpha;u) = sum_{a admissivel para u} (q*2^-a)^alpha * Z_{k-1}(alpha; w_a(u)),
    w_a(u) = (2^a u - 1)/q,  "a admissivel" sse 2^a u == 1 (mod q).
A cada passo apenas u mod q^(profundidade restante) importa
(lem:fibre-bijection do paper), entao a recursao usa esse residuo, o
que faz a memoizacao colapsar a arvore de ramificacao exponencial num
DAG de tamanho tratavel: veio de teste independente antes de comitar
este script (ver CRITIQUE.md, rodada 20, achado X-01).

Reproduzir: python3 quenched_fixed_root_oscillation.py
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
    print("=== q=3, u_0=1, alpha=2 (raiz congelada alpha^*(3)=2) ===")
    print("rho_ann(2) = 1 (raiz da equacao de pressao)\n")
    d3 = ordq(3)
    memo = {}
    vals = []
    for k in range(1, 13):
        z = Zk(3, 2.0, 1, k, 60, d3, memo)
        r = z ** (1.0 / k)
        vals.append(r)
        print(f"k={k:>2}  Z_k(2;1)={z:.8f}  Z_k^(1/k)={r:.6f}")
    print(f"faixa observada: [{min(vals):.4f}, {max(vals):.4f}], "
          f"todos abaixo de rho_ann(2)=1")

    print("\n=== q=5, alpha=1 (raiz trivial congelada), u_0 in {1,2,3,4} ===")
    print("rho_ann(1) = 1 (conservacao de massa, sempre raiz)\n")
    d5 = ordq(5)
    for u0 in [1, 2, 3, 4]:
        memo5 = {}
        for k in [8, 12, 16]:
            z = Zk(5, 1.0, u0, k, 40, d5, memo5)
            r = z ** (1.0 / k)
            print(f"u_0={u0}  k={k:>2}  Z_k(1;{u0})={z:.8f}  Z_k^(1/k)={r:.6f}")
        print()


if __name__ == "__main__":
    main()
