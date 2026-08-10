"""
E-105 — Verificação numérica do gap espectral do operador de transferência dual M_alpha.

Contexto: consulta ao Fable (H-129) resolveu uma tensão nas notas antigas do
projeto ("raiz complexa subdominante do operador causando o transiente
k^-0,222") mostrando que ela mistura duas camadas distintas:

  - L_alpha (Koopman ponderado, filho -> pai): NAO preserva funcoes
    localmente constantes de nivel K (LC_K) -- eleva o nivel em 1 a cada
    aplicacao. Essa e exatamente a Prop. de impossibilidade de estado
    finito ja provada no paper (Exemplo ex:naive-fails).

  - M_alpha (dual, pai -> filho, via os mapas afins phi_a(w) = (q*w+1)*2^-a):
    PRESERVA LC_K exatamente (baixa o nivel em 1). E o operador certo para
    perguntar "existe gap espectral na camada linear?".

  Afirmacao teorica do Fable (triangularidade estrita em niveis de Fourier):
  o espectro de M_alpha restrito a LC_K e EXATAMENTE {Lambda} uniao {0},
  com Lambda = q^alpha/(2^alpha - 1) autovalor simples (autofuncao = 1),
  e bloco nilpotente nos niveis >=1. Ou seja: gap espectral perfeito,
  SEM autovalor subdominante isolado intrinseco -- o "0,222" nao pode vir
  do espectro deste operador.

Este script verifica essa afirmacao numericamente, para q=5, nas duas
raizes da equacao de pressao (alpha_+ = 1, congelada; alpha_- = 0,650919,
nao-congelada), em varios niveis de truncamento K.
"""
import numpy as np

def modinv(a, m):
    return pow(a, -1, m)

def build_M_alpha(q, alpha, K, a_max):
    n = q ** K
    M = np.zeros((n, n))
    inv2 = modinv(2, n)
    inv2a = 1
    for a in range(1, a_max + 1):
        inv2a = (inv2a * inv2) % n
        weight = (q * 2.0 ** (-a)) ** alpha
        if weight < 1e-16:
            break
        for w in range(n):
            v = ((q * w + 1) * inv2a) % n
            M[w, v] += weight
    return M

def report(q, alpha, K, a_max=200):
    Lambda_theory = q ** alpha / (2 ** alpha - 1)
    M = build_M_alpha(q, alpha, K, a_max)
    row_sums = M.sum(axis=1)
    eigvals = np.linalg.eigvals(M)
    eigvals_sorted = sorted(eigvals, key=lambda z: -abs(z))
    top = eigvals_sorted[0]
    rest = eigvals_sorted[1:]
    max_rest_abs = max(abs(z) for z in rest) if rest else 0.0
    print(f"q={q} alpha={alpha:.6f} K={K} (dim={q**K})")
    print(f"  Lambda teorico        = {Lambda_theory:.10f}")
    print(f"  row sums (min,max)    = {row_sums.min():.10f}, {row_sums.max():.10f}")
    print(f"  autovalor dominante   = {top.real:.10f} {'+ '+format(top.imag,'.2e')+'i' if abs(top.imag)>1e-9 else ''}")
    print(f"  |maior autovalor resto| = {max_rest_abs:.3e}  (esperado: ~0, i.e. nilpotente)")
    print()

if __name__ == "__main__":
    print("### q=5 ###")
    for alpha, label in [(1.0, "alpha_+ (congelada)"), (0.6509190247, "alpha_- (nao-congelada)")]:
        print(f"=== {label} ===")
        for K in [2, 3, 4]:
            report(5, alpha, K)

    # q=7: 2 nao e raiz primitiva mod 7 (ord_7(2)=3, <2>={1,2,4}) -- caso
    # estruturalmente diferente sinalizado pelo Fable. Confirma que o
    # espectro {Lambda,0} nao depende de 2 ser raiz primitiva mod q.
    print("### q=7 (2 nao e raiz primitiva mod 7 -- caso estrutural distinto) ###")
    print("=== alpha_+ (congelada) ===")
    for K in [2, 3]:
        report(7, 1.0, K)
