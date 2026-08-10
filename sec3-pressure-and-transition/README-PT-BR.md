# §3 — Equação de pressão em forma fechada

Verifica o Teorema da Seção 3 do paper: o expoente de cauda $\alpha$
satisfaz $q^{\alpha-1}=2^\alpha-1$, com transição estrutural em $q=5$.

## Arquivos

- **`pressure_qx1.py`** — constrói o operador de transferência
  $M_{q,k}(\alpha)$ explicitamente (como matriz numérica) em resíduos
  mod $q^k$, para vários $k$, e verifica que seu autovalor de Perron
  bate exatamente com a forma fechada $c_q(\alpha)=q^{\alpha-1}/(2^\alpha-1)$
  — inclusive checando que **as somas de coluna são constantes e
  independentes de $k$** (a parte central do argumento). Ao final,
  resolve por bisseção (via `scipy.optimize.brentq`) as duas raízes de
  $q^{\alpha-1}=2^\alpha-1$ para $q=3,5,7,9,11,13,15$.
- **`empirical_qx1_tree.py`** — enumera a árvore reversa REAL (não a
  matriz idealizada) de $q=5$ e $q=7$ por busca em profundidade, medindo
  a inclinação de contagem por década e comparando com a previsão
  teórica. Também expõe `count_tree` e `CYCLES`, reusados pelo script
  abaixo.
- **`tail_index_q5_rigorous.py`** — primeira versão do teste do índice
  de cauda de $W_v$ para $q=5$: 5000 raízes, 4 níveis de headroom
  ($10^5$–$10^8$), estimador de Hill com IC via bootstrap em 4 frações
  de cauda, mais regressão rank-size (Zipf) independente. Superado por
  `full_battery.py` abaixo (mantido por histórico). Resultado de
  referência em `tail_index_q5_results_reference.json`.
- **`full_battery.py`** — bateria completa de 4 estimadores sobre a
  mesma amostra (5000 raízes, 4 headrooms): regressão Gabaix-Ibragimov
  (correção rank−1/2), Hill com correção de viés (Huisman et al.),
  MLE de GPD com varredura de estabilidade de limiar, e
  Clauset-Shalizi-Newman + teste de Vuong contra lognormal truncada.
  Autocontido (gera as próprias amostras via `count_tree`). Ver
  "Resultado" abaixo.
- **`exact_moment_test.py`** — teste EXATO (não estatístico) do índice
  de cauda: reusa a DP de `Z_k(\theta;u)` (função de partição da
  recursão quenched) sobre a população COMPLETA de resíduos mod $5^k$
  (não amostra), calculando o momento populacional $M_k(p)$ para
  $k=5,\ldots,11$ (teto seguro de memória — $k=12$ estoura). O ponto
  onde $M_k(p)$ deixa de saturar e passa a divergir com $k$ é o índice
  de cauda medido sem ruído de estimador. Resultado de referência em
  `exact_moment_results_reference.json`.
- **`experiment_gap_check.py`** — verificação numérica exata do gap
  espectral do operador de transferência linear dual $M_\alpha$ (a
  formalização correta, distinta do Koopman $L_\alpha$ que a
  impossibilidade de estado finito do paper já proíbe): confirma que o
  espectro de $M_\alpha$ restrito a qualquer nível de truncamento $K$ é
  exatamente $\{\Lambda,0\}$, sem autovalor subdominante isolado — o
  transiente $k^{-0{,}222}$ mencionado abaixo NÃO vem do espectro deste
  operador (ver correção de terminologia mais abaixo).
- **`stage2_periodogram.py`** — testa a hipótese log-periódica para o
  transiente (pesos dos ramos são potências de 2): deriva o período
  teórico via a dicotomia aritmético/não-aritmético da teoria de
  renovação implícita (log₂5 é irracional ⟹ sem log-periodicidade
  assintótica esperada), depois testa isso contra um periodograma
  calculado sobre os dados. Autocontido (reusa `csn_fit`/`generate_raw_samples`
  de `full_battery.py`).
- **`stage4_type_constants_check.py`** — testa a previsão de família de
  escala exata por tipo de resíduo do raiz ($u_0\bmod5$): confirma
  $W_i\stackrel{d}{=}2^{-a_0(i)\theta}\cdot W^*$ (mesma distribuição
  para todo tipo, só reescalada), mas mostra que essa razão de quantis
  é invariante ao índice de cauda $\kappa$ por construção — não o
  testa. Autocontido (regenera raízes+tipo+contagens do zero).
- **`stage6_large_sample_generation.py`** / **`stage6_large_sample_battery.py`**
  / **`stage6_calibration_checks.py`** — amostra 20× maior (100.000
  raízes, paralelizada, ~70-75 min com 12 processos) e a mesma bateria
  de 4 estimadores, mais duas calibrações de sanidade (nulo sintético
  de Pareto exato; invariância a um expoente de normalização
  deliberadamente errado). Ver "Resultado" abaixo — é a evidência mais
  forte reunida para a Conjectura do índice de cauda.

- **`experiment_type_rescaling_sterility.py`** — testa se a família de
  escala por tipo ($W_i\stackrel{d}{=}2^{-a_0(i)\theta}\cdot W^*$,
  acima) sobrevive quando $2$ não é raiz primitiva mod $q$ (a árvore
  reversa então tem classes de resíduo estéreis extras fora de
  $\langle2\rangle$, além de $u\equiv0$; ver §2 do paper). Testado
  para $q=7$ (primo, 3 de 6 resíduos não-nulos não-estéreis) e $q=15$
  (composto, 4 de 8 resíduos coprimos não-estéreis). Autocontido
  (`tree_lib_sterility.py` é uma cópia isolada de `count_tree`/`CYCLES`
  de `empirical_qx1_tree.py`, sem efeitos colaterais de import).

- **`iid_tail_check_assumptions.py`** — para o martingale aditivo na
  raiz menor de pressão, no modelo de ramificação iid correspondente,
  confere numericamente as quatro hipóteses do teorema de renovação
  implícita de Liu (2000, Teorema 2.2): não degenerescência
  ($\psi'(1)<0$), existência do segundo zero $\kappa>1$, momentos
  finitos numa vizinhança compacta, e a condição não reticulada
  (irracionalidade de $\log q/\log2$). Sustenta
  Theorem (`thm:iid-tail` no paper) (§3 do paper).
- **`lp_collision_spectrum.py`** — calcula a estatística de colisão
  $L^p$ $\|M_\ell\|_p^p=3^{\ell(p-1)}\sum_x\mu_\ell(x)^p$ para vários
  $p$, reaproveitando a recursão exata de
  `sec10-l2-refutation-and-jensen/experiment_k_ell.py`. Sustenta
  Theorem (`thm:lp-collision` no paper) (§3 do paper) e, em nível finito,
  o Resultado Empírico (`thm:lp-spectrum` no paper) (§10 do paper, também
  espelhado em `sec10-l2-refutation-and-jensen/`).

## Como rodar

```
python3 pressure_qx1.py                 # ~1s, imprime a tabela de validacao + raizes
python3 empirical_qx1_tree.py           # mais lento (enumeracao de arvore real ate 1e12-1e13)
python3 tail_index_q5_rigorous.py       # ~20 min (versao antiga, superada por full_battery.py)
python3 full_battery.py                 # ~25 min, gera amostras + roda os 4 estimadores
python3 exact_moment_test.py            # ~15 min (k ate 11), usa ~10-15GB de RAM no pico
python3 experiment_gap_check.py          # segundos, so numpy
python3 stage2_periodogram.py            # ~20-25 min (gera as proprias amostras)
python3 stage4_type_constants_check.py   # ~20-25 min (idem)
python3 stage6_large_sample_generation.py  # ~70-75 min, 12 processos, escreve
                                            # tail_index_q5_large_sample.json local
python3 stage6_large_sample_battery.py      # ~2 min, requer o arquivo acima
python3 stage6_calibration_checks.py        # ~2 min, idem
python3 experiment_type_rescaling_sterility.py  # ~11s, q=7 e q=15 juntos
python3 iid_tail_check_assumptions.py           # <1s
python3 lp_collision_spectrum.py                # poucos segundos ate ell=14
```

## Resultado esperado (`pressure_qx1.py`)

Para cada `q,k,alpha` testado, `rho` (autovalor calculado numericamente)
deve bater com `c` (forma fechada) a menos de erro de ponto flutuante
(~1e-15), com `colsum[min,max]` sendo um único valor repetido (soma de
coluna constante). A tabela final de raízes deve reproduzir:

```
  q    raiz menor a1    raiz maior a2
  3   1.000000000000   2.000000000000
  5   0.650918639898   1.000000000000
  7   0.373501034431   1.000000000000
  9   0.258108023834   1.000000000000
```

## Resultado esperado (`tail_index_q5_rigorous.py`, versão antiga)

Estável entre os 4 níveis de headroom (ex. Hill em fração=5%: ~1,58 em
todos). Na fração de 5% (a mais equilibrada), Hill ≈ 1,58 (IC95%
≈ [1,41; 1,80]), próximo do previsto 1,536 — mas instável entre
frações diferentes (de ~1,39 em 10% a ~2,10 em 1%).

## Resultado (`full_battery.py`) — quadro misto, não confirmatório

O Hill com correção de viés (Huisman) é estável entre headrooms e cai
perto do previsto 1,536 — mas a varredura de estabilidade de limiar
(GPD) não mostra platô limpo, e o teste de Vuong favorece a alternativa
**lognormal** sobre a lei de potência, com significância, em 3 dos 4
níveis de headroom. Lendo os 4 estimadores pela profundidade de cauda
que cada um resume, o índice local aparente sobe suavemente de ~1,3
(janela larga) a ~2,2 (janela estreita) — sem se estabilizar perto de
um único valor. Consistente com convergência pré-assintótica lenta, não
com confirmação nem refutação.

## Resultado (`exact_moment_test.py`) — inconclusivo, com motivo identificado

Checagem de sanidade: $M_k(1{,}0)=1{,}0$ exatamente em todo $k$ (forçado
pela identidade de pressão anelada, Teorema da §3 — confirma a
implementação). Para o índice de cauda propriamente dito: $M_k(p)$
satura (incrementos decrescentes) para $p\le1{,}6$ e diverge
(incrementos crescentes) para $p\ge1{,}7$ — o que colocaria o índice
real acima do previsto 1,536 se tomado ao pé da letra. Mas a RAZÃO
entre incrementos sucessivos ainda não estabilizou para $p\le1{,}6$ em
$k=11$ (ao contrário de $p\ge1{,}7$, já estável) — assinatura clássica
de sistema ainda relaxando, não que já convergiu. Dado o transiente
conhecido de $q=5$ (decaimento $k^{-0{,}222}$ — ver correção de
terminologia abaixo), reduzir esse transiente pela metade exigiria
$k\approx250$ — inalcançável por enumeração exaustiva ($5^k$ resíduos).
**Inconclusivo, não desconfirmatório**: não dá para distinguir, com
este método, entre o valor previsto estar logo abaixo do índice real ou
bem abaixo dele.

## Correção de terminologia: o transiente $k^{-0{,}222}$ NÃO é espectral

Uma nota de sessão anterior atribuía esse transiente a "uma raiz
complexa subdominante do operador de transferência" — essa atribuição
estava **errada**. `experiment_gap_check.py` confirma exatamente a
formalização correta (o operador dual $M_\alpha$ tem espectro
$\{\Lambda,0\}$, gap perfeito, sem autovalor subdominante isolado).
`stage2_periodogram.py` testa e refuta a hipótese alternativa de que
o transiente fosse log-periódico. A origem exata do expoente
$0{,}222$ permanece sem localização — ver
`ResearchOS/projects/collatz/hypotheses/H-129-*.md` para o registro
completo desta investigação.

## Resultado (`stage4_type_constants_check.py`) — família de escala confirmada, não testa κ

Razões de quantis por tipo de resíduo batem com a previsão
$W_i\stackrel{d}{=}2^{-a_0(i)\theta}\cdot W^*$ a 2–9% de desvio, em
todos os 4 headrooms e 3 níveis de cauda testados — estável ao longo
de 4 ordens de grandeza. Mas o índice de cauda $\kappa$ se cancela
algebricamente nessa razão (verificado): este teste confirma $\theta$
e a decomposição multi-tipo, não $\kappa$.

## Resultado (Estágio 6: amostra 20× maior) — evidência passa de inconclusiva para fortemente favorável

Com 100.000 raízes (vs. 5.000 nas rodadas anteriores): GPD mostra
platô de limiar limpo pela primeira vez (ξ estável ≈0,63–0,68 em
todos os 9 níveis de limiar testados, previsto 0,6509); Huisman muito
estável (~1,545, IC95% cobrindo 1,536290, idêntico nos 4 headrooms);
Vuong deixa de favorecer lognormal (era 3 dos 4 casos antes; agora
"indistinguível" nos 4). Duas calibrações de sanidade
(`stage6_calibration_checks.py`) não revelam artefato: um Pareto
sintético exato de índice 1,536290 reproduz o mesmo padrão de vieses
dos estimadores visto nos dados reais (confirma calibração, não
enviesamento); recalcular com um expoente de normalização
deliberadamente errado ($\theta'=0{,}60$) reproduz os MESMOS números —
descarta circularidade.

**Não é confirmação nem fechamento**: o teste de Vuong dá
não-rejeição, não "lei de potência vence"; e o martingale $W$
provadamente ainda não convergiu no headroom alcançado (a mediana
cai monotonicamente com o headroom mesmo com o índice de cauda já
estável). Mas é a evidência mais forte reunida até hoje a favor da
Conjectura do índice de cauda para $q=5$ — exatamente o padrão que o
próprio paper propôs como necessário para decidir a questão.

## Resultado (`experiment_type_rescaling_sterility.py`) — família de escala sobrevive à esterilidade extra

Tanto para $q=7$ (3 tipos não-estéreis, $a_0=3,2,1$) quanto $q=15$ (4
tipos não-estéreis, $a_0=4,3,2,1$), todas as razões par-a-par
$W_i/W_j$ batem com o previsto $2^{-(a_0(i)-a_0(j))\theta}$ a
$1$–$4\%$ (mesma precisão do achado original de $q=5$ acima, "2–9%").
Esta é a metade empírica da nota de rodapé do paper (§3, discussão da
conjectura de índice de cauda) sobre por que a esterilidade extra
(quando $2$ não é raiz primitiva mod $q$) não perturba a família de
escala: analiticamente, a matriz de pressão multitipo restrita aos
tipos sobreviventes $\langle2\rangle$ tem posto $1$, e seu autovalor
de Perron colapsa, por um cancelamento independente de $d$,
exatamente para a mesma equação de pressão $q^{s-1}/(2^s-1)$ — ver
`ResearchOS/projects/collatz/hypotheses/H-130-*.md` para a derivação
completa.

## Resultado (`iid_tail_check_assumptions.py`) — o teorema de renovação implícita de Liu se aplica

Para $q=3,5,7,9,15$, as quatro hipóteses valem: $\psi'(1)<0$ em todos
os casos, o segundo zero $\kappa=\alpha_+(q)/\alpha_-(q)$ existe e
excede 1, e a checagem de irracionalidade (necessária para a condição
não reticulada) passa. Esta é a metade numérica do
Theorem (`thm:iid-tail` no paper); a citação em si (Q. Liu, *On generalized
multiplicative cascades*, SPA 86 (2000), 263–286, Teorema 2.2) foi
conferida diretamente contra a fonte primária — ver
`ResearchOS/projects/collatz/hypotheses/H-132-*.md`.

## Resultado (`lp_collision_spectrum.py`) — uma família exata de um parâmetro em torno da estatística $L^2$

Para $\ell$ até 14, momentos abaixo de $p=2$ crescem mais devagar que
$p=2$ no intervalo testado, e momentos acima de $p=2$ crescem mais
rápido — compatível com, mas não prova de, um índice crítico genuíno
em $p=2$. Este cálculo finito é compartilhado com
`sec10-l2-refutation-and-jensen/`, onde sustenta o Resultado Empírico
do paper sobre o espectro $L^p$ em nível finito.

## Nota sobre a redação do paper (histórico)

Uma revisão externa apontou que uma versão anterior da prosa da Seção
3 descrevia o mecanismo de forma imprecisa, como um autômato finito em
resíduos mod $q$ (em vez de mod $q^k$). Isso já foi corrigido no paper
publicado: a Seção 3 agora enuncia e prova uma identidade de pressão
ANELADA exata (via um lema de bijeção de fibra), com uma Proposição
separada sobre a transição de congelamento quenched/anelado, e rebaixa
o índice de cauda de teorema a conjectura para $q\ge5$ — ver
`ResearchOS/projects/collatz/hypotheses/H-109-*.md` para o histórico
completo da correção.
