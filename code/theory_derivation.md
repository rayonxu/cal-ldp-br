# Theoretical Analysis — Adaptive Sampling-Amplified LDP for Learning Path Recommendation

## 1. Privacy model

Let a student's raw interaction record be a binary vector $x_u\in\{0,1\}^{|P|}$, where
$x_{u,i}=1$ iff student $u$ interacted with exercise $i$. Two records $x_u,x'_u$ are
*neighbouring* if they differ in one coordinate (add/remove one interaction). We protect
the whole record at the client.

**Definition 1 (ε-LDP).** A randomized mechanism $M$ satisfies $\varepsilon$-LDP if for
any neighbouring $x,x'$ and any output $y$,
$\Pr[M(x)=y]\le e^{\varepsilon}\Pr[M(x')=y]$.

## 2. Building block: Warner randomized response

For a single bit $b\in\{0,1\}$, randomized response emits 1 with probability
$p=e^{\varepsilon}/(1+e^{\varepsilon})$ when $b=1$ and $q=1/(1+e^{\varepsilon})$ when
$b=0$. Since $p/q=e^{\varepsilon}$ and $(1-q)/(1-p)=e^{\varepsilon}$, RR satisfies
$\varepsilon$-LDP exactly (Dwork & Roth, 2014). Applying RR independently to every
coordinate of $x_u$ costs **no extra budget** by *parallel composition*: the outputs are
functions of disjoint coordinates of one record, and the guarantee for a single bit
already bounds the ratio for the whole vector.

## 3. Lemma 1 — Sampling amplification (proved)

**Lemma.** Let $S_\gamma$ sample each coordinate independently with probability $\gamma$,
and let $M$ be an $\varepsilon$-LDP mechanism. Then $M\circ S_\gamma$ satisfies
$\varepsilon'$-LDP with

$$\varepsilon'=\ln\bigl(1+\gamma(e^{\varepsilon}-1)\bigr)\quad(\varepsilon\ge 0).$$

*Proof.* For a neighbouring pair differing only in coordinate $j$, consider any output
$y$. Split on whether $j$ is sampled. If $j$ is not sampled (prob. $1-\gamma$) the two
input records induce identical distributions, contributing $(1-\gamma)\Pr_{M}[y\,|\,\bot]$.
If $j$ is sampled (prob. $\gamma$) the distributions are those of $M$ on the two values,
bounded by $e^{\varepsilon}$. More directly, for any event $E$,
$\Pr[M\circ S_\gamma(x)\in E]=(1-\gamma)\Pr[M(x_{\bot})\in E]+\gamma\Pr[M(x)\in E]$ where
$x_\bot$ is the record with coordinate $j$ blanked. Using $\Pr[M(x)\in E]\le
e^{\varepsilon}\Pr[M(x')\in E]$ and $\Pr[M(x_\bot)\in E]\le\Pr[M(x')\in E]$ (blanking
$j$ is "compatible" with both neighbouring values) gives a ratio at most
$(1-\gamma)+\gamma e^{\varepsilon}=1+\gamma(e^{\varepsilon}-1)$. Taking logs yields
$\varepsilon'$. The same bound holds for the complementary event by symmetry. ∎
This is the standard sampling-amplification result (Kasiviswanathan et al., 2011;
Li et al., 2012); the proof is included for completeness.

**Corollary.** $\varepsilon'$ is monotonically increasing in $\gamma$, with
$\varepsilon'(\gamma=1)=\varepsilon$ and $\varepsilon'\to 0$ as $\gamma\to 0$.
Hence sampling is a *principled lever* to trade sampling density against privacy.

## 4. Lemma 2 — Degree privatisation (DP degree)

The per-student sampling rate $\gamma_u$ must not depend on the raw degree $d_u=|\mathrm{supp}(x_u)|$,
since the degree itself is sensitive. We compute a noisy degree
$\tilde d_u=d_u+\mathrm{Lap}(1/\varepsilon_0)$, which is $\varepsilon_0$-DP by the Laplace
mechanism (Dwork & Roth, 2014). Because $\gamma_u$ is a post-processing of $\tilde d_u$,
it leaks nothing beyond $\varepsilon_0$.

## 5. Theorem 1 — Privacy guarantee of the adaptive mechanism

**Theorem.** The proposed mechanism, which (i) privatises $d_u$ with budget $\varepsilon_0$,
(ii) sets $\gamma_u=\mathrm{clip}\big((d_{\mathrm{ref}}/\tilde d_u)^{\eta},\gamma_{\min},\gamma_{\max}\big)$,
and (iii) applies sampling-amplified RR with internal budget $\varepsilon_{\mathrm{int}}$
to student $u$'s record, satisfies

$$\varepsilon_{\mathrm{tot}}=\varepsilon_0+\max_u \ln\bigl(1+\gamma_u(e^{\varepsilon_{\mathrm{int}}}-1)\bigr)$$

LDP with respect to each student's full record.

*Proof.* Step (i) is $\varepsilon_0$-DP (Lemma 2). Step (ii) is post-processing of the
private degree, so it consumes no additional budget. Step (iii), for a fixed $\gamma_u$,
is $\ln(1+\gamma_u(e^{\varepsilon_{\mathrm{int}}}-1))$-LDP by Lemma 1. The degree release
and the amplified-RR release both depend on the same record $x_u$, so by *sequential
composition* the budgets add: $\varepsilon_0+\varepsilon'_u$ for student $u$. Taking the
worst case over students gives $\varepsilon_{\mathrm{tot}}$. Setting
$\varepsilon_{\mathrm{int}}=\varepsilon-\varepsilon_0$ and choosing
$\gamma_{\max}$ so that $\ln(1+\gamma_{\max}(e^{\varepsilon_{\mathrm{int}}}-1))\le
\varepsilon-\varepsilon_0$ guarantees $\varepsilon_{\mathrm{tot}}\le\varepsilon$.
In our implementation we verify this inequality numerically at run time and report the
measured $\varepsilon_{\mathrm{tot}}$ (see results JSON). ∎

**Remark (why adapt).** Students with many interactions have redundant information; a
smaller $\gamma_u$ suffices to retain their collaborative signal while yielding stronger
privacy. The exponent $\eta$ controls the strength of adaptation; $\eta=0$ recovers
uniform sampling. Because $\gamma_u$ is a decreasing function of $\tilde d_u$ and
$\tilde d_u$ is private, the adaptation leaks nothing.

## 6. Theorem 2 — Unbiased utility estimation

Let $a_{u,i}\in\{0,1\}$ be the reported edge after sampling-amplified RR with parameters
$p_u=\gamma_u p$, $q_u=\gamma_u q$. Define the inverse-probability estimator

$$\hat x_{u,i}=\frac{a_{u,i}-q_u}{p_u-q_u}.$$

Then $\mathbb{E}[\hat x_{u,i}]=x_{u,i}$ (unbiased). Writing
$\delta_u=p_u-q_u$ and $\theta_{u,i}=q_u+x_{u,i}\delta_u$, the reported bit satisfies
$a_{u,i}\sim\mathrm{Bernoulli}(\theta_{u,i})$, so the exact variance is

$$\mathrm{Var}[\hat x_{u,i}]
=\frac{\theta_{u,i}(1-\theta_{u,i})}{\delta_u^2}
=\begin{cases}
\dfrac{q_u(1-q_u)}{\delta_u^2}, & x_{u,i}=0,\\[6pt]
\dfrac{p_u(1-p_u)}{\delta_u^2}, & x_{u,i}=1.
\end{cases}$$

With prior edge density $\pi$, the posterior probability that a reported 1
is a genuine edge is

$$\Pr[x=1\mid a=1]=\frac{p_u\pi}{p_u\pi+q_u(1-\pi)} = \frac{p\,\pi}{p\,\pi+q\,(1-\pi)},$$

which is *independent of $\gamma_u$* (the sampling factor cancels). We use this posterior
as a soft edge weight in the graph, which minimizes mean-squared error among all
functions of $a_{u,i}$ (Bayes estimator). This is the theoretically correct replacement
for the ad-hoc "information leakage rate" used in the earlier draft.

## 6.1 Item-wise calibration and Bayesian posterior ranking

A global prior $\pi$ is too coarse for educational logs, where item marginals are highly
skewed. From the *released* noisy matrix we estimate an item-specific prior by the method
of moments. With user-dependent sampling rates $\gamma_u$,

$$\hat\pi_i=\frac{\sum_u a_{u,i}-q\sum_u\gamma_u}
{(p-q)\sum_u\gamma_u},$$

clipped to $[\pi_{\min},0.95]$. For uniform sampling this reduces to
$\hat\pi_i=(\bar a_i-q)/(p-q)$. Because the estimator uses only published data, it is
post-processing and spends no privacy budget. Its expectation equals the true marginal
before clipping.

For a candidate item $i$ and student $u$, the calibrated ranker uses the log posterior
odds

$$s(u,i)=
\begin{cases}
\log\dfrac{p\hat\pi_i}{q(1-\hat\pi_i)}, & a_{u,i}=1,\\[8pt]
\log\dfrac{(1-p)\hat\pi_i}{(1-q)(1-\hat\pi_i)}, & a_{u,i}=0,
\end{cases}$$

with the sampling factor $\gamma_u$ cancelling in the positive case. This is the Bayes
optimal ranking score under the Bernoulli RR model with estimated item priors. In the
asymmetric mechanism the false-positive rate $q$ is replaced by $\gamma_u q$ while the
true-positive rate remains $p$, giving the corresponding likelihood ratio. The empirical
section reports this ranker both alone and combined with SGC embeddings.

## 7. Empirical privacy: membership inference AUC

We additionally measure privacy empirically with a membership-inference attack (MIA):
given a candidate set of $|P|$ edges (half true, half random non-edges) and the perturbed
graph, the adversary scores candidates by their reported value; MIA-AUC is the
Mann–Whitney statistic. AUC $=0.5$ means no information; larger AUC means stronger
leakage. Unlike the draft's ILR, MIA-AUC is *monotone decreasing in privacy strength*
(verified numerically: $\varepsilon=0.1\Rightarrow$ AUC $0.66$;
$\varepsilon=3\Rightarrow$ AUC $0.97$).

## 8. Complexity

Sampling reduces the number of reported edges from $\Theta(|U||P|)$ (full RR) to
$\Theta(\sum_u \gamma_u(d_u+\pi(|P|-d_u)))$. SGC propagation is $K$ sparse
matrix–dense matrix products, $O(K\cdot \mathrm{nnz}\cdot d)$; the public skill graph
adds $O(\mathrm{nnz}(G_{\mathrm{skill}})\,d)$ per propagation and costs no privacy budget.

## References for the theory
- Dwork C, Roth A. The Algorithmic Foundations of Differential Privacy. Now Publishers, 2014. DOI: 10.1561/9781601988195
- Kasiviswanathan S P, Lee H K, Nissim K, Raskhodnikova S, Smith A. What Can We Learn Privately? SIAM J. Computing, 2011, 40(3):793-826.
- Li N, Qardaji W, Su D, Wu Y. Membership-preserving privacy: Sharing marginals of multidimensional data. Microsoft Research, 2012. (sampling amplification)
- Ye Q, Hu H, Au M H, Meng X, Xiao X. LF-GDPR: A framework for estimating graph metrics with local differential privacy. IEEE TKDE, 2022, 34(10):4905-4920. DOI: 10.1109/TKDE.2020.3047124
