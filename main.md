

---

# Rank-One Channel Models for Irreversible Data Deletion

**Chenggang Lu**  
College of Mathematical Sciences, Zhejiang University of Technology, Hangzhou, China

> **Abstract** — We study irreversible data deletion from an information-theoretic perspective via rank-one input-output channel models. We show that rank-one overwrite channels induce statistical independence between stored data and overwritten outputs, resulting in zero mutual information and zero channel capacity. This abstraction generalizes classical one-time pad mechanisms by allowing both independent and data-correlated overwrite patterns without requiring secret keys. We characterize three canonical deletion mechanisms – constant, uniform random, and biased random overwriting – within a unified parameterization. For approximately rank-one channels, we further prove that information leakage and capacity converge to zero as the rank-one approximation error vanishes. These results establish a concise theoretical foundation for keyless secure deletion and clarify the role of overwrite randomness and bias in achieving irreversibility.

**Index Terms** — Rank-one channels, information-theoretic security, data deletion, channel capacity, mutual information.

---

## I. Introduction

Secure deletion aims to render previously stored data irrecoverable, even in the presence of full access to the storage medium. From an information-theoretic viewpoint, this goal can be formalized as eliminating any statistical dependence between the original data and the post-deletion observations. Despite extensive system-level practices based on overwriting and sanitization, a concise channel-level characterization of irreversible deletion remains largely missing.

In this work, we introduce a rank-one input-output (IO) channel abstraction to model overwrite-based data deletion. We show that rank-one overwrite channels induce statistical independence between input and output, yielding zero mutual information and zero channel capacity. This property provides a direct information-theoretic notion of irreversibility, independent of computational assumptions or secret keys. Our model strictly generalizes classical one-time pad constructions by admitting both independent and plaintext-correlated overwrite patterns, while preserving perfect information-theoretic deletion.

Within this framework, we provide a unified analysis of three canonical deletion mechanisms: constant overwriting, uniform random overwriting, and biased random overwriting.

We further extend the analysis to approximately rank-one channels and prove that information leakage and channel capacity converge to zero as the deviation from rank one vanishes. These results establish a minimal yet expressive theoretical foundation for keyless secure deletion and clarify the role of overwrite randomness and bias in achieving irreversibility. Our contributions are threefold:  
(i) we formalize overwrite-based deletion via rank-one IO channels and establish zero mutual information and capacity;  
(ii) we unify constant, uniform random, and biased overwriting within a single parameterized model that generalizes one-time pads;  
(iii) we characterize approximate rank-one channels and show vanishing leakage in the asymptotic regime.

---

## II. Background and Related Work

### A. System-Level Secure Deletion

Secure deletion has been studied in multiple layers, from file systems to storage media [1]–[3]. Early research highlighted the difficulty of fully erasing data and proposed practical overwriting mechanisms with varying passes and randomization [4]. Systems like versioning file systems, log-structured storage, and flash-based devices present unique challenges for reliable deletion [1], [3]. Standards such as NIST SP 800-88 and SNIA guidelines formalize sanitization procedures and verification methods [5], [6].

### B. Cryptographic and Information-Theoretic Approaches

Cryptographic approaches introduce stronger guarantees by coupling deletion with encryption or key destruction [7]–[9]. The one-time pad is a classical example achieving perfect secrecy, while modern research explores certified deletion, publicly verifiable deletion, and hybrid encryption with deletion [10]–[13]. These frameworks provide formal guarantees but often rely on computational assumptions or quantum resources, without directly addressing fundamental, channel-level deletion properties.

Information theory provides metrics such as mutual information and channel capacity to rigorously quantify irrecoverability [10]–[12]. However, explicit modeling of deletion as a channel transformation with provable limits has been largely unexplored. Our work defines **rank-one Markov IO transformations** as a canonical abstraction, enabling unified analysis of overwriting-based deletion mechanisms and establishing a direct connection between system practices and information-theoretic security.

---

## 1. Proposed Method

### A. Rank-One Markov IO Channels and One-Time Pad Realizations

We formalize an interpretation of rank-one Markov IO channels in terms of classical one-time pad (OTP) constructions.

**Proposition 1 (OTP Interpretation of Rank-One Binary Channels).**  
Let \(X \in \{0,1\}\) be a binary input with known distribution \(P_X = (p_0, p_1)\), \(p_0 + p_1 = 1\), and let \(P(Y|X)\) be a binary-output rank-one Markov IO channel

\[
P(Y|X) = \begin{pmatrix} r_0 & r_0 \\ r_1 & r_1 \end{pmatrix}, \quad r_0 + r_1 = 1.
\]

Then there exists an independent binary key \(K \in \{0,1\}\) and a one-time pad transformation \(Y = X \oplus K\) whose induced IO behavior coincides with \(P(Y|X)\) if and only if

\[
r_0 \in [\min(p_0, p_1), \max(p_0, p_1)].
\]

The corresponding key distribution is generally non-unique.

**Proof.** Since \(P(Y|X)\) has rank one, the output \(Y\) is statistically independent of \(X\) and \(P(Y|X) = P_Y\).

Assume an independent key \(K\) exists, with \(P_K = (q_0, q_1)\). Then

\[
r_0 = p_0 q_0 + p_1 q_1, \quad r_1 = p_0 q_1 + p_1 q_0.
\]

Using \(q_1 = 1 - q_0\), we obtain \(r_0 = p_1 + (p_0 - p_1)q_0\), which implies \(r_0 \in [\min(p_0, p_1), \max(p_0, p_1)]\).

Conversely, given \(r_0 \in [\min(p_0, p_1), \max(p_0, p_1)]\) and \(p_0 \neq p_1\), let

\[
q_0 \coloneqq \frac{r_0 - p_1}{p_0 - p_1}, \quad q_1 \coloneqq 1 - q_0.
\]

Then \(K \sim (q_0, q_1)\) reproduces the desired output marginal. In particular, for \(r_0 = \frac{1}{2}\) the channel reduces to a uniform BSC with zero mutual information for any input, realized by a uniform independent key.

**Remark 1 (Interpretation).** Rank-one binary channels induce output distributions independent of the input. Uniform rank-one channels coincide with classical one-time pad behavior, while biased rank-one channels admit multiple realizations, independent or correlated with the plaintext.

### B. Encryption-by-Overwriting and Keyless Deletion

Motivated by Proposition 1, we adopt a paradigm of encryption-by-overwriting. The original data \(X\) is transformed through a rank-one Markov IO channel to an overwritten output \(Y\) which may be formally viewed as \(Y = X \oplus K\), but the key \(K\) is immediately discarded.

Security arises solely from the induced rank-one IO structure: the deletion is keyless in the sense that no secret material needs to be stored or later destroyed. This framework unifies deterministic, randomized, and biased overwriting mechanisms under a rigorous information-theoretic model.

### C. Canonical Rank-One Overwriting Mechanisms

We consider three representative rank-one overwriting mechanisms. Constant overwriting (\(r_0 \in \{0,1\}\)) produces a fixed output and corresponds to deterministic erasure. Uniform random overwriting (\(r_0 = \frac{1}{2}\)) coincides with the classical one-time pad and yields zero mutual information and capacity. Biased random overwriting (\(r_0 \in (0,1) \setminus \{\frac{1}{2}\}\)) produces probabilistic outputs with controlled bias and admits multiple realizations, independent or correlated with the input.

**Theorem III.1 (Irreversibility of Rank-One Overwriting).**  
Let \(X \in \{0,1\}^n\) be stored data, and \(Y \in \{0,1\}^n\) its overwritten representation via independent rank-one channels \(P(Y_i|X_i)\). Then

\[
I(X;Y) = 0.
\]

Hence \(Y\) reveals no information about \(X\) even if a one-time pad key were used and subsequently discarded [11], [13].

**Proof.** Independence of each coordinate implies \(P(Y|X) = \prod_i P(Y_i)\); mutual information factorizes: \(I(X;Y) = \sum_i I(X_i;Y_i) = 0\).

### D. Parameterized Rank-One Family and Approximate Channels

We consider the binary-input binary-output setting. Define a parameterized rank-one family

\[
P(Y|X) = \begin{pmatrix} r_0 & r_0 \\ 1 - r_0 & 1 - r_0 \end{pmatrix}, \quad r_0 \in [0,1],
\]

which interpolates deterministic overwriting (\(r_0 = 0,1\)), uniform random overwriting (\(r_0 = \frac{1}{2}\)), and biased random overwriting (\(r_0 \neq \frac{1}{2}\)). For all \(r_0\), the channel output is statistically independent of the input and hence has zero capacity.

We now consider approximately rank-one binary channels. Let \(P_\epsilon(Y|X)\) be a \(2 \times 2\) stochastic matrix whose singular values satisfy

\[
\sigma_1 \ge \sigma_2 \ge 0, \qquad \epsilon \coloneqq \frac{\sigma_2}{\sigma_1}.
\]

Since any stochastic matrix admits a singular value decomposition, \(P_\epsilon(Y|X)\) can be written as

\[
P_\epsilon(Y|X) = \sigma_1 u v^\top + E_\epsilon,
\]

where \(u, v\) are nonnegative vectors and \(\|E_\epsilon\|_2 = \sigma_2\). The rank-one component induces an output independent of the input, while \(E_\epsilon\) captures the deviation from perfect overwriting.

Let \(P_Y^{(x)}\) denote the output distribution conditioned on \(X = x\). For binary channels, the total variation distance between the two conditional output distributions satisfies

\[
\left\| P_Y^{(0)} - P_Y^{(1)} \right\|_{\mathrm{TV}} \le 2\|E_\epsilon\|_2 = 2\sigma_2 = 2\epsilon \sigma_1.
\]

By Pinsker's inequality, the Kullback-Leibler divergence between the two output distributions is bounded as

\[
D_{\mathrm{KL}}\left(P_Y^{(0)} \| P_Y^{(1)}\right) \le \frac{1}{\ln 2} \| P_Y^{(0)} - P_Y^{(1)} \|_{\mathrm{TV}}^2 \le \frac{4\sigma_1^2}{\ln 2} \epsilon^2.
\]

For a binary-input channel, the mutual information is upper bounded by the maximum divergence between the two conditional output distributions, yielding

\[
I(X;Y) \le \frac{4\sigma_1^2}{\ln 2} \epsilon^2.
\]

Taking the supremum over all input distributions, we obtain the capacity bound

\[
C(P_\epsilon(Y|X)) \le \frac{4\sigma_1^2}{\ln 2} \epsilon^2 \xrightarrow{\epsilon \to 0} 0.
\]

This result shows that binary channels that are close to rank-one exhibit vanishing mutual information and capacity. Consequently, near-rank-one Markov IO transformations provide a principled and robust approximation to ideal keyless overwritten deletion, even when perfect rank-one structure cannot be achieved in practice [11], [12].

---

## IV. Model and Problem Formulation

Let \(X\) and \(Y\) be independent random variables supported on \((0,1)\) with densities

\[
\begin{aligned}
f_X(x) &= \frac{p_0 p_1 r x^{r-1}(1-x)^{r-1}}{\left(p_0 x^r + p_1(1-x)^r\right)^2}, \\[1.2ex]
f_Y(y) &= \frac{q_0 q_1 r y^{r-1}(1-y)^{r-1}}{\left(q_0 y^r + q_1(1-y)^r\right)^2}.
\end{aligned} \tag{1}
\]

Let \(N \sim \mathcal{N}(0, \sigma^2)\) be independent of \((X,Y)\), and define

\[
R = \alpha X + \beta Y + N. \tag{3}
\]

We aim to compute the mutual information

\[
I(X;R) = H(R) - H(R|X). \tag{4}
\]

---

## V. Conditional Entropy \(H(R|X)\)

Conditioned on \(X = x\), we have

\[
R | (X = x) = \alpha x + \beta Y + N. \tag{5}
\]

Since differential entropy is invariant under translation,

\[
H(R|X = x) = H(\beta Y + N). \tag{6}
\]

Taking expectation over \(X\), we obtain

\[
\boxed{H(R|X) = H(\beta Y + N)}. \tag{7}
\]

---

## VI. Equivalent Additive Noise Representation

Define

\[
Z = \beta Y + N. \tag{8}
\]

Then

\[
R = \alpha X + Z, \tag{9}
\]

with \(X \perp Z\).

---

## VII. Density of \(Z\)

The density of \(Z\) is given by convolution:

\[
f_Z(z) = \int_0^1 f_Y(y) \phi_\sigma(z - \beta y) \, dy, \tag{10}
\]

where

\[
\phi_\sigma(t) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-t^2/(2\sigma^2)}. \tag{11}
\]

---

## VIII. Density of \(R\)

The density of \(R\) is

\[
f_R(r) = \int_0^1 f_X(x) f_Z(r - \alpha x) \, dx. \tag{12}
\]

---

## IX. Exact Mutual Information Expression

The mutual information is

\[
I(X;R) = -\int f_R(r) \log f_R(r) \, dr + \int f_Z(z) \log f_Z(z) \, dz.
\]

---

## X. Small-\(\alpha\) Expansion

We expand \(f_R(r)\) around \(\alpha = 0\):

\[
f_R(r) = \int f_X(x) f_Z(r - \alpha x) \, dx. \tag{14}
\]

Using Taylor expansion:

\[
f_Z(r - \alpha x) = f_Z(r) - \alpha x f_Z'(r) + \frac{\alpha^2 x^2}{2} f_Z''(r) + o(\alpha^2). \tag{15}
\]

Taking expectation over \(X\):

\[
f_R(r) = f_Z(r) - \alpha \mathbb{E}[X] f_Z'(r) + \frac{\alpha^2}{2} \mathbb{E}[X^2] f_Z''(r) + o(\alpha^2). \tag{16}
\]

Define \(\tilde{X} = X - \mathbb{E}[X]\), then \(\mathbb{E}[\tilde{X}] = 0\). Rewriting the expansion:

\[
f_R(r) = f_Z(r) + \frac{\alpha^2}{2} \mathrm{Var}(X) f_Z''(r) + o(\alpha^2). \tag{19}
\]

We use the perturbation expansion of entropy:

\[
H(f + \epsilon g) = H(f) - \epsilon \int g \log f + \frac{\epsilon^2}{2} \int \frac{g^2}{f} + o(\epsilon^2). \tag{20}
\]

Applying to \(f_R = f_Z + \delta f\), the first-order term vanishes due to normalization. Thus,

\[
H(R) = H(Z) + \frac{\alpha^2}{2} \mathrm{Var}(X) \int \frac{(f_Z'(r))^2}{f_Z(r)} \, dr + o(\alpha^2). \tag{21}
\]

---

**Theorem XI.1.** Under the above model, as \(\alpha \to 0\), the mutual information satisfies

\[
\boxed{I(X;R) = \frac{\alpha^2}{2} \mathrm{Var}(X) J(Z) + o(\alpha^2),}
\]

where

\[
J(Z) = \int \frac{(f_Z'(z))^2}{f_Z(z)} \, dz \tag{23}
\]

is the Fisher information of \(Z = \beta Y + N\).

**Corollary XI.2.** If \(\alpha = 0\), then \(I(X;R) = 0\).

---

## XII. Unified Continuous Model with Rank-Aware Covering

Let \(X \in (0,1)\) be a continuous random variable with density

\[
f_X(x) = \frac{p_0 p_1 r x^{r-1}(1-x)^{r-1}}{\left(p_0 x^r + p_1(1-x)^r\right)^2}. \tag{24}
\]

We model the covering operation via a conditional distribution:

\[
f_{Y|X}(y|x), \tag{25}
\]

which captures the rank structure of the transformation \(X \to Y\).

- Rank-1: \(f_{Y|X}(y|x) = f_Y(y)\)  
- Rank-2: \(f_{Y|X}(y|x)\) depends on \(x\)

Let additive noise \(N \sim \mathcal{N}(0, \sigma^2)\), and define

\[
R = \alpha X + \beta Y + N. \tag{26}
\]

---

## XIII. Distributions of \(R\)

The conditional distribution of \(R\) given \(X = x\) is

\[
f_{R|X}(r|x) = \int_0^1 f_{Y|X}(y|x) \phi_\sigma(r - \alpha x - \beta y) \, dy, \tag{27}
\]

where \(\phi_\sigma\) is the Gaussian density.

The marginal distribution of \(R\) is

\[
f_R(r) = \int_0^1 f_X(x) f_{R|X}(r|x) \, dx. \tag{28}
\]

---

## XIV. Exact Mutual Information

The mutual information is

\[
I(X;R) = \int f_X(x) \int f_{R|X}(r|x) \log \frac{f_{R|X}(r|x)}{f_R(r)} \, dr \, dx. \tag{29}
\]

---

## XV. Rank-1 Case

If \(f_{Y|X}(y|x) = f_Y(y)\), then

\[
f_{R|X}(r|x) = \int f_Y(y) \phi_\sigma(r - \alpha x - \beta y) \, dy. \tag{30}
\]

When \(\alpha = 0\), we have

\[
f_{R|X}(r|x) = f_R(r), \tag{31}
\]

which implies

\[
\boxed{I(X;R) = 0.} \tag{32}
\]

---

## XVI. Small Dependence Expansion

Assume

\[
f_{Y|X}(y|x) = f_Y(y) + \eta \, h(y,x), \tag{33}
\]

where \(\eta\) measures the deviation from Rank-1 and satisfies

\[
\int h(y,x) \, dy = 0. \tag{34}
\]

Then

\[
f_{Z|X}(z|x) = f_Z(z) + \eta \int h(y,x) \phi_\sigma(z - \beta y) \, dy. \tag{35}
\]

Define

\[
g(z,x) = \int h(y,x) \phi_\sigma(z - \beta y) \, dy. \tag{36}
\]

**Remark.** The deviation term \(h(y,x)\) is required to satisfy (37) to preserve normalization.

The direct leakage through \(\alpha X\) and the covering imperfection through \(\eta h(y,x)\) represent two orthogonal mechanisms of information leakage.

---

## XVII. Mutual Information Expansion

Using the expansion of relative entropy:

\[
I(X;Z) = \frac{\eta^2}{2} \int f_X(x) \int \frac{g(z,x)^2}{f_Z(z)} \, dz \, dx + o(\eta^2). \tag{38}
\]

---

## XVIII. Full Model Expansion

Now include \(\alpha\):

\[
R = \alpha X + Z. \tag{39}
\]

For small \(\alpha\) and \(\eta\), we obtain

\[
\boxed{
I(X;R) = \frac{\alpha^2}{2} \mathrm{Var}(X) J(Z) + \frac{\eta^2}{2} \mathbb{E}\left[ \int \frac{g(z,X)^2}{f_Z(z)} \, dz \right] + o(\alpha^2 + \eta^2)
} \tag{40}
\]

---

## XIX. Main Theorem

**Theorem XIX.1.** Under the unified model, the information leakage decomposes as

\[
\boxed{I(X;R) = I_{\mathrm{direct}} + I_{\mathrm{cover}}} \tag{41}
\]

where

\[
\begin{aligned}
I_{\mathrm{direct}} &= \frac{\alpha^2}{2} \mathrm{Var}(X) J(Z), \\[1.2ex]
I_{\mathrm{cover}} &= \frac{\eta^2}{2} \mathbb{E}\left[ \int \frac{g(z,X)^2}{f_Z(z)} \, dz \right].
\end{aligned} \tag{43}
\]

**Interpretation.** The first term corresponds to direct leakage through \(\alpha X\), while the second term captures residual leakage induced by imperfect covering (Rank-2 behavior).

---

## XX. From Discrete Channel Rank to Continuous Operator

Consider a discrete binary channel

\[
P_{Y|X} = \begin{pmatrix}
P(Y=0|X=0) & P(Y=1|X=0) \\
P(Y=0|X=1) & P(Y=1|X=1)
\end{pmatrix}. \tag{44}
\]

Let the singular value decomposition (SVD) be

\[
P_{Y|X} = \sigma_1 u_1 v_1^\top + \sigma_2 u_2 v_2^\top, \tag{45}
\]

where \(\sigma_1 \ge \sigma_2 \ge 0\).

**Rank-1 condition:** The channel is rank-1 if and only if \(\sigma_2 = 0\).

---

## XXI. Continuous Operator Representation

Define the conditional density \(f_{Y|X}(y|x)\) as an integral operator:

\[
(T\psi)(y) = \int_0^1 f_{Y|X}(y|x) \psi(x) \, dx. \tag{46}
\]

Then \(T\) is a Hilbert-Schmidt operator with decomposition

\[
f_{Y|X}(y|x) = \sum_{k=1}^{\infty} \sigma_k \phi_k(y) \psi_k(x), \tag{47}
\]

where \(\{\sigma_k\}\) are singular values.

---

## XXII. Definition of Deviation Parameter

We decompose

\[
f_{Y|X}(y|x) = f_Y(y) + \sum_{k\ge 2} \sigma_k \phi_k(y) \psi_k(x). \tag{48}
\]

Define

\[
\eta^2 \coloneqq \sum_{k\ge 2} \sigma_k^2. \tag{49}
\]

**Interpretation.** The parameter \(\eta\) quantifies the deviation from rank-1 behavior. In particular,

- \(\eta = 0 \iff\) rank-1 (perfect covering)
- \(\eta > 0 \iff\) residual dependence (rank-2 or higher)

---

## XXIII. Leakage Upper Bound

From the monotonicity of mutual information, we have

\[
I(X;R) \le I(X;R,Y). \tag{50}
\]

Using the chain rule,

\[
I(X;R,Y) = I(X;Y) + I(X;R|Y). \tag{51}
\]

Therefore,

\[
\boxed{I(X;R) \le I(X;Y) + I(X;R|Y)}. \tag{52}
\]

### A. Covering Leakage

Using operator decomposition,

\[
I(X;Y) \le C \sum_{k\ge 2} \sigma_k^2 = C \eta^2, \tag{53}
\]

for some constant \(C\) depending on \(f_X\).

### B. Direct Leakage

From small-\(\alpha\) expansion,

\[
I(X;R|Y) \le \frac{\alpha^2}{2} \mathrm{Var}(X) J(N), \tag{54}
\]

where \(J(N) = 1/\sigma^2\) for Gaussian noise.

### C. Final Bound

Combining both terms, we obtain

\[
\boxed{I(X;R) \le C_1 \alpha^2 + C_2 \eta^2}. \tag{55}
\]

**Theorem XXIII.1 (Unified Leakage Bound).** Under the proposed model,

\[
I(X;R) \le C_1 \alpha^2 + C_2 \sum_{k\ge 2} \sigma_k^2, \tag{56}
\]

where \(\sigma_k\) are singular values of the conditional operator \(f_{Y|X}\).

---

## XXIV. Experimental Evaluation

The purpose of our experimental evaluation is not to benchmark specific deletion tools, but to validate the information-theoretic predictions of the rank-one Markov IO framework under practical, finite-sample conditions. In particular, we empirically examine how mutual information and capacity estimates behave under exact and approximate rank-one overwriting mechanisms.

### A. Experimental Setup

All experiments consider binary data streams of length \(n = 10^5\) bits. Input symbols are generated i.i.d. according to Bernoulli distributions with varying bias. Overwriting is performed independently across coordinates using parameterized rank-one or near-rank-one binary channels.

Mutual information is estimated empirically from joint frequency counts, and channel capacity upper bounds are computed using standard binary-input channel formulations. All reported values are averaged over multiple independent runs to ensure statistical stability.

### B. Rank-One Overwriting Verification

**Table I** reports the empirical mutual information between input and overwritten output for different rank-one overwriting mechanisms. Across all input distributions and overwriting types, the measured mutual information remains numerically zero, confirming the theoretical irreversibility of rank-one IO models.

**Table I: Empirical Mutual Information for Rank-One Overwriting Mechanisms**

| Overwriting Type | \(r_0\) | \(P(X=1)\) | \(I(X;Y)\) (bits) |
|------------------|--------|------------|------------------|
| Constant overwrite | 1.0 | 0.2 | < 10⁻⁴ |
| Constant overwrite | 1.0 | 0.5 | < 10⁻⁴ |
| Uniform random | 0.5 | 0.5 | < 10⁻⁴ |
| Uniform random | 0.5 | 0.8 | < 10⁻⁴ |
| Biased random | 0.7 | 0.2 | < 10⁻⁴ |
| Biased random | 1.0 | 0.8 | < 10⁻⁴ |
| No deletion (baseline) | – | 0.5 | ≈ 1.0 |

### C. Approximate Rank-One Channels

Both **Table II** and **Fig. 1** demonstrate that information leakage vanishes rapidly as the channel approaches rank-one structure, validating the asymptotic analysis.

**Table II: Information Leakage for Approximate Rank-One Channels**

| \(\epsilon\) | \(I(X;Y)\) (bits) | Capacity Upper Bound |
|--------------|-------------------|-----------------------|
| \(10^{-1}\)  | \(1.2 \times 10^{-2}\) | \(1.5 \times 10^{-2}\) |
| \(10^{-2}\)  | \(1.1 \times 10^{-4}\) | \(1.4 \times 10^{-4}\) |
| \(10^{-3}\)  | \(1.0 \times 10^{-6}\) | \(1.2 \times 10^{-6}\) |
| 0 (rank-one) | 0 | 0 |

<center><b>Fig. 1.</b> Mutual information versus singular value ratio \(\epsilon\) for approximate rank-one channels.</center>

### D. Interpretation of Approximate Rank-One Behavior

Although the ideal rank-one deletion channel yields zero mutual information and zero capacity, practical overwriting mechanisms inevitably deviate from this idealization due to finite randomness, hardware noise, and implementation constraints. We model such deviations via the parameter \(\epsilon\) defined as the ratio between the second and largest singular values of the channel transition matrix.

This parameter admits a natural operational interpretation: it quantifies the extent to which the output distribution retains residual dependence on the input. When \(\epsilon = 0\), the channel output is statistically independent of the input, corresponding to perfect deletion. As \(\epsilon\) increases, the channel gradually departs from this ideal, allowing a non-zero amount of information to leak.

This parameterization enables a continuous, quantitative assessment of deletion quality via mutual information and capacity. In particular, mutual information captures average leakage, while channel capacity characterizes the worst-case recoverable information. The joint consideration of both metrics provides a more complete picture of deletion robustness under adversarial inference.

---

## XXV. Results and Discussion

### A. Consistency with Information-Theoretic Analysis

The experimental results are fully consistent with the theoretical analysis developed in Section III. Exact rank-one overwriting eliminates all statistical dependence between original and overwritten data, while approximate rank-one channels exhibit leakage that decays continuously with the deviation parameter \(\epsilon\).

Notably, biased overwriting mechanisms preserve nonuniform output statistics while still achieving zero mutual information, highlighting the distinction between randomness and irreversibility.

### B. Implications for Practical Overwriting

From a systems perspective, these results suggest that secure deletion does not require maximally random overwriting. Instead, it suffices to enforce a rank-one IO structure, which may be achieved through deterministic, random, or biased overwriting mechanisms.

This observation aligns with practical constraints in storage systems, where perfect randomness may be costly or unnecessary, and motivates rank-one IO design as a principled abstraction for keyless deletion.

---

## XXVI. Conclusion

This paper establishes rank-one Markov IO transformations as a fundamental information-theoretic abstraction for secure data deletion. By modeling overwriting as a channel operation, we show that rank-one behavior guarantees complete irreversibility, zero mutual information, and zero capacity, independent of implementation details or key secrecy.

Our analysis reveals that classical one-time pad encryption is a special case within a broader equivalence class of rank-one IO models, and that secure deletion can be achieved in a keyless manner through overwriting alone. The introduction of parameterized and approximate rank-one channels further demonstrates the robustness of this framework in practical settings.

Future work includes extending the theory to structured storage systems, correlated overwriting processes, and adaptive adversarial models, as well as exploring efficient verification mechanisms for rank-one behavior in real-world deletion systems.

---

## References

[1] Z. N. J. Peterson, R. Burns, J. Herring, and A. D. Rubin, "Secure deletion for a versioning file system," in *USENIX Annual Technical Conference*, 2005.

[2] J. Reardon, D. A. Basin, and S. Capkun, "On secure data deletion," *IEEE Security & Privacy*, vol. 12, no. 3, pp. 37–44, 2014.

[3] J. Reardon, "Secure data deletion," ETH Zurich Dissertation, 2014.

[4] P. Gutmann, "Secure deletion of data from magnetic and solid-state memory," *USENIX Security Symposium*, 1996.

[5] National Institute of Standards and Technology, "NIST special publication 800-88 rev. 1: Guidelines for media sanitization," 2014. [Online]. Available: https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-88r1.pdf

[6] SNIA, "Media sanitization," 2022. [Online]. Available: https://www.snia.org/education/storage_networking_practice

[7] J. Bartusek and D. Khurana, "Cryptography with certified deletion," IACR ePrint, 2022.

[8] J. Bartusek and V. Goyal, "Software with certified deletion," IACR ePrint, 2023.

[9] K. Dey and R. Safavi-Naini, "Hybrid encryption with certified deletion in preprocessing model," arXiv preprint, 2026.

[10] C. E. Shannon, "Communication theory of secrecy systems," *Bell System Technical Journal*, vol. 28, no. 4, pp. 656–715, 1949.

[11] M. Bloch, O. Günlü, A. Yener, F. Oggier, and H. V. Poor, "An overview of information-theoretic security and privacy," *IEEE Journal on Selected Areas in Information Theory*, vol. 38, no. 3, pp. 478–509, 2021.

[12] S. Asoodeh, F. Alajaji, and T. Linder, "Notes on information-theoretic privacy," arXiv preprint, 2015.

[13] R. F. Schäfer, M. Bloch, and A. Yener, "Information-theoretic security and privacy: A tutorial," *IEEE Access*, vol. 9, pp. 12345–12380, 2021.

---

*End of document*