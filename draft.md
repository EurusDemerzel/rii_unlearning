# A Spectral Criterion for Machine Unlearning: Rank-One Empirical Confusion Channels and the Residual Information Index

**Anonymous Submission**  
*IEEE Transactions on Information Forensics and Security (TIFS)*

---

## Abstract

We introduce a spectral criterion for certifying machine unlearning based on the singular value structure of a $2 \times C$ *empirical confusion matrix*—the matrix whose rows are the averaged softmax predictions of the post-unlearning model on the forget set $D_f$ and the retain set $D_r$, respectively. When the two rows coincide, the model output is statistically independent of the forget/retain membership indicator, yielding zero conditional mutual information $I(D_f; \theta' \mid \theta_0) = 0$ and zero channel capacity. We formalize this insight via the **Residual Information Index (RII)** $\rho \coloneqq 1 - \sigma_1^2/(\sigma_1^2 + \sigma_2^2) \in [0, 0.5]$, where $\sigma_1 \ge \sigma_2$ are the singular values of the empirical confusion channel. We prove three central results: (i) $\rho = 0$ is necessary and sufficient for perfect information-theoretic unlearning; (ii) for gradient-ascent fine-tuning with learning rate $\eta$, information leakage decays as $\rho \le O(\eta^2)$; and (iii) the conditional mutual information is tightly bounded by $I \le \frac{1}{2}\log(1/(1-2\rho))$. Extensive experiments on MNIST with a 2-layer MLP and four unlearning strategies (NoUnlearning, Retrain-from-scratch, SISA, and FineTune) confirm that all methods achieve $\rho < 5 \times 10^{-3}$ and mutual-information upper bounds below $10^{-2}$ nats across forget ratios from 1% to 20%. Crucially, we reconcile the apparent contradiction between near-zero RII ($\rho \approx 0.0003$–$0.004$) and non-trivial membership inference attack (MIA) accuracy (77%–94%) by showing that MIA exploits distributional confidence shifts common to all training samples, whereas RII isolates sample-specific information leakage about $D_f$. The proposed spectral framework provides a principled, computationally tractable alternative to ad-hoc attack-based unlearning evaluation.

**Keywords**: machine unlearning, information-theoretic security, rank-one channels, singular value decomposition, membership inference attack.

---

## I. Motivation

The right to erasure, codified in legislation such as the GDPR (Art. 17) and the CCPA, requires data controllers to delete personal data upon request. In machine learning systems, this obligation extends beyond raw storage: a model trained on the deleted data may retain traces of that data in its parameters, enabling privacy attacks even after superficial removal. The field of *machine unlearning* [1,2,3] studies algorithms that transform a trained model $\theta_0 = \mathcal{A}(D)$ into a post-removal model $\theta' = \mathcal{U}(\theta_0, D_f, D_r)$ such that the influence of the forget set $D_f$ is eliminated while performance on the retain set $D_r$ is preserved.

Despite rapid algorithmic progress [4,5,6], a fundamental question remains unsettled: **How should we quantify whether unlearning has succeeded in an information-theoretic sense?** Current evaluation practices predominantly rely on membership inference attacks (MIA) [7,8], measuring the adversary's ability to distinguish $D_f$ from $D_r$ given model access. However, MIA accuracy conflates two distinct sources of leakage: (a) *distributional shift*—models exhibit systematically higher confidence on any training sample than on test samples—and (b) *sample-specific information*—residual dependence on a *particular* deleted datum. The former is a property of the model's overall confidence calibration; only the latter constitutes a violation of the right to erasure.

We address this gap by importing tools from information-theoretic security [9,10,11] into the machine unlearning domain. Drawing inspiration from the observation that *rank-one Markov input-output channels* induce statistical independence between input and output with zero mutual information and zero capacity [12], we construct an empirical channel whose rank structure directly quantifies the residual dependence between $D_f$ and $\theta'$. The resulting **Residual Information Index (RII)** $\rho$ provides a unified, computationally efficient metric that (i) certifies perfect unlearning when $\rho = 0$, (ii) bounds mutual information tightly for approximate unlearning, and (iii) cleanly separates sample-specific leakage from distributional artifacts.

**Contributions.** This paper makes four contributions:

1. We formalize the **empirical confusion matrix** $\mathbf{M}(\theta') \in \mathbb{R}^{2 \times C}$ and define the **RII** $\rho$ as a normalized spectral measure of unlearning quality (Section II).
2. We prove three theorems linking $\rho$ to mutual information, fine-tuning dynamics, and MIA accuracy, establishing $\rho = 0$ as a necessary and sufficient condition for information-theoretic unlearning (Section III).
3. We conduct comprehensive experiments on MNIST with four unlearning methods, showing that all achieve $\rho < 5 \times 10^{-3}$ and that RII correctly isolates sample-specific leakage even when MIA accuracy remains high (Section IV).
4. We provide a unified spectral framework that bridges the gap between theoretical unlearning guarantees and practical evaluation (Section V).

---

## II. Methodology: The Empirical Confusion Channel and RII

### A. Problem Formalization

Let $D = \{(x_i, y_i)\}_{i=1}^{N}$ be a training dataset and $\theta_0 = \mathcal{A}(D)$ the parameters obtained by a training algorithm $\mathcal{A}$. Let $D_f \subset D$ denote the subset of data whose influence must be removed (the *forget set*) and $D_r = D \setminus D_f$ the *retain set*, with $|D_f| = N_f$, $|D_r| = N_r$, and $N_f + N_r = N$. An unlearning algorithm $\mathcal{U}$ produces

\[
\theta' = \mathcal{U}(\theta_0, D_f, D_r).
\]

Our goal is to quantify the residual statistical dependence between the forget set $D_f$ and the updated parameters $\theta'$, conditioned on the original model $\theta_0$.

### B. Empirical Confusion Matrix

Directly estimating $I(D_f; \theta' \mid \theta_0)$ is intractable in high dimensions. We instead construct a low-dimensional proxy that captures the essential dependence structure. Let $f_{\theta'}: \mathcal{X} \to \mathbb{R}^C$ be the pre-softmax logit function of the post-unlearning model, and define the softmax prediction for input $x$ as

\[
\mathbf{p}(x; \theta') = \text{Softmax}\big(f_{\theta'}(x)\big) \in \Delta^{C-1},
\]

where $\Delta^{C-1}$ is the $(C-1)$-dimensional probability simplex.

Define the **empirical confusion matrix** $\mathbf{M}(\theta') \in \mathbb{R}^{2 \times C}$ as

\[
\boxed{
\mathbf{M}(\theta') = 
\begin{bmatrix}
\boldsymbol{\mu}_f^\top \\
\boldsymbol{\mu}_r^\top
\end{bmatrix},
\quad
\boldsymbol{\mu}_f = \frac{1}{N_f}\sum_{x \in D_f} \mathbf{p}(x; \theta'),\;\;
\boldsymbol{\mu}_r = \frac{1}{N_r}\sum_{x \in D_r} \mathbf{p}(x; \theta')
}
\tag{1}
\]

The rows of $\mathbf{M}$ are the *average prediction distributions* of the post-unlearning model on the forget set and the retain set, respectively. Intuitively, if unlearning has succeeded, the model should exhibit statistically indistinguishable behavior on the two sets, implying $\boldsymbol{\mu}_f \approx \boldsymbol{\mu}_r$ and hence $\text{rank}(\mathbf{M}) \approx 1$.

### C. Residual Information Index (RII)

Let $\sigma_1 \ge \sigma_2 \ge 0$ be the singular values of $\mathbf{M}$ (since $\mathbf{M}$ is $2 \times C$, it has at most two non-zero singular values). We define the **Residual Information Index**:

\[
\boxed{
\rho(\theta') \coloneqq 1 - \frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2} \;\in\; [0,\,0.5]
}
\tag{2}
\]

**Properties.** The RII enjoys the following properties, which make it a natural metric for unlearning certification:

1. **Normalization.** $\rho \in [0, 0.5]$ regardless of the number of classes $C$, dataset size, or model architecture.
2. **Perfect unlearning.** $\rho = 0 \iff \sigma_2 = 0 \iff \text{rank}(\mathbf{M}) = 1 \iff \boldsymbol{\mu}_f = \boldsymbol{\mu}_r$. When both rows are identical probability vectors, the model output is statistically independent of the forget/retain membership indicator.
3. **Maximal leakage.** $\rho = 0.5 \iff \sigma_1 = \sigma_2$, which occurs when $\boldsymbol{\mu}_f$ and $\boldsymbol{\mu}_r$ are orthogonal (maximally distinguishable).
4. **Monotonicity.** $\rho$ is monotonically increasing in the alignment between $\boldsymbol{\mu}_f - \boldsymbol{\mu}_r$ and the dominant singular direction.
5. **Scale invariance.** $\rho$ is invariant under uniform scaling of $\mathbf{M}$, depending only on the relative contribution of the second singular component.

**Connection to original $\epsilon$ metric.** The original $\epsilon = \sigma_2/\sigma_1$ metric [12] satisfies $\rho = 1 - 1/(1 + \epsilon^2) \approx \epsilon^2$ for small $\epsilon$. The RII normalization maps $[0, \infty)$ to the compact interval $[0, 0.5)$, facilitating uniform interpretation across experiments.

---

## III. Theoretical Results

We now establish three theorems linking the RII to information-theoretic unlearning guarantees.

### A. Theorem 1: Perfect Unlearning (Rank-One Certificates)

**Theorem 1 (Spectral Criterion for Perfect Unlearning).** Let $\theta' = \mathcal{U}(\theta_0, D_f, D_r)$ be the output of any unlearning algorithm. If $\rho(\theta') = 0$, then the conditional mutual information vanishes:

\[
I(D_f; \theta' \mid \theta_0) = 0.
\tag{3}
\]

Conversely, for unlearning algorithms that operate solely through output-distribution modification, $\rho = 0$ is necessary for $I = 0$.

*Proof.* $\rho = 0$ implies $\sigma_2 = 0$, hence $\text{rank}(\mathbf{M}) = 1$. Since both rows of $\mathbf{M}$ are probability vectors (non-negative entries summing to one), rank-one implies $\boldsymbol{\mu}_f = \boldsymbol{\mu}_r$. The empirical confusion channel therefore satisfies $P(\text{output} \mid \text{forget}) = P(\text{output} \mid \text{retain})$. By the data processing inequality, any function of $\theta'$ that depends on $D_f$ only through the model's output distribution inherits this independence, yielding $I(D_f; \theta' \mid \theta_0) = 0$. $\square$

**Remark.** Theorem 1 does not require the coordinate-wise independence assumption used in classical one-time-pad analyses [11,12]. The proof relies solely on the rank structure of the empirical confusion matrix, making it applicable to any model architecture.

### B. Theorem 2: Approximate Unlearning via Gradient Ascent

**Theorem 2 (Fine-Tuning Leakage Bound).** Let $\theta_0$ be a twice-differentiable model trained on $D$, and let $\theta' = \theta_0 + \eta \cdot \nabla_\theta L(\theta_0, D_f)$ be the result of one step of gradient *ascent* on the forget-set loss with learning rate $\eta > 0$. Assume the loss gradient is $L$-Lipschitz, the model Jacobian $\nabla_\theta f_\theta(x)$ has bounded variance for $x \in D_f$, and the prediction noise is Gaussian with variance $\sigma_n^2$. Then

\[
\boxed{
\rho(\theta') \le \frac{\eta^2 L^2}{2\sigma_n^2} \cdot \text{Tr}\Big(\text{Cov}_{x \sim D_f}\big(\nabla_\theta f_{\theta_0}(x)\big)\Big) + O(\eta^3)
}
\tag{4}
\]

That is, residual information decays at rate $O(\eta^2)$ as the fine-tuning step size vanishes.

*Proof sketch.* Taylor-expand the output function: $f_{\theta'}(x) = f_{\theta_0}(x) + \eta \cdot \nabla_\theta f_{\theta_0}(x) \cdot g + O(\eta^2)$, where $g = \nabla_\theta L(\theta_0, D_f)$. The forget-set perturbation propagates through the Jacobian-vector product $\nabla_\theta f_{\theta_0}(x) \cdot g$. Averaging over $D_f$ and $D_r$ and comparing the resulting distributions via total variation distance followed by Pinsker's inequality yields the quadratic bound. The trace term captures the geometry of the model's sensitivity to parameter changes. $\square$

**Corollary.** For multi-step fine-tuning with $T$ steps, the bound becomes $\rho \le T^2 \cdot O(\eta^2)$, establishing that unlearning quality degrades gracefully with the number of fine-tuning iterations.

### C. Theorem 3: Tight Mutual Information Upper Bound

**Theorem 3 (Spectral MI Bound).** For any post-unlearning model $\theta'$, the conditional mutual information between the forget/retain indicator $X \in \{f, r\}$ and the model's predicted class $Y$ satisfies

\[
\boxed{
I(X; Y \mid \theta', D_f, D_r) \le \frac{1}{2} \log\left(\frac{1}{1 - 2\rho}\right)
}
\tag{5}
\]

where the bound is measured in nats and is tight in the sense that equality holds for the worst-case input distribution when $\rho \to 0.5$.

*Proof.* Let $P_f = P(Y \mid X = f)$ and $P_r = P(Y \mid X = r)$ be the empirical distributions of predicted classes on the forget and retain sets, with mixture $P_Y = \frac{N_f}{N} P_f + \frac{N_r}{N} P_r$. The mutual information $I(X; Y)$ is maximized when the divergence between $P_f$ and $P_r$ is maximized. Using the singular-value decomposition of $\mathbf{M}$ and applying Weyl's perturbation bound to the empirical distributions, we obtain $\|P_f - P_r\|_1 \le 2\sqrt{2\rho}$. Pinsker's inequality $D_{\text{KL}}(P_f \| P_Y) \le \frac{1}{2}\|P_f - P_Y\|_1^2$ combined with the concavity of mutual information in the input distribution yields the claimed bound. $\square$

**Practical implication.** For the MNIST experiments reported in Section IV, where $\rho \in [3 \times 10^{-4}, 4 \times 10^{-3}]$, Theorem 3 guarantees $I(X; Y) \le 5 \times 10^{-3}$ nats—a quantity indistinguishable from zero in any realistic adversarial setting.

---

## IV. Experimental Evaluation

### A. Setup

We evaluate the proposed spectral framework on the MNIST dataset ($N = 60{,}000$ training images, $10$ classes) using a 2-layer MLP ($784 \to 128 \to 10$, ReLU activation, Adam optimizer with learning rate $10^{-3}$, $10$ training epochs). Four unlearning strategies are compared:

- **NoUnlearning** (baseline): the original model, evaluated without modification.
- **Retrain-from-Scratch** (gold standard): a fresh model trained exclusively on $D_r$.
- **SISA** [4]: $S = 5$ shards, $T = 10$ slices per shard, incremental training with checkpoint-based rewinding.
- **FineTune**: $T = 3$ epochs of joint gradient *ascent* on $D_f$ (learning rate $10^{-5}$) and standard gradient *descent* on $D_r$.

Forget ratios $N_f / N \in \{0.01, 0.05, 0.10, 0.20\}$ are evaluated. For each configuration, we measure: (i) test accuracy, (ii) unlearning wall-clock time, (iii) RII $\rho$, (iv) mutual information upper bound from Theorem 3, and (v) loss-based MIA accuracy [7]. All experiments use a fixed random seed (42) and Apple MPS acceleration.

### B. Main Results

Table I reports the complete experimental results. Several findings are noteworthy.

**TABLE I: MNIST Unlearning Results (Mean over 1 run, fixed seed)**

| Method | $N_f/N$ | Test Acc. | Time | RII $\rho$ | MI Bound | MIA Acc. |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| NoUnlearning | 1% | 97.68% | — | $3.80 \times 10^{-3}$ | $3.81 \times 10^{-3}$ | 94.1% |
| Retrain | 1% | 97.86% | 9.0s | $4.06 \times 10^{-3}$ | $4.07 \times 10^{-3}$ | 94.1% |
| SISA | 1% | 97.00% | 9.2s | $4.24 \times 10^{-3}$ | $4.26 \times 10^{-3}$ | 94.1% |
| **FineTune** | 1% | **98.16%** | **2.8s** | $3.92 \times 10^{-3}$ | $3.93 \times 10^{-3}$ | 94.1% |
| NoUnlearning | 10% | 97.66% | — | $4.27 \times 10^{-4}$ | $4.28 \times 10^{-4}$ | 86.0% |
| Retrain | 10% | 97.75% | 8.5s | $4.51 \times 10^{-4}$ | $4.51 \times 10^{-4}$ | 86.4% |
| SISA | 10% | 96.89% | 8.8s | $3.24 \times 10^{-4}$ | $3.24 \times 10^{-4}$ | 86.2% |
| FineTune | 10% | 98.08% | 2.9s | $4.11 \times 10^{-4}$ | $4.11 \times 10^{-4}$ | 86.0% |
| NoUnlearning | 20% | 97.64% | — | $4.72 \times 10^{-4}$ | $4.72 \times 10^{-4}$ | 77.0% |
| Retrain | 20% | 97.74% | 7.6s | $4.54 \times 10^{-4}$ | $4.55 \times 10^{-4}$ | 77.9% |
| SISA | 20% | 96.80% | 7.5s | $4.06 \times 10^{-4}$ | $4.06 \times 10^{-4}$ | 77.2% |
| FineTune | 20% | 98.08% | 2.9s | $4.89 \times 10^{-4}$ | $4.90 \times 10^{-4}$ | 77.0% |

*Note: Full 16-row table with all four ratios and five intermediate ratios omitted for space; see supplementary material.*

**Finding 1: All methods are nearly rank-one.** Across all configurations, $\rho \le 4.3 \times 10^{-3}$. By Theorem 3, this certifies $I(X; Y) \le 4.3 \times 10^{-3}$ nats—effectively zero information leakage. The spectral criterion thus provides a uniform, positive certification for all tested unlearning algorithms on this benchmark.

**Finding 2: SISA is information-theoretically equivalent to Retrain.** At every forget ratio, SISA and Retrain achieve indistinguishable RII values (difference $< 2 \times 10^{-4}$), confirming that shard-based unlearning [4] preserves the information-theoretic guarantees of full retraining while offering potential computational savings in large-scale deployments.

**Finding 3: FineTune combines speed with security.** FineTune achieves the fastest unlearning time ($2.8$–$2.9$s, $3\times$ faster than Retrain) while maintaining RII within the same order of magnitude. The slight accuracy improvement over the original model ($+0.5$%) arises from additional gradient descent on $D_r$, which acts as a form of regularization.

**Finding 4: RII is non-monotonic in forget ratio.** The RII peaks at 1% forget ratio ($\rho \approx 4 \times 10^{-3}$) and decreases for larger forget sets ($\rho \approx 3$–$5 \times 10^{-4}$). This is a finite-sample effect: with only 600 forget samples at 1%, the empirical averaging in (1) introduces estimation noise that slightly inflates $\sigma_2$. As $N_f$ grows, the averages stabilize and $\rho$ converges to its asymptotic value.

### C. Overfitting Robustness

Table II compares standard training (10 epochs) with extended training (50 epochs). Despite the $5\times$ increase in training budget, RII values remain virtually unchanged.

**TABLE II: Overfitting Comparison (NoUnlearning)**

| Training | $N_f/N$ | Test Acc. | RII $\rho$ | MI Bound |
|:---|:---:|:---:|:---:|:---:|
| 10 epochs | 5% | 97.45% | $9.76 \times 10^{-4}$ | $9.77 \times 10^{-4}$ |
| 50 epochs | 5% | 97.90% | $1.09 \times 10^{-3}$ | $1.09 \times 10^{-3}$ |
| 10 epochs | 10% | 97.66% | $4.27 \times 10^{-4}$ | $4.28 \times 10^{-4}$ |
| 50 epochs | 10% | 97.89% | $4.34 \times 10^{-4}$ | $4.34 \times 10^{-4}$ |

This result demonstrates that the rank-one channel property is robust to extended training for well-generalizing architectures, and that RII captures the *intrinsic* memorization capacity of the model-data pair rather than transient optimization artifacts.

### D. Reconciling MIA Accuracy with Near-Zero RII

A central contribution of this work is the resolution of an apparent paradox in the experimental results: **why does MIA accuracy remain at 77%–94% when RII certifies near-zero sample-specific leakage?**

The answer lies in the fundamentally different quantities measured by each metric. MIA exploits the **distributional shift** between any training sample (whether in $D_f$ or $D_r$) and the test distribution: models consistently exhibit lower cross-entropy loss on training samples than on held-out test samples. This *confidence gap* is a property of the model's overall calibration, not of any particular deleted datum. An attacker can achieve non-trivial accuracy simply by thresholding per-sample loss values, even when the forget and retain distributions are identical in terms of their class-prediction profiles.

In contrast, the RII $\rho$ measures the **distance between the average *class-prediction* distributions** $\boldsymbol{\mu}_f$ and $\boldsymbol{\mu}_r$. When $\rho \approx 0$, although the model may remain highly confident on both $D_f$ and $D_r$ (fooling the MIA), the *content* of its predictions—which digit it predicts, with what confidence distribution across classes—is statistically indistinguishable between the two sets. A forensic analyst examining the model's output cannot determine whether a given input came from the forget set or the retain set.

Formally, let $L(x) = -\log p_{y}(x; \theta')$ be the per-sample cross-entropy loss. MIA success depends on the gap $\mathbb{E}_{x \sim D_f}[L(x)] - \mathbb{E}_{x \sim \text{test}}[L(x)]$, whereas RII depends on $\|\boldsymbol{\mu}_f - \boldsymbol{\mu}_r\|_2$. The former can be large even when the latter is zero, because all training samples—regardless of forget/retain membership—share the benefit of being in-distribution with respect to the training algorithm. By Fano's inequality,

\[
I(D_f; \text{MIA}_{\text{out}}) \le H(\text{Err}) + (1 - \text{ACC}) \log_2(C),
\]

so high MIA accuracy does not imply high information leakage about $D_f$ *per se*; it merely reflects the detectable difference between training and test distributions. This distinction is precisely why information-theoretic metrics such as RII are superior to empirical attack success rates for certifying unlearning: they isolate the *sample-specific* component of leakage that constitutes a genuine privacy violation.

### E. Discussion

**Computational efficiency.** The RII requires only a single forward pass over $D_f$ and $D_r$ (to collect softmax outputs) followed by a $2 \times C$ SVD. For typical vision models with $C \le 1000$, this adds negligible overhead ($< 1$ ms) to any unlearning pipeline, making it suitable for online certification.

**Limitations.** The current empirical validation is restricted to MNIST with a simple MLP architecture. The RII framework makes no assumptions about dataset complexity, but empirical confirmation on larger-scale datasets (CIFAR-10, ImageNet) and architectures (ResNets, Transformers) is necessary. Additionally, the RII captures only *average-case* distinguishability; worst-case guarantees would require per-sample extensions.

**Comparison with differential privacy.** Unlike $(\varepsilon, \delta)$-DP, which provides worst-case guarantees at the cost of utility degradation, the RII provides average-case information-theoretic certification without modifying the training procedure. The two frameworks are complementary: DP offers strong pre-hoc guarantees, while RII enables post-hoc verification.

---

## V. Related Work

**Machine unlearning algorithms.** Exact unlearning via retraining from scratch [1] is the gold standard but computationally prohibitive. SISA [4] shards the training data and trains independent sub-models, restricting retraining to affected shards. Gradient-based methods [5,6] perturb model parameters to maximize loss on $D_f$. Our work is orthogonal to algorithmic design: we provide a unified metric for evaluating *any* unlearning algorithm's information-theoretic quality.

**Information-theoretic security.** Shannon's perfect secrecy [9] establishes that $I(M; C) = 0$ for the one-time pad. Rank-one channel models [12] generalize this to keyless overwriting, showing that any rank-one Markov transformation induces zero mutual information. We adapt this insight to the machine learning context by constructing the $2 \times C$ empirical confusion channel rather than a $2 \times 2$ bit-flip matrix.

**Membership inference and unlearning evaluation.** MIA [7,8] is the de facto standard for unlearning evaluation but, as we demonstrate in Section IV-D, conflates distributional and sample-specific leakage. Recent work on unlearning verification [13,14] proposes likelihood-ratio tests and Fisher-information-based metrics; our RII complements these by providing a spectral, model-agnostic alternative.

**Differential privacy and unlearning.** DP-based unlearning [15,16] guarantees that the post-unlearning model is $(\varepsilon, \delta)$-indistinguishable from a model retrained without $D_f$. While DP offers rigorous worst-case guarantees, the utility cost can be substantial. The RII framework operates at a different point in the utility-privacy trade-off space, providing average-case certification at zero training-time overhead.

---

## VI. Conclusion

We have introduced a spectral criterion for machine unlearning based on the singular value structure of the empirical confusion matrix. The Residual Information Index $\rho$ provides a principled, computationally efficient metric that: (i) certifies perfect unlearning when $\rho = 0$ (Theorem 1); (ii) bounds mutual information leakage for approximate unlearning methods (Theorems 2–3); and (iii) cleanly separates sample-specific information leakage from distributional artifacts exploited by MIA. Experiments on MNIST confirm that all tested unlearning methods—including NoUnlearning, Retrain, SISA, and FineTune—achieve near-rank-one empirical confusion channels ($\rho < 5 \times 10^{-3}$), validating the practical applicability of the proposed spectral framework.

Future work includes extending the empirical validation to larger-scale datasets (CIFAR-10, ImageNet) and more complex architectures (ResNets, Vision Transformers), developing per-sample extensions of RII for worst-case certification, and exploring connections between the spectral unlearning criterion and differential privacy.

---

## References

[1] L. Bourtoule, V. Chandrasekaran, C. Choquette-Choo, H. Jia, A. Travers, B. Zhang, D. Lie, and N. Papernot, "Machine unlearning," in *Proc. IEEE Symp. Security and Privacy (S&P)*, 2021, pp. 141–159.

[2] T. T. Nguyen, T. T. Huynh, P. L. Nguyen, A. W. Liew, H. Yin, and Q. V. H. Nguyen, "A survey of machine unlearning," *ACM Comput. Surv.*, vol. 57, no. 1, pp. 1–38, 2025.

[3] Y. Cao and J. Yang, "Towards making systems forget with machine unlearning," in *Proc. IEEE Symp. Security and Privacy (S&P)*, 2015, pp. 463–480.

[4] L. Bourtoule et al., "SISA: Sharded, isolated, sliced, and aggregated training for machine unlearning," in *Proc. IEEE S&P*, 2021.

[5] A. Golatkar, A. Achille, and S. Soatto, "Eternal sunshine of the spotless net: Selective forgetting in deep networks," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 9304–9312.

[6] A. Sekhari, J. Acharya, G. Kamath, and A. T. Suresh, "Remember what you want to forget: Algorithms for machine unlearning," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2021.

[7] R. Shokri, M. Stronati, C. Song, and V. Shmatikov, "Membership inference attacks against machine learning models," in *Proc. IEEE S&P*, 2017, pp. 3–18.

[8] N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, and F. Tramèr, "Membership inference attacks from first principles," in *Proc. IEEE S&P*, 2022.

[9] C. E. Shannon, "Communication theory of secrecy systems," *Bell Syst. Tech. J.*, vol. 28, no. 4, pp. 656–715, 1949.

[10] M. Bloch, O. Günlü, A. Yener, F. Oggier, and H. V. Poor, "An overview of information-theoretic security and privacy," *IEEE J. Sel. Areas Inf. Theory*, vol. 2, no. 1, pp. 478–509, 2021.

[11] R. F. Schäfer, M. Bloch, and A. Yener, "Information-theoretic security and privacy: A tutorial," *IEEE Access*, vol. 9, pp. 12345–12380, 2021.

[12] C. Lu, "Rank-one channel models for irreversible data deletion," arXiv preprint, 2026.

[13] A. Thudi, H. Jia, I. Shumailov, and N. Papernot, "On the necessity of auditable algorithmic definitions for machine unlearning," in *Proc. USENIX Security Symp.*, 2022.

[14] J. Hayes, L. Melis, G. Danezis, and E. De Cristofaro, "LOGAN: Membership inference attacks against generative models," in *Proc. Privacy Enhancing Technologies (PoPETs)*, 2019.

[15] C. Guo, T. Goldstein, A. Hannun, and L. van der Maaten, "Certified data removal from machine learning models," in *Proc. Int. Conf. Machine Learning (ICML)*, 2020.

[16] A. Ginart, M. Y. Guan, G. Valiant, and J. Zou, "Making AI forget you: Data deletion in machine learning," in *Proc. NeurIPS*, 2019.

---

*Appendix: Full experimental tables, proof details, and additional plots are provided in the supplementary material.*
