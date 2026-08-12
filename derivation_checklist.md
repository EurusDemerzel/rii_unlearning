# 推导辅助清单

## 符号速查

| 符号 | 含义 | 维度 |
|------|------|:----:|
| $\mathbf{M}$ | 经验混淆矩阵 | $2\times C$ |
| $\boldsymbol{\mu}_f, \boldsymbol{\mu}_r$ | 遗忘集/保留集平均 softmax 向量 | $C\times 1$ |
| $\sigma_1 \ge \sigma_2$ | $\mathbf{M}$ 的奇异值 | 标量 |
| $\rho = \sigma_2^2/(\sigma_1^2+\sigma_2^2)$ | RII | $[0,0.5]$ |
| $\mathbf{H}$ | $K$ 个未见类均值堆叠的矩阵 | $K\times C$ |
| $\rho_H$ | MHPR | $[0,1]$ |

---

## 第一阶段：基础框架

### Step 1: 验证 $\mathbf{M}$ 的 SVD

$$
\mathbf{M} = \begin{bmatrix}\boldsymbol{\mu}_f^\top \\ \boldsymbol{\mu}_r^\top\end{bmatrix}
\quad\Rightarrow\quad
\mathbf{M}\mathbf{M}^\top = \begin{bmatrix}
\|\boldsymbol{\mu}_f\|^2 & \boldsymbol{\mu}_f^\top\boldsymbol{\mu}_r \\
\boldsymbol{\mu}_f^\top\boldsymbol{\mu}_r & \|\boldsymbol{\mu}_r\|^2
\end{bmatrix}
$$

**检查点** □ 计算 $\mathbf{M}\mathbf{M}^\top$ 的特征值：
$$
\lambda_{1,2} = \frac{\|\boldsymbol{\mu}_f\|^2 + \|\boldsymbol{\mu}_r\|^2 \pm \sqrt{(\|\boldsymbol{\mu}_f\|^2 - \|\boldsymbol{\mu}_r\|^2)^2 + 4(\boldsymbol{\mu}_f^\top\boldsymbol{\mu}_r)^2}}{2}
$$

**检查点** □ 当 $\|\boldsymbol{\mu}_f\| = \|\boldsymbol{\mu}_r\|$ 时简化：
$$
\sigma_1^2 = \|\boldsymbol{\mu}\|^2 + \boldsymbol{\mu}_f^\top\boldsymbol{\mu}_r,\quad
\sigma_2^2 = \|\boldsymbol{\mu}\|^2 - \boldsymbol{\mu}_f^\top\boldsymbol{\mu}_r
$$

**检查点** □ 验证 $\|\boldsymbol{\mu}_f - \boldsymbol{\mu}_r\|_2 = \sqrt{2}\,\sigma_2$

---

### Step 2: RII 的基本性质

$$ \rho = \frac{\sigma_2^2}{\sigma_1^2+\sigma_2^2} $$

**检查点** □ $\rho=0 \iff \sigma_2=0 \iff \boldsymbol{\mu}_f=\boldsymbol{\mu}_r$
- $(\Rightarrow)$: $\rho=0 \Rightarrow \sigma_2=0 \Rightarrow \text{rank}(\mathbf{M})=1$，两行成比例。又每行和为 $1$，比例系数必为 $1$，故 $\boldsymbol{\mu}_f=\boldsymbol{\mu}_r$ ✅
- $(\Leftarrow)$: 显然

**检查点** □ $\rho \le 0.5$，等号当 $\sigma_1=\sigma_2$

---

## 第二阶段：Theorem 1 — 完美遗忘判据

### 要证明的核心链条

$$
\rho=0 \iff \boldsymbol{\mu}_f=\boldsymbol{\mu}_r \iff I(X;Y|\theta')=0
$$

**检查点** □ **$\rho=0 \Rightarrow I=0$**：
1. $\rho=0 \Rightarrow \boldsymbol{\mu}_f=\boldsymbol{\mu}_r$（已证）
2. Softmax 是 categorical 分布，$\boldsymbol{\mu} = \mathbf{p}(x;\theta')$ 是均值参数
3. 指数族中均值参数与自然参数一一对应（softmax 是双射）
4. 因此 $\boldsymbol{\mu}_f=\boldsymbol{\mu}_r$ 意味着 $P(Y|X=1) = P(Y|X=0)$
5. 即 $Y \perp X$，故 $I=0$ ✅

**检查点** □ **$I=0 \Rightarrow \rho=0$**：
1. $I=0 \Rightarrow P(Y|X=1)=P(Y|X=0)$
2. $\Rightarrow \mathbb{E}[Y|X=1]=\mathbb{E}[Y|X=0] \Rightarrow \boldsymbol{\mu}_f=\boldsymbol{\mu}_r$
3. $\Rightarrow \sigma_2=0 \Rightarrow \rho=0$ ✅

**⚠️ 这里有个微妙处**：$I=0$ 均值相等，但反过来均值相等并不总是推出 $I=0$（不同分布可以有相同均值）。你的论证使用了指数族性质——softmax 是**最小指数族**，均值参数是充分统计量，均值相等 ⟹ 分布相等。这一步需要自己确认理解。

---

## 第三阶段：Proposition 1 — 置信界

### 推导链

$$\hat{\mathbf{M}} \xrightarrow{\text{Hoeffding}} \|\hat{\mathbf{M}}-\mathbf{M}\|_F \le O(\sqrt{C/N_{\min}}) \xrightarrow{\text{Weyl}} |\hat{\sigma}_i-\sigma_i| \le O(\sqrt{C/N_{\min}}) \xrightarrow{\text{Delta}} |\hat{\rho}-\rho^*| \le O(1/\sigma_1^2 \cdot \sqrt{C/N_{\min}})$$

**检查点** □ **Hoeffding 步骤**：
- 每个 $\hat{M}_{ij}$ 是 $N_f$ 或 $N_r$ 个 sub-Gaussian($\nu^2$) 变量的均值
- $\mathbb{P}(|\hat{M}_{ij}-M_{ij}| \ge t) \le 2\exp(-N_{\min}t^2/2\nu^2)$
- Union bound over $2C$ 个 entry：$\|\hat{\mathbf{M}}-\mathbf{M}\|_F \le 2\nu\sqrt{2C\log(4C/\delta)/N_{\min}}$

**检查点** □ **Weyl 步骤**：
- $|\hat{\sigma}_i - \sigma_i| \le \|\hat{\mathbf{M}} - \mathbf{M}\|_2 \le \|\hat{\mathbf{M}} - \mathbf{M}\|_F$ ✅

**检查点** □ **Delta 方法**：
- $\rho = f(\sigma_1,\sigma_2) = 1 - \sigma_1^2/(\sigma_1^2+\sigma_2^2)$
- $\frac{\partial\rho}{\partial\sigma_1} = -\frac{2\sigma_1\sigma_2^2}{(\sigma_1^2+\sigma_2^2)^2},\quad \frac{\partial\rho}{\partial\sigma_2} = \frac{2\sigma_1^2\sigma_2}{(\sigma_1^2+\sigma_2^2)^2}$
- $\|\nabla f\|_2 \approx 2\sigma_2/\sigma_1^2$ 当 $\rho\ll 1$
- $|\hat{\rho}-\rho^*| \le \|\nabla f\|_2 \cdot \|\hat{\mathbf{M}}-\mathbf{M}\|_F$

---

## 第四阶段：Theorem 2 — 梯度上升泄漏界

### 假设
- 单步梯度上升 $\theta' = \theta_0 + \eta\nabla_\theta\mathcal{L}(\theta_0, D_f)$
- $\mathcal{L}$ 有 $L$-Lipschitz 梯度
- **sample-level 遗忘**：$D_f$ 随机抽样 ⇒ 梯度近似不相关

### 关键步骤

**检查点** □ 理解这里需要你手动推导的核心近似：
$$
\Delta f(x) = f_{\theta'}(x) - f_{\theta_0}(x) \approx \eta\langle \nabla_\theta f_{\theta_0}(x), \nabla_\theta\mathcal{L}(\theta_0, D_f)\rangle
$$

**检查点** □ 然后 $\Delta\boldsymbol{\mu} = \frac{1}{N_f}\sum_{x\in D_f} \mathbf{J}_{\mathrm{softmax}}(f(x)) \cdot \Delta f(x)$

**检查点** □ 验证 $\|\Delta\boldsymbol{\mu}\|_2^2 \le O(\eta^2)\cdot\mathrm{Tr}(\mathrm{Cov}(\nabla f))$

---

## 第五阶段：Theorem 3 — $\chi^2$-MI 界

### 核心推导

$$
\|\boldsymbol{\mu}_f - \boldsymbol{\mu}_r\|_1 \le \sqrt{C}\|\boldsymbol{\mu}_f - \boldsymbol{\mu}_r\|_2 = \sqrt{2C}\sigma_2
$$

**检查点** □ $\chi^2(P_f\|P_r) = \sum_y \frac{(P_f(y)-P_r(y))^2}{P_r(y)} = \frac{\|\boldsymbol{\mu}_f - \boldsymbol{\mu}_r\|_2^2}{\delta_{\mathrm{eff}}}$

**检查点** □ $\delta_{\mathrm{eff}}^{-1} = \frac{\|\boldsymbol{\mu}_f-\boldsymbol{\mu}_r\|_2^2}{\sum_y (P_f-P_r)^2/P_r}$ 可直接从数据计算（无需 bound）

**检查点** □ $D_{\mathrm{KL}}(P_f\|P_r) \le \log(1+\chi^2) \le \chi^2$
- 这里的 $\log(1+x) \le x$ 对 $x > -1$ 成立 ✅

**检查点** □ 二元输入信道：$I(X;Y) \le \max\{D_{\mathrm{KL}}(P_f\|P_r), D_{\mathrm{KL}}(P_r\|P_f)\}$

---

## 第六阶段：Theorem 4 — 统一泄漏分解

### 信号模型

$$
R = \alpha X + \beta Y + N,\quad N\sim\mathcal{N}(0,\sigma^2)
$$

### 分解推导

**检查点** □ **Chain rule**:
$$
I(X;R) \le I(X;R,Y) = I(X;Y) + I(X;R|Y)
$$

**检查点** □ **Cover 项** $I(X;Y)$:
- Hilbert-Schmidt 分解 $f_{Y|X} = f_Y + \sum_{k\ge 2}\sigma_k \phi_k \psi_k$
- $I(X;Y) \le C_2\sum_{k\ge 2}\sigma_k^2 = C_2\eta^2$，其中 $C_2 = \log C$

**检查点** □ **Direct 项** $I(X;R|Y)$:
- 给定 $Y$，$R = \alpha X + \beta y + N$ → 剩下只有 $\alpha X + N$
- Lemma 1 (Fisher 展开): $I(X;\alpha X+N) = \frac{\alpha^2}{2\sigma^2}\mathrm{Var}(X) + o(\alpha^2)$
- 这里 $J(N) = 1/\sigma^2$

---

## 第七阶段：MHPR 理论

### 投影操作

$$
\mu_f \approx \sum_{k=1}^K \alpha_k \boldsymbol{\mu}_{h_k} = \mathbf{H}^\top \alpha
$$

求解 $\alpha = (\mathbf{H}\mathbf{H}^\top)^{-1}\mathbf{H}\boldsymbol{\mu}_f$（最小二乘）

**检查点** □ 验证：$\hat{\boldsymbol{\mu}}_f = \mathbf{H}^\top(\mathbf{H}\mathbf{H}^\top)^{-1}\mathbf{H}\boldsymbol{\mu}_f$ 是 $\boldsymbol{\mu}_f$ 在 $\mathrm{rowspan}(\mathbf{H})$ 上的正交投影

### 三个命题

**检查点** □ **Proposition 2 (单调性)**: $\rho_H(K_2) \le \rho_H(K_1)$ 对 $K_2 \ge K_1$
- 证明：$\mathcal{S}_{K_1} \subseteq \mathcal{S}_{K_2}$ ⇒ 投影残差单调不增 ✅

**检查点** □ **Proposition 3 (偏差-方差)**: $\mathbb{E}[\rho_H] = O(C/N_{\min})$ 在 $H_0$ 下
- $H_0$: $\boldsymbol{\mu}_f^* \in \mathcal{S}_K^*$（完美遗忘）
- 两个估计误差来源：$\boldsymbol{\mu}_f$ 的估计 + 子空间估计

**检查点** □ **Proposition 4 (置信界)**:
$$
|\hat{\rho}_H - \rho_H^*| \le \frac{8\nu\sqrt{2}}{\|\boldsymbol{\mu}_f^*\|_2^2} \sqrt{\frac{C(K+1)\log((K+1)C/\delta)}{N_{\min}}}
$$

主要依赖 Lipschitz 常数 $L = 8/\|\boldsymbol{\mu}_f^*\|_2^2$

---

## 第八阶段：温度缩放

### 条件数自适应准则

**检查点** □ 计算 $\kappa(\mathbf{H}(T)) = \sigma_{\max}(\mathbf{H}(T)) / \sigma_{\min}(\mathbf{H}(T))$

**检查点** □ 选最小 $T$ 使 $\kappa(\mathbf{H}(T)) < 10$

---

## 推荐手推路线

```
Step 1: 写一遍 M 的 SVD → 验证 Gram 矩阵 → 求 σ₁, σ₂
Step 2: 证明 ρ=0 ⟺ μ_f=μ_r ⟺ I=0
Step 3: 手算 ρ 的梯度 ∂ρ/∂σ₁, ∂ρ/∂σ₂
Step 4: 写出 Hoeffding → Weyl → Delta 的完整链条
Step 5: 验证 η² 界的泰勒展开一阶项
Step 6: 写出 χ² 到 KL 到 MI 的放缩链
Step 7: 验证 chain rule 分解 I(X;R) ≤ I(X;Y) + I(X;R|Y)
Step 8: 写出 MHPR 的投影矩阵并验证正交性
Step 9: 计算温度缩放后的 H(T) 条件数
```

每个检查点过了就打勾 □，遇到卡住的地方告诉我，我帮你拆解。
