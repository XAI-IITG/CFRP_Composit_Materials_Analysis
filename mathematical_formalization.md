# Mathematical Formalization of the KG-XAI Pipeline

## Table of Contents

1. [Input & Preprocessing](#1-input--preprocessing)
2. [Transformer RUL Predictor](#2-transformer-rul-predictor)
3. [RuleFit Rule Extraction](#3-rulefit-rule-extraction)
4. [Temporal RuleFit-AF](#4-temporal-rulefit-af)
5. [STL RuleFit](#5-stl-rulefit)
6. [Query Engine](#6-query-engine)
7. [XAI Benchmarking Metrics](#7-xai-benchmarking-metrics)

---

## 1. Input & Preprocessing

### 1.1 Raw Feature Space

Let the raw dataset consist of $N$ specimens, each producing a multivariate time series of $d = 16$ sensor features sampled at discrete fatigue cycles.

$$
\mathbf{x}^{(i)}_t \in \mathbb{R}^d, \quad i = 1, \dots, N, \quad t = 1, \dots, T_i
$$

where $T_i$ is the total number of observations for specimen $i$ and $d = 16$ features include PZT signals, acoustic emission, strain, and optical measurements.

### 1.2 Feature Standardization (Z-Score)

Each feature $j$ is standardized across all training specimens:

$$
z_{t,j} = \frac{x_{t,j} - \hat{\mu}_j}{\hat{\sigma}_j}
$$

where $\hat{\mu}_j$ and $\hat{\sigma}_j$ are the training-set mean and standard deviation of feature $j$. After scaling, $z_{t,j} \sim \mathcal{N}(0, 1)$ approximately.

### 1.3 Target Normalization (MinMax)

The RUL target $y^{(i)} \in \mathbb{R}_+$ (in fatigue cycles) is normalized to $[0, 1]$:

$$
\tilde{y}^{(i)} = \frac{y^{(i)} - y_{\min}}{y_{\max} - y_{\min}}
$$

The inverse transform recovers actual cycles:

$$
y^{(i)} = \tilde{y}^{(i)} \cdot (y_{\max} - y_{\min}) + y_{\min}
$$

### 1.4 Sliding Window

A sliding window of length $T = 10$ converts each time series into overlapping subsequences:

$$
\mathbf{X}_i = [\mathbf{z}_{t-T+1}, \, \mathbf{z}_{t-T+2}, \, \dots, \, \mathbf{z}_t] \in \mathbb{R}^{T \times d}
$$

The associated target is computed as:

$$
y_i = \text{RUL at timestep } t \text{ (cycles to failure)}
$$

After windowing, we obtain $n$ samples, each of shape $(T, d) = (10, 16)$.

### 1.5 Data Split

$$
\mathcal{D}_{\text{train}}: \sim 160 \text{ samples}, \quad
\mathcal{D}_{\text{val}}: \sim 84 \text{ samples}, \quad
\mathcal{D}_{\text{test}}: \sim 62 \text{ samples}
$$

---

## 2. Transformer RUL Predictor

The black-box model $f_\theta : \mathbb{R}^{T \times d} \to [0,1]$ is a Transformer encoder followed by an MLP regression head.

### 2.1 Input Projection

Each input $\mathbf{X} \in \mathbb{R}^{T \times d}$ is linearly projected to dimension $d_{\text{model}} = 128$:

$$
\mathbf{H}^{(0)} = \mathbf{X} \mathbf{W}_{\text{proj}} + \mathbf{b}_{\text{proj}}, \quad \mathbf{W}_{\text{proj}} \in \mathbb{R}^{d \times d_{\text{model}}}
$$

### 2.2 Positional Encoding

Sinusoidal positional encoding is added to preserve temporal order:

$$
\text{PE}(t, 2k) = \sin\!\left(\frac{t}{10000^{2k/d_{\text{model}}}}\right), \quad
\text{PE}(t, 2k+1) = \cos\!\left(\frac{t}{10000^{2k/d_{\text{model}}}}\right)
$$

$$
\mathbf{H}^{(0)} \leftarrow \mathbf{H}^{(0)} + \text{PE}
$$

### 2.3 Multi-Head Self-Attention

For each of $h = 8$ heads in each of $L = 4$ encoder layers:

$$
\text{head}_i = \text{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i) = \text{softmax}\!\left(\frac{\mathbf{Q}_i \mathbf{K}_i^\top}{\sqrt{d_k}}\right) \mathbf{V}_i
$$

where $d_k = d_{\text{model}} / h = 16$ and:

$$
\mathbf{Q}_i = \mathbf{H} \mathbf{W}_i^Q, \quad \mathbf{K}_i = \mathbf{H} \mathbf{W}_i^K, \quad \mathbf{V}_i = \mathbf{H} \mathbf{W}_i^V
$$

$$
\text{MultiHead}(\mathbf{H}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O
$$

### 2.4 Transformer Encoder Layer

Each of $L = 4$ layers applies:

$$
\mathbf{H}' = \text{LayerNorm}\!\big(\mathbf{H} + \text{MultiHead}(\mathbf{H})\big)
$$

$$
\mathbf{H}^{(\ell+1)} = \text{LayerNorm}\!\big(\mathbf{H}' + \text{FFN}(\mathbf{H}')\big)
$$

where the feed-forward network is:

$$
\text{FFN}(\mathbf{h}) = \text{ReLU}(\mathbf{h} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2
$$

with $\mathbf{W}_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $d_{\text{ff}} = 512$.

### 2.5 Global Average Pooling

After $L$ encoder layers, the sequence is pooled:

$$
\mathbf{h}_{\text{pool}} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{H}^{(L)}_t \in \mathbb{R}^{d_{\text{model}}}
$$

### 2.6 MLP Regression Head

$$
\mathbf{a}_1 = \text{ReLU}\!\big(\text{Dropout}(\mathbf{h}_{\text{pool}}) \, \mathbf{W}_{\text{fc1}} + \mathbf{b}_{\text{fc1}}\big) \in \mathbb{R}^{64}
$$

$$
\mathbf{a}_2 = \text{ReLU}(\mathbf{a}_1 \, \mathbf{W}_{\text{fc2}} + \mathbf{b}_{\text{fc2}}) \in \mathbb{R}^{32}
$$

$$
\hat{\tilde{y}} = \sigma(\mathbf{a}_2 \, \mathbf{w}_{\text{fc3}} + b_{\text{fc3}}) \in [0, 1]
$$

where $\sigma$ is the sigmoid function. The final RUL prediction in cycles is:

$$
\hat{y} = f_\theta(\mathbf{X}) = \hat{\tilde{y}} \cdot (y_{\max} - y_{\min}) + y_{\min}
$$

### 2.7 Hyperparameters

| Symbol | Value | Description |
|--------|-------|-------------|
| $d$ | 16 | Input feature dimension |
| $T$ | 10 | Sequence length |
| $d_{\text{model}}$ | 128 | Transformer hidden dimension |
| $h$ | 8 | Number of attention heads |
| $L$ | 4 | Number of encoder layers |
| $d_{\text{ff}}$ | 512 | Feed-forward hidden dimension |
| $p_{\text{drop}}$ | 0.1 | Dropout rate |

---

## 3. RuleFit Rule Extraction

RuleFit (Friedman & Popescu, 2008) distills the black-box model into interpretable symbolic rules.

### 3.1 Teacher Predictions

The teacher signal for rule learning is the Transformer's prediction (not ground truth):

$$
\tilde{y}_i = f_\theta(\mathbf{X}_i), \quad i = 1, \dots, n
$$

### 3.2 Snapshot Representation

The standard RuleFit operates on the last-timestep 2D snapshot:

$$
\mathbf{x}_i = \mathbf{X}_{i}[T, :] \in \mathbb{R}^d
$$

### 3.3 Tree Ensemble

A Gradient Boosting Regressor with $M = 80$ trees, max depth $D = 4$, learning rate $\eta = 0.01$, and subsample ratio 0.5 is trained:

$$
g(\mathbf{x}) = \sum_{m=1}^{M} \nu \cdot h_m(\mathbf{x})
$$

where $h_m$ is the $m$-th decision tree, trained to fit:

$$
h_m(\mathbf{x}) \approx \tilde{y}_i - g_{m-1}(\mathbf{x}_i)
$$

### 3.4 Rule Extraction

Each root-to-leaf path in each tree $h_m$ defines a conjunctive rule $r_k$:

$$
r_k(\mathbf{x}) = \mathbb{1}\!\left[\bigwedge_{\ell=1}^{L_k} \left( x_{a_{k\ell}} \circ_{k\ell} \tau_{k\ell} \right)\right]
$$

where:
- $L_k$ = number of conditions (splits) in the path
- $a_{k\ell} \in \{1, \dots, d\}$ = feature index at split $\ell$
- $\circ_{k\ell} \in \{\leq, >\}$ = comparison operator
- $\tau_{k\ell} \in \mathbb{R}$ = threshold

Total candidate rules: $K \approx 1241$ (from 80 trees × ~15 leaves each).

### 3.5 Binary Rule Design Matrix

$$
\mathbf{R} \in \{0, 1\}^{n \times K}, \quad R_{ik} = r_k(\mathbf{x}_i)
$$

### 3.6 Lasso Sparse Selection

A linear model is fit with $\ell_1$ regularization:

$$
\min_{\beta_0, \boldsymbol{\beta}} \; \frac{1}{n} \sum_{i=1}^{n} \left( \tilde{y}_i - \beta_0 - \sum_{k=1}^{K} \beta_k R_{ik} \right)^2 + \lambda \|\boldsymbol{\beta}\|_1
$$

where $\lambda$ is chosen by 5-fold cross-validation (LassoCV). Rules with $|\beta_k| > \epsilon$ are selected:

$$
\mathcal{K}_{\text{sel}} = \{k : |\beta_k| > 10^{-6}\}
$$

The Lasso prediction for a new sample is:

$$
\hat{y}_{\text{Lasso}}(\mathbf{x}) = \beta_0 + \sum_{k \in \mathcal{K}_{\text{sel}}} \beta_k \, r_k(\mathbf{x})
$$

The **importance** of each selected rule is $w_k = |\beta_k|$.

### 3.7 Rule Categorization

For each selected rule $r_k$, define its support set:

$$
\mathcal{S}_k = \{i : r_k(\mathbf{x}_i) = 1\}
$$

Then compute:
- **Coverage:** $\text{cov}(r_k) = |\mathcal{S}_k| / n$
- **Average RUL:** $\bar{y}_k = \frac{1}{|\mathcal{S}_k|} \sum_{i \in \mathcal{S}_k} \tilde{y}_i$
- **Majority stage:** $s_k = \text{mode}\{s_i : i \in \mathcal{S}_k\}$
- **Prediction label:**

$$
\text{pred}(r_k) = \begin{cases}
\texttt{Normal} & \text{if } \bar{y}_k \geq B_{\text{Early}} \\
\texttt{Warning} & \text{if } B_{\text{Mid}} \leq \bar{y}_k < B_{\text{Early}} \\
\texttt{Critical} & \text{if } \bar{y}_k < B_{\text{Mid}}
\end{cases}
$$

with stage boundaries $B_{\text{Early}} = 130{,}000$ and $B_{\text{Mid}} = 80{,}000$ cycles.

---

## 4. Temporal RuleFit-AF

### 4.1 Temporal Augmentation Map

Let $J \subseteq \{1, \dots, d\}$ be the set of features selected for augmentation. For feature $j \in J$, define temporal statistics over the sequence $\mathbf{X}_i[:, j] = (x_{i,1,j}, \dots, x_{i,T,j})$:

**Last-step delta:**
$$
\phi_j^{\Delta}(\mathbf{X}_i) = x_{i,T,j} - x_{i,T-1,j}
$$

**Window delta:**
$$
\phi_j^{\Delta_W}(\mathbf{X}_i) = x_{i,T,j} - x_{i,1,j}
$$

**OLS slope:**
$$
\phi_j^{\text{slope}}(\mathbf{X}_i) = \frac{\sum_{t=1}^{T}(t - \bar{t})(x_{i,t,j} - \bar{x}_{i,j})}{\sum_{t=1}^{T}(t - \bar{t})^2}
$$

where $\bar{t} = \frac{1}{T}\sum_{t=1}^T t$ and $\bar{x}_{i,j} = \frac{1}{T}\sum_{t=1}^T x_{i,t,j}$.

**Volatility:**
$$
\phi_j^{\text{vol}}(\mathbf{X}_i) = \text{std}\!\left(x_{i,2,j} - x_{i,1,j}, \;\dots,\; x_{i,T,j} - x_{i,T-1,j}\right)
$$

### 4.2 Augmented Representation

$$
\mathbf{z}_i = \Phi(\mathbf{X}_i) = \Big[\mathbf{x}_{i,T,:} \;;\; \phi_j^s(\mathbf{X}_i)\Big]_{j \in J,\; s \in \mathcal{S}} \;\in\; \mathbb{R}^{d + |J||\mathcal{S}|}
$$

With $d = 16$, $|J| = 16$, $|\mathcal{S}| = 4$: $\;\mathbf{z}_i \in \mathbb{R}^{80}$.

### 4.3 TemporalRuleFit

$$
\text{TemporalRuleFit} = \text{RuleFit} \;\circ\; \Phi
$$

The tree ensemble is trained on $(\mathbf{z}_i, \tilde{y}_i)$ instead of $(\mathbf{x}_i, \tilde{y}_i)$. All subsequent steps (rule extraction, Lasso selection, categorization) proceed identically.

---

## 5. STL RuleFit

### 5.1 Signal Temporal Logic (STL) Robustness

The STL operators **G** (Globally/Always) and **F** (Finally/Eventually) have quantitative semantics defined via robustness functions:

$$
\rho(G_{[a,b]}(x_j > \tau), \mathbf{X}) = \min_{t \in [a,b]} (x_{t,j} - \tau)
$$

$$
\rho(F_{[a,b]}(x_j > \tau), \mathbf{X}) = \max_{t \in [a,b]} (x_{t,j} - \tau)
$$

The predicate is satisfied iff $\rho > 0$.

### 5.2 STL Feature Extraction

For feature $j \in J$ and interval $[a, b]$:

$$
\phi_j^{\min[a,b]}(\mathbf{X}_i) = \min_{t \in [a,b]} x_{i,t,j}
$$

$$
\phi_j^{\max[a,b]}(\mathbf{X}_i) = \max_{t \in [a,b]} x_{i,t,j}
$$

Default intervals (for $T = 10$):
- **Early:** $[0, 4]$ (first half)
- **Late:** $[5, 9]$ (second half)
- **Full:** $[0, 9]$ (entire window)

Feature naming convention: `min_a_b__feature`, `max_a_b__feature`.

### 5.3 STL ↔ Decision Tree Equivalence

When a decision tree splits on an STL feature, the split has provably correct temporal logic semantics:

| Feature | Split | Equivalent STL | Meaning |
|---------|-------|---------------|---------|
| $\min_{[a,b]}(x_j)$ | $> \tau$ | $G_{[a,b]}(x_j > \tau)$ | Always above $\tau$ |
| $\min_{[a,b]}(x_j)$ | $\leq \tau$ | $F_{[a,b]}(x_j \leq \tau)$ | Eventually drops to $\leq \tau$ |
| $\max_{[a,b]}(x_j)$ | $> \tau$ | $F_{[a,b]}(x_j > \tau)$ | Eventually exceeds $\tau$ |
| $\max_{[a,b]}(x_j)$ | $\leq \tau$ | $G_{[a,b]}(x_j \leq \tau)$ | Always below $\tau$ |

### 5.4 STL Augmented Representation

$$
\mathbf{z}_i^{\text{STL}} = \Big[\mathbf{x}_{i,T,:} \;;\; \phi_j^{\min[a,b]}(\mathbf{X}_i) \;;\; \phi_j^{\max[a,b]}(\mathbf{X}_i)\Big]_{j \in J,\; [a,b] \in \mathcal{I}}
$$

With $|J| = 8$ sensor features, $|\mathcal{I}| = 3$ intervals, 2 operators: $\mathbf{z}_i^{\text{STL}} \in \mathbb{R}^{16 + 48} = \mathbb{R}^{64}$.

### 5.5 STLRuleFit

$$
\text{STLRuleFit} = \text{RuleFit} \;\circ\; \Phi^{\text{STL}}
$$

An extracted STL rule example:

$$
r_k(\mathbf{z}^{\text{STL}}) = \mathbb{1}\!\Big[G_{[5,9]}(\text{scatter\_energy} > 0.84) \;\wedge\; G_{[0,9]}(\text{rms} \leq 2.02)\Big]
$$

---

## 6. Query Engine

The query engine operates on the set of selected rules $\mathcal{R} = \{r_k\}_{k \in \mathcal{K}_{\text{sel}}}$, each annotated with importance $w_k$, stage $s_k$, average RUL $\bar{y}_k$, and prediction label $p_k$.

### 6.1 Rule Matching

For a query sample $\mathbf{x}^*$, the set of **active rules** is:

$$
\mathcal{A}(\mathbf{x}^*) = \{k \in \mathcal{K}_{\text{sel}} : r_k(\mathbf{x}^*) = 1\}
$$

### 6.2 Point Query

**Predicted RUL** (importance-weighted average):

$$
\hat{y}_{\text{QE}}(\mathbf{x}^*) = \frac{\sum_{k \in \mathcal{A}} w_k \, \bar{y}_k}{\sum_{k \in \mathcal{A}} w_k}
$$

**Predicted stage:**

$$
\hat{s}(\mathbf{x}^*) = \text{stage\_from\_rul}\!\big(\hat{y}_{\text{QE}}(\mathbf{x}^*)\big)
$$

**Prediction label** (majority vote):

$$
\hat{p}(\mathbf{x}^*) = \arg\max_{p \in \{N, W, C\}} \sum_{k \in \mathcal{A}} \mathbb{1}[p_k = p]
$$

**Confidence:**

$$
\text{conf}(\mathbf{x}^*) = \min\!\left(1, \; 0.7 \cdot \frac{\sum_{k \in \mathcal{A}} w_k}{\sum_{k \in \mathcal{K}_{\text{sel}}} w_k} + 0.3 \cdot \max_p \frac{|\{k \in \mathcal{A} : p_k = p\}|}{|\mathcal{A}|}\right)
$$

### 6.3 Why Query

For each active rule $r_k \in \mathcal{A}(\mathbf{x}^*)$, produce **evidence**:

For each condition $(a_{k\ell}, \circ_{k\ell}, \tau_{k\ell})$ in rule $r_k$:

$$
\text{evidence}_{k\ell} = \begin{cases}
\text{margin} = \tau_{k\ell} - x^*_{a_{k\ell}} & \text{if } \circ_{k\ell} = \leq \\
\text{margin} = x^*_{a_{k\ell}} - \tau_{k\ell} & \text{if } \circ_{k\ell} = >
\end{cases}
$$

$$
\text{satisfied}_{k\ell} = \mathbb{1}[\text{margin} \geq 0]
$$

Rules are ranked by a **ranking score**:

$$
\text{score}(r_k) = w_k \cdot (1 + 0.15 \cdot L_k)
$$

where $L_k$ is the number of conditions (specificity).

### 6.4 Why-Not Query

Given a target stage $s^*$ or prediction label $p^*$:

1. Retrieve candidate rules: $\mathcal{C} = \{k : s_k = s^* \text{ or } p_k = p^*\}$
2. For each candidate $r_k \in \mathcal{C}$, partition conditions into:

$$
\mathcal{F}_k = \{(a, \circ, \tau) : \text{condition NOT satisfied by } \mathbf{x}^*\}
$$

$$
\mathcal{P}_k = \{(a, \circ, \tau) : \text{condition satisfied by } \mathbf{x}^*\}
$$

3. Rank candidates by $\big(|\mathcal{F}_k|, \; -w_k\big)$ ascending — the best why-not rule is the one with fewest failures.

### 6.5 What-If Query

Given user-specified feature changes $\Delta = \{(j, v_j)\}$ (set feature $j$ to value $v_j$):

1. Create modified sample: $\mathbf{x}' = \mathbf{x}^*$, then $x'_j = v_j$ for each $(j, v_j) \in \Delta$
2. Run point query on $\mathbf{x}'$
3. Report:

$$
\Delta_{\text{RUL}} = \hat{y}_{\text{QE}}(\mathbf{x}') - \hat{y}_{\text{QE}}(\mathbf{x}^*)
$$

$$
\text{stage\_changed} = \mathbb{1}[\hat{s}(\mathbf{x}') \neq \hat{s}(\mathbf{x}^*)]
$$

### 6.6 Counterfactual Query

Given target stage $s^*$, find the **minimum-cost feature change** to satisfy a target rule:

1. For each candidate rule $r_k$ with $s_k = s^*$:
   - Identify failed conditions $\mathcal{F}_k$
   - For each failed condition $(j, \circ, \tau)$, propose:

$$
v_j^* = \begin{cases}
\tau & \text{if } \circ = \leq \\
\tau + \epsilon & \text{if } \circ = >
\end{cases}
$$

   - Compute cost: $\text{cost}_k = \sum_{(j, \circ, \tau) \in \mathcal{F}_k} |v_j^* - x^*_j|$

2. Select the candidate with minimal $(|\mathcal{F}_k|, \; \text{cost}_k, \; -w_k)$.

### 6.7 Global Feature Influence Query

For a filtered rule set (optionally by stage $s$ or prediction $p$), compute for each feature $j$:

$$
\text{GFI}(j) = \sum_{\substack{k \in \mathcal{K}_{\text{sel}} \\ j \in \text{features}(r_k)}} w_k
$$

Features are ranked by $\text{GFI}(j)$ descending and normalized:

$$
\text{GFI}_{\text{norm}}(j) = \frac{\text{GFI}(j)}{\max_{j'} \text{GFI}(j')}
$$

### 6.8 Cohort Pattern Query

Given a cohort-defining criterion $(j, \circ, \tau)$:

1. Partition samples: $\mathcal{C} = \{i : x_{i,j} \circ \tau\}$, $\bar{\mathcal{C}} = \{i : x_{i,j} \not\circ \tau\}$

2. Compute feature-wise mean differences:

$$
\delta_j = \frac{1}{|\mathcal{C}|} \sum_{i \in \mathcal{C}} x_{i,j} - \frac{1}{|\bar{\mathcal{C}}|} \sum_{i \in \bar{\mathcal{C}}} x_{i,j}
$$

3. Rank features by $|\delta_j|$ descending.

4. Optionally compute stage distribution differences between $\mathcal{C}$ and $\bar{\mathcal{C}}$.

---

## 7. XAI Benchmarking Metrics

### 7.1 Faithfulness

**Intuition:** If the explanation correctly identifies important features, perturbing them should cause a large prediction change.

For a sample $\mathbf{X}$, let $\mathcal{T}_K$ be the top-$K$ features identified by the XAI method. Define:

$$
\Delta_{\text{top}} = \left| f_\theta(\mathbf{X}) - f_\theta\!\big(\text{perturb}(\mathbf{X}, \mathcal{T}_K)\big) \right|
$$

$$
\Delta_{\text{rand}}^{(b)} = \left| f_\theta(\mathbf{X}) - f_\theta\!\big(\text{perturb}(\mathbf{X}, \mathcal{R}_K^{(b)})\big) \right|, \quad b = 1, \dots, B
$$

where $\mathcal{R}_K^{(b)}$ is a random set of $K$ features and $B = 5$ baselines.

$$
\text{Faithfulness}(\mathbf{X}) = \frac{\Delta_{\text{top}}}{\frac{1}{B}\sum_{b=1}^B \Delta_{\text{rand}}^{(b)} + \epsilon}
$$

**Interpretation:** Ratio $> 1$ means the identified features matter more than random ones.

### 7.2 Stability

**Intuition:** Small input perturbations should not drastically change the explanation.

Add Gaussian noise $\boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ with $\sigma = 0.05$:

$$
\mathbf{X}' = \mathbf{X} + \boldsymbol{\eta}
$$

Repeat $P = 10$ times. Measure Jaccard similarity of matched rule sets:

$$
\text{Rule Stability} = \frac{1}{P} \sum_{p=1}^P J\!\big(\mathcal{A}(\mathbf{X}), \; \mathcal{A}(\mathbf{X}'_p)\big)
$$

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

Similarly for feature stability (Jaccard of top-$K$ feature sets) and stage stability (fraction of perturbations that preserve the predicted stage).

### 7.3 Compactness

$$
\text{Compactness} = \begin{cases}
\text{avg\_rules\_per\_sample} = \frac{1}{n}\sum_{i=1}^n |\mathcal{A}(\mathbf{x}_i)| \\[6pt]
\text{avg\_conditions\_per\_rule} = \frac{1}{|\mathcal{K}_{\text{sel}}|}\sum_{k \in \mathcal{K}_{\text{sel}}} L_k \\[6pt]
\text{feature\_coverage} = \frac{|\{j : \exists k, \; j \in \text{features}(r_k)\}|}{d}
\end{cases}
$$

### 7.4 Concordance (RuleFit vs SHAP)

**Spearman Rank Correlation:** For a sample, let $\pi^{\text{RF}}$ and $\pi^{\text{SHAP}}$ be the feature importance rankings from RuleFit and SHAP respectively:

$$
\rho_s = 1 - \frac{6 \sum_{j=1}^{d} (\pi^{\text{RF}}_j - \pi^{\text{SHAP}}_j)^2}{d(d^2 - 1)}
$$

**Top-K Overlap:**

$$
\text{Overlap}_K = \frac{|\text{Top}_K^{\text{RF}} \cap \text{Top}_K^{\text{SHAP}}|}{K}
$$

Computed for $K = 3$ and $K = 5$.

---

## Symbol Reference

| Symbol | Definition |
|--------|-----------|
| $N$ | Number of specimens |
| $n$ | Number of windowed samples |
| $d$ | Input feature dimension (16) |
| $T$ | Sequence length (10) |
| $f_\theta$ | Transformer black-box model |
| $\tilde{y}_i$ | Teacher prediction (Transformer output in cycles) |
| $\mathbf{x}_i$ | Last-timestep snapshot, $\mathbb{R}^d$ |
| $\mathbf{X}_i$ | Full sequence window, $\mathbb{R}^{T \times d}$ |
| $\mathbf{z}_i$ | Augmented representation (temporal or STL) |
| $r_k$ | Conjunctive rule (binary indicator function) |
| $K$ | Total candidate rules |
| $\mathcal{K}_{\text{sel}}$ | Selected rule indices (non-zero Lasso) |
| $\mathbf{R}$ | Binary rule design matrix |
| $w_k = |\beta_k|$ | Rule importance (Lasso coefficient magnitude) |
| $\bar{y}_k$ | Average teacher RUL of rule $k$'s support |
| $s_k$ | Majority stage label of rule $k$ |
| $\mathcal{A}(\mathbf{x}^*)$ | Set of active (matched) rules for query $\mathbf{x}^*$ |
| $\Phi$ | Temporal augmentation map |
| $\Phi^{\text{STL}}$ | STL robustness augmentation map |
| $G_{[a,b]}$ | STL "Globally" (Always) operator |
| $F_{[a,b]}$ | STL "Finally" (Eventually) operator |
| $\rho(\cdot)$ | STL robustness function |
| $B_{\text{Early}}, B_{\text{Mid}}$ | Stage boundaries (130k, 80k cycles) |
| $\lambda$ | Lasso regularization parameter |
| $J(\cdot, \cdot)$ | Jaccard similarity |
| $\rho_s$ | Spearman rank correlation |
