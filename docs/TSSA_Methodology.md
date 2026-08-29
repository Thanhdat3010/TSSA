# Section 3: Proposed Method — Target-Side Semantic Anchoring (TSSA)

## 3.1 Problem Setting and Baseline Objective

Let $\mathbf{x} = (x_1, x_2, \dots, x_m)$ denote a sequence of tokens in a severely low-resource source language $S$, and let $\mathbf{y} = (y_1, y_2, \dots, y_n)$ denote its corresponding translation in a high-resource target language $T$. The trainable translation model comprises a source encoder $E_S$, an autoregressive decoder $D$, and parameter set $\theta$. In parallel, we introduce a frozen target teacher encoder $E_T$, initialized from a high-capacity pretrained target-language model (e.g., BARTpho) or multilingual sequence-to-sequence model (e.g., mBART).

In standard neural machine translation (NMT), the parameters $\theta$ are optimized solely via the conditional teacher-forced cross-entropy loss:

$$\mathcal{L}_{\text{MT}}(\theta) = - \frac{1}{n} \sum_{t=1}^{n} \log p_\theta(y_t \mid y_{<t}, \mathbf{x})$$

A fundamental design requirement of TSSA is that the target teacher $E_T$ processes $\mathbf{y}$ **exclusively during training**. At inference time, $E_T$ is entirely discarded, and translation decoding proceeds without requiring target references, reverse translation models, or pivot intermediate representations.

```
Training Time Pipeline:
[Source x] ---> Encoder E_S ------------> Hidden States h_i^S ---+
                    |                                            |
                    +-- (L_prime Contrastive)                    +---> [Cross-Attention] ---> Decoder D ---> Output y
                    |                                            |       ^
[Target y] ---> Frozen Teacher E_T ----> Teacher States h_j^T --+       | (L_route Gating)
                    |                                            |       |
                    +-- Posterior Matrix A (Aligner) ------------+-------+
```

---

## 3.2 Confidence-Weighted Structural Anchoring ($\mathcal{L}_{\text{struct}}$)

To transfer token-level contextual geometry from the rich target space to the source encoder, TSSA aligns source representations with target barycenters derived from an offline word-alignment posterior.

For each parallel pair $(\mathbf{x}, \mathbf{y})$, an external aligner produces a soft alignment posterior matrix $\mathbf{A} \in [0, 1]^{m \times n}$, where $A_{ij}$ denotes the alignment confidence between source token $x_i$ and target token $y_j$. The contextual representations produced by the source encoder and frozen teacher are:

$$h_i^S = E_S(\mathbf{x})_i \in \mathbb{R}^{d_S}, \quad h_j^T = E_T(\mathbf{y})_j \in \mathbb{R}^{d_T}$$

### 1. Target Barycenter Construction
For each source position $i$, we compute a normalized target barycenter $\bar{h}_i^T$ by taking the convex combination of teacher token states weighted by normalized alignment probabilities $\tilde{A}_{ij}$:

$$\bar{h}_i^T = \sum_{j=1}^{n} \tilde{A}_{ij} h_j^T, \quad \text{where} \quad \tilde{A}_{ij} = \frac{A_{ij}}{\sum_{k=1}^{n} A_{ik} + \varepsilon}$$

### 2. Confidence-Weighted Loss Formulation
Let $P_S \in \mathbb{R}^{d \times d_S}$ and $P_T \in \mathbb{R}^{d \times d_T}$ be learned linear projection matrices mapping representations into a shared $d$-dimensional space. We define the token-level confidence weight as $c_i = \min\left(1, \sum_{j=1}^{n} A_{ij}\right)$. Unaligned source tokens (where $\sum_j A_{ij} \approx 0$) receive near-zero weight and are masked out.

The structural anchoring loss is formulated as:

$$\mathcal{L}_{\text{struct}} = \frac{1}{\sum_{i=1}^{m} c_i + \varepsilon} \sum_{i=1}^{m} c_i \left\| \frac{P_S h_i^S}{\|P_S h_i^S\|_2 + \varepsilon} - \text{sg}\left( \frac{P_T \bar{h}_i^T}{\|P_T \bar{h}_i^T\|_2 + \varepsilon} \right) \right\|_2^2$$

where $\text{sg}(\cdot)$ denotes the **stop-gradient operator**. Applying $\text{sg}(\cdot)$ to the teacher representations prevents reciprocal parameter co-adaptation, ensuring that the target semantic space serves as a fixed geometric anchor rather than a moving target.

---

## 3.3 Sentence-Level Semantic Priming ($\mathcal{L}_{\text{prime}}$)

While $\mathcal{L}_{\text{struct}}$ enforces fine-grained token correspondences, rare morphological variants, divergent syntactic structures, and unaligned tokens require global sequence-level semantic guidance. TSSA incorporates an in-batch contrastive learning objective to align whole-sentence representations.

Let $z_b^S$ and $z_b^T$ denote the $\ell_2$-normalized, mask-aware mean-pooled projected representations of the source and target sentences for batch item $b \in \{1, \dots, B\}$:

$$z_b^S = \text{Normalize}\left( \frac{1}{m_b} \sum_{i=1}^{m_b} P_S h_{b, i}^S \right), \quad z_b^T = \text{Normalize}\left( \frac{1}{n_b} \sum_{j=1}^{n_b} P_T h_{b, j}^T \right)$$

Given temperature hyperparameter $\tau > 0$, the sentence-level priming loss is defined via InfoNCE over in-batch negatives:

$$\mathcal{L}_{\text{prime}} = - \frac{1}{B} \sum_{b=1}^{B} \log \frac{\exp\left( (z_b^S)^\top \text{sg}(z_b^T) / \tau \right)}{\sum_{k=1}^{B} \exp\left( (z_b^S)^\top \text{sg}(z_k^T) / \tau \right)}$$

Semantically near-duplicate target sentences within the minibatch are screened via metadata and masked to prevent false negative penalization.

---

## 3.4 Anchor-Consistent Head-Wise Routing ($\mathcal{L}_{\text{route}}$)

Standard cross-attention in Transformer decoders attends broadly across all encoder positions, potentially transmitting ungrounded or noisy source signals under low-resource regimes. Instead of costly edge-wise token gates, TSSA deploys lightweight **head-wise routing** at each cross-attention head.

### 1. Inference-Compatible Routing Gate
At decoder layer $\ell$, attention head $h$, and target generation step $t$, let $o_{\ell h t}$ denote the un-gated head output, $q_{\ell h t}$ the attention query vector, and $\mathbf{s} = \frac{1}{m} \sum_{i=1}^{m} h_i^S$ the pooled source context summary. A lightweight multi-layer perceptron ($\text{MLP}_{\ell h}$) predicts a dynamic scalar gate $g_{\ell h t} \in [0, 1]$:

$$g_{\ell h t} = \sigma\left( \text{MLP}_{\ell h} \left( [q_{\ell h t}; \mathbf{s}; q_{\ell h t} \odot \mathbf{s}] \right) \right)$$

$$\tilde{o}_{\ell h t} = g_{\ell h t} \cdot o_{\ell h t}$$

All inputs to the gate ($q_{\ell h t}$ and $\mathbf{s}$) are available at inference time, ensuring zero reference leakage.

### 2. Detached Teacher-Supervised Reliability Target
During teacher-forced training, we compute an anchor-consistency reliability target $r_{\ell h t} \in [0, 1]$ by evaluating the cosine similarity between the projected head output and the corresponding frozen teacher state at target step $t$:

$$r_{\ell h t} = \frac{1 + \cos\left( R_{\ell h} o_{\ell h t}, \text{sg}(P_T h_t^T) \right)}{2}$$

where $R_{\ell h}$ is a learned projection matrix. The routing supervision is optimized via soft binary cross-entropy:

$$\mathcal{L}_{\text{route}} = - \frac{1}{N} \sum_{\ell, h, t} \left[ r_{\ell h t} \log g_{\ell h t} + (1 - r_{\ell h t}) \log (1 - g_{\ell h t}) \right]$$

Crucially, the reliability target $r_{\ell h t}$ is **detached** from the computational graph when updating the router $\text{MLP}_{\ell h}$, preventing trivial degenerate solutions.

---

## 3.5 Unified Objective and Training Schedule

The total training objective of TSSA combines translation cross-entropy with the three semantic anchoring losses:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MT}} + \lambda_1 \mathcal{L}_{\text{struct}} + \lambda_2 \mathcal{L}_{\text{prime}} + \lambda_3 \mathcal{L}_{\text{route}}$$

### Training Schedule & Stabilization
1. **Warm-up Phase**: The model begins with pure translation loss $\mathcal{L}_{\text{MT}}$ with gating set to $g=1$ to establish baseline sequence generation capability.
2. **Linear Ramp Phase**: Structural anchoring $\lambda_1$ and priming $\lambda_2$ are linearly ramped up as the source encoder space aligns with the target geometry.
3. **Routing Activation Phase**: $\mathcal{L}_{\text{route}}$ is activated after token and sentence representations stabilize, training cross-attention heads to selectively filter anchor-consistent evidence.

### Optional Extension: Wake–Sleep Semantic Consolidation (TSSA+Sleep)
Once the core model converges, an optional offline consolidation step generates synthetic source-target bitext, filters pseudo-pairs based on translation confidence, anchor agreement, and structural coverage, and fine-tunes the model on the filtered synthetic data with a capped ratio.
