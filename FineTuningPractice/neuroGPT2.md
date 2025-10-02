# **NeuroGPT 2.0: An Enhanced Foundation Model for EEG via Advanced Architectures, Causal Masking Optimization, and PeFT Finetuning Adaptation**

#### **1. Introduction**

Electroencephalography (EEG) foundation models address key challenges of data scarcity and heterogeneity in Brain-Computer Interface (BCI) applications. Prior work (e.g., Neuro-GPT by Cui et al.) demonstrated that EEG encoders combined with decoder-only transformers and a causal reconstruction objective can learn useful representations at scale, improving downstream motor imagery performance. This proposal builds on that foundation with targeted architectural and training refinements.

Building upon these substantial contributions, this proposal identifies several opportunities to further advance EEG foundation models through modern architectural innovations and training techniques. Current approaches, while effective, can benefit from:
1.  **Enhanced encoder architectures** that leverage recent advances in multimodal and contrastive learning to capture complex neural dynamics.
2.  **Refined causal masking strategies** that better align with the stochastic nature of brain signals.
3.  **Parameter-Efficient Fine-Tuning (PeFT)** to maximize model utilization while addressing the overfitting concerns identified in the original work.

This work introduces NeuroGPT 2.0, an enhanced EEG foundation model that integrates these advancements while retaining the encoder–GPT structure.

#### **2. Background: The Foundation Established by NeuroGPT**

The original Neuro-GPT by Cui et al. successfully demonstrated that treating EEG chunks as tokens and using a causal reconstruction loss is an effective method for self-supervised pre-training. The model architecture consists of an **EEG encoder** with two convolutional layers and six self-attention layers, and a **GPT model**. The encoder generates 1,080-dimensional embeddings which are then projected to 1,024 dimensions to be compatible with the GPT-2 backbone.

The model was pre-trained on the extensive TUH EEG dataset (over 20,000 recordings, totaling 5,656 hours). A key finding was that **encoder-only fine-tuning achieved superior performance** on the downstream motor imagery task compared to fine-tuning the complete model, which tended to overfit on the small BCI 2a dataset. This critical insight provides the primary motivation for integrating PeFT in NeuroGPT 2.0.

As reported by the original authors, that prior study acknowledged public funding (e.g., DARPA, NIH); these acknowledgements pertain solely to the cited work and are unrelated to the present submission.

#### **3. Datasets and Preprocessing**

To ensure direct comparability and follow established protocols, NeuroGPT 2.0 uses the same datasets as the referenced prior work and specifies a transparent preprocessing pipeline.

**3.1 Pre-training: TUH EEG Corpus**

We will pre-train on the Temple University Hospital (TUH) EEG Corpus, a large-scale clinical dataset with diverse pathologies and recording conditions (\(~20{,}000\) recordings from \(~15{,}000\) subjects, totaling \(~5{,}656\) hours).

- Heterogeneity handling: TUH includes > 40 channel montages and variable sampling rates. We map all recordings to a common 10–20 subset (22 channels), re-reference to common average, apply a 0.5–100 Hz bandpass and a 60 Hz notch, then resample to 250 Hz.
- Artifact mitigation: We will employ automatic artifact detection (blink/muscle heuristics) and, where needed, lightweight ICA or regression for ocular channels if available; segments with extreme amplitudes are discarded.
- Channel alignment: When montages lack a target channel, we use spherical interpolation to the 10–20 grid before selection; missing channels are imputed by interpolation from nearest neighbors.

**3.2 Downstream: BCI Competition IV Dataset 2a (Motor Imagery)**

We focus on motor imagery (MI), a canonical BCI task with well-defined protocols.

- Dataset: 9 subjects, 2 sessions recorded on different days; each session comprises 288 trials (72 per class) across 4 classes: left hand, right hand, feet, tongue. EEG: 22 channels (10–20), 250 Hz. EOG channels are also provided.
- Trial timing: Each trial contains a cue onset; we extract an MI window of \([0.5, 4.0]\) s post-cue (robust in prior literature), preceded by baseline correction using \([-0.5, 0]\) s.
- Preprocessing: Common average reference, bandpass 4–38 Hz (sensorimotor rhythms), optional 50/60 Hz notch. Per-channel z-score normalization within session. Optional filter-bank variant (e.g., 4–8, 8–12, 12–26, 26–38 Hz) for an ablation.
- Channel mapping: We ensure the TUH-to-BCI2a 22-channel mapping is consistent; any missing TUH channels during pretraining are interpolated on the 10–20 scalp grid to avoid representation shift during fine-tuning.
- Label format: 4-way classification with class weights \(w_c\) to address potential subject-specific imbalance.

Objective for MI classification (class-weighted cross-entropy):

$$
\mathcal{L}_{\text{CE}} = - \sum_{c=1}^{4} w_c\, y_c \log p_c, \quad \text{where } p = \operatorname{softmax}(h_\theta(x)),\; y \in \{0,1\}^4.
$$

**3.3 EEG Tokenization and Chunking**

We treat short EEG segments as tokens for the decoder-only backbone. Given a continuous sequence of length \(T\) seconds at \(f_s\) Hz, we use fixed-length windows of \(T_w\) seconds with stride \(\Delta\) to produce a token sequence of length

$$
L = 1 + \left\lfloor \frac{f_s T - f_s T_w}{f_s \Delta} \right\rfloor.
$$

Unless otherwise noted, we use \(T_w = 1.0\,\text{s}\) (250 samples), \(\Delta = 0.5\,\text{s}\), and limit context length to \(L \in [256, 512]\) depending on GPU memory.

**3.4 Practical challenges and options**

- Montage mismatch and missing channels: mitigate with spherical interpolation and consistent 22-channel mapping.
- Nonstationarity across sessions/subjects: employ per-session normalization and subject-aware augmentations during pretraining.
- Class imbalance and calibration: use class weights \(w_c\), optionally focal loss as an ablation.
- Artifacts: automatic detection and exclusion; compare with light ICA.
- Domain shift TUH \(\to\) BCI2a: consider feature normalization transfers and small adapter layers per dataset as an ablation.

#### **4. Proposed Enhancements: The NeuroGPT 2.0 Methodology**

NeuroGPT 2.0 retains the core Encoder-GPT structure but introduces significant upgrades to each component.

**4.1 Enhanced EEG Encoder**
We enhance the encoder's ability to learn robust spatio-temporal features via two components:

*   **Neuroanatomically-Informed Spatial Priors**: We augment attention with a spatial bias matrix \(M_{\text{spatial}}\) that encodes electrode proximity, injecting anatomical structure directly into the model:

    $$
    \mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left( \frac{QK^\top}{\sqrt{d_k}} + \alpha M_{\text{spatial}} \right) V.
    $$

    One practical choice is an RBF kernel over 3D electrode coordinates \(p_i\):

    $$
    [M_{\text{spatial}}]_{ij} = -\frac{\lVert p_i - p_j \rVert_2^2}{2\sigma^2}, \quad \text{with scale } \alpha \text{ and bandwidth } \sigma.
    $$

    The bias can be shared across heads or made per-head; \(\alpha\) controls its contribution relative to \(\tfrac{QK^\top}{\sqrt{d_k}}\).

*   **Contrastive Learning Integration**: To complement reconstruction, we add a subject-aware InfoNCE loss that encourages invariance within subject/session and separation across subjects:

    $$
    \mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp\big(\operatorname{sim}(z_i, z_{j^+})/\tau\big)}{\sum_{k=1}^{2N} \mathbf{1}[k\neq i] \, \exp\big(\operatorname{sim}(z_i, z_k)/\tau\big)}.
    $$

    We use cosine similarity \(\operatorname{sim}(a,b) = \tfrac{a^\top b}{\lVert a\rVert\,\lVert b\rVert}\), temperature \(\tau>0\), positives \(z_{j^+}\) from the same subject/session, and augmentations such as time-warping, jitter, frequency masking, and channel dropout.

**4.2 Refined Causal Masking Strategies**
Building upon progressive causal masking with a reconstruction loss, we explore dynamic variants during pretraining:

*   **Adaptive Probabilistic Masking**: A lightweight network \(f_\theta\) adapts token-wise masking probabilities based on local signal features:

    $$
    \Pr(\text{mask}_i) = \sigma\big(f_\theta(x_i)\big) \cdot \rho,
    $$

    where \(\sigma\) is the logistic function and \(\rho\) scales overall sparsity.

*   **Constrained Future-Aware Masking (pretraining only)**: We allow a limited lookahead window \(w\) with logarithmic decay, while preserving temporal structure:

    $$
    M^{\text{future}}_{ij} = \begin{cases}
    0, & i \ge j \\
    -\beta\, \log (j - i + 1), & i < j < i + w \\
    -\infty, & j \ge i + w
    \end{cases}
    $$

    This variant is used only in pretraining. All downstream classification uses strictly causal attention to avoid temporal leakage.

 **4.3 Modern LLM Backbone Integration**
While the original model successfully used GPT-2, we will explore integration with more recent open-source models like Llama or Mistral. These architectures offer enhanced representational capacity, native support for PeFT, and better optimization stability, directly addressing the fine-tuning challenges identified in the original work.

#### **5. Parameter-Efficient Fine-Tuning (PeFT) for Downstream Adaptation**

**5.1 LoRA vs. Traditional Fine-Tuning: Overcoming the Overfitting Challenge**
A key finding from the original study was that fine-tuning the entire `Encoder+GPT` model led to overfitting and performed worse than fine-tuning the encoder alone. This is a classic problem when adapting large models to small datasets.
*   **Traditional Fine-Tuning** updates all `W` parameters of the model. With millions of parameters in the GPT component and a small downstream dataset (only 9 subjects in BCI 2a), this approach is highly susceptible to memorizing the training data, leading to poor generalization. It also risks "catastrophic forgetting," where the valuable general-purpose features learned during pre-training are overwritten.
*   **Low-Rank Adaptation (LoRA)** offers a direct solution. It freezes the massive pre-trained weights (\(W_0\)) and injects small, trainable low-rank matrices \(A\) and \(B\) to approximate the weight update:

    $$
    \Delta W = BA, \qquad W = W_0 + \Delta W, \qquad \operatorname{rank}(A) = \operatorname{rank}(B) = r.
    $$

    Training only \(A,B\) reduces the number of trainable parameters by orders of magnitude, mitigating overfitting while preserving pretraining knowledge.

**5.2 Advanced PeFT Implementations**
To maximize the benefits of this approach, we will implement several advanced PeFT techniques:

*   **LoRA Integration**: We will apply LoRA to both encoder and GPT components, hypothesizing this will enable the `Encoder+GPT` fine-tuning strategy to finally outperform the `Encoder-only` approach.
*   **Adaptive Rank Selection**: Dynamic rank allocation based on layer-wise gradient norms focuses the parameter budget where it is most needed:

    $$
    r_\ell = \max\!\Big( r_{\min}, 
    \min\!\big( r_{\max}, \big\lfloor \frac{\lVert \nabla_\ell \rVert_2}{\max_k \lVert \nabla_k \rVert_2} \, r_{\text{budget}} \big\rfloor \big) \Big).
    $$

*   **Decoupled Learning Rates**: Use separate learning rates for \(A\) and \(B\) to stabilize optimization:

    $$
    \eta_A = \eta_{\text{base}}, \qquad \eta_B = \lambda \, \eta_{\text{base}}, \quad \lambda > 1.
    $$

#### **6. Evaluation Protocol**

Our evaluation will be rigorous and directly comparable to the original work.

*   **Datasets**: We will use the **TUH EEG Corpus** for pre-training and the **BCI Competition IV Dataset 2a** for the downstream motor imagery task, replicating the original experimental setup. This includes performing the critical **channel resampling** step to align the two datasets' sensor configurations.
*   **Cross-Subject Validation**: We use strict leave-one-subject-out cross-validation (LOSO). The overall accuracy is

    $$
    \operatorname{Acc}_{\text{LOSO}} = \frac{1}{N}\sum_{i=1}^{N} \operatorname{Acc}\big( (\mathcal{D} \setminus S_i) \to S_i \big).
    $$
*   **Ablation Studies**: We will conduct systematic ablation studies to isolate the contribution of each enhancement: encoder architecture, masking strategy, PeFT methods, and LLM backbone choice.
*   **Experimental Protocol**: The process will be systematic: (1) Pre-train NeuroGPT 2.0 on TUH EEG; (2) Evaluate PeFT approaches on downstream tasks; (3) Compare results against Neuro-GPT baselines; (4) Statistically validate all improvements.

#### **7. Expected Findings**

1.  **Enhanced Representation Quality**: The advanced encoder should capture more nuanced spatio-temporal patterns, leading to improved downstream performance.
2.  **Improved Fine-tuning Effectiveness**: PeFT should enable effective utilization of the complete model architecture, avoiding the overfitting issues identified in the original work and allowing the `Encoder+GPT` strategy to achieve its full potential.
3.  **Better Generalization**: The combination of enhanced representations and optimized fine-tuning should result in improved cross-subject generalization, addressing a key challenge in EEG applications.

#### **8. Broader Impact and Future Applications**

The primary goal of this research is to develop a more powerful and generalizable foundation model for EEG, directly addressing the core challenges of data scarcity and heterogeneity. The advancements in NeuroGPT 2.0 will catalyze progress in several key domains:

*   **Clinical Neuroscience**: Enable more robust systems for the early detection and monitoring of neurological disorders like epilepsy or Alzheimer's disease from routine clinical recordings.
*   **Brain-Computer Interfaces (BCIs)**: Significantly improve the performance of assistive technologies, such as communication devices for "locked-in" patients and more intuitive control for advanced neuroprosthetics, by enhancing motor imagery classification.
*   **Neurotechnology Applications**: Provide scalable solutions for real-time cognitive state monitoring in high-stakes professions (e.g., pilots, surgeons) to assess mental workload and prevent human error.

#### **9. Conclusion**

NeuroGPT 2.0 builds directly upon prior work on encoder–decoder self-supervised learning for EEG. By integrating enhanced encoder architectures with spatial priors, dynamic (pretraining-only) causal masking, and parameter-efficient fine-tuning, this work aims to advance scalable and generalizable EEG analysis. A rigorous LOSO evaluation and clear ablations will quantify each contribution, and the resulting code and configurations will be released to support reproducibility.