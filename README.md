---
license: apache-2.0
language:
- ar
pipeline_tag: text-generation
tags:
- arabic
- islamic
- heritage
- shamela
- classical-arabic
- fiqh
- tafsir
- hadith
- aqeedah
- nahw
- continued-pretraining
- representation-analysis
- cka
- stable-rank
- linear-probe
- domain-specialization
---

# 🕌 3arabLM-4B-Islamic-v2

**A Specialized Arabic Language Model for Islamic Heritage**

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-blue)](https://huggingface.co/sherif1313/3arabLM-4B-islamic-v2)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Research Paper](https://img.shields.io/badge/arXiv-Research%20Paper-red)](https://arxiv.org/abs/XXXX.XXXXX)

---

## 📖 Abstract

Large language models adapted to specialized historical corpora may develop substantial changes in their internal representations, yet conventional evaluation metrics provide limited information about how such changes evolve across network depth. 

This paper presents a **representation-diagnostic framework** for examining layer-wise representation structure in Arabic heritage language models undergoing continued pretraining on the Shamela corpus.

The framework characterizes representations using complementary descriptive diagnostics, including:

- Centered Kernel Alignment (CKA)
- Linear probe performance
- Macro-F1
- Silhouette and Davies–Bouldin scores
- Stable rank
- Far-layer representation similarity

We apply the framework to a **4B-parameter Qwen3.5-based model** after continued pretraining on six domains of Islamic heritage literature:

| Domain | Description |
| :--- | :--- |
| **Aqeedah** | Islamic Creed and Theology |
| **Tafsir** | Quranic Exegesis |
| **Nahw and Sarf** | Arabic Grammar and Morphology |
| **Fiqh** | Islamic Jurisprudence |
| **Hadith Mutoon** | Prophetic Traditions |
| **Fatawa** | Legal Opinions and Verdicts |

---

## 📊 Key Results

### Linear Probe Accuracy Across Layers

| Layer | Accuracy | Macro-F1 | Silhouette | Davies-Bouldin |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 53.80% | 0.537 | -0.0075 | 13.5336 |
| 4 | 70.28% | 0.703 | -0.0060 | 11.1941 |
| 8 | 72.87% | 0.728 | -0.0112 | 11.2660 |
| 12 | 76.76% | 0.766 | -0.0089 | 11.0906 |
| 16 | 78.06% | 0.782 | -0.0068 | 10.1647 |
| 20 | 79.54% | 0.797 | -0.0029 | 8.1488 |
| 24 | 79.72% | 0.799 | -0.0110 | 8.3116 |
| 28 | 77.69% | 0.777 | -0.0124 | 8.4527 |
| **31** | **82.31%** | **0.824** | **0.0031** | **6.7378** |

### Cross-Domain CKA at Layer 31

|  | Aqeedah | Tafsir | Nahw | Fiqh | Mutoon | Fatawa |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Aqeedah** | 1.0000 | 0.0276 | 0.0258 | 0.0262 | 0.0312 | 0.0375 |
| **Tafsir** | 0.0276 | 1.0000 | 0.0228 | 0.0172 | 0.0293 | 0.0217 |
| **Nahw** | 0.0258 | 0.0228 | 1.0000 | 0.0173 | 0.0254 | 0.0258 |
| **Fiqh** | 0.0262 | 0.0172 | 0.0173 | 1.0000 | 0.0213 | 0.0222 |
| **Mutoon** | 0.0312 | 0.0293 | 0.0254 | 0.0213 | 1.0000 | 0.0371 |
| **Fatawa** | 0.0375 | 0.0217 | 0.0258 | 0.0222 | 0.0371 | 1.0000 |

### Stable Rank Across Layers

| Layer | Stable Rank | Stable Rank / 2560 |
| :---: | :---: | :---: |
| 0 | 10.593 | 0.004138 |
| 4 | 20.324 | 0.007939 |
| 8 | 24.854 | 0.009709 |
| 12 | 23.687 | 0.009253 |
| 16 | 18.208 | 0.007113 |
| 20 | 16.453 | 0.006427 |
| 24 | 20.457 | 0.007991 |
| 28 | 18.448 | 0.007206 |
| **31** | **5.762** | **0.002251** |

### Average Domain CKA Across Layers

| Layer | Average Domain CKA |
| :---: | :---: |
| 0 | 0.054200 |
| 4 | 0.127515 |
| 8 | 0.148687 |
| 12 | 0.146753 |
| 16 | 0.130844 |
| 20 | 0.112298 |
| 24 | 0.116429 |
| 28 | 0.117917 |
| **31** | **0.025891** |

---

## 🧠 Research Questions

The study addresses four research questions:

1. **RQ1:** How does domain discriminability change across model depth?
2. **RQ2:** How does cross-domain representational similarity change across layers?
3. **RQ3:** Does the effective rank of the activation space exhibit a characteristic depth-wise trajectory?
4. **RQ4:** Does the final layer exhibit a distinctive combination of domain discriminability, representational similarity, and rank?

These questions are deliberately descriptive. The study does not attempt to infer from representation statistics alone whether the model has reached an architectural capacity limit.

---

## 🏗️ Methodology

### Framework Components

1. **Domain-level CKA** – Measures representational similarity across domains.
2. **Linear-probe domain classification** – Quantifies domain discriminability.
3. **Layer-wise stable-rank analysis** – Measures effective dimensionality.
4. **Far-layer CKA analysis** – Compares representations from distant layers.

### Centered Kernel Alignment (CKA)

For centered activation matrices $X, Y \in \mathbb{R}^{n \times d}$:

$$
\text{CKA}(X, Y) = \frac{\|X^T Y\|_F^2}{\|X^T X\|_F \|Y^T Y\|_F}
$$

### Stable Rank

For an activation matrix $A$ with singular values $\sigma_1, \dots, \sigma_r$:

$$
\text{StableRank}(A) = \frac{\|A\|_F^2}{\|A\|_2^2} = \frac{\sum_i \sigma_i^2}{\sigma_{\max}^2}
$$

---

## 📚 Training Corpus

The model was continued-pretrained on six domains from the Shamela collection:

| Domain | Texts Available |
| :--- | :---: |
| Aqeedah | 100,000 |
| Tafsir | 100,000 |
| Nahw and Sarf | 100,000 |
| General Fiqh | 100,000 |
| Hadith Mutoon | 100,000 |
| Fatawa | 100,000 |
| **Total** | **600,000** |

---

## ⚙️ Activation Collection

- **Layers analyzed:** `{0, 4, 8, 12, 16, 20, 24, 28, 31}`
- **Vectors per domain per layer:** 800
- **Total activation vectors:** 43,200
- **Matrix dimensions:** 800 × 2560
- **Hardware:** NVIDIA RTX 3090 (24 GB VRAM)

---

## 🔬 Key Findings

1. **Domain discriminability increases substantially with depth** – Linear probe accuracy rises from 53.80% (L0) to 82.78% (L31).

2. **Cross-domain representational similarity decreases strongly in the final layer** – Average domain CKA falls from ~0.11–0.15 in middle layers to 0.025891 at L31.

3. **Stable rank exhibits a non-monotonic trajectory** – Reaches 24.854 at L8, then decreases to 5.762 at the final layer.

4. **The final layer has a distinctive representational regime** – Characterized by high domain discriminability and strong spectral compression.

---

## ⚠️ Limitations

| Limitation | Description |
| :--- | :--- |
| **No Base-Model Comparison** | The original Qwen3.5-4B base model was not included under identical diagnostic conditions. |
| **Single Main Checkpoint** | Analysis focuses on checkpoint 97,500; no trajectory across training checkpoints. |
| **No Statistical Confidence Intervals** | Single diagnostic run; confidence intervals not reported. |
| **Not Mechanistic Interpretability** | The framework is representational, not causal. |
| **No Catastrophic-Forgetting Experiment** | SFT was not tested. |

---

## 🗓️ Future Work

- **Checkpoint Trajectory:** 10k, 30k, 50k, 75k, 97.5k steps.
- **Base-Model Comparison:** Original Qwen3.5-4B.
- **Intrinsic Dimension:** Levina–Bickel MLE.
- **Effective Rank and Participation Ratio:** Additional spectral measures.
- **SFT and Forgetting Experiments:** Domain-specific SFT with pre/post evaluation.
- **Causal Representation Interventions:** Activation patching, ablation, feature steering.

---

## 📜 Citation

If you use this model in academic research, please cite:

```bibtex
@article{hassan_2026_3arabLM,
    title={A Representation-Diagnostic Framework for Arabic Heritage Language Models: A Case Study on Continued Pretraining with the Shamela Corpus},
    author={Hassan, Sherif},
    year={2026},
    url={https://huggingface.co/sherif1313/3arabLM-4B-islamic-v2}
}
