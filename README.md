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
| **31** | **82.78%** | **0.824** | **0.0031** | **6.7378** |

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

## 📦 Current Release
Model: sherif1313/3arabLM-4B-islamic-v2

This release represents a new stage in the 3arabLM research project. The previous development focused more narrowly on Islamic scholarly domains such as Fiqh and Tafsir. The new direction expands the training curriculum toward a broader Islamic Heritage Foundation Model covering multiple branches of Islamic and Arabic scholarship. The model should still be considered an early research milestone. A sustantial portion of the planned training curriculum remains to be completed.

```text

These books have been preserved in the model weightsو 

النحو الوافي
تمهيد القواعد بشرح تسهيل الفوائد
شرح ألفية ابن مالك للحازمي
شرح ألفية ابن مالك للشاطبي = المقاصد الشافية
شرح المفصل لابن يعيش
الموسوعة الفقهية الكويتية
موسوعة الإجماع في الفقه الإسلامي
موسوعة فقه العبادات
فتاوى الشبكة الإسلامية
مجموع فتاوى ورسائل العثيمين
السنن الكبرى للبيهقي ت التركي
المحيط في الاحاديث النبوية والسنن والاثار
جامع الرويات
حلية الأولياء وطبقات الأصفياء
صحيح البخاري 
الجامع لشعب الإيمان للبيهقي
الموسوعة العقدية - الدرر السنية
المهذب النقي الجامع لتفسير ابن جرير الطبري
الموسوعة القرآنية
تفسير ابن كثير _ 
تفسير القرطبي
روح البيان

When selecting other books, please modify the code.
            do_sample=True,               
            repetition_penalty=1.08,
            no_repeat_ngram_size=4,


```
---

## 🗓️ ⚠️ Current Limitations
The current model has several limitations:

The training curriculum is still incomplete.
Some disciplines remain underrepresented.
Source attribution is still under development.
Generated claims should be verified against original sources.
🔬 Research Directions
The project explores several research questions:

Full-Parameter Memorization
How much classical scholarly knowledge can be encoded directly into model parameters?

Knowledge Reconstruction
Can a language model reconstruct scholarly passages without relying entirely on external retrieval?

Hierarchical Representation
Can لbook, chapter, section, and paragraph structures improve scholarly representations?

Metadata-Aware Pretraining
Can author, book, century, madhhab, and discipline metadata improve knowledge organization?

Knowledge-Preserving Scaling
Can model capacity be increased without catastrophic forgetting?

Expert Models
Can specialized Islamic experts be efficiently developed from a shared foundation?

📖 Large Language Models as Compressed Digital Libraries
The central research hypothesis of 3arabLM is that language models may function not only as generators, but also as compressed representations of large scholarly corpora. The project therefore explores the possibility of building:

📖 Knowledge Recall Models
🧠 Memorization-Oriented Language Models
🏛 Compressed Digital Libraries
The ultimate objective is to study how much classical Arabic scholarly knowledge can be represented and reconstructed from model parameters.

🎯 Project Goal
The ultimate objective of 3arabLM is:

To preserve, recall, and reconstruct the Arabic Islamic scholarly heritage with the highest possible degree of fidelity while maintaining the language, terminology, methodologies, and diversity of the original scholarly tradition.

The project is not intended to replace scholars.

It is a research effort exploring how modern language-model technology can contribute to:

Digital preservation.
Computational study of Islamic heritage.
Classical Arabic NLP.
Scholarly knowledge representation.
Knowledge memorization.
Text reconstruction.
🗓️ Roadmap
Stage	Description	Status
Stage 1	Foundation — Continual pretraining on selected Islamic scholarly domains	✅ Complete
Stage 2	Scholarly Expansion — Expansion into Hadith, Sirah, History, Aqeedah, Arabic Language, Literature, and Spiritual Literature	🔄 In Progress
Stage 3	Metadata Integration — Adding structured information about books, authors, periods, schools, and disciplines	⏳ Planned
Stage 4	Capacity Scaling — Increasing model capacity while preserving previously learned knowledge	⏳ Planned
Stage 5	Expert Models — Developing specialized experts for individual scholarly disciplines	⏳ Planned
Stage 6	Islamic Heritage Foundation — Building a large-scale Arabic foundation model	⏳ Long-Term

---

## 📜 Citation

If you use this model in academic research, please cite:

```bibtex
@article{hassan_2026_3arabLM,
    title={3arabLM-4B-islamic-v2},
    author={Hassan, Sherif},
    year={2026},
    url={https://huggingface.co/sherif1313/3arabLM-4B-islamic-v2}
}
