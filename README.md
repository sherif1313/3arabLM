A Representation-Diagnostic Framework for Arabic
Heritage Language Models: A Case Study on
Fine-Tuning with the Shamela Corpus
Sheriff Hassan
Shamela Builder Project
August 2026
Abstract
Large language models adapted to specialized historical corpora may develop substantial
changes in their internal representations, yet conventional evaluation metrics provide limited
information about how such changes evolve across network depth. This paper presents a
representation-diagnostic framework for examining layer-wise representation structure in
Arabic heritage language models undergoing continued pretraining on the Shamela corpus.
Rather than treating internal representations as direct evidence of causal mechanisms,
the proposed framework characterizes them using complementary descriptive diagnostics,
including centered kernel alignment (CKA), linear probe performance, macro-F1, silhouette
and Davies–Bouldin scores, stable rank, and far-layer representation similarity. We apply
the framework to a 4B-parameter Qwen3.5-based model after continued pretraining on six
domains of Islamic heritage literature: Aqeedah, Tafsir, Nahw and Sarf, Fiqh, Hadith Mutoon,
and Fatawa.
The analysis reveals a pronounced increase in domain discriminability with depth. Linear-
probe accuracy rises from 53.80% at layer 0 to 82.31% at the final layer, while macro-F1
increases from 0.537 to 0.824. At the same time, the average cross-domain CKA reaches a
maximum of 0.1487 at layer 8 and decreases sharply to 0.0259 at layer 31. Stable rank follows
a non-monotonic trajectory, reaching 24.854 at layer 8 before decreasing to 5.762 at the final
layer.
These observations indicate that the final representation is substantially more domain-
discriminative and considerably lower-rank than intermediate representations. However,
these measurements do not by themselves establish knowledge compartmentalization, causal
specialization, architectural saturation, or protection against catastrophic forgetting. We
therefore interpret the findings as evidence of representational specialization and compression,
while identifying several experiments required to determine their functional consequences.
Keywords: Arabic language models; representation analysis; CKA; linear probes; stable
rank; Shamela; continued pretraining; heritage Arabic; LLMs; representation diagnostics.
1
Introduction
Large language models have demonstrated increasingly strong capabilities across multilingual and
domain-specific applications. However, adapting general-purpose language models to historical
and specialized corpora raises an important question: how do internal representations change as
the model becomes increasingly exposed to a specialized domain?
This question is particularly relevant for Arabic heritage literature. Classical Islamic texts
contain substantial lexical, syntactic, stylistic, and terminological variation. The Shamela collec-
tion provides a large body of historical Islamic literature covering multiple disciplines, including
jurisprudence, Qur’anic exegesis, Hadith, Arabic grammar, theology, and fatwa literature.
Conventional evaluation based on language-model loss or downstream task accuracy can
reveal whether a model improves, but does not directly describe the geometry of its internal
1representations. Representation similarity analysis provides one complementary approach. In
particular, centered kernel alignment (CKA) has been widely used to compare neural repre-
sentations across layers and models [Kornblith et al., 2019]. CKA is attractive because it can
capture similarity between high-dimensional representations without requiring a direct one-to-one
correspondence between individual features.
Another useful perspective is provided by dimensionality measures. Previous work has
shown that neural representations can have effective dimensionality far smaller than the nominal
number of hidden units, with dimensionality changing substantially across depth [Ansuini et al.,
2019]. Such findings motivate examining rank-related statistics in language models rather than
interpreting hidden size alone as an indication of effective representational capacity.
The present study therefore focuses on a descriptive question: How does the representational
structure of a Shamela-continued-pretrained Arabic language model change across network depth,
and how does this change relate to the discriminability of Islamic textual domains?
The contribution of this work is a practical diagnostic framework combining several com-
plementary measurements. The framework is intended to help researchers inspect internal
representation trajectories before making stronger claims about model scaling, further continued
pretraining, or downstream adaptation.
Importantly, this work does not claim to provide mechanistic interpretability in the strict
causal sense. The measurements presented here characterize representational geometry and
predictive information. They do not identify individual causal circuits, neurons, attention heads,
or mechanisms responsible for the observed behavior.
2
Research Questions
The study addresses four research questions.
RQ1: How does domain discriminability change across model depth?
RQ2: How does cross-domain representational similarity change across layers?
RQ3: Does the effective rank of the activation space exhibit a characteristic depth-wise trajectory?
RQ4: Does the final layer exhibit a distinctive combination of domain discriminability, represen-
tational similarity, and rank?
These questions are deliberately descriptive. In particular, the study does not attempt to
infer from representation statistics alone whether the model has reached an architectural capacity
limit.
3Related Work
3.1Representation Similarity
Representation similarity analysis has become an important methodology for studying neural
networks. CKA was introduced as a robust similarity measure for comparing representations
and addressing limitations of conventional CCA-based approaches [Kornblith et al., 2019].
For centered activation matrices X and Y , linear CKA can be expressed as
CKA(X, Y ) =
∥X T Y ∥2F
.
∥X T X∥F ∥Y T Y ∥F
(1)
CKA values close to one indicate stronger representational similarity, whereas lower values
indicate weaker similarity under the selected representation and sample distribution.
2An important methodological point is that low CKA between two domains does not necessarily
mean that the corresponding knowledge is independent. Similarity is affected by the activation
distribution, preprocessing, sampling strategy, layer location, and the statistical properties of the
representation.
3.2
Dimensionality of Neural Representations
Ansuini et al. [Ansuini et al., 2019] showed that the intrinsic dimensionality of neural representa-
tions can be substantially smaller than the nominal number of hidden units. Their work also
demonstrated non-monotonic dimensionality trajectories across layers.
This motivates the use of rank-based diagnostics in the present study. Stable rank provides a
relatively simple measure of the effective spectral spread of an activation matrix.
3.3
Depth Scaling and Continued Pretraining
Depth up-scaling methods such as SOLAR demonstrate that increasing model depth through
layer-level transformations followed by continued pretraining can be an effective scaling strategy
[Kim et al., 2024]. However, whether such an intervention is appropriate for a particular model
cannot be inferred from a single representation statistic.
The present work therefore treats architectural expansion as a possible future intervention
rather than as a conclusion derived directly from CKA or stable rank.
4Methodology
4.1Framework Overview
The diagnostic framework contains four principal analytical components:
1. domain-level CKA;
2. linear-probe domain classification;
3. layer-wise stable-rank analysis;
4. far-layer CKA analysis.
Together, these measurements provide complementary information about representational
similarity, domain discriminability, and spectral structure.
4.2
Centered Kernel Alignment
For two centered activation matrices X, Y ∈ Rn×d , linear CKA is calculated as
CKA(X, Y ) =
∥X T Y ∥2F
.
∥X T X∥F ∥Y T Y ∥F
(2)
For cross-domain analysis, the activations of two different textual domains are compared at
the same model layer.
The resulting quantity should be interpreted as a measure of representational similarity rather
than as a direct measure of semantic independence or knowledge separation.
34.3
Linear Probe
A linear classifier is trained on the activation vectors to predict the source domain of each text.
Six classes are used: Aqeedah; Tafsir; Nahw and Sarf; General Fiqh; Hadith Mutoon; Fatawa.
Performance is reported using accuracy and macro-F1.
A high linear-probe score indicates that domain-discriminative information is accessible
through a linear transformation of the representation. It does not imply that the model explicitly
contains a symbolic domain classifier.
4.4
Stable Rank
For an activation matrix A with singular values σ1 , . . . , σr , stable rank is defined as
2
∥A∥2F
i σi
StableRank(A) =
=
.
2
σmax
∥A∥22
∑︁
(3)
Stable rank is bounded above by the ordinary matrix rank and provides a spectral measure
of effective dimensionality.
In this study, stable rank is reported both in absolute terms and relative to the hidden
dimension of 2560.
A low stable rank should not automatically be interpreted as either efficient representation
or insufficient model capacity. Both interpretations require additional evidence.
4.5
Far-Layer CKA
Far-layer CKA measures similarity between representations from layers that are separated by
substantial network depth.
The analysis includes:
• L00 ↔ L16;
• L00 ↔ L31;
• L04 ↔ L28;
• L08 ↔ L31.
The L05 ↔ L25 entries in the raw diagnostic output are zero because those layer activations
were not collected in the V10 layer set. They are therefore treated as unavailable rather than as
genuine evidence of zero similarity.
5Experimental Setup
5.1Model
The evaluated model is a Qwen3.5-based 4B-parameter language model continued-pretrained on
the Shamela corpus.
The architecture used in the diagnostic run has:
• hidden dimension: 2560;
• number of layers: 32;
• diagnostic layers: {0,4,8,12,16,20,24,28,31};
• final diagnostic layer: L31.
The publicly documented Qwen3.5-4B architecture specifies a 2560-dimensional hidden
representation and 32 layers, with its attention configuration containing 16 query attention heads
in the gated-attention component.
45.2
Continued Pretraining Data
Six domains from the Shamela collection were used:
Table 1: Domains used in the representation analysis.
Domain
Texts available
Aqeedah
Tafsir
Nahw and Sarf
General Fiqh
Hadith Mutoon
Fatawa100,000
100,000
100,000
100,000
100,000
100,000
Total600,000
The diagnostic collection itself used 800 activation vectors per domain and layer.
Thus, for every domain-layer pair, the activation matrix had dimensions 800 × 2560.
The activation vector corresponds to the selected final real token after attention processing,
according to the V10 collection procedure.
5.3
Activation Collection
Activations were collected from nine layers: L = {0, 4, 8, 12, 16, 20, 24, 28, 31}.
For each of the six domains, 800 vectors were collected at each diagnostic layer.
Consequently, the complete collection contained 6 × 9 × 800 = 43, 200 activation vectors.
The resulting matrices were retained separately by domain and layer to permit both within-
layer cross-domain analysis and within-domain cross-layer analysis.
5.4
Hardware and Inference
The V10 analysis was executed on a single NVIDIA RTX 3090 GPU with 24 GB of VRAM.
During the run, approximately 7.8 GB of GPU memory was allocated by the diagnostic
process.
Activation collection completed successfully over 1200 collection batches.
6Results
6.1Layer-wise Domain Discriminability
The strongest result is the progressive increase in linear-probe domain-discriminability across
depth.
5Table 2: Linear-probe performance across diagnostic layers.
LayerAccuracyMacro-F1SilhouetteDavies-Bouldin
0
4
8
12
16
20
24
28
3153.80%
70.28%
72.87%
76.76%
78.06%
79.54%
79.72%
77.69%
82.31%0.537
0.703
0.728
0.766
0.782
0.797
0.799
0.777
0.824-0.0075
-0.0060
-0.0112
-0.0089
-0.0068
-0.0029
-0.0110
-0.0124
0.003113.5336
11.1941
11.2660
11.0906
10.1647
8.1488
8.3116
8.4527
6.7378
Accuracy increases from 53.80% at layer 0 to 82.31% at layer 31. Macro-F1 follows a similar
trajectory, increasing from 0.537 to 0.824.
Interestingly, the progression is not strictly monotonic. Performance reaches 79.72% at L24,
decreases to 77.69% at L28, and then increases to 82.31% at L31.
This suggests that the final layer contains particularly accessible domain-discriminative
information, but the non-monotonic behavior indicates that representation development is more
complex than a simple linear increase in specialization.
6.2
Cross-Domain CKA at the Final Layer
Table 3 reports the cross-domain CKA matrix at L31.
Table 3: Cross-domain CKA matrix at layer 31.
Aqeedah
Tafsir
Nahw
Fiqh
Mutoon
Fatawa
AqeedahTafsirNahwFiqhMutoonFatawa
1.0000
0.0276
0.0258
0.0262
0.0312
0.03750.0276
1.0000
0.0228
0.0172
0.0293
0.02170.0258
0.0228
1.0000
0.0173
0.0254
0.02580.0262
0.0172
0.0173
1.0000
0.0213
0.02220.0312
0.0293
0.0254
0.0213
1.0000
0.03710.0375
0.0217
0.0258
0.0222
0.0371
1.0000
All off-diagonal CKA values are below 0.04.
These low values indicate that the activation distributions associated with the six domains
are substantially dissimilar at L31 under the present CKA measurement.
However, low CKA should not be interpreted as proof that the model stores knowledge in
independent compartments. CKA measures representational similarity, not causal independence
or resistance to interference.
6.3
Average Domain CKA Across Depth
A particularly notable result is the non-monotonic evolution of average cross-domain CKA.
6Table 4: Average cross-domain CKA by layer.
LayerAverage Domain CKA
0
4
8
12
16
20
24
28
310.054200
0.127515
0.148687
0.146753
0.130844
0.112298
0.116429
0.117917
0.025891
Average domain CKA rises from 0.0542 at L0 to a maximum of 0.1487 at L8. It then gradually
decreases and reaches 0.025891 at L31.
The sharp reduction between L28 and L31 is particularly notable because it occurs simulta-
neously with the highest linear-probe accuracy.
One possible descriptive interpretation is that the final representation makes domain-
discriminative information more accessible while becoming less similar across domains. However,
establishing whether this transition is functionally necessary would require intervention experi-
ments.
6.4
Stable Rank
The stable-rank trajectory is shown in Table 5.
Table 5: Layer-wise stable rank of activation representations.
LayerStable RankStable Rank / 2560
0
4
8
12
16
20
24
28
3110.593
20.324
24.854
23.687
18.208
16.453
20.457
18.448
5.7620.004138
0.007939
0.009709
0.009253
0.007113
0.006427
0.007991
0.007206
0.002251
Stable rank increases substantially during the early layers, reaching 24.854 at L8. It subse-
quently decreases, with a pronounced reduction at the final layer.
The L31 value of 5.762 is substantially lower than the nominal hidden dimension of 2560.
This result should be interpreted cautiously. Stable rank captures spectral structure, but it
cannot by itself distinguish between efficient compression, task-specific specialization, anisotropy,
or under-utilization of representational dimensions.
6.5
Far-Layer CKA
Far-layer CKA provides additional information about the evolution of representations within
each domain.
7Table 6: Far-layer CKA for the six domains.
DomainL00–L16L00–L31L04–L28L08–L31
Aqeedah
Tafsir
Nahw/Sarf
Fiqh
Mutoon
Fatawa0.58293
0.52570
0.53741
0.86112
0.55544
0.750680.34888
0.39176
0.37435
0.49430
0.34209
0.586180.59194
0.70091
0.65235
0.90041
0.65623
0.701060.43168
0.45480
0.50708
0.57715
0.41128
0.57495
The results show substantial within-domain similarity across distant layers. For example,
Fiqh reaches 0.90041 for L04–L28, while Fatawa reaches 0.75068 for L00–L16.
Importantly, these results also demonstrate why a single far-layer CKA value should not be
used to characterize the entire network. Similarity depends on the selected layers and domain.
The raw V10 output contains an L05–L25 value of 0.00000 for every domain. Because L5
and L25 were not included in the activation collection set, these values represent unavailable
measurements rather than observed zero similarity. They are therefore excluded from substantive
interpretation.
6.6
Integrated Layer Analysis
Figure 1 summarizes the principal V10 measurements.
Figure 1: V10 representation diagnostics across model depth. The panels show linear-probe
accuracy, macro-F1, stable rank, and average cross-domain CKA. The final layer exhibits the
highest domain-probe performance, the lowest stable rank, and the lowest average cross-domain
CKA.
The joint pattern is more informative than any single metric.
8At intermediate layers, stable rank is comparatively higher and cross-domain CKA is also
higher. At L31, domain discriminability reaches its maximum while both stable rank and
cross-domain CKA reach pronounced minima.
This convergence of three measurements makes the final layer particularly interesting for
further study.
7Discussion
7.1Increasing Domain Discriminability
The clearest empirical observation is that domain identity becomes increasingly linearly decodable
with depth.
The increase from 53.80% at L0 to 82.31% at L31 indicates that the final representations
contain substantially more linearly accessible information about the source discipline.
This is consistent with the general observation that deeper neural representations can become
increasingly specialized for task-relevant information.
Nevertheless, domain classification accuracy should not be equated with “knowledge sep-
aration.” A representation may encode domain identity for many reasons, including lexical
distributions, stylistic patterns, document structure, and semantic content.
Therefore, the present result is best described as: The model’s final-layer representations
contain strongly domain-discriminative information.
This statement is directly supported by the linear-probe results without requiring assumptions
about the internal organization of knowledge.
7.2
The Final-Layer Compression Pattern
The most interesting structural observation is the simultaneous occurrence of:
1. maximum linear-probe accuracy;
2. minimum average cross-domain CKA;
3. very low stable rank.
At L31, the model reaches 82.31% linear-probe accuracy while stable rank falls to 5.762 and
average cross-domain CKA falls to 0.025891.
One hypothesis is that the final representation compresses information into a smaller number
of directions that remain highly discriminative for the domains represented in the training data.
However, this remains a hypothesis.
An equally plausible alternative is that the final layer produces an anisotropic representation
in which only a small number of directions dominate the variance.
Distinguishing these interpretations requires additional measurements, including covariance
spectra, participation ratio, effective rank, intrinsic dimension, and comparisons against the
original base model.
7.3
Why Low CKA Does Not Establish Knowledge Compartmentalization
The low cross-domain CKA values are striking, but they should not be interpreted as evidence
that knowledge is stored in independent modules.
If two domains have different lexical, stylistic, and semantic distributions, their activations
can naturally become dissimilar even when they share large amounts of underlying knowledge.
Furthermore, CKA is a statistical similarity measure. It does not provide a causal intervention.
9Consequently, the present results support the weaker and more defensible claim that: The
model produces substantially different activation patterns for the six textual domains at its final
layer.
Whether those differences reduce interference during subsequent training is an open empirical
question.
7.4
Implications for Continued Pretraining
The results do not provide sufficient evidence to claim that the model has reached an architectural
saturation point.
In particular, stable rank of 5.762 does not imply that the model has “used only 5.762
dimensions” of its 2560-dimensional hidden state in a literal sense.
Likewise, low CKA does not establish that adding layers would be unnecessary.
Architectural decisions should instead be informed by a combination of:
• validation loss;
• per-domain validation loss;
• downstream task performance;
• representation trajectories across checkpoints;
• probe performance;
• forgetting measurements;
• comparisons with the original base model.
Depth up-scaling is therefore best considered a future experimental condition rather than a
recommendation derived from the current diagnostics.
8
Limitations
This study has several important limitations.
8.1
No Base-Model Comparison
The current analysis focuses on the continued-pretrained checkpoint and does not include the
original Qwen3.5-4B base model under identical diagnostic conditions.
Therefore, we cannot determine which representational changes are caused specifically by
continued pretraining on Shamela.
A direct base-versus-continued-pretraining comparison is necessary.
8.2
Single Main Checkpoint
The analysis focuses on checkpoint 97,500.
Although intermediate layers within this checkpoint were analyzed, the study does not yet
provide a trajectory across training checkpoints.
Consequently, the designation of a “saturation point” cannot be established from these data.
Future experiments should analyze checkpoints such as 10k, 30k, 50k, 75k, and 97.5k steps.
8.3
No Statistical Confidence Intervals
The reported linear-probe accuracy of 82.31% is based on a single diagnostic run.
Confidence intervals, bootstrap estimates, repeated splits, and multiple random seeds should
be reported in future experiments.
108.4
Representation Analysis Is Not Mechanistic Interpretability
The current methodology analyzes representation geometry and predictive information.
It does not identify causal mechanisms, circuits, neurons, attention heads, or interventions
that change model behavior.
For this reason, the paper deliberately uses the term representation-diagnostic rather than
claiming mechanistic interpretability in the strict sense.
8.5
No Direct Catastrophic-Forgetting Experiment
The current experiments do not test whether subsequent supervised fine-tuning would cause
catastrophic forgetting.
Low cross-domain CKA and high domain-probe accuracy cannot establish this.
A proper forgetting experiment would require domain-specific evaluation before and after
SFT or additional continued pretraining.
9
Future Work
Several experiments follow naturally from the present findings.
9.1
Checkpoint Trajectory
Future work will measure the same metrics across multiple checkpoints: 10k, 30k, 50k, 75k,
97.5k.
This would determine whether the final-layer rank collapse and CKA reduction emerge
progressively or occur only at a late stage.
9.2
Base-Model Comparison
The same diagnostic pipeline should be applied to the original Qwen3.5-4B model using matched
input samples.
The difference between base and continued-pretrained representations would provide a more
direct estimate of the effect of Shamela training.
9.3
Intrinsic Dimension
The current V10 analysis emphasizes stable rank. Future work should add Levina–Bickel
maximum-likelihood intrinsic-dimension estimation.
.
For k nearest neighbors, the estimator can be expressed as dˆ = ∑︁k−1k−1
i=1
log(rk /ri )
This would provide a complementary nonlinear estimate of representation dimensionality.
9.4
Effective Rank and Participation Ratio
Additional spectral measures should be computed to determine whether the stable-rank collapse
corresponds to broader spectral concentration.
In particular:
(︄
)︄
EffectiveRank(A) = exp −
∑︂
pi log pi ,
(4)
i
where pi = ∑︁σiσ .
j
j
Participation ratio can also be evaluated as
(
∑︁
σi )2
P R(A) = ∑︁i 2 .
i σi
11
(5)9.5
SFT and Forgetting Experiments
The current representation results motivate, but do not establish, an interesting downstream
hypothesis: Domain-discriminative representations may influence the extent to which domain-
specific SFT causes interference with other domains.
This hypothesis should be tested experimentally.
Performance should be measured on every domain before and after SFT, allowing a direct
forgetting matrix to be constructed.
9.6
Causal Representation Interventions
A stronger mechanistic interpretation would require interventions such as activation patching,
representation ablation, feature steering, or layer-specific causal mediation experiments.
Such experiments are outside the scope of the present study.
10
Conclusion
We presented a representation-diagnostic framework for studying how internal representations
evolve in an Arabic heritage language model continued-pretrained on the Shamela corpus.
The V10 analysis reveals three principal observations.
First, domain discriminability increases substantially with depth. Linear probe accuracy rises
from 53.80% at L0 to 82.31% at L31, while macro-F1 increases from 0.537 to 0.824.
Second, cross-domain representational similarity decreases strongly in the final layer. Average
domain CKA falls from values around 0.11–0.15 in the middle layers to 0.025891 at L31.
Third, stable rank exhibits a non-monotonic trajectory and reaches 5.762 at the final layer,
substantially below the nominal hidden dimension of 2560.
The combination of these results suggests that the final layer has a distinctive representational
regime characterized by high domain discriminability and strong spectral compression.
However, these observations should not be interpreted as proof of causal knowledge compart-
mentalization, architectural saturation, or protection against catastrophic forgetting.
The principal contribution of this work is therefore methodological: representation diagnostics
can reveal meaningful changes in the geometry and domain accessibility of Arabic heritage
language model representations, while also identifying questions that cannot be answered without
additional experiments.
Future work should focus on checkpoint trajectories, base-model comparisons, statistical
uncertainty, intrinsic-dimensionality analysis, and direct forgetting experiments. These extensions
will determine whether the representation patterns observed here correspond to functional
advantages during subsequent training and downstream adaptation.
Data and Code Availability
The diagnostic implementation and model artifacts are associated with the Shamela Builder
Project.
The continued-pretrained model checkpoint analyzed in this study is: https://huggingface.
co/sherif1313/3arabLM-4B-islamic-v2
The diagnostic outputs include:
• linear_probe_results.csv;
• stable_rank_results.csv;
• domain_cka_results.csv;
12• far_cka.csv;
• combined_layer_analysis.csv;
• layer_analysis_v10.png.
Acknowledgments
The author acknowledges the open-source Arabic NLP community and the developers of the
Shamela corpus and Qwen model family.
References
Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of Neural
Network Representations Revisited. In Proceedings of the 36th International Conference on
Machine Learning, volume 97, pages 3519–3529, 2019.
Alessio Ansuini, Alessandro Laio, Jakob H. Macke, and Davide Zoccolan. Intrinsic Dimension of
Data Representations in Deep Neural Networks. In Advances in Neural Information Processing
Systems, 2019.
Dahyun Kim, Sanghoon Kim, Chanjun Park, Wonsung Lee, Wonho Song, Yunsu Kim, Hyeonwoo
Kim, Yungi Kim, Hyeonju Lee, Jihoo Kim, Changbae Ahn, Seonghoon Yang, Sukyung Lee,
Hyunbyung Park, Gyoungjin Gim, Mikyoung Cha, Hwalsuk Lee, and Sunghun Kim. SOLAR
10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling. In
Proceedings of the 2024 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies, Industry Track, pages 23–35, 2024.
Qwen Team. Qwen3.5: Towards Native Multimodal Agents. Technical report and model
documentation, 2026.
Elizaveta Levina and Peter Bickel. Maximum Likelihood Estimation of Intrinsic Dimension from
Incomplete Nearest Neighbor Data. In Advances in Neural Information Processing Systems,
