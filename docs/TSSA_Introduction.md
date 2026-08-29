# Target-Side Semantic Anchoring for Low-Resource Neural Machine Translation

## Introduction

Neural machine translation (NMT) for minoritized and extremely low-resource languages faces severe data scarcity, lexical sparsity, and noisy parallel supervision [20, 25]. In settings such as Bahnaric–Vietnamese, Tay–Vietnamese, and Rhade–Vietnamese translation, standard sequence-to-sequence architectures are forced to optimize an unconstrained conditional generation objective from fragile, weakly pretrained source representations [3, 20]. This paradigm overlooks a fundamental structural asymmetry: while the source language lacks large-scale corpora, the target language is frequently supported by high-capacity monolingual or multilingual pretrained representations, such as BARTpho or multilingual BART [17, 20, 29]. Bilingual cognitive processing and cross-lingual lexical access models demonstrate that conceptual retrieval is inherently asymmetric, allowing a rich lexical-semantic system to scaffold weaker linguistic inputs [7, 15, 23, 24]. Consequently, an essential research question emerges: can a rich, pretrained target-language model serve as a training-time latent teacher to organize and regularize source representations before and during translation?

Despite the prevalence of low-resource scenarios, existing methodologies struggle to exploit this target-side advantage without introducing severe bottlenecks [12]. Back-translation leverages target-side monolingual text but requires an auxiliary target-to-source model; in low-resource regimes, this reverse generator is fragile and propagates synthetic errors into the training pipeline [26]. Pivot translation routes generation through intermediary languages, compounding error accumulation, domain mismatch, and decoding latency across sequential translation hops. Furthermore, standard sequence-to-sequence cross-attention mechanisms cannot be treated as true cross-lingual word alignment, frequently failing to prevent omissions, hallucinations, and catastrophic semantic drift under limited bitext [11, 19, 22, 30]. While bilingual lexicon induction and representation alignment have advanced from static linear mappings to dynamic subspaces and contextual embeddings [1, 2, 13, 14, 16], directly transferring structured target geometry into encoder representations during end-to-end NMT training remains an open problem.

To overcome these structural limitations, the goal of this work is to establish an asymmetric, training-time semantic scaffolding framework that directly aligns source representations to a rich target semantic space without introducing intermediate models or test-time inference overhead. Translating between typologically diverse, low-resource language pairs requires three stringent conditions: (i) the target teacher must act as a stable geometric anchor rather than a moving co-adaptation target; (ii) token-level alignment must remain robust against imperfect, noisy external posteriors without mistaking soft alignments for gold linguistic boundaries [4, 9, 31]; and (iii) the target teacher must be consulted strictly during training, guaranteeing zero target-reference leakage or latency degradation during autoregressive inference.

Realizing this framework presents three central technical challenges. First, at the token level, word-alignment posteriors derived from external aligners are inherently noisy and incomplete in low-resource regimes; unconstrained cross-lingual projection risks collapsing source geometries or over-regularizing toward spurious alignments [4, 5, 8, 10]. Second, at the sentence level, divergent word orders, dialectal variations, and rare morphological forms frequently lack explicit token correspondences, demanding a global semantic objective that preserves sequence-level intent without suffering from false contrastive negatives [1, 28]. Third, at the decoding level, conventional cross-attention indiscriminately attends across all source positions; dynamically routing attention to selectively transmit anchor-consistent representations requires lightweight, inference-compatible gating that does not overfit teacher-forced alignment states [9, 30].

In this paper, we propose Target-Side Semantic Anchoring (TSSA), a unified framework that shapes the source encoder toward a frozen target teacher while dynamically routing cross-attention channels. To address noisy token links, TSSA introduces confidence-weighted structural anchoring ($\mathcal{L}_{\text{struct}}$), projecting source states toward target barycenters with a stop-gradient operator to stabilize the reference anchor geometry. To ensure sentence-level coherence across unaligned forms, TSSA incorporates in-batch semantic priming ($\mathcal{L}_{\text{prime}}$), optimizing a temperature-scaled contrastive loss over mask-aware pooled representations. Finally, to filter cross-attention evidence, TSSA deploys lightweight head-wise routing ($\mathcal{L}_{\text{route}}$) in the decoder, supervising scalar attention gates with detached teacher agreement during training while operating entirely on inference-available signals during decoding.

We summarize our primary contributions as follows:
- **Asymmetric Anchoring Framework**: We formalize the principle of target-side semantic anchoring for low-resource NMT, treating the target language as a training-time latent teacher to regularize source representations without requiring reverse models, pivot systems, or target-reference inputs at inference.
- **Unified Training Objective**: We design a three-part objective coupling confidence-weighted barycentric token anchoring, sentence-level semantic priming, and inference-compatible head-wise routing with detached reliability supervision.
- **Extensive Controlled Evaluation**: We evaluate TSSA across three low-resource language pairs (Bahnaric, Tay, and Rhade translating to Vietnamese) under multiple data budgets, demonstrating consistent improvements over five baseline families in translation adequacy, COMET, chrF++, and alignment precision.
- **Mechanistic and Causal Validation**: We conduct factorial ablations, causal head-pruning experiments, and controlled perturbation tests, confirming that performance gains stem specifically from target semantic anchoring rather than parameter scaling or generic regularization.

---

## References

[1] Mikel Artetxe, Gorka Labaka, and Eneko Agirre. A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 789–798, 2018.

[2] Mikel Artetxe, Gorka Labaka, and Eneko Agirre. Bilingual lexicon induction through unsupervised machine translation. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 5002–5007, 2019.

[3] Mengyu Bu, Shuhao Gu, and Yang Feng. Improving multilingual neural machine translation by utilizing semantic and linguistic features. In *Findings of the Association for Computational Linguistics: ACL 2024*, pages 10410–10423, 2024.

[4] Chi Chen, Maosong Sun, and Yang Liu. Mask-align: Self-supervised neural word alignment. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 4781–4791, 2021.

[5] Yun Chen, Yang Liu, Guanhua Chen, Xin Jiang, and Qun Liu. Accurate word alignment induction from neural machine translation. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 566–576, 2020.

[6] Noam Chomsky. A review of B. F. Skinner's Verbal Behavior. *Language*, 35(1):26–58, 1959.

[7] Ton Dijkstra and Walter J. B. Van Heuven. The architecture of the bilingual word recognition system: From identification to decision. *Bilingualism: Language and Cognition*, 5(3):175–197, 2002.

[8] Zi-Yi Dou and Graham Neubig. Word alignment by fine-tuning embeddings on parallel corpora. In *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume*, pages 2112–2128, 2021.

[9] Benedikt Ebing, Christian Goldschmied, and Goran Glavaš. TransAlign: Machine translation encoders are strong word aligners, too. In *Findings of the Association for Computational Linguistics: EMNLP 2025*, pages 20736–20749, 2025.

[10] Sarthak Garg, Stephan Peitz, Udhyakumar Nallasamy, and Matthias Paulik. Jointly learning to align and translate with transformer models. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 4453–4462, 2019.

[11] Hamidreza Ghader and Christof Monz. What does attention in neural machine translation pay attention to? In *Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 30–39, 2017.

[12] Katharina Hämmerl, Jindřich Libovický, and Alexander Fraser. Understanding cross-lingual alignment—a survey. In *Findings of the Association for Computational Linguistics: ACL 2024*, pages 10922–10943, 2024.

[13] Ling Hu and Yuemei Xu. DM-BLI: Dynamic multiple subspaces alignment for unsupervised bilingual lexicon induction. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 2041–2052, 2024.

[14] Masoud Jalili Sabet, Philipp Dufter, François Yvon, and Hinrich Schütze. SimAlign: High quality word alignments without parallel training data using static and contextualized embeddings. In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pages 1627–1643, 2020.

[15] Judith F. Kroll and Erika Stewart. Category interference in translation and picture naming: Evidence for asymmetric connections between bilingual memory representations. *Journal of Memory and Language*, 33(2):149–174, 1994.

[16] Guillaume Lample, Alexis Conneau, Marc'Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. Word translation without parallel data. In *International Conference on Learning Representations (ICLR)*, 2018.

[17] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 7871–7880, 2020.

[18] Ping Li, Jennifer Legault, and Kaitlyn A. Litcofsky. Neuroplasticity as a function of second language learning: Anatomical changes in the human brain. *Cortex*, 58:301–324, 2014.

[19] Xintong Li, Guanlin Li, Lemao Liu, Max Meng, and Shuming Shi. On the word alignment from neural machine translation. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 1293–1303, 2019.

[20] Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, and Luke Zettlemoyer. Multilingual denoising pre-training for neural machine translation. *Transactions of the Association for Computational Linguistics*, 8:726–742, 2020.

[21] Matthew W. Lowder and Fernanda Ferreira. Prediction in the processing of repair disfluencies. *Language, Cognition and Neuroscience*, 31(1):73–79, 2016.

[22] Chunpeng Ma, Akihiro Tamura, Masao Utiyama, Tiejun Zhao, and Eiichiro Sumita. Encoder-decoder attention $\neq$ word alignment: Axiomatic method of learning word alignments for neural machine translation. *Journal of Natural Language Processing*, 27(3):531–552, 2020.

[23] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S. Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In *Advances in Neural Information Processing Systems (NeurIPS)*, volume 26, 2013.

[24] Tomáš Mikolov, Wen-tau Yih, and Geoffrey Zweig. Linguistic regularities in continuous space word representations. In *Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)*, pages 746–751, 2013.

[25] Rico Sennrich and Biao Zhang. Revisiting low-resource neural machine translation: A case study. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 211–221, 2019.

[26] Rico Sennrich, Barry Haddow, and Alexandra Birch. Improving neural machine translation models with monolingual data. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 86–96, 2016.

[27] Freda Shi, Luke Zettlemoyer, and Sida I. Wang. Bilingual lexicon induction via unsupervised bitext construction and word alignment. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 813–826, 2021.

[28] Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. MASS: Masked sequence to sequence pre-training for language generation. In *Proceedings of the 36th International Conference on Machine Learning (ICML)*, pages 5926–5936, 2019.

[29] Yuqing Tang, Chau Tran, Xian Li, Peng-Jen Chen, Naman Goyal, Vishrav Chaudhary, Jiatao Gu, and Angela Fan. Multilingual translation with extensible multilingual pretraining and finetuning. *arXiv preprint arXiv:2008.00401*, 2020.

[30] Qiyu Wu, Masaaki Nagata, Zhongtao Miao, and Yoshimasa Tsuruoka. Word alignment as preference for machine translation. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 3223–3239, 2024.

[31] Thomas Zenkel, Joern Wuebker, and John DeNero. End-to-end neural word alignment outperforms GIZA++. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 1605–1617, 2020.
