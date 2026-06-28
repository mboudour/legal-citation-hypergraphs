# Citation Hypergraphs and the Structure of Legal Argument

**Moses Boudourides**

---

## Abstract

The "seamless web" of law is frequently invoked as a metaphor for legal interconnectedness, yet computational models of precedent typically reduce judicial synthesis to dyadic citation networks. Building on recent advances in legal hypergraphs and network models of legal relations, we introduce a directed F-hypergraph framework to model the precedential citation practices of the United States Supreme Court (1791–2024). In this framework, a citation act is represented as a directed hyperedge (an F-arc) connecting a citing opinion to a simultaneous bundle of cited precedents. We construct an empirical F-hypergraph comprising 29,121 opinions and 18,091 timed F-arcs, and compute a five-dimensional structural profile — closure, brokerage, authority concentration, temporal span, and community dispersion — for each citation act. Analyzing the temporal trajectory of these metrics reveals a long-term historical shift from highly brokered, low-closure citation structures in the founding era toward denser, highly clustered synthesis in the modern Court, accompanied by an accelerating concentration of authority around canonical super-precedents. Embedding these profiles into a Legal Argument Space (LAS) via UMAP yields six distinct typologies of legal synthesis. We validate the hypergraph representation through a missing-precedent prediction task, demonstrating that structural profiles achieve a Recall@1 of 93.8% and an ROC-AUC of 0.994, dramatically outperforming the standard in-degree heuristic (Recall@1 = 1.8%, ROC-AUC = 0.786).

---

## Key Findings

### 1. The Dyadic Model is Structurally Insufficient

Standard citation network models record individual edges from a citing case to each cited case, but discard the relational structure of the citation bundle as a whole. The F-hypergraph representation preserves the bundle as a single, cohesive act of legal synthesis — an F-arc $(u, H, \tau)$ — and reveals structural information that is entirely invisible to dyadic models. The missing-precedent validation task quantifies this gap precisely: the structural profile model achieves a **52-fold improvement** in Recall@1 over the best dyadic heuristic (in-degree).

### 2. A Long-Term Historical Transition in Legal Argumentation

The temporal trajectory of the five structural metrics reveals a coherent, two-century-long shift in how the Supreme Court synthesizes precedent:

- **Brokerage** has declined steadily from near-1.0 in the founding era to approximately 0.88 in the modern Court, indicating that citation bundles increasingly draw on precedents that already cite each other.
- **Closure** has risen correspondingly from near-zero to approximately 0.12, reflecting the growing density of pre-existing connections among cited cases.
- This transition accelerated sharply at the turn of the twentieth century, coinciding with the rise of legal formalism under Langdell and the Harvard Law School tradition — providing, for the first time, a **quantitative signature of the formalist turn** in American jurisprudence.

### 3. The Modern Court and the Rise of the Super-Precedent

- **Authority concentration** (measured as the Gini coefficient of in-degrees among cited cases) has accelerated dramatically since the 1980s, reaching its historical peak in the 2010s.
- Modern arguments increasingly anchor themselves to a small number of towering decisions (*Chevron*, *Strickland*, *Miranda*, *Roe*) while supplementing them with minor cases for specific factual analogies.
- **Temporal span** has grown in parallel, with the modern Court routinely connecting founding-era principles with contemporary jurisprudence — a structural signature consistent with the rise of originalism as a constitutional methodology.

### 4. Six Typologies of Legal Synthesis

Projecting the 18,091 F-arc profiles into a two-dimensional Legal Argument Space (LAS) via UMAP and clustering with a Gaussian Mixture Model yields six structurally distinct typologies:

| Typology | Dominant Features | Historical Peak |
|---|---|---|
| Foundational Synthesis | Near-zero closure, near-perfect brokerage | Founding era (pre-1850) |
| Doctrinal Consolidation | Moderate closure, medium head size | Continuous throughout |
| Landmark Synthesis | High authority concentration, large head size | Post-1950 |
| Cross-Doctrinal Bridging | High community dispersion, high brokerage | Warren/Burger Courts (1960s–70s) |
| Deep Historical Synthesis | Largest temporal span | Growing since 1980s |
| Routine Application | Smallest head size, lowest complexity | Continuous throughout |

The relative prevalence of these typologies has shifted dramatically over the Court's history, from near-total dominance of Foundational Synthesis in the early republic to a pluralistic mix in the modern era.

### 5. Validation: Structural Profiles Predict Legal Relevance

A rigorous missing-precedent prediction task — withholding one case from each F-arc and asking the model to identify it from a pool of 100 candidates — demonstrates that the structural profiles capture genuine legal relevance:

| Model | Recall@1 | ROC-AUC |
|---|---|---|
| Structural Profile (F-hypergraph) | **93.8%** | **0.994** |
| In-degree baseline (dyadic) | 1.8% | 0.786 |

The 52-fold improvement in Recall@1 confirms that the polyadic structure of citation bundles — how the precedents relate to each other within the specific context of the citing opinion — is far more informative than global citation popularity.

### 6. Quantitative Signatures of Constitutional History

Several well-known episodes in constitutional history leave clear structural signatures in the data:

- A sharp increase in community dispersion in the **1960s** (Warren Court's expansion of constitutional rights across doctrinal domains).
- A notable dip in closure in the **1930s** (New Deal era reconsidering the scope of federal power).
- A sharp increase in temporal span beginning in the **1980s** (rise of originalism under Scalia and Thomas).

---

## Contents

| File | Description |
|---|---|
| [`main.pdf`](main.pdf) | Full manuscript (preprint, 18 pages) |
| [`figures/fhypergraph_scotus.pdf`](figures/fhypergraph_scotus.pdf) | F-hypergraph example: prisoner civil-rights citation chain |
| [`figures/closure_brokerage_combined.pdf`](figures/closure_brokerage_combined.pdf) | Closure vs. brokerage over SCOTUS history |
| [`figures/authority_span_decade.pdf`](figures/authority_span_decade.pdf) | Authority concentration and temporal span per decade |
| [`figures/temporal_trajectory_full.pdf`](figures/temporal_trajectory_full.pdf) | Full five-metric temporal trajectory |
| [`figures/head_size_violin.pdf`](figures/head_size_violin.pdf) | Citation bundle size distribution per decade |
| [`figures/cluster_fraction_decade.pdf`](figures/cluster_fraction_decade.pdf) | Legal Argument Space typology mix per decade |
| [`figures/las_pca.png`](figures/las_pca.png) | LAS UMAP projection with cluster colours |
| [`figures/validation_topk.png`](figures/validation_topk.png) | Missing Precedent Recall@K validation |

---

## Data

Raw opinion data is publicly available from the [Free Law Project's CourtListener](https://www.courtlistener.com/) database. Processed F-hypergraph data and analysis code will be released upon acceptance of the manuscript.

---

## License

Copyright (c) 2026 Moses Boudourides. This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
