# Citation Hypergraphs and the Structure of Legal Argument

**Moses Boudourides**  
School of Professional Studies, Northwestern University  
moses.boudourides@northwestern.edu

---

## Abstract

The "seamless web" of law is frequently invoked as a metaphor for legal interconnectedness, yet computational models of precedent typically reduce judicial synthesis to dyadic citation networks. Building on recent advances in legal hypergraphs and network models of legal relations, we introduce a directed F-hypergraph framework to model the precedential citation practices of the United States Supreme Court (1791–2024). In this framework, a citation act is represented as a directed hyperedge (an F-arc) connecting a citing opinion to a simultaneous bundle of cited precedents. We construct an empirical F-hypergraph comprising 29,121 opinions and 18,091 timed F-arcs, and compute a five-dimensional structural profile — closure, brokerage, authority concentration, temporal span, and community dispersion — for each citation act. Analyzing the temporal trajectory of these metrics reveals a long-term historical shift from highly brokered, low-closure citation structures in the founding era toward denser, highly clustered synthesis in the modern Court, accompanied by an accelerating concentration of authority around canonical super-precedents. Embedding these profiles into a Legal Argument Space (LAS) via UMAP yields six distinct typologies of legal synthesis. We validate the hypergraph representation through a missing-precedent prediction task, demonstrating that structural profiles achieve a Recall@1 of 93.8% and an ROC-AUC of 0.994, dramatically outperforming the standard in-degree heuristic (Recall@1 = 1.8%, ROC-AUC = 0.786).

---

## Contents

| File | Description |
|---|---|
| [`main.pdf`](main.pdf) | Full manuscript (preprint) |
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

## Citation

If you use this work, please cite:

```bibtex
@article{boudourides2026citation,
  author  = {Boudourides, Moses},
  title   = {Citation Hypergraphs and the Structure of Legal Argument},
  year    = {2026},
  note    = {Preprint}
}
```

---

## License

Copyright (c) 2026 Moses Boudourides. This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
