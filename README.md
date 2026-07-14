# ALS Blood Transcriptomics Dashboard

An interactive Streamlit dashboard built from the manuscript **"Integrated Blood
Transcriptomic Analysis of ALS: Machine Learning Classification, Molecular
Subtyping, and Subtype-Specific Drug Repurposing Using GSE112676."**

## What's inside

- `app.py` — the full multi-page dashboard (12 sections, sidebar navigation)
- `assets/` — all 13 figures extracted from the source document
- `requirements.txt` — Python dependencies

## Sections

1. **Overview** — abstract, key metrics, pipeline flow
2. **Dataset & Preprocessing** — cohort composition, leak-free split, normalisation steps
3. **Differential Expression** — volcano plot, top DE probe selection
4. **ML Classification** — model performance, confusion matrix, ROC curve, hyperparameter tuning
5. **SHAP & Biomarkers** — top 20 SHAP-ranked genes, known vs. novel candidates
6. **PPI Network & Pathways** — STRING network, hub gene centrality, g:Profiler enrichment
7. **Molecular Subtyping** — k-means clustering, cluster quality metrics, subtype genes
8. **Drug Repurposing** — filterable FDA-stratified drug-gene candidate table
9. **External Validation** — cross-platform performance with/without ComBat
10. **Literature Landscape** — comparative table of related studies & addressed gaps
11. **Discussion & Limitations** — expandable sections mirroring the manuscript discussion
12. **References & Code** — key citations and Colab notebook link

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Notes

- All charts and tables are built from figures and numeric results reported directly
  in the source manuscript — no data has been fabricated or extrapolated beyond what
  is stated in the text.
- Figures without exact underlying numeric data (e.g. ROC curves for Random Forest/SVM/
  XGBoost, SHAP beeswarm plots) are shown as the original extracted images rather than
  reconstructed charts.
