"""
Integrated Blood Transcriptomic Analysis of ALS — Interactive Dashboard
Machine Learning Classification, Molecular Subtyping & Drug Repurposing (GSE112676)
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLE
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="ALS Blood Transcriptomics Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASSETS = Path(__file__).parent / "assets"

PRIMARY = "#7C3AED"
ACCENT = "#06B6D4"
ALS_COLOR = "#EF4444"
CTRL_COLOR = "#3B82F6"
BG_CARD = "#141824"

st.markdown(f"""
<style>
    .stApp {{ background-color: #0B0E14; }}
    h1, h2, h3, h4 {{ color: #F1F5F9 !important; font-family: 'Georgia', serif; }}
    p, li, span, label, .stMarkdown {{ color: #CBD5E1; }}
    [data-testid="stSidebar"] {{ background-color: #10141F; border-right: 1px solid #232838; }}
    .metric-card {{
        background: linear-gradient(145deg, #171B29, #10131C);
        border: 1px solid #262B3D;
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
    }}
    .metric-value {{ font-size: 28px; font-weight: 700; color: {ACCENT}; margin-bottom: 2px;}}
    .metric-label {{ font-size: 12.5px; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em;}}
    .section-tag {{
        display:inline-block; background: rgba(124,58,237,0.15); color: {PRIMARY};
        padding: 3px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
        letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 8px; border: 1px solid rgba(124,58,237,0.4);
    }}
    .callout {{
        background: rgba(6,182,212,0.08); border-left: 3px solid {ACCENT};
        padding: 12px 16px; border-radius: 6px; font-size: 14.5px; color:#E2E8F0;
    }}
    .warn {{
        background: rgba(239,68,68,0.08); border-left: 3px solid {ALS_COLOR};
        padding: 12px 16px; border-radius: 6px; font-size: 14.5px; color:#E2E8F0;
    }}
    .gene-chip {{
        display:inline-block; background:#1B2130; border:1px solid #2E3448; color:#E2E8F0;
        padding: 4px 12px; border-radius: 8px; margin: 3px; font-size: 13px; font-family: monospace;
    }}
    hr {{ border-color: #232838; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{ background-color: #141824; border-radius: 8px 8px 0 0; padding: 8px 16px;}}
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
def style_fig(fig, height=420):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1", family="Helvetica"),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    return fig

def metric_card(value, label):
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{value}</div>
    <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

def tag(text):
    st.markdown(f'<span class="section-tag">{text}</span>', unsafe_allow_html=True)

def img(name, caption=None, width=None):
    path = ASSETS / name
    if path.exists():
        st.image(str(path), caption=caption, width=width, use_container_width=(width is None))
    else:
        st.warning(f"Image {name} not found.")

# ----------------------------------------------------------------------------
# DATA (all figures sourced directly from the manuscript)
# ----------------------------------------------------------------------------

TOP20_BIOMARKERS = [
    dict(gene="ABCA1", category="Lipid metabolism", note="Cholesterol efflux transporter; dysregulated cholesterol homeostasis affects axonal membrane integrity & synaptic vesicle recycling.", status="Known"),
    dict(gene="SLC25A20", category="Mitochondrial", note="Mitochondrial carnitine transporter; relevant to bioenergetic dysfunction in ALS.", status="Known"),
    dict(gene="BORCS7", category="Vesicular trafficking", note="BORC complex component controlling lysosomal positioning; interacts with C9orf72, the most common ALS genetic cause.", status="Known"),
    dict(gene="SRPK1", category="RNA metabolism", note="Splicing kinase that directly regulates TDP-43 subcellular localisation and aggregation; SRPK1 inhibition is an emerging ALS therapeutic strategy.", status="Known"),
    dict(gene="TNFRSF10B", category="Apoptotic signalling", note="TRAIL receptor 2; mediates extrinsic apoptotic death-receptor signalling, elevated in ALS peripheral blood.", status="Known"),
    dict(gene="PAK2", category="Cytoskeletal / axonal transport", note="Involved in cytoskeletal dynamics and axonal transport machinery.", status="Known"),
    dict(gene="NDFIP1", category="Ubiquitin-proteasome", note="NEDD4-family adaptor; ubiquitin-mediated degradation of pro-apoptotic substrates. Principal PPI network hub gene.", status="Known"),
    dict(gene="DNAJB14", category="Protein quality control", note="Molecular co-chaperone implicated in ALS-relevant proteostasis pathways.", status="Known"),
    dict(gene="QPCT", category="Unclassified", note="No prior ALS-specific literature support — flagged as a novel candidate biomarker requiring experimental validation.", status="Novel"),
    dict(gene="SAR1B", category="Vesicular trafficking", note="COPII vesicle GTPase; secondary PPI hub gene. No prior ALS-specific literature support — novel candidate.", status="Novel"),
    dict(gene="SLC44A1", category="Unclassified", note="Not present in curated ALS causative gene panels — promising novel candidate identified via SHAP.", status="Novel"),
    dict(gene="HIBADH", category="Metabolic", note="Not present in curated ALS causative gene panels — promising novel candidate identified via SHAP.", status="Novel"),
    dict(gene="SAMSN1", category="Immune signalling", note="Not present in curated ALS causative gene panels — promising novel candidate identified via SHAP.", status="Novel"),
    dict(gene="NT5DC1", category="Metabolic", note="Not present in curated ALS causative gene panels — promising novel candidate identified via SHAP.", status="Novel"),
    dict(gene="ARL5B", category="Vesicular trafficking", note="Secondary PPI hub gene (degree centrality 0.053).", status="Known"),
]

PPI_HUBS = pd.DataFrame([
    dict(gene="NDFIP1", degree=0.105, betweenness=0.006, role="Principal hub — ubiquitin-mediated protein QC"),
    dict(gene="SAR1B", degree=0.053, betweenness=0.0, role="Secondary hub — vesicular trafficking"),
    dict(gene="TNFRSF10B", degree=0.053, betweenness=0.0, role="Secondary hub — apoptotic signalling"),
    dict(gene="BORCS7", degree=0.053, betweenness=0.0, role="Secondary hub — lysosomal positioning"),
    dict(gene="ARL5B", degree=0.053, betweenness=0.0, role="Secondary hub — vesicular trafficking"),
])

CLUSTER_METRICS = pd.DataFrame([
    dict(k=2, silhouette=0.478, ch_index=334.6),
    dict(k=3, silhouette=0.382, ch_index=297.9),
    dict(k=4, silhouette=0.351, ch_index=292.1),
])

CLUSTER1_GENES = ["FEZ2", "FAM122B", "FBXO11", "MICU2", "ARGLU1", "MGAT4A", "TBPL1"]

DRUG_TABLE = pd.DataFrame([
    dict(target="FEZ2", drug="Lithium Chloride", status="FDA approved", area="Neurological"),
    dict(target="FEZ2", drug="Valproic Acid", status="FDA approved", area="Neurological"),
    dict(target="FAM122B", drug="Okadaic Acid", status="Research", area="Cell signalling"),
    dict(target="FBXO11", drug="MLN4924", status="Clinical trial", area="Oncology"),
    dict(target="MICU2", drug="Ruthenium Red", status="Research", area="Mitochondrial"),
    dict(target="MICU2", drug="DS16570511", status="Preclinical", area="Mitochondrial"),
    dict(target="MGAT4A", drug="Swainsonine", status="Preclinical", area="Glycobiology"),
    dict(target="MGAT4A", drug="2-DG", status="Clinical trial", area="Metabolic"),
    dict(target="TBPL1", drug="Triptolide", status="Preclinical", area="Anti-inflammatory"),
    dict(target="ALG6", drug="Tunicamycin", status="Research", area="ER stress"),
    dict(target="ALG6", drug="4-PBA", status="FDA approved", area="Metabolic"),
    dict(target="FOPNL", drug="Taxol", status="FDA approved", area="Oncology"),
    dict(target="CD58", drug="Efalizumab", status="Research", area="Immunotherapy"),
    dict(target="CD58", drug="Natalizumab", status="FDA approved", area="Neurological"),
])

VALIDATION_TABLE = pd.DataFrame([
    dict(Metric="AUC", Internal_GSE112676=0.944, External_no_ComBat=0.758, External_ComBat=0.732),
    dict(Metric="Accuracy", Internal_GSE112676=0.893, External_no_ComBat=0.671, External_ComBat=0.628),
])

LIT_TABLE = pd.DataFrame([
    dict(no=1, author="Mougeot et al. (2011)", work="Microarray profiling of ALS peripheral blood lymphocytes", ml="Statistical DE analysis", gap="No ML classification or biomarker ranking", addressed="Supervised ML classification with SHAP-based biomarker ID"),
    dict(no=2, author="Saris et al. (2013)", work="Cross-cohort whole blood transcriptomic comparison in ALS", ml="Statistical analysis", gap="No predictive model built or externally validated", addressed="Logistic Regression classifier with external validation on independent cohort"),
    dict(no=3, author="van Rheenen et al. (2018)", work="Two-stage whole blood transcriptome-wide ALS biomarker study", ml="Statistical DE analysis", gap="No ML classification, subtyping, or drug repurposing", addressed="Integrated ML pipeline with subtyping and FDA-stratified drug repurposing"),
    dict(no=4, author="Dimitriu et al. (2021)", work="Systematic evaluation of ML approaches in ALS transcriptomics", ml="LR, RF, SVM, XGBoost", gap="Data leakage in preprocessing and feature selection bias", addressed="Strictly leak-free pipeline with within-CV feature selection"),
    dict(no=5, author="Prudencio et al. (2015)", work="Transcriptomic subtyping of ALS spinal cord tissue", ml="Unsupervised clustering", gap="Spinal cord inaccessible during life; no blood-based subtyping", addressed="Unsupervised k-means subtyping of ALS patients in whole blood"),
    dict(no=6, author="Blasco et al. (2014)", work="Blood and CSF biomarker analysis in ALS", ml="Statistical analysis", gap="No integrated classification and biological interpretation pipeline", addressed="SHAP interpretability linking classifier predictions to biological mechanisms"),
    dict(no=7, author="Pushpakom et al. (2019)", work="Review of computational drug repurposing methodologies", ml="Literature review", gap="No ALS-specific transcriptomics-driven repurposing with FDA stratification", addressed="Cluster-specific drug repurposing stratified by FDA approval status"),
    dict(no=8, author="Johnson et al. (2007)", work="ComBat batch correction for microarray meta-analysis", ml="Empirical Bayes", gap="Batch correction not applied in ALS cross-platform validation studies", addressed="ComBat applied for V3/V4 cross-platform harmonisation, transparently reported"),
    dict(no=9, author="Lundberg & Lee (2017)", work="SHAP unified framework for model interpretation", ml="Game theory", gap="SHAP not applied to ALS blood transcriptomics classification", addressed="SHAP LinearExplainer applied for exact gene-level attribution"),
    dict(no=10, author="Freshour et al. (2021)", work="DGIdb 4.0 drug-gene interaction database", ml="Database curation", gap="DGIdb not applied to ALS molecular subtype-specific gene targets", addressed="DGIdb queried against cluster-specific upregulated genes for subtype drug repurposing"),
])

PATHWAYS = [
    dict(name="RNA binding & processing", detail="Consistent with TDP-43 / FUS RNA metabolism dysfunction — the defining pathological hallmark of ALS."),
    dict(name="Ubiquitin-dependent protein degradation", detail="Corroborates SHAP-identified proteasomal pathway importance (NDFIP1 hub gene)."),
    dict(name="Mitochondrial function & oxidative stress", detail="Reflects bioenergetic failure, echoed in blood cells and peripheral tissue."),
    dict(name="Intracellular transport & vesicular trafficking", detail="Aligned with BORCS7 / SAR1B / ARL5B hub gene functions."),
    dict(name="KEGG: Neurodegeneration & apoptotic signalling", detail="Independent pathway-level validation of the DE gene set (TNFRSF10B)."),
]

# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧬 ALS Blood Transcriptomics")
    st.caption("GSE112676 · Integrated ML Pipeline")
    page = st.radio(
        "Navigate",
        [
            "🏠  Overview",
            "🧪  Dataset & Preprocessing",
            "📈  Differential Expression",
            "🤖  ML Classification",
            "🔍  SHAP & Biomarkers",
            "🕸️  PPI Network & Pathways",
            "🧩  Molecular Subtyping",
            "💊  Drug Repurposing",
            "🌍  External Validation",
            "📚  Literature Landscape",
            "🗒️  Discussion & Limitations",
            "📖  References & Code",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:12px; color:#64748B;'>"
        "Dashboard generated from the manuscript "
        "<i>'Integrated Blood Transcriptomic Analysis of ALS: Machine Learning "
        "Classification, Molecular Subtyping, and Subtype-Specific Drug Repurposing "
        "Using GSE112676'</i>.</div>",
        unsafe_allow_html=True,
    )

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if page.startswith("🏠"):
    st.markdown("<div class='section-tag'>Whole-Blood Microarray · n = 741 + 301 external</div>", unsafe_allow_html=True)
    st.title("Integrated Blood Transcriptomic Analysis of ALS")
    st.markdown("#### Machine Learning Classification, Molecular Subtyping & Subtype-Specific Drug Repurposing")
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("0.944", "Test AUC (LR)")
    with c2: metric_card("89.3%", "Test Accuracy")
    with c3: metric_card("0.758", "External AUC")
    with c4: metric_card("2", "ALS Subtypes")
    with c5: metric_card("43", "Drug Candidates")

    st.write("")
    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("Abstract")
        st.markdown("""
Amyotrophic lateral sclerosis (ALS) is a fatal motor neuron disease with rapid progression
and no validated diagnostic biomarkers. This study builds an **end-to-end, leak-free**
machine learning and bioinformatics pipeline on whole blood microarray data from
**GSE112676** (n = 741; 233 ALS, 508 controls; Illumina HumanHT-12 V3).

After log2 transformation, quantile normalisation, and FDR-corrected differential
expression analysis, the **top 1,000 differentially expressed probes** trained four
classifiers — Logistic Regression, Random Forest, SVM, and XGBoost. **Logistic
Regression (C = 0.1)** was the top performer (Test AUC **0.944**, Accuracy **89.3%**).

Predictions were interpreted with **SHAP**, and top genes were mapped onto a
**STRING PPI network** (hub gene: **NDFIP1**) and enriched via **g:Profiler**.
Unsupervised **k-means clustering** on ALS-only samples revealed **two molecular
subtypes** (Cluster 1: n=121, Cluster 2: n=112), each triaged for **FDA-stratified
computational drug repurposing**. **External cross-platform validation** on GSE112680
(n=301, HT-12 V4) achieved AUC **0.758** (uncorrected) / **0.732** (ComBat-corrected),
confirming generalisability with expected cross-platform attenuation.
        """)
    with right:
        img("image1.png", caption="Graphical summary of the end-to-end analytical pipeline")

    st.markdown("---")
    st.subheader("Pipeline at a Glance")
    stages = ["Data\nAcquisition", "Leak-free\nPreprocessing", "Differential\nExpression",
              "ML\nClassification", "SHAP\nExplainability", "PPI + Pathway\nEnrichment",
              "Molecular\nSubtyping", "Drug\nRepurposing", "External\nValidation"]
    fig = go.Figure()
    xs = list(range(len(stages)))
    fig.add_trace(go.Scatter(
        x=xs, y=[1]*len(stages), mode="markers+text",
        marker=dict(size=34, color=ACCENT, line=dict(width=2, color=PRIMARY)),
        text=[str(i+1) for i in xs], textfont=dict(color="white", size=13),
        hovertext=stages, hoverinfo="text"
    ))
    fig.add_trace(go.Scatter(x=xs, y=[1]*len(stages), mode="lines",
                              line=dict(color="#2E3448", width=3), hoverinfo="skip"))
    for i, s in enumerate(stages):
        fig.add_annotation(x=i, y=0.85, text=s.replace("\n","<br>"), showarrow=False,
                            font=dict(size=11, color="#94A3B8"))
    fig.update_layout(showlegend=False, height=210,
                       xaxis=dict(visible=False, range=[-0.5, len(stages)-0.5]),
                       yaxis=dict(visible=False, range=[0.5, 1.3]),
                       margin=dict(l=10,r=10,t=10,b=10),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="callout">
    💡 Use the sidebar to explore each stage of the pipeline in depth — from raw dataset
    composition through to subtype-specific FDA-stratified drug candidates.
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE: DATASET & PREPROCESSING
# ============================================================================
elif page.startswith("🧪"):
    tag("Data Acquisition & Preprocessing")
    st.title("Dataset & Preprocessing Pipeline")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Primary cohort — GSE112676")
        fig = px.pie(values=[233, 508], names=["ALS", "Control"],
                      color=["ALS","Control"], color_discrete_map={"ALS":ALS_COLOR,"Control":CTRL_COLOR},
                      hole=0.55, title="n = 741 · Illumina HumanHT-12 V3 (GPL6947)")
        style_fig(fig, 340)
        st.plotly_chart(fig, use_container_width=True)
        cc1, cc2, cc3 = st.columns(3)
        with cc1: metric_card("741", "Total samples")
        with cc2: metric_card("233", "ALS")
        with cc3: metric_card("508", "Controls")

    with c2:
        st.subheader("External validation cohort — GSE112680")
        fig = px.pie(values=[164, 137], names=["ALS", "Control"],
                      color=["ALS","Control"], color_discrete_map={"ALS":ALS_COLOR,"Control":CTRL_COLOR},
                      hole=0.55, title="n = 301 · Illumina HumanHT-12 V4 (GPL10558)")
        style_fig(fig, 340)
        st.plotly_chart(fig, use_container_width=True)
        cc1, cc2, cc3 = st.columns(3)
        with cc1: metric_card("301", "Total samples")
        with cc2: metric_card("164", "ALS")
        with cc3: metric_card("137", "Controls")

    st.markdown("---")
    st.subheader("Train / Test Split (strictly leak-free)")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        fig = go.Figure(go.Bar(
            x=["Training (80%)", "Held-out Test (20%)"], y=[592, 149],
            marker_color=[PRIMARY, ACCENT], text=[592, 149], textposition="outside"
        ))
        fig.update_layout(title="Stratified split, n = 741 → 592 / 149", yaxis_title="Samples")
        style_fig(fig, 360)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("""
        <div class="callout">
        <b>Why split-first matters:</b> the dataset was partitioned into training (80%, n=592)
        and test (20%, n=149) sets via stratified random sampling <i>before</i> any
        normalisation, imputation, or feature selection — preventing test-set information
        from leaking into training-stage transformations. A fixed random seed (42) ensures
        full reproducibility.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### Pipeline steps")
        steps = [
            ("1", "QC filtering", "Probes with >30% missing values in training samples removed (48,803 initial probes → QC subset)."),
            ("2", "Imputation", "Residual missing values imputed using training-set medians (applied identically to test set)."),
            ("3", "Log2 transform", "log2(x + 1) pseudocount stabilises variance across probe intensities."),
            ("4", "Quantile normalisation", "QuantileTransformer fit exclusively on training data: X_norm = Φ⁻¹(F_train(X))."),
            ("5", "Standard scaling", "StandardScaler fit on training data only — zero mean, unit variance."),
        ]
        for n, title, desc in steps:
            st.markdown(f"**{n}. {title}** — {desc}")

    st.markdown("---")
    st.markdown("""
    <div class="warn">
    <b>Cross-platform note:</b> GSE112680 was selected because it shares tissue type,
    research group, and disease context with GSE112676, while introducing deliberate
    technical heterogeneity via a different BeadChip version (V3 → V4) — a stringent
    test of generalisability. Probe overlap between platforms: <b>39,426 common probes
    (80.8% of training probes)</b>.
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE: DIFFERENTIAL EXPRESSION
# ============================================================================
elif page.startswith("📈"):
    tag("Section 3.3 / 4.3 — Differential Expression Analysis")
    st.title("Differential Expression Analysis")
    st.markdown("""
Welch's two-sample t-test was applied independently to every probe (accounting for
unequal variances between the 233 ALS vs. 508 control groups), with resulting p-values
corrected via **Benjamini–Hochberg FDR** (α = 0.05). Probes were ranked by absolute
t-statistic magnitude and the **top 1,000** were carried forward as the feature set for
all downstream modelling.
    """)

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("48,803", "Initial probes")
    with c2: metric_card("1,000", "Top DE probes selected")
    with c3: metric_card("FDR < 0.05", "Significance threshold")

    st.write("")
    left, right = st.columns([1.4, 1])
    with left:
        img("image2.png", caption="Fig. 1 — Volcano plot: differential expression, ALS vs. Control")
    with right:
        st.markdown("""
        <div class="callout">
        Top-ranked probes span <b>both up- and down-regulated</b> transcripts —
        reflecting the bidirectional nature of ALS-associated blood transcriptomic
        dysregulation. Following GPL6947 gene-symbol mapping, top probes converged on
        genes implicated in:
        </div>
        """, unsafe_allow_html=True)
        for c in ["RNA metabolism", "Ubiquitin-mediated protein degradation",
                  "Mitochondrial function", "Vesicular trafficking", "Apoptotic signalling"]:
            st.markdown(f'<span class="gene-chip">{c}</span>', unsafe_allow_html=True)

# ============================================================================
# PAGE: ML CLASSIFICATION
# ============================================================================
elif page.startswith("🤖"):
    tag("Section 3.4–3.6 / 4.4–4.6 — Machine Learning Classification")
    st.title("Machine Learning Classification")
    st.markdown("""
Four supervised classifiers were trained on the top 1,000 DE probes: **Logistic
Regression (L2)**, **Random Forest**, **SVM (RBF kernel)**, and **XGBoost** — a
deliberate spectrum from linear to non-linear model complexity.
    """)

    tabs = st.tabs(["🏆 Best Model — Logistic Regression", "🧮 Confusion Matrix", "📉 ROC Curve", "⚙️ Hyperparameter Tuning"])

    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("0.944", "Test AUC")
        with c2: metric_card("89.3%", "Test Accuracy")
        with c3: metric_card("80.2%", "Sensitivity")
        with c4: metric_card("93.5%", "Specificity")
        st.markdown("""
        <div class="callout">
        Logistic Regression <b>outperformed</b> the more complex ensemble methods (Random
        Forest, SVM, XGBoost) — consistent with the well-established observation that
        regularised linear models excel in high-dimensional, low-sample-size transcriptomics
        settings. Higher specificity (93.5%) vs. sensitivity (80.2%) reflects the decision
        boundary's slight bias toward the more abundant control class (508 vs 233).
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        gauge_c1, gauge_c2 = st.columns(2)
        with gauge_c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=0.944,
                title={'text': "Test AUC-ROC"},
                gauge={'axis': {'range': [0.5, 1]}, 'bar': {'color': ACCENT},
                       'steps': [{'range': [0.5, 0.7], 'color': '#1F2433'},
                                 {'range': [0.7, 0.9], 'color': '#262D40'},
                                 {'range': [0.9, 1], 'color': '#333B55'}]}))
            style_fig(fig, 280)
            st.plotly_chart(fig, use_container_width=True)
        with gauge_c2:
            fig = go.Figure(go.Bar(
                x=["Sensitivity", "Specificity", "Accuracy"], y=[0.802, 0.935, 0.893],
                marker_color=[ALS_COLOR, CTRL_COLOR, PRIMARY],
                text=["80.2%","93.5%","89.3%"], textposition="outside"))
            fig.update_layout(yaxis_range=[0,1.1], title="Classification metrics")
            style_fig(fig, 280)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        c1, c2 = st.columns([1,1.2])
        with c1:
            img("image3.png", caption="Fig. 2 — Confusion matrix, Logistic Regression (test set)")
        with c2:
            st.markdown(f"""
            On the held-out test set (n = 149):
            - **Sensitivity (Recall, ALS): 80.2%** — proportion of true ALS patients correctly identified.
            - **Specificity: 93.5%** — proportion of true controls correctly identified.
            - The model shows a modest bias toward the majority (control) class, consistent
              with the underlying **1 : 2.18 class imbalance** (233 ALS : 508 controls).
            """)

    with tabs[2]:
        c1, c2 = st.columns([1,1.2])
        with c1:
            img("image4.png", caption="Fig. 3 — ROC curve comparison across classifiers")
        with c2:
            st.markdown("""
            All four classifiers (Logistic Regression, Random Forest, SVM-RBF, XGBoost)
            were trained on the identical StandardScaler-normalised feature matrix and
            evaluated via 5-fold stratified cross-validation and on the independent
            held-out test set. **Logistic Regression achieved the top test AUC of 0.944**,
            reported here as the primary model carried forward into SHAP explainability,
            PPI analysis, and external validation.
            """)

    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            cs = [0.01, 0.1, 1, 10, 100]
            highlight = [PRIMARY if c == 0.1 else "#2E3448" for c in cs]
            fig = go.Figure(go.Bar(x=[str(c) for c in cs], y=[0.80,0.872,0.85,0.83,0.81],
                                     marker_color=highlight))
            fig.update_layout(title="GridSearchCV over C (5-fold CV) — optimum at C = 0.1",
                               xaxis_title="Regularisation strength C", yaxis_title="Mean CV AUC (illustrative)")
            style_fig(fig, 340)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Bars illustrate the search grid; the reported optimum (C = 0.1, CV AUC = 0.872) is exact, intermediate values are for visual context only.")
        with c2:
            st.markdown("""
            **Within-CV leak-free feature selection:** a scikit-learn `Pipeline` encapsulated
            `SelectKBest` (k = 500, ANOVA F-statistic) and Logistic Regression as sequential
            steps *within each fold* — ensuring feature selection uses only that fold's
            training data.

            - **Optimal regularisation:** C = 0.1 (strong L2 penalty)
            - **Cross-validation AUC:** 0.872
            - **Test AUC:** 0.944
            - **Bootstrap validation:** Mean AUC = 0.943, 95% CI [0.905 – 0.973]

            The modest gap between CV AUC (0.872) and test AUC (0.944) indicates the
            model generalises well without substantial overfitting. Strong regularisation
            in a 1,000-probe / 592-sample feature space enforces a **distributed
            multi-gene signature** rather than reliance on any single transcript —
            appropriate given the polygenic nature of ALS transcriptomic dysregulation.
            """)

# ============================================================================
# PAGE: SHAP & BIOMARKERS
# ============================================================================
elif page.startswith("🔍"):
    tag("Section 3.7 / 4.7 — Explainability & Biomarker Discovery")
    st.title("SHAP Analysis & Biomarker Identification")
    st.markdown("""
**SHapley Additive exPlanations (SHAP)** decomposed individual Logistic Regression
predictions into per-gene contribution scores. A `LinearExplainer` used the training
set as background reference; SHAP values were computed for all held-out test samples,
with mean absolute SHAP value used to rank genes globally.
    """)

    c1, c2 = st.columns([1, 1])
    with c1:
        img("image5.png", caption="Fig. 4 — Top 20 biomarkers by mean |SHAP value|")
    with c2:
        img("image6.png", caption="SHAP beeswarm / individual prediction attribution")

    st.markdown("---")
    st.subheader("Candidate biomarker panel")

    filt = st.multiselect("Filter by status", ["Known", "Novel"], default=["Known", "Novel"])
    df = pd.DataFrame(TOP20_BIOMARKERS)
    df = df[df.status.isin(filt)]

    fig = px.bar(df.groupby("status").size().reset_index(name="count"), x="status", y="count",
                 color="status", color_discrete_map={"Known": CTRL_COLOR, "Novel": ALS_COLOR})
    fig.update_layout(title="Known ALS-associated vs. novel candidate biomarkers", showlegend=False)
    style_fig(fig, 280)
    st.plotly_chart(fig, use_container_width=True)

    for _, row in df.iterrows():
        badge_color = CTRL_COLOR if row.status == "Known" else ALS_COLOR
        st.markdown(f"""
        <div style="background:#141824; border:1px solid #262B3D; border-left:4px solid {badge_color};
        border-radius:8px; padding:12px 16px; margin-bottom:8px;">
        <b style="font-family:monospace; font-size:15px; color:#F1F5F9;">{row.gene}</b>
        &nbsp;<span style="font-size:11px; background:{badge_color}22; color:{badge_color};
        padding:2px 8px; border-radius:10px;">{row.status}</span>
        &nbsp;<span style="font-size:11px; color:#94A3B8;">· {row.category}</span>
        <div style="font-size:13.5px; color:#CBD5E1; margin-top:4px;">{row.note}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warn">
    Novel candidates (QPCT, SAR1B, SLC44A1, HIBADH, SAMSN1, NT5DC1) lack established
    ALS literature support and are explicitly flagged as requiring experimental
    validation — not clinical inference.
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE: PPI NETWORK & PATHWAYS
# ============================================================================
elif page.startswith("🕸️"):
    tag("Section 3.9 / 4.9 — PPI Network & Pathway Enrichment")
    st.title("Protein–Protein Interaction Network & Pathway Enrichment")

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("20", "Network nodes")
    with c2: metric_card("3", "High-confidence edges")
    with c3: metric_card("NDFIP1", "Principal hub gene")

    left, right = st.columns([1.2, 1])
    with left:
        img("image9.png", caption="Fig. 7 — STRING PPI network (confidence ≥ 0.4) of top SHAP-ranked genes")
    with right:
        st.subheader("Hub genes by centrality")
        fig = px.bar(PPI_HUBS.sort_values("degree"), x="degree", y="gene", orientation="h",
                     color="degree", color_continuous_scale=["#2E3448", ACCENT],
                     labels={"degree":"Degree centrality","gene":""})
        style_fig(fig, 300)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(PPI_HUBS, hide_index=True, use_container_width=True)
        st.caption("Sparse topology (3 edges / 20 nodes) is expected for blood-transcriptomics-derived gene sets, which don't recapitulate dense motor-neuron proteome interaction networks.")

    st.markdown("---")
    st.subheader("Pathway Enrichment (g:Profiler, FDR-adjusted p < 0.05)")
    c1, c2 = st.columns([1, 1.1])
    with c1:
        img("image7.png", caption="Fig. 5 — Pathway enrichment of top differentially expressed genes")
    with c2:
        for p in PATHWAYS:
            st.markdown(f"**• {p['name']}**")
            st.caption(p["detail"])

    st.markdown("---")
    st.subheader("ALS Risk Score Prediction")
    img("image8.png", caption="Fig. 6 — Predicted ALS risk score distribution", width=650)

# ============================================================================
# PAGE: MOLECULAR SUBTYPING
# ============================================================================
elif page.startswith("🧩"):
    tag("Section 3.10 / 4.10 — Unsupervised Molecular Subtyping")
    st.title("Molecular Subtyping of ALS Patients")
    st.markdown("""
Unsupervised clustering was performed **exclusively on the 233 ALS patient samples**
(not controls) using the top 1,000 DE probes — isolating genuine within-disease
transcriptomic heterogeneity from the ALS-vs-control classification signal.
    """)

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("96.2%", "Variance explained (top 50 PCs)")
    with c2: metric_card("k = 2", "Optimal cluster number")
    with c3: metric_card("0.478", "Silhouette score at k=2")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Cluster quality across k")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=CLUSTER_METRICS.k, y=CLUSTER_METRICS.silhouette,
                              name="Silhouette score", marker_color=PRIMARY, yaxis="y1"))
        fig.add_trace(go.Scatter(x=CLUSTER_METRICS.k, y=CLUSTER_METRICS.ch_index,
                                  name="Calinski-Harabasz index", mode="lines+markers",
                                  marker_color=ACCENT, yaxis="y2"))
        fig.update_layout(
            xaxis=dict(title="k", tickmode="array", tickvals=[2,3,4]),
            yaxis=dict(title="Silhouette score", side="left"),
            yaxis2=dict(title="CH index", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.15),
            title="Both metrics agree: k = 2 is optimal"
        )
        style_fig(fig, 380)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(CLUSTER_METRICS, hide_index=True, use_container_width=True)

    with right:
        st.subheader("Subtype composition")
        fig = px.pie(values=[121, 112], names=["Cluster 1", "Cluster 2"], hole=0.55,
                     color_discrete_sequence=[PRIMARY, ACCENT], title="233 ALS patients")
        style_fig(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    img("image12.png", caption="Fig. 10 — Molecular subtyping: PCA / cluster visualisation")

    st.markdown("---")
    st.subheader("Cluster 1 — Upregulated Genes")
    st.markdown(" ".join([f'<span class="gene-chip">{g}</span>' for g in CLUSTER1_GENES]), unsafe_allow_html=True)
    st.markdown("""
    <div class="callout" style="margin-top:12px;">
    Cluster 1 (n=121) shows a molecularly distinct signature suggesting predominant
    dysregulation of <b>mitochondrial calcium homeostasis</b> (MICU2), <b>axonal
    transport</b> (FEZ2), and <b>glycosylation</b> (MGAT4A). Cluster 2 (n=112) did not
    show significant upregulation under the applied threshold — possibly a
    transcriptomically quieter subtype, or one requiring more sensitive detection.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="warn" style="margin-top:8px;">
    These subtypes are <b>computationally derived</b> and require clinical validation
    against ALSFRS-R progression scores, survival data, and site-of-onset metadata
    before any prognostic interpretation.
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE: DRUG REPURPOSING
# ============================================================================
elif page.startswith("💊"):
    tag("Section 3.11 / 4.11 — Computational Drug Repurposing")
    st.title("Subtype-Specific Drug Repurposing")
    st.markdown("""
Cluster 1 subtype-defining upregulated genes were mapped to a curated drug–gene
interaction table, with candidates stratified by regulatory approval status to
prioritise the most immediately translatable agents.
    """)

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("43", "Total candidates identified")
    with c2: metric_card("27", "FDA-approved compounds")
    with c3: metric_card("14", "Cluster 1 candidates shown")

    left, right = st.columns([1, 1.1])
    with left:
        img("image13.png", caption="Fig. 11 — Drug repurposing overview")
        status_counts = DRUG_TABLE.status.value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig = px.pie(status_counts, values="count", names="status", hole=0.5,
                     color="status",
                     color_discrete_map={"FDA approved": "#22C55E", "Clinical trial": ACCENT,
                                          "Preclinical": "#F59E0B", "Research": "#94A3B8"},
                     title="Cluster 1 candidates by approval status (n=14 shown)")
        style_fig(fig, 340)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Filter candidates")
        status_filter = st.multiselect("Approval status", DRUG_TABLE.status.unique().tolist(),
                                        default=DRUG_TABLE.status.unique().tolist())
        filtered = DRUG_TABLE[DRUG_TABLE.status.isin(status_filter)]
        st.dataframe(
            filtered.rename(columns={"target":"Target Gene","drug":"Drug","status":"Approval Status","area":"Disease Area"}),
            hide_index=True, use_container_width=True, height=440
        )

    st.markdown("""
    <div class="callout">
    <b>Internal biological control:</b> Lithium Chloride and Valproic Acid (targeting
    FEZ2) have <b>both been previously evaluated in ALS clinical trials</b> — the
    pipeline recovered known ALS therapeutic candidates <i>without any prior encoding
    of ALS trial history</i>, validating that it captures genuine disease-relevant
    biology rather than spurious associations.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="warn">
    All drug–gene associations are <b>computationally derived</b> and require
    experimental validation in appropriate ALS cellular/animal models before any
    clinical relevance can be claimed.
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE: EXTERNAL VALIDATION
# ============================================================================
elif page.startswith("🌍"):
    tag("Section 3.8 / 4.8 — Cross-Platform External Validation")
    st.title("External Validation — GSE112680 (HT-12 V4)")
    st.markdown("""
Model generalisability was tested on an independent cross-platform cohort
(n = 301; 164 ALS, 137 controls; Illumina HumanHT-12 **V4**), using preprocessing
parameters fitted exclusively on GSE112676 training data.
    """)

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("39,426", "Common probes (80.8%)")
    with c2: metric_card("0.758", "AUC, no batch correction")
    with c3: metric_card("0.732", "AUC, ComBat corrected")

    left, right = st.columns([1.2, 1])
    with left:
        vt = VALIDATION_TABLE.melt(id_vars="Metric", var_name="Cohort", value_name="Value")
        vt["Cohort"] = vt["Cohort"].replace({
            "Internal_GSE112676": "Internal (GSE112676)",
            "External_no_ComBat": "External — no ComBat",
            "External_ComBat": "External — ComBat"
        })
        fig = px.bar(vt, x="Metric", y="Value", color="Cohort", barmode="group",
                     color_discrete_sequence=[PRIMARY, ACCENT, "#F59E0B"],
                     text_auto=".3f")
        fig.update_layout(title="Internal vs. External validation performance", yaxis_range=[0,1])
        style_fig(fig, 400)
        st.plotly_chart(fig, use_container_width=True)

        display_tbl = pd.DataFrame({
            "Metric": ["AUC", "Accuracy", "N samples", "Platform", "Batch corrected"],
            "Internal GSE112676": [0.944, 0.893, 741, "HT-12 V3", "No"],
            "External (no ComBat)": [0.758, 0.671, 301, "HT-12 V4", "No"],
            "External (ComBat)": [0.732, 0.628, 301, "HT-12 V4", "Yes"],
        })
        st.dataframe(display_tbl, hide_index=True, use_container_width=True)

    with right:
        st.markdown("""
        **Three validation dimensions assessed:**
        1. **Expression consistency** — directional DE patterns of SHAP-selected genes
           (ABCA1, SRPK1, NDFIP1) were preserved in the external cohort.
        2. **Predictive robustness** — direct application without batch correction:
           AUC 0.758, Accuracy 67.1%, ALS recall 0.88, control precision 0.75.
        3. **SHAP gene reproducibility** — top-ranked genes retained biological
           relevance in the external transcriptomic profile.

        **On ComBat:** correction produced a *marginal* AUC reduction (0.758 → 0.732,
        Δ = 0.026), consistent with partial confounding between biological and
        technical variation across platforms — ComBat can remove genuine ALS signal
        alongside technical noise when group and batch membership aren't fully
        independent. Both results significantly exceed chance (AUC = 0.5).
        """)
        st.markdown("""
        <div class="callout">
        Performance reduction vs. the internal test AUC (0.944) reflects genuine
        cross-platform technical variation (V3 → V4 BeadChip), <b>not model
        overfitting</b> — supported by directionally consistent predictions on the
        external cohort. Same-platform validation would be expected to approach the
        within-CV estimate of 0.872.
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE: LITERATURE LANDSCAPE
# ============================================================================
elif page.startswith("📚"):
    tag("Section 2 — Literature Review")
    st.title("Literature Landscape & Research Gaps")
    st.markdown("""
Comparative analysis of recent ALS transcriptomics / biomarker studies, their
methodologies, key limitations, and how this study's integrated pipeline addresses
each identified gap.
    """)
    st.dataframe(
        LIT_TABLE.rename(columns={
            "no": "#", "author": "Author(s) & Year", "work": "Work Done",
            "ml": "ML Approach", "gap": "Identified Gap", "addressed": "Gap Addressed by This Study"
        }),
        hide_index=True, use_container_width=True, height=460
    )
    st.markdown("---")
    st.markdown("""
    <div class="callout">
    <b>Synthesis:</b> existing literature establishes a strong biological rationale for
    blood transcriptomics-based ALS biomarker discovery, highlights methodological
    pitfalls (data leakage, improper CV), and identifies the absence of a
    <b>fully integrated pipeline</b> combining leak-free classification, explainable AI,
    molecular subtyping, FDA-stratified drug repurposing, and cross-platform external
    validation — the gap this study was designed to fill.
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE: DISCUSSION & LIMITATIONS
# ============================================================================
elif page.startswith("🗒️"):
    tag("Section 5 — Discussion")
    st.title("Discussion & Limitations")

    with st.expander("5.1 — Why Blood, Not Muscle or CNS Tissue?", expanded=True):
        st.markdown("""
        Muscle weakness in ALS is a **downstream consequence** of denervation, not a
        primary myopathic process — muscle biopsies show neurogenic, not myopathic,
        atrophy. CNS tissue is inaccessible during life; CSF requires invasive lumbar
        puncture and reflects nonspecific neurodegeneration. **Whole blood** is
        accessible via routine venepuncture, enables longitudinal sampling, and is an
        **active participant** in ALS pathophysiology through peripheral immune
        dysregulation, monocyte/neutrophil activation, and systemic neuroinflammatory
        signalling.
        """)

    with st.expander("5.2 — Biological Significance of the DE Signature"):
        st.markdown("""
        Enrichment of **RNA binding/processing** pathways aligns with TDP-43/FUS
        dysfunction (present in >97% of ALS cases). **Ubiquitin-dependent degradation**
        pathway enrichment corroborates elevated ubiquitinated protein levels reported
        in ALS lymphocytes. **Mitochondrial/oxidative stress** enrichment reflects
        bioenergetic dysfunction increasingly recognised as systemic, not solely
        neuronal.
        """)

    with st.expander("5.3 — Interpreting the Top SHAP Biomarkers"):
        st.markdown("""
        **SRPK1** is mechanistically striking: it directly regulates **TDP-43**
        subcellular localisation and aggregation — the defining ALS pathological event
        — and SRPK1 inhibition is an emerging preclinical therapeutic strategy.
        **BORCS7** connects to **C9orf72**, the most common ALS genetic cause, via
        lysosomal biogenesis/autophagy regulation. **NDFIP1**, the PPI network hub,
        may represent a **coordinating node** through which multiple ALS-associated
        molecular alterations converge systemically.
        """)

    with st.expander("5.4 — Molecular Subtypes & Disease Heterogeneity"):
        st.markdown("""
        The near-balanced Cluster 1 (n=121) / Cluster 2 (n=112) split suggests the
        transcriptomic space of ALS whole blood is genuinely **bimodal**. This
        heterogeneity has been widely implicated in repeated ALS clinical trial
        failures, where responsive subgroups may be diluted within unselected
        populations.
        """)

    with st.expander("5.5 — Translational Implications of Drug Repurposing"):
        st.markdown("""
        Recovery of **Lithium Chloride** and **Valproic Acid** (both previously
        tested in ALS clinical trials) from a purely data-driven pipeline —
        with no prior trial history encoded — is an important internal validity
        control for the repurposing approach.
        """)

    with st.expander("5.6 — External Validation & Batch Effects"):
        st.markdown("""
        The AUC reduction on external validation (0.944 → 0.758) is attributable to
        **systematic intensity differences between BeadChip V3/V4**, not overfitting.
        ComBat's marginal negative effect (0.758 → 0.732) reflects a known limitation:
        when biological and technical variation are partially confounded, batch
        correction can remove genuine signal alongside noise.
        """)

    with st.expander("5.7 — Limitations & Future Directions", expanded=True):
        st.markdown("""
        - Cross-platform external validation limits interpretability of internal vs.
          external comparisons.
        - Molecular subtypes are derived from transcriptomics **alone**, without
          clinical outcome correlation (ALSFRS-R, survival) — prognostic significance
          remains to be established.
        - Drug repurposing candidates are **computationally derived**, requiring
          experimental testing in ALS cellular/animal models.
        - Blood transcriptomics captures **systemic correlates**, not primary motor
          neuron pathology directly.

        **Future work:** prospective longitudinal cohorts, multi-omic integration
        (proteomics, metabolomics), ComBat-seq for RNA-seq data, experimental
        validation of SRPK1 inhibitors and 4-PBA, and incorporation of clinical
        metadata (site of onset, genetics, ALSFRS-R) into the subtyping framework.
        """)

    st.markdown("---")
    st.subheader("Conclusion")
    st.markdown("""
    <div class="callout">
    This project delivers a robust, leak-free ML pipeline for ALS diagnosis and
    biomarker discovery from the GSE112676 blood transcriptome. Logistic Regression
    achieved Test AUC = 0.944 (Bootstrap Mean AUC 0.943, 95% CI [0.905, 0.973]).
    External validation (AUC 0.758) confirmed genuine, transferable transcriptomic
    signal. SHAP identified known ALS-relevant biomarkers (ABCA1, SLC25A20, BORCS7,
    TNFRSF10B, DNAJB14, SRPK1) alongside four novel candidates (SLC44A1, HIBADH,
    SAMSN1, NT5DC1). Unsupervised clustering revealed two molecular subtypes
    (k=2, silhouette = 0.478), feeding a drug-repurposing screen that yielded
    <b>43 candidates including 27 FDA-approved compounds</b>.
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE: REFERENCES & CODE
# ============================================================================
elif page.startswith("📖"):
    tag("Reproducibility")
    st.title("References & Code Availability")

    st.markdown("""
    <div class="callout">
    <b>Code availability:</b> the full analysis pipeline is available at —<br>
    <a href="https://colab.research.google.com/drive/10ndcEVdGPpgxNRmV7b_njGuayO-wJHUr#scrollTo=NtZrHtXBZ0QA"
    target="_blank" style="color:#06B6D4;">Google Colab Notebook ↗</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Key References")
    refs = [
        "van Rheenen W, et al. (2018). Whole blood transcriptome analysis in amyotrophic lateral sclerosis: a biomarker study. PLOS ONE, 13(6), e0198874.",
        "Hardiman O, et al. (2017). Amyotrophic lateral sclerosis. Nature Reviews Disease Primers, 3, 17071.",
        "van Es MA, et al. (2017). Amyotrophic lateral sclerosis. Lancet, 390(10107), 2084–2098.",
        "Chio A, et al. (2013). Global epidemiology of ALS: a systematic review. Neuroepidemiology, 41(2), 118–130.",
        "Mougeot JL, et al. (2011). Microarray analysis of peripheral blood lymphocytes from ALS patients and the SAFE study. Amyotrophic Lateral Sclerosis, 12(5), 324–330.",
        "Pedregosa F, et al. (2011). Scikit-learn: machine learning in Python. JMLR, 12, 2825–2830.",
        "Lundberg SM, Lee SI. (2017). A unified approach to interpreting model predictions. NeurIPS, 30, 4765–4774.",
        "Johnson WE, et al. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. Biostatistics, 8(1), 118–127.",
        "Raudvere U, et al. (2019). g:Profiler: a web server for functional enrichment analysis. Nucleic Acids Research, 47(W1), W191–W198.",
        "Szklarczyk D, et al. (2019). STRING v11: protein-protein association networks. Nucleic Acids Research, 47(D1), D607–D613.",
        "Freshour SL, et al. (2021). Integration of the Drug-Gene Interaction Database (DGIdb 4.0). Nucleic Acids Research, 49(D1), D1144–D1151.",
        "Chen T, Guestrin C. (2016). XGBoost: a scalable tree boosting system. KDD 2016, 785–794.",
        "Benjamini Y, Hochberg Y. (1995). Controlling the false discovery rate. JRSS B, 57(1), 289–300.",
        "Pushpakom S, et al. (2019). Drug repurposing: progress, challenges and recommendations. Nature Reviews Drug Discovery, 18(1), 41–58.",
        "Prudencio M, et al. (2015). Distinct brain transcriptome profiles in C9orf72-associated and sporadic ALS. Nature Neuroscience, 18(8), 1175–1182.",
    ]
    for r in refs:
        st.markdown(f"- {r}")
    st.caption("Full reference list (30 items) is available in the source manuscript.")

    st.markdown("---")
    st.subheader("About this dashboard")
    st.markdown("""
    This Streamlit dashboard was generated from the manuscript
    *"Integrated Blood Transcriptomic Analysis of ALS: Machine Learning Classification,
    Molecular Subtyping, and Subtype-Specific Drug Repurposing Using GSE112676."*
    All figures are reproduced directly from the source document; interactive charts
    (dataset composition, cluster metrics, validation comparisons, drug repurposing
    table, PPI hub centralities) are built from the exact numeric results reported
    in the text.
    """)

st.markdown("---")
st.caption("ALS Blood Transcriptomics Dashboard · Built with Streamlit & Plotly · Source: GSE112676 / GSE112680 (NCBI GEO)")
