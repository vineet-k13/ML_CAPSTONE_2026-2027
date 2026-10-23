"""Streamlit Web Application: Video Game Sales & Market Intelligence

Demonstrates Review 1 and Review 2 deliverables:
1. Interactive Prediction & Market Simulator (Trained Pipeline)
2. Consolidated 10-Classifier Comparison & Leaderboard
3. Unsupervised Market Clusters Explorer (PCA 2D projection)
4. Capstone Methodology & Pipeline Overview
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ui

# Page configuration
st.set_page_config(
    page_title="Video Game Market Intelligence | ML Capstone",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject custom aesthetic CSS & animations
ui.inject_css()

# Root paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "Video_Games_Sales_as_at_22_Dec_2016.csv"


@st.cache_resource
def load_models():
    clf_path = MODELS_DIR / "sales_tier_classifier.joblib"
    reg_path = MODELS_DIR / "global_sales_regressor.joblib"
    metrics_path = MODELS_DIR / "metrics.json"

    clf = joblib.load(clf_path) if clf_path.exists() else None
    reg = joblib.load(reg_path) if reg_path.exists() else None

    metrics = None
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return clf, reg, metrics


@st.cache_data
def load_cluster_data():
    pts_path = MODELS_DIR / "cluster_points.csv"
    prof_path = MODELS_DIR / "cluster_profile.csv"
    pts = pd.read_csv(pts_path) if pts_path.exists() else None
    prof = pd.read_csv(prof_path) if prof_path.exists() else None
    return pts, prof


@st.cache_data
def load_raw_dataset():
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
        df["User_Score"] = pd.to_numeric(df["User_Score"], errors="coerce")
        return df
    return None


clf_model, reg_model, metrics = load_models()
cluster_pts, cluster_prof = load_cluster_data()
raw_df = load_raw_dataset()

# Masthead
ui.render(f"""
<div class="mast">
  <div>
    <h1 class="display">Video Game Market Intelligence <em>&</em> Sales Forecaster</h1>
  </div>
  <div class="meta">
    <div>
      <span class="eyebrow">DATASET</span>
      <div class="v">16.7k</div>
      <span class="muted" style="font-size:0.75rem;">Game Releases</span>
    </div>
    <div>
      <span class="eyebrow">MODELS EVALUATED</span>
      <div class="v">22</div>
      <span class="muted" style="font-size:0.75rem;">10 Reg + 10 Clf + 2 Clu</span>
    </div>
    <div>
      <span class="eyebrow">BEST CLASSIFIER</span>
      <div class="v" style="color:var(--accent);">
        {metrics["classification"]["selected_model"] if metrics else "Random Forest"}
      </div>
      <span class="muted" style="font-size:0.75rem;">Tuned 5-Fold CV</span>
    </div>
  </div>
</div>
""")

# Primary Navigation Tabs
tab_pred, tab_board, tab_cluster, tab_method = st.tabs([
    "  Market Forecaster & Simulator",
    "  Classification Leaderboard (10 Models)",
    "  Market Segment Clusters (PCA & Profiles)",
    "  Architecture & Methodology",
])

# ==============================================================================
# TAB 1: PREDICTOR & SIMULATOR
# ==============================================================================
with tab_pred:
    col_input, col_result = st.columns([1.15, 1.45], gap="large")

    with col_input:
        ui.render(f"""
        <div class="card-head" style="margin-bottom:0.75rem;">
          <div class="t">
            {ui.icon("target", 18)}
            <h3 style="margin:0;">Title Parameters</h3>
          </div>
          <span class="chip live"><span class="dot"></span>LIVE PIPELINE</span>
        </div>
        """)

        with st.container(key="input_panel"):
            c_plat, c_genre = st.columns(2)
            with c_plat:
                platform_opts = [
                    "PS4", "PS3", "PS2", "X360", "XOne", "Wii", "WiiU", "PC", "DS", "3DS", "PSP", "GBA", "XB", "GC"
                ]
                platform = st.selectbox("Platform", platform_opts, index=0)
            with c_genre:
                genre_opts = [
                    "Action", "Shooter", "Role-Playing", "Sports", "Platform",
                    "Racing", "Misc", "Fighting", "Simulation", "Adventure", "Strategy", "Puzzle"
                ]
                genre = st.selectbox("Genre", genre_opts, index=1)

            c_year, c_rate = st.columns(2)
            with c_year:
                year = st.slider("Release Year", min_value=1995, max_value=2026, value=2015, step=1)
            with c_rate:
                rating_opts = ["M", "T", "E", "E10+"]
                rating = st.selectbox("ESRB Rating", rating_opts, index=0)

            st.markdown("---")
            c_cscore, c_ccnt = st.columns(2)
            with c_cscore:
                critic_score = st.slider("Metacritic Critic Score", 20.0, 100.0, 84.0, 1.0)
            with c_ccnt:
                critic_count = st.slider("Critic Reviews Count", 5, 120, 65, 1)

            c_uscore, c_ucnt = st.columns(2)
            with c_uscore:
                user_score = st.slider("User Rating (0 - 10)", 1.0, 10.0, 7.8, 0.1)
            with c_ucnt:
                user_count = st.slider("User Reviews Count", 10, 8000, 1450, 10)

            # Calculated Gap
            score_gap = (critic_score / 10.0) - user_score
            gap_badge = (
                f"<span style='color:var(--accent); font-weight:600;'>+{score_gap:.2f} (Critics favor)</span>"
                if score_gap > 0
                else f"<span style='color:var(--ink); font-weight:600;'>{score_gap:.2f} (Users favor)</span>"
            )
            ui.render(f"""
            <div style="font-size:0.8rem; color:var(--muted); margin-top:0.4rem;">
              Calculated Critic-User Gap: {gap_badge}
            </div>
            """)

    with col_result:
        # Create input dataframe
        input_data = pd.DataFrame([{
            "Year_of_Release": float(year),
            "Critic_Score": float(critic_score),
            "Critic_Count": float(critic_count),
            "User_Score": float(user_score),
            "User_Count": float(user_count),
            "Platform": platform,
            "Genre": genre,
            "Rating": rating,
        }])

        if clf_model is not None:
            # Predict Tier
            pred_tier = clf_model.predict(input_data)[0]
            pred_proba = clf_model.predict_proba(input_data)[0]
            classes = list(clf_model.classes_)

            # Probability mapping
            proba_dict = {cls: float(p) for cls, p in zip(classes, pred_proba)}
            top_prob = proba_dict.get(pred_tier, 0.0)

            # Regression estimate
            pred_sales = 0.0
            if reg_model is not None:
                pred_sales = max(0.01, float(reg_model.predict(input_data)[0]))

            tier_c = ui.TIER_COLOURS.get(pred_tier, ui.INK)
            tier_ico = ui.icon(ui.TIER_ICON.get(pred_tier, "trophy"), 36, "bob")

            tier_desc = {
                "Hit": "Blockbuster tier (> 1.0M units). Strong critical reception and massive player engagement momentum.",
                "Moderate": "Mid-tier commercial performance (0.2M – 1.0M units). Solid player base with profitable long tail.",
                "Low": "Niche/Limited volume (< 0.2M units). Restricted reach or specialized audience appeal.",
            }.get(pred_tier, "")

            # Result verdict block
            ui.render(f"""
            <div class="card d1" style="--tier: {tier_c};">
              <div class="card-head">
                <span class="eyebrow">PREDICTED CLASSIFICATION RESULT</span>
                <span class="chip accent"><span class="dot"></span>TUNED {metrics['classification']['selected_model'].upper() if metrics else 'MODEL'}</span>
              </div>
              <div class="verdict">
                <div class="ring-wrap">
                  <div class="ring" style="--deg: {top_prob * 100:.1f};"></div>
                  <div class="ring-center">
                    <div class="pct" style="--n: {int(top_prob * 100)};"><small>%</small></div>
                    <div class="lbl">Confidence</div>
                  </div>
                </div>
                <div>
                  <div class="tier-name display" style="color:{tier_c};">
                    {tier_ico} <span>{pred_tier}</span>
                  </div>
                  <p>{tier_desc}</p>
                </div>
              </div>
            </div>
            """)

            # Probability distribution breakdown
            low_p = proba_dict.get("Low", 0.0) * 100
            mod_p = proba_dict.get("Moderate", 0.0) * 100
            hit_p = proba_dict.get("Hit", 0.0) * 100

            ui.render(f"""
            <div class="card d2">
              <div class="card-head">
                <div class="t">
                  {ui.icon("layers", 18)}
                  <h3>Class Probabilities Distribution</h3>
                </div>
                <span class="muted mono" style="font-size:0.75rem;">Softmax / OvR</span>
              </div>
              <div class="bars">
                <div class="bar-row {'top' if pred_tier == 'Hit' else ''}">
                  <span class="name"><i class="sw" style="--c:{ui.ACCENT};"></i>Hit Tier</span>
                  <div class="bar-track">
                    <div class="bar-fill" style="--w:{hit_p:.1f}%; --c:{ui.ACCENT}; --delay:0.1s;"></div>
                  </div>
                  <span class="val" style="--n:{int(hit_p)};"></span>
                </div>
                <div class="bar-row {'top' if pred_tier == 'Moderate' else ''}">
                  <span class="name"><i class="sw" style="--c:{ui.INK_SOFT};"></i>Moderate</span>
                  <div class="bar-track">
                    <div class="bar-fill" style="--w:{mod_p:.1f}%; --c:{ui.INK_SOFT}; --delay:0.2s;"></div>
                  </div>
                  <span class="val" style="--n:{int(mod_p)};"></span>
                </div>
                <div class="bar-row {'top' if pred_tier == 'Low' else ''}">
                  <span class="name"><i class="sw" style="--c:{ui.STONE};"></i>Low Volume</span>
                  <div class="bar-track">
                    <div class="bar-fill" style="--w:{low_p:.1f}%; --c:{ui.STONE}; --delay:0.3s;"></div>
                  </div>
                  <span class="val" style="--n:{int(low_p)};"></span>
                </div>
              </div>
            </div>
            """)

            # Key Estimated Metrics
            ui.render(f"""
            <div class="stats d3" style="margin-top:1rem;">
              <div class="stat">
                <div class="k">{ui.icon("globe", 16)} Indicative Volume</div>
                <div class="v">{pred_sales:.2f}M</div>
                <div class="s">Tuned Gradient Boost Regressor</div>
              </div>
              <div class="stat">
                <div class="k">{ui.icon("star", 16)} Critic Metascore</div>
                <div class="v">{critic_score:.0f}/100</div>
                <div class="s">{critic_count} accredited publications</div>
              </div>
              <div class="stat">
                <div class="k">{ui.icon("users", 16)} Player Sentiment</div>
                <div class="v">{user_score:.1f}/10</div>
                <div class="s">{user_count:,} public user ratings</div>
              </div>
            </div>
            """)

            # Comparable Titles from the dataset
            if raw_df is not None:
                matches = raw_df[
                    (raw_df["Genre"] == genre) &
                    (raw_df["Critic_Score"].between(critic_score - 7, critic_score + 7))
                ].sort_values("Global_Sales", ascending=False).head(4)

                if not matches.empty:
                    ui.render(f"""
                    <div class="card d4" style="margin-top:1rem;">
                      <div class="card-head">
                        <div class="t">
                          {ui.icon("database", 18)}
                          <h3>Historical Market Comparables ({genre})</h3>
                        </div>
                        <span class="muted" style="font-size:0.75rem;">Metacritic ±7 pts</span>
                      </div>
                      <div class="titles">
                        {"".join([
                            f'''<div class="title-row">
                              <span class="idx">#{i+1}</span>
                              <div>
                                <div class="nm">{ui.esc(row["Name"])}</div>
                                <div class="sub">{ui.esc(row["Platform"])} • {int(row["Year_of_Release"]) if pd.notna(row["Year_of_Release"]) else "N/A"} • Critic: {row["Critic_Score"]:.0f}</div>
                              </div>
                              <span class="sales">{row["Global_Sales"]:.2f}M</span>
                              <span class="tierchip {pred_tier}">{pred_tier}</span>
                            </div>'''
                            for i, (_, row) in enumerate(matches.iterrows())
                        ])}
                      </div>
                    </div>
                    """)
        else:
            st.info("Pipeline models are currently training in the background. Artifacts will display immediately upon completion.")

# ==============================================================================
# TAB 2: MODEL LEADERBOARD (10 CLASSIFIERS)
# ==============================================================================
with tab_board:
    ui.render(f"""
    <div class="card" style="margin-bottom:1.5rem;">
      <div class="card-head">
        <div>
          <span class="eyebrow">CONSOLIDATED REVIEW 1 & REVIEW 2 EVALUATION</span>
          <h2 class="display" style="font-size:2rem; margin:0.3rem 0 0;">The 10-Classifier Tournament</h2>
        </div>
        <span class="chip"><span class="dot"></span>TEST SET SIZE: 3,290 TITLES</span>
      </div>
      <p class="muted" style="margin:0; font-size:0.92rem; max-width:55rem;">
        All 10 algorithms trained on the identical 80:20 stratified training split and evaluated on the same held-out test split.
        No data leakage: imputers and scalers fitted on train set only.
        Metrics computed using weighted averages and multiclass One-vs-Rest (OvR) ROC-AUC.
      </p>
    </div>
    """)

    if metrics and "classification" in metrics:
        clf_data = metrics["classification"]["table"]
        df_table = pd.DataFrame(clf_data).sort_values("F1", ascending=False).reset_index(drop=True)
        df_table.index = df_table.index + 1
        df_table.index.name = "Rank"

        # Table rows formatting
        sel_model = metrics["classification"]["selected_model"]

        rows_html = []
        for rank, row in df_table.iterrows():
            is_deployed = (row["Algorithm"] == sel_model)
            deployed_cls = "deployed" if is_deployed else ""
            badge = f'<span class="chip accent" style="font-size:0.65rem; padding:0.15rem 0.4rem; margin-left:0.5rem;">DEPLOYED</span>' if is_deployed else ""
            part_badge = f'<span class="tierchip {"Hit" if row["Part"] == "B" else "Moderate"}" style="font-size:0.65rem;">PART {row["Part"]}</span>'

            f1_pct = float(row["F1"]) * 100
            rows_html.append(f"""
            <tr class="{deployed_cls}">
              <td style="font-weight:600;">#{rank}</td>
              <td style="font-weight:500;">{row['Algorithm']} {badge}</td>
              <td>{part_badge}</td>
              <td>{row['Accuracy']:.4f}</td>
              <td>{row['Precision']:.4f}</td>
              <td>{row['Recall']:.4f}</td>
              <td>
                <div class="f1cell">
                  <span>{row['F1']:.4f}</span>
                  <div class="f1bar"><i style="--w:{f1_pct:.1f}%;"></i></div>
                </div>
              </td>
              <td>{row['ROC_AUC']:.4f}</td>
            </tr>
            """)

        ui.render(f"""
        <div class="card" style="padding:0; overflow-x:auto;">
          <table class="board">
            <thead>
              <tr>
                <th style="width:4rem;">Rank</th>
                <th>Algorithm</th>
                <th>Review Part</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th style="text-align:right;">F1-Score (Weighted)</th>
                <th>ROC-AUC (OvR)</th>
              </tr>
            </thead>
            <tbody>
              {"".join(rows_html)}
            </tbody>
          </table>
        </div>
        """)

        # Tuning Delta & Comparison
        col_t1, col_t2 = st.columns([1, 1], gap="large")
        with col_t1:
            ui.render(f"""
            <div class="card d2" style="margin-top:1.5rem;">
              <div class="card-head">
                <div class="t">
                  {ui.icon("cpu", 18)}
                  <h3>Hyperparameter Tuning Impact ({sel_model})</h3>
                </div>
                <span class="chip accent"><span class="dot"></span>OPTIMIZED</span>
              </div>
              <p class="muted" style="font-size:0.85rem; margin-top:-0.5rem;">
                5-Fold Stratified Cross-Validation on training split with RandomizedSearchCV.
              </p>
            """)

            ba = metrics["classification"]["before_after"]
            ba_df = pd.DataFrame(ba)
            st.dataframe(
                ba_df.style.format("{:.4f}").map(
                    lambda v: "color: #3E6F4B; font-weight:600;" if isinstance(v, (int, float)) and v > 0 else "",
                    subset=["Change"]
                ),
                use_container_width=True
            )
            ui.render("</div>")

        with col_t2:
            ui.render(f"""
            <div class="card d3" style="margin-top:1.5rem;">
              <div class="card-head">
                <div class="t">
                  {ui.icon("trending", 18)}
                  <h3>F1 vs ROC-AUC Performance Map</h3>
                </div>
              </div>
            """)

            fig = px.scatter(
                df_table,
                x="F1",
                y="ROC_AUC",
                text="Algorithm",
                color="Part",
                color_discrete_map={"A": ui.INK_SOFT, "B": ui.ACCENT},
                hover_data=["Accuracy", "Precision", "Recall"],
            )
            fig.update_traces(textposition="top center", marker=dict(size=12, line=dict(width=1, color="DarkSlateGrey")))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=20),
                font=dict(family="Geist, sans-serif", color=ui.INK),
                xaxis=dict(gridcolor=ui.LINE, title="Weighted F1-Score"),
                yaxis=dict(gridcolor=ui.LINE, title="ROC-AUC (OvR)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True)
            ui.render("</div>")
    else:
        st.info("Leaderboard metrics will display here after execution.")

# ==============================================================================
# TAB 3: CLUSTERING (PCA 2D & MARKET SEGMENTS)
# ==============================================================================
with tab_cluster:
    ui.render(f"""
    <div class="card" style="margin-bottom:1.5rem;">
      <div class="card-head">
        <div>
          <span class="eyebrow">UNSUPERVISED CLUSTERING TRACK</span>
          <h2 class="display" style="font-size:2rem; margin:0.3rem 0 0;">Market Segmentation & PCA Projection</h2>
        </div>
        <span class="chip"><span class="dot"></span>NO LABELS USED IN TRAINING</span>
      </div>
      <p class="muted" style="margin:0; font-size:0.92rem; max-width:55rem;">
        Trained on 6,894 complete titles using regional sales (NA, EU, JP, Other) and engagement signals (Critic Score, User Score, Review counts).
        Evaluated using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.
      </p>
    </div>
    """)

    if cluster_pts is not None and metrics and "clustering" in metrics:
        col_pca, col_prof = st.columns([1.3, 1], gap="large")

        with col_pca:
            ui.render(f"""
            <div class="card">
              <div class="card-head">
                <div class="t">
                  {ui.icon("scatter", 18)}
                  <h3>PCA 2D Dimensionality Reduction (k={metrics['clustering']['k']})</h3>
                </div>
                <span class="muted mono" style="font-size:0.75rem;">
                  Explained Var: {sum(metrics['clustering']['pca_variance'])*100:.1f}%
                </span>
              </div>
            """)

            clu_palette = [ui.ACCENT, ui.INK, ui.TEAL, "#B89A5E", "#8C6A9E"]
            cluster_pts["Cluster_Name"] = cluster_pts["KMeans_Cluster"].apply(lambda c: f"Cluster {c}")

            fig_pca = px.scatter(
                cluster_pts,
                x="PC1",
                y="PC2",
                color="Cluster_Name",
                color_discrete_sequence=clu_palette,
                hover_data=["Name", "Platform", "Genre", "Year_of_Release", "Global_Sales", "Critic_Score"],
                opacity=0.6,
            )
            fig_pca.update_traces(marker=dict(size=5))
            fig_pca.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                font=dict(family="Geist, sans-serif", color=ui.INK),
                xaxis=dict(gridcolor=ui.LINE, title="Principal Component 1"),
                yaxis=dict(gridcolor=ui.LINE, title="Principal Component 2"),
                height=520,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_pca, use_container_width=True)
            ui.render("</div>")

        with col_prof:
            ui.render(f"""
            <div class="card" style="margin-bottom:1rem;">
              <div class="card-head">
                <div class="t">
                  {ui.icon("target", 18)}
                  <h3>Internal Cluster Validation Metrics</h3>
                </div>
              </div>
            """)
            clu_metrics_df = pd.DataFrame(metrics["clustering"]["table"])
            st.dataframe(clu_metrics_df.style.format({
                "Silhouette": "{:.4f}",
                "Davies_Bouldin": "{:.4f}",
                "Calinski_Harabasz": "{:.1f}",
            }), use_container_width=True)
            ui.render("</div>")

            # Cluster Profile Cards
            if cluster_prof is not None:
                seg_cards = []
                for idx, row in cluster_prof.iterrows():
                    c_color = clu_palette[int(idx) % len(clu_palette)]
                    seg_cards.append(f"""
                    <div class="seg" style="--c:{c_color};">
                      <div class="hd"><i style="background:{c_color};"></i><span class="eyebrow">CLUSTER {idx}</span></div>
                      <h4>Segment Profile {idx}</h4>
                      <div class="kv">
                        <div><span class="muted">Titles:</span> <b>{int(row['Games']):,}</b></div>
                        <div><span class="muted">Median Sales:</span> <b>{row['Global_Sales_median']:.2f}M</b></div>
                        <div><span class="muted">Avg Critic:</span> <b>{row['Critic_Score']:.1f}/100</b></div>
                        <div><span class="muted">Avg User:</span> <b>{row['User_Score']:.1f}/10</b></div>
                      </div>
                      <div class="ex">
                        <b>Regional:</b> NA {row['NA_Sales']:.2f}M • EU {row['EU_Sales']:.2f}M • JP {row['JP_Sales']:.2f}M
                      </div>
                    </div>
                    """)

                ui.render(f"""
                <div class="card">
                  <div class="card-head">
                    <div class="t">{ui.icon("layers", 18)}<h3>Cluster Archetypes</h3></div>
                  </div>
                  <div class="segs">
                    {"".join(seg_cards)}
                  </div>
                </div>
                """)
    else:
        st.info("Clustering projection and metrics will display here after execution.")

# ==============================================================================
# TAB 4: METHODOLOGY & ARCHITECTURE
# ==============================================================================
with tab_method:
    ui.render(f"""
    <div class="card" style="margin-bottom:1.5rem;">
      <span class="eyebrow">CAPSTONE SPECIFICATION COMPLIANCE</span>
      <h2 class="display" style="font-size:2.2rem; margin:0.3rem 0 0.8rem;">Project Pipeline & Rubric Integrity</h2>
      <p class="muted" style="margin:0; font-size:0.95rem; line-height:1.6; max-width:55rem;">
        Built strictly in accordance with <b>23CSE301 Machine Learning Guidelines & Rubrics</b>.
        Demonstrates complete data integrity: stratified splitting, training-only preprocessor fitting,
        systematic evaluation across regression, classification, and clustering tracks, and bonus web interface.
      </p>
    </div>

    <div class="card" style="margin-bottom:1.5rem;">
      <div class="card-head">
        <div class="t">{ui.icon("split", 18)}<h3>5-Stage Machine Learning Pipeline</h3></div>
      </div>
      <div class="steps">
        <div class="step">
          {ui.icon("database", 20)}
          <b>1. Audit & EDA</b>
          <span>16,719 rows audited. Distribution skew, missingness, correlation heatmaps, target analysis.</span>
        </div>
        <div class="step">
          {ui.icon("broom", 20)}
          <b>2. Preprocessing</b>
          <span>Median imputation, standard scaling, one-hot encoding fit strictly on training split only.</span>
        </div>
        <div class="step">
          {ui.icon("trending", 20)}
          <b>3. Regression (10)</b>
          <span>Linear, Ridge, Lasso, ElasticNet, Poly, DecisionTree, RF, GradBoost, SVR, KNN. R², RMSE, MAE.</span>
        </div>
        <div class="step">
          {ui.icon("layers", 20)}
          <b>4. Classification (10)</b>
          <span>Parts A & B unified. Accuracy, Precision, Recall, Weighted F1, ROC-AUC (OvR) on common test split.</span>
        </div>
        <div class="step">
          {ui.icon("scatter", 20)}
          <b>5. Clustering (2)</b>
          <span>K-Means (Elbow) & Agglomerative (Dendrogram). Silhouette, Davies-Bouldin, Calinski-Harabasz, PCA, t-SNE.</span>
        </div>
      </div>
    </div>
    """)
