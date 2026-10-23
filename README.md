# 23CSE301 — Machine Learning Capstone Project
**Video Game Market Intelligence & Predictive Sales Architecture**  
*B.Tech. Computer Science and Engineering — Academic Year 2026–27*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/Academic-Capstone-lightgrey.svg)]()

---

## 1. Executive Summary & Problem Statement

This repository hosts the end-to-end Machine Learning pipeline for the **23CSE301 ML Capstone Project**, encompassing all specifications across **Review 1** and **Review 2**.

The commercial success of video game titles exhibits extreme variance and heavy right-skewed distribution—a rare minority of blockbuster releases generate the overwhelming share of revenue, while thousands of titles struggle to break even. This project investigates and models video game commercial performance across three distinct machine learning tracks:

1. **Regression Track (10 Algorithms):** Predicting exact continuous `Global_Sales` (in millions of units) from release attributes, platform indicators, and critic/user reception signals.
2. **Classification Track (10 Algorithms — Parts A & B):** Categorizing titles into distinct commercial tiers (`Low`: $<0.2\text{M}$, `Moderate`: $0.2\text{M}-1.0\text{M}$, `Hit`: $\ge 1.0\text{M}$) and conducting a consolidated tournament across all 10 algorithms on identical test splits.
3. **Unsupervised Clustering Track (2 Algorithms):** Discovering intrinsic market segment archetypes using multi-regional sales distributions and user engagement volume without labels, evaluated via internal cluster validity indices, PCA 2D projections, and t-SNE manifolds.
4. **Interactive GUI Application (Bonus Opportunity):** A production-grade Streamlit web interface featuring real-time tier forecasting, class probability distributions, model leaderboards, and interactive 2D cluster exploration.

---

## 2. Dataset Description & Preprocessing Pipeline

- **Source File:** `data/Video_Games_Sales_as_at_22_Dec_2016.csv`
- **Volume:** 16,719 game titles, 16 attributes spanning 1980 to 2016.
- **Attributes:**
  - *Categorical:* `Platform` (31 unique hardware platforms), `Genre` (12 categories), `Publisher`, `Developer`, `Rating` (ESRB classifications: E, T, M, E10+, etc.).
  - *Numerical (Sales):* `NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`, `Global_Sales` (millions of units).
  - *Numerical (Reception & Activity):* `Critic_Score` (0–100), `Critic_Count`, `User_Score` (0–10), `User_Count`, `Year_of_Release`.

### Data Cleaning & Methodological Safeguards
- **Audit & Type Resolution:** The string `"tbd"` ("to be decided") in `User_Score` was coerced to `NaN`, restoring column numerical types.
- **Missing Value Handling:** Titles missing critical structural keys (`Global_Sales`, `Year_of_Release`, `Genre`) were dropped ($16,448$ titles retained). Median imputation was used for continuous review scores (robust to skew); mode imputation was used for categorical features.
- **Strict Data-Leakage Prevention (Rubric Rule 7.1):** All imputers, standardizers (`StandardScaler`), and categorical encoders (`OneHotEncoder`) were **fitted strictly on the training split only** ($80\%$) and applied via out-of-fold transformation to the test split ($20\%$).
- **Stratified Partitioning:** Continuous sales in regression were binned into quintiles for stratified train/test splitting; classification used stratified splits on `Sales_Tier` to guarantee identical class support across splits ($13,158$ train / $3,290$ test).
- **Engineered Feature:** `Score_Gap = (Critic_Score / 10) - User_Score`.
  *Justification:* Divergence between professional critics and grassroots player reception captures grassroots polarization and word-of-mouth dynamics not reflected in raw scores alone.

---

## 3. Results Summary Tables

### 3.1 Consolidated 10-Classifier Tournament Table (Review 2 Core Deliverable)
All 10 algorithms were trained on the identical training set and evaluated on the same held-out test split ($3,290$ samples). Multiclass ROC-AUC is computed using One-vs-Rest (OvR), weighted by class support.

| Rank | Algorithm | Review Part | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | ROC-AUC (OvR) | Status |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Gradient Boosting** | **Part B** | **0.6675** | **0.6554** | **0.6675** | **0.6507** | **0.7904** | **Selected** |
| **2** | Random Forest | Part B | 0.6526 | 0.6424 | 0.6526 | 0.6440 | 0.7631 | Runner-up |
| **3** | MLP Neural Network | Part B | 0.6620 | 0.6479 | 0.6620 | 0.6411 | 0.7830 | Top 3 |
| **4** | Bagging (Decision Tree) | Part B | 0.6459 | 0.6365 | 0.6459 | 0.6388 | 0.7652 | Ensemble |
| **5** | Support Vector Machine (SVC) | Part A | 0.6526 | 0.6432 | 0.6526 | 0.6286 | 0.7615 | Top Part A |
| **6** | Decision Tree Classifier | Part A | 0.6368 | 0.6183 | 0.6368 | 0.6097 | 0.7477 | Baseline |
| **7** | Logistic Regression | Part A | 0.6292 | 0.6126 | 0.6292 | 0.6076 | 0.7398 | Baseline |
| **8** | K-Nearest Neighbors (KNN) | Part A | 0.6185 | 0.6026 | 0.6185 | 0.6050 | 0.7308 | Baseline |
| **9** | AdaBoost | Part B | 0.6283 | 0.6077 | 0.6283 | 0.5963 | 0.7345 | Ensemble |
| **10** | Gaussian Naive Bayes | Part A | 0.2416 | 0.5747 | 0.2416 | 0.1827 | 0.6393 | Lower Bound |

> **Key Tournament Insight:** Part B ensemble algorithms decisively outperformed Part A single estimators. Boosting and Bagging overcome the high variance of individual decision trees, while Gaussian Naive Bayes fails due to extreme feature codependence between review counts and mutual exclusivity of one-hot platform indicators.

### 3.2 Hyperparameter Tuning & Final Model Selection
- **Selected Champion Model:** **Gradient Boosting Classifier**
- **5-Fold Stratified Cross-Validation on Training Split:**
  - $\text{Mean CV } F1 = 0.6451 \pm 0.0052$
  - $\text{Mean CV ROC-AUC} = 0.7856$
- **Optimization Strategy:** `RandomizedSearchCV` exploring `n_estimators`, `learning_rate`, `max_depth`, and `subsample`.
- **Tuned Model Generalization Improvement:**

| Metric | Untuned Gradient Boosting | Tuned Gradient Boosting | Absolute Delta |
|:---|:---:|:---:|:---:|
| **Test Accuracy** | $0.6675$ | **$0.6726$** | $+0.0051$ |
| **Test Precision (Weighted)** | $0.6554$ | **$0.6616$** | $+0.0062$ |
| **Test Recall (Weighted)** | $0.6675$ | **$0.6726$** | $+0.0051$ |
| **Test F1-Score (Weighted)** | $0.6507$ | **$0.6600$** | **$+0.0093$** |
| **Test ROC-AUC (OvR)** | $0.7904$ | **$0.7992$** | **$+0.0088$** |

---

### 3.3 Regression Track Summary (10 Algorithms, Review 1)
Evaluated on continuous `Global_Sales` (held-out test split, ranked by $R^2$):

| Rank | Algorithm | $R^2$ Score | RMSE ($M$) | MAE ($M$) | 5-Fold CV $R^2$ |
|:---:|:---|:---:|:---:|:---:|:---:|
| **1** | Random Forest Regressor | **0.2326** | **1.7913** | 0.4454 | 0.2038 |
| **2** | Support Vector Regressor (SVR) | 0.1916 | 1.8384 | **0.3993** | **0.2645** |
| **3** | KNN Regressor | 0.1880 | 1.8425 | 0.4824 | 0.1669 |
| **4** | Polynomial Regression (deg=2) | 0.1786 | 1.8531 | 0.5086 | 0.1906 |
| **5** | Gradient Boosting Regressor | 0.1743 | 1.8579 | 0.4547 | 0.2472 |
| **6** | Decision Tree Regressor | 0.1244 | 1.9133 | 0.4790 | -0.0383 |
| **7** | Ridge Regression | 0.0745 | 1.9671 | 0.5550 | 0.1815 |
| **8** | Linear Regression | 0.0744 | 1.9671 | 0.5552 | 0.1815 |
| **9** | ElasticNet Regression | 0.0699 | 1.9720 | 0.5492 | 0.1665 |
| **10** | Lasso Regression | 0.0639 | 1.9783 | 0.5540 | 0.1561 |

*(Note: Degree-3 Polynomial Regression exhibited catastrophic divergence $R^2 < -7.0$ due to exponential feature space expansion, demonstrating severe overfitting).*

---

### 3.4 Unsupervised Clustering Track (Review 2 Core Deliverable)
Fitted on $6,894$ complete titles using 6 log-transformed regional sales & activity volume features plus 2 score dimensions. **Zero labels were exposed during fitting.**

| Algorithm | Clusters ($k$) | Linkage | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | Concordance (ARI) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **K-Means Clustering** | 4 | N/A (Centroid) | **0.2095** | **1.4089** | **2,135.0** | \multirow{2}{*}{$0.4471$} |
| **Agglomerative Hierarchical** | 4 | Ward's Variance | 0.1678 | 1.5270 | 1,784.3 | |

#### Market Archetype Profiles Discovered ($k=4$)
- **Cluster 0 — Global Mega-Blockbusters ($n=302$):** High average global sales ($>4.8\text{M}$ units), dominant Western market presence ($2.97\text{M}$ NA, $2.22\text{M}$ EU), massive review volume ($>1,000$ user reviews). Over $85\%$ map to the *Hit* tier.
- **Cluster 1 — Japan-Heavy & Specialized Niche Titles ($n=3,137$):** Moderate-to-low Western sales but distinct Japanese market engagement.
- **Cluster 2 — Mainstream Mid-Tier Hits ($n=1,916$):** Solid critic scores ($\approx 80/100$), steady cross-regional adoption ($0.89\text{M}$ median global sales), predominantly *Moderate* tier.
- **Cluster 3 — Budget & Low-Engagement Titles ($n=1,539$):** Modest Metascores ($\approx 52/100$), low review counts, minimal commercial velocity ($0.16\text{M}$ median sales). Over $87\%$ map to the *Low* tier.

---

## 4. Repository Structure

This repository strictly complies with the capstone organization schema:

```text
ML_PROJECT/
├── README.md                               # Project overview, consolidated metrics, setup guide
├── requirements.txt                         # Pinned Python package dependencies
├── 23CSE301_ML_26_27_Capstone_Guidelines.pdf # Official capstone syllabus and evaluation rubric
│
├── data/
│   └── Video_Games_Sales_as_at_22_Dec_2016.csv # Primary raw dataset (16,719 rows)
│
├── notebooks/
│   └── ML_Capstone_VideoGameSales.ipynb    # Clean, fully-executed, zero-error master notebook
│
├── models/
│   ├── sales_tier_classifier.joblib        # Deployed Pipeline (Preprocessor + Tuned Gradient Boosting)
│   ├── global_sales_regressor.joblib       # Indicative Sales Regressor (Tuned GB Regressor)
│   ├── cluster_points.csv                  # 2D PCA projected coordinates with cluster memberships
│   ├── cluster_profile.csv                 # Cluster centroid statistics and commercial profiles
│   ├── classification_comparison.csv       # Consolidated 10-classifier metric export
│   └── metrics.json                        # Serialized benchmark metrics for GUI consumption
│
├── app/
│   ├── app.py                              # Streamlit interactive application
│   └── ui.py                               # Custom minimalist aesthetic engine, SVG icons, animations
│
└── .streamlit/
    └── config.toml                         # Warm paper theme & typography configuration
```

---

## 5. Environment Setup & How-to-Run Instructions

### 5.1 Prerequisites
- Python 3.10, 3.11, or 3.12 installed.
- Recommended: Virtual environment (`venv` or `conda`).

### 5.2 Installation
Clone or navigate to the repository directory and install all dependencies:
```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# 2. Install all pinned requirements
pip install -r requirements.txt
```

### 5.3 Executing the Master Jupyter Notebook
The notebook runs end-to-end without interactive prompts and automatically resolves data paths relative to the project root:
```bash
jupyter notebook notebooks/ML_Capstone_VideoGameSales.ipynb
```
*To re-run headlessly via terminal:*
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/ML_Capstone_VideoGameSales.ipynb
```

### 5.4 Launching the Interactive Web Application (Bonus GUI)
Launch the Streamlit web application:
```bash
streamlit run app/app.py
```
Once launched, open your web browser at `http://localhost:8501`.

---

## 6. Web Application Interface & Design Aesthetics

The web GUI was designed under modern UI guidelines:
- **Restrained Editorial Palette:** Warm textured paper background (`#F3F0E8`), high-contrast ink typography (`#1E1D1B`), and a single terracotta/burnt-orange accent (`#C8512A`). Avoids generic AI aesthetics.
- **Interactive SVG Visuals:** Hand-crafted, lightweight inline SVG icons with stroke-draw entry animations.
- **Hardware-Accelerated CSS Micro-Animations:** Progress rings, probability distribution transitions, and hover-responsive cards utilizing compositor threads with full `prefers-reduced-motion` accessibility support.
- **4 Dedicated Modules:**
  1. *Market Forecaster & Simulator:* Real-time parameter tweaking with immediate tier prediction, confidence ring, probability breakdown, and historical benchmark title matching.
  2. *Classification Leaderboard:* Comprehensive 10-classifier sortable tournament table, interactive F1 vs ROC-AUC scatter map, and hyperparameter delta analysis.
  3. *Market Segment Explorer:* Interactive Plotly 2D PCA cluster map with hover inspect and archetype financial cards.
  4. *Architecture & Methodology:* Detailed 5-stage pipeline diagram and rubric requirement checklist.

---

## 7. Compliance Checklist

- [x] **Section A — Review 1 Part B Classification:** All 5 Part-B algorithms implemented and trained on identical split without errors.
- [x] **Consolidated 10-Model Comparison:** Single unified table comparing Accuracy, Precision, Recall, F1, and ROC-AUC.
- [x] **Final Model Selection:** Formally documented with 5-fold cross-validation justification and hyperparameter tuning progression.
- [x] **Section B — Clustering Track:** Both K-Means and Agglomerative Hierarchical clustering implemented; cluster labels extracted.
- [x] **Cluster Validation Indices:** Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index reported.
- [x] **Required Visualizations:** K-Means Elbow Curve, Hierarchical Dendrogram, PCA 2D scatter plots for all algorithms, and t-SNE non-linear projection.
- [x] **Code Quality & Cleanliness:** Modular functions, commented logic, random seeds set (`random_state=42`), zero execution errors.
- [x] **Bonus Deliverables:** Working Streamlit GUI with live predictions and deployment architecture.
