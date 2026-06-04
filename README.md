# 📊 Macro-Economic Scenario Planner (MESP)
### Quantifying Macroeconomic Shocks on US Ready-to-Eat Cereal Demand

> **MTech Industry Project** — General Mills India × NMIMS MPSTME | Dec 2025 – May 2026  
> Industry Mentor: Ms. Manali Yardi (Manager, Data Science, PCA — General Mills)  
> Academic Supervisor: Prof. Swati Vaishnav (MPSTME, NMIMS)  
>
> *Note: Source code, data, and all quantitative outputs are confidential to General Mills India and cannot be shared publicly. This repository documents the methodology, architecture, and problem-solving approach.*

---

## 🎯 Business Problem

General Mills' RTE Cereal category runs a **3-year baseline forecast** built on historical sales trends — robust extrapolation, but fundamentally backward-looking. The gap:

> *"If CPI rises 1% above baseline — what happens to our cereal volume?"*

Without a quantitative tool to answer this, scenario analysis required **manual, ad-hoc estimation** by finance and commercial teams — slow, inconsistent across analysts, and untraceable to a validated model.

**This project delivers that missing capability.**

---

## 📈 Context

The project integrates monthly US cereal retail sales data with four macroeconomic indicators spanning several years of US retail history:
- **CPI Food at Home** (BLS)
- **Civilian Unemployment Rate** (BLS)
- **Consumer Sentiment Index** (University of Michigan)
- **Real GDP Growth** (BEA, quarterly → monthly via linear interpolation)

The cereal category experienced significant post-COVID structural dynamics — a divergence between revenue and volume trajectories that made macro scenario planning operationally critical for leadership.

---

## 🏗️ Architecture — Two-Stage Log-Log Ridge Regression

```
Macroeconomic Inputs
┌─────────────────────────────────────────────────┐
│  CPI Food at Home  │  Unemployment  │  GDP Growth │
└──────────────────────────┬──────────────────────┘
                           │  log-transform (YoY %)
                           ▼
              ┌─────────────────────────┐
              │   STAGE 1: RidgeCV      │
              │   Macro → Avg Price     │
              │   TimeSeriesSplit CV    │
              └───────────┬─────────────┘
                          │  Predicted log(Price)
                          │
Consumer Sentiment ───────┤
(direct volume channel)   │
                          ▼
              ┌──────────────────────────────────┐
              │   STAGE 2: RidgeCV               │
              │   Price + Sentiment → Volume     │
              │   Target: STL De-Seas. Volume    │
              └───────────┬──────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │   CHAIN-RULE COMPOSITION           │
         │   ε(macro→vol) = ε(macro→price)   │
         │                × ε(price→vol)      │
         └────────────────┬───────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │   SENSITIVITY TABLE                │
         │   Unit volume Δ per 1% macro shift │
         │   Instantly usable by business     │
         └────────────────────────────────────┘
```

### Why Two Stages?

A single-stage model regressing macro drivers directly on volume conflates two economically distinct mechanisms:

1. **Price transmission channel** — CPI / Unemployment / GDP → shelf price → volume (indirect, mediated)
2. **Confidence channel** — Consumer Sentiment → volume (direct, bypasses price)

Separating these allows practitioners to independently modify price pass-through assumptions (Stage 1) or demand elasticity assumptions (Stage 2) without refitting the full model — a critical property for flexible scenario planning across different economic regimes.

---

## 🔬 Methodology

See [`methodology.md`](methodology.md) for full technical detail. Summary of the five-phase pipeline:

### Phase 1 — Data Integration
Five sources merged on a common monthly period index. Key preprocessing: GDP published quarterly is linearly interpolated to monthly; YoY growth rate computed as a 12-month percentage change.

### Phase 2 — Exploratory Data Analysis (10 Workstreams)
- Time series trend analysis (revenue, volume, price trajectories)
- Seasonality analysis — calendar pattern identification and quantification
- Year-over-Year decomposition
- Price analysis and empirical elasticity estimation (scatter + OLS)
- Pearson / Spearman correlation across all variables
- Z-score outlier detection and COVID structural break analysis
- Augmented Dickey-Fuller (ADF) stationarity testing — all series confirmed I(1)
- Classical additive decomposition + ACF/PACF analysis
- COVID-19 three-phase comparison (pre / peak / post)

### Phase 3 — Feature Relevance Testing (Dual-Channel Routing)
Each macro driver evaluated against **both** the price channel and the volume channel using four metrics:
- Pearson correlation
- Spearman rank correlation
- Univariate Ridge R²
- Mutual Information

**Routing outcome:** CPI, Unemployment, and GDP Growth route to Stage 1 (price channel dominates). Consumer Sentiment routes to Stage 2 (volume channel dominates — supported by the defensive-category business mechanism).

### Phase 4 — Two-Stage Log-Log Ridge Regression

**STL Decomposition** (pre-Stage 2):
- Applied to log(EQ Volume) using `robust=True` to limit outlier influence
- De-seasonalised volume = exp(trend + residual) → Stage 2 target
- Multiplicative seasonal factor = exp(seasonal component) → re-seasonalisation

**Stage 1 — Macro → Average Equivalent Price:**
- Log-log transformation of all inputs and target
- GDP Growth sign-preserving transform: `sign(x) × log1p(|x|/100)`
- VIF check confirms no problematic multicollinearity
- RidgeCV with 200 log-spaced α values and 5-fold `TimeSeriesSplit`
- Coefficients back-transformed to elasticity space via scaler std devs

**Stage 2 — Price + Sentiment → De-Seasonalised Volume:**
- Log-log specification — coefficients directly interpretable as elasticities
- Price elasticity captures direct demand response to shelf price changes
- Sentiment elasticity captures defensive-category mechanism (confidence ↓ → at-home cereal ↑)

**Chain-Rule Composition:**
```
ε(CPI → volume)   = ε(CPI → price) × ε(price → volume)
ε(Unemp → volume) = ε(Unemp → price) × ε(price → volume)
ε(GDP → volume)   = corrected delta (percentage-scale input)
ε(Sentiment → volume) = direct Stage 2 coefficient
```

### Phase 5 — Sensitivity Table
Translates chain-rule elasticities into absolute unit volume impacts per 1% macro shift at baseline volume — directly answering the business question for finance and IBP teams.

---

## 📊 Key EDA Insights (Qualitative)

**Revenue–Volume Structural Divergence (Post-2021)**  
Revenue and volume diverged sharply post-COVID — category revenue growth was entirely price-driven, not volume recovery. This structural dynamic is the primary context motivating the scenario planner.

**CPI–Price Transmission**  
Near-perfect correlation between CPI Food at Home and average shelf price — food cost inflation passes through to cereal prices with high fidelity and low lag. CPI is consequently the highest-impact scenario lever.

**Elastic Demand**  
Empirical price elasticity computed month-over-month confirms elastic demand: price increases generate proportionally larger volume decreases. The Stage 2 model refines this on de-seasonalised volume.

**Defensive Category Mechanism**  
Consumer Sentiment shows a counter-intuitive negative relationship with revenue but positive with volume — households shift toward affordable at-home meals when confidence falls, benefiting cereal volumes.

**Strong Quarter-End Seasonality**  
Revenue peaks consistently in March, June, September, and December — promotional cycles drive this pattern. Seasonal strength is very high, confirming that STL de-seasonalisation is essential before macro regression.

**January 2022 Outlier**  
A trade-loading event (z-score well above threshold) identified and confirmed as non-consumer-demand signal. Handled via STL `robust=True`.

---

## 🏢 Business Applications

| Function | Use Case |
|---|---|
| **Finance / IBP** | Generate low/base/high volume scenarios from published macro outlook ranges (Oxford Economics, IHS Markit) |
| **Commercial Pricing** | Simulate volume impact of price hold vs. competitor price increase — Stage 2 elasticity directly applicable |
| **Supply Chain / S&OP** | Adjust production plans and procurement ahead of macro shifts, not reactively after POS signals arrive |
| **Brand Planning** | Adapt framework to brand-level data for SKU/brand revenue planning |
| **Cross-Category** | Methodology reusable for any FMCG category with historical sales + macro data |

---

## ⚠️ Limitations

| Limitation | Impact |
|---|---|
| Short training window | Limits statistical precision; improves with each quarter of new data |
| Constant elasticity assumption | Cannot capture regime-switching — pass-through rates differ across high/low inflation environments |
| No competitive variables | Omits private-label pricing, promotional intensity — potential omitted variable bias |
| Category-level only | Brand/SKU disaggregation requires additional modelling layer |
| Point estimates only | No confidence intervals — bootstrap resampling recommended for enterprise use |

---

## 🚀 Future Scope

1. **Rolling re-estimation** — quarterly cadence, auto-incorporating new months
2. **Bootstrap confidence intervals** — block-bootstrap draws for uncertainty quantification around each sensitivity estimate
3. **Competitive variable inclusion** — private-label price index, promotional intensity
4. **Brand/SKU disaggregation** — separate elasticity models per brand tier
5. **Regime-switching model** — Markov-switching regression for state-dependent elasticities
6. **Interactive dashboard** — Streamlit/Dash app with macro sliders and real-time volume readouts
7. **External macro forecast integration** — direct feed from Oxford Economics / CBO outlook revisions

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | `pandas`, `numpy`, `openpyxl` |
| Statistics | `statsmodels` (STL, ADF, VIF) |
| Modelling | `scikit-learn` (RidgeCV, TimeSeriesSplit, StandardScaler) |
| Visualisation | `matplotlib`, `seaborn` |
| Environment | Jupyter Notebook |

---

## 📚 Key References

1. Bijmolt et al. (2005) — Meta-analysis of price elasticity estimates across FMCG categories
2. Nakamura & Steinsson (2008) — Retail food price transmission mechanisms and lags
3. Cleveland et al. (1990) — STL decomposition (original paper)
4. Hoerl & Kennard (1970) — Ridge Regression
5. Baron & Kenny (1986) — Mediation analysis framework (two-stage model justification)
6. Hyndman & Athanasopoulos (2021) — *Forecasting: Principles and Practice* (STL best practices)

Full reference list in [`references.md`](references.md).

---

## 👤 Author

**Jeet Nakrani** (D010)  
MTech Data Science & Business Analytics — NMIMS MPSTME  
Industry Project @ General Mills India (CMI Function) | Dec 2025 – May 2026  
[LinkedIn](https://linkedin.com/in/your-profile) · [Portfolio](https://your-portfolio.com)

---

> *All quantitative outputs, model results, and General Mills data are proprietary and confidential. This repository shares only the methodology, architecture, and problem-solving approach of the MTech industry project.*
