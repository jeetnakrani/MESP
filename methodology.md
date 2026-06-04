# Methodology — Macro-Economic Scenario Planner

Full technical walkthrough of the five-phase pipeline. All quantitative outputs are proprietary to General Mills and omitted per confidentiality agreement.

---

## Phase 1 — Data Integration

### Sources and Variables

| Dataset | Source | Variables | Period | Frequency |
|---|---|---|---|---|
| Cereal Sales | General Mills Internal | eq_vol, dol_val, avg_eq_price | Multi-year US retail history | Monthly |
| CPI Food at Home | BLS | cpi_home | Aligned to sales period | Monthly |
| Unemployment Rate | BLS | unemp | Aligned to sales period | Monthly |
| Consumer Sentiment | U. Michigan | sent | Aligned to sales period | Monthly |
| Real GDP | BEA | gdp, gdp_growth | Aligned to sales period | Quarterly → Monthly |

### GDP Interpolation
GDP is published quarterly. Raw data contains the same quarterly figure repeated across all three months (stair-step pattern). Fix:
```python
# Replace duplicate quarterly values with NaN, then linear interpolate
df['gdp'] = df['gdp'].where(df['gdp'] != df['gdp'].shift(1))
df['gdp'] = df['gdp'].interpolate(method='linear')
df['gdp_growth'] = df['gdp'].pct_change(12) * 100  # 12-month YoY %
```

All timeseries aligned to a common monthly period index and merged on shared month key. Series continuity confirmed — no missing months.

---

## Phase 2 — Exploratory Data Analysis (10 Workstreams)

### 2.1 Summary Statistics
Descriptive statistics computed for all three cereal sales variables (revenue, equivalent volume, average price) and all four macro indicators. Coefficient of variation quantifies relative variability across series.

### 2.2 Time Series Trends
Three structural trends characterise the study period — revenue trajectory, volume trajectory, and price trajectory — analysed via dual-axis visualisation and 12-month rolling averages. The divergence between revenue and volume post-COVID is the defining structural finding of the EDA.

### 2.3 Seasonality Analysis
Strong quarter-end seasonal pattern confirmed by three analytical approaches:
- Monthly average bar chart
- Year-by-month revenue heatmap
- Calendar-month boxplots (inter-quartile range analysis)

Seasonal peaks occur in March, June, September, and December — corresponding to quarter-end promotional cycles.

### 2.4 Year-over-Year Analysis
Annual aggregation over all complete years. Revenue vs. volume YoY transitions classified to distinguish price-driven vs. volume-driven growth episodes.

### 2.5 Price Analysis & Empirical Elasticity
Price trajectory computed (CAGR, cumulative change, peak year). Empirical price elasticity estimated month-over-month:
```python
elasticity = (pct_change_volume) / (pct_change_price)
# Trim extreme values beyond ±20 before computing median
```
Scatter of price vs. volume (coloured by year) with OLS fit quantifies the empirical price-demand relationship. Confirms elastic demand in the category.

### 2.6 Correlation Analysis (Pearson + Spearman)
Full correlation matrix across all variables. Key relationships identified:
- CPI ↔ Avg Price: near-perfect positive correlation — cost pass-through confirmed
- Consumer Sentiment ↔ EQ Volume: negative correlation — defensive category mechanism confirmed
- Unemployment ↔ Avg Price: positive — cost pressure during high-unemployment episodes

### 2.7 Outlier Detection (Z-Score)
Z-score analysis identifies a single high-confidence outlier in the revenue series corresponding to a known trade-loading event. Not a consumer demand signal.

**Handling:** STL `robust=True` absorbs into residual without explicit imputation. For standalone SARIMA models: binary intervention dummy recommended.

**COVID Period (structural break, not outlier):** Three-phase comparative analysis — Pre-COVID / COVID Peak / Post-COVID. Revenue, volume, and price comparison across phases establishes the post-COVID baseline regime.

### 2.8 Stationarity Testing (ADF)
Augmented Dickey-Fuller test applied to all variables in levels and first differences.
- **Levels:** Revenue, volume, price, CPI, GDP confirmed non-stationary (p > 0.05)
- **First differences:** All series confirmed stationary (p < 0.01)
- **Result:** All series are I(1) — integrated of order 1

### 2.9 Classical Decomposition + ACF/PACF
Additive decomposition (period = 12) applied to revenue. Seasonal, trend, and residual contribution percentages quantified. ACF/PACF analysis informs SARIMA specification for any standalone volume forecast model.

### 2.10 COVID-19 Phase Analysis
Three-phase segmentation (Pre-COVID / COVID Peak / Post-COVID) with average revenue, volume, and price per phase. Key finding: post-COVID revenue premium is entirely attributable to price inflation — volume has not recovered to pre-COVID levels.

---

## Phase 3 — Feature Relevance Testing (Dual-Channel Routing)

Each macro driver evaluated against **both** the price channel and the volume channel using four metrics:

| Metric | What It Tests |
|---|---|
| Pearson correlation | Linear relationship strength |
| Spearman rank correlation | Monotonic relationship (rank-based, robust to outliers) |
| Univariate Ridge R² | Predictive power in a regularised linear framework |
| Mutual Information | Non-linear association (model-free) |

**Routing decision rule:** Driver assigned to the channel where its aggregate signal strength (particularly MI score) is higher.

**Outcome:**
- CPI Food at Home → Stage 1 (Price Channel)
- Unemployment Rate → Stage 1 (Price Channel)
- GDP Growth → Stage 1 (Price Channel)
- Consumer Sentiment → Stage 2 (Volume Channel — confirmed by defensive-category business logic)

---

## Phase 4 — Two-Stage Log-Log Ridge Regression

### Training Window
A rolling window of recent post-disruption months, intentionally excluding the COVID-19 period and supply-chain disruption years. Rationale: reflects the current macro-consumer dynamics most relevant for near-term scenario planning.

### STL Decomposition (Pre-Stage 2)

```python
from statsmodels.tsa.seasonal import STL
import numpy as np

# Fit on a broader window than training for stable seasonal factor estimation
stl = STL(np.log(volume_series), period=12, robust=True)
result = stl.fit()

# De-seasonalised volume (Stage 2 target)
vol_deseasoned = np.exp(result.trend + result.resid)

# Multiplicative seasonal factor (for re-seasonalising output forecasts)
seasonal_factor = np.exp(result.seasonal)

# Diagnostic metrics
seasonal_strength = max(0, 1 - np.var(result.resid) / np.var(result.resid + result.seasonal))
trend_strength    = max(0, 1 - np.var(result.resid) / np.var(result.resid + result.trend))
```

Both seasonal strength and trend strength confirm the decomposition is capturing genuine signal — not noise.

### Stage 1 — Macro Drivers → Average Equivalent Price

```python
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Log-transform inputs
X_cpi  = np.log(df['cpi_home'].values)
X_unem = np.log(df['unemp'].values)
# GDP Growth: sign-preserving log transform (handles percentage scale + sign)
X_gdp  = np.sign(df['gdp_growth']) * np.log1p(np.abs(df['gdp_growth']) / 100)

X1 = np.column_stack([X_cpi, X_unem, X_gdp])
y1 = np.log(df['avg_eq_price'].values)   # log target → log-log model

# Standardise (required for Ridge coefficient comparability)
scaler1 = StandardScaler()
X1_scaled = scaler1.fit_transform(X1)

# RidgeCV — forward-expanding folds only (no future leakage)
cv = TimeSeriesSplit(n_splits=5)
ridge1 = RidgeCV(alphas=np.logspace(-3, 3, 200), cv=cv)
ridge1.fit(X1_scaled, y1)

# Back-transform coefficients to log-log elasticity space
elasticities_s1 = ridge1.coef_ / scaler1.scale_
```

**VIF Check (Stage 1):** All features confirmed below threshold — no problematic multicollinearity.

**Stage 1 produces:**
- Elasticity of avg equivalent price with respect to each macro driver
- Interpretable as: "A 1% rise in CPI → X% change in cereal shelf price"

### Stage 2 — Price + Consumer Sentiment → De-Seasonalised Volume

```python
X2 = np.log(df[['avg_eq_price', 'sent']].values)
y2 = np.log(vol_deseasoned)   # STL de-seasonalised target

scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)

ridge2 = RidgeCV(alphas=np.logspace(-3, 3, 200), cv=cv)
ridge2.fit(X2_scaled, y2)

elasticities_s2 = ridge2.coef_ / scaler2.scale_
```

**Stage 2 produces:**
- Price elasticity: direct demand response to shelf price changes
- Sentiment elasticity: defensive-category volume effect

### Chain-Rule Composition

```python
# For each Stage 1 driver:
# ε(driver → volume) = ε(driver → price) × ε(price → volume)

price_to_vol_elasticity = elasticities_s2[0]   # from Stage 2

combined = {}
for driver, e_stage1 in zip(['cpi', 'unemp', 'gdp'], elasticities_s1):
    combined[driver] = e_stage1 * price_to_vol_elasticity

# Consumer Sentiment: direct from Stage 2 (no Stage 1 mediation)
combined['sentiment'] = elasticities_s2[1]
```

---

## Phase 5 — Sensitivity Table Construction

```python
# Baseline volume: mean de-seasonalised volume over training window
baseline_volume = vol_deseasoned.mean()

sensitivity = {}
for driver, elasticity in combined.items():
    unit_impact = baseline_volume * (elasticity / 100)
    sensitivity[driver] = {
        'combined_elasticity' : round(elasticity, 4),
        'unit_impact_per_1pct': round(unit_impact, 0)
    }
```

The resulting table directly answers the business question for each driver:
> *"If [driver] changes by 1%, our monthly cereal volume changes by [unit_impact] equivalent units."*

**Validation checks applied:**
1. **Directional consistency** — all elasticity signs verified against economic theory and EDA correlations
2. **Magnitude plausibility** — combined elasticities cross-checked against empirical EDA elasticity estimates
3. **Business interpretability** — unit impacts scaled and expressed appropriately for finance/IBP teams

---

## Model Evaluation

Both stages evaluated on training window via `TimeSeriesSplit` cross-validation in original (non-log) scale:

```python
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

# Metrics computed in original scale (exp of log predictions)
y_pred_original = np.exp(ridge.predict(X_scaled))
y_true_original = np.exp(y_log)

r2   = r2_score(y_true_original, y_pred_original)
mape = mean_absolute_percentage_error(y_true_original, y_pred_original) * 100
rmse = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
```

> Specific metric values are proprietary and not disclosed.
