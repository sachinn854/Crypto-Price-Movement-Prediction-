# Crypto Price Movement Prediction

Hourly price-movement modelling on 8 cryptocurrencies, with a leakage-free
feature pipeline, technical indicators written from scratch, ADF stationarity
testing, and 5-fold time-series cross-validation.

**Live app:** https://crypto-price-movement-prediction.streamlit.app/

> **Summary of results:** the direction classifier reaches **52.20% ± 0.53%**
> accuracy against a 50% baseline. The return regressor **does not work** — every
> model tested scored a negative R², i.e. worse than forecasting zero. Both are
> reported as measured.

---

## Results

From `python src/main.py --kbest 20` on 519,220 hourly bars.
Full log: `models/full_training_log.txt` · metrics: `models/training_report.json`

### Direction (up / down)

| Model | CV accuracy (mean ± std) |
|---|---|
| **RandomForest** *(selected)* | **0.5220 ± 0.0053** |
| LightGBM | 0.5209 ± 0.0049 |
| XGBoost | 0.5200 ± 0.0063 |
| DecisionTree | 0.5171 ± 0.0030 |
| Logit *(statsmodels, 1 fold)* | 0.5109 |
| Baseline | 0.5000 |

Precision 0.5228 · Recall 0.4558 · F1 0.4862.

The 2.2-point edge sits about 4 fold-standard-deviations above 50%, so it holds
across all five time regimes rather than coming from one lucky split. It has not
been tested against transaction costs.

### Return (next-hour log return)

| Model | CV R² (mean ± std) |
|---|---|
| **RandomForest** *(selected)* | **-0.0072 ± 0.0053** |
| LightGBM | -0.0073 ± 0.0043 |
| XGBoost | -0.0095 ± 0.0089 |
| DecisionTree | -0.0592 ± 0.0327 |
| OLS *(statsmodels, 1 fold)* | -0.0009 |
| Baseline *(predict zero)* | -0.0000 |

Every model scores below the zero-forecast baseline. Best MAE is 0.006530 vs the
baseline's 0.006521 — indistinguishable. SMAPE is 197% against a maximum of 200%.
The regressor is kept only because the app pairs it with the classifier for a
magnitude estimate; it should not be described as working.

---

## Validation

**5-fold expanding-window time-series cross-validation**, not a single split.

`TimeSeriesSplit(n_splits=5, gap=24)` is applied *within each symbol's own
chronology*, then fold k of every symbol is concatenated into one global fold. A
single split over the stacked frame would not be sound, because it interleaves 8
assets whose histories start on different dates.

- Train always precedes test, per symbol. No shuffling.
- 24-bar embargo, so the 20-bar rolling windows and 26-period MACD EMA of the
  first test row cannot overlap the last training row.
- Scaling and feature selection sit inside the sklearn `Pipeline`, so they are
  fit on training folds only.

Folds grow from 86,358 to 432,494 training rows, with 86,534 test rows each.

### Stationarity (ADF test)

`statsmodels.tsa.stattools.adfuller` on every symbol, before modelling:

| Series | Stationary at 5% |
|---|---|
| Raw close price | **0 / 8** (ADF -0.04 to -2.54, p 0.11–0.95) |
| Log returns | **8 / 8** (ADF -29 to -84, p ≈ 0) |

This is why the target is the next-period **log return**, never the price level.
Fitting on the non-stationary series is what produced this project's earlier
R² = 0.9998 — the model was carrying the price forward, not forecasting.

Interpretable baselines with full `summary()` saved to `models/`:
OLS (R² -0.0009, 18/21 coefficients significant at 5%) and
Logit (accuracy 0.5109, pseudo-R² 0.00168).

---

## Features

134 predictors built, narrowed to 20 by `SelectKBest` inside the pipeline.

| Family | Count |
|---|---|
| Lagged OHLCV (t-1 … t-5) | 30 |
| Bollinger Bands (20, 2σ) — bands, %B, bandwidth, distance-in-σ | 21 |
| Return moments — mean/std/skew/kurtosis, return_lag1…lag7 | 19 |
| MACD (12/26/9) — line, signal, histogram + normalised | 18 |
| Rolling price/volume stats — SMA 5/10/20, std, volume SMAs | 18 |
| Calendar + cyclical encodings | 13 |
| Price/volume ratios | 12 |
| RSI (14, Wilder) | 3 |

RSI, MACD and Bollinger Bands are implemented directly in pandas/numpy in
`src/feature_engineering_module.py` — no TA wrapper library — and shared by the
training and inference paths through one `build_indicator_frame()` definition so
the two cannot drift apart.

RSI uses Wilder's seeding (simple mean of the first 14 deltas, then α = 1/14
smoothing) and is verified against Wilder's published worked example. MACD and
Bollinger also emit close-normalised variants so features stay comparable across
assets whose price levels differ by four orders of magnitude.

They earn their place: of the 20 features retained, **7 are RSI/MACD/Bollinger in
the regressor and 12 in the classifier**.

### Leakage controls

Every predictor comes from bars strictly before the one being predicted.

- Indicators are computed on the unshifted close, then shifted by ≥1 bar.
- Rolling stats `.shift()` first, then `.rolling()` — the reverse order silently
  includes the current bar.
- All feature construction runs inside a per-symbol loop, so nothing crosses an
  asset boundary.
- A whitelist is the final gate: only `_lag1`…`_lag7` columns and a fixed list of
  known-in-advance calendar columns reach the model.
- The pipeline warns if CV R² > 0.80 or accuracy > 0.85, which on this data would
  mean leakage rather than success.

---

## Data

CryptoCompare `histohour` API, scraped with backward pagination.

- **Assets:** ADA, BTC, DOGE, DOT, ETH, LTC, SOL, XRP
- **Range:** 22 May 2017 → 9 Aug 2025, hourly
- **Cleaning:** 576,288 raw bars → 519,308 clean (90.1% retained). Dropped 56,980
  rows where all price *and* volume fields were zero — API padding from before
  each coin listed, which is non-existent data rather than missing data. OHLC
  violations repaired rather than dropped; returns capped at +100% / -99%.
- **Modelling rows:** 519,220 after target creation and indicator warm-up.

---

## Layout

```
app.py                              Streamlit UI
Datasracpe/datascrape.py            CryptoCompare scraper
notebooks/01_data_cleaning.ipynb    raw → processed
notebooks/02_eda_analysis.ipynb     exploratory analysis
src/main.py                         CLI entry point
src/model_training.py               ADF, features, CV, baselines, training
src/feature_engineering_module.py   RSI / MACD / Bollinger + lag builder
src/preprocessing_module.py         encode → scale → SelectKBest
src/unified_pipeline.py             dual-model facade + inference
models/                             pipelines, training report, statsmodels summaries
```

---

## Usage

```bash
pip install -r requirements.txt

# Retrain: ADF test → features → 5-fold CV → baselines → save
python src/main.py --data "Data/processed/final_cleaned_crypto_zero_removed.csv" --kbest 20

# App
streamlit run app.py
```

### Predicting

The models read a 20-bar Bollinger/SMA window at lag 3 and a 26-period MACD EMA,
so **at least 50 consecutive hourly bars of one symbol** are required. A single
candle cannot produce a feature vector.

```python
from src.unified_pipeline import CompleteCryptoPipeline

cp = CompleteCryptoPipeline(models_dir="models")
cp.load_pipelines()

cp.predict_from_history(history_df)   # ≥ 50 rows, one symbol, oldest first
# {'return_pct': ..., 'direction': 1, 'direction_label': 'Bullish', 'as_of': ..., 'bars_used': 200}
```

Fewer rows raises `InsufficientHistoryError` with an actionable message.

---

## Limitations

1. **The regressor does not work** — negative R² across every model and fold.
2. **The Streamlit UI collects a single candle**, which is insufficient for the
   feature set, so its prediction path errors until it is changed to supply
   history. This predates the current models.
3. **No backtest.** 52.2% accuracy is a statistical result, not a demonstrated
   trading edge; transaction costs and slippage are unmodelled.
4. **No probability calibration**, so outputs should not drive position sizing.
5. **Symbol is label-encoded**, giving a nominal variable an ordinal value. Trees
   mostly tolerate this; one-hot or native categorical support would be cleaner.
6. **No hyperparameter search** — a single configuration per model. The CV
   harness now exists to support one.

---

## Stack

Python 3.13 · pandas · numpy · scikit-learn · XGBoost · LightGBM · statsmodels ·
Streamlit · Plotly · joblib
