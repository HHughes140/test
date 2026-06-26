# Axioma Insurance Factor Analysis

Single-file analysis of **Axioma style-factor z-scores** (standardized exposures) within an
**insurance** equity universe. It builds factor histories and exposures, runs factor
long/short backtests, and produces a deep statistical + visual breakdown — with extra
modules tailored to how factor premia actually behave across insurance sub-industries and
rate regimes.

## What it produces

**Core** (`insurance_factor_dashboard.png` + console report)
- Per-factor dollar-neutral long/short return (long top-quintile z, short bottom; signal at
  *t* → return at *t+1*, no look-ahead): annualized return/vol, **Sharpe**, Newey-West *t*,
  rank **IC** + IC-IR, hit rate, max drawdown, autocorrelation half-life.
- 2-year & full-history cumulative L/S, factor-return correlation heatmap, **PCA** scree
  (effective number of independent bets), exposure dispersion, rolling Sharpe, drawdown.
- Insurance-basket regression on the factors (R², spanned variance, implied alpha).

**Advanced** (`insurance_factor_dashboard_advanced.png` + console report)
- **A. Sub-industry decomposition** — L/S Sharpe & IC within P&C / Life / Brokers /
  Reinsurance / Multiline / Title / Health. Surfaces when a factor only works in one segment.
- **B. Rate / macro regime conditioning** — each factor premium conditioned on rate
  direction, VIX level, and credit-spread level; plus per-name **rate beta** (the duration trade).
- **C. Statistical rigor** — **deflated Sharpe** (multiple-testing aware across the N factors),
  **Lo** autocorrelation-corrected Sharpe, **block-bootstrap** Sharpe CIs, **Fama-MacBeth**
  pure factor premia (controls for cross-factor collinearity), multi-horizon **IC decay**
  (1/5/21/63d), and quintile **monotonicity**.

## Usage

Runs out of the box on a realistic **synthetic** insurance panel (clearly labelled in every
output) so the full pipeline is verifiable before wiring in real data:

```bash
python3 axioma_insurance_factor_analysis.py
```

Point it at real Axioma exports:

```bash
python3 axioma_insurance_factor_analysis.py \
    --exposures expo.parquet --returns rets.parquet \
    --universe univ.csv --macro macro.csv
```

### Data schema (long / tidy, CSV or Parquet)

| file | columns |
|------|---------|
| `exposures` | `date, asset_id, factor, zscore` |
| `returns`   | `date, asset_id, ret` (decimal, 0.01 = 1%) |
| `universe`  | `asset_id, name, in_insurance, sub_industry` (last two optional) |
| `macro`     | `date, ten_year, credit_spread, vix` (optional; enables module B) |

Periodicity (daily/weekly/monthly) is inferred from date spacing for annualization.

## Requirements

```
python>=3.9
numpy, pandas, matplotlib, scipy
```

> The committed PNGs are generated from synthetic demo data and are labelled as such.
