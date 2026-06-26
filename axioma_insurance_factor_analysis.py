#!/usr/bin/env python3
"""
axioma_insurance_factor_analysis.py
===================================

Single-file analysis of Axioma style-factor z-scores (standardized exposures)
within an *insurance* equity universe.

What it does
------------
1. Loads Axioma factor exposures (z-scores) + asset returns + universe membership.
   - Real data: point --exposures / --returns / --universe at your exports.
   - No data?  It synthesizes a realistic insurance-universe panel so the whole
     pipeline runs end-to-end (clearly labelled SYNTHETIC in every output).

2. Restricts to the insurance universe and computes, per factor:
     - Total factor history  : cross-sectional exposure mean & dispersion over time
     - Exposure to our stocks : latest z-score per name, plus exposure stability
     - Factor long/short return: each period rank names by z-score, long top quintile,
       short bottom quintile (dollar-neutral, equal weight), realize next-period return.

3. Mathematical deep dive:
     - Information Coefficient (rank corr of z-score vs forward return) + IC IR + t-stat
     - Annualized return / vol / Sharpe of each factor L/S leg, with Newey-West t-stats
     - Max drawdown & hit rate of each factor
     - Factor-return correlation matrix + PCA (eigenvalues, variance explained) =>
       how many *independent* bets the factor set really represents (crowding)
     - Exposure crowding: cross-sectional correlation of the z-scores themselves
     - Half-life of factor-return autocorrelation (momentum vs mean-reversion in the premium)
     - Multi-factor OLS of insurance basket return on factor L/S returns (R^2, betas)

4. Charts (one PNG dashboard):
     - 2-year cumulative L/S per factor
     - Full-history cumulative L/S per factor
     - Bar chart: annualized L/S return (+ Sharpe) per factor
     - Factor-return correlation heatmap
     - PCA scree (variance explained)
     - Exposure dispersion (breadth of the bet) over time

Data schema (long / tidy format, CSV or Parquet)
------------------------------------------------
exposures : columns = [date, asset_id, factor, zscore]
returns   : columns = [date, asset_id, ret]      # simple period return, decimal (0.01 = 1%)
universe  : columns = [asset_id, name, in_insurance(bool)]   # optional; default = all assets

Dates are parsed automatically. Returns should be aligned to the same periodicity as
exposures (daily, weekly, or monthly). The script infers periodicity for annualization.

Usage
-----
    python3 axioma_insurance_factor_analysis.py                 # synthetic demo
    python3 axioma_insurance_factor_analysis.py \
        --exposures expo.parquet --returns rets.parquet --universe univ.csv \
        --out insurance_factors.png

Author: generated for henry's insurance factor workflow.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")  # headless / file output
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    from scipy import stats as sps
except Exception:  # scipy optional; degrade gracefully
    sps = None


# ----------------------------------------------------------------------------- #
# Synthetic data (only used when no real files are supplied)
# ----------------------------------------------------------------------------- #
# A representative US insurance universe (GICS 4030 Insurance industry group).
INSURANCE_TICKERS = [
    ("BRK.B", "Berkshire Hathaway"), ("PGR", "Progressive"), ("CB", "Chubb"),
    ("MMC", "Marsh & McLennan"), ("AON", "Aon"), ("ELV", "Elevance Health"),
    ("CI", "Cigna"), ("AIG", "American Intl Group"), ("MET", "MetLife"),
    ("PRU", "Prudential Financial"), ("TRV", "Travelers"), ("ALL", "Allstate"),
    ("AFL", "Aflac"), ("HIG", "Hartford"), ("ACGL", "Arch Capital"),
    ("WTW", "Willis Towers Watson"), ("AJG", "Arthur J Gallagher"),
    ("MKL", "Markel"), ("GL", "Globe Life"), ("CINF", "Cincinnati Financial"),
    ("PFG", "Principal Financial"), ("RGA", "Reinsurance Group"),
    ("L", "Loews"), ("FNF", "Fidelity National Fin"), ("EG", "Everest Group"),
    ("WRB", "W.R. Berkley"), ("AIZ", "Assurant"), ("UNM", "Unum"),
    ("BRO", "Brown & Brown"), ("RYAN", "Ryan Specialty"),
    ("KNSL", "Kinsale Capital"), ("ERIE", "Erie Indemnity"),
    ("FAF", "First American Fin"), ("CNO", "CNO Financial"),
    ("AFG", "American Financial"), ("ORI", "Old Republic"),
    ("RLI", "RLI Corp"), ("SIGI", "Selective Insurance"),
    ("THG", "Hanover Insurance"), ("MCY", "Mercury General"),
]

# Insurance sub-industry classification (GICS 4030 sub-groups, hand-mapped).
# Factor premia behave very differently across these buckets, so we decompose by it.
SUBINDUSTRY = {
    "BRK.B": "Multiline", "PGR": "P&C", "CB": "P&C", "MMC": "Brokers",
    "AON": "Brokers", "ELV": "Health", "CI": "Health", "AIG": "Multiline",
    "MET": "Life", "PRU": "Life", "TRV": "P&C", "ALL": "P&C", "AFL": "Life",
    "HIG": "P&C", "ACGL": "P&C", "WTW": "Brokers", "AJG": "Brokers",
    "MKL": "P&C", "GL": "Life", "CINF": "P&C", "PFG": "Life",
    "RGA": "Reinsurance", "L": "Multiline", "FNF": "Title", "EG": "Reinsurance",
    "WRB": "P&C", "AIZ": "Multiline", "UNM": "Life", "BRO": "Brokers",
    "RYAN": "Brokers", "KNSL": "P&C", "ERIE": "Brokers", "FAF": "Title",
    "CNO": "Life", "AFG": "P&C", "ORI": "Title", "RLI": "P&C",
    "SIGI": "P&C", "THG": "P&C", "MCY": "P&C",
}

# Per-sub-industry sensitivity to a +1 move in the 10Y yield (contemporaneous return).
# Life insurers benefit from higher rates (spread/discounting); brokers ~rate-neutral.
SUBIND_RATE_BETA = {
    "Life": 0.9, "Reinsurance": 0.5, "P&C": 0.35, "Multiline": 0.4,
    "Title": -0.6, "Brokers": -0.05, "Health": 0.1,
}

# Axioma-US4 style factors most relevant to financials/insurance.
AXIOMA_STYLE_FACTORS = [
    "Value", "Earnings Yield", "Dividend Yield", "Profitability", "Growth",
    "Leverage", "Size", "Market Sensitivity", "Volatility", "Liquidity",
    "Medium-Term Momentum", "Exchange Rate Sensitivity",
]

# True forward-return premia baked into synthetic data (per period, annualized-ish
# intuition). Some factors "work" inside insurance, some are noise -> realistic output.
SYNTH_TRUE_PREMIA = {
    "Value": 0.030, "Earnings Yield": 0.045, "Dividend Yield": 0.020,
    "Profitability": 0.055, "Growth": -0.010, "Leverage": -0.025,
    "Size": -0.015, "Market Sensitivity": 0.000, "Volatility": -0.040,
    "Liquidity": 0.005, "Medium-Term Momentum": 0.035,
    "Exchange Rate Sensitivity": 0.000,
}


def synthesize(seed: int = 7, n_periods: int = 252 * 6):
    """Build a realistic daily insurance panel: exposures, returns, universe, macro."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_periods)
    assets = [t for t, _ in INSURANCE_TICKERS]
    names = dict(INSURANCE_TICKERS)
    factors = AXIOMA_STYLE_FACTORS
    n_a, n_f = len(assets), len(factors)
    rate_beta = np.array([SUBIND_RATE_BETA[SUBINDUSTRY[a]] for a in assets])

    # Persistent latent z-scores: AR(1) per asset/factor, cross-sectionally standardized.
    rho = 0.985  # exposures are slow-moving
    z = rng.standard_normal((n_a, n_f))
    expo_rows = []
    ret_rows = []

    # A single market factor + idiosyncratic noise drives returns; style premia add tilt.
    daily_premia = {f: SYNTH_TRUE_PREMIA[f] / 252.0 for f in factors}
    mkt_vol = 0.011
    idio_vol = 0.013

    z_hist = np.empty((n_periods, n_a, n_f))
    for t in range(n_periods):
        z = rho * z + np.sqrt(1 - rho**2) * rng.standard_normal((n_a, n_f))
        # cross-sectionally standardize each factor (this is what "z-score" means)
        z = (z - z.mean(0)) / (z.std(0) + 1e-9)
        z_hist[t] = z

    # Macro paths: 10Y yield (random walk), credit spread (mean-reverting), VIX (lognormal).
    d_rate = rng.standard_normal(n_periods) * 0.05          # daily Δ10Y in %-points
    ten_year = 2.5 + np.cumsum(d_rate)
    cs = np.empty(n_periods); cs[0] = 4.0
    vix = np.empty(n_periods); vix[0] = 18.0
    for t in range(1, n_periods):
        cs[t] = max(1.0, 0.97 * cs[t - 1] + 0.03 * 4.0 + rng.standard_normal() * 0.12)
        vix[t] = max(9.0, 0.94 * vix[t - 1] + 0.06 * 18.0 + rng.standard_normal() * 1.3)
    # market return is fatter-tailed when VIX is high
    mkt = rng.standard_normal(n_periods) * mkt_vol * (vix / 18.0)

    for t in range(n_periods):
        zt = z_hist[t]
        idio = rng.standard_normal(n_a) * idio_vol
        # use *previous* period's z to predict *this* period's return (no look-ahead)
        zt_prev = z_hist[t - 1] if t > 0 else zt
        signal_prev = np.zeros(n_a)
        for j, f in enumerate(factors):
            signal_prev += daily_premia[f] * zt_prev[:, j]
        # contemporaneous rate shock scaled by each name's rate beta (the duration trade)
        rate_effect = rate_beta * d_rate[t] * 0.01
        r = signal_prev + 0.9 * mkt[t] + rate_effect + idio
        for i, a in enumerate(assets):
            ret_rows.append((dates[t], a, r[i]))
            for j, f in enumerate(factors):
                expo_rows.append((dates[t], a, f, zt[i, j]))

    exposures = pd.DataFrame(expo_rows, columns=["date", "asset_id", "factor", "zscore"])
    returns = pd.DataFrame(ret_rows, columns=["date", "asset_id", "ret"])
    universe = pd.DataFrame(
        [(a, names[a], True, SUBINDUSTRY[a]) for a in assets],
        columns=["asset_id", "name", "in_insurance", "sub_industry"],
    )
    macro = pd.DataFrame({"date": dates, "ten_year": ten_year,
                          "credit_spread": cs, "vix": vix})
    return exposures, returns, universe, macro


# ----------------------------------------------------------------------------- #
# Loading
# ----------------------------------------------------------------------------- #
def _read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_data(args):
    """Return (exposures, returns, universe, macro, is_synthetic)."""
    if not args.exposures or not args.returns:
        print(">> No --exposures/--returns supplied: running on SYNTHETIC insurance data.\n")
        e, r, u, m = synthesize()
        return e, r, u, m, True

    exposures = _read_any(args.exposures)
    returns = _read_any(args.returns)
    exposures["date"] = pd.to_datetime(exposures["date"])
    returns["date"] = pd.to_datetime(returns["date"])

    if args.universe:
        universe = _read_any(args.universe)
    else:
        ids = sorted(set(exposures["asset_id"]) | set(returns["asset_id"]))
        universe = pd.DataFrame({"asset_id": ids, "name": ids, "in_insurance": True})
    if "in_insurance" not in universe.columns:
        universe["in_insurance"] = True
    if "name" not in universe.columns:
        universe["name"] = universe["asset_id"]
    if "sub_industry" not in universe.columns:
        universe["sub_industry"] = universe["asset_id"].map(SUBINDUSTRY).fillna("Other")

    macro = None
    if args.macro:
        macro = _read_any(args.macro)
        macro["date"] = pd.to_datetime(macro["date"])
    return exposures, returns, universe, macro, False


# ----------------------------------------------------------------------------- #
# Core analytics
# ----------------------------------------------------------------------------- #
@dataclass
class FactorResult:
    factor: str
    ls_returns: pd.Series        # per-period long/short return
    cum: pd.Series               # cumulative (compounded) L/S return
    ann_ret: float
    ann_vol: float
    sharpe: float
    t_stat: float                # Newey-West t on mean L/S return
    ic_mean: float               # mean cross-sectional rank IC
    ic_ir: float                 # IC information ratio
    ic_t: float
    hit_rate: float
    max_dd: float
    half_life: float             # of factor-return autocorrelation
    latest_dispersion: float


def _infer_ppy(idx: pd.DatetimeIndex) -> int:
    """Periods per year from median spacing."""
    if len(idx) < 3:
        return 252
    med = np.median(np.diff(idx.values).astype("timedelta64[D]").astype(float))
    if med <= 2:
        return 252
    if med <= 9:
        return 52
    return 12


def _newey_west_t(x: np.ndarray, lags: int = 5) -> float:
    """t-stat of the mean of x with Newey-West HAC standard error."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return np.nan
    mu = x.mean()
    e = x - mu
    gamma0 = (e @ e) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1 - l / (lags + 1)
        cov = (e[l:] @ e[:-l]) / n
        var += 2 * w * cov
    se = np.sqrt(var / n)
    return mu / se if se > 0 else np.nan


def _half_life(series: pd.Series) -> float:
    """Half-life implied by lag-1 autocorrelation: ln(0.5)/ln(rho). NaN if not mean-reverting."""
    s = series.dropna()
    if len(s) < 20:
        return np.nan
    rho = s.autocorr(lag=1)
    if rho is None or np.isnan(rho) or rho <= 0 or rho >= 1:
        return np.nan
    return np.log(0.5) / np.log(rho)


def _max_drawdown(cum_growth: pd.Series) -> float:
    peak = cum_growth.cummax()
    dd = cum_growth / peak - 1.0
    return dd.min()


def quintile_long_short(
    expo: pd.DataFrame, rets: pd.DataFrame, factor: str, q: float = 0.2
) -> pd.Series:
    """Per-period dollar-neutral L/S return: long top-q by z-score, short bottom-q.

    Uses period-t exposure to weight period-(t+1) return (no look-ahead).
    """
    z = (
        expo[expo["factor"] == factor]
        .pivot_table(index="date", columns="asset_id", values="zscore")
        .sort_index()
    )
    r = rets.pivot_table(index="date", columns="asset_id", values="ret").sort_index()
    common = z.columns.intersection(r.columns)
    z, r = z[common], r[common]
    # align: signal at t -> return at t+1
    r_fwd = r.shift(-1)
    out = {}
    for dt, row in z.iterrows():
        vals = row.dropna()
        if len(vals) < 5:
            continue
        hi = vals[vals >= vals.quantile(1 - q)].index
        lo = vals[vals <= vals.quantile(q)].index
        fr = r_fwd.loc[dt]
        long_r = fr[hi].mean()
        short_r = fr[lo].mean()
        if np.isnan(long_r) or np.isnan(short_r):
            continue
        out[dt] = long_r - short_r
    return pd.Series(out, name=factor).sort_index()


def cross_sectional_ic(expo: pd.DataFrame, rets: pd.DataFrame, factor: str) -> pd.Series:
    """Per-period Spearman rank IC between z-score(t) and forward return(t+1)."""
    z = (
        expo[expo["factor"] == factor]
        .pivot_table(index="date", columns="asset_id", values="zscore")
        .sort_index()
    )
    r = rets.pivot_table(index="date", columns="asset_id", values="ret").sort_index()
    common = z.columns.intersection(r.columns)
    z, r = z[common], r[common]
    r_fwd = r.shift(-1)
    ics = {}
    for dt, row in z.iterrows():
        a = row
        b = r_fwd.loc[dt]
        m = (~a.isna()) & (~b.isna())
        if m.sum() < 5:
            continue
        ra = a[m].rank()
        rb = b[m].rank()
        if ra.std() == 0 or rb.std() == 0:
            continue
        ics[dt] = np.corrcoef(ra, rb)[0, 1]
    return pd.Series(ics, name=factor).sort_index()


def analyze_factor(expo, rets, factor, ppy) -> FactorResult:
    ls = quintile_long_short(expo, rets, factor)
    ic = cross_sectional_ic(expo, rets, factor)
    cum = (1 + ls).cumprod()

    ann_ret = ls.mean() * ppy
    ann_vol = ls.std() * np.sqrt(ppy)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    t_stat = _newey_west_t(ls.values)
    ic_mean = ic.mean()
    ic_ir = ic.mean() / ic.std() if ic.std() > 0 else np.nan
    ic_t = ic_ir * np.sqrt(len(ic)) if len(ic) else np.nan
    hit = (ls > 0).mean()
    mdd = _max_drawdown(cum)
    hl = _half_life(ls)

    # exposure dispersion (breadth of the active bet) at latest date
    z_latest = (
        expo[expo["factor"] == factor]
        .sort_values("date")
        .groupby("asset_id")
        .tail(1)["zscore"]
    )
    disp = float(z_latest.std())

    return FactorResult(
        factor, ls, cum, ann_ret, ann_vol, sharpe, t_stat,
        ic_mean, ic_ir, ic_t, hit, mdd, hl, disp,
    )


# ----------------------------------------------------------------------------- #
# Mathematical deep dive
# ----------------------------------------------------------------------------- #
def pca_on_factor_returns(ls_matrix: pd.DataFrame):
    """Eigendecomposition of the factor-return correlation matrix.

    Tells us how many *independent* bets the factor set represents inside insurance.
    """
    X = ls_matrix.dropna()
    C = np.corrcoef(X.values.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    var_explained = eigvals / eigvals.sum()
    # Effective number of bets = exp(entropy of normalized eigenvalues)
    p = var_explained[var_explained > 0]
    eff_bets = float(np.exp(-(p * np.log(p)).sum()))
    return C, eigvals, var_explained, eff_bets, eigvecs, ls_matrix.columns.tolist()


def basket_factor_regression(expo, rets, ls_matrix, universe):
    """OLS of equal-weight insurance basket return on factor L/S returns -> R^2, betas."""
    ins_ids = universe.loc[universe["in_insurance"], "asset_id"]
    r = (
        rets[rets["asset_id"].isin(ins_ids)]
        .pivot_table(index="date", columns="asset_id", values="ret")
        .sort_index()
    )
    basket = r.mean(axis=1)
    df = ls_matrix.join(basket.rename("basket"), how="inner").dropna()
    if len(df) < 30:
        return None
    y = df["basket"].values
    X = df.drop(columns="basket").values
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return {
        "r2": r2,
        "alpha_per_period": beta[0],
        "betas": dict(zip(ls_matrix.columns, beta[1:])),
        "n": len(df),
    }


def exposure_crowding(expo, factors):
    """Average absolute cross-sectional correlation between the z-scores themselves.

    High => factors are loading on the same names (the active bets overlap)."""
    latest = (
        expo.sort_values("date")
        .groupby(["asset_id", "factor"])
        .tail(1)
        .pivot_table(index="asset_id", columns="factor", values="zscore")
    )
    latest = latest[[f for f in factors if f in latest.columns]]
    C = latest.corr()
    off = C.where(~np.eye(len(C), dtype=bool))
    return C, float(off.abs().stack().mean())


# ----------------------------------------------------------------------------- #
# (A) Sub-industry decomposition
# ----------------------------------------------------------------------------- #
def _median_split_ls(z: pd.DataFrame, r_fwd: pd.DataFrame) -> pd.Series:
    """Above-median minus below-median forward return — robust for small buckets."""
    out = {}
    for dt, row in z.iterrows():
        vals = row.dropna()
        if len(vals) < 4:
            continue
        med = vals.median()
        hi, lo = vals[vals > med].index, vals[vals <= med].index
        fr = r_fwd.loc[dt]
        a, b = fr[hi].mean(), fr[lo].mean()
        if not (np.isnan(a) or np.isnan(b)):
            out[dt] = a - b
    return pd.Series(out).sort_index()


def subindustry_decomposition(expo, rets, universe, factors, ppy):
    """Per (factor, sub-industry): Sharpe of an above/below-median L/S and mean IC.

    Returns two DataFrames (sharpe, ic) indexed by factor, columns = sub-industries.
    """
    sub_map = universe.set_index("asset_id")["sub_industry"].to_dict()
    subs = sorted(set(sub_map.values()))
    r = rets.pivot_table(index="date", columns="asset_id", values="ret").sort_index()
    r_fwd = r.shift(-1)
    sharpe = pd.DataFrame(index=factors, columns=subs, dtype=float)
    ic = pd.DataFrame(index=factors, columns=subs, dtype=float)
    for f in factors:
        zf = (expo[expo["factor"] == f]
              .pivot_table(index="date", columns="asset_id", values="zscore").sort_index())
        for s in subs:
            ids = [a for a in zf.columns if sub_map.get(a) == s]
            if len(ids) < 4:
                continue
            z = zf[ids]
            rf = r_fwd[[c for c in ids if c in r_fwd.columns]]
            ls = _median_split_ls(z, rf)
            if len(ls) > 20 and ls.std() > 0:
                sharpe.loc[f, s] = ls.mean() / ls.std() * np.sqrt(ppy)
            # mean cross-sectional rank IC within the bucket
            ics = []
            for dt, row in z.iterrows():
                b = r_fwd.loc[dt, [c for c in ids if c in r_fwd.columns]]
                m = (~row.isna()) & (~b.isna())
                if m.sum() >= 4 and row[m].std() and b[m].std():
                    ics.append(np.corrcoef(row[m].rank(), b[m].rank())[0, 1])
            if ics:
                ic.loc[f, s] = np.mean(ics)
    return sharpe.astype(float), ic.astype(float)


# ----------------------------------------------------------------------------- #
# (B) Rate / macro regime conditioning
# ----------------------------------------------------------------------------- #
def regime_conditioning(ls_matrix, macro, ppy, lookback=21):
    """Mean annualized L/S return of each factor split by macro regime.

    Regimes: rising vs falling 10Y (Δ over `lookback`), wide vs tight credit spread,
    high vs low VIX (median splits). Returns dict[label] -> Series(factor -> ann ret).
    """
    if macro is None:
        return None
    m = macro.set_index("date").sort_index()
    idx = ls_matrix.index
    m = m.reindex(idx).ffill()
    out = {}
    if "ten_year" in m:
        d = m["ten_year"].diff(lookback)
        out["Rising rates"] = ls_matrix[d > 0].mean() * ppy
        out["Falling rates"] = ls_matrix[d <= 0].mean() * ppy
    if "credit_spread" in m:
        hi = m["credit_spread"] > m["credit_spread"].median()
        out["Wide credit"] = ls_matrix[hi].mean() * ppy
        out["Tight credit"] = ls_matrix[~hi].mean() * ppy
    if "vix" in m:
        hi = m["vix"] > m["vix"].median()
        out["High VIX"] = ls_matrix[hi].mean() * ppy
        out["Low VIX"] = ls_matrix[~hi].mean() * ppy
    return out


def per_name_rate_beta(rets, macro, universe, lookback=1):
    """OLS beta of each name's return on Δ10Y (the duration trade). Returns DataFrame."""
    if macro is None or "ten_year" not in macro.columns:
        return None
    m = macro.set_index("date").sort_index()
    d_rate = m["ten_year"].diff(lookback)
    r = rets.pivot_table(index="date", columns="asset_id", values="ret").sort_index()
    d_rate = d_rate.reindex(r.index)
    sub_map = universe.set_index("asset_id")["sub_industry"].to_dict()
    rows = []
    x = d_rate.values
    for a in r.columns:
        y = r[a].values
        m_ok = ~(np.isnan(x) | np.isnan(y))
        if m_ok.sum() < 30 or np.nanstd(x[m_ok]) == 0:
            continue
        beta = np.polyfit(x[m_ok], y[m_ok], 1)[0]
        rows.append((a, sub_map.get(a, "Other"), beta, np.nanmean(y) * 252))
    return pd.DataFrame(rows, columns=["asset_id", "sub_industry", "rate_beta", "ann_ret"])


# ----------------------------------------------------------------------------- #
# (C) Statistical hardening: IC decay, monotonicity, deflated/bootstrap Sharpe
# ----------------------------------------------------------------------------- #
def ic_decay(expo, rets, factor, horizons=(1, 5, 21, 63)):
    """Mean Spearman IC of z-score(t) vs cumulative forward return over h periods."""
    z = (expo[expo["factor"] == factor]
         .pivot_table(index="date", columns="asset_id", values="zscore").sort_index())
    r = rets.pivot_table(index="date", columns="asset_id", values="ret").sort_index()
    common = z.columns.intersection(r.columns)
    z, r = z[common], r[common]
    logr = np.log1p(r)
    out = {}
    for h in horizons:
        # cumulative return over the NEXT h periods (t+1 .. t+h), no look-ahead
        fwd = np.expm1(logr[::-1].rolling(h).sum()[::-1].shift(-1))
        ics = []
        for dt, row in z.iterrows():
            b = fwd.loc[dt] if dt in fwd.index else None
            if b is None:
                continue
            mok = (~row.isna()) & (~b.isna())
            if mok.sum() >= 5 and row[mok].std() and b[mok].std():
                ics.append(np.corrcoef(row[mok].rank(), b[mok].rank())[0, 1])
        out[h] = np.mean(ics) if ics else np.nan
    return out


def quintile_monotonicity(expo, rets, factor, nq=5):
    """Mean forward return of each z-score quintile (Q1 low ... Q5 high)."""
    z = (expo[expo["factor"] == factor]
         .pivot_table(index="date", columns="asset_id", values="zscore").sort_index())
    r = rets.pivot_table(index="date", columns="asset_id", values="ret").sort_index()
    common = z.columns.intersection(r.columns)
    z, r = z[common], r[common]
    r_fwd = r.shift(-1)
    buckets = {q: [] for q in range(nq)}
    for dt, row in z.iterrows():
        vals = row.dropna()
        if len(vals) < nq * 2:
            continue
        try:
            labels = pd.qcut(vals, nq, labels=False, duplicates="drop")
        except ValueError:
            continue
        fr = r_fwd.loc[dt]
        for q in range(nq):
            ids = labels[labels == q].index
            v = fr[ids].mean()
            if not np.isnan(v):
                buckets[q].append(v)
    means = np.array([np.mean(buckets[q]) if buckets[q] else np.nan for q in range(nq)])
    # Spearman rank corr between quintile index and mean return = monotonicity score
    valid = ~np.isnan(means)
    mono = (np.corrcoef(np.arange(nq)[valid], means[valid])[0, 1]
            if valid.sum() > 2 else np.nan)
    return means, mono


def deflated_sharpe(observed_sr, sr_list, n_obs, skew=0.0, kurt=3.0):
    """Bailey & Lopez de Prado Deflated Sharpe Ratio.

    Adjusts an annualized... no — works on PER-PERIOD Sharpe. We pass per-period SR.
    Accounts for selecting the best of N trials. Returns P(true SR>0 | best of N)."""
    if sps is None or len(sr_list) < 2:
        return np.nan
    sr = np.array([s for s in sr_list if not np.isnan(s)])
    if len(sr) < 2:
        return np.nan
    var_sr = np.var(sr, ddof=1)
    n_trials = len(sr)
    emc = 0.5772156649
    # expected max Sharpe under the null across N independent trials
    z1 = sps.norm.ppf(1 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr0 = np.sqrt(var_sr) * ((1 - emc) * z1 + emc * z2)
    denom = np.sqrt(1 - skew * observed_sr + (kurt - 1) / 4.0 * observed_sr**2)
    dsr = sps.norm.cdf(((observed_sr - sr0) * np.sqrt(n_obs - 1)) / denom)
    return dsr


def block_bootstrap_sharpe_ci(x, ppy, block=21, n_boot=2000, seed=0):
    """Stationary block bootstrap CI for the annualized Sharpe of return series x."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    srs = np.empty(n_boot)
    n_blocks = int(np.ceil(n / block))
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]
        s = x[idx]
        srs[b] = s.mean() / s.std() * np.sqrt(ppy) if s.std() > 0 else 0.0
    return (np.percentile(srs, 2.5), np.percentile(srs, 97.5))


def fama_macbeth(expo, rets, factors):
    """Cross-sectional OLS each period -> mean 'pure' factor return controlling for
    the other factors. Returns DataFrame[factor -> (premium_ann, t_stat)]."""
    r = rets.pivot_table(index="date", columns="asset_id", values="ret").sort_index()
    r_fwd = r.shift(-1)
    zpanels = {f: expo[expo["factor"] == f]
               .pivot_table(index="date", columns="asset_id", values="zscore").sort_index()
               for f in factors}
    coefs = {f: [] for f in factors}
    for dt in r_fwd.index:
        y = r_fwd.loc[dt].dropna()
        if len(y) < len(factors) + 3:
            continue
        cols, X = [], []
        for f in factors:
            zp = zpanels[f]
            if dt in zp.index:
                cols.append(f)
                X.append(zp.loc[dt].reindex(y.index))
        if not cols:
            continue
        Xm = pd.concat(X, axis=1, keys=cols)
        df = Xm.join(y.rename("y")).dropna()
        if len(df) < len(cols) + 3:
            continue
        A = np.column_stack([np.ones(len(df)), df[cols].values])
        beta, *_ = np.linalg.lstsq(A, df["y"].values, rcond=None)
        for i, f in enumerate(cols):
            coefs[f].append(beta[i + 1])
    rows = {}
    for f in factors:
        c = np.array(coefs[f])
        if len(c) > 10:
            rows[f] = (c.mean() * 252, _newey_west_t(c))
    return pd.DataFrame(rows, index=["premium_ann", "t_stat"]).T


# ----------------------------------------------------------------------------- #
# Reporting
# ----------------------------------------------------------------------------- #
def print_report(results, ls_matrix, pca, reg, crowd_avg, ppy, is_synth, expo, universe):
    line = "=" * 100
    tag = "  [SYNTHETIC DEMO DATA]" if is_synth else ""
    print(line)
    print(f"AXIOMA STYLE-FACTOR ANALYSIS  —  INSURANCE UNIVERSE{tag}")
    print(line)

    n_names = expo["asset_id"].nunique()
    d0, d1 = expo["date"].min().date(), expo["date"].max().date()
    print(f"Universe size      : {n_names} insurance names")
    print(f"History            : {d0}  ->  {d1}   ({ppy} periods/yr assumed)")
    print(f"Factors analyzed   : {len(results)}")
    print()

    # ---- per-factor performance table ----
    print("FACTOR LONG/SHORT PERFORMANCE  (long top quintile z-score, short bottom, dollar-neutral)")
    print("-" * 100)
    hdr = (f"{'Factor':<26}{'AnnRet':>8}{'AnnVol':>8}{'Sharpe':>8}"
           f"{'NW-t':>7}{'IC':>7}{'IC-IR':>7}{'Hit%':>7}{'MaxDD':>8}{'HalfLife':>9}")
    print(hdr)
    print("-" * 100)
    rows = sorted(results, key=lambda x: (x.sharpe if not np.isnan(x.sharpe) else -9))[::-1]
    for fr in rows:
        hl = f"{fr.half_life:6.1f}" if not np.isnan(fr.half_life) else "   n/a"
        print(f"{fr.factor:<26}{fr.ann_ret*100:>7.1f}%{fr.ann_vol*100:>7.1f}%"
              f"{fr.sharpe:>8.2f}{fr.t_stat:>7.2f}{fr.ic_mean:>7.3f}{fr.ic_ir:>7.2f}"
              f"{fr.hit_rate*100:>6.1f}%{fr.max_dd*100:>7.1f}%{hl:>9}")
    print("-" * 100)
    print("Reading it: |NW-t|>2 => mean L/S return reliably non-zero. IC>0 => z-score ranks")
    print("forward returns the intended way. IC-IR>0.5 is strong, >0.3 decent. HalfLife small")
    print("=> the premium mean-reverts fast (timing/contrarian); large/n.a. => persistent trend.")
    print()

    # ---- math deep dive ----
    C, eigvals, var_exp, eff_bets, eigvecs, cols = pca
    print("MATHEMATICAL DEEP DIVE")
    print("-" * 100)
    print("1) PCA of the factor-return correlation matrix (independent bets inside insurance):")
    cum_v = np.cumsum(var_exp)
    for i in range(min(6, len(eigvals))):
        bar = "#" * int(round(var_exp[i] * 50))
        print(f"   PC{i+1}:  eigenvalue={eigvals[i]:5.2f}   var={var_exp[i]*100:5.1f}%   "
              f"cum={cum_v[i]*100:5.1f}%  {bar}")
    print(f"   -> Effective # of independent factor bets (entropy): {eff_bets:.2f} "
          f"out of {len(cols)} nominal factors.")
    # interpret PC1
    load1 = pd.Series(eigvecs[:, 0], index=cols).sort_values(key=abs, ascending=False)
    top = ", ".join(f"{k}({v:+.2f})" for k, v in load1.head(4).items())
    print(f"   -> PC1 (dominant common mode) loads most on: {top}")
    print()

    print(f"2) Exposure crowding: avg |cross-sectional corr| between factor z-scores = {crowd_avg:.2f}")
    print("   (>0.4 means several factors are effectively betting on the same insurance names.)")
    print()

    if reg:
        print(f"3) Insurance basket return regressed on factor L/S returns (n={reg['n']}):")
        print(f"   R^2 = {reg['r2']*100:.1f}%   "
              f"=> share of equal-weight insurance-basket variance spanned by these factors.")
        print(f"   Implied annualized alpha (intercept) = {reg['alpha_per_period']*ppy*100:+.2f}% "
              f"(return NOT explained by the factors).")
        bser = pd.Series(reg["betas"]).sort_values(key=abs, ascending=False)
        topb = ", ".join(f"{k}({v:+.2f})" for k, v in bser.head(4).items())
        print(f"   Largest basket betas: {topb}")
        print()

    # ---- actionable insight ----
    best = rows[0]
    worst = rows[-1]
    sig = [fr for fr in rows if abs(fr.t_stat) > 2]
    print("INSIGHTS FOR STOCK PERFORMANCE")
    print("-" * 100)
    print(f"* Strongest paying factor: {best.factor} "
          f"(Sharpe {best.sharpe:.2f}, IC {best.ic_mean:+.3f}, NW-t {best.t_stat:+.2f}).")
    print(f"* Weakest / fade candidate: {worst.factor} "
          f"(Sharpe {worst.sharpe:.2f}, IC {worst.ic_mean:+.3f}).")
    print(f"* Statistically reliable factors (|NW-t|>2): "
          f"{', '.join(fr.factor for fr in sig) if sig else 'none at this sample size'}.")
    if eff_bets < len(cols) * 0.5:
        print(f"* Diversification warning: only ~{eff_bets:.1f} independent bets — the factor menu "
              f"is redundant inside insurance; don't size them as if uncorrelated.")
    # name-level tilt: which stocks are extreme on the best factor right now
    z_best = (
        expo[expo["factor"] == best.factor]
        .sort_values("date").groupby("asset_id").tail(1)
        .merge(universe[["asset_id", "name"]], on="asset_id", how="left")
        .sort_values("zscore", ascending=False)
    )
    longs = z_best.head(5)
    shorts = z_best.tail(5)
    nm = lambda df: ", ".join(f"{r.asset_id}({r.zscore:+.2f})" for r in df.itertuples())
    print(f"* On {best.factor} today — most exposed (long bias): {nm(longs)}")
    print(f"*                          least exposed (short bias): {nm(shorts)}")
    print(line)


# ----------------------------------------------------------------------------- #
# Charts
# ----------------------------------------------------------------------------- #
def make_dashboard(results, ls_matrix, pca, expo, out_path, is_synth, ppy, focus=None):
    C, eigvals, var_exp, eff_bets, eigvecs, cols = pca
    fig = plt.figure(figsize=(20, 13))
    gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.28)
    title = "Axioma Style Factors — Insurance Universe"
    if is_synth:
        title += "   [SYNTHETIC DEMO DATA]"
    fig.suptitle(title, fontsize=17, fontweight="bold")

    cmap = plt.cm.tab20(np.linspace(0, 1, len(results)))
    color = {fr.factor: cmap[i] for i, fr in enumerate(results)}
    # focus = factor(s) to feature; default to the single top-Sharpe factor
    focus_list = focus or [max(results, key=lambda x: (x.sharpe if not np.isnan(x.sharpe) else -9))]
    best = focus_list[0]                       # primary, drives single-factor panels (7-9)
    single = len(focus_list) == 1
    ttl = best.factor if single else "focus factors"

    # (1) 2-year cumulative L/S — focus factor(s)
    ax = fig.add_subplot(gs[0, 0])
    cutoff = expo["date"].max() - pd.Timedelta(days=730)
    for fr in focus_list:
        c = fr.cum[fr.cum.index >= cutoff]
        if len(c):
            c = c / c.iloc[0]
            ax.plot(c.index, c.values, lw=1.8, color=color[fr.factor], label=fr.factor)
            if single:
                ax.fill_between(c.index, 1.0, c.values, where=c.values >= 1.0,
                                color=color[fr.factor], alpha=0.12)
    ax.set_title(f"2-Year Cumulative L/S — {ttl}", fontweight="bold")
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.grid(alpha=0.3)
    if not single:
        ax.legend(fontsize=7, loc="upper left")

    # (2) Full-history cumulative L/S — focus factor(s)
    ax = fig.add_subplot(gs[0, 1])
    for fr in focus_list:
        ax.plot(fr.cum.index, fr.cum.values, lw=1.8, color=color[fr.factor], label=fr.factor)
        if single:
            ax.fill_between(fr.cum.index, 1.0, fr.cum.values, where=fr.cum.values >= 1.0,
                            color=color[fr.factor], alpha=0.12)
    ax.set_title(f"Full-History Cumulative L/S — {ttl}", fontweight="bold")
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.grid(alpha=0.3)
    if not single:
        ax.legend(fontsize=7, loc="upper left")

    # (3) Annualized L/S return bar (color by Sharpe)
    ax = fig.add_subplot(gs[0, 2])
    rs = sorted(results, key=lambda x: x.ann_ret)
    yy = np.arange(len(rs))
    vals = [fr.ann_ret * 100 for fr in rs]
    barcolors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
    ax.barh(yy, vals, color=barcolors)
    ax.set_yticks(yy)
    ax.set_yticklabels([fr.factor for fr in rs], fontsize=8)
    for i, fr in enumerate(rs):
        ax.text(vals[i], i, f"  S={fr.sharpe:.2f}", va="center",
                fontsize=7, color="black")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Annualized L/S Return (label = Sharpe)", fontweight="bold")
    ax.set_xlabel("% / yr")
    ax.grid(alpha=0.3, axis="x")

    # (4) Factor-return correlation heatmap
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=6.5)
    ax.set_yticklabels(cols, fontsize=6.5)
    ax.set_title("Factor L/S Return Correlation", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (5) PCA scree
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(range(1, len(var_exp) + 1), var_exp * 100, color="#1f77b4", alpha=0.8)
    ax.plot(range(1, len(var_exp) + 1), np.cumsum(var_exp) * 100, "o-",
            color="#ff7f0e", label="cumulative")
    ax.set_title(f"PCA Scree — eff. independent bets ≈ {eff_bets:.1f}", fontweight="bold")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("% variance explained")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (6) IC bar
    ax = fig.add_subplot(gs[1, 2])
    rs2 = sorted(results, key=lambda x: x.ic_mean)
    yy = np.arange(len(rs2))
    icv = [fr.ic_mean for fr in rs2]
    ax.barh(yy, icv, color=["#2ca02c" if v >= 0 else "#d62728" for v in icv])
    ax.set_yticks(yy)
    ax.set_yticklabels([fr.factor for fr in rs2], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Mean Cross-Sectional Rank IC", fontweight="bold")
    ax.grid(alpha=0.3, axis="x")

    # (7) Exposure dispersion over time (breadth of the bet) for top-Sharpe factor
    ax = fig.add_subplot(gs[2, 0])
    zb = (expo[expo["factor"] == best.factor]
          .groupby("date")["zscore"].std())
    ax.plot(zb.index, zb.values, color=color[best.factor], lw=1.1)
    ax.set_title(f"Exposure Dispersion over Time — {best.factor}", fontweight="bold")
    ax.set_ylabel("cross-sectional std of z")
    ax.grid(alpha=0.3)

    # (8) Rolling 1y Sharpe of best factor
    ax = fig.add_subplot(gs[2, 1])
    win = ppy
    rmean = best.ls_returns.rolling(win).mean()
    rstd = best.ls_returns.rolling(win).std()
    rsharpe = (rmean / rstd) * np.sqrt(ppy)
    ax.plot(rsharpe.index, rsharpe.values, color=color[best.factor], lw=1.1)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_title(f"Rolling 1Y Sharpe — {best.factor}", fontweight="bold")
    ax.grid(alpha=0.3)

    # (9) Drawdown of best factor
    ax = fig.add_subplot(gs[2, 2])
    cum = best.cum
    dd = cum / cum.cummax() - 1.0
    ax.fill_between(dd.index, dd.values * 100, 0, color="#d62728", alpha=0.5)
    ax.set_title(f"Drawdown — {best.factor} (maxDD {best.max_dd*100:.1f}%)", fontweight="bold")
    ax.set_ylabel("%")
    ax.grid(alpha=0.3)

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n>> Dashboard written to: {out_path}")


def make_advanced_dashboard(results, ls_matrix, expo, rets, universe, macro,
                            sub_sharpe, sub_ic, regimes, rate_beta_df,
                            out_path, is_synth, ppy, focus=None):
    """Page 2: charts for sub-industry, regime conditioning, and statistical rigor."""
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.32)
    title = "Axioma Insurance Factors — Sub-Industry / Regime / Rigor"
    if is_synth:
        title += "   [SYNTHETIC DEMO DATA]"
    fig.suptitle(title, fontsize=17, fontweight="bold")

    factors = list(ls_matrix.columns)

    # (1) Sub-industry Sharpe heatmap
    ax = fig.add_subplot(gs[0, 0])
    S = sub_sharpe.loc[factors]
    im = ax.imshow(S.values.astype(float), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(S.columns))); ax.set_xticklabels(S.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(S.index))); ax.set_yticklabels(S.index, fontsize=7)
    ax.set_title("Factor L/S Sharpe by Sub-Industry", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (2) Sub-industry IC heatmap
    ax = fig.add_subplot(gs[0, 1])
    I = sub_ic.loc[factors]
    im = ax.imshow(I.values.astype(float), cmap="RdBu_r", vmin=-0.06, vmax=0.06, aspect="auto")
    ax.set_xticks(range(len(I.columns))); ax.set_xticklabels(I.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(I.index))); ax.set_yticklabels(I.index, fontsize=7)
    ax.set_title("Factor IC by Sub-Industry", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (3) Regime conditioning — rising vs falling rates (grouped bars)
    ax = fig.add_subplot(gs[0, 2])
    if regimes and "Rising rates" in regimes:
        rr = regimes["Rising rates"].reindex(factors) * 100
        fr = regimes["Falling rates"].reindex(factors) * 100
        y = np.arange(len(factors))
        ax.barh(y - 0.2, rr.values, height=0.4, color="#d62728", label="Rising rates")
        ax.barh(y + 0.2, fr.values, height=0.4, color="#1f77b4", label="Falling rates")
        ax.set_yticks(y); ax.set_yticklabels(factors, fontsize=7)
        ax.axvline(0, color="k", lw=0.8); ax.legend(fontsize=7)
        ax.set_title("Factor L/S by Rate Regime (ann %)", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No macro series\n(pass --macro)", ha="center", va="center")
        ax.set_title("Rate Regime", fontweight="bold"); ax.axis("off")
    ax.grid(alpha=0.3, axis="x")

    # (4) Regime conditioning — high vs low VIX
    ax = fig.add_subplot(gs[1, 0])
    if regimes and "High VIX" in regimes:
        hv = regimes["High VIX"].reindex(factors) * 100
        lv = regimes["Low VIX"].reindex(factors) * 100
        y = np.arange(len(factors))
        ax.barh(y - 0.2, hv.values, height=0.4, color="#9467bd", label="High VIX")
        ax.barh(y + 0.2, lv.values, height=0.4, color="#2ca02c", label="Low VIX")
        ax.set_yticks(y); ax.set_yticklabels(factors, fontsize=7)
        ax.axvline(0, color="k", lw=0.8); ax.legend(fontsize=7)
        ax.set_title("Factor L/S by VIX Regime (ann %)", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No VIX series", ha="center", va="center"); ax.axis("off")
    ax.grid(alpha=0.3, axis="x")

    # (5) Per-name rate beta scatter (the duration trade)
    ax = fig.add_subplot(gs[1, 1])
    if rate_beta_df is not None and len(rate_beta_df):
        subs = sorted(rate_beta_df["sub_industry"].unique())
        cmap = plt.cm.tab10(np.linspace(0, 1, len(subs)))
        for c, s in zip(cmap, subs):
            d = rate_beta_df[rate_beta_df["sub_industry"] == s]
            ax.scatter(d["rate_beta"], d["ann_ret"] * 100, color=c, s=28, label=s, alpha=0.8)
        for _, row in rate_beta_df.iterrows():
            ax.annotate(row["asset_id"], (row["rate_beta"], row["ann_ret"] * 100),
                        fontsize=5.5, alpha=0.6)
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_xlabel("rate beta (return per +1 Δ10Y)"); ax.set_ylabel("ann return %")
        ax.legend(fontsize=6, ncol=2); ax.set_title("Per-Name Rate Beta", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No macro series", ha="center", va="center"); ax.axis("off")
    ax.grid(alpha=0.3)

    # (6) IC decay curve — focus factor(s)
    ax = fig.add_subplot(gs[1, 2])
    horizons = (1, 5, 21, 63)
    focus_list = focus or [max(results, key=lambda x: abs(x.ic_mean) if not np.isnan(x.ic_mean) else 0)]
    if len(focus_list) == 1:
        fr = focus_list[0]
        yv = [ic_decay(expo, rets, fr.factor, horizons)[h] for h in horizons]
        ax.plot(horizons, yv, "o-", color="#1f77b4", lw=1.8)
        ax.fill_between(horizons, 0, yv, color="#1f77b4", alpha=0.12)
        ax.set_title(f"IC Decay by Horizon — {fr.factor}", fontweight="bold")
    else:
        cmap2 = plt.cm.tab10(np.linspace(0, 1, len(focus_list)))
        for col, fr in zip(cmap2, focus_list):
            yv = [ic_decay(expo, rets, fr.factor, horizons)[h] for h in horizons]
            ax.plot(horizons, yv, "o-", color=col, lw=1.5, label=fr.factor)
        ax.legend(fontsize=6.5)
        ax.set_title("IC Decay by Horizon — focus factors", fontweight="bold")
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("forward horizon (periods)"); ax.set_ylabel("mean IC")
    ax.grid(alpha=0.3)

    # (7) Quintile monotonicity for top-4 by |Sharpe|
    ax = fig.add_subplot(gs[2, 0])
    topf = sorted(results, key=lambda x: abs(x.sharpe) if not np.isnan(x.sharpe) else 0,
                  reverse=True)[:4]
    w = 0.2
    for k, fr in enumerate(topf):
        means, mono = quintile_monotonicity(expo, rets, fr.factor)
        x = np.arange(len(means)) + (k - 1.5) * w
        ax.bar(x, means * 1e4, width=w, label=f"{fr.factor} (ρ={mono:+.2f})")
    ax.set_xticks(range(5)); ax.set_xticklabels([f"Q{i+1}" for i in range(5)])
    ax.set_ylabel("fwd return (bps)")
    ax.set_title("Quintile Monotonicity (Q1→Q5)", fontweight="bold")
    ax.legend(fontsize=6.5); ax.grid(alpha=0.3, axis="y")

    # (8) Bootstrap Sharpe CI whiskers + deflated-Sharpe flag
    ax = fig.add_subplot(gs[2, 1])
    per_period_sr = {fr.factor: fr.ann_ret / ppy / (fr.ann_vol / np.sqrt(ppy))
                     if fr.ann_vol > 0 else np.nan for fr in results}
    sr_list = list(per_period_sr.values())
    rows = sorted(results, key=lambda x: x.sharpe if not np.isnan(x.sharpe) else -9)
    y = np.arange(len(rows))
    for i, fr in enumerate(rows):
        lo, hi = block_bootstrap_sharpe_ci(fr.ls_returns.values, ppy)
        dsr = deflated_sharpe(per_period_sr[fr.factor], sr_list, len(fr.ls_returns))
        sig = (not np.isnan(dsr)) and dsr > 0.95
        col = "#2ca02c" if sig else "#7f7f7f"
        ax.plot([lo, hi], [i, i], color=col, lw=2)
        ax.plot(fr.sharpe, i, "o", color=col, ms=5)
        if not np.isnan(dsr):
            ax.text(hi, i, f"  DSR={dsr:.2f}", va="center", fontsize=6)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([fr.factor for fr in rows], fontsize=7)
    ax.set_xlabel("annualized Sharpe (95% block-bootstrap CI)")
    ax.set_title("Sharpe CI + Deflated SR (green=DSR>0.95)", fontweight="bold")
    ax.grid(alpha=0.3, axis="x")

    # (9) Fama-MacBeth pure premia vs marginal (quintile) premia
    ax = fig.add_subplot(gs[2, 2])
    fm = fama_macbeth(expo, rets, factors)
    if len(fm):
        marg = pd.Series({fr.factor: fr.ann_ret for fr in results}).reindex(fm.index) * 100
        y = np.arange(len(fm))
        ax.barh(y - 0.2, fm["premium_ann"].values * 100, height=0.4,
                color="#ff7f0e", label="Fama-MacBeth (pure)")
        ax.barh(y + 0.2, marg.values, height=0.4, color="#1f77b4", label="Quintile (marginal)")
        ax.set_yticks(y); ax.set_yticklabels(fm.index, fontsize=7)
        ax.axvline(0, color="k", lw=0.8); ax.legend(fontsize=7)
        ax.set_title("Pure vs Marginal Factor Premia (ann %)", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center"); ax.axis("off")
    ax.grid(alpha=0.3, axis="x")

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f">> Advanced dashboard written to: {out_path}")


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exposures", help="CSV/Parquet: [date, asset_id, factor, zscore]")
    p.add_argument("--returns", help="CSV/Parquet: [date, asset_id, ret]")
    p.add_argument("--universe", help="CSV/Parquet: [asset_id, name, in_insurance, sub_industry]")
    p.add_argument("--macro", help="CSV/Parquet: [date, ten_year, credit_spread, vix]")
    p.add_argument("--out", default="insurance_factor_dashboard.png", help="output PNG")
    p.add_argument("--factors", nargs="*", help="restrict analysis to these factor names")
    p.add_argument("--focus-factors", nargs="*",
                   help="factor name(s) to feature in the single-line charts "
                        "(default: top factor by Sharpe). One name = single line; "
                        'a few = a small labeled set. e.g. --focus-factors "Value" "Momentum"')
    args = p.parse_args(argv)

    expo, rets, universe, macro, is_synth = load_data(args)

    # restrict to insurance universe
    ins_ids = set(universe.loc[universe["in_insurance"], "asset_id"])
    if ins_ids:
        expo = expo[expo["asset_id"].isin(ins_ids)]
        rets = rets[rets["asset_id"].isin(ins_ids)]

    factors = args.factors or sorted(expo["factor"].unique())
    expo = expo[expo["factor"].isin(factors)]
    ppy = _infer_ppy(pd.DatetimeIndex(sorted(expo["date"].unique())))

    print(f"Analyzing {len(factors)} factors over {expo['asset_id'].nunique()} insurance names...\n")
    results = []
    for f in factors:
        fr = analyze_factor(expo, rets, f, ppy)
        if len(fr.ls_returns) >= 10:
            results.append(fr)
    if not results:
        sys.exit("No factor produced enough overlapping signal/return periods.")

    ls_matrix = pd.concat([fr.ls_returns.rename(fr.factor) for fr in results], axis=1)
    pca = pca_on_factor_returns(ls_matrix)
    reg = basket_factor_regression(expo, rets, ls_matrix, universe)
    _, crowd_avg = exposure_crowding(expo, factors)

    # resolve --focus-factors to result objects (preserve requested order)
    name_to_fr = {fr.factor: fr for fr in results}
    focus = None
    if args.focus_factors:
        focus = [name_to_fr[f] for f in args.focus_factors if f in name_to_fr]
        missing = [f for f in args.focus_factors if f not in name_to_fr]
        if missing:
            print(f">> Note: focus factor(s) unavailable / no signal: {', '.join(missing)}")
        if not focus:
            print(">> No requested focus factors available; defaulting to top factor by Sharpe.")
            focus = None
        else:
            print(f">> Charts focused on: {', '.join(fr.factor for fr in focus)}\n")

    print_report(results, ls_matrix, pca, reg, crowd_avg, ppy, is_synth, expo, universe)
    make_dashboard(results, ls_matrix, pca, expo, args.out, is_synth, ppy, focus=focus)

    # ---- advanced modules: sub-industry, regime, statistical rigor ----
    sub_sharpe, sub_ic = subindustry_decomposition(expo, rets, universe, factors, ppy)
    regimes = regime_conditioning(ls_matrix, macro, ppy)
    rate_beta_df = per_name_rate_beta(rets, macro, universe)
    print_advanced(sub_sharpe, sub_ic, regimes, rate_beta_df, results, ls_matrix,
                   expo, rets, factors, ppy)

    stem = args.out.rsplit(".", 1)
    adv_out = f"{stem[0]}_advanced.{stem[1] if len(stem) > 1 else 'png'}"
    make_advanced_dashboard(results, ls_matrix, expo, rets, universe, macro,
                            sub_sharpe, sub_ic, regimes, rate_beta_df,
                            adv_out, is_synth, ppy, focus=focus)


def print_advanced(sub_sharpe, sub_ic, regimes, rate_beta_df, results, ls_matrix,
                   expo, rets, factors, ppy):
    line = "=" * 100
    print("\n" + line)
    print("ADVANCED: SUB-INDUSTRY DECOMPOSITION / REGIME CONDITIONING / STATISTICAL RIGOR")
    print(line)

    # --- sub-industry ---
    print("1) FACTOR L/S SHARPE BY SUB-INDUSTRY (above/below-median split within bucket)")
    print("-" * 100)
    with pd.option_context("display.width", 120, "display.max_columns", 20):
        print(sub_sharpe.round(2).fillna("  .").to_string())
    print("\n   -> Where each factor actually pays. A factor strong only in one column is a")
    print("      sub-industry effect, not a broad insurance premium.")
    # most concentrated factor
    spread = (sub_sharpe.max(axis=1) - sub_sharpe.min(axis=1)).sort_values(ascending=False)
    if len(spread):
        f0 = spread.index[0]
        best_sub = sub_sharpe.loc[f0].idxmax()
        print(f"   -> Most sub-industry-dependent: {f0} "
              f"(best in {best_sub}, Sharpe spread {spread.iloc[0]:.2f} across buckets).")
    print()

    # --- regime ---
    if regimes:
        print("2) FACTOR L/S ANNUALIZED RETURN BY MACRO REGIME (%)")
        print("-" * 100)
        reg_df = pd.DataFrame(regimes) * 100
        with pd.option_context("display.width", 120):
            print(reg_df.round(1).to_string())
        if "Rising rates" in regimes:
            diff = (regimes["Rising rates"] - regimes["Falling rates"]).sort_values()
            print(f"\n   -> Most rate-sensitive factor premia: "
                  f"{diff.index[0]} (better when rates fall), "
                  f"{diff.index[-1]} (better when rates rise).")
        print()

    if rate_beta_df is not None and len(rate_beta_df):
        print("3) PER-NAME RATE BETA (return per +1pt Δ10Y) — the duration trade")
        print("-" * 100)
        by_sub = (rate_beta_df.groupby("sub_industry")["rate_beta"].mean()
                  .sort_values(ascending=False))
        print("   Avg rate beta by sub-industry: " +
              ", ".join(f"{s}={v:+.2f}" for s, v in by_sub.items()))
        top = rate_beta_df.reindex(rate_beta_df["rate_beta"].abs().sort_values(ascending=False).index).head(5)
        print("   Most rate-sensitive names: " +
              ", ".join(f"{r.asset_id}({r.rate_beta:+.2f})" for r in top.itertuples()))
        print()

    # --- statistical rigor ---
    print("4) STATISTICAL RIGOR")
    print("-" * 100)
    per_period_sr = {fr.factor: (fr.ann_ret / ppy) / (fr.ann_vol / np.sqrt(ppy))
                     if fr.ann_vol > 0 else np.nan for fr in results}
    sr_list = list(per_period_sr.values())
    hdr = f"{'Factor':<26}{'Sharpe':>8}{'BootCI95':>20}{'DeflatedSR':>12}{'Mono ρ':>9}"
    print(hdr); print("-" * 100)
    for fr in sorted(results, key=lambda x: x.sharpe if not np.isnan(x.sharpe) else -9, reverse=True):
        lo, hi = block_bootstrap_sharpe_ci(fr.ls_returns.values, ppy)
        dsr = deflated_sharpe(per_period_sr[fr.factor], sr_list, len(fr.ls_returns))
        _, mono = quintile_monotonicity(expo, rets, fr.factor)
        ci = f"[{lo:+.2f}, {hi:+.2f}]"
        flag = " *" if (not np.isnan(dsr) and dsr > 0.95) else "  "
        print(f"{fr.factor:<26}{fr.sharpe:>8.2f}{ci:>20}{dsr:>11.2f}{flag}{mono:>8.2f}")
    print("-" * 100)
    print("   BootCI excludes 0 => Sharpe robust to resampling. DeflatedSR>0.95 (*) => survives")
    print("   the multiple-testing haircut from scanning all factors. Mono ρ near +1 => clean")
    print("   monotone Q1->Q5 (real signal), near 0 => tail-driven / noisy.")

    fm = fama_macbeth(expo, rets, factors)
    if len(fm):
        print("\n5) FAMA-MACBETH PURE FACTOR PREMIA (controls for cross-factor collinearity)")
        print("-" * 100)
        fm2 = fm.copy(); fm2["premium_ann"] = fm2["premium_ann"] * 100
        fm2 = fm2.sort_values("premium_ann", ascending=False)
        for f, row in fm2.iterrows():
            mark = " *" if abs(row["t_stat"]) > 2 else "  "
            print(f"   {f:<26}{row['premium_ann']:>7.1f}%/yr   t={row['t_stat']:+.2f}{mark}")
        print("   -> 'Pure' premia after orthogonalizing against the other factors; compare to")
        print("      the marginal quintile premia. Shrinkage here = the bet was mostly crowding.")
    print(line)


if __name__ == "__main__":
    main()
