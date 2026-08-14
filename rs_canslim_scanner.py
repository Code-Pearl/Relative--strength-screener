#!/usr/bin/env python3
"""
RS & CAN SLIM Standalone Scanner
=================================

Feed it a list of tickers and it will:
  1. Pull daily price history for each symbol + a benchmark (default SPY)
  2. Run the existing Relative Strength engine (relative_strength.py) to
     produce a weighted RS rating plus 1M / 3M / 12M RS ratings, ranked
     cross-sectionally (percentile) against the tickers you fed it
  3. Pull light fundamentals from yfinance and run them through the existing
     CAN SLIM grading engine (canslim_calculations.py) to get the 7 letter
     grades, a composite score, and an A-F band
  4. Render a single self-contained HTML report styled after the
     Investing Compass DataDownloader Pro UI (index.html / style.css),
     with the Investing Compass logo in the sidebar footer

Usage
-----
    pip install yfinance pandas numpy --break-system-packages

    # Straight list of tickers
    python rs_canslim_scanner.py AAPL MSFT NVDA GOOGL AMZN

    # From a file (one ticker per line, or comma/space separated)
    python rs_canslim_scanner.py --file tickers.txt

    # Preview the report with synthetic data, no network / yfinance needed
    python rs_canslim_scanner.py --demo

    # Custom benchmark / output path
    python rs_canslim_scanner.py AAPL MSFT --benchmark QQQ -o report.html

Notes on data honesty
----------------------
Several CAN SLIM letters are documented *proxies* in canslim_calculations.py
because the full pipeline (finviz institutional snapshots, float history,
regime.json) isn't available outside that codebase. This scanner is explicit
about which letters are full-signal vs. proxy vs. neutral-default in the
"Methodology" panel of the generated report - it does not pretend to have
data it doesn't have. See canslim_calculations.py's module docstring for the
full per-letter rationale.
"""
from __future__ import annotations

import argparse
import base64
import html
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relative_strength import RelativeStrengthCalculator  # noqa: E402
from rs_sparkline import RSSparklineCalculator  # noqa: E402
from canslim_calculations import score_ticker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                     datefmt="%H:%M:%S")
logger = logging.getLogger("rs_canslim_scanner")

LOOKBACK_DAYS = 420  # trading-day buffer so a full 252-day RS window survives holidays/gaps
HERE = Path(__file__).resolve().parent


# ──────────────────────────────────────────────────────────────────────────
# Data fetching
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class TickerRaw:
    symbol: str
    close_asc: Optional[pd.Series] = None   # chronological, oldest first (for sparkline)
    info: dict = field(default_factory=dict)
    error: Optional[str] = None


def fetch_price_history(symbol: str, period_days: int = LOOKBACK_DAYS) -> Optional[pd.Series]:
    """Daily close prices, chronological (oldest first). None on failure."""
    import yfinance as yf
    try:
        df = yf.Ticker(symbol).history(period=f"{period_days}d", auto_adjust=True)
        if df is None or df.empty or "Close" not in df:
            return None
        s = df["Close"].dropna()
        return s if len(s) else None
    except Exception as e:  # noqa: BLE001 - network/library errors are all "couldn't fetch"
        logger.warning("Price fetch failed for %s: %s", symbol, e)
        return None


def fetch_fundamentals(symbol: str) -> dict:
    """Light fundamentals snapshot from yfinance's .info. Missing fields are
    left absent - callers must treat absence as 'unknown', not zero."""
    import yfinance as yf
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as e:  # noqa: BLE001
        logger.warning("Fundamentals fetch failed for %s: %s", symbol, e)
        return {}
    return info


def annual_eps_growth_from_income_stmt(symbol: str) -> Optional[float]:
    """Best-effort 2-year annual EPS growth %, from yfinance's annual income
    statement (Net Income / Basic shares across the two most recent fiscal
    years). yfinance's annual history is shallow (3-4 yrs) and sometimes
    unavailable entirely - None means 'couldn't compute', not 'no growth'.
    """
    import yfinance as yf
    try:
        t = yf.Ticker(symbol)
        fin = t.income_stmt
        if fin is None or fin.empty:
            return None
        cols = list(fin.columns)[:2]  # two most recent fiscal years
        if len(cols) < 2:
            return None

        def _eps_for(col):
            for row_name in ("Diluted EPS", "Basic EPS"):
                if row_name in fin.index:
                    v = fin.loc[row_name, col]
                    if pd.notna(v):
                        return float(v)
            ni_row = "Net Income" if "Net Income" in fin.index else None
            shares = t.info.get("basicAverageShares") or t.info.get("sharesOutstanding")
            if ni_row and shares:
                v = fin.loc[ni_row, col]
                if pd.notna(v) and shares:
                    return float(v) / float(shares)
            return None

        recent, prior = _eps_for(cols[0]), _eps_for(cols[1])
        if recent is None or prior is None or prior == 0:
            return None
        return (recent - prior) / abs(prior) * 100.0
    except Exception as e:  # noqa: BLE001
        logger.debug("Annual EPS growth unavailable for %s: %s", symbol, e)
        return None


def simple_market_regime(benchmark_close_asc: pd.Series) -> str:
    """Coarse regime proxy from the benchmark's own trend - price vs its 50
    and 200-day SMAs. This is NOT the pipeline's regime.json (which blends
    breadth, momentum, and more); it's a transparent stand-in so grade_m has
    something real to read in a standalone context."""
    if benchmark_close_asc is None or len(benchmark_close_asc) < 200:
        return "NEUTRAL"
    price = float(benchmark_close_asc.iloc[-1])
    sma50 = float(benchmark_close_asc.iloc[-50:].mean())
    sma200 = float(benchmark_close_asc.iloc[-200:].mean())
    if price > sma50 > sma200:
        return "RISK_ON"
    if price < sma50 < sma200:
        return "RISK_OFF"
    if price > sma200:
        return "NEUTRAL"
    return "CAUTION"


# ──────────────────────────────────────────────────────────────────────────
# Synthetic demo data (no network required)
# ──────────────────────────────────────────────────────────────────────────

def _synthetic_series(n_days: int, drift: float, vol: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n_days)
    price = 100 * np.cumprod(1 + rets)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    return pd.Series(price, index=idx)


def demo_universe() -> tuple[dict[str, TickerRaw], pd.Series]:
    random.seed(7)
    profiles = {
        # symbol: (drift, vol, growth-ish info)
        "LEAD1": (0.0035, 0.018, dict(eps=45, sales=32, ann_eps=38, float_pct=0.15,
                                       short=6, inst_own=72, held_growing=True)),
        "LEAD2": (0.0028, 0.02, dict(eps=30, sales=22, ann_eps=25, float_pct=0.30,
                                      short=10, inst_own=61, held_growing=True)),
        "MID1": (0.0012, 0.017, dict(eps=12, sales=9, ann_eps=8, float_pct=0.5,
                                      short=4, inst_own=45, held_growing=None)),
        "MID2": (0.0006, 0.015, dict(eps=4, sales=3, ann_eps=2, float_pct=0.55,
                                      short=3, inst_own=40, held_growing=None)),
        "LAG1": (-0.0008, 0.022, dict(eps=-8, sales=-4, ann_eps=-12, float_pct=0.8,
                                       short=18, inst_own=22, held_growing=False)),
        "LAG2": (-0.0018, 0.026, dict(eps=-20, sales=-15, ann_eps=-25, float_pct=0.9,
                                       short=28, inst_own=15, held_growing=False)),
    }
    out: dict[str, TickerRaw] = {}
    for i, (sym, (drift, vol, meta)) in enumerate(profiles.items()):
        close = _synthetic_series(LOOKBACK_DAYS, drift, vol, seed=100 + i)
        info = {
            "regularMarketPrice": float(close.iloc[-1]),
            "shortName": sym,
            "_demo_eps_growth": meta["eps"],
            "_demo_sales_growth": meta["sales"],
            "_demo_ann_eps_growth": meta["ann_eps"],
            "_demo_float_pct": meta["float_pct"],
            "shortPercentOfFloat": meta["short"] / 100.0,
            "heldPercentInstitutions": meta["inst_own"] / 100.0,
            "_demo_held_growing": meta["held_growing"],
        }
        out[sym] = TickerRaw(symbol=sym, close_asc=close, info=info)
    bench = _synthetic_series(LOOKBACK_DAYS, 0.0006, 0.011, seed=999)
    return out, bench


# ──────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class ScanRow:
    symbol: str
    price: Optional[float]
    rs_rating: float
    rs_1m: float
    rs_3m: float
    rs_12m: float
    sparkline: Optional[list]
    sparkline_trend: int
    canslim: dict
    error: Optional[str] = None


def run_scan(tickers: list[str], benchmark: str, demo: bool) -> tuple[list[ScanRow], dict]:
    calc = RelativeStrengthCalculator(benchmark=benchmark)
    sparkcalc = RSSparklineCalculator()

    raw: dict[str, TickerRaw] = {}
    bench_close_asc: Optional[pd.Series] = None

    if demo:
        raw, bench_close_asc = demo_universe()
        tickers = list(raw.keys())
    else:
        logger.info("Fetching benchmark %s ...", benchmark)
        bench_close_asc = fetch_price_history(benchmark)
        if bench_close_asc is None:
            raise SystemExit(f"Could not fetch benchmark data for {benchmark}. "
                              f"Check your network / that yfinance is installed, "
                              f"or run with --demo to preview the report.")
        for sym in tickers:
            logger.info("Fetching %s ...", sym)
            close = fetch_price_history(sym)
            info = fetch_fundamentals(sym) if close is not None else {}
            err = None if close is not None else "no price data returned"
            raw[sym] = TickerRaw(symbol=sym, close_asc=close, info=info, error=err)

    bench_desc = bench_close_asc.iloc[::-1].reset_index(drop=True)

    # Pass 1: raw relative performances (needed to build the cross-sectional
    # universe for percentile ranking, same contract calculate_all_rs_ratings
    # expects via universe_performances).
    weighted_pool, pool_1m, pool_3m, pool_12m = [], [], [], []
    stock_desc: dict[str, pd.Series] = {}
    for sym, t in raw.items():
        if t.close_asc is None or len(t.close_asc) < 60:
            continue
        desc = t.close_asc.iloc[::-1].reset_index(drop=True)
        stock_desc[sym] = desc
        w = calc.calculate_weighted_performance(desc, bench_desc)
        if w is not None:
            weighted_pool.append(w)
        for period, pool in ((21, pool_1m), (63, pool_3m), (252, pool_12m)):
            sr = calc.calculate_return(desc, period)
            br = calc.calculate_return(bench_desc, period)
            if sr is not None and br is not None:
                pool.append(sr - br)

    universe = {"weighted": weighted_pool, 21: pool_1m, 63: pool_3m, 252: pool_12m}

    # Cross-sectional float percentile (0 = smallest float = best S score)
    float_shares = {}
    for sym, t in raw.items():
        fv = t.info.get("_demo_float_pct")
        if fv is None:
            fv = t.info.get("floatShares")
        if fv is not None:
            float_shares[sym] = float(fv)
    float_ranked = sorted(float_shares.items(), key=lambda kv: kv[1])
    float_percentile = {}
    if float_ranked:
        n = len(float_ranked)
        for rank, (sym, _) in enumerate(float_ranked):
            float_percentile[sym] = rank / max(1, n - 1)

    regime_verdict = simple_market_regime(bench_close_asc)

    rows: list[ScanRow] = []
    for sym in tickers:
        t = raw.get(sym)
        if t is None or t.close_asc is None or sym not in stock_desc:
            rows.append(ScanRow(sym, None, 0, 0, 0, 0, None, 0, {}, error=(t.error if t else "not found")))
            continue

        desc = stock_desc[sym]
        rs = calc.calculate_all_rs_ratings(sym, desc, bench_desc, universe)

        spark = sparkcalc.calculate_rs_sparkline(t.close_asc, bench_close_asc)

        info = t.info
        eps_growth = info.get("_demo_eps_growth")
        if eps_growth is None:
            eps_growth = info.get("earningsQuarterlyGrowth")
            eps_growth = eps_growth * 100 if eps_growth is not None else None
        sales_growth = info.get("_demo_sales_growth")
        if sales_growth is None:
            sales_growth = info.get("revenueGrowth")
            sales_growth = sales_growth * 100 if sales_growth is not None else None
        ann_eps_growth = info.get("_demo_ann_eps_growth")
        if ann_eps_growth is None:
            ann_eps_growth = annual_eps_growth_from_income_stmt(sym)

        off_52w = None
        high52 = info.get("fiftyTwoWeekHigh")
        cur_price = float(t.close_asc.iloc[-1])
        if high52:
            off_52w = max(0.0, (float(high52) - cur_price) / float(high52) * 100.0)
        elif len(t.close_asc) >= 252:
            hi = float(t.close_asc.iloc[-252:].max())
            off_52w = max(0.0, (hi - cur_price) / hi * 100.0) if hi else None

        short_float = info.get("shortPercentOfFloat")
        short_float = short_float * 100 if short_float is not None else None
        inst_own = info.get("heldPercentInstitutions")
        inst_own = inst_own * 100 if inst_own is not None else None

        float_trend = "insufficient_data"
        if info.get("_demo_held_growing") is True:
            float_trend = "shrinking"
        elif info.get("_demo_held_growing") is False:
            float_trend = "growing"

        canslim = score_ticker(
            eps_growth_pct=eps_growth,
            sales_growth_pct=sales_growth,
            annual_eps_growth_pct=ann_eps_growth,
            off_52w_high_pct=off_52w,
            float_percentile=float_percentile.get(sym),
            short_float_pct=short_float,
            rs_rating=rs["rs_rating"],
            leadership_score=None,
            inst_trend="insufficient_data",
            inst_own_pct=inst_own,
            float_trend=float_trend,
            regime_verdict=regime_verdict,
        )

        rows.append(ScanRow(
            symbol=sym,
            price=cur_price,
            rs_rating=rs["rs_rating"],
            rs_1m=rs["rs_rating_1m"],
            rs_3m=rs["rs_rating_3m"],
            rs_12m=rs["rs_rating_12m"],
            sparkline=spark["rs_data"],
            sparkline_trend=spark["rs_trend"],
            canslim=canslim,
        ))

    rows.sort(key=lambda r: (r.canslim.get("canslim_composite", -1), r.rs_rating), reverse=True)
    meta = {"benchmark": benchmark, "regime_verdict": regime_verdict, "demo": demo}
    return rows, meta


# ──────────────────────────────────────────────────────────────────────────
# HTML report
# ──────────────────────────────────────────────────────────────────────────

LETTER_BAND_COLOR = {
    "A": ("#38a169", "#f0fff4"),
    "B": ("#3182ce", "#ebf8ff"),
    "C": ("#d69e2e", "#fffff0"),
    "D": ("#dd6b20", "#fffaf0"),
    "F": ("#e53e3e", "#fff5f5"),
}


def _rs_tier_color(v: float) -> str:
    if v >= 80:
        return "#38a169"
    if v >= 60:
        return "#3182ce"
    if v >= 40:
        return "#d69e2e"
    return "#e53e3e"


def _grade_cell(v: float) -> str:
    color = _rs_tier_color(v)
    return f'<span class="g-cell" style="color:{color}">{v:.0f}</span>'


def _sparkline_svg(data: Optional[list], trend: int) -> str:
    if not data or len(data) < 2:
        return '<span class="muted">—</span>'
    w, h, pad = 64, 22, 2
    lo, hi = min(data), max(data)
    rng = (hi - lo) or 1.0
    step = (w - 2 * pad) / (len(data) - 1)
    pts = []
    for i, v in enumerate(data):
        x = pad + i * step
        y = h - pad - ((v - lo) / rng) * (h - 2 * pad)
        pts.append(f"{x:.1f},{y:.1f}")
    color = "#38a169" if trend > 0 else ("#e53e3e" if trend < 0 else "#8a93a6")
    poly = " ".join(pts)
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" class="spark">'
            f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="1.6" '
            f'stroke-linejoin="round" stroke-linecap="round"/></svg>')


def render_html(rows: list[ScanRow], meta: dict, logo_svg: str) -> str:
    n = len(rows)
    scored = [r for r in rows if not r.error]
    avg_rs = round(sum(r.rs_rating for r in scored) / len(scored), 1) if scored else 0
    a_count = sum(1 for r in scored if r.canslim.get("canslim_letter") == "A")
    f_count = sum(1 for r in scored if r.canslim.get("canslim_letter") == "F")
    err_count = n - len(scored)

    def row_html(r: ScanRow) -> str:
        sym = html.escape(r.symbol)
        if r.error:
            return (f'<tr><td class="sym">{sym}</td>'
                     f'<td colspan="16" class="muted">Skipped — {html.escape(r.error)}</td></tr>')
        c = r.canslim
        band = c.get("canslim_letter", "F")
        fg, bg = LETTER_BAND_COLOR.get(band, LETTER_BAND_COLOR["F"])
        price_txt = f"${r.price:,.2f}" if r.price is not None else "—"
        letters = "".join(
            f'<td>{_grade_cell(c.get(f"canslim_{L}_grade", 0))}</td>'
            for L in ("c", "a", "n", "s", "l", "i", "m")
        )
        return f"""<tr>
      <td class="sym">{sym}</td>
      <td>{price_txt}</td>
      <td><span class="rs-pill" style="background:{_rs_tier_color(r.rs_rating)}1a;color:{_rs_tier_color(r.rs_rating)}">{r.rs_rating:.0f}</span></td>
      <td>{_grade_cell(r.rs_1m)}</td>
      <td>{_grade_cell(r.rs_3m)}</td>
      <td>{_grade_cell(r.rs_12m)}</td>
      <td>{_sparkline_svg(r.sparkline, r.sparkline_trend)}</td>
      {letters}
      <td><b>{c.get('canslim_composite', 0):.1f}</b></td>
      <td><span class="band-pill" style="background:{bg};color:{fg}">{band}</span></td>
    </tr>"""

    rows_html = "\n".join(row_html(r) for r in rows)
    demo_banner = ('<div class="demo-banner"><i class="fa fa-flask"></i> Demo mode — '
                   'synthetic prices/fundamentals, not real market data. Run without '
                   '<code>--demo</code> and with tickers to scan real symbols.</div>') if meta.get("demo") else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>RS &amp; CAN SLIM Scanner</title>
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet"/>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Faculty+Glyphic:wght@400;600;700&display=swap" rel="stylesheet">
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0 }}
:root {{
  --bg:#f0f2f5; --surface:#fff; --surface2:#f8f9fb; --border:#e8ecf0; --border2:#d4dae2;
  --text:#5a6478; --text2:#1e2535; --text3:#8a93a6; --primary:#3182ce; --primary-light:#ebf4ff;
  --charcoal:#1e2535; --green:#38a169; --green-bg:#f0fff4; --red:#e53e3e; --red-bg:#fff5f5;
  --yellow:#d69e2e; --r:12px; --rs:8px;
  --shadow-sm:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04);
}}
body {{ font-family:'Faculty Glyphic',sans-serif; background:var(--bg); color:var(--text); font-size:14px; line-height:1.5 }}
#app {{ display:flex; min-height:100vh }}
#sidebar {{ width:240px; background:var(--charcoal); display:flex; flex-direction:column; flex-shrink:0 }}
.logo-area {{ padding:24px 20px 20px; border-bottom:1px solid rgba(255,255,255,.06); display:flex; align-items:center; gap:11px }}
.logo-icon {{ width:34px; height:34px; flex-shrink:0 }}
.brand {{ font-weight:700; font-size:15px; color:#fff; letter-spacing:-.02em; text-transform:uppercase }}
.brand-sub {{ font-size:10px; color:rgba(255,255,255,.3); margin-top:1px; font-weight:600; letter-spacing:.1em; text-transform:uppercase }}
.nav-sec {{ padding:18px 12px 8px }}
.nav-lbl {{ font-size:10px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:rgba(255,255,255,.18); padding:0 10px 10px }}
.meta-item {{ display:flex; justify-content:space-between; padding:8px 12px; font-size:11px; color:rgba(255,255,255,.55); font-weight:600 }}
.meta-item b {{ color:#fff }}
.sidebar-footer {{ margin-top:auto; padding:16px 20px; border-top:1px solid rgba(255,255,255,.05) }}
.sidebar-footer svg {{ max-width:100%; height:auto; opacity:.85 }}
#main {{ flex:1; display:flex; flex-direction:column; min-width:0 }}
.topbar {{ height:64px; background:var(--surface); border-bottom:1px solid var(--border); display:flex; align-items:center; padding:0 28px; gap:10px; flex-shrink:0 }}
.topbar-title {{ font-size:18px; font-weight:700; flex:1; color:var(--text2); letter-spacing:-.02em }}
.topbar-sub {{ font-size:11px; color:var(--text3); margin-top:1px; font-weight:600; text-transform:uppercase; letter-spacing:.08em }}
.demo-banner {{ background:var(--yellow); color:#fff; padding:8px 28px; font-size:12px; font-weight:700 }}
.demo-banner code {{ background:rgba(255,255,255,.25); padding:1px 5px; border-radius:3px }}
#content {{ flex:1; overflow:auto; padding:20px 28px 32px }}
.stats-row {{ display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap }}
.stat-chip {{ display:flex; align-items:center; gap:7px; padding:8px 14px; border-radius:20px; font-size:12px; font-weight:700; border:1.5px solid var(--border); background:var(--surface) }}
.stat-chip .sv {{ font-size:15px; font-weight:800; color:var(--text2) }}
.stat-total {{ border-left:3px solid var(--text3) }}
.stat-a {{ border-left:3px solid var(--green); color:var(--green) }}
.stat-f {{ border-left:3px solid var(--red); color:var(--red) }}
.stat-err {{ border-left:3px solid var(--yellow); color:var(--yellow) }}
.table-scroll {{ border:1.5px solid var(--border); border-radius:var(--r); background:var(--surface); box-shadow:var(--shadow-sm); overflow:auto }}
table {{ width:100%; border-collapse:collapse }}
thead th {{ background:var(--surface2); padding:11px 10px; text-align:left; font-size:9.5px; font-weight:800; text-transform:uppercase; letter-spacing:.08em; color:var(--text3); border-bottom:1.5px solid var(--border); position:sticky; top:0; cursor:pointer; white-space:nowrap }}
thead th:hover {{ color:var(--primary) }}
tbody tr {{ border-bottom:1px solid var(--border) }}
tbody tr:last-child {{ border-bottom:none }}
tbody tr:hover {{ background:var(--surface2) }}
tbody td {{ padding:9px 10px; font-size:12.5px; vertical-align:middle; color:var(--text2); font-weight:500; white-space:nowrap }}
.sym {{ font-weight:800; color:var(--primary); font-size:13.5px }}
.muted {{ color:var(--text3); font-size:12px; font-weight:600 }}
.g-cell {{ font-weight:800 }}
.rs-pill {{ display:inline-flex; padding:3px 10px; border-radius:20px; font-weight:800; font-size:12px }}
.band-pill {{ display:inline-flex; align-items:center; justify-content:center; width:26px; height:22px; border-radius:6px; font-weight:800; font-size:12px }}
.spark {{ display:block }}
.methodology {{ margin-top:20px; background:var(--surface); border:1.5px solid var(--border); border-radius:var(--r); padding:16px 20px; font-size:12px; color:var(--text3); line-height:1.7 }}
.methodology b {{ color:var(--text2) }}
.methodology .disclaimer {{ margin-top:10px; padding-top:10px; border-top:1px dashed var(--border2); color:var(--text3); font-style:italic }}
</style>
</head>
<body>
<div id="app">
  <nav id="sidebar">
    <div class="logo-area">
      <svg class="logo-icon" viewBox="0 0 1000 1000" xmlns="http://www.w3.org/2000/svg">
        <rect x="361" y="86" width="310" height="741" rx="18" fill="white" opacity=".9" />
        <ellipse cx="641" cy="348" rx="262" ry="262" fill="white" opacity=".75" />
        <ellipse cx="362" cy="650" rx="262" ry="262" fill="white" opacity=".6" />
        <ellipse cx="803" cy="816" rx="82" ry="82" fill="white" opacity=".9" />
      </svg>
      <div>
        <div class="brand">RS Scanner</div>
        <div class="brand-sub">RS &amp; CAN SLIM</div>
      </div>
    </div>
    <div class="nav-sec">
      <div class="nav-lbl">Scan</div>
      <div class="meta-item"><span>Benchmark</span><b>{html.escape(meta['benchmark'])}</b></div>
      <div class="meta-item"><span>Tickers</span><b>{n}</b></div>
      <div class="meta-item"><span>Regime (proxy)</span><b>{html.escape(meta['regime_verdict'])}</b></div>
    </div>
    <div class="sidebar-footer">{logo_svg}</div>
  </nav>

  <div id="main">
    <div class="topbar">
      <div style="flex:1">
        <div class="topbar-title">RS &amp; CAN SLIM Scan Results</div>
        <div class="topbar-sub">{n} tickers · ranked by CAN SLIM composite</div>
      </div>
    </div>
    {demo_banner}
    <div id="content">
      <div class="stats-row">
        <div class="stat-chip stat-total"><span>Total</span><span class="sv">{n}</span></div>
        <div class="stat-chip"><span>Avg RS</span><span class="sv">{avg_rs}</span></div>
        <div class="stat-chip stat-a"><span>A-grade</span><span class="sv">{a_count}</span></div>
        <div class="stat-chip stat-f"><span>F-grade</span><span class="sv">{f_count}</span></div>
        <div class="stat-chip stat-err"><span>Skipped</span><span class="sv">{err_count}</span></div>
      </div>

      <div class="table-scroll">
        <table id="tbl">
          <thead>
            <tr>
              <th onclick="sortTbl(0)">Symbol</th>
              <th onclick="sortTbl(1)">Price</th>
              <th onclick="sortTbl(2)">RS</th>
              <th onclick="sortTbl(3)">RS 1M</th>
              <th onclick="sortTbl(4)">RS 3M</th>
              <th onclick="sortTbl(5)">RS 12M</th>
              <th>RS Trend</th>
              <th onclick="sortTbl(7)">C</th>
              <th onclick="sortTbl(8)">A</th>
              <th onclick="sortTbl(9)">N</th>
              <th onclick="sortTbl(10)">S</th>
              <th onclick="sortTbl(11)">L</th>
              <th onclick="sortTbl(12)">I</th>
              <th onclick="sortTbl(13)">M</th>
              <th onclick="sortTbl(14)">Composite</th>
              <th onclick="sortTbl(15)">Grade</th>
            </tr>
          </thead>
          <tbody>
{rows_html}
          </tbody>
        </table>
      </div>

      <div class="methodology">
        <b>Methodology.</b> RS rating = <code>relative_strength.py</code>'s weighted engine
        (63/126/189/252-day stock-vs-benchmark performance, 40/20/20/20 weighted), percentile-ranked
        against the tickers in this scan. RS 1M/3M/12M are single-period percentile ranks (21/63/252
        trading days). CAN SLIM letters come from <code>canslim_calculations.py</code>: <b>C</b> and
        <b>L</b> are full-signal (quarterly EPS/sales growth; this scan's RS rating). <b>A</b>
        (annual EPS growth), <b>N</b> (52-week-high proximity), and <b>S</b> (float size + short
        interest) are documented proxies. <b>I</b> (institutional trend) and float trend for
        <b>S</b> need multi-day accumulated snapshots this single run doesn't have, so they fall back
        to their neutral defaults. <b>M</b> uses a simplified benchmark-trend proxy (price vs. 50/200-day
        SMA), not a full regime model.
        <div class="disclaimer">This report is a quantitative screening aid only, not investment advice.
        Data quality depends on your data source; verify anything before acting on it.</div>
      </div>
    </div>
  </div>
</div>
<script>
function sortTbl(col) {{
  const tbody = document.querySelector('#tbl tbody');
  const rows = Array.from(tbody.querySelectorAll('tr')).filter(r => r.children.length > 3);
  const skip = Array.from(tbody.querySelectorAll('tr')).filter(r => r.children.length <= 3);
  const dir = tbody.dataset.sortCol == col && tbody.dataset.sortDir == '1' ? -1 : 1;
  tbody.dataset.sortCol = col; tbody.dataset.sortDir = dir;
  rows.sort((a, b) => {{
    const av = a.children[col].innerText.replace(/[$,]/g, '');
    const bv = b.children[col].innerText.replace(/[$,]/g, '');
    const an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return (an - bn) * dir;
    return av.localeCompare(bv) * dir;
  }});
  rows.forEach(r => tbody.appendChild(r));
  skip.forEach(r => tbody.appendChild(r));
}}
</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def parse_ticker_args(args: argparse.Namespace) -> list[str]:
    raw: list[str] = []
    if args.file:
        text = Path(args.file).read_text()
        for chunk in text.replace(",", " ").split():
            raw.append(chunk.strip())
    raw.extend(args.tickers or [])
    seen, out = set(), []
    for t in raw:
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser(description="RS & CAN SLIM standalone scanner")
    ap.add_argument("tickers", nargs="*", help="Ticker symbols, space or comma separated")
    ap.add_argument("--file", "-f", help="Path to a file with tickers (one per line, or comma/space separated)")
    ap.add_argument("--benchmark", "-b", default="SPY", help="Benchmark symbol (default: SPY)")
    ap.add_argument("--output", "-o", default="rs_canslim_report.html", help="Output HTML path")
    ap.add_argument("--demo", action="store_true", help="Use synthetic data, no network/yfinance required")
    args = ap.parse_args()

    tickers = parse_ticker_args(args)
    if not tickers and not args.demo:
        ap.error("No tickers given. Pass symbols, use --file, or use --demo to preview the report.")

    logger.info("Starting scan of %d ticker(s)%s", len(tickers) or 6, " [DEMO]" if args.demo else "")
    rows, meta = run_scan(tickers, args.benchmark, args.demo)

    logo_path = HERE / "investing_compass_logo_footer.svg"
    logo_svg = logo_path.read_text() if logo_path.exists() else ""

    out_html = render_html(rows, meta, logo_svg)
    out_path = Path(args.output)
    out_path.write_text(out_html, encoding="utf-8")
    logger.info("Wrote report: %s (%d rows)", out_path.resolve(), len(rows))


if __name__ == "__main__":
    main()
