"""CAN SLIM per-letter grading (William O'Neil's 7-factor system).

Pure grading functions only - no file/parquet/network I/O here. The pipeline
runner (run_setups.py) reads fundamentals.parquet, rs.parquet, leaders.parquet,
regime.json and analysis.parquet and passes the raw values in, so these
functions stay unit-testable without touching disk.

Signal quality per letter (see docs / owner report for full rationale):
  C - full signal: eps_growth_pct + sales_growth_pct already computed from
      real quarterly yfinance data (tscreener/pipeline/fundamentals.py).
  A - proxy, data-dependent: annual_eps_growth_pct comes from yfinance's
      ANNUAL income statement, which is only 3-4 years deep and sometimes
      unavailable (delisted comps, recent IPOs, data gaps). None -> a
      documented neutral default, not a real annual-growth read.
  N - proxy only: "new highs" is approximated by proximity to the 52-week
      high (off_52w_high_pct from analysis.parquet). "New products / new
      management / new highs in the group" are NOT captured - there is no
      data source for that in this pipeline yet.
  S - proxy + historical trend (once accumulated): float_shares percentile
      rank across the scored universe (crude size proxy) and short_float_pct
      (supply/squeeze signal) remain the base; float_trend from
      fundamentals.py's float_history/ snapshots ("shrinking"/"growing"/
      "flat"/"insufficient_data", the same accumulate-daily-snapshots
      mechanism inst_trend uses) is now blended on top when available -
      shrinking float (buybacks) adds a bonus, growing float (dilution)
      subtracts. Until ~3-6 daily pipeline runs have accumulated snapshots
      for a ticker, float_trend is "insufficient_data" and the letter falls
      back to the snapshot-only proxy (float percentile + short interest),
      identical to the pre-trend behavior.
  L - full signal: rs_rating (IBD-style, 0-99) and/or leadership_score,
      already computed cross-sectionally in rs.py / leadership.py.
  I - full signal: inst_trend ("accumulating"/"distributing"/"flat"/
      "insufficient_data") from the finviz snapshot history, optionally
      boosted by inst_own_pct.
  M - full signal, but market-wide not stock-specific: regime.json's
      benchmark.composite.verdict is the same value for every ticker on a
      given day - it answers "is the market in a confirmed uptrend"
      (O'Neil's 'M'), not anything about the individual stock.
"""

LETTERS = ("c", "a", "n", "s", "l", "i", "m")

DEFAULT_CFG = {
    "c": {"eps_target_pct": 25.0, "sales_target_pct": 20.0,
          "eps_weight": 0.6, "sales_weight": 0.4},
    "a": {"target_pct": 25.0, "missing_default": 50.0},
    "n": {"max_below_pct": 15.0},
    "s": {"short_neutral_max_pct": 5.0, "short_squeeze_max_pct": 20.0,
          "short_squeeze_bonus": 10.0, "short_overhang_penalty": 5.0,
          "missing_default": 50.0,
          "float_shrinking_bonus": 15.0, "float_growing_penalty": 15.0},
    "i": {"accumulating": 85.0, "flat": 55.0, "distributing": 25.0,
          "insufficient_data": 50.0, "high_own_threshold_pct": 60.0,
          "high_own_bonus": 10.0},
    "m": {"RISK_ON": 90.0, "NEUTRAL": 60.0, "CAUTION": 35.0, "RISK_OFF": 10.0,
          "missing_default": 50.0},
    "letter_bands": {"A": 80.0, "B": 65.0, "C": 50.0, "D": 35.0},
}


def _cfg(canslim_settings: dict | None, section: str) -> dict:
    """Merge configured overrides for `section` over DEFAULT_CFG[section]."""
    merged = dict(DEFAULT_CFG[section])
    if canslim_settings and section in canslim_settings:
        merged.update(canslim_settings[section])
    return merged


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _is_nan(v) -> bool:
    try:
        return v is None or v != v  # NaN != NaN
    except TypeError:
        return v is None


def _linear_band(value: float, target: float) -> float:
    """0% -> 50, +target% -> 100, -target% -> 0, clamp 0-100."""
    return _clamp(50.0 + (value / target) * 50.0)


# ── C: current quarterly earnings/sales ──────────────────────────────────────

def grade_c(eps_growth_pct, sales_growth_pct, cfg: dict | None = None) -> float:
    """Blend eps_growth_pct and sales_growth_pct into 0-100 (default 60/40).
    Missing component(s) fall back to a neutral 50 for that half of the blend."""
    c = _cfg(cfg, "c")
    eps_score = 50.0 if _is_nan(eps_growth_pct) else _linear_band(
        eps_growth_pct, c["eps_target_pct"])
    sales_score = 50.0 if _is_nan(sales_growth_pct) else _linear_band(
        sales_growth_pct, c["sales_target_pct"])
    return round(c["eps_weight"] * eps_score + c["sales_weight"] * sales_score, 1)


# ── A: annual earnings growth (3-year trend) ─────────────────────────────────

def grade_a(annual_eps_growth_pct, cfg: dict | None = None) -> float:
    """>= target% CAGR-style growth -> 100, 0% -> 50, negative -> below 50.
    None/NaN (yfinance annual data unavailable) -> documented neutral default."""
    c = _cfg(cfg, "a")
    if _is_nan(annual_eps_growth_pct):
        return c["missing_default"]
    return round(_linear_band(annual_eps_growth_pct, c["target_pct"]), 1)


# ── N: new highs proxy (new products/management NOT captured) ───────────────

def grade_n(off_52w_high_pct, cfg: dict | None = None) -> float:
    """off_52w_high_pct = % below the 52-week high (0 = at/above the high).
    At-or-above high -> 100, max_below_pct+ below -> 0, linear between.
    NOTE: this is a proxy for the "new highs" component only - "new
    products / new management" are not modeled."""
    c = _cfg(cfg, "n")
    if _is_nan(off_52w_high_pct):
        return 50.0
    off = max(0.0, float(off_52w_high_pct))
    return round(_clamp(100.0 - (off / c["max_below_pct"]) * 100.0), 1)


# ── S: supply and demand (float size + short interest proxy) ────────────────

def grade_s(float_percentile: float | None, short_float_pct, cfg: dict | None = None,
            float_trend: str | None = None) -> float:
    """float_percentile: this ticker's float_shares percentile rank across the
    scored universe, in [0, 1] where 0 = smallest float (best score). Pass
    None when no universe comparison is available (single-ticker context).

    Short interest treated as: <= short_neutral_max_pct neutral (no
    adjustment), between neutral and short_squeeze_max_pct -> squeeze-
    potential bonus, above short_squeeze_max_pct -> flagged with a mild
    (not severe) overhang penalty - O'Neil's supply/demand letter isn't
    exclusively bearish on high short interest, since a crowded short can
    also fuel a squeeze.

    float_trend (optional): "shrinking" / "growing" / "flat" /
    "insufficient_data" from fundamentals.py's float_trend_from_history(),
    i.e. the actual historical share-count trend once >= 3 daily snapshots
    have accumulated. "shrinking" (buybacks reducing float) adds
    float_shrinking_bonus - bullish per O'Neil's supply/demand thesis: fewer
    shares + demand -> price moves faster. "growing" (dilution) subtracts
    float_growing_penalty. "flat" and "insufficient_data" (or None, the
    default) leave the score unchanged, so this stays backward-compatible
    with the pre-trend snapshot-only proxy until history accumulates."""
    c = _cfg(cfg, "s")
    base = c["missing_default"] if float_percentile is None else _clamp(
        100.0 * (1.0 - float(float_percentile)))
    adj = 0.0
    if not _is_nan(short_float_pct):
        sf = float(short_float_pct)
        if sf > c["short_squeeze_max_pct"]:
            adj = -c["short_overhang_penalty"]
        elif sf > c["short_neutral_max_pct"]:
            adj = c["short_squeeze_bonus"]
    if float_trend == "shrinking":
        adj += c["float_shrinking_bonus"]
    elif float_trend == "growing":
        adj -= c["float_growing_penalty"]
    return round(_clamp(base + adj), 1)


# ── L: leader or laggard ─────────────────────────────────────────────────────

def grade_l(rs_rating, leadership_score=None, cfg: dict | None = None) -> float:
    """rs_rating and leadership_score are both already 0-99 (IBD-style);
    normalize to 0-100 and average when both are present."""
    scores = []
    if not _is_nan(rs_rating):
        scores.append(_clamp(float(rs_rating) / 99.0 * 100.0))
    if not _is_nan(leadership_score):
        scores.append(_clamp(float(leadership_score) / 99.0 * 100.0))
    if not scores:
        return 50.0
    return round(sum(scores) / len(scores), 1)


# ── I: institutional sponsorship ─────────────────────────────────────────────

def grade_i(inst_trend, inst_own_pct=None, cfg: dict | None = None) -> float:
    c = _cfg(cfg, "i")
    base = c.get(inst_trend, c["insufficient_data"])
    if not _is_nan(inst_own_pct) and float(inst_own_pct) > c["high_own_threshold_pct"]:
        base += c["high_own_bonus"]
    return round(_clamp(base), 1)


# ── M: market direction (market-wide, same for every ticker) ────────────────

def grade_m(regime_verdict, cfg: dict | None = None) -> float:
    c = _cfg(cfg, "m")
    return c.get(regime_verdict, c["missing_default"])


# ── composite ─────────────────────────────────────────────────────────────────

def composite_grade(letter_scores: dict, weights: dict | None = None) -> float:
    """Equal-weighted (1/7 each) average of the 7 letter scores unless
    `weights` overrides. Equal weighting was chosen because O'Neil doesn't
    rank the letters against each other in the source material - CAN SLIM is
    presented as 7 necessary conditions, not a ranked list."""
    weights = weights or {k: 1.0 / len(LETTERS) for k in LETTERS}
    total_w = sum(weights.get(k, 0.0) for k in LETTERS)
    if total_w <= 0:
        return 0.0
    total = sum(letter_scores.get(k, 0.0) * weights.get(k, 0.0) for k in LETTERS)
    return round(total / total_w, 1)


def letter_band(composite: float, cfg: dict | None = None) -> str:
    """Map a 0-100 composite to an A-F display band."""
    bands = _cfg(cfg, "letter_bands") if cfg else DEFAULT_CFG["letter_bands"]
    if composite >= bands["A"]:
        return "A"
    if composite >= bands["B"]:
        return "B"
    if composite >= bands["C"]:
        return "C"
    if composite >= bands["D"]:
        return "D"
    return "F"


def score_ticker(*, eps_growth_pct=None, sales_growth_pct=None,
                  annual_eps_growth_pct=None, off_52w_high_pct=None,
                  float_percentile=None, short_float_pct=None,
                  rs_rating=None, leadership_score=None,
                  inst_trend="insufficient_data", inst_own_pct=None,
                  float_trend="insufficient_data",
                  regime_verdict="NEUTRAL", settings: dict | None = None) -> dict:
    """Compute all 7 letter grades + composite + letter band for one ticker.
    `settings` is the full config["canslim"] dict (or None for defaults)."""
    grades = {
        "c": grade_c(eps_growth_pct, sales_growth_pct, settings),
        "a": grade_a(annual_eps_growth_pct, settings),
        "n": grade_n(off_52w_high_pct, settings),
        "s": grade_s(float_percentile, short_float_pct, settings, float_trend=float_trend),
        "l": grade_l(rs_rating, leadership_score, settings),
        "i": grade_i(inst_trend, inst_own_pct, settings),
        "m": grade_m(regime_verdict, settings),
    }
    weights = (settings or {}).get("weights")
    comp = composite_grade(grades, weights)
    return {
        "canslim_c_grade": grades["c"],
        "canslim_a_grade": grades["a"],
        "canslim_n_grade": grades["n"],
        "canslim_s_grade": grades["s"],
        "canslim_l_grade": grades["l"],
        "canslim_i_grade": grades["i"],
        "canslim_m_grade": grades["m"],
        "canslim_composite": comp,
        "canslim_letter": letter_band(comp, settings),
    }


if __name__ == "__main__":
    # Self-check: strong ticker -> high grades, weak ticker -> low grades.
    strong = score_ticker(eps_growth_pct=40, sales_growth_pct=30,
                          annual_eps_growth_pct=30, off_52w_high_pct=2,
                          float_percentile=0.2, short_float_pct=8,
                          rs_rating=92, leadership_score=85,
                          inst_trend="accumulating", inst_own_pct=70,
                          regime_verdict="RISK_ON")
    weak = score_ticker(eps_growth_pct=-10, sales_growth_pct=-5,
                        annual_eps_growth_pct=-10, off_52w_high_pct=40,
                        float_percentile=0.9, short_float_pct=30,
                        rs_rating=20, inst_trend="distributing",
                        regime_verdict="RISK_OFF")
    assert strong["canslim_letter"] == "A", strong
    assert weak["canslim_letter"] == "F", weak
    assert strong["canslim_composite"] > weak["canslim_composite"], (strong, weak)
    for k, v in strong.items():
        print(f"{k}: {v}")
    print("OK: strong=A, weak=F")
