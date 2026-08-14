"""
RS Sparkline Calculator.

Calculates the RS ratio (stock_price / SPY_price) for the last 30 trading days
for sparkline visualization in the bulk screener results.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RSSparklineCalculator:
    """
    Calculate RS ratio series for sparkline visualization.

    Replicates Google Sheets formula:
    =SPARKLINE(QUERY(HSTACK(
      GOOGLEFINANCE(stock,"price",WORKDAY(TODAY(),-30),TODAY()),
      GOOGLEFINANCE(SPY,"price",WORKDAY(TODAY(),-30),TODAY())
    ),"SELECT Col2/Col4"))

    This gets stock and SPY prices for last 30 trading days,
    divides stock by SPY to get RS ratio.
    """

    SPARKLINE_DAYS = 30  # Number of trading days for sparkline

    def calculate_rs_sparkline(
        self,
        stock_prices: pd.Series,
        spy_prices: pd.Series,
        normalize: bool = True
    ) -> Dict:
        """
        Calculate RS ratio series for the last 30 trading days.

        Args:
            stock_prices: Stock closing prices (chronological order, oldest first)
            spy_prices: SPY closing prices (chronological order, oldest first)
            normalize: If True, normalize to start at 1.0 for better visual comparison

        Returns:
            Dict with:
            - rs_data: List of 30 RS ratio values (or None if insufficient data)
            - rs_trend: -1 (declining), 0 (flat), 1 (improving)
        """
        if len(stock_prices) < self.SPARKLINE_DAYS or len(spy_prices) < self.SPARKLINE_DAYS:
            logger.debug(
                f"Insufficient data for RS sparkline: stock={len(stock_prices)}, spy={len(spy_prices)}"
            )
            return {
                "rs_data": None,
                "rs_trend": 0,
            }

        try:
            # Get last 30 trading days (most recent data) as float so NaN/Inf
            # checks are consistent regardless of source dtype.
            stock_last_30 = np.asarray(
                stock_prices.iloc[-self.SPARKLINE_DAYS:].values, dtype=float
            )
            spy_last_30 = np.asarray(
                spy_prices.iloc[-self.SPARKLINE_DAYS:].values, dtype=float
            )

            # Calculate RS ratio (stock / SPY). Suppress the divide-by-zero
            # warning so the downstream NaN/Inf scrubbing path stays quiet.
            with np.errstate(divide="ignore", invalid="ignore"):
                rs_ratio = stock_last_30 / spy_last_30

            # If the window contains no finite samples, bail out to None.
            # Previously NaN/Inf were rewritten to 1.0, which after the
            # normalization step (divide by rs_ratio[0]) produced a large
            # spike on the affected bar — most visible on markets where the
            # raw ratio is far from 1 (e.g. CN stock_CNY / CSI300 ≈ 0.01,
            # so a planted 1.0 became a ~100× spike on the latest date).
            finite_mask = np.isfinite(rs_ratio)
            if not finite_mask.any():
                logger.debug("All-non-finite RS series; returning None sparkline")
                return {
                    "rs_data": None,
                    "rs_trend": 0,
                }

            # Replace non-finite values with the first finite sample so the
            # series is fully finite and the trailing bar flattens to the
            # leading level (which normalization will collapse to 1.0)
            # instead of producing a spike.
            first_finite = float(rs_ratio[np.argmax(finite_mask)])
            rs_ratio = np.where(finite_mask, rs_ratio, first_finite)

            # Normalize to start at 1.0 if requested (better for visual comparison)
            if normalize and rs_ratio[0] != 0:
                rs_ratio = rs_ratio / rs_ratio[0]

            # Calculate trend using linear regression slope
            x = np.arange(len(rs_ratio))
            slope, _ = np.polyfit(x, rs_ratio, 1)

            # Determine trend direction
            # Threshold: slope must be significant relative to the data range
            data_range = np.max(rs_ratio) - np.min(rs_ratio)
            slope_threshold = data_range * 0.01 if data_range > 0 else 0.0001

            if slope > slope_threshold:
                trend = 1  # Improving
            elif slope < -slope_threshold:
                trend = -1  # Declining
            else:
                trend = 0  # Flat

            # Round values for JSON storage efficiency (4 decimal places)
            rs_data = [round(float(v), 4) for v in rs_ratio]

            # Final safety net: if rounding/normalization produced any
            # non-finite value, collapse the whole payload to None so the
            # schema sanitizer drops it cleanly.
            if not all(np.isfinite(v) for v in rs_data):
                logger.debug("Non-finite values after normalization; returning None sparkline")
                return {
                    "rs_data": None,
                    "rs_trend": 0,
                }

            return {
                "rs_data": rs_data,
                "rs_trend": trend,
            }

        except Exception as e:
            logger.warning(f"Error calculating RS sparkline: {e}")
            return {
                "rs_data": None,
                "rs_trend": 0,
            }
