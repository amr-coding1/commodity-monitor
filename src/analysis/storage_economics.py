"""Storage economics analysis — inventory-spread correlations and sensitivity."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.database import (
    get_analytics_series,
    get_connection,
    get_inventory_series,
    get_futures_series,
    list_commodities,
)
from src.settings import (
    CORRELATION_LAGS_WEEKS,
    ROLLING_CORR_WINDOW,
    WALK_FORWARD_TRAIN_FRACTION,
)

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Result of a Pearson correlation with confidence interval."""

    lag_weeks: int
    r: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "lag_weeks": self.lag_weeks,
            "r": round(self.r, 4),
            "p_value": round(self.p_value, 6),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "n": self.n,
        }

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05


@dataclass
class SensitivityResult:
    """OLS regression result: spread ~ beta * inventory_zscore."""

    beta: float
    std_err: float
    t_stat: float
    p_value: float
    r_squared: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "beta": round(self.beta, 6),
            "std_err": round(self.std_err, 6),
            "t_stat": round(self.t_stat, 4),
            "p_value": round(self.p_value, 6),
            "r_squared": round(self.r_squared, 4),
            "n": self.n,
        }


@dataclass
class WalkForwardResult:
    """Walk-forward validation result."""

    in_sample_r: float
    out_of_sample_r: float
    in_sample_n: int
    out_of_sample_n: int
    stable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_sample_r": round(self.in_sample_r, 4),
            "out_of_sample_r": round(self.out_of_sample_r, 4),
            "in_sample_n": self.in_sample_n,
            "out_of_sample_n": self.out_of_sample_n,
            "stable": self.stable,
        }


@dataclass
class CommodityAnalysis:
    """Full analysis result for a single commodity."""

    commodity: str
    correlations: list[CorrelationResult] = field(default_factory=list)
    sensitivity: SensitivityResult | None = None
    walk_forward: WalkForwardResult | None = None
    rolling_correlation: pd.Series | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "commodity": self.commodity,
            "correlations": [c.to_dict() for c in self.correlations],
            "sensitivity": self.sensitivity.to_dict() if self.sensitivity else None,
            "walk_forward": self.walk_forward.to_dict() if self.walk_forward else None,
        }


def pearson_with_ci(
    x: pd.Series,
    y: pd.Series,
    confidence: float = 0.95,
) -> tuple[float, float, float, float]:
    """Pearson correlation with Fisher z-transform confidence interval.

    Args:
        x, y: Paired data series.
        confidence: Confidence level (default 0.95).

    Returns:
        (r, p_value, ci_lower, ci_upper)
    """
    mask = x.notna() & y.notna()
    x_clean, y_clean = x[mask].values, y[mask].values
    n = len(x_clean)

    if n < 10:
        return 0.0, 1.0, -1.0, 1.0

    r, p = stats.pearsonr(x_clean, y_clean)

    # Fisher z-transform for CI
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf((1 + confidence) / 2)

    ci_lower = np.tanh(z - z_crit * se)
    ci_upper = np.tanh(z + z_crit * se)

    return float(r), float(p), float(ci_lower), float(ci_upper)


def lagged_correlation(
    inventory: pd.Series,
    spread: pd.Series,
    lag_weeks: int,
) -> CorrelationResult:
    """Compute Pearson correlation between inventory (lagged) and spread.

    Positive lag means inventory leads spread by N weeks.
    """
    lag_days = lag_weeks * 5  # trading days
    if lag_days > 0:
        inv_lagged = inventory.shift(lag_days)
    else:
        inv_lagged = inventory

    r, p, ci_lo, ci_hi = pearson_with_ci(inv_lagged, spread)
    mask = inv_lagged.notna() & spread.notna()
    n = mask.sum()

    return CorrelationResult(
        lag_weeks=lag_weeks,
        r=r,
        p_value=p,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n=int(n),
    )


def multi_lag_correlation(
    inventory: pd.Series,
    spread: pd.Series,
    lags: list[int] | None = None,
) -> list[CorrelationResult]:
    """Compute correlations at multiple lag offsets."""
    lags = lags or CORRELATION_LAGS_WEEKS
    return [lagged_correlation(inventory, spread, lag) for lag in lags]


def walk_forward_test(
    inventory: pd.Series,
    spread: pd.Series,
    train_fraction: float | None = None,
) -> WalkForwardResult | None:
    """Walk-forward validation: compare in-sample vs out-of-sample correlation.

    If correlations are stable (same sign, OOS > 50% of IS), marks as stable.
    """
    fraction = train_fraction or WALK_FORWARD_TRAIN_FRACTION
    mask = inventory.notna() & spread.notna()
    inv_clean = inventory[mask]
    spread_clean = spread[mask]

    n = len(inv_clean)
    if n < 60:
        return None

    split = int(n * fraction)
    is_inv, is_spread = inv_clean.iloc[:split], spread_clean.iloc[:split]
    oos_inv, oos_spread = inv_clean.iloc[split:], spread_clean.iloc[split:]

    if len(is_inv) < 20 or len(oos_inv) < 20:
        return None

    is_r, _, _, _ = pearson_with_ci(is_inv, is_spread)
    oos_r, _, _, _ = pearson_with_ci(oos_inv, oos_spread)

    # Stability: same sign and OOS magnitude ≥ 50% of IS
    same_sign = (is_r * oos_r) > 0
    magnitude_ok = abs(oos_r) >= 0.5 * abs(is_r) if is_r != 0 else True
    stable = same_sign and magnitude_ok

    return WalkForwardResult(
        in_sample_r=is_r,
        out_of_sample_r=oos_r,
        in_sample_n=len(is_inv),
        out_of_sample_n=len(oos_inv),
        stable=stable,
    )


def compute_sensitivity(
    inventory_zscore: pd.Series,
    spread_pct: pd.Series,
) -> SensitivityResult | None:
    """OLS regression: spread_pct ~ beta * inventory_zscore.

    Sensitivity = how much the spread changes per unit z-score change.
    """
    mask = inventory_zscore.notna() & spread_pct.notna()
    x = inventory_zscore[mask].values
    y = spread_pct[mask].values
    n = len(x)

    if n < 20:
        return None

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    t_stat = slope / std_err if std_err != 0 else 0.0

    return SensitivityResult(
        beta=float(slope),
        std_err=float(std_err),
        t_stat=float(t_stat),
        p_value=float(p_value),
        r_squared=float(r_value ** 2),
        n=n,
    )


def compute_rolling_correlation(
    inventory: pd.Series,
    spread: pd.Series,
    window: int | None = None,
) -> pd.Series:
    """Rolling Pearson correlation between inventory and spread."""
    w = window or ROLLING_CORR_WINDOW
    return inventory.rolling(window=w, min_periods=w).corr(spread)


class StorageEconomicsAnalyser:
    """Run full storage economics analysis for one or all commodities."""

    def __init__(self, conn=None):
        self.conn = conn or get_connection()

    def analyse_commodity(self, commodity: str) -> CommodityAnalysis:
        """Full analysis for a single commodity."""
        result = CommodityAnalysis(commodity=commodity)

        analytics = get_analytics_series(self.conn, commodity)
        if analytics.empty or "zscore_1y" not in analytics.columns:
            logger.warning("Insufficient analytics data for %s", commodity)
            return result

        inv_z = analytics["zscore_1y"].astype(float)
        spread = analytics["spread_pct"].astype(float) if "spread_pct" in analytics.columns else None

        if spread is None or spread.isna().all():
            logger.warning("No spread data for %s", commodity)
            return result

        # Multi-lag correlations
        result.correlations = multi_lag_correlation(inv_z, spread)

        # Walk-forward validation
        result.walk_forward = walk_forward_test(inv_z, spread)

        # Sensitivity (OLS)
        result.sensitivity = compute_sensitivity(inv_z, spread)

        # Rolling correlation
        result.rolling_correlation = compute_rolling_correlation(inv_z, spread)

        return result

    def run_all(self) -> dict[str, CommodityAnalysis]:
        """Run analysis for all commodities."""
        results = {}
        for commodity in list_commodities():
            logger.info("Analysing %s", commodity)
            results[commodity] = self.analyse_commodity(commodity)
        return results
