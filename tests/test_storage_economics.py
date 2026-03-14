"""Tests for storage economics analysis — correlations, walk-forward, sensitivity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.storage_economics import (
    compute_rolling_correlation,
    compute_sensitivity,
    lagged_correlation,
    multi_lag_correlation,
    pearson_with_ci,
    walk_forward_test,
)


class TestPearsonWithCI:
    """Test Pearson correlation with Fisher z-transform CI."""

    def test_perfect_positive(self):
        """Perfectly correlated data should give r ≈ 1."""
        x = pd.Series(range(100), dtype=float)
        y = pd.Series(range(100), dtype=float)
        r, p, ci_lo, ci_hi = pearson_with_ci(x, y)
        assert abs(r - 1.0) < 0.001
        assert p < 0.001
        assert ci_lo > 0.95

    def test_perfect_negative(self):
        """Perfectly anti-correlated data should give r ≈ -1."""
        x = pd.Series(range(100), dtype=float)
        y = pd.Series(range(100, 0, -1), dtype=float)
        r, p, ci_lo, ci_hi = pearson_with_ci(x, y)
        assert abs(r - (-1.0)) < 0.001
        assert ci_hi < -0.95

    def test_uncorrelated(self):
        """Random data should have r ≈ 0 with wide CI."""
        np.random.seed(42)
        x = pd.Series(np.random.randn(200))
        y = pd.Series(np.random.randn(200))
        r, p, ci_lo, ci_hi = pearson_with_ci(x, y)
        assert abs(r) < 0.2
        assert ci_lo < 0 < ci_hi

    def test_insufficient_data(self):
        """Fewer than 10 observations should return defaults."""
        x = pd.Series([1.0, 2.0, 3.0])
        y = pd.Series([4.0, 5.0, 6.0])
        r, p, ci_lo, ci_hi = pearson_with_ci(x, y)
        assert r == 0.0
        assert p == 1.0

    def test_with_nans(self):
        """NaN values should be handled gracefully."""
        x = pd.Series([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)
        y = pd.Series([2, 4, 6, np.nan, 10, 12, 14, 16, 18, 20, 22, 24], dtype=float)
        r, p, ci_lo, ci_hi = pearson_with_ci(x, y)
        assert r > 0.9  # strong positive after removing NaNs

    def test_ci_contains_r(self):
        """Confidence interval should contain the point estimate."""
        np.random.seed(42)
        x = pd.Series(np.random.randn(100))
        y = x + np.random.randn(100) * 0.5  # correlated
        r, p, ci_lo, ci_hi = pearson_with_ci(x, y)
        assert ci_lo <= r <= ci_hi

    def test_ci_width_decreases_with_n(self):
        """Larger samples should give narrower CIs."""
        np.random.seed(42)
        x50 = pd.Series(np.random.randn(50))
        y50 = x50 * 0.8 + np.random.randn(50) * 0.3
        _, _, lo50, hi50 = pearson_with_ci(x50, y50)

        x200 = pd.Series(np.random.randn(200))
        y200 = x200 * 0.8 + np.random.randn(200) * 0.3
        _, _, lo200, hi200 = pearson_with_ci(x200, y200)

        assert (hi200 - lo200) < (hi50 - lo50)


class TestLaggedCorrelation:
    """Test lagged correlation computation."""

    def test_zero_lag(self):
        """Zero lag should match direct pearson."""
        np.random.seed(42)
        inv = pd.Series(np.random.randn(100))
        spread = inv * 0.6 + np.random.randn(100) * 0.3
        result = lagged_correlation(inv, spread, lag_weeks=0)
        assert result.lag_weeks == 0
        assert result.r > 0.5

    def test_positive_lag(self):
        """Positive lag means inventory leads spread."""
        np.random.seed(42)
        inv = pd.Series(np.random.randn(200))
        # Spread follows inventory with a 5-day delay
        spread = pd.Series(np.zeros(200))
        spread.iloc[5:] = inv.iloc[:-5].values
        result = lagged_correlation(inv, spread, lag_weeks=1)
        assert result.lag_weeks == 1
        assert isinstance(result.r, float)
        assert isinstance(result.n, int)


class TestMultiLagCorrelation:
    """Test multi-lag correlation computation."""

    def test_returns_all_lags(self):
        """Should return a result for each requested lag."""
        np.random.seed(42)
        inv = pd.Series(np.random.randn(200))
        spread = pd.Series(np.random.randn(200))
        results = multi_lag_correlation(inv, spread, lags=[0, 1, 2, 4])
        assert len(results) == 4
        assert [r.lag_weeks for r in results] == [0, 1, 2, 4]


class TestWalkForwardTest:
    """Test walk-forward validation."""

    def test_stable_relationship(self):
        """Consistent correlation should be marked stable."""
        np.random.seed(42)
        n = 200
        inv = pd.Series(np.random.randn(n))
        spread = inv * 0.7 + np.random.randn(n) * 0.2
        result = walk_forward_test(inv, spread)
        assert result is not None
        assert result.stable is True
        assert result.in_sample_r > 0
        assert result.out_of_sample_r > 0

    def test_insufficient_data(self):
        """Too few observations should return None."""
        inv = pd.Series([1.0, 2.0, 3.0])
        spread = pd.Series([4.0, 5.0, 6.0])
        result = walk_forward_test(inv, spread)
        assert result is None

    def test_split_sizes(self):
        """Verify in-sample and out-of-sample sizes sum correctly."""
        np.random.seed(42)
        n = 100
        inv = pd.Series(np.random.randn(n))
        spread = pd.Series(np.random.randn(n))
        result = walk_forward_test(inv, spread, train_fraction=0.6)
        if result:
            assert result.in_sample_n + result.out_of_sample_n == n


class TestComputeSensitivity:
    """Test OLS sensitivity regression."""

    def test_positive_beta(self):
        """Positive relationship should give positive beta."""
        np.random.seed(42)
        zscore = pd.Series(np.random.randn(100))
        spread = zscore * 0.005 + np.random.randn(100) * 0.001
        result = compute_sensitivity(zscore, spread)
        assert result is not None
        assert result.beta > 0

    def test_insufficient_data(self):
        """Too few observations should return None."""
        zscore = pd.Series([1.0, 2.0])
        spread = pd.Series([0.01, 0.02])
        result = compute_sensitivity(zscore, spread)
        assert result is None

    def test_r_squared_range(self):
        """R² should be between 0 and 1."""
        np.random.seed(42)
        zscore = pd.Series(np.random.randn(200))
        spread = zscore * 0.01 + np.random.randn(200) * 0.005
        result = compute_sensitivity(zscore, spread)
        assert result is not None
        assert 0 <= result.r_squared <= 1

    def test_with_nans(self):
        """NaN values should be excluded."""
        np.random.seed(42)
        zscore = pd.Series(np.random.randn(100))
        spread = pd.Series(np.random.randn(100) * 0.01)
        zscore.iloc[10:15] = np.nan
        spread.iloc[50:55] = np.nan
        result = compute_sensitivity(zscore, spread)
        assert result is not None
        assert result.n < 100


class TestRollingCorrelation:
    """Test rolling Pearson correlation."""

    def test_output_length(self):
        """Output should have same length as input."""
        np.random.seed(42)
        inv = pd.Series(np.random.randn(100))
        spread = pd.Series(np.random.randn(100))
        result = compute_rolling_correlation(inv, spread, window=20)
        assert len(result) == 100

    def test_nan_warmup(self):
        """First window-1 values should be NaN."""
        inv = pd.Series(np.random.randn(50))
        spread = pd.Series(np.random.randn(50))
        result = compute_rolling_correlation(inv, spread, window=20)
        assert result.iloc[:19].isna().all()
        assert result.iloc[19:].notna().any()

    def test_values_in_range(self):
        """All non-NaN values should be between -1 and 1."""
        np.random.seed(42)
        inv = pd.Series(np.random.randn(200))
        spread = pd.Series(np.random.randn(200))
        result = compute_rolling_correlation(inv, spread, window=30)
        valid = result.dropna()
        assert (valid >= -1.0).all()
        assert (valid <= 1.0).all()
