"""Tests for inventory normalisation — z-scores and stocks-to-use."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.processing.normaliser import (
    compute_inventory_zscore,
    compute_stocks_to_use_days,
)


class TestComputeInventoryZscore:
    """Test rolling z-score computation."""

    def test_basic_zscore(self):
        """Z-score of constant series should be NaN (zero std)."""
        series = pd.Series([100.0] * 30)
        result = compute_inventory_zscore(series, window=20)
        # All values after warmup should be NaN (zero variance)
        assert result.iloc[-1] != result.iloc[-1]  # NaN check

    def test_increasing_series(self):
        """Linearly increasing series should have positive z-score at end."""
        series = pd.Series(range(100), dtype=float)
        result = compute_inventory_zscore(series, window=20)
        assert result.iloc[-1] > 0

    def test_decreasing_series(self):
        """Linearly decreasing series should have negative z-score at end."""
        series = pd.Series(list(range(100, 0, -1)), dtype=float)
        result = compute_inventory_zscore(series, window=20)
        assert result.iloc[-1] < 0

    def test_insufficient_data(self):
        """With fewer observations than window, z-score should be NaN."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = compute_inventory_zscore(series, window=10)
        assert result.isna().all()

    def test_window_parameter(self):
        """Different windows should produce different z-scores."""
        np.random.seed(42)
        series = pd.Series(np.random.randn(500).cumsum() + 100)
        z20 = compute_inventory_zscore(series, window=20)
        z50 = compute_inventory_zscore(series, window=50)
        # They should differ (not identical)
        overlap = z20.dropna().index.intersection(z50.dropna().index)
        assert not np.allclose(z20[overlap].values, z50[overlap].values)

    def test_known_values(self):
        """Verify z-score against manual calculation."""
        data = [10, 12, 11, 13, 14, 12, 15, 13, 11, 14, 16]
        series = pd.Series(data, dtype=float)
        result = compute_inventory_zscore(series, window=5)

        # Manual: for last value (16), window is [14, 12, 15, 13, 11]... wait
        # Actually rolling window at index 10 includes indices 6-10: [15, 13, 11, 14, 16]
        # But z-score uses the rolling stats including the current value
        window_vals = series.iloc[6:11]  # [15, 13, 11, 14, 16]
        expected_mean = window_vals.mean()
        expected_std = window_vals.std()
        expected_z = (16 - expected_mean) / expected_std

        assert abs(result.iloc[-1] - expected_z) < 0.01


class TestComputeStocksToUseDays:
    """Test stocks-to-use ratio calculation."""

    def test_basic_calculation(self):
        """100 tonnes stock with 365 tonnes/year consumption = 100 days."""
        result = compute_stocks_to_use_days(100, 365)
        assert abs(result - 100.0) < 0.01

    def test_one_year_supply(self):
        """Stock equal to annual consumption = 365 days."""
        result = compute_stocks_to_use_days(1000, 1000)
        assert abs(result - 365.0) < 0.01

    def test_zero_consumption(self):
        """Zero consumption should return None."""
        assert compute_stocks_to_use_days(100, 0) is None

    def test_negative_consumption(self):
        """Negative consumption should return None."""
        assert compute_stocks_to_use_days(100, -50) is None

    def test_large_stocks(self):
        """Large stock should give proportionally large days."""
        result = compute_stocks_to_use_days(200000, 26000000)
        assert result is not None
        assert result > 0
        expected = 200000 / (26000000 / 365)
        assert abs(result - expected) < 0.01

    def test_zero_stock(self):
        """Zero stock should return 0 days."""
        result = compute_stocks_to_use_days(0, 1000)
        assert result == 0.0
