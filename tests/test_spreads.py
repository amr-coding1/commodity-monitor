"""Tests for spread regime classification and signal alignment."""
from __future__ import annotations

import pytest

from src.processing.spreads import classify_signal_alignment, classify_spread_regime


class TestClassifySpreadRegime:
    """Test calendar spread regime classification."""

    def test_backwardation(self):
        """Positive spread (M1 > M2) = backwardation."""
        assert classify_spread_regime(5.0, 100.0) == "backwardation"

    def test_contango(self):
        """Negative spread (M1 < M2) = contango."""
        assert classify_spread_regime(-5.0, 100.0) == "contango"

    def test_flat_regime(self):
        """Very small spread relative to price = flat."""
        assert classify_spread_regime(0.1, 100.0) == "flat"
        assert classify_spread_regime(-0.1, 100.0) == "flat"

    def test_flat_threshold(self):
        """Spread exactly at threshold boundary."""
        # 0.2% of 100 = 0.2, so spread of 0.19 should be flat
        assert classify_spread_regime(0.19, 100.0, flat_threshold_pct=0.002) == "flat"
        # 0.21 should be backwardation
        assert classify_spread_regime(0.21, 100.0, flat_threshold_pct=0.002) == "backwardation"

    def test_none_spread(self):
        """None spread returns unknown."""
        assert classify_spread_regime(None, 100.0) == "unknown"

    def test_none_m1(self):
        """None M1 returns unknown."""
        assert classify_spread_regime(5.0, None) == "unknown"

    def test_zero_m1(self):
        """Zero M1 returns unknown (division by zero guard)."""
        assert classify_spread_regime(5.0, 0.0) == "unknown"

    def test_large_backwardation(self):
        """Large positive spread = backwardation."""
        assert classify_spread_regime(500.0, 8500.0) == "backwardation"

    def test_large_contango(self):
        """Large negative spread = contango."""
        assert classify_spread_regime(-200.0, 8500.0) == "contango"


class TestClassifySignalAlignment:
    """Test signal alignment classification."""

    def test_aligned_tight(self):
        """Low z-score + backwardation = aligned."""
        assert classify_signal_alignment(-1.5, "backwardation") == "aligned"

    def test_aligned_surplus(self):
        """High z-score + contango = aligned."""
        assert classify_signal_alignment(1.5, "contango") == "aligned"

    def test_divergent_low_stock_contango(self):
        """Low stocks but contango = divergent."""
        assert classify_signal_alignment(-1.5, "contango") == "divergent"

    def test_divergent_high_stock_backwardation(self):
        """High stocks but backwardation = divergent."""
        assert classify_signal_alignment(1.5, "backwardation") == "divergent"

    def test_neutral_normal_zscore(self):
        """Z-score between thresholds = neutral."""
        assert classify_signal_alignment(0.0, "backwardation") == "neutral"
        assert classify_signal_alignment(0.5, "contango") == "neutral"
        assert classify_signal_alignment(-0.5, "backwardation") == "neutral"

    def test_neutral_flat_spread(self):
        """Flat spread = neutral regardless of z-score."""
        assert classify_signal_alignment(-2.0, "flat") == "neutral"
        assert classify_signal_alignment(2.0, "flat") == "neutral"

    def test_neutral_unknown_regime(self):
        """Unknown regime = neutral."""
        assert classify_signal_alignment(-2.0, "unknown") == "neutral"

    def test_none_zscore(self):
        """None z-score = neutral."""
        assert classify_signal_alignment(None, "backwardation") == "neutral"

    def test_custom_thresholds(self):
        """Custom z-score thresholds work correctly."""
        # With tighter thresholds
        assert classify_signal_alignment(
            -0.5, "backwardation", zscore_tight=-0.5, zscore_surplus=0.5
        ) == "neutral"  # exactly at threshold, not below
        assert classify_signal_alignment(
            -0.6, "backwardation", zscore_tight=-0.5, zscore_surplus=0.5
        ) == "aligned"

    def test_boundary_zscore(self):
        """Z-score exactly at threshold is NOT tight/surplus (strict inequality)."""
        assert classify_signal_alignment(-1.0, "backwardation") == "neutral"
        assert classify_signal_alignment(1.0, "contango") == "neutral"
