"""Integration tests — full pipeline from seeded DB through analysis and reporting."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.analysis.cross_commodity import (
    build_zscore_matrix,
    compute_cross_correlation_matrix,
    compute_regime_summary,
    compute_sensitivity_comparison,
)
from src.analysis.snapshot import CommoditySnapshot, get_market_snapshot
from src.analysis.storage_economics import StorageEconomicsAnalyser
from src.database import (
    get_analytics_series,
    get_futures_series,
    get_inventory_series,
    get_latest_analytics,
)
from src.processing.normaliser import process_inventory_analytics
from src.processing.spreads import process_spread_analytics
from src.reporting.commentary import generate_commentary


# ── Processing pipeline ──────────────────────────────────────────────────


class TestProcessingPipeline:
    """Test the full processing pipeline on seeded data."""

    def test_inventory_analytics_writes_zscores(self, seeded_db):
        """process_inventory_analytics should populate zscore_1y in daily_analytics."""
        n = process_inventory_analytics(seeded_db, "copper")
        assert n > 0

        df = get_analytics_series(seeded_db, "copper")
        assert not df.empty
        assert "zscore_1y" in df.columns
        # After warmup period, z-scores should be non-null
        valid = df["zscore_1y"].dropna()
        assert len(valid) > 0

    def test_inventory_analytics_all_commodities(self, seeded_db):
        """All 5 commodities should produce analytics rows."""
        for commodity in ["copper", "aluminium", "coffee", "cocoa", "sugar"]:
            n = process_inventory_analytics(seeded_db, commodity)
            assert n > 0, f"No analytics rows for {commodity}"

    def test_spread_analytics_writes_regime(self, seeded_db):
        """process_spread_analytics should populate regime and spread_pct."""
        process_inventory_analytics(seeded_db, "copper")
        n = process_spread_analytics(seeded_db, "copper")
        assert n > 0

        df = get_analytics_series(seeded_db, "copper")
        assert "regime" in df.columns
        assert "spread_pct" in df.columns
        # Should have actual regime values (not all unknown)
        regimes = df["regime"].dropna().unique()
        assert len(regimes) > 0
        assert any(r in regimes for r in ["backwardation", "contango", "flat"])

    def test_spread_analytics_signal_alignment(self, seeded_db):
        """Signal alignment should be computed when z-scores exist."""
        process_inventory_analytics(seeded_db, "copper")
        process_spread_analytics(seeded_db, "copper")

        df = get_analytics_series(seeded_db, "copper")
        alignments = df["signal_alignment"].dropna().unique()
        assert len(alignments) > 0
        for a in alignments:
            assert a in ("aligned", "divergent", "neutral")

    def test_stocks_to_use_computed_for_metals(self, seeded_db):
        """Stocks-to-use should be computed when consumption data exists."""
        n = process_inventory_analytics(seeded_db, "copper")
        assert n > 0

        df = get_analytics_series(seeded_db, "copper")
        stu = df["stocks_to_use"].dropna()
        assert len(stu) > 0
        # Should be positive and reasonable (days of supply)
        assert (stu > 0).all()
        assert (stu < 365).all()  # less than a year of supply

    def test_full_pipeline_all_commodities(self, seeded_db):
        """Run processing for all commodities end to end."""
        for commodity in ["copper", "aluminium", "coffee", "cocoa", "sugar"]:
            process_inventory_analytics(seeded_db, commodity)
            process_spread_analytics(seeded_db, commodity)

        # Every commodity should now have analytics
        for commodity in ["copper", "aluminium", "coffee", "cocoa", "sugar"]:
            df = get_analytics_series(seeded_db, commodity)
            assert not df.empty, f"No analytics for {commodity}"
            assert df["regime"].notna().any(), f"No regime for {commodity}"


# ── Analysis layer ───────────────────────────────────────────────────────


class TestStorageEconomicsIntegration:
    """Test the StorageEconomicsAnalyser on seeded + processed data."""

    @pytest.fixture(autouse=True)
    def _process_all(self, seeded_db):
        """Run processing pipeline before each test in this class."""
        self.conn = seeded_db
        for commodity in ["copper", "aluminium", "coffee", "cocoa", "sugar"]:
            process_inventory_analytics(seeded_db, commodity)
            process_spread_analytics(seeded_db, commodity)

    def test_analyse_commodity_returns_correlations(self):
        """Analyser should produce correlation results."""
        analyser = StorageEconomicsAnalyser(self.conn)
        result = analyser.analyse_commodity("copper")
        assert len(result.correlations) > 0
        for corr in result.correlations:
            assert -1.0 <= corr.r <= 1.0
            assert corr.n > 0
            assert corr.ci_lower <= corr.r <= corr.ci_upper

    def test_analyse_commodity_returns_sensitivity(self):
        """Analyser should produce a sensitivity (OLS) result."""
        analyser = StorageEconomicsAnalyser(self.conn)
        result = analyser.analyse_commodity("copper")
        assert result.sensitivity is not None
        assert 0 <= result.sensitivity.r_squared <= 1

    def test_analyse_commodity_returns_walk_forward(self):
        """Walk-forward validation should run on seeded data."""
        analyser = StorageEconomicsAnalyser(self.conn)
        result = analyser.analyse_commodity("copper")
        assert result.walk_forward is not None
        assert result.walk_forward.in_sample_n > 0
        assert result.walk_forward.out_of_sample_n > 0

    def test_analyse_commodity_returns_rolling_correlation(self):
        """Rolling correlation series should be computed."""
        analyser = StorageEconomicsAnalyser(self.conn)
        result = analyser.analyse_commodity("copper")
        assert result.rolling_correlation is not None
        valid = result.rolling_correlation.dropna()
        assert len(valid) > 0
        assert (valid >= -1.0).all()
        assert (valid <= 1.0).all()

    def test_run_all_covers_every_commodity(self):
        """run_all should return results for all 5 commodities."""
        analyser = StorageEconomicsAnalyser(self.conn)
        results = analyser.run_all()
        assert len(results) == 5
        for commodity in ["copper", "aluminium", "coffee", "cocoa", "sugar"]:
            assert commodity in results
            assert len(results[commodity].correlations) > 0

    def test_to_dict_serialisation(self):
        """CommodityAnalysis.to_dict() should produce clean JSON-serialisable output."""
        analyser = StorageEconomicsAnalyser(self.conn)
        result = analyser.analyse_commodity("copper")
        d = result.to_dict()
        assert d["commodity"] == "copper"
        assert isinstance(d["correlations"], list)
        assert d["sensitivity"] is not None
        # Verify no NaN/inf in output
        for corr in d["correlations"]:
            for val in corr.values():
                if isinstance(val, float):
                    assert not math.isnan(val)
                    assert not math.isinf(val)


# ── Cross-commodity analysis ─────────────────────────────────────────────


class TestCrossCommodityIntegration:
    """Test cross-commodity analysis on seeded + processed data."""

    @pytest.fixture(autouse=True)
    def _process_all(self, seeded_db):
        self.conn = seeded_db
        for commodity in ["copper", "aluminium", "coffee", "cocoa", "sugar"]:
            process_inventory_analytics(seeded_db, commodity)
            process_spread_analytics(seeded_db, commodity)

    def test_zscore_matrix_shape(self):
        """Z-score matrix should have date rows and commodity columns."""
        matrix = build_zscore_matrix(self.conn)
        assert not matrix.empty
        assert len(matrix.columns) == 5
        assert set(matrix.columns) == {"copper", "aluminium", "coffee", "cocoa", "sugar"}

    def test_cross_correlation_matrix(self):
        """Cross-correlation matrix should be square and symmetric."""
        corr = compute_cross_correlation_matrix(self.conn)
        assert corr.shape == (5, 5)
        # Diagonal should be 1.0
        for i in range(5):
            assert corr.iloc[i, i] == 1.0
        # Should be symmetric
        for i in range(5):
            for j in range(i + 1, 5):
                assert abs(corr.iloc[i, j] - corr.iloc[j, i]) < 1e-10

    def test_sensitivity_comparison(self):
        """Sensitivity comparison should return a row per commodity."""
        df = compute_sensitivity_comparison(self.conn)
        assert len(df) == 5
        assert "beta" in df.columns
        assert "r_squared" in df.columns

    def test_regime_summary(self):
        """Regime summary should return current state per commodity."""
        df = compute_regime_summary(self.conn)
        assert len(df) == 5
        assert "regime" in df.columns
        assert "signal_alignment" in df.columns


# ── Snapshot ─────────────────────────────────────────────────────────────


class TestSnapshotIntegration:
    """Test market snapshot on seeded + processed data."""

    @pytest.fixture(autouse=True)
    def _process_all(self, seeded_db):
        self.conn = seeded_db
        for commodity in ["copper", "aluminium", "coffee", "cocoa", "sugar"]:
            process_inventory_analytics(seeded_db, commodity)
            process_spread_analytics(seeded_db, commodity)

    def test_snapshot_returns_all_commodities(self):
        """Snapshot should have an entry for every commodity."""
        snapshots = get_market_snapshot(self.conn)
        assert len(snapshots) == 5
        names = {s.commodity for s in snapshots}
        assert names == {"copper", "aluminium", "coffee", "cocoa", "sugar"}

    def test_snapshot_has_regime(self):
        """Each snapshot should have a regime classification."""
        snapshots = get_market_snapshot(self.conn)
        for s in snapshots:
            assert s.overall_regime in ("tight", "surplus", "balanced", "unknown")

    def test_snapshot_to_dict_no_nan(self):
        """Snapshot to_dict should not contain NaN values."""
        snapshots = get_market_snapshot(self.conn)
        for s in snapshots:
            d = s.to_dict()
            for key, val in d.items():
                if isinstance(val, float):
                    assert not math.isnan(val), f"NaN in {s.commodity}.{key}"

    def test_snapshot_zero_zscore_preserved(self):
        """A z-score of exactly 0.0 should not be suppressed to None."""
        snapshot = CommoditySnapshot(commodity="test", zscore_1y=0.0)
        d = snapshot.to_dict()
        assert d["zscore_1y"] == 0.0, "Zero z-score was suppressed to None"

    def test_snapshot_zero_spread_preserved(self):
        """A spread_pct of exactly 0.0 should not be suppressed to None."""
        snapshot = CommoditySnapshot(commodity="test", spread_pct=0.0)
        d = snapshot.to_dict()
        assert d["spread_pct"] == 0.0, "Zero spread was suppressed to None"


# ── Commentary ───────────────────────────────────────────────────────────


class TestCommentaryIntegration:
    """Test the commentary generation on seeded + processed data."""

    @pytest.fixture(autouse=True)
    def _process_all(self, seeded_db):
        self.conn = seeded_db
        for commodity in ["copper", "aluminium", "coffee", "cocoa", "sugar"]:
            process_inventory_analytics(seeded_db, commodity)
            process_spread_analytics(seeded_db, commodity)

    def test_commentary_renders(self):
        """Commentary should render without errors."""
        md = generate_commentary(self.conn)
        assert isinstance(md, str)
        assert len(md) > 100

    def test_commentary_contains_all_commodities(self):
        """Commentary should mention every commodity."""
        md = generate_commentary(self.conn)
        for commodity in ["Copper", "Aluminium", "Coffee", "Cocoa", "Sugar"]:
            assert commodity in md, f"{commodity} missing from commentary"

    def test_commentary_no_nan_strings(self):
        """Commentary should not contain literal 'nan' values."""
        md = generate_commentary(self.conn)
        # Split into lines and check each (excluding the word "analysis" etc.)
        for line in md.split("\n"):
            # Only flag standalone 'nan', not words containing 'nan'
            cells = line.split("|")
            for cell in cells:
                stripped = cell.strip()
                assert stripped != "nan", f"Found 'nan' in commentary: {line}"

    def test_commentary_has_regime_info(self):
        """Commentary should include regime classifications."""
        md = generate_commentary(self.conn)
        assert any(
            word in md.lower()
            for word in ["backwardation", "contango", "balanced", "tight", "surplus"]
        )

    def test_commentary_has_sensitivity_table(self):
        """Commentary should include the sensitivity comparison table."""
        md = generate_commentary(self.conn)
        assert "Sensitivity" in md
        assert "β" in md or "beta" in md.lower()


# ── Database query helpers ───────────────────────────────────────────────


class TestDatabaseQueryHelpers:
    """Test the database query helper functions on seeded data."""

    def test_get_inventory_series(self, seeded_db):
        """get_inventory_series should return a proper DataFrame."""
        df = get_inventory_series(seeded_db, "copper")
        assert not df.empty
        assert "stock_level" in df.columns
        assert df.index.name == "date"

    def test_get_inventory_series_date_filter(self, seeded_db):
        """Date filtering should narrow the result set."""
        full = get_inventory_series(seeded_db, "copper")
        filtered = get_inventory_series(
            seeded_db, "copper",
            start_date="2024-03-01", end_date="2024-04-01",
        )
        assert len(filtered) < len(full)
        assert len(filtered) > 0

    def test_get_futures_series(self, seeded_db):
        """get_futures_series should return M1, M2, and spread."""
        df = get_futures_series(seeded_db, "copper")
        assert not df.empty
        assert "m1_close" in df.columns
        assert "m2_close" in df.columns
        assert "spread" in df.columns

    def test_get_latest_analytics_single(self, seeded_db):
        """get_latest_analytics for one commodity should return one row."""
        process_inventory_analytics(seeded_db, "copper")
        df = get_latest_analytics(seeded_db, "copper")
        assert len(df) == 1
        assert df.iloc[0]["commodity"] == "copper"

    def test_get_latest_analytics_all(self, seeded_db):
        """get_latest_analytics without commodity should return all."""
        for c in ["copper", "aluminium", "coffee", "cocoa", "sugar"]:
            process_inventory_analytics(seeded_db, c)
        df = get_latest_analytics(seeded_db)
        assert len(df) == 5
