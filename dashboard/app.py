"""Streamlit dashboard — Cross-Commodity Inventory & Term Structure Monitor."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.analysis.cross_commodity import (
    build_zscore_matrix,
    compute_cross_correlation_matrix,
    compute_regime_summary,
    compute_sensitivity_comparison,
)
from src.analysis.snapshot import get_market_snapshot
from src.analysis.storage_economics import StorageEconomicsAnalyser
from src.database import (
    get_analytics_series,
    get_connection,
    get_futures_series,
    get_inventory_series,
    init_db,
    list_commodities,
)

st.set_page_config(
    page_title="Commodity Monitor",
    page_icon="📊",
    layout="wide",
)

REGIME_COLOUR_MAP = {
    "tight": "#d32f2f",
    "surplus": "#2e7d32",
    "balanced": "#ff9800",
    "unknown": "#9e9e9e",
}
ALIGNMENT_COLOUR_MAP = {
    "aligned": "#2e7d32",
    "divergent": "#d32f2f",
    "neutral": "#9e9e9e",
}


@st.cache_resource(ttl=300)
def get_db():
    conn = get_connection()
    init_db(conn)
    return conn


def main():
    conn = get_db()
    st.title("Cross-Commodity Inventory & Term Structure Monitor")

    view = st.sidebar.radio(
        "View",
        ["Overview", "Individual Commodity", "Cross-Commodity", "Regime Analysis"],
    )

    if view == "Overview":
        render_overview(conn)
    elif view == "Individual Commodity":
        render_individual(conn)
    elif view == "Cross-Commodity":
        render_cross_commodity(conn)
    elif view == "Regime Analysis":
        render_regime_analysis(conn)


# ── Overview ─────────────────────────────────────────────────────────────
def render_overview(conn):
    st.header("Market Overview")

    snapshots = get_market_snapshot(conn)
    if not snapshots:
        st.warning("No data available. Run ingestion and analysis first.")
        return

    rows = []
    for s in snapshots:
        d = s.to_dict()
        rows.append({
            "Commodity": d["commodity"].title(),
            "Z-Score (1Y)": d["zscore_1y"],
            "Stocks-to-Use (days)": d["stocks_to_use"],
            "Spread (%)": round(d["spread_pct"] * 100, 2) if d["spread_pct"] is not None else None,
            "Regime": d["regime"],
            "Alignment": d["signal_alignment"],
            "Overall": d["overall_regime"],
        })

    df = pd.DataFrame(rows)

    def colour_regime(val):
        if val == "tight":
            return "background-color: #ffcdd2"
        elif val == "surplus":
            return "background-color: #c8e6c9"
        elif val == "divergent":
            return "background-color: #fff3e0"
        return ""

    styled = df.style.map(colour_regime, subset=["Overall", "Alignment"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    tight = sum(1 for s in snapshots if s.is_tight)
    surplus = sum(1 for s in snapshots if s.is_surplus)
    divergent = sum(1 for s in snapshots if s.is_divergent)
    col1.metric("Tight Markets", tight)
    col2.metric("Surplus Markets", surplus)
    col3.metric("Divergent Signals", divergent)

    # Z-score heatmap
    st.subheader("Inventory Tightness Over Time")
    matrix = build_zscore_matrix(conn)
    if not matrix.empty:
        weekly = matrix.resample("W").last()
        fig = px.imshow(
            weekly.T,
            color_continuous_scale="RdYlGn_r",
            zmin=-3,
            zmax=3,
            labels={"color": "Z-Score"},
            aspect="auto",
        )
        fig.update_yaxes(ticktext=[c.title() for c in weekly.columns], tickvals=list(range(len(weekly.columns))))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


# ── Individual Commodity ─────────────────────────────────────────────────
def render_individual(conn):
    commodities = list_commodities()
    commodity = st.sidebar.selectbox(
        "Select Commodity",
        commodities,
        format_func=lambda x: x.title(),
    )

    st.header(f"{commodity.title()} — Detailed Analysis")

    analytics = get_analytics_series(conn, commodity)
    if analytics.empty:
        st.warning(f"No analytics data for {commodity}. Run processing first.")
        return

    # Dual-axis: Z-score vs Spread
    st.subheader("Inventory Z-Score vs Calendar Spread")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=analytics.index,
        y=analytics["zscore_1y"].astype(float),
        name="Z-Score (1Y)",
        line=dict(color="#1976d2"),
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=analytics.index,
        y=analytics["spread_pct"].astype(float) * 100,
        name="Spread (%)",
        line=dict(color="#d32f2f"),
        yaxis="y2",
    ))
    fig.add_hline(y=-1.0, line_dash="dash", line_color="#d32f2f", opacity=0.4)
    fig.add_hline(y=1.0, line_dash="dash", line_color="#2e7d32", opacity=0.4)
    fig.update_layout(
        yaxis=dict(title="Z-Score", side="left"),
        yaxis2=dict(title="Spread (%)", side="right", overlaying="y"),
        height=450,
        legend=dict(x=0, y=1.1, orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Scatter: Z-score vs Spread by regime
    st.subheader("Stock Level vs Spread (by Regime)")
    scatter_df = analytics[["zscore_1y", "spread_pct", "regime"]].dropna()
    if not scatter_df.empty:
        scatter_df["spread_pct_100"] = scatter_df["spread_pct"].astype(float) * 100
        fig_scatter = px.scatter(
            scatter_df,
            x="zscore_1y",
            y="spread_pct_100",
            color="regime",
            color_discrete_map={
                "backwardation": "#d32f2f",
                "contango": "#1976d2",
                "flat": "#9e9e9e",
                "unknown": "#e0e0e0",
            },
            labels={"zscore_1y": "Inventory Z-Score", "spread_pct_100": "Spread (%)"},
            opacity=0.6,
        )
        fig_scatter.add_vline(x=-1.0, line_dash="dash", line_color="#d32f2f", opacity=0.4)
        fig_scatter.add_vline(x=1.0, line_dash="dash", line_color="#2e7d32", opacity=0.4)
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Rolling correlation
    st.subheader("Rolling Correlation (63-day)")
    analyser = StorageEconomicsAnalyser(conn)
    result = analyser.analyse_commodity(commodity)
    if result.rolling_correlation is not None:
        rc = result.rolling_correlation.dropna()
        if not rc.empty:
            fig_rc = go.Figure()
            fig_rc.add_trace(go.Scatter(
                x=rc.index, y=rc.values,
                fill="tozeroy",
                line=dict(color="#1976d2"),
            ))
            fig_rc.add_hline(y=0, line_color="black", opacity=0.3)
            fig_rc.update_layout(
                yaxis_title="Pearson r",
                height=300,
            )
            st.plotly_chart(fig_rc, use_container_width=True)

    # Summary stats
    st.subheader("Analysis Summary")
    col1, col2 = st.columns(2)
    with col1:
        if result.correlations:
            st.write("**Lagged Correlations (inventory → spread):**")
            corr_rows = [c.to_dict() for c in result.correlations]
            st.dataframe(pd.DataFrame(corr_rows), hide_index=True)
    with col2:
        if result.sensitivity:
            s = result.sensitivity.to_dict()
            st.write("**Sensitivity (OLS):**")
            st.write(f"β = {s['beta']:.4f} (spread % per z-score unit)")
            st.write(f"R² = {s['r_squared']:.3f}")
            st.write(f"p-value = {s['p_value']:.4f}")
        if result.walk_forward:
            wf = result.walk_forward.to_dict()
            st.write("**Walk-Forward Validation:**")
            st.write(f"In-sample r = {wf['in_sample_r']:.3f} (n={wf['in_sample_n']})")
            st.write(f"Out-of-sample r = {wf['out_of_sample_r']:.3f} (n={wf['out_of_sample_n']})")
            st.write(f"Stable: {'Yes' if wf['stable'] else 'No'}")


# ── Cross-Commodity ──────────────────────────────────────────────────────
def render_cross_commodity(conn):
    st.header("Cross-Commodity Analysis")

    # Z-score overlay
    st.subheader("Normalised Inventory Levels")
    matrix = build_zscore_matrix(conn)
    if not matrix.empty:
        fig = go.Figure()
        for commodity in matrix.columns:
            fig.add_trace(go.Scatter(
                x=matrix.index,
                y=matrix[commodity],
                name=commodity.title(),
                mode="lines",
            ))
        fig.add_hline(y=-1.0, line_dash="dash", line_color="#d32f2f", opacity=0.4, annotation_text="Tight")
        fig.add_hline(y=1.0, line_dash="dash", line_color="#2e7d32", opacity=0.4, annotation_text="Surplus")
        fig.update_layout(
            yaxis_title="Inventory Z-Score (1Y)",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Sensitivity comparison
    st.subheader("Inventory→Spread Sensitivity")
    sens_df = compute_sensitivity_comparison(conn)
    if not sens_df.empty and sens_df["beta"].notna().any():
        plot_df = sens_df.dropna(subset=["beta"])
        fig_bar = px.bar(
            plot_df,
            x=plot_df["commodity"].str.title(),
            y="beta",
            text=plot_df["r_squared"].apply(lambda x: f"R²={x:.2f}" if pd.notna(x) else ""),
            labels={"beta": "β (spread % per z-score unit)", "x": ""},
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Correlation matrix
    st.subheader("Z-Score Cross-Correlations")
    corr = compute_cross_correlation_matrix(conn)
    if not corr.empty:
        fig_hm = px.imshow(
            corr.astype(float),
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            labels={"color": "Correlation"},
        )
        fig_hm.update_xaxes(ticktext=[c.title() for c in corr.columns], tickvals=list(range(len(corr.columns))))
        fig_hm.update_yaxes(ticktext=[c.title() for c in corr.index], tickvals=list(range(len(corr.index))))
        fig_hm.update_layout(height=450)
        st.plotly_chart(fig_hm, use_container_width=True)


# ── Regime Analysis ──────────────────────────────────────────────────────
def render_regime_analysis(conn):
    st.header("Regime Analysis")

    # Current regimes
    st.subheader("Current Regime State")
    summary = compute_regime_summary(conn)
    if not summary.empty:
        display_df = summary.copy()
        display_df["commodity"] = display_df["commodity"].str.title()
        display_df["spread_pct"] = display_df["spread_pct"].apply(
            lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "N/A"
        )
        display_df["zscore_1y"] = display_df["zscore_1y"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )
        display_df["stocks_to_use"] = display_df["stocks_to_use"].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
        )

        def colour_row(row):
            regime = row.get("regime", "")
            if regime == "backwardation":
                return ["background-color: #ffcdd2"] * len(row)
            elif regime == "contango":
                return ["background-color: #bbdefb"] * len(row)
            return [""] * len(row)

        styled = display_df.style.apply(colour_row, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # Regime transitions over time
    st.subheader("Regime History")
    commodities = list_commodities()
    commodity = st.selectbox(
        "Select commodity for regime history",
        commodities,
        format_func=lambda x: x.title(),
    )

    analytics = get_analytics_series(conn, commodity)
    if not analytics.empty:
        # Regime timeline
        regime_map = {"backwardation": -1, "flat": 0, "contango": 1, "unknown": 0}
        analytics["regime_num"] = analytics["regime"].map(regime_map).fillna(0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=analytics.index,
            y=analytics["zscore_1y"].astype(float),
            name="Z-Score (1Y)",
            line=dict(color="#1976d2"),
        ))

        # Shade backwardation/contango periods
        for regime, colour in [("backwardation", "rgba(211,47,47,0.1)"), ("contango", "rgba(25,118,210,0.1)")]:
            mask = analytics["regime"] == regime
            if mask.any():
                changes = mask.astype(int).diff().fillna(0)
                starts = analytics.index[changes == 1].tolist()
                ends = analytics.index[changes == -1].tolist()
                if mask.iloc[0]:
                    starts.insert(0, analytics.index[0])
                if mask.iloc[-1]:
                    ends.append(analytics.index[-1])
                for s, e in zip(starts, ends):
                    fig.add_vrect(x0=s, x1=e, fillcolor=colour, line_width=0, layer="below")

        fig.add_hline(y=-1.0, line_dash="dash", line_color="#d32f2f", opacity=0.4)
        fig.add_hline(y=1.0, line_dash="dash", line_color="#2e7d32", opacity=0.4)
        fig.update_layout(
            yaxis_title="Z-Score",
            height=400,
            title=f"{commodity.title()} — Z-Score with Regime Shading",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Backtest: what happened to spreads when z-score crossed thresholds
        st.subheader("Threshold Crossings")
        z = analytics["zscore_1y"].astype(float)
        spread = analytics["spread_pct"].astype(float) * 100

        tight_crosses = z[(z.shift(1) >= -1.0) & (z < -1.0)].index
        surplus_crosses = z[(z.shift(1) <= 1.0) & (z > 1.0)].index

        events = []
        for dt in tight_crosses:
            loc = analytics.index.get_loc(dt)
            spread_at = spread.iloc[loc] if loc < len(spread) else None
            spread_20d = spread.iloc[loc + 20] if loc + 20 < len(spread) else None
            events.append({
                "Date": str(dt.date()) if hasattr(dt, "date") else str(dt),
                "Type": "Tight",
                "Z-Score": round(float(z.iloc[loc]), 2),
                "Spread (%)": round(float(spread_at), 2) if spread_at is not None else None,
                "Spread +20d (%)": round(float(spread_20d), 2) if spread_20d is not None else None,
            })
        for dt in surplus_crosses:
            loc = analytics.index.get_loc(dt)
            spread_at = spread.iloc[loc] if loc < len(spread) else None
            spread_20d = spread.iloc[loc + 20] if loc + 20 < len(spread) else None
            events.append({
                "Date": str(dt.date()) if hasattr(dt, "date") else str(dt),
                "Type": "Surplus",
                "Z-Score": round(float(z.iloc[loc]), 2),
                "Spread (%)": round(float(spread_at), 2) if spread_at is not None else None,
                "Spread +20d (%)": round(float(spread_20d), 2) if spread_20d is not None else None,
            })

        if events:
            st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)
        else:
            st.info("No threshold crossings detected in the available data.")


if __name__ == "__main__":
    main()
