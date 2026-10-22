"""
SPC & OEE Quality Dashboard

Run:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import oee
import spc

DATA = Path(__file__).parent / "data"

st.set_page_config(page_title="SPC & OEE Dashboard", page_icon="📊", layout="wide")

TEAL, RED, ORANGE, BLUE, MUTED = "#00d4aa", "#ff4757", "#ffa502", "#3d8eff", "#8b92a8"
BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c9cfdd", size=12), margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)"), yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
)


def layout(**over):
    """Merge overrides into the base layout (axis dicts are merged, not replaced)."""
    out = dict(BASE_LAYOUT)
    for k, v in over.items():
        out[k] = {**BASE_LAYOUT[k], **v} if k in ("xaxis", "yaxis") and isinstance(v, dict) else v
    return out


# ─── Data loading ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def sample_spc() -> pd.DataFrame:
    return pd.read_csv(DATA / "sample_spc.csv")


@st.cache_data(show_spinner=False)
def sample_oee():
    return pd.read_csv(DATA / "sample_production_log.csv"), pd.read_csv(DATA / "sample_downtime_log.csv")


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Data")
    source = st.radio("Source", ["Sample data (simulated)", "Upload my CSV files"], label_visibility="collapsed")
    spc_df, prod_df, down_df = sample_spc(), *sample_oee()
    if source.startswith("Upload"):
        up_spc = st.file_uploader("SPC data: subgroup, measurement", type=["csv"], key="u1")
        up_prod = st.file_uploader("OEE production log", type=["csv"], key="u2")
        up_down = st.file_uploader("OEE downtime log", type=["csv"], key="u3")
        with st.expander("Required columns"):
            st.code("SPC: subgroup, measurement\n"
                    "Production: date, shift, planned_min,\n  ideal_cycle_min, total_count, good_count\n"
                    "Downtime: date, shift, reason, minutes", language="text")
        if up_spc is not None:
            spc_df = pd.read_csv(up_spc)
        if up_prod is not None and up_down is not None:
            prod_df, down_df = pd.read_csv(up_prod), pd.read_csv(up_down)
        st.caption("Any file left empty keeps the sample data for that tab.")
    st.caption("Sample data is simulated, not from a real factory.")

st.title("📊 SPC & OEE Quality Dashboard")
st.caption("Statistical process control, process capability and equipment effectiveness for manufacturing.")

tab_spc, tab_cap, tab_oee, tab_method = st.tabs(
    ["📈 Control charts", "🎯 Capability", "⚙️ OEE", "ℹ️ Method"])

# ─── Prepare SPC data ───────────────────────────────────────────────────────
spc_error = None
try:
    ids, matrix = spc.to_subgroup_matrix(spc_df)
except Exception as exc:  # noqa: BLE001
    spc_error, ids, matrix = str(exc), [], np.empty((0, 0))

# ═══════════════════════════════ CONTROL CHARTS ═════════════════════════════
with tab_spc:
    if spc_error:
        st.error(spc_error)
    else:
        k, n = matrix.shape
        c1, c2, c3 = st.columns([1.2, 1.2, 2])
        chart_kind = c1.selectbox("Chart type", ["X̄-R (subgroups)", "I-MR (individuals)"])
        baseline = c2.slider("Baseline subgroups (Phase I)", 10, k, min(30, k), key="baseline_sub",
                             help="Control limits are computed from these first subgroups, "
                                  "then used to monitor the rest.")
        rules = c3.multiselect("Run rules", list(spc.RULE_LABELS), default=list(spc.RULE_LABELS),
                               format_func=lambda r: f"{r}: {spc.RULE_LABELS[r]}")

        if chart_kind.startswith("X̄"):
            cc = spc.xbar_r_chart(matrix, baseline, rules)
            x_label, s_label, x_idx = "Subgroup mean (X̄)", "Range (R)", list(range(1, k + 1))
        else:
            flat = matrix.ravel()
            cc = spc.imr_chart(flat, baseline * n, rules)
            x_label, s_label, x_idx = "Individual value", "Moving range", list(range(1, len(flat) + 1))

        flagged = sorted(cc.signals)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Center line", f"{cc.cl:.4f}")
        m2.metric("UCL / LCL", f"{cc.ucl:.4f} / {cc.lcl:.4f}")
        m3.metric("Out-of-control points", f"{len(flagged)}")
        m4.metric("Range signals", f"{len(cc.spread_signals)}")

        # --- main chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_idx, y=cc.x, mode="lines+markers", name=x_label,
                                 line=dict(color=BLUE, width=1.6), marker=dict(size=5, color=BLUE)))
        for mult, dash, col in ((1, "dot", "#2a3550"), (2, "dot", "#3a4a70")):
            for sgn in (1, -1):
                fig.add_hline(y=cc.cl + sgn * mult * cc.sigma_stat, line=dict(color=col, width=1, dash=dash))
        fig.add_hline(y=cc.cl, line=dict(color=TEAL, width=2), annotation_text="CL", annotation_position="right")
        fig.add_hline(y=cc.ucl, line=dict(color=RED, width=2, dash="dash"), annotation_text="UCL", annotation_position="right")
        fig.add_hline(y=cc.lcl, line=dict(color=RED, width=2, dash="dash"), annotation_text="LCL", annotation_position="right")
        if flagged:
            fig.add_trace(go.Scatter(
                x=[x_idx[i] for i in flagged], y=[cc.x[i] for i in flagged], mode="markers",
                name="Signal", marker=dict(size=10, color=RED, line=dict(color="white", width=1)),
                text=[", ".join(cc.signals[i]) for i in flagged],
                hovertemplate="#%{x}<br>%{y:.4f}<br>Rules: %{text}<extra></extra>"))
        fig.add_vrect(x0=0.5, x1=cc.baseline + 0.5, fillcolor=TEAL, opacity=0.06, line_width=0,
                      annotation_text="Baseline (Phase I)", annotation_position="top left")
        fig.update_layout(**layout(height=380, title=f"{cc.kind} chart: {x_label}", showlegend=False,
                                   xaxis=dict(title="Subgroup" if chart_kind.startswith("X̄") else "Observation")))
        st.plotly_chart(fig, width="stretch")

        # --- spread chart
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x_idx, y=cc.spread, mode="lines+markers", name=s_label,
                                  line=dict(color=ORANGE, width=1.6), marker=dict(size=5, color=ORANGE)))
        fig2.add_hline(y=cc.cl_spread, line=dict(color=TEAL, width=2))
        fig2.add_hline(y=cc.ucl_spread, line=dict(color=RED, width=2, dash="dash"))
        if cc.lcl_spread > 0:
            fig2.add_hline(y=cc.lcl_spread, line=dict(color=RED, width=2, dash="dash"))
        if cc.spread_signals:
            fig2.add_trace(go.Scatter(x=[x_idx[i] for i in cc.spread_signals],
                                      y=[cc.spread[i] for i in cc.spread_signals], mode="markers",
                                      marker=dict(size=10, color=RED, line=dict(color="white", width=1))))
        fig2.update_layout(**layout(height=260, title=f"{s_label} chart", showlegend=False))
        st.plotly_chart(fig2, width="stretch")

        if flagged:
            st.markdown("**Signals detected**")
            rows = [{"Point": x_idx[i], "Value": round(float(cc.x[i]), 4),
                     "Rules": ", ".join(cc.signals[i])} for i in flagged]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=220)
        else:
            st.success("No signals: the process looks statistically stable.")
        st.caption(f"Limits are computed from the first {cc.baseline} "
                   f"{'subgroups' if chart_kind.startswith('X̄') else 'observations'} and applied to all data.")

# ═══════════════════════════════ CAPABILITY ═════════════════════════════════
with tab_cap:
    if spc_error:
        st.error(spc_error)
    else:
        flat = matrix.ravel()
        cc_cap = spc.xbar_r_chart(matrix, st.session_state.get("baseline_sub", min(30, matrix.shape[0])))
        if source.startswith("Sample"):
            default_lsl, default_usl = 24.960, 25.040          # spec of the simulated shaft: 25.000 ± 0.040
        else:
            half = 4 * float(flat.std(ddof=1))                  # placeholder: replace with your real limits
            default_lsl, default_usl = float(round(flat.mean() - half, 3)), float(round(flat.mean() + half, 3))
        a, b, _ = st.columns([1, 1, 2])
        lsl = a.number_input("LSL (lower spec limit)", value=default_lsl, format="%.4f", step=0.001)
        usl = b.number_input("USL (upper spec limit)", value=default_usl, format="%.4f", step=0.001)
        if usl <= lsl:
            st.error("USL must be greater than LSL.")
        else:
            cap = spc.capability(flat, cc_cap.sigma_within, lsl, usl)
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Cp", f"{cap.cp:.2f}", help="Potential capability, short-term sigma, ignores centering")
            k2.metric("Cpk", f"{cap.cpk:.2f}", help="Capability including centering, short-term sigma")
            k3.metric("Pp", f"{cap.pp:.2f}", help="Overall performance, long-term sigma")
            k4.metric("Ppk", f"{cap.ppk:.2f}", help="Overall performance including centering")
            k5.metric("Expected PPM out of spec", f"{cap.ppm_overall:,.0f}")

            st.info(f"**Short-term (Cpk):** {spc.verdict(cap.cpk)}    |    "
                    f"**Long-term (Ppk):** {spc.verdict(cap.ppk)}")
            if cap.cp - cap.cpk > 0.15:
                st.warning(f"Cp − Cpk = {cap.cp - cap.cpk:.2f}: the process is off-center. "
                           "Re-centering the mean would recover capability.")
            if cap.cpk - cap.ppk > 0.15:
                st.warning("Long-term performance (Ppk) is clearly worse than short-term capability (Cpk): "
                           "shifts or drift between subgroups are inflating the overall variation.")
            if cc_cap.signals or cc_cap.spread_signals:
                st.caption("⚠️ The process is not in statistical control "
                           f"({len(cc_cap.signals)} signal points), so capability indices are indicative only. "
                           "Short-term sigma is estimated from the baseline subgroups.")

            xs = np.linspace(min(flat.min(), lsl) - cap.sigma_overall, max(flat.max(), usl) + cap.sigma_overall, 300)
            pdf = np.exp(-0.5 * ((xs - cap.mean) / cap.sigma_overall) ** 2) / (cap.sigma_overall * np.sqrt(2 * np.pi))
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=flat, histnorm="probability density", nbinsx=40,
                                       marker_color=BLUE, opacity=0.75, name="Measurements"))
            fig.add_trace(go.Scatter(x=xs, y=pdf, mode="lines", line=dict(color=TEAL, width=3), name="Normal fit"))
            for val, name in ((lsl, "LSL"), (usl, "USL")):
                fig.add_vline(x=val, line=dict(color=RED, width=2, dash="dash"), annotation_text=name)
            fig.add_vline(x=cap.mean, line=dict(color=TEAL, width=1, dash="dot"), annotation_text="mean")
            fig.update_layout(**layout(height=380, title="Process capability histogram",
                                       xaxis=dict(title="Measurement"), yaxis=dict(title="Density")))
            st.plotly_chart(fig, width="stretch")

# ═══════════════════════════════ OEE ════════════════════════════════════════
with tab_oee:
    try:
        table = oee.shift_table(prod_df, down_df)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        table = None
    if table is not None:
        shifts = sorted(table["shift"].unique())
        sel = st.multiselect("Shifts", shifts, default=shifts)
        t = table[table["shift"].isin(sel)]
        d = down_df[down_df["shift"].isin(sel)]
        if t.empty:
            st.warning("Select at least one shift.")
        else:
            r = oee.ratios(t)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("OEE", f"{r['oee'] * 100:.1f}%", f"{(r['oee'] - oee.BENCHMARK) * 100:+.1f} pts vs 85% benchmark")
            c2.metric("Availability", f"{r['availability'] * 100:.1f}%")
            c3.metric("Performance", f"{r['performance'] * 100:.1f}%")
            c4.metric("Quality", f"{r['quality'] * 100:.1f}%")

            w = oee.loss_waterfall(t)
            fig = go.Figure(go.Waterfall(
                x=["Planned time", "Downtime", "Speed loss", "Quality loss", "Fully productive"],
                y=[w["planned"], -w["downtime"], -w["speed"], -w["quality"], 0],
                measure=["absolute", "relative", "relative", "relative", "total"],
                text=[f"{v / 60:,.0f} h" for v in (w["planned"], w["downtime"], w["speed"], w["quality"], w["productive"])],
                textposition="outside",
                decreasing=dict(marker=dict(color=RED)), increasing=dict(marker=dict(color=TEAL)),
                totals=dict(marker=dict(color=TEAL)), connector=dict(line=dict(color=MUTED))))
            fig.update_layout(**layout(height=380, title="Where the planned time goes (OEE loss waterfall)",
                                       yaxis=dict(title="Minutes"), showlegend=False))
            st.plotly_chart(fig, width="stretch")

            left, right = st.columns(2)
            with left:
                p = oee.pareto(d)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=p["reason"], y=p["minutes"], marker_color=ORANGE, name="Minutes"))
                fig.add_trace(go.Scatter(x=p["reason"], y=p["cum_pct"], yaxis="y2", mode="lines+markers",
                                         line=dict(color=TEAL, width=3), name="Cumulative %"))
                fig.add_hline(y=80, yref="y2", line=dict(color=MUTED, dash="dot"))
                fig.update_layout(**layout(height=360, title="Downtime Pareto", showlegend=False,
                                           yaxis=dict(title="Minutes"),
                                           yaxis2=dict(overlaying="y", side="right", range=[0, 105],
                                                       title="Cumulative %", showgrid=False)))
                st.plotly_chart(fig, width="stretch")
            with right:
                dd = oee.daily_oee(t)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dd["date"], y=dd["oee"] * 100, mode="lines+markers",
                                         line=dict(color=BLUE, width=2), name="Daily OEE"))
                fig.add_hline(y=oee.BENCHMARK * 100, line=dict(color=TEAL, dash="dash"),
                              annotation_text="85% benchmark")
                fig.update_layout(**layout(height=360, title="Daily OEE (%)", showlegend=False,
                                           yaxis=dict(range=[40, 100], title="%")))
                st.plotly_chart(fig, width="stretch")

            st.markdown("**OEE by shift**")
            bs = oee.by_shift(t).copy()
            for col in ("availability", "performance", "quality", "oee"):
                bs[col] = (bs[col] * 100).round(1)
            st.dataframe(bs.rename(columns={"shift": "Shift", "availability": "Availability %",
                                            "performance": "Performance %", "quality": "Quality %",
                                            "oee": "OEE %"}), hide_index=True, width="stretch")

            top = oee.pareto(d)
            if len(top):
                n80 = int((top["cum_pct"] < 80).sum()) + 1
                st.info(f"**{', '.join(top['reason'].head(n80))}** account for "
                        f"{top['cum_pct'].iloc[n80 - 1]:.0f}% of downtime: start improvement there.")

# ═══════════════════════════════ METHOD ═════════════════════════════════════
with tab_method:
    st.markdown("""
**Control charts.** Limits are computed from a baseline window (Phase I) and then used to monitor the
remaining data (Phase II). For X̄-R charts: UCL/LCL = X̿ ± A₂·R̄ and the R chart uses D₃·R̄ / D₄·R̄.
The within-subgroup sigma is **σ = R̄ / d₂**. For I-MR charts: limits = X̄ ± 3·(MR̄ / 1.128).

**Run rules (Western Electric / Nelson).**
`R1` one point beyond 3σ · `R2` nine in a row on one side · `R3` six in a row rising or falling ·
`R5` two of three beyond 2σ on the same side.

**Capability.**
`Cp = (USL − LSL) / 6σ_within` · `Cpk = min(USL − μ, μ − LSL) / 3σ_within`.
`Pp` and `Ppk` use the overall standard deviation. A large gap between Cpk and Ppk means shifts or drift
between subgroups. Capability indices are only meaningful when the process is stable.
Common reading: Cpk ≥ 1.33 capable, 1.00 to 1.33 marginal, < 1.00 not capable.

**OEE.** `OEE = Availability × Performance × Quality`, with
Availability = run time / planned time, Performance = (ideal cycle time × total count) / run time,
Quality = good count / total count. Aggregates are time-weighted (sum first, then divide).
The 85% benchmark is a commonly quoted reference, not a universal target.

**Data.** The sample data is simulated and reproducible (`data_gen.py`). Replace it with your own CSV files.
""")
