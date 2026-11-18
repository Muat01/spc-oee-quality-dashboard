"""
Export README figures (docs/) from the same Plotly charts used in the app.
Optional helper: needs `kaleido` (plotly 6+ also needs Chrome installed).
"""
import os

import pandas as pd

import charts
import oee
import spc

os.makedirs("docs", exist_ok=True)
ids, M = spc.to_subgroup_matrix(pd.read_csv("data/sample_spc.csv"))
cc = spc.xbar_r_chart(M, 30)
cap = spc.capability(M.ravel(), cc.sigma_within, 24.96, 25.04)
x_idx = list(range(1, len(cc.x) + 1))
prod, down = pd.read_csv("data/sample_production_log.csv"), pd.read_csv("data/sample_downtime_log.csv")
T = oee.shift_table(prod, down)

figs = {
    "control_chart.png": charts.control_chart(cc, x_idx, "Subgroup mean (X̄)", "Subgroup"),
    "capability_gauges.png": charts.capability_gauges(cap),
    "capability_histogram.png": charts.capability_histogram(M.ravel(), cap, 24.96, 25.04),
    "oee_gauge.png": charts.oee_gauge(oee.ratios(T)["oee"]),
    "oee_waterfall.png": charts.waterfall(oee.loss_waterfall(T)),
    "downtime_pareto.png": charts.pareto(oee.pareto(down)),
}
for name, fig in figs.items():
    fig.update_layout(paper_bgcolor="#171E1B")
    fig.write_image(f"docs/{name}", width=1200, scale=2)
print("docs figures written")
