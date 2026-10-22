"""
Generate synthetic sample data (reproducible, seed fixed).

SPC  : shaft diameter (mm), target 25.000, subgroups of 5. Injected events:
       one outlier, a sudden mean shift, tool-wear drift with a tool change,
       and a short period of higher variability.
OEE  : 30 days x 3 shifts of a machining cell with downtime reasons.

All data is SIMULATED. No real factory data is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OUT = "data"


def make_spc(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_sub, n = 120, 5
    mu = np.full(n_sub, 25.000)
    sigma = np.full(n_sub, 0.010)

    mu[45:70] += 0.012                      # sudden shift (e.g. new material lot)
    drift = np.linspace(0, 0.016, 20)       # tool wear
    mu[75:95] += drift                      # ... then a tool change at 95 resets it
    sigma[105:113] = 0.026                  # unstable fixture -> more variation

    rows = []
    for i in range(n_sub):
        vals = rng.normal(mu[i], sigma[i], n)
        if i == 30:
            vals[2] += 0.050                # a single outlier
        for j, v in enumerate(vals):
            rows.append({"subgroup": i + 1, "obs": j + 1, "measurement": round(float(v), 4)})
    return pd.DataFrame(rows)


def make_oee(seed: int = 11):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-03-02", periods=30, freq="D")
    shifts = ["Morning", "Afternoon", "Night"]
    shift_penalty = {"Morning": 0.0, "Afternoon": 0.03, "Night": 0.07}
    reasons = ["Breakdown", "Changeover", "Minor stops", "Material shortage", "Quality check"]
    weights = np.array([0.36, 0.26, 0.20, 0.10, 0.08])

    prod_rows, down_rows = [], []
    for d_idx, d in enumerate(dates):
        improvement = 0.04 * (d_idx / 29)          # gradual improvement over the month
        for s in shifts:
            planned, ideal = 450.0, 0.50
            n_events = rng.integers(2, 6)
            total_down = 0.0
            for _ in range(n_events):
                reason = rng.choice(reasons, p=weights)
                scale = {"Breakdown": 28, "Changeover": 22, "Minor stops": 7,
                         "Material shortage": 15, "Quality check": 9}[reason]
                minutes = float(round(rng.gamma(2.0, scale / 2.0) * (1 + shift_penalty[s] * 3 - improvement), 1))
                minutes = max(2.0, minutes)
                total_down += minutes
                down_rows.append({"date": d.date().isoformat(), "shift": s, "reason": reason, "minutes": minutes})
            run = max(60.0, planned - total_down)
            perf = np.clip(rng.normal(0.89 + improvement - shift_penalty[s], 0.02), 0.7, 0.99)
            total = int(run / ideal * perf)
            qual = np.clip(rng.normal(0.975 + improvement / 2 - shift_penalty[s] / 2, 0.006), 0.9, 0.995)
            good = int(round(total * qual))
            prod_rows.append({"date": d.date().isoformat(), "shift": s, "planned_min": planned,
                              "ideal_cycle_min": ideal, "total_count": total, "good_count": good})
    return pd.DataFrame(prod_rows), pd.DataFrame(down_rows)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    make_spc().to_csv(f"{OUT}/sample_spc.csv", index=False)
    prod, down = make_oee()
    prod.to_csv(f"{OUT}/sample_production_log.csv", index=False)
    down.to_csv(f"{OUT}/sample_downtime_log.csv", index=False)
    print("Sample data written to ./data/")
