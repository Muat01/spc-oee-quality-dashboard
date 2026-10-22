"""
Overall Equipment Effectiveness (OEE) calculations.

OEE = Availability x Performance x Quality
  Availability = run time / planned production time
  Performance  = (ideal cycle time x total count) / run time
  Quality      = good count / total count

Aggregates are time-weighted (sums first, then ratios), the correct way to
combine shifts or days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROD_COLS = ["date", "shift", "planned_min", "ideal_cycle_min", "total_count", "good_count"]
DOWN_COLS = ["date", "shift", "reason", "minutes"]
BENCHMARK = 0.85  # commonly quoted "world-class" OEE


def validate(prod: pd.DataFrame, down: pd.DataFrame) -> None:
    for name, df, cols in (("production log", prod, PROD_COLS), ("downtime log", down, DOWN_COLS)):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"The {name} is missing columns: {', '.join(missing)}")


def shift_table(prod: pd.DataFrame, down: pd.DataFrame) -> pd.DataFrame:
    """One row per date/shift with run time and the base quantities."""
    validate(prod, down)
    d = down.assign(minutes=pd.to_numeric(down["minutes"], errors="coerce").fillna(0))
    dt = d.groupby(["date", "shift"], as_index=False)["minutes"].sum().rename(columns={"minutes": "downtime_min"})
    t = prod.merge(dt, on=["date", "shift"], how="left")
    t["downtime_min"] = t["downtime_min"].fillna(0.0)
    t["run_min"] = (t["planned_min"] - t["downtime_min"]).clip(lower=0)
    t["ideal_min"] = t["ideal_cycle_min"] * t["total_count"]
    t["good_ideal_min"] = t["ideal_cycle_min"] * t["good_count"]
    return t


def ratios(t: pd.DataFrame) -> dict:
    """Time-weighted availability, performance, quality and OEE."""
    planned, run = t["planned_min"].sum(), t["run_min"].sum()
    ideal, total, good = t["ideal_min"].sum(), t["total_count"].sum(), t["good_count"].sum()
    a = run / planned if planned else 0.0
    p = min(1.0, ideal / run) if run else 0.0
    q = good / total if total else 0.0
    return {"availability": a, "performance": p, "quality": q, "oee": a * p * q}


def daily_oee(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, g in t.groupby("date"):
        r = ratios(g)
        rows.append({"date": date, **r})
    return pd.DataFrame(rows)


def by_shift(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for shift, g in t.groupby("shift"):
        rows.append({"shift": shift, **ratios(g)})
    return pd.DataFrame(rows)


def loss_waterfall(t: pd.DataFrame) -> dict:
    """Minutes lost to availability, performance (speed) and quality."""
    planned = t["planned_min"].sum()
    downtime = t["downtime_min"].sum()
    speed = max(0.0, t["run_min"].sum() - t["ideal_min"].sum())
    quality = t["ideal_min"].sum() - t["good_ideal_min"].sum()
    productive = t["good_ideal_min"].sum()
    return {"planned": planned, "downtime": downtime, "speed": speed,
            "quality": quality, "productive": productive}


def pareto(down: pd.DataFrame) -> pd.DataFrame:
    d = down.assign(minutes=pd.to_numeric(down["minutes"], errors="coerce").fillna(0))
    p = d.groupby("reason", as_index=False)["minutes"].sum().sort_values("minutes", ascending=False)
    p["pct"] = p["minutes"] / p["minutes"].sum() * 100
    p["cum_pct"] = p["pct"].cumsum()
    return p.reset_index(drop=True)
