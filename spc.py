"""
Statistical Process Control (SPC) calculations.

* X-bar/R and I-MR control charts (Phase I limits from a baseline window)
* Western Electric / Nelson run rules
* Process capability: Cp, Cpk (within-subgroup sigma) and Pp, Ppk (overall sigma)

Control-chart constants are the standard tabulated values for subgroup sizes 2-10.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

# n: (A2, D3, D4, d2)
CONSTANTS = {
    2: (1.880, 0.000, 3.267, 1.128),
    3: (1.023, 0.000, 2.574, 1.693),
    4: (0.729, 0.000, 2.282, 2.059),
    5: (0.577, 0.000, 2.114, 2.326),
    6: (0.483, 0.000, 2.004, 2.534),
    7: (0.419, 0.076, 1.924, 2.704),
    8: (0.373, 0.136, 1.864, 2.847),
    9: (0.337, 0.184, 1.816, 2.970),
    10: (0.308, 0.223, 1.777, 3.078),
}
D2_MR = 1.128  # d2 for moving range of 2

RULE_LABELS = {
    "R1": "1 point beyond 3σ",
    "R2": "9 points in a row on one side of the center line",
    "R3": "6 points in a row steadily rising or falling",
    "R5": "2 of 3 points beyond 2σ (same side)",
}


# ─── Data shaping ───────────────────────────────────────────────────────────
def to_subgroup_matrix(df: pd.DataFrame, subgroup_col: str = "subgroup",
                       value_col: str = "measurement"):
    """Return (subgroup ids, matrix [n_subgroups x n]) from long-format data."""
    for col in (subgroup_col, value_col):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Expected columns: {subgroup_col}, {value_col}")
    clean = df[[subgroup_col, value_col]].dropna()
    clean = clean.assign(**{value_col: pd.to_numeric(clean[value_col], errors="coerce")}).dropna()
    groups = [(k, g[value_col].to_numpy()) for k, g in clean.groupby(subgroup_col, sort=False)]
    sizes = {len(v) for _, v in groups}
    if len(sizes) != 1:
        raise ValueError("All subgroups must have the same number of measurements.")
    n = sizes.pop()
    if n not in CONSTANTS:
        raise ValueError("Subgroup size must be between 2 and 10 for X-bar/R charts.")
    ids = [k for k, _ in groups]
    return ids, np.vstack([v for _, v in groups])


# ─── Run rules ──────────────────────────────────────────────────────────────
def detect_signals(values: Sequence[float], center: float, sigma: float,
                   rules: Sequence[str] = ("R1", "R2", "R3", "R5")) -> Dict[int, List[str]]:
    """Return {index: [rules violated]} using Western Electric / Nelson rules."""
    v = np.asarray(values, dtype=float)
    z = (v - center) / sigma if sigma > 0 else np.zeros_like(v)
    out: Dict[int, List[str]] = {}

    def flag(i: int, rule: str) -> None:
        out.setdefault(i, [])
        if rule not in out[i]:
            out[i].append(rule)

    n = len(v)
    for i in range(n):
        if "R1" in rules and abs(z[i]) > 3:
            flag(i, "R1")
        if "R2" in rules and i >= 8:
            w = z[i - 8:i + 1]
            if np.all(w > 0) or np.all(w < 0):
                flag(i, "R2")
        if "R3" in rules and i >= 5:
            w = v[i - 5:i + 1]
            d = np.diff(w)
            if np.all(d > 0) or np.all(d < 0):
                flag(i, "R3")
        if "R5" in rules and i >= 2:
            w = z[i - 2:i + 1]
            if abs(z[i]) > 2 and ((w > 2).sum() >= 2 or (w < -2).sum() >= 2):
                if (z[i] > 2 and (w > 2).sum() >= 2) or (z[i] < -2 and (w < -2).sum() >= 2):
                    flag(i, "R5")
    return out


# ─── X-bar / R ──────────────────────────────────────────────────────────────
@dataclass
class ControlChart:
    kind: str
    x: np.ndarray          # plotted statistic (subgroup means or individuals)
    spread: np.ndarray     # ranges (R) or moving ranges (MR)
    cl: float
    ucl: float
    lcl: float
    sigma_stat: float      # sigma of the plotted statistic
    cl_spread: float
    ucl_spread: float
    lcl_spread: float
    sigma_within: float    # estimated within-subgroup / short-term sigma
    n: int
    baseline: int
    signals: Dict[int, List[str]]
    spread_signals: List[int]


def xbar_r_chart(matrix: np.ndarray, baseline: int,
                 rules: Sequence[str] = ("R1", "R2", "R3", "R5")) -> ControlChart:
    k, n = matrix.shape
    baseline = int(max(2, min(baseline, k)))
    xbar, r = matrix.mean(axis=1), np.ptp(matrix, axis=1)
    a2, d3, d4, d2 = CONSTANTS[n]
    xbb, rbar = xbar[:baseline].mean(), r[:baseline].mean()
    sigma_within = rbar / d2
    sigma_x = sigma_within / math.sqrt(n)
    ucl_r, lcl_r = d4 * rbar, d3 * rbar
    spread_sig = [i for i, val in enumerate(r) if val > ucl_r or val < lcl_r]
    return ControlChart(
        kind="X̄-R", x=xbar, spread=r, cl=xbb, ucl=xbb + a2 * rbar, lcl=xbb - a2 * rbar,
        sigma_stat=sigma_x, cl_spread=rbar, ucl_spread=ucl_r, lcl_spread=lcl_r,
        sigma_within=sigma_within, n=n, baseline=baseline,
        signals=detect_signals(xbar, xbb, sigma_x, rules), spread_signals=spread_sig)


def imr_chart(values: np.ndarray, baseline: int,
              rules: Sequence[str] = ("R1", "R2", "R3", "R5")) -> ControlChart:
    x = np.asarray(values, dtype=float)
    mr = np.concatenate([[np.nan], np.abs(np.diff(x))])
    baseline = int(max(3, min(baseline, len(x))))
    xbar = x[:baseline].mean()
    mrbar = np.nanmean(mr[1:baseline])
    sigma = mrbar / D2_MR
    ucl_mr = 3.267 * mrbar
    spread_sig = [i for i, val in enumerate(mr) if not np.isnan(val) and val > ucl_mr]
    return ControlChart(
        kind="I-MR", x=x, spread=mr, cl=xbar, ucl=xbar + 3 * sigma, lcl=xbar - 3 * sigma,
        sigma_stat=sigma, cl_spread=mrbar, ucl_spread=ucl_mr, lcl_spread=0.0,
        sigma_within=sigma, n=1, baseline=baseline,
        signals=detect_signals(x, xbar, sigma, rules), spread_signals=spread_sig)


# ─── Capability ─────────────────────────────────────────────────────────────
def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass
class Capability:
    mean: float
    sigma_within: float
    sigma_overall: float
    cp: float
    cpk: float
    pp: float
    ppk: float
    ppm_overall: float


def capability(values: np.ndarray, sigma_within: float, lsl: float, usl: float) -> Capability:
    if usl <= lsl:
        raise ValueError("USL must be greater than LSL.")
    v = np.asarray(values, dtype=float).ravel()
    mu, s_all = float(v.mean()), float(v.std(ddof=1))
    cp = (usl - lsl) / (6 * sigma_within)
    cpk = min(usl - mu, mu - lsl) / (3 * sigma_within)
    pp = (usl - lsl) / (6 * s_all)
    ppk = min(usl - mu, mu - lsl) / (3 * s_all)
    ppm = (_phi((lsl - mu) / s_all) + 1.0 - _phi((usl - mu) / s_all)) * 1e6
    return Capability(mu, sigma_within, s_all, cp, cpk, pp, ppk, ppm)


def verdict(cpk: float) -> str:
    if cpk >= 1.67:
        return "Excellent (Cpk ≥ 1.67)"
    if cpk >= 1.33:
        return "Capable (Cpk ≥ 1.33)"
    if cpk >= 1.0:
        return "Marginal (1.00 ≤ Cpk < 1.33)"
    return "Not capable (Cpk < 1.00)"
