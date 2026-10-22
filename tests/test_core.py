import numpy as np
import pandas as pd
import pytest

import oee
import spc


def test_constants_n5():
    assert spc.CONSTANTS[5] == (0.577, 0.0, 2.114, 2.326)


def test_xbar_r_limits_match_hand_calculation():
    m = np.array([[10, 11, 12, 13, 14],      # mean 12, R 4
                  [11, 12, 13, 14, 15],      # mean 13, R 4
                  [12, 13, 14, 15, 16]],     # mean 14, R 4
                 dtype=float)
    cc = spc.xbar_r_chart(m, baseline=3)
    assert cc.cl == pytest.approx(13.0)
    assert cc.cl_spread == pytest.approx(4.0)
    assert cc.ucl == pytest.approx(13 + 0.577 * 4)
    assert cc.lcl == pytest.approx(13 - 0.577 * 4)
    assert cc.ucl_spread == pytest.approx(2.114 * 4)
    assert cc.sigma_within == pytest.approx(4 / 2.326)


def test_rule1_detects_point_beyond_3_sigma():
    x = np.zeros(20)
    x[10] = 5.0
    sig = spc.detect_signals(x, 0.0, 1.0, rules=["R1"])
    assert list(sig.keys()) == [10]


def test_rule2_nine_points_one_side():
    x = np.array([0.5] * 9 + [-0.5] * 3)
    sig = spc.detect_signals(x, 0.0, 1.0, rules=["R2"])
    assert 8 in sig and 7 not in sig


def test_rule3_six_points_trend():
    x = np.array([0, 1, 2, 3, 4, 5, 0], dtype=float) * 0.1
    sig = spc.detect_signals(x, 0.0, 10.0, rules=["R3"])
    assert 5 in sig and 6 not in sig


def test_rule5_two_of_three_beyond_2_sigma():
    x = np.array([0, 2.5, 0.1, 2.6, 0], dtype=float)
    sig = spc.detect_signals(x, 0.0, 1.0, rules=["R5"])
    assert 3 in sig


def test_capability_known_values():
    v = np.random.default_rng(1).normal(0, 1, 100000)
    c = spc.capability(v, sigma_within=1.0, lsl=-4, usl=4)
    assert c.cp == pytest.approx(8 / 6)
    assert c.cpk == pytest.approx(4 / 3, rel=0.01)
    assert c.ppm_overall == pytest.approx(63.3, rel=0.3)   # about 63 ppm at +/-4 sigma


def test_capability_off_center_has_lower_cpk():
    v = np.random.default_rng(2).normal(1.0, 1.0, 5000)
    c = spc.capability(v, sigma_within=1.0, lsl=-4, usl=4)
    assert c.cpk < c.cp


def test_unequal_subgroups_rejected():
    df = pd.DataFrame({"subgroup": [1, 1, 1, 2, 2], "measurement": [1, 2, 3, 4, 5]})
    with pytest.raises(ValueError):
        spc.to_subgroup_matrix(df)


def _example():
    prod = pd.DataFrame([{"date": "d1", "shift": "A", "planned_min": 480,
                          "ideal_cycle_min": 1.0, "total_count": 400, "good_count": 380}])
    down = pd.DataFrame([{"date": "d1", "shift": "A", "reason": "Breakdown", "minutes": 60}])
    return prod, down


def test_oee_known_example():
    t = oee.shift_table(*_example())
    r = oee.ratios(t)
    assert r["availability"] == pytest.approx(420 / 480)
    assert r["performance"] == pytest.approx(400 / 420)
    assert r["quality"] == pytest.approx(380 / 400)
    assert r["oee"] == pytest.approx((420 / 480) * (400 / 420) * (380 / 400))


def test_loss_waterfall_adds_up():
    w = oee.loss_waterfall(oee.shift_table(*_example()))
    assert w["planned"] - w["downtime"] - w["speed"] - w["quality"] == pytest.approx(w["productive"])


def test_pareto_sorted_and_cumulative():
    d = pd.DataFrame({"date": ["d"] * 3, "shift": ["A"] * 3,
                      "reason": ["x", "y", "x"], "minutes": [10, 50, 30]})
    p = oee.pareto(d)
    assert p.loc[0, "minutes"] == 50
    assert p["cum_pct"].iloc[-1] == pytest.approx(100.0)
