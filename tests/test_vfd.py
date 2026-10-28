import math

import numpy as np
import pytest

from study import fan_load, rated, starts
from vfd import pwm
from vfd.energy import Application, annual, speed_for
from vfd.motor import Motor, slip_for_torque, steady_state, vf_voltage


@pytest.fixture(scope="module")
def m():
    return Motor()


def test_rated_point_is_realistic(m):
    r = rated(m)
    assert r["p_out"] == pytest.approx(15e3, rel=0.02)
    assert 0.015 < r["slip"] < 0.03
    assert 0.90 < r["efficiency"] < 0.95
    assert 0.82 < r["pf"] < 0.90
    locked = steady_state(m, m.v_ll, 50, 1.0)["i_a"] / r["i_a"]
    assert 5.5 < locked < 7.5


def test_vf_law(m):
    assert vf_voltage(m, 0) == pytest.approx(8.0)
    assert vf_voltage(m, 50) == pytest.approx(415.0)
    assert vf_voltage(m, 25) == pytest.approx(8 + 407 / 2)


@pytest.fixture(scope="module")
def start_runs(m):
    load, op = fan_load(m)
    dol, vf, summary = starts(m, load, rated(m)["i_a"])
    return dol, vf, summary, op


def test_dynamic_model_settles_on_equivalent_circuit(start_runs):
    dol, vf, _, op = start_runs
    # no core loss in the dynamic model, so allow a few percent
    assert dol["i_rms"][-1] == pytest.approx(op["i_a"], rel=0.04)
    assert dol["speed_rpm"][-1] == pytest.approx(op["speed_rpm"], abs=3)
    assert vf["speed_rpm"][-1] == pytest.approx(dol["speed_rpm"][-1], abs=1)


def test_soft_start_limits_inrush(start_runs):
    _, _, s, _ = start_runs
    assert s["DOL inrush / rated"] > 6.0
    assert s["V/f max current / rated"] < 1.2
    assert s["DOL peak torque (N m)"] > 3 * s["V/f peak torque (N m)"]


def test_pwm_fundamental_and_symmetry(m):
    f = 40.0
    v = vf_voltage(m, f)
    r = pwm.run(m, f, v, 50.0, f_carrier=99 * f, settle_cycles=6)
    _, v1 = pwm.thd(r["t"], r["vab"], f)
    assert v1 / math.sqrt(2) == pytest.approx(v, rel=0.01)
    assert abs(r["ia"].mean()) < 0.05
    thd_i, _ = pwm.thd(r["t"], r["ia"], f)
    assert thd_i < 8


def test_affinity_for_a_pure_friction_system():
    app = Application("fan", 13.5, 0.0)
    for q in (0.6, 0.8, 1.0):
        assert speed_for(app, q) == pytest.approx(q)


def test_drive_saves_energy_at_part_flow(m):
    fan = annual(m, Application("fan", 13.5, 0.0))
    pump = annual(m, Application("pump", 13.5, 0.4))
    for row in fan["rows"]:
        if row["flow_pct"] < 100:
            assert row["input_kw_vfd"] < row["input_kw_dol"]
    # static head limits how far the speed can fall
    assert fan["saving_pct"] > pump["saving_pct"] > 15
