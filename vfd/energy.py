"""Annual energy of a fan or pump: constant-speed motor with a damper or
throttling valve against the same motor on a V/f drive.

Machine characteristics in per unit of the design point (flow 1, head 1):
  head at full speed           H(Q) = 1.25 - 0.25 Q^2
  head at speed n (affinity)   H(Q, n) = 1.25 n^2 - 0.25 Q^2
  efficiency                   eta(x) = eta_bep (2x - x^2),  x = Q / (n Q_bep)
  system curve                 H_sys(Q) = h_static + (1 - h_static) Q^2

With a damper/valve the machine stays at full speed and the operating point
moves along H(Q); the extra head is burnt across the damper. With a drive
the speed drops until H(Q, n) meets the system curve.
"""

import math
from dataclasses import dataclass

from scipy.optimize import brentq

from .motor import Motor, slip_for_torque, steady_state, vf_voltage

# share of running hours at each flow, typical of a variable-demand fan or pump
DUTY = [(1.0, 0.10), (0.9, 0.20), (0.8, 0.30), (0.7, 0.25), (0.6, 0.15)]
HOURS_PER_YEAR = 6000
TARIFF_INR_PER_KWH = 8.0


@dataclass
class Application:
    name: str
    shaft_kw_design: float      # shaft power at design flow, full speed
    h_static: float             # static head as a fraction of design head
    eta_bep: float = 0.80


def head(q, n=1.0):
    return 1.25 * n * n - 0.25 * q * q


def eta(q, n=1.0, eta_bep=0.80):
    x = q / n
    return eta_bep * (2 * x - x * x)


def shaft_kw_throttled(app, q):
    return app.shaft_kw_design * (q * head(q) / eta(q, 1.0, app.eta_bep)) / (1.0 / app.eta_bep)


def speed_for(app, q):
    h_sys = app.h_static + (1 - app.h_static) * q * q
    return math.sqrt((h_sys + 0.25 * q * q) / 1.25)


def shaft_kw_vfd(app, q):
    n = speed_for(app, q)
    h_sys = app.h_static + (1 - app.h_static) * q * q
    return app.shaft_kw_design * (q * h_sys / eta(q, n, app.eta_bep)) / (1.0 / app.eta_bep), n


def drive_efficiency(load_frac):
    return 0.96 + 0.015 * min(load_frac, 1.0)


def motor_input_dol(m: Motor, shaft_kw):
    """Motor on the 50 Hz supply delivering shaft_kw: solve speed and torque together."""
    def mismatch(s):
        r = steady_state(m, m.v_ll, m.f_rated, s)
        return r["p_out"] - shaft_kw * 1e3
    s_max = m.rr / math.hypot(m.rs, 2 * math.pi * m.f_rated * (m.lls + m.llr))
    s = brentq(mismatch, 1e-5, s_max)
    return steady_state(m, m.v_ll, m.f_rated, s)


def motor_input_vfd(m: Motor, shaft_kw, n_frac, n_design_rpm):
    """Find the drive frequency that runs the motor at the required speed
    under the required torque."""
    target_wm = n_frac * n_design_rpm * 2 * math.pi / 60
    torque = shaft_kw * 1e3 / target_wm

    def speed_error(f):
        v = vf_voltage(m, f)
        s = slip_for_torque(m, v, f, torque)
        return steady_state(m, v, f, s)["wm"] - target_wm

    f = brentq(speed_error, 5.0, 60.0)
    v = vf_voltage(m, f)
    r = steady_state(m, v, f, slip_for_torque(m, v, f, torque))
    r["f"] = f
    r["v"] = v
    return r


def annual(m: Motor, app: Application):
    design = motor_input_dol(m, app.shaft_kw_design)
    n_design_rpm = design["speed_rpm"]
    rows = []
    kwh_dol = kwh_vfd = 0.0
    for q, share in DUTY:
        hours = share * HOURS_PER_YEAR
        p_thr = shaft_kw_throttled(app, q)
        dol = motor_input_dol(m, p_thr)
        p_vfd, n = shaft_kw_vfd(app, q)
        vfd = motor_input_vfd(m, p_vfd, n, n_design_rpm)
        p_in_vfd = vfd["p_in"] / drive_efficiency(p_vfd / app.shaft_kw_design)
        kwh_dol += dol["p_in"] / 1e3 * hours
        kwh_vfd += p_in_vfd / 1e3 * hours
        rows.append({
            "flow_pct": round(100 * q), "hours": hours,
            "shaft_kw_throttled": p_thr, "input_kw_dol": dol["p_in"] / 1e3,
            "speed_pct": 100 * n, "drive_hz": vfd["f"], "shaft_kw_vfd": p_vfd,
            "motor_eff_vfd": vfd["efficiency"], "input_kw_vfd": p_in_vfd / 1e3,
        })
    saving = kwh_dol - kwh_vfd
    return {"rows": rows, "kwh_dol": kwh_dol, "kwh_vfd": kwh_vfd, "saving_kwh": saving,
            "saving_pct": 100 * saving / kwh_dol,
            "saving_inr": saving * TARIFF_INR_PER_KWH}
