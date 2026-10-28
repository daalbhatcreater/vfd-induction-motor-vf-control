"""Induction motor: steady-state equivalent circuit and dynamic dq model.

Parameters are per phase, star equivalent, referred to the stator. The dynamic
model runs in a reference frame turning with the supply voltage vector, so a
balanced sinusoidal supply appears as a constant voltage and the solver can
take long steps once the electrical transient has died away.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


@dataclass
class Motor:
    v_ll: float = 415.0
    f_rated: float = 50.0
    poles: int = 4
    rs: float = 0.25
    rr: float = 0.20
    lls: float = 2.2e-3
    llr: float = 2.2e-3
    lm: float = 70e-3
    rc: float = 574.0          # core-loss resistance, parallel with Lm (~300 W)
    friction_w: float = 150.0  # friction and windage at synchronous speed
    j: float = 1.5             # motor plus fan impeller, kg m2

    @property
    def p(self):
        return self.poles // 2

    @property
    def ls(self):
        return self.lls + self.lm

    @property
    def lr(self):
        return self.llr + self.lm

    @property
    def w_sync(self):
        return 2 * math.pi * self.f_rated / self.p


def friction_torque(m: Motor, wm):
    """Windage grows with speed cubed in power, so torque with speed squared."""
    return m.friction_w / m.w_sync * (wm / m.w_sync) ** 2


def steady_state(m: Motor, v_ll, f, slip):
    """Per-phase equivalent circuit at supply voltage v_ll and frequency f."""
    w = 2 * math.pi * f
    vph = v_ll / math.sqrt(3)
    zs = complex(m.rs, w * m.lls)
    zr = complex(m.rr / slip, w * m.llr)
    zm = 1 / (1 / complex(0, w * m.lm) + 1 / (m.rc * f / m.f_rated))
    z = zs + zm * zr / (zm + zr)
    i_s = vph / z
    e = i_s * zm * zr / (zm + zr)
    i_r = e / zr
    p_gap = 3 * abs(i_r) ** 2 * m.rr / slip
    w_s = w / m.p
    wm = w_s * (1 - slip)
    torque = p_gap / w_s - friction_torque(m, wm)
    p_in = 3 * (vph * i_s.conjugate()).real
    return {"i_a": abs(i_s), "torque": torque, "speed_rpm": wm * 60 / (2 * math.pi),
            "p_out": torque * wm, "p_in": p_in, "pf": math.cos(math.atan2(z.imag, z.real)),
            "efficiency": torque * wm / p_in if p_in > 0 else 0.0, "wm": wm}


def slip_for_torque(m: Motor, v_ll, f, torque):
    """Operating slip on the stable side of the torque curve."""
    s_max = m.rr / math.hypot(m.rs, 2 * math.pi * f * (m.lls + m.llr))
    return brentq(lambda s: steady_state(m, v_ll, f, s)["torque"] - torque, 1e-5, s_max)


def vf_voltage(m: Motor, f, boost_v=8.0):
    """Linear V/f law with a low-frequency boost for the stator IR drop."""
    return min(boost_v + (m.v_ll - boost_v) * f / m.f_rated, m.v_ll)


# ---- dynamic model --------------------------------------------------------------
def simulate(m: Motor, freq, volts, load, t_end, t_eval, max_step=1e-3):
    """Integrate the motor in the supply-voltage frame.

    freq(t), volts(t): supply frequency (Hz) and line voltage (V rms)
    load(wm): load torque (N m) at mechanical speed wm
    State: stator and rotor flux (d, q) and mechanical speed.
    """
    sigma = m.ls * m.lr - m.lm ** 2

    def currents(psd, psq, prd, prq):
        isd = (m.lr * psd - m.lm * prd) / sigma
        isq = (m.lr * psq - m.lm * prq) / sigma
        ird = (m.ls * prd - m.lm * psd) / sigma
        irq = (m.ls * prq - m.lm * psq) / sigma
        return isd, isq, ird, irq

    def rhs(t, x):
        psd, psq, prd, prq, wm = x
        we = 2 * math.pi * freq(t)
        wr = m.p * wm
        vd = math.sqrt(2) * volts(t) / math.sqrt(3)
        isd, isq, ird, irq = currents(psd, psq, prd, prq)
        te = 1.5 * m.p * (psd * isq - psq * isd)
        return [vd - m.rs * isd + we * psq,
                -m.rs * isq - we * psd,
                -m.rr * ird + (we - wr) * prq,
                -m.rr * irq - (we - wr) * prd,
                (te - load(wm) - friction_torque(m, wm)) / m.j]

    sol = solve_ivp(rhs, (0, t_end), [0, 0, 0, 0, 0], t_eval=t_eval, max_step=max_step,
                    method="RK45", rtol=1e-6, atol=1e-6)
    psd, psq, prd, prq, wm = sol.y
    isd, isq, _, _ = currents(psd, psq, prd, prq)
    te = 1.5 * m.p * (psd * isq - psq * isd)
    # phase-a current: rotate back by the supply angle
    theta = np.array([_angle(freq, t) for t in sol.t])
    ia = np.real((isd + 1j * isq) * np.exp(1j * theta))
    return {"t": sol.t, "i_rms": np.hypot(isd, isq) / math.sqrt(2), "ia": ia, "te": te,
            "speed_rpm": wm * 60 / (2 * math.pi), "wm": wm, "freq": np.array([freq(t) for t in sol.t]),
            "flux": np.hypot(psd, psq)}


def _angle(freq, t, n=200):
    """Supply angle, the integral of 2 pi f."""
    if t == 0:
        return 0.0
    ts = np.linspace(0, t, n)
    return float(np.trapezoid([2 * math.pi * freq(x) for x in ts], ts))
