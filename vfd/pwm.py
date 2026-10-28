"""Switching-level simulation of the inverter and motor for a few cycles.

A three-phase two-level inverter on a diode-rectified DC bus is modulated with
sine-triangle PWM plus min-max (space-vector equivalent) zero-sequence
injection, which raises the linear range to Vdc / sqrt(3) phase peak. The motor
is integrated in the stationary frame with a fixed 1 us step, starting from its
steady state so no start-up transient is mixed into the waveforms.
"""

import math

import numpy as np

from .motor import Motor, slip_for_torque, steady_state


def run(m: Motor, f, v_ll, load_torque, f_carrier=4000.0, vdc=None, cycles=3,
        settle_cycles=12, dt=1e-6):
    vdc = vdc or 1.35 * m.v_ll
    slip = slip_for_torque(m, v_ll, f, load_torque)
    w = 2 * math.pi * f
    wr = m.p * steady_state(m, v_ll, f, slip)["wm"]

    # initial fluxes from the steady-state phasors (peak, stationary frame)
    vph = v_ll / math.sqrt(3)
    zs = complex(m.rs, w * m.lls)
    zr = complex(m.rr / slip, w * m.llr)
    zm = complex(0, w * m.lm)
    i_s = vph / (zs + zm * zr / (zm + zr))
    i_r = -i_s * zm / (zm + zr)
    psi_s = math.sqrt(2) * (m.ls * i_s + m.lm * i_r)
    psi_r = math.sqrt(2) * (m.lr * i_r + m.lm * i_s)

    sigma = m.ls * m.lr - m.lm ** 2
    a = complex(-0.5, math.sqrt(3) / 2)
    v_peak = math.sqrt(2) * vph
    n = int((cycles + settle_cycles) / f / dt)
    keep = int(settle_cycles / f / dt)
    stride = 5
    t_out, ia_out, vab_out = [], [], []

    for k in range(n):
        t = k * dt
        refs = [v_peak * math.cos(w * t - 2 * math.pi * ph / 3) for ph in range(3)]
        offset = -(max(refs) + min(refs)) / 2
        # triangular carrier between -Vdc/2 and +Vdc/2
        x = (t * f_carrier) % 1.0
        carrier = (4 * x - 1 if x < 0.5 else 3 - 4 * x) * vdc / 2
        legs = [vdc / 2 if r + offset > carrier else -vdc / 2 for r in refs]
        vs = (2 / 3) * (legs[0] + a * legs[1] + a * a * legs[2])

        i_s = (m.lr * psi_s - m.lm * psi_r) / sigma
        i_r = (m.ls * psi_r - m.lm * psi_s) / sigma
        psi_s += (vs - m.rs * i_s) * dt
        psi_r += (-m.rr * i_r + 1j * wr * psi_r) * dt

        if k >= keep and k % stride == 0:
            t_out.append(t)
            ia_out.append(i_s.real)
            vab_out.append(legs[0] - legs[1])

    t_out = np.array(t_out) - t_out[0]
    ia = np.array(ia_out)
    return {"t": t_out, "ia": ia, "vab": np.array(vab_out), "f": f, "vdc": vdc,
            "f_carrier": f_carrier, "dt": dt * stride}


def spectrum(t, x, f1):
    """Amplitude spectrum over a whole number of fundamental cycles."""
    n_cycle = int(round(1 / f1 / (t[1] - t[0])))
    n = (len(x) // n_cycle) * n_cycle
    x = x[:n]
    amp = np.abs(np.fft.rfft(x)) * 2 / n
    freqs = np.fft.rfftfreq(n, t[1] - t[0])
    return freqs, amp


def thd(t, x, f1, up_to_hz=20e3):
    freqs, amp = spectrum(t, x, f1)
    k1 = np.argmin(np.abs(freqs - f1))
    harm = amp[(freqs > 1.5 * f1) & (freqs <= up_to_hz)]
    return 100 * math.sqrt(np.sum(harm ** 2)) / amp[k1], amp[k1]
