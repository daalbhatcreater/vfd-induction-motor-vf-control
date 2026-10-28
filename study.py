"""DOL against V/f starting, PWM waveforms and annual energy.

    python study.py
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vfd import pwm
from vfd.energy import DUTY, HOURS_PER_YEAR, TARIFF_INR_PER_KWH, Application, annual
from vfd.motor import Motor, simulate, slip_for_torque, steady_state, vf_voltage

OUT = Path(__file__).parent / "results"
FAN_KW = 13.5           # fan shaft power at design flow
RAMP_S = 10.0           # drive acceleration time 0 -> 50 Hz


def rated(m):
    """Nameplate point: rated torque at 50 Hz, 415 V."""
    t_rated = 97.4
    s = slip_for_torque(m, m.v_ll, m.f_rated, t_rated)
    r = steady_state(m, m.v_ll, m.f_rated, s)
    r["slip"] = s
    r["torque_rated"] = t_rated
    return r


def fan_load(m):
    from vfd.energy import motor_input_dol
    op = motor_input_dol(m, FAN_KW)
    t_op = FAN_KW * 1e3 / op["wm"]
    return (lambda wm: t_op * (max(wm, 0) / op["wm"]) ** 2), op


def sliding_rms(t, x, window):
    dt = t[1] - t[0]
    n = max(int(round(window / dt)), 1)
    c = np.cumsum(np.insert(x * x, 0, 0.0))
    out = np.sqrt((c[n:] - c[:-n]) / n)
    return t[n - 1:], out


def starts(m, load, i_rated):
    t_dol = np.arange(0, 4.0, 2e-4)
    dol = simulate(m, lambda t: 50.0, lambda t: m.v_ll, load, 4.0, t_dol, max_step=2e-4)
    fq = lambda t: min(50.0 * t / RAMP_S, 50.0)
    t_vf = np.arange(0, RAMP_S + 4.0, 2e-3)
    vf = simulate(m, fq, lambda t: vf_voltage(m, fq(t)), load, RAMP_S + 4.0, t_vf, max_step=2e-3)

    _, dol_rms = sliding_rms(dol["t"], dol["ia"], 0.02)
    def run_up(res, n_target):
        idx = np.argmax(res["speed_rpm"] >= 0.98 * n_target)
        return res["t"][idx]
    n_final = dol["speed_rpm"][-1]
    summary = {
        "DOL peak instantaneous current (A)": np.abs(dol["ia"]).max(),
        "DOL max 1-cycle rms current (A)": dol_rms.max(),
        "DOL inrush / rated": dol_rms.max() / i_rated,
        "DOL first-cycle peak / rated rms": np.abs(dol["ia"]).max() / i_rated,
        "DOL peak torque (N m)": dol["te"].max(),
        "DOL time to 98 % speed (s)": run_up(dol, n_final),
        "V/f max rms current (A)": vf["i_rms"].max(),
        "V/f max current / rated": vf["i_rms"].max() / i_rated,
        "V/f peak torque (N m)": vf["te"].max(),
        "V/f time to 98 % speed (s)": run_up(vf, n_final),
    }
    return dol, vf, summary


def plot_starts(dol, vf, i_rated, path):
    fig, ax = plt.subplots(3, 2, figsize=(11, 8.5), sharex="col")
    for col, (res, title) in enumerate([(dol, "Direct-on-line"), (vf, f"V/f drive, {RAMP_S:g} s ramp")]):
        if col == 0:
            ax[0, col].plot(res["t"], res["ia"], lw=0.5, color="tab:red", label="phase current")
            tr, rms = sliding_rms(res["t"], res["ia"], 0.02)
            ax[0, col].plot(tr, rms, color="black", lw=1.2, label="1-cycle rms")
        else:
            ax[0, col].plot(res["t"], res["i_rms"], color="black", lw=1.2, label="rms current")
        ax[0, col].axhline(i_rated, color="tab:blue", ls="--", lw=0.8, label="rated current")
        ax[0, col].set_ylabel("Current (A)")
        ax[0, col].legend(fontsize=7, loc="upper right")
        ax[0, col].set_title(title)
        ax[1, col].plot(res["t"], res["te"], color="tab:purple")
        ax[1, col].set_ylabel("Torque (N m)")
        ax[2, col].plot(res["t"], res["speed_rpm"], color="tab:green", label="speed")
        ax[2, col].set_ylabel("Speed (rpm)")
        ax[2, col].set_xlabel("Time (s)")
        if col == 1:
            a2 = ax[2, col].twinx()
            a2.plot(res["t"], res["freq"], color="grey", ls=":", label="drive frequency")
            a2.set_ylabel("Frequency (Hz)")
        for a in ax[:, col]:
            a.grid(True, lw=0.3)
    ax[0, 0].set_ylim(-280, 280)
    ax[0, 1].set_ylim(0, 280)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_torque_speed(m, load, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for f in (10, 20, 30, 40, 50):
        v = vf_voltage(m, f)
        s = np.linspace(1e-4, 1.0, 600)
        rows = [steady_state(m, v, f, x) for x in s]
        ax.plot([r["speed_rpm"] for r in rows], [r["torque"] for r in rows],
                label=f"{f} Hz, {v:.0f} V")
    w = np.linspace(0, m.w_sync, 200)
    ax.plot(w * 60 / (2 * math.pi), [load(x) for x in w], "k--", label="fan load")
    ax.set_xlim(0, 1550)
    ax.set_ylim(0, 300)
    ax.set_xlabel("Speed (rpm)")
    ax.set_ylabel("Torque (N m)")
    ax.set_title("Torque-speed curves under the V/f law")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_pwm(r, path):
    t = r["t"] * 1e3
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.5))
    ax[0].plot(t, r["vab"], lw=0.4, color="tab:blue")
    ax[0].set_ylabel("V_ab (V)")
    ax[0].set_title(f"Inverter at {r['f']:g} Hz: {r['f_carrier'] / 1e3:g} kHz carrier, "
                    f"Vdc {r['vdc']:.0f} V, SVPWM-equivalent modulation")
    ax[1].plot(t, r["ia"], color="tab:red")
    ax[1].set_ylabel("i_a (A)")
    ax[1].set_xlabel("Time (ms)")
    freqs, amp = pwm.spectrum(r["t"], r["ia"], r["f"])
    ax[2].bar(freqs / 1e3, amp, width=0.03, color="tab:red")
    ax[2].set_xlim(0, 10)
    ax[2].set_yscale("log")
    ax[2].set_ylim(1e-3, 50)
    ax[2].set_xlabel("Frequency (kHz)")
    ax[2].set_ylabel("|i_a| (A peak)")
    for a in ax:
        a.grid(True, lw=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_energy(results, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, (name, a) in enumerate(results.items()):
        rows = a["rows"]
        flows = [r["flow_pct"] for r in rows]
        ax[k].plot(flows, [r["input_kw_dol"] for r in rows], "o-", label="DOL + damper / valve")
        ax[k].plot(flows, [r["input_kw_vfd"] for r in rows], "s-", label="V/f drive")
        ax[k].set_title(f"{name}: {a['saving_pct']:.1f} % less energy")
        ax[k].set_xlabel("Flow (%)")
        ax[k].set_ylabel("Electrical input (kW)")
        ax[k].invert_xaxis()
        ax[k].grid(True, lw=0.3)
        ax[k].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    m = Motor()
    r = rated(m)
    load, op = fan_load(m)

    dol, vf, start_summary = starts(m, load, r["i_a"])
    plot_starts(dol, vf, r["i_a"], OUT / "dol_vs_vf_start.png")
    plot_torque_speed(m, load, OUT / "torque_speed_vf.png")

    f_pwm = 40.0
    t_load = load(0.8 * op["wm"])
    # synchronous carrier, odd multiple of 3 (mf = 99): keeps half-wave and
    # three-phase symmetry, so no DC or even harmonics in the output
    pw = pwm.run(m, f_pwm, vf_voltage(m, f_pwm), t_load, f_carrier=99 * f_pwm)
    thd_i, i1 = pwm.thd(pw["t"], pw["ia"], f_pwm)
    thd_v, v1 = pwm.thd(pw["t"], pw["vab"], f_pwm)
    plot_pwm(pw, OUT / "pwm_waveforms.png")

    apps = [Application("Fan with outlet damper", FAN_KW, 0.0),
            Application("Pump with throttling valve, 40 % static head", FAN_KW, 0.4)]
    energy = {a.name: annual(m, a) for a in apps}
    plot_energy(energy, OUT / "energy_vfd_vs_dol.png")

    motor = pd.DataFrame([
        ("Rating", "15 kW, 415 V, 50 Hz, 4-pole"),
        ("Equivalent circuit", f"Rs {m.rs} Ohm, Rr {m.rr} Ohm, Lls = Llr {m.lls * 1e3:g} mH, "
                               f"Lm {m.lm * 1e3:g} mH, Rc {m.rc:g} Ohm"),
        ("Rated point", f"{r['torque_rated']:.1f} N m, {r['speed_rpm']:.0f} rpm, {r['i_a']:.1f} A, "
                        f"pf {r['pf']:.3f}, efficiency {100 * r['efficiency']:.1f} %"),
        ("Locked rotor at 415 V", f"{steady_state(m, m.v_ll, 50, 1.0)['i_a']:.0f} A "
                                  f"({steady_state(m, m.v_ll, 50, 1.0)['i_a'] / r['i_a']:.1f} x rated)"),
        ("Fan operating point", f"{FAN_KW} kW at {op['speed_rpm']:.0f} rpm, {op['i_a']:.1f} A"),
        ("Inertia (motor + impeller)", f"{m.j} kg m2"),
    ], columns=["Item", "Value"])
    starts_df = pd.DataFrame([(k, round(v, 2)) for k, v in start_summary.items()],
                             columns=["Quantity", "Value"])
    pwm_df = pd.DataFrame([
        ("Operating point", f"{f_pwm:g} Hz, {vf_voltage(m, f_pwm):.0f} V (V/f), 80 % fan speed"),
        ("Carrier / DC bus", f"{pw['f_carrier'] / 1e3:g} kHz (mf = 99) / {pw['vdc']:.0f} V"),
        ("Fundamental line voltage", f"{v1 / math.sqrt(2):.1f} V rms"),
        ("Fundamental current", f"{i1 / math.sqrt(2):.2f} A rms"),
        ("Current THD (to 20 kHz)", f"{thd_i:.2f} %"),
        ("Line voltage THD (to 20 kHz)", f"{thd_v:.1f} %"),
    ], columns=["Item", "Value"])
    energy_rows = []
    for name, a in energy.items():
        for row in a["rows"]:
            energy_rows.append({"Application": name, **{k: round(v, 2) for k, v in row.items()}})
    energy_df = pd.DataFrame(energy_rows)
    energy_sum = pd.DataFrame([{
        "Application": name, "kWh/year DOL": round(a["kwh_dol"]),
        "kWh/year V/f drive": round(a["kwh_vfd"]), "Saving (kWh)": round(a["saving_kwh"]),
        "Saving (%)": round(a["saving_pct"], 1),
        f"Saving (INR at {TARIFF_INR_PER_KWH:g}/kWh)": round(a["saving_inr"]),
    } for name, a in energy.items()])

    with pd.ExcelWriter(OUT / "vfd_study.xlsx") as xl:
        motor.to_excel(xl, sheet_name="Motor", index=False)
        starts_df.to_excel(xl, sheet_name="Starting", index=False)
        pwm_df.to_excel(xl, sheet_name="PWM", index=False)
        energy_sum.to_excel(xl, sheet_name="Energy summary", index=False)
        energy_df.to_excel(xl, sheet_name="Energy by flow", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    for title, df in [("Motor", motor), ("Starting", starts_df), ("PWM", pwm_df),
                      (f"Energy, {HOURS_PER_YEAR} h/year, duty {DUTY}", energy_sum),
                      ("Energy by flow", energy_df)]:
        print(title)
        print(df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
