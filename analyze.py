#!/usr/bin/env python3
"""
analyze.py — Bridge hypothesis test: Mycelium ≈ Memristor

Sweep dV/dt over 8 log-spaced values, simulate both systems, compute
hysteresis area A, fit power-law scaling A ∝ (dV/dt)^(-α), and
compute Pearson cross-correlation with bootstrap confidence intervals.

Outputs:
    ~/citations-needed/experiments/mycelium-memristor/
        ├── simulators.py
        ├── analyze.py
        ├── results.json
        └── plot.png           (3-panel figure)
"""

import json
import sys
import os
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── local ──────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulators import MyceliumSimulator, MemristorSimulator

# ── configuration ──────────────────────────────────────────────────────────
N_RATES = 8
DV_DT_MIN = 0.1          # V/s
DV_DT_MAX = 10.0         # V/s
V_AMP = 1.0              # triangle wave amplitude (V)
N_CYCLES = 3             # simulate 3 cycles, analyse the last one
N_BOOTSTRAP = 2000       # resamples for CI

OUT_DIR = os.path.expanduser("~/citations-needed/experiments/mycelium-memristor")

# ── helpers ────────────────────────────────────────────────────────────────


def extract_last_cycle(t, V, I):
    """Return (V_cycle, I_cycle) for the final complete cycle."""
    # Find zero-crossing indices
    signs = np.signbit(V)
    idx = np.where(np.diff(signs.astype(int)))[0]
    # A full cycle has 4 zero crossings: 0→+ → 0→− → 0
    if len(idx) < 4:
        # fallback: use the last third of the signal
        n = len(V)
        return V[-n // 3:], I[-n // 3:]
    # Last 4 zero crossings span one full cycle
    start = idx[-4]
    end = idx[-1] + 1
    return V[start:end], I[start:end]


def power_law(x, C, alpha):
    """A = C · (dV/dt)^(-alpha).  We fit log(A) = log(C) - alpha·log(x)."""
    return C * x ** (-alpha)


def fit_power_law(rate_vals, area_vals):
    """Fit A = C * rate^(-alpha) via log-space linear regression."""
    log_x = np.log(rate_vals)
    log_y = np.log(area_vals)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    C = np.exp(intercept)
    alpha = -slope  # because log(A) = log(C) - alpha·log(rate)
    # Compute R² manually
    residuals = log_y - (intercept + slope * log_x)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r_sq = 1.0 - ss_res / ss_tot
    return C, alpha, r_sq


def bootstrap_ci(rate_vals, area_vals, n_iter=N_BOOTSTRAP, ci=95):
    """
    Bootstrap confidence interval for the power-law exponent α.
    Resamples with replacement n_iter times.
    """
    n = len(rate_vals)
    alpha_samples = np.empty(n_iter)
    for i in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        try:
            _, alpha_i, _ = fit_power_law(rate_vals[idx], area_vals[idx])
            alpha_samples[i] = alpha_i
        except Exception:
            alpha_samples[i] = np.nan
    alpha_samples = alpha_samples[~np.isnan(alpha_samples)]
    low_pct = (100.0 - ci) / 2.0
    high_pct = 100.0 - low_pct
    ci_low = float(np.percentile(alpha_samples, low_pct))
    ci_high = float(np.percentile(alpha_samples, high_pct))
    return ci_low, ci_high, alpha_samples


# ── run sweep for one simulator ────────────────────────────────────────────


def sweep(sim, rates, V_amp=V_AMP, n_cycles=N_CYCLES):
    """Run simulator at each rate and return (rates, areas, iv_data)."""
    rates = np.asarray(rates, dtype=float)
    areas = np.empty_like(rates)
    iv_data = []          # list of (V_cycle, I_cycle, label) for plotting
    for j, dvdt in enumerate(rates):
        t, V, I, freq = sim.simulate(V_amp, dvdt, n_cycles=n_cycles)
        Vc, Ic = extract_last_cycle(t, V, I)
        area = sim.hysteresis_area(Vc, Ic)
        areas[j] = abs(area)   # polarity may flip; take absolute
        iv_data.append((Vc, Ic, f"dV/dt={dvdt:.2f} V/s"))
    return rates, areas, iv_data


# ── main ───────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)

    # Log-spaced sweep rates
    rates = np.logspace(np.log10(DV_DT_MIN), np.log10(DV_DT_MAX), N_RATES)
    print(f"Sweep rates (V/s): {np.round(rates, 3).tolist()}")

    # ── Simulate ───────────────────────────────────────────────────────
    print("\n--- Mycelium simulator ---")
    myc = MyceliumSimulator()
    r_myc, a_myc, iv_myc = sweep(myc, rates)
    for r, a in zip(r_myc, a_myc):
        print(f"  dV/dt = {r:6.3f}  →  area = {a:.6e}")

    print("\n--- Memristor simulator ---")
    mem = MemristorSimulator()
    r_mem, a_mem, iv_mem = sweep(mem, rates)
    for r, a in zip(r_mem, a_mem):
        print(f"  dV/dt = {r:6.3f}  →  area = {a:.6e}")

    # ── Power-law fits ──────────────────────────────────────────────────
    C_myc, alpha_myc, r2_myc = fit_power_law(r_myc, a_myc)
    print(f"\nMycelium:   A = {C_myc:.4e} · rate^({-alpha_myc:.4f})   R² = {r2_myc:.4f}")

    C_mem, alpha_mem, r2_mem = fit_power_law(r_mem, a_mem)
    print(f"Memristor:  A = {C_mem:.4e} · rate^({-alpha_mem:.4f})   R² = {r2_mem:.4f}")

    # ── Bootstrap CI for α ──────────────────────────────────────────────
    ci_low_myc, ci_high_myc, alpha_samples_myc = bootstrap_ci(r_myc, a_myc)
    ci_low_mem, ci_high_mem, alpha_samples_mem = bootstrap_ci(r_mem, a_mem)

    print(f"\nMycelium α  95 % CI:  [{ci_low_myc:.4f}, {ci_high_myc:.4f}]")
    print(f"Memristor α 95 % CI:  [{ci_low_mem:.4f}, {ci_high_mem:.4f}]")

    # ── Pearson correlation of log-areas ────────────────────────────────
    log_a_myc = np.log(a_myc)
    log_a_mem = np.log(a_mem)
    pearson_r, pearson_p = stats.pearsonr(log_a_myc, log_a_mem)
    print(f"\nPearson r(log A_myc, log A_mem) = {pearson_r:.6f}  (p = {pearson_p:.2e})")

    # ── 3-panel figure ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(
        "Bridge Hypothesis: Mycelium AP ~ Memristor Switching\n"
        "A ∝ (dV/dt)^(-α)   (shared power-law scaling A ∝ rate⁻α)",
        fontsize=14, y=1.02,
    )

    # Panel A — Mycelium I-V loops (last 4 rates for readability)
    ax = axes[0]
    n_plot = min(4, len(iv_myc))
    step = max(1, len(iv_myc) // n_plot)
    for idx in range(0, len(iv_myc), step):
        Vc, Ic, label = iv_myc[idx]
        ax.plot(Vc, Ic, lw=1.2, label=label.split("=")[1].split(" ")[0] + " V/s")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (arb.)")
    ax.set_title("A  Mycelium Ca²⁺ AP  (I–V loops)")
    ax.legend(fontsize=7, title="dV/dt")
    ax.grid(True, alpha=0.3)

    # Panel B — Memristor I-V loops
    ax = axes[1]
    for idx in range(0, len(iv_mem), step):
        Vc, Ic, label = iv_mem[idx]
        ax.plot(Vc, Ic, lw=1.2, label=label.split("=")[1].split(" ")[0] + " V/s")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (A)")
    ax.set_title("B  HP Memristor  (I–V loops)")
    ax.legend(fontsize=7, title="dV/dt")
    ax.grid(True, alpha=0.3)

    # Panel C — Scaling correlation: log-area vs log-rate
    ax = axes[2]
    ax.loglog(r_myc, a_myc, "o-", color="#2c7bb6", lw=1.8, ms=6, label="Mycelium")
    ax.loglog(r_mem, a_mem, "s-", color="#d7191c", lw=1.8, ms=6, label="Memristor")

    # Power-law fit lines
    rate_fit = np.logspace(np.log10(DV_DT_MIN), np.log10(DV_DT_MAX), 200)
    ax.loglog(rate_fit, power_law(rate_fit, C_myc, alpha_myc),
              "--", color="#2c7bb6", alpha=0.5, lw=1,
              label=rf"Mycel fit: a={alpha_myc:.3f}")
    ax.loglog(rate_fit, power_law(rate_fit, C_mem, alpha_mem),
              "--", color="#d7191c", alpha=0.5, lw=1,
              label=rf"Mem fit: a={alpha_mem:.3f}")

    # Annotate Pearson r
    ax.text(0.05, 0.05,
            rf"Pearson $r = {pearson_r:.4f}$" + "\n" + rf"$p = {pearson_p:.2e}$",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    ax.set_xlabel(r"dV/dt  (V/s)")
    ax.set_ylabel("Hysteresis Area  $A$")
    ax.set_title("C  Scaling:  A  =  C · (dV/dt)^(-a)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "plot.png")
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {plot_path}")

    # ── Save results.json ───────────────────────────────────────────────
    results = {
        "bridge_hypothesis": "Mycelium action potentials ≈ Memristor switching dynamics",
        "description": (
            "Both systems are governed by voltage-driven slow state variables "
            "that modulate conductance, producing pinched I-V hysteresis loops "
            "whose area scales as A ∝ (dV/dt)^(-α)."
        ),
        "parameters": {
            "rates_log10": {
                "min": DV_DT_MIN,
                "max": DV_DT_MAX,
                "n": N_RATES,
            },
            "mycelium": {
                "g_max": myc.g_max,
                "V_half": myc.V_half,
                "V_slope": myc.V_slope,
                "tau_w": myc.tau_w,
            },
            "memristor": {
                "R_on": mem.R_on,
                "R_off": mem.R_off,
                "k": mem.k,
                "x0": mem.x0,
            },
        },
        "sweep": {
            "dVdt_Vs": rates.tolist(),
        },
        "mycelium": {
            "areas": a_myc.tolist(),
            "fit": {
                "C": C_myc,
                "alpha": alpha_myc,
                "R_squared": r2_myc,
                "bootstrap_95_ci": [ci_low_myc, ci_high_myc],
            },
        },
        "memristor": {
            "areas": a_mem.tolist(),
            "fit": {
                "C": C_mem,
                "alpha": alpha_mem,
                "R_squared": r2_mem,
                "bootstrap_95_ci": [ci_low_mem, ci_high_mem],
            },
        },
        "cross_correlation": {
            "pearson_r": pearson_r,
            "p_value": pearson_p,
            "n_samples": N_RATES,
        },
        "files": {
            "simulators": "simulators.py",
            "analysis": "analyze.py",
            "results": "results.json",
            "plot": "plot.png",
        },
    }

    results_path = os.path.join(OUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {results_path}")

    # ── Final summary ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  BRIDGE HYPOTHESIS TEST SUMMARY")
    print("=" * 65)
    print(f"  Mycelium α  = {alpha_myc:.4f}  [{ci_low_myc:.4f}, {ci_high_myc:.4f}]  R²={r2_myc:.4f}")
    print(f"  Memristor α = {alpha_mem:.4f}  [{ci_low_mem:.4f}, {ci_high_mem:.4f}]  R²={r2_mem:.4f}")
    print(f"  α overlap?  {ci_low_myc <= ci_high_mem and ci_low_mem <= ci_high_myc}")
    print(f"  Pearson r(log A) = {pearson_r:.4f}  (p={pearson_p:.2e})")
    print("=" * 65)


if __name__ == "__main__":
    main()