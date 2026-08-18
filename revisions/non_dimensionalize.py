#!/usr/bin/env python3
"""
non_dimensionalize.py — R2-c4: non-dimensionalization of both models.

Reviewer #2 point 4: "The HP memristor model utilizes physical units (Ohms),
whereas the fungal model uses normalized/arbitrary conductance. Normalizing both
models into non-dimensional units would make the Pearson correlation metrics
(r > 0.98) more mathematically formal."

Two claims addressed here:
  (A) Derive the non-dimensional (dimensionless) form of BOTH models.
  (B) Show the Pearson r on log-areas is INVARIANT under arbitrary
      (multiplicative) rescaling of current — i.e. r > 0.98 does not depend on
      the conductivity unit convention (S, arb., A/V). Compute r under several
      conductance scales to demonstrate.

Claim A — dimensionless rescaling:
  Mycelium:  I = g_max · w · V. Let V' = V / V_ref, I' = I / I_ref.
             The loop pinching + power-law exponent alpha are unchanged by a
             constant factor because area scales as A = ∮ I dV
             = (I_ref · V_ref) · ∮ I' dV'. So alpha is invariant.
  Memristor: I = V / R(x), R in Ohms. Same argument: A = I_ref·V_ref · A'.

The shared quantity is the exponent alpha (dimensionless) and the Pearson
correlation r on log(A) (invariant to scaling + log). We demonstrate both.
"""

import json, os, sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulators import MyceliumSimulator, MemristorSimulator

OUT = os.path.expanduser("/home/ubuntu/mycelium-revision/experiment")


def sweep_areas(sim, rates, V_amp=1.0, n_cycles=3):
    """Return areas at each ramp rate (reuses simulators.extract_last_cycle logic)."""
    from analyze import extract_last_cycle
    areas = []
    for dvdt in rates:
        t, V, I, f = sim.simulate(V_amp, dvdt, n_cycles=n_cycles)
        Vc, Ic = extract_last_cycle(t, V, I)
        areas.append(abs(sim.hysteresis_area(Vc, Ic)))
    return np.array(areas)


def fit_alpha(rate, area):
    logx, logy = np.log(rate), np.log(area)
    sl, ic, r, p, se = stats.linregress(logx, logy)
    return -sl  # A = C·rate^(-alpha)


def main():
    os.makedirs(OUT, exist_ok=True)
    np.random.seed(7)
    rates = np.logspace(np.log10(0.1), np.log10(10.0), 8)

    mycel = MyceliumSimulator(tau_w=8.0)
    mem = MemristorSimulator()

    a_myc_orig = sweep_areas(mycel, rates)
    a_mem_orig = sweep_areas(mem, rates)

    alpha_myc = fit_alpha(rates, a_myc_orig)
    alpha_mem = fit_alpha(rates, a_mem_orig)
    # r on log-areas (the published metric)
    r_pub, p_pub = stats.pearsonr(np.log(a_myc_orig), np.log(a_mem_orig))
    r_scaled, _ = stats.pearsonr(np.log(a_myc_orig), np.log(np.abs(a_mem_orig)))
    r_scaled2, _ = stats.pearsonr(np.log(np.abs(a_myc_orig)), np.log(a_mem_orig))

    # --- (A) dimensionless form derivations ---
    derivations = {
        "mycelium": (
            "I = g_max * w(V) * V   ;   V' = V/V_ref, I' = I/(g_max*V_ref)  ->  "
            "I' = w(v) * v,  dw/ds = (w_inf(v)-w)/tau_w'   where s = t/tau_w, "
            "tau_w' = 1.  Dimensionless hysteresis area A' = A/(g_max*V_ref^2). "
            "No physical constant remains; alpha identical."
        ),
        "memristor": (
            "I = v / R(x), R(x)=R_on*x+R_off*(1-x). Set r=R_off/R_on, "
            "v'=v/V_ref, i'=i*R_on/V_ref -> i' = v'/(x + r*(1-x)). "
            "dx/ds = k' * i', k'=k*V_ref*R_on. Only RATIO r=R_off/R_on matters; "
            "absolute Ohms vanish. Dimensionless loop area A' = A*R_on/V_ref^2. "
            "alpha identical."
        ),
    }

    # --- (B) scale-invariance of Pearson r ---
    # Multiply the memristor current by arbitrary conductance scales; watch r hold.
    scaling_checks = []
    for scale in [1.0, 1e-3, 1e6, 1e-9]:
        a_scaled = a_mem_orig * scale
        if np.all(a_scaled > 0):
            r, p = stats.pearsonr(np.log(a_myc_orig), np.log(a_scaled))
            scaling_checks.append({"current_scale": scale, "pearson_r": r})
        else:
            # areas positive; log fine; guard anyway
            r, p = stats.pearsonr(np.log(np.abs(a_myc_orig)), np.log(np.abs(a_scaled)))
            scaling_checks.append({"current_scale": scale, "pearson_r": r})

    r_full, p_full = stats.pearsonr(np.log(a_myc_orig), np.log(a_mem_orig))
    result = {
        "reviewer_point": "R2-c4 non-dimensionalization",
        "alpha_mycelium": alpha_myc,
        "alpha_memristor": alpha_mem,
        "pearson_r_log_areas_original_units": r_full,
        "pearson_p": p_full,
        "pearson_r_under_arbitrary_current_scale": scaling_checks,
        "invariance_conclusion": (
            "Pearson r on log-transformed hysteresis areas is invariant to arbitrary "
            "multiplicative rescaling of current (a conductance-unit change maps to an "
            "additive shift in log-space, which Pearson r ignores). "
            f"r={r_full:.4f} is therefore unit-independent."
        ),
        "dimensionless_derivations": derivations,
        "note": "Both models reduce to dimensionless forms governed only by ratios "
                "(R_off/R_on) or unit-removed constants; alpha (exponent) is a pure number.",
        "files": ["non_dimensionalize.py", "non_dimensionalize_results.json"],
    }

    path = os.path.join(OUT, "non_dimensionalize_results.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 70)
    print("  NON-DIMENSIONALIZATION (R2-c4) — VERIFICATION")
    print("=" * 70)
    print(f"  alpha_mycelium  = {alpha_myc:.4f}")
    print(f"  alpha_memristor = {alpha_mem:.4f}")
    print(f"  Pearson r(log A) original units = {r_full:.6f}  (p={p_full:.2e})")
    print("\n  r under arbitrary current-scale rescaling:")
    for c in scaling_checks:
        print(f"    scale={c['current_scale']:>8.0e}  ->  r={c['pearson_r']:.6f}")
    print("\n  dimensionless forms:")
    for k, v in derivations.items():
        print(f"    [{k}] {v[:110]}")
    print("\n  CONCLUSION: r is unit-independent; alpha is a pure dimensionless exponent.")
    print("=" * 70)
    print(f"  saved -> {path}")


if __name__ == "__main__":
    main()