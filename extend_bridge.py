#!/usr/bin/env python3
"""Fast parameter sensitivity: fewer rates, fewer taus, vectorized."""
import json, sys, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulators import MyceliumSimulator, MemristorSimulator

OUT_DIR = os.path.expanduser("~/citations-needed/experiments/mycelium-memristor")
V_AMP = 1.0
N_CYCLES = 2  # only 2 cycles instead of 3

def fit_power_law(rate_vals, area_vals):
    log_x = np.log(rate_vals)
    log_y = np.log(area_vals)
    slope, intercept, _, _, _ = stats.linregress(log_x, log_y)
    r = log_y - (intercept + slope * log_x)
    ss_res = np.sum(r**2); ss_tot = np.sum((log_y - np.mean(log_y))**2)
    return np.exp(intercept), -slope, 1.0 - ss_res/ss_tot if ss_tot > 0 else 0

def extract_last_cycle(t, V, I):
    signs = np.signbit(V)
    idx = np.where(np.diff(signs.astype(int)))[0]
    if len(idx) < 4:
        n = len(V)
        return V[-n//3:], I[-n//3:]
    return V[idx[-4]:idx[-1]+1], I[idx[-4]:idx[-1]+1]

def sweep_areas(sim, dvdt_list):
    areas = []
    for dvdt in dvdt_list:
        t, V, I, _ = sim.simulate(V_AMP, dvdt, n_cycles=N_CYCLES)
        Vc, Ic = extract_last_cycle(t, V, I)
        areas.append(abs(sim.hysteresis_area(Vc, Ic)))
    return np.array(areas)

# Use 6 rates, not 8 — cuts ~30% of runtime
rates = np.array([0.2, 0.5, 1.0, 2.5, 5.0, 10.0])

# ── A. Parameter sensitivity ──
print("Sweeping τ_w...")
tau_vals = np.array([1, 2, 4, 8, 12, 20, 30, 50])
alpha_list = []
for tau in tau_vals:
    myc = MyceliumSimulator(tau_w=tau)
    a = sweep_areas(myc, rates)
    _, alpha, r2 = fit_power_law(rates, a)
    alpha_list.append(alpha)
    print(f"  τ_w={tau:2d}s  α={alpha:.4f}  R²={r2:.4f}")

# Fewer fine points
tau_fine = np.linspace(1, 50, 50)
alpha_fine = []
for tau in tau_fine:
    myc = MyceliumSimulator(tau_w=tau)
    a = sweep_areas(myc, rates)
    _, alpha, _ = fit_power_law(rates, a)
    alpha_fine.append(alpha)
alpha_fine = np.array(alpha_fine)

def find_cross(tau_arr, alpha_arr, target):
    for i in range(len(alpha_arr)-1):
        if alpha_arr[i] < target <= alpha_arr[i+1]:
            f = (target - alpha_arr[i]) / (alpha_arr[i+1] - alpha_arr[i])
            return float(tau_arr[i] + f * (tau_arr[i+1] - tau_arr[i]))
    return None

t08 = find_cross(tau_fine, alpha_fine, 0.8)
t10 = find_cross(tau_fine, alpha_fine, 1.0)
print(f"  α=0.8 at τ_w≈{t08:.0f}s" if t08 else "  α never reaches 0.8")
print(f"  α=1.0 at τ_w≈{t10:.0f}s" if t10 else "  α never reaches 1.0")

# ── B. Reference memristor ──
mem = MemristorSimulator()
a_mem = sweep_areas(mem, rates)
C_mem, alpha_mem, r2_mem = fit_power_law(rates, a_mem)

# ── C. Comparison at key τ_w values ──
myc8 = MyceliumSimulator(tau_w=8.0)
myc32 = MyceliumSimulator(tau_w=32.0)
a8 = sweep_areas(myc8, rates)
a32 = sweep_areas(myc32, rates)
_, a8_alpha, r2_8 = fit_power_law(rates, a8)
_, a32_alpha, r2_32 = fit_power_law(rates, a32)
p8_m, _ = stats.pearsonr(np.log(a8), np.log(a_mem))
p32_m, _ = stats.pearsonr(np.log(a32), np.log(a_mem))

# ── D. Figure ──
print("\nGenerating figure...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle("Fungal Mycelium as a Biological Memristor: Computational Validation",
             fontsize=14, y=1.02)

ax = axes[0]
ax.plot(tau_fine, alpha_fine, '-', color='#2c7bb6', lw=2)
ax.axhline(alpha_mem, color='#d7191c', lw=1.5, ls='--', label=f'Memristor α={alpha_mem:.3f}')
ax.axhline(1.0, color='gray', lw=0.8, ls=':', label='α=1.0')
ax.plot(tau_vals, alpha_list, 'o', color='#2c7bb6', ms=5)
if t08: ax.axvline(t08, color='green', lw=0.8, ls='--', alpha=0.6)
if t10: ax.axvline(t10, color='purple', lw=0.8, ls='--', alpha=0.6)
ax.set_xlabel(r'Gating time constant $\tau_w$ (s)')
ax.set_ylabel(r'Scaling exponent $\alpha$')
ax.set_title('A  Mycelium α(τ_w) crosses memristor α')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.loglog(rates, a8, 'o-', color='#2c7bb6', lw=1.8, ms=6, label=f'Single hypha τ_w=8s (α={a8_alpha:.3f})')
ax.loglog(rates, a32, 's-', color='#0571b0', lw=1.8, ms=6, label=f'Single hypha τ_w=32s (α={a32_alpha:.3f})')
ax.loglog(rates, a_mem, 'D-', color='#d7191c', lw=1.8, ms=6, label=f'HP memristor (α={alpha_mem:.3f})')
rf = np.logspace(np.log10(rates[0])-0.1, np.log10(rates[-1])+0.1, 100)
def pl(x, C, a): return C * x**(-a)
ax.loglog(rf, pl(rf, *fit_power_law(rates, a8)[:2]), '--', color='#2c7bb6', alpha=0.3)
ax.loglog(rf, pl(rf, *fit_power_law(rates, a32)[:2]), '--', color='#0571b0', alpha=0.3)
ax.loglog(rf, pl(rf, C_mem, alpha_mem), '--', color='#d7191c', alpha=0.3)
ax.text(0.05, 0.05, f'Single(8s) v Mem r={p8_m:.4f}\nSingle(32s) v Mem r={p32_m:.4f}',
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
ax.set_xlabel(r'dV/dt (V/s)'); ax.set_ylabel('Hysteresis Area A')
ax.set_title('B  Power-law scaling: A = C·(dV/dt)^(-α)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3, which='both')

ax = axes[2]; ax.axis('off')
tbl = ax.table(cellText=[
    [f'Single hypha (τ_w=8s)', f'{a8_alpha:.3f}', f'{r2_8:.4f}', f'{p8_m:.4f}'],
    [f'Single hypha (τ_w=32s)', f'{a32_alpha:.3f}', f'{r2_32:.4f}', f'{p32_m:.4f}'],
    [f'HP memristor', f'{alpha_mem:.3f}', f'{r2_mem:.4f}', '—'],
], colLabels=['System', 'α', 'R²', 'r vs memristor'],
   loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.8)
ax.set_title('C  Quantitative Summary', fontsize=11, pad=12)

plt.tight_layout()
p = os.path.join(OUT_DIR, "plot_extended.png")
fig.savefig(p, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Plot → {p}")

json.dump({
    "parameter_sensitivity": {
        "tau_vals_s": tau_vals.tolist(), "alpha_vals": alpha_list,
        "tau_at_alpha_08": t08, "tau_at_alpha_10": t10,
        "finding": f"α crosses 0.8 at τ_w≈{t08:.0f}s, approaches 1.0 at τ_w≈{t10 or '>50'}s"
    },
    "reference": {
        "single_tau8_alpha": a8_alpha, "single_tau8_r2": float(r2_8), "single_tau8_r_vs_mem": p8_m,
        "single_tau32_alpha": a32_alpha, "single_tau32_r2": float(r2_32), "single_tau32_r_vs_mem": p32_m,
        "memristor_alpha": alpha_mem, "memristor_r2": float(r2_mem),
    }
}, open(os.path.join(OUT_DIR, "results_extended.json"), "w"), indent=2)

print("\n=== SUMMARY ===")
print(f"  α range: τ_w=1 → {alpha_list[0]:.3f}, τ_w=50 → {alpha_list[-1]:.3f}")
print(f"  α=0.8 at τ_w≈{t08:.0f}s, α=1.0 at τ_w≈{t10 or '>50'}s")
print(f"  τ_w=8:  α={a8_alpha:.3f}, r vs mem={p8_m:.4f}")
print(f"  τ_w=32: α={a32_alpha:.3f}, r vs mem={p32_m:.4f}")
print("=== DONE ===")