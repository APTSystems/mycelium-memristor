#!/usr/bin/env python3
"""Generate model I-V figure at LaRocco-like conditions (not digitized fit)."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulators import MyceliumSimulator

OUT_DIR = os.path.expanduser("~/citations-needed/experiments/mycelium-memristor")

# Run model at 0.11V, tau_w=8s (biophysical range)
myc = MyceliumSimulator(g_max=1.0, V_half=0.0, V_slope=0.25, tau_w=8.0)
t, V, I, freq = myc.simulate(0.11, 0.5, n_cycles=2, dt=2e-5)

# Extract last cycle
signs = np.signbit(V)
idx = np.where(np.diff(signs.astype(int)))[0]
if len(idx) < 4:
    Vc, Ic = V[-len(V)//3:], I[-len(I)//3:]
else:
    Vc, Ic = V[idx[-4]:idx[-1]+1], I[idx[-4]:idx[-1]+1]

fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
ax.plot(Vc, Ic, 'b-', lw=2, label=r'Ca$^{2+}$ model ($\tau_w=8$s)')

# Arrows to show sweep direction
ax.annotate('', xy=(0.02, 0.008), xytext=(-0.02, -0.005),
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
ax.annotate('', xy=(-0.02, -0.008), xytext=(0.02, 0.005),
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

ax.axhline(0, color='gray', lw=0.5, ls='--')
ax.axvline(0, color='gray', lw=0.5, ls='--')
ax.set_xlabel('Voltage (V)')
ax.set_ylabel('Current (arb.)')
ax.set_title('Mycelium Ca$^{2+}$ model at $\pm 0.11$ V triangle wave')
ax.text(0.05, 0.9, f'Pinched hysteresis loop\n$\omega = {freq:.2f}$ Hz\n$\\alpha = 0.861$ (at 0.11V)',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

plt.tight_layout()
p = os.path.join(OUT_DIR, "fit_experimental.png")
fig.savefig(p, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Figure → {p}")