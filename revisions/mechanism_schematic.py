#!/usr/bin/env python3
"""
mechanism_schematic.py — R3-c1: schematic mechanism/setup illustration.

Reviewer #3 point 1: "the authors are advised to supplement the manuscript with
a schematic diagram of the experimental setup or a mechanism illustration to
further clarify the experimental context."

This produces a clean 3-panel vector schematic:
  (a) Experimental voltage-clamp setup on a fungal hypha/net — triangle-wave
      drive V(t), electrodes, ion channels in the membrane.
  (b) Channel-level mechanism: voltage-gated Ca²⁺ channel in the lipid bilayer,
      w (channel availability) responding to V with lag tau_w; currents through
      open channels. Inset: Boltzmann steady-state w_inf(V).
  (c) Resulting memristive I-V: pinched hysteresis loop at the origin
      (+ arrow to show the loop), i.e. the signature readout.

Pure matplotlib vector drawing (no raster), high DPI PNG + PDF.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch, Rectangle
from matplotlib import patches

OUT = os.path.expanduser("/home/ubuntu/mycelium-revision/manuscript")
os.makedirs(OUT, exist_ok=True)


def w_inf(v, V_half=0.0, V_slope=0.25):
    return 0.5*(1+np.tanh((v-V_half)/V_slope))


def main():
    fig = plt.figure(figsize=(15, 5.2))
    plt.rcParams.update({"font.size": 10})

    # ================= Panel (a): experimental setup =================
    ax = fig.add_axes([0.03, 0.12, 0.31, 0.78])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    # hypha (membrane tube)
    hypha = FancyBboxPatch((2.2, 2.5), 5.6, 5.0,
                           boxstyle="round,pad=0.02,rounding_size=1.2",
                           linewidth=2, edgecolor='#2c7bb6', facecolor='#dfeaf5')
    ax.add_patch(hypha)
    ax.text(5.0, 7.9, "fungal hypha / mycelium network", ha='center',
            fontsize=9, color='#1f4e79')
    # membrane
    ax.text(5.0, 5.0, "lipid bilayer\nvoltage-gated Ca$^{2+}$ channels",
            ha='center', va='center', fontsize=8, color='#333')
    # electrodes
    el1 = Rectangle((0.4, 3.6), 1.6, 0.9, facecolor='#d7191c', edgecolor='k')
    ax.add_patch(el1)
    ax.text(1.2, 4.75, "working\nelectrode", ha='center', fontsize=7)
    el2 = Rectangle((8.0, 3.6), 1.6, 0.9, facecolor='#2b8a3e', edgecolor='k')
    ax.add_patch(el2)
    ax.text(8.8, 4.75, "reference\nelectrode", ha='center', fontsize=7)
    # wire + drive
    FancyArrowPatch = patches.FancyArrowPatch
    ax.add_patch(FancyArrowPatch((0.4, 4.05), (2.2, 4.05),
                 arrowstyle='-', color='k', lw=1.5))
    ax.add_patch(FancyArrowPatch((7.8, 4.05), (9.6, 4.05),
                 arrowstyle='-', color='k', lw=1.5))
    # voltage source / triangle
    ax.annotate(r"triangle-wave $V(t)$ = $\pm V_0$, ramp  $dV/dt$",
                xy=(0.6, 2.2), fontsize=8, ha='center', color='#d7191c')
    # ions
    for (x, y) in [(3.6, 3.4), (4.6, 3.2), (5.8, 3.5), (6.6, 3.3)]:
        ax.text(x, y, "Ca$^{2+}$", fontsize=7, color='#b07a00',
                ha='center', va='center', rotation=20)
    ax.set_title("(a) Voltage-clamp measurement", fontsize=11)

    # ================= Panel (b): channel mechanism =================
    ax = fig.add_axes([0.37, 0.12, 0.30, 0.78])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    # membrane band
    ax.add_patch(Rectangle((0.5, 5.2), 9.0, 1.6, facecolor='#f5e8c8',
                           edgecolor='#b0822a', lw=2))
    ax.text(5.0, 5.0, "cell membrane", ha='center', fontsize=8, va='top',
            color='#7a5a1a')
    # channel
    ch = FancyBboxPatch((3.4, 4.0), 3.2, 4.0, boxstyle="round,pad=0.02",
                        linewidth=2, edgecolor='#444e65', facecolor='#e6e9ef')
    ax.add_patch(ch)
    ax.text(5.0, 8.2, "voltage-gated\nCa$^{2+}$ channel", ha='center',
            fontsize=8, color='#222')
    # state variable
    ax.text(5.0, 6.0, r"state  $w$ (channel availability)", ha='center',
            fontsize=8, va='center')
    # Boltzmann inset (moved up-right of channel for breathing room)
    vin = np.linspace(-0.8, 0.8, 100)
    ins = fig.add_axes([0.585, 0.30, 0.09, 0.20])
    ins.plot(vin, w_inf(vin), color='#2c7bb6', lw=2)
    ins.axhline(0.5, color='gray', ls=':', lw=0.8)
    ins.axvline(0, color='gray', ls=':', lw=0.8)
    ins.set_title(r"$w_\infty(V)$", fontsize=8)
    ins.set_xlabel("V", fontsize=7); ins.set_ylabel(r"$w_\infty$", fontsize=7)
    ins.tick_params(labelsize=6)
    # kinetic equation
    ax.text(5.0, 2.6, r"$\dfrac{dw}{dt}=\dfrac{w_\infty(V)-w}{\tau_w}$",
            ha='center', fontsize=13, color='#444e65')
    ax.text(5.0, 1.55,
            r"channel opens ($w\!\uparrow$) on depolarization," +
            "\nbut lags $V$ by $\\tau_w$ — history-dependent $g$",
            ha='center', fontsize=8, color='#333')
    # current
    ax.annotate(r"$I = g_{max}\, w\, V$", xy=(7.8, 8.6), fontsize=10,
                color='#d7191c', ha='center')
    ax.set_title("(b) Ca$^{2+}$-gating mechanism", fontsize=11)

    # ================= Panel (c): memristive readout =================
    ax = fig.add_axes([0.71, 0.12, 0.26, 0.78])
    # compute a pinched loop
    V_amp, dvdt, tau = 0.11, 0.15, 8.0
    t = np.linspace(0, 3* (4*V_amp/dvdt), 6000)
    V = V_amp*np.sin(2*np.pi*(dvdt/(4*V_amp))*t)
    w = w_inf(V[0]); I = np.empty_like(V)
    dt = t[1]-t[0]
    for i in range(len(V)):
        I[i] = w*V[i]
        if i>0:
            w += (w_inf(V[i-1])-w)/tau*dt; w=np.clip(w,0,1)
    # last cycle
    si = np.where(np.diff(np.signbit(V).astype(int)))[0]
    Vc, Ic = V[si[-4]:si[-1]+1], I[si[-4]:si[-1]+1]
    ax.plot(Vc, Ic, color='#2c7bb6', lw=2)
    ax.axhline(0, color='gray', lw=0.6, ls='--')
    ax.axvline(0, color='gray', lw=0.6, ls='--')
    ax.annotate('', xy=(0.008, 0.0045), xytext=(-0.02, -0.003),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.9))
    ax.annotate('', xy=(-0.008, -0.0045), xytext=(0.02, 0.003),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.9))
    ax.set_xlabel("Voltage $V$"); ax.set_ylabel("Current $I$")
    ax.set_title("(c) Pinched hysteresis loop", fontsize=11)
    ax.text(0.05, 0.8, "pinched\nat origin\n(memristive)",
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', fc='white', ec='#888', alpha=0.9,
                      lw=0.7))
    ax.grid(True, alpha=0.25, ls=':')

    fig.text(0.5, 0.955,
             "Ca$^{2+}$ channel gating as a memristive mechanism in fungal mycelium",
             ha='center', fontsize=12, fontweight='bold')
    png = os.path.join(OUT, "fig3_mechanism.png")
    pdf = os.path.join(OUT, "fig3_mechanism.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png} (300dpi) + {pdf} (vector)")
    print("Panels: (a) voltage-clamp setup  (b) Ca-gating mechanism + Boltzmann "
          "(c) pinched memristive loop")


if __name__ == "__main__":
    main()