"""
simulators.py — Dual simulators for the bridge hypothesis test:
  Mycelium action potentials ≈ Memristor switching dynamics

Both systems share a core dynamical motif:
  voltage → drives a slow state variable → modulates conductance → pinched I-V hysteresis
"""

import numpy as np
from scipy import signal


class MyceliumSimulator:
    """
    Simplified Hodgkin-Huxley–style model for fungal calcium action potentials.

    Voltage-clamp mode: V(t) is forced (triangle wave), and a slow calcium-gating
    variable w ∈ [0,1] evolves as:
        dw/dt = (w_inf(V) - w) / τ_w
        w_inf(V) = 0.5 * (1 + tanh((V - V_half) / V_slope))

    Measured current:  I = g_max · w · V   (pinched at origin by construction)

    Hysteresis arises because w lags behind the driving voltage — the same
    state-dependent conductance motif found in memristors.
    """

    def __init__(self, g_max=1.0, V_half=0.0, V_slope=0.25, tau_w=8.0):
        self.g_max = g_max        # max conductance (mS, normalised)
        self.V_half = V_half      # half-activation potential (V)
        self.V_slope = V_slope    # slope of activation curve (V)
        self.tau_w = tau_w        # slow gating time constant (s)

    def w_inf(self, V):
        """Steady-state calcium gating variable."""
        return 0.5 * (1.0 + np.tanh((V - self.V_half) / self.V_slope))

    def simulate(self, V_amp, dVdt, n_cycles=3, dt=1e-4):
        """
        Run a voltage-clamp simulation with a triangle-wave drive.

        Parameters
        ----------
        V_amp : float       amplitude of triangle wave (V)
        dVdt  : float       maximum ramp rate (V/s) — the shared parameter
        n_cycles : int      number of drive cycles (last cycle used for analysis)
        dt    : float       integration timestep (s)

        Returns
        -------
        t : ndarray         time vector (s)
        V : ndarray         applied voltage (V)
        I : ndarray         measured current (arb. units)
        freq : float        drive frequency (Hz)
        """
        # Triangle-wave frequency from max slope:  dV/dt = 4·V_amp·f
        freq = dVdt / (4.0 * V_amp)
        T = 1.0 / freq
        n_steps = int(np.ceil(n_cycles * T / dt))
        t = np.arange(n_steps) * dt

        # Triangle wave — symmetric ramp up/down, amplitude V_amp
        V = V_amp * signal.sawtooth(2.0 * np.pi * freq * t, width=0.5)

        # Integrate state variable (forward Euler)
        w = self.w_inf(V[0])
        I = np.empty_like(t)
        for i in range(n_steps):
            if i > 0:
                dw = (self.w_inf(V[i - 1]) - w) / self.tau_w
                w += dw * dt
                w = np.clip(w, 0.0, 1.0)
            I[i] = self.g_max * w * V[i]

        return t, V, I, freq

    @staticmethod
    def hysteresis_area(V, I):
        """
        Signed area of I–V loop via Green's theorem (shoelace / trapezoidal).

        For a pinched hysteresis loop traced once per cycle, np.trapz(I, V)
        gives the total enclosed area (both lobes contribute the same sign).
        """
        return float(np.trapezoid(I, V))


class MemristorSimulator:
    """
    HP TiO₂ memristor model (Strukov et al., Nature 2008).

    State variable x = w/D ∈ [0,1] (normalised doped-region width):
        dx/dt = k · i(t)          where  k = μ_v · R_on / D²

    Resistance:  R(x) = R_on·x + R_off·(1-x)
    Current:     i = v / R(x)

    The same qualitative motif: voltage drives a slow state → modulates
    conductance → pinched hysteresis loop.
    """

    def __init__(self, R_on=100, R_off=10000, k=200.0, x0=0.3):
        self.R_on = R_on      # doped-state resistance (Ω)
        self.R_off = R_off    # undoped-state resistance (Ω)
        self.k = k            # drift-rate constant (m²/(V·s) collapsed into 1/(C·Ω))
        self.x0 = x0          # initial normalised dopant position

    def resistance(self, x):
        return self.R_on * x + self.R_off * (1.0 - x)

    def simulate(self, V_amp, dVdt, n_cycles=3, dt=1e-4):
        """
        Identical interface to MyceliumSimulator.simulate.

        Parameters
        ----------
        V_amp : float       amplitude of triangle wave (V)
        dVdt  : float       maximum ramp rate (V/s)
        n_cycles : int      number of drive cycles
        dt    : float       integration timestep (s)

        Returns
        -------
        t, V, I, freq
        """
        freq = dVdt / (4.0 * V_amp)
        T = 1.0 / freq
        n_steps = int(np.ceil(n_cycles * T / dt))
        t = np.arange(n_steps) * dt

        V = V_amp * signal.sawtooth(2.0 * np.pi * freq * t, width=0.5)

        x = self.x0
        I = np.empty_like(t)
        for i in range(n_steps):
            R = self.resistance(x)
            I[i] = V[i] / R
            if i > 0:
                dx = self.k * I[i - 1]
                x += dx * dt
                x = np.clip(x, 0.0, 1.0)

        return t, V, I, freq

    @staticmethod
    def hysteresis_area(V, I):
        return float(np.trapezoid(I, V))