# Mycelium Memristor — Ca²⁺ Channel Hysteresis Model

**Calcium Action Potential Hysteresis in Fungal Mycelium as a Biological Memristor: A Computational Validation**

Computational model and analysis code supporting the submission to *Chaos: An Interdisciplinary Journal of Nonlinear Science* (manuscript 26-AR-01834).

## Overview

This repository implements a biophysical model of voltage-gated calcium (Ca²⁺) channel hysteresis in fungal mycelium and compares its dynamical scaling behavior to the canonical HP TiO₂ memristor model. The key finding is that Ca²⁺ channel gating hysteresis — a slow, voltage-dependent recovery process — produces pinched current-voltage loops with power-law area scaling $A \propto (dV/dt)^{-\alpha}$ that closely matches solid-state memristor dynamics.

## Repository Structure

```
├── simulators.py          # Ca²⁺ AP and HP memristor models
├── analyze.py             # Parameter sensitivity sweep (τ_w from 1–50 s)
├── extend_bridge.py       # 5-node ring-coupled network + LaRocco-comparable simulation
├── generate_figure.py     # Generate main figure panels
├── fit_experimental.py    # Legacy fit script (see paper §3.6 for current approach)
├── results.json           # Gold bridge numerical results
├── results_extended.json  # Sweep and network numerical results
├── fit_experimental.json  # LaRocco-comparable simulation data
├── requirements.txt       # Python dependencies
├── revisions/             # Analyses added during the reviewer-revision round
│   ├── multi_state_model.py         # Two-gate model (activation m + inactivation h)
│   ├── multi_state_results.json    # α(τ_w) sweep for single- vs two-gate models
│   ├── multi_state_fit.png         # Figure 1 in the revised manuscript
│   ├── freq_sweep.py               # Band-pass frequency response + high-frequency cutoff
│   ├── freq_sweep_results.json     # A(f) band-pass data
│   ├── freq_sweep.png              # Frequency-dependence figure (rev §4.2)
│   ├── non_dimensionalize.py       # Scale-invariance of Pearson r (rev §2.3)
│   ├── non_dimensionalize_results.json
│   ├── reverse_fit.py              # Reverse-fit attempt / non-identifiability (rev §3.7)
│   ├── reverse_fit_results.json
│   ├── generate_figure1.py         # Clean high-res vector regenerated main figure
│   └── mechanism_schematic.py      # 3-panel mechanism/setup schematic (rev §2, Fig. 2)
└── README.md
```

## Requirements

- Python ≥ 3.9
- NumPy
- SciPy
- Matplotlib

Install with:

```bash
pip install -r requirements.txt
```

## Usage

### Reproduce the main results

```bash
# Run the parameter sensitivity sweep (Table 1)
python3 analyze.py

# Run the gold bridge + network comparison
python3 extend_bridge.py

# Generate the main figure (Fig 1)
python3 generate_figure.py
```

### Model description

The `MyceliumSimulator` class in `simulators.py` implements a Hodgkin-Huxley style gating variable:

- **Fast variable:** Membrane voltage $V$ driven by a triangular/sinusoidal stimulus
- **Slow variable:** Calcium channel inactivation gate $w$ with time constant $\tau_w$ (1–50 s)
- **Conductance:** $g = g_{max} \cdot w \cdot (1 + \tanh((V - V_{half}) / V_{slope})) / 2$

The hysteresis area $A$ is computed as the area enclosed by the I-V loop for each cycle, and the scaling exponent $\alpha$ is extracted from a power-law fit $A = C \cdot (dV/dt)^{-\alpha}$.

## Revision artifacts (reviewer-response round)

The `revisions/` subdirectory holds the additional analyses produced in response
to the reviewer comments on CHA26-AR-01834:

- **`multi_state_model.py`** — a two-gate formulation explicitly separating fast
  activation ($m$) from slow Ca$^{2+}$-dependent inactivation ($h$). Confirms the
  power-law scaling $A \propto (\mathrm{d}V/\mathrm{d}t)^{-\alpha}$ is robust to
  multi-state channel dynamics (revised manuscript, "Robustness to Multi-State
  Channel Dynamics").
- **`freq_sweep.py`** — computes the hysteresis-area frequency response, showing it
  is band-pass with a cutoff $f_c \approx V_s/(4V_0\tau_w)$ well below 1 kHz over the
  biophysical range, and that the Ca²⁺ loop area is negligible at the 5.85 kHz
  switching reported experimentally (revised manuscript, §4.2).
- **`non_dimensionalize.py`** — verifies that the Pearson correlation of
  log-transformed hysteresis areas is invariant under arbitrary conductance
  rescaling (revised manuscript, §2.3 "Scale invariance of the comparison").
- **`reverse_fit.py`** — documents the reverse-fitting attempt against digitized
  published I-V data and the resulting parameter non-identifiability (revised
  manuscript, §3.7 "Inverse parameter fitting").
- **`generate_figure1.py`** — regenerates the main validation figure as clean,
  high-resolution vector graphics.
- **`mechanism_schematic.py`** — generates the 3-panel mechanism/setup schematic
  (revised manuscript, Fig. 2).

## Data

All numerical results are stored as JSON files:

- `results.json` — Single-hypha gold bridge at $\tau_w = 8$ s and $\tau_w = 32$ s, with HP memristor reference
- `results_extended.json` — Full $\tau_w$ sweep (1–50 s), network model, LaRocco-comparable simulation
- `fit_experimental.json` — I-V loop data at LaRocco-comparable conditions

## Citation

If you use this code or data, please cite:

> Scott, C.L. (2026). *Calcium Action Potential Hysteresis in Fungal Mycelium as a Biological Memristor: A Computational Validation.* Chaos: An Interdisciplinary Journal of Nonlinear Science.

## License

MIT License — see [LICENSE](LICENSE).

## Data Availability Statement

As stated in the manuscript: "The simulation code and data are available at https://github.com/aptsystems/mycelium-memristor (DOI will be assigned upon publication). All experiments were run on a standard Linux workstation; no specialized hardware is required to reproduce the results."