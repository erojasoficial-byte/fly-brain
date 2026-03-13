# Emergent Individuality in Whole-Brain Connectome Simulations of *Drosophila melanogaster*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GPU Accelerated](https://img.shields.io/badge/GPU-CUDA%2012.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![NeuroMechFly](https://img.shields.io/badge/body-NeuroMechFly%20v2-orange.svg)](https://neuromechfly.org/)
[![FlyWire Connectome](https://img.shields.io/badge/brain-FlyWire%20v783-purple.svg)](https://flywire.ai/)
[![bioRxiv](https://img.shields.io/badge/preprint-bioRxiv-B31B1B.svg)](https://doi.org/10.1101/2026.XX.XX.XXXXXX)

> **Enrique Manuel Rojas Aliaga** · Independent Researcher · Lima, Peru · erojasoficial@gmail.com

<div align="center">
  <img src="demo_preview.gif" alt="Embodied Drosophila simulation" width="640">
  <br>
  <em>The fly sees, walks, grooms, and escapes — driven entirely by 138,639 spiking neurons from the FlyWire connectome.</em>
  <br>
  <a href="demo.mp4"><strong>Watch full demo video</strong></a>
</div>

---

## Paper

This repository accompanies the preprint:

> **Rojas Aliaga, E. M.** (2026). *Emergent Individuality and Neural Integration in Whole-Brain Connectome Simulations of Drosophila melanogaster.* bioRxiv. doi: [10.1101/2026.XX.XX.XXXXXX](https://doi.org/10.1101/2026.XX.XX.XXXXXX)

Pre-built PDFs are included in this repository:

| Document | File | Pages |
|---|---|---|
| **English** | [`paper_emergent_individuality.pdf`](paper_emergent_individuality.pdf) | 14 |
| **Spanish** | [`paper_individualidad_emergente_ES.pdf`](paper_individualidad_emergente_ES.pdf) | 14 |

Both papers contain 10 publication-quality figures (300 DPI) generated from the included experimental data.

---

## Abstract

We present the first embodied whole-brain spiking simulation of *Drosophila melanogaster* that combines the complete FlyWire v783 connectome (138,639 neurons; 15,091,983 directed weighted synapses) with biomechanical embodiment, multi-modal sensory processing, Hebbian synaptic plasticity, and multi-theory neural integration metrics. Two flies initialized with identical connectomes develop distinct behavioral profiles (81% vs. 47% escape), different neural integration signatures (CI = 0.221 vs. 0.198), and 76,034 divergent synapses after 24 hours of independent embodied experience — demonstrating that connectome architecture + embodied experience + Hebbian plasticity is sufficient to generate computational individuality from identical initial conditions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BRAIN (GPU / PyTorch)                     │
│  138,639 LIF neurons · 15M synapses · 5 kHz timestep       │
│                                                              │
│  Visual  ─── T1→T2→T3→T4→T5 ─── LC4 ─── GF ──→ DNs       │
│  cortex       (motion detection)   (loom)  (escape)          │
│                                                              │
│  Olfactory ─── ORN→PN→KC→MBON ─── DN turn commands         │
│  Gustatory ─── GRN→SEZ→MN ─── feeding/avoidance            │
│  Somatosensory ─── mechanoreceptors→IN→MN                   │
│                                                              │
│  Hebbian Plasticity: dW = eta*(r_i*r_j) - alpha*W          │
└──────────────────────┬──────────────────────────────────────┘
                       │ DN spike rates (~1,100 descending neurons)
              ┌────────▼────────┐
              │  Brain-Body     │
              │  Bridge         │
              │  DN→drive rates │
              │  mode selection │
              │  (walk/escape/  │
              │   groom/feed/   │
              │   flight)       │
              └────────┬────────┘
                       │ joint torques
┌──────────────────────▼──────────────────────────────────────┐
│              BODY (NeuroMechFly v2 / MuJoCo)                │
│  87 joints · 6 legs · 2 wings · head · abdomen              │
│  Compound eyes (721 ommatidia) · Contact sensors             │
│  Arena: terrain, sky, sunlight, odors, threats               │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/erojasoficial-byte/fly-brain.git
cd fly-brain
git lfs pull   # downloads large data files (~270 MB)
```

### 2. Install Dependencies

```bash
# PyTorch with CUDA 12.1 (adjust for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install flygym mujoco numpy scipy pandas matplotlib fpdf2 pygame pyarrow
```

Or use conda:

```bash
conda env create -f environment.yml
conda activate brain-fly
pip install flygym mujoco fpdf2 pygame
```

### 3. Run

```bash
# Single fly — visual mode with all sensory systems
python fly_embodied.py --visual --monitor --flight --olfactory --gustatory --somatosensory

# Two-fly experiment with neural integration metrics
python two_flies.py --visual

# Headless mode (faster, no display)
python fly_embodied.py --steps 10000
python two_flies.py --steps 500000
```

---

## Reproducing the Paper

All figures and statistics in the paper are generated programmatically from the experimental data included in this repository. No manual editing or cherry-picking was performed.

### Step 1: Verify Data Integrity

The repository includes all raw data from 20 measurement sessions (8 paired two-fly sessions + 3 single-fly baseline sessions + 1 early single-fly session):

```bash
# Check consciousness session data
ls consciousness_history/
# Expected: 20 session directories (3 single + 8 paired × 2 flies + 1 early)

# Check plasticity weight snapshots
ls data/plastic_weights*.pt
# Expected: 3 files (~60 MB each)
```

### Step 2: Regenerate All Figures and the Paper

```bash
# Generate English paper with 10 figures (300 DPI)
python generate_paper.py
# Output: paper_emergent_individuality.pdf + paper_figures/*.png

# Generate Spanish paper (reuses same figures)
python generate_paper_es.py
# Output: paper_individualidad_emergente_ES.pdf
```

This regenerates every figure from raw data:

| Figure | Script section | Source data |
|---|---|---|
| Fig. 1 — Architecture | `generate_paper.py` | Schematic (programmatic) |
| Fig. 2 — CI timelines | `generate_paper.py` | `consciousness_history/session_*_fly{0,1}/` |
| Fig. 3 — CI by behavioral mode | `generate_paper.py` | Same sessions |
| Fig. 4 — Integration components | `generate_paper.py` | Same sessions |
| Fig. 5 — Cross-session evolution | `generate_paper.py` | All 8 paired sessions |
| Fig. 6 — CI distributions | `generate_paper.py` | Same sessions |
| Fig. 7 — Behavior montage | `generate_paper.py` | Rendered frames |
| Fig. 8 — Plasticity analysis | `generate_paper.py` | `data/plastic_weights_{fly0,fly1}.pt` |
| Fig. 9 — Cross-correlation | `generate_paper.py` | Paired session data |
| Fig. 10 — Compound eyes | `generate_paper.py` | `data/eye_{L,R}_{0,20}.png` |

### Step 3: Run Your Own Experiment

```bash
# Run a new two-fly experiment (headless, ~30 min for 100k steps)
python two_flies.py --steps 100000

# Results saved to consciousness_history/session_YYYYMMDD_HHMMSS_fly{0,1}/
# Plasticity weights saved to data/plastic_weights_fly{0,1}.pt
```

### Step 4: Independent Analysis Scripts

```bash
# Analyze plasticity divergence between two flies
python analyze_plasticity_divergence.py

# Compare overnight plasticity evolution
python analyze_overnight.py

# Direct weight comparison
python compare_plasticity.py
```

---

## Project Structure

```
fly-brain/
├── paper_emergent_individuality.pdf          # Paper (English, 14 pages)
├── paper_individualidad_emergente_ES.pdf     # Paper (Spanish, 14 pages)
├── paper_figures/                            # 10 publication figures (300 DPI)
│
├── fly_embodied.py          # Main closed-loop brain-body simulation     (1,035 lines)
├── brain_body_bridge.py     # DN decoding, motor commands, behavior      (  747 lines)
├── consciousness.py         # Multi-theory neural integration metrics    (  957 lines)
├── two_flies.py             # Two-fly experiment + plasticity tracking   (1,140 lines)
├── code/run_pytorch.py      # GPU LIF neuron simulation engine           (  514 lines)
│
├── visual_system.py         # Compound eye → motion detection pipeline   (  422 lines)
├── olfactory.py             # Bilateral ORN chemotaxis                   (  335 lines)
├── gustatory.py             # Tarsal taste detection (sugar/bitter)      (  197 lines)
├── somatosensory.py         # Touch, vibration, Johnston's organ         (  349 lines)
│
├── brain_monitor.py         # Real-time neural activity visualization    (1,253 lines)
├── looming_arena.py         # Natural environment with threats           (  255 lines)
├── procedural_arena.py      # Procedural world generation (Minecraft-like)(  410 lines)
├── flight.py                # Virtual flight state machine               (  200 lines)
├── vocalization.py          # Wing courtship song generation             (  191 lines)
├── fly_alive.py             # Autonomous neural-driven demo              (  260 lines)
│
├── generate_paper.py        # Paper + figure generation (English)        (1,333 lines)
├── generate_paper_es.py     # Paper + figure generation (Spanish)        (  674 lines)
├── analyze_plasticity_divergence.py  # Plasticity divergence analysis    (  778 lines)
├── analyze_overnight.py     # Overnight session analysis                 (  251 lines)
├── compare_plasticity.py    # Direct weight comparison tool              (  254 lines)
│
├── data/
│   ├── 2025_Completeness_783.csv        # FlyWire v783 neuron table (138,639 neurons)
│   ├── 2025_Connectivity_783.parquet    # Synaptic connectivity (15,091,983 edges)
│   ├── flywire_annotations.tsv          # Neuron type annotations
│   ├── sez_neurons.pickle               # SEZ neuron IDs (for gustation)
│   ├── plastic_weights.pt               # Baseline plasticity snapshot
│   ├── plastic_weights_fly0.pt          # Fly 0 final weights (after 24h)
│   ├── plastic_weights_fly1.pt          # Fly 1 final weights (after 24h)
│   ├── eye_{L,R}_{0,20}.png            # Compound eye renders
│   └── benchmark-results.csv            # GPU benchmark data
│
├── consciousness_history/               # 20 experimental sessions
│   ├── session_20260311_132935/         # Single-fly baseline sessions
│   ├── session_20260311_133411/
│   ├── session_20260311_134345/
│   ├── session_20260311_210436/         # Extended single-fly session
│   ├── session_20260311_225504_fly0/    # ┐ Paired session 1
│   ├── session_20260311_225556_fly1/    # ┘
│   ├── session_20260311_230255_fly0/    # ┐ Paired session 2
│   ├── session_20260311_230347_fly1/    # ┘
│   ├── session_20260311_230853_fly0/    # ┐ Paired session 3
│   ├── session_20260311_230945_fly1/    # ┘
│   ├── session_20260311_233655_fly0/    # ┐ Paired session 4
│   ├── session_20260311_233746_fly1/    # ┘
│   ├── session_20260312_071403_fly0/    # ┐ Paired session 5
│   ├── session_20260312_071508_fly1/    # ┘
│   ├── session_20260312_074258_fly0/    # ┐ Paired session 6
│   ├── session_20260312_074353_fly1/    # ┘
│   ├── session_20260312_083414_fly0/    # ┐ Paired session 7
│   ├── session_20260312_083508_fly1/    # ┘
│   ├── session_20260312_094510_fly0/    # ┐ Paired session 8
│   └── session_20260312_094601_fly1/    # ┘
│
├── code/                    # GPU simulation backends
│   ├── run_pytorch.py       # Primary: PyTorch sparse LIF engine
│   ├── run_brian2_cuda.py   # Alternative: Brian2CUDA backend
│   ├── run_nestgpu.py       # Alternative: NEST GPU backend
│   └── benchmark.py         # Backend comparison tool
│
├── scripts/
│   └── setup_WSL_CUDA.sh    # WSL2 + CUDA setup guide
│
├── docs/index.html          # Project webpage
├── environment.yml          # Conda environment specification
├── LICENSE                  # MIT License
└── demo.mp4 / demo_preview.gif  # Demo video
```

**Total**: ~11,500 lines of Python across 20 modules.

---

## Neural Model

The brain implements a **Leaky Integrate-and-Fire (LIF)** network from the [FlyWire connectome](https://flywire.ai/) v783 (Dorkenwald et al., 2024):

| Parameter | Value |
|---|---|
| Neurons | 138,639 |
| Synapses | 15,091,983 (directed, weighted) |
| Timestep | 0.2 ms (5 kHz) |
| Membrane time constant (tau_m) | 10 ms |
| Synaptic time constant (tau_s) | 5 ms |
| Resting potential (V_rest) | -65 mV |
| Threshold (V_th) | -50 mV |
| Reset potential (V_reset) | -65 mV |
| Refractory period (t_ref) | 2 ms |

Neurotransmitter identity determines synapse sign:
- **Excitatory**: acetylcholine, glutamate
- **Inhibitory**: GABA, glycine

### Hebbian Plasticity

All 15,091,983 synapses undergo continuous modification:

```
dW_ij = eta * (r_i * r_j) - alpha * W_ij
```

- `eta = 1e-4` — learning rate
- `alpha = 1e-7` — weight decay (homeostatic depression bias)
- Operates on GPU alongside neural dynamics (no additional memory overhead)

## Sensory Systems

| Modality | Neurons | Pathway | Reference |
|---|---|---|---|
| **Vision** | 721 ommatidia/eye | T1→T2→T3→T4/T5→LC4→GF→DN | Reichardt-like motion detection |
| **Olfaction** | ~2,600 ORNs | ORN→PN→KC→MBON→DN | Bilateral chemotaxis |
| **Gustation** | ~200 GRNs | GRN→SEZ→MN | Tarsal sugar/bitter |
| **Mechanosensation** | JO + leg sensors | Mechanoreceptor→IN→MN | Vibration + proprioception |

## Neural Integration Metrics

Four proxy metrics computed every 500 ms (see paper Section 2.5 for mathematical definitions):

| Metric | Theory | Range | Mean value |
|---|---|---|---|
| Phi | Integrated Information Theory (Tononi) | [0, 1] | ~0.15 |
| Broadcast | Global Workspace Theory (Baars, Dehaene) | [0, 1] | ~0.60 |
| Self-Model | Self-Model Theory (Metzinger) | [0, 1] | ~0.04 |
| Complexity | Perturbation Complexity (Koch) | [0, 1] | ~0.08 |

**Composite Index**: CI = 0.3 × Phi + 0.3 × Broadcast + 0.2 × Self-Model + 0.2 × Complexity

---

## Key Results

From 8 paired two-fly sessions over 24 hours of simulated time:

| Metric | Fly 0 | Fly 1 | Interpretation |
|---|---|---|---|
| Mean CI | 0.221 | 0.198 | 12% asymmetry |
| Escape behavior | 81.4% | 46.6% | Distinct motor profiles |
| Grooming behavior | 4.7% | 37.2% | Stable individual preferences |
| Divergent synapses | — | 76,034 (0.50%) | Experience-dependent plasticity |
| Cross-individual r | — | 0.19 | Near-independent dynamics |

See Figures 2–9 in the paper for detailed visualizations.

---

## System Requirements

| Component | Minimum | Tested |
|---|---|---|
| **GPU** | NVIDIA with CUDA 12.x, 6 GB VRAM | RTX 4060 Laptop (8 GB) |
| **RAM** | 32 GB | 64 GB DDR5 |
| **CPU** | Intel i5 / AMD Ryzen 5 | Intel i7-13620H |
| **OS** | Windows 10 / Ubuntu 20.04 | Windows 11 Pro |
| **Python** | 3.10+ | 3.10.16 |
| **PyTorch** | 2.0+ (CUDA) | 2.5.1+cu121 |
| **MuJoCo** | 3.0+ | 3.2.7 |

---

## Citation

```bibtex
@article{rojas_aliaga_2026,
  author  = {Rojas Aliaga, Enrique Manuel},
  title   = {Emergent Individuality and Neural Integration in Whole-Brain
             Connectome Simulations of {Drosophila melanogaster}},
  year    = {2026},
  journal = {bioRxiv},
  doi     = {10.1101/2026.XX.XX.XXXXXX},
  url     = {https://github.com/erojasoficial-byte/fly-brain}
}
```

## Acknowledgments

- [FlyWire Consortium](https://flywire.ai/) — Complete adult *Drosophila* connectome (Dorkenwald et al., 2024; Schlegel et al., 2024)
- [NeuroMechFly v2](https://neuromechfly.org/) — Biomechanical fly model (Lobato-Rios et al., 2024)
- [MuJoCo](https://mujoco.org/) — Physics engine (DeepMind)
- [Shiu et al. (2024)](https://doi.org/10.1038/s41586-024-07763-9) — LIF connectome model reference

## License

[MIT License](LICENSE) — free to use, modify, and distribute.

---

**Enrique Manuel Rojas Aliaga** · erojasoficial@gmail.com · Independent Researcher, Lima, Peru
