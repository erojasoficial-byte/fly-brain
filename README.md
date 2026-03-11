# Embodied Drosophila: 138,000-Neuron Whole-Brain Simulation in a Biomechanical Body

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GPU Accelerated](https://img.shields.io/badge/GPU-CUDA%2012.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![NeuroMechFly](https://img.shields.io/badge/body-NeuroMechFly%20v2-orange.svg)](https://neuromechfly.org/)
[![FlyWire Connectome](https://img.shields.io/badge/brain-FlyWire%20v783-purple.svg)](https://flywire.ai/)

> **A complete virtual fruit fly**: 138,639 LIF neurons running on GPU (PyTorch) connected to a biomechanical body (NeuroMechFly v2 / MuJoCo) with compound-eye vision, olfaction, gustation, somatosensation, courtship song, and virtual flight — all driven by the real connectome.

<div align="center">
  <img src="demo_preview.gif" alt="Embodied Drosophila simulation demo" width="640">
  <br>
  <em>Real-time embodied simulation: the fly sees, walks, and escapes — all driven by 138,639 spiking neurons from the FlyWire connectome.</em>
  <br>
  <a href="demo.mp4"><strong>Watch full video</strong></a>
</div>

---

## What is this?

This project connects a **whole-brain spiking neural network** of the fruit fly *Drosophila melanogaster* to a **physics-simulated biomechanical body**, creating a fully embodied virtual organism that:

- **Sees** through compound eyes (750 ommatidia per eye) with motion-sensitive neurons (T1-T5, lobula plate tangential cells)
- **Smells** via ~2,600 olfactory receptor neurons with bilateral gradient sensing (chemotaxis)
- **Tastes** through tarsal contact sensors (sugar attraction, bitter aversion)
- **Feels** ground contact forces and proprioceptive feedback across all 6 legs
- **Walks** with biologically realistic hexapod gaits (tripod, tetrapod, wave)
- **Escapes** looming threats via the Giant Fiber pathway (visual looming → LC4 → GF → motor)
- **Flies** using virtual wing forces when the Giant Fiber fires above threshold
- **Courts** with species-specific wing vibration songs (pulse and sine patterns)
- **Feeds** with proboscis extension triggered by gustatory neurons

All behaviors emerge from the connectome — no hardcoded behaviors, no if-else chains for actions.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    BRAIN (GPU/PyTorch)                   │
│  138,639 LIF neurons · 5M+ synapses · 0.1ms timestep   │
│                                                          │
│  Visual  ─── T1→T2→T3→T4→T5 ─── LC4 ─── GF ──→ DNs   │
│  cortex       (motion detection)   (loom)  (escape)      │
│                                                          │
│  Olfactory ─── ORN→PN→KC→MBON ─── DN turn commands     │
│  Gustatory ─── GRN→SEZ→MN ─── feeding/avoidance        │
│  Somatosensory ─── mechanoreceptors→IN→MN               │
└──────────────────────┬──────────────────────────────────┘
                       │ DN rates (descending neurons)
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
┌──────────────────────▼──────────────────────────────────┐
│              BODY (NeuroMechFly v2 / MuJoCo)            │
│  Biomechanical mesh · 6 legs · 3 joints each            │
│  Compound eyes · Contact sensors · Adhesion pads        │
│  Arena: ground, sky, sunlight, looming sphere            │
│  Virtual flight via xfrc_applied forces                  │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Clone
git clone https://github.com/eonsystemspbc/fly-brain.git
cd fly-brain

# Install dependencies
conda env create -f environment.yml
conda activate brain-fly
pip install flygym mujoco

# Run the embodied simulation (visual mode with brain monitor)
python fly_embodied.py --visual --monitor

# Run with flight enabled
python fly_embodied.py --visual --monitor --flight

# Run with olfactory/gustatory stimuli
python fly_embodied.py --visual --monitor --flight --sugar 10,0 --odor attractive,15,5

# Headless mode (faster, logs to console)
python fly_embodied.py --steps 5000
```

## Key Modules

| File | Description |
|---|---|
| `fly_embodied.py` | Main simulation loop — connects brain, body, and all sensory systems |
| `brain_body_bridge.py` | Translates descending neuron rates into locomotion drives and mode selection |
| `visual_system.py` | Compound eye rendering → contrast detection → motion energy (T1-T5 pathway) |
| `olfactory.py` | Bilateral ORN activation from odor sources with distance/wind model |
| `gustatory.py` | Tarsal taste detection (sugar/bitter) via ground contact zones |
| `somatosensory.py` | Leg contact forces, proprioception, and tactile escape reflexes |
| `vocalization.py` | Courtship song generation (P1→pulse/sine wing vibration patterns) |
| `flight.py` | Virtual flight state machine (GF→takeoff→fly→land) via MuJoCo external forces |
| `looming_arena.py` | Natural arena with sky, ground, sunlight, looming sphere, taste/odor zones |
| `brain_monitor.py` | Real-time neural activity visualization (pygame overlay) |
| `code/run_pytorch.py` | GPU-accelerated LIF network (sparse matrix, 0.1ms timestep) |

## Neural Model

The brain simulation implements a **Leaky Integrate-and-Fire (LIF)** network based on the [FlyWire connectome](https://flywire.ai/) (v783):

- **138,639 neurons** with biologically-derived connectivity
- **~5 million synapses** as a sparse matrix on GPU
- **0.1ms timestep**, 10,000 simulation steps per second of biological time
- Parameters: `tau_mem=20ms`, `tau_syn=5ms`, `V_thresh=-45mV`, `V_rest=-52mV`

Based on [Shiu et al. (2023)](https://www.biorxiv.org/content/10.1101/2023.05.02.539144v1) — *"A leaky integrate-and-fire computational model based on the connectome of the entire adult Drosophila brain."*

## Sensory Systems

### Vision (Compound Eyes)
Each compound eye has **750 ommatidia** rendered by MuJoCo. The visual pipeline implements the biological motion detection cascade:
- **T1**: Luminance → contrast (highpass filter)
- **T2**: Temporal derivative (ON transients)
- **T3**: Half-wave rectification
- **T4/T5**: Directional motion energy (Reichardt-like)
- **LC4 → Giant Fiber**: Looming detection → escape response

### Olfaction (Chemotaxis)
~2,600 olfactory receptor neurons (ORNs) respond to odor concentration gradients. Bilateral comparison (left vs right antenna) drives turning toward attractive odors and away from repulsive ones.

### Gustation (Taste)
Tarsal gustatory receptor neurons (GRNs) detect sugar and bitter compounds through leg contact with taste zones on the ground. Sugar triggers proboscis extension and feeding; bitter triggers avoidance.

### Somatosensation (Touch)
Contact force sensors on all 6 legs provide ground-truth proprioception. Sudden high forces trigger tactile escape reflexes via ascending mechanosensory neurons.

## Virtual Flight

When the Giant Fiber fires above threshold (GF > 0.4), the fly enters a **virtual flight mode**:

1. **Takeoff**: 2.5× body weight lift force, legs freeze in tucked position
2. **Flying**: Hover at ~1.05× body weight, DN-based steering (DNa01/DNa02 for thrust and yaw)
3. **Landing**: Gradual descent with altitude-proportional force reduction
4. **Cooldown**: 3-second refractory period prevents immediate re-takeoff

Flight uses MuJoCo's `xfrc_applied` to apply forces directly to the thorax body. Orientation is maintained via quaternion control.

## Benchmark Suite

The project also includes a benchmark comparing four neural simulation frameworks:

```bash
# Run benchmarks
python main.py --t_run 0.1 1 10 --n_run 1

# Specific framework
python main.py --pytorch --t_run 1 --n_run 1
```

| Framework | Backend | Status |
|---|---|---|
| Brian2 | C++ standalone (CPU) | ready |
| Brian2CUDA | CUDA standalone (GPU) | ready |
| PyTorch | CUDA sparse (GPU) | ready |
| NEST GPU | Custom CUDA kernel | ready |

## Data

The model uses [FlyWire](https://flywire.ai/) connectome data version **783** (public release):

| File | Description | Size |
|---|---|---|
| `data/2025_Completeness_783.csv` | Neuron IDs and metadata | 3.2 MB |
| `data/2025_Connectivity_783.parquet` | Synaptic connectivity matrix | 97 MB |
| `data/flywire_annotations.tsv` | Neuron type annotations | 32 MB |

## System Requirements

- **GPU**: NVIDIA with CUDA 12.x (tested on RTX 4070)
- **RAM**: 16 GB+ recommended
- **OS**: Windows 11 (native) or Linux (WSL2)
- **Python**: 3.10+
- **Key packages**: PyTorch (CUDA), flygym, MuJoCo, NumPy, SciPy, pygame

## Papers

Two technical papers are included documenting the full system:

- [`paper_embodied_drosophila.md`](paper_embodied_drosophila.md) — English
- [`paper_embodied_drosophila_es.md`](paper_embodied_drosophila_es.md) — Spanish (Español)

Generate PDFs with: `python md_to_pdf.py`

## Contributing

This is an open research project and **contributions are welcome**! Whether you're a neuroscientist, roboticist, ML engineer, or just curious about computational biology — there's room for you.

Ways to contribute:
- **New sensory modalities**: Auditory (Johnston's organ), hygrosensation, thermosensation
- **Circuit analysis**: Identify and validate specific neural circuits in the connectome
- **Behavioral validation**: Compare simulated behaviors with real *Drosophila* data
- **Performance optimization**: Improve GPU utilization, reduce memory footprint
- **Neuromorphic hardware**: Port the LIF network to Intel Loihi, SpiNNaker, or other neuromorphic chips
- **New behaviors**: Social interactions, learning, memory (mushroom body circuits)
- **Visualization**: Better real-time monitors, VR integration, data dashboards

Please open an issue or pull request. All skill levels welcome.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{embodied_drosophila_2026,
  author = {Rojas, E.},
  title = {Embodied Drosophila: Whole-Brain Connectome Simulation in a Biomechanical Body},
  year = {2026},
  url = {https://github.com/eonsystemspbc/fly-brain},
  note = {138,639 LIF neurons (FlyWire v783) + NeuroMechFly v2 + MuJoCo}
}
```

## License

MIT License — free to use, modify, and distribute.

## Acknowledgments

- [FlyWire](https://flywire.ai/) — Adult *Drosophila* whole-brain connectome
- [NeuroMechFly](https://neuromechfly.org/) — Biomechanical fly model (flygym)
- [MuJoCo](https://mujoco.org/) — Physics engine
- [Shiu et al. (2023)](https://www.biorxiv.org/content/10.1101/2023.05.02.539144v1) — LIF model foundation

---

**Keywords**: Drosophila, fruit fly, brain simulation, connectome, whole-brain model, spiking neural network, LIF neurons, NeuroMechFly, MuJoCo, embodied neuroscience, computational neuroscience, FlyWire, insect brain, virtual flight, escape behavior, Giant Fiber, GPU simulation, PyTorch, biomechanics, sensorimotor, compound eye, olfaction, chemotaxis
