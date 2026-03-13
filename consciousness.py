#!/usr/bin/env python3
"""
Consciousness Detection Module — Multi-theory proxy measurement.

Measures objective correlates of neural integration and complexity in the
Drosophila whole-brain simulation. Four independent metrics:

  1. Phi Proxy (IIT)      — mutual information between brain partitions
  2. Global Workspace     — broadcast coverage of high fan-out hub neurons
  3. Self-Model           — sensorimotor prediction (proprioceptive→behavior)
  4. Perturbation Complexity — cascade richness after random spike injection

Combined into a single Consciousness Index (CI) for tracking.
Does NOT claim consciousness — lets data speak.

Usage:
    Activated by --consciousness flag in fly_embodied.py
"""

import os
import csv
import time
import math
import numpy as np
import torch
from pathlib import Path
from collections import deque
from datetime import datetime


# ============================================================================
# Constants
# ============================================================================

# Composite index weights
W_PHI = 0.3
W_BROADCAST = 0.3
W_SELF = 0.2
W_COMPLEXITY = 0.2

# Partition caps (max neurons per partition for tractability)
VISUAL_CAP = 10000
MOTOR_CAP = 2000
OLFACTORY_CAP = 4000
INTEGRATOR_CAP = 9000

# Update intervals (in update() calls — each call ≈ 100ms body sim time)
# update() is called once per BRAIN_RATIO body steps (every ~100ms)
CI_RECORD_INTERVAL = 5     # record CI every ~500ms
PHI_INTERVAL = 5           # compute Phi every ~500ms (uses sliding window)
GW_INTERVAL = 5            # compute GW every ~500ms
SELF_INTERVAL = 3          # compute Self every ~300ms
PERTURB_INTERVAL = 50      # trigger perturbation every ~5s

# Phi time-series window (accumulate rate per partition each call)
PHI_WINDOW = 50            # ~5s of history for MI computation

# Self-model parameters
SELF_WINDOW = 10           # ~1s sliding window
SELF_LAG = 3               # ~300ms lag

# Perturbation parameters
PERTURB_N_NEURONS = 10
PERTURB_INJECT_STEPS = 3   # keep injection for 3 calls (~300ms body time)
PERTURB_OBS_STEPS = 50     # observe for ~5s (enough for synaptic delay + cascade)

# MI histogram bins (for time-series MI, fewer bins for short series)
MI_BINS = 8

# Global workspace fan-out threshold
FANOUT_THRESHOLD = 100     # avg fan-out ~36, so >100 catches top ~5% hubs


# ============================================================================
# Phi Proxy — IIT-inspired mutual information between brain partitions
# ============================================================================

class PhiProxy:
    """
    IIT-inspired measure: mutual information between 4 brain partitions.

    Uses TIME-SERIES MI: each call records aggregate firing rate per
    partition. MI is computed between partition rate time-series over a
    sliding window, measuring whether partitions share information
    (co-vary) over time.
    """

    def __init__(self, partition_indices, device='cpu'):
        self.device = device
        self.partitions = partition_indices
        self.partition_names = list(partition_indices.keys())
        self.n_partitions = len(self.partition_names)

        # Sliding window of per-partition mean firing rates
        self.rate_history = {
            name: deque(maxlen=PHI_WINDOW)
            for name in self.partition_names
        }
        self.phi = 0.0
        self.history = []

    def accumulate(self, spikes):
        """Record mean firing rate per partition for this timestep."""
        for name, idx in self.partitions.items():
            rate = float(spikes[0, idx].float().mean())
            self.rate_history[name].append(rate)

    def compute(self):
        """Compute Phi proxy as mean pairwise MI across partition time-series."""
        # Need enough data points for meaningful MI
        min_len = min(len(h) for h in self.rate_history.values())
        if min_len < 15:
            return self.phi

        # Compute mean pairwise MI between partition rate time-series
        mi_sum = 0.0
        n_pairs = 0

        for i in range(self.n_partitions):
            for j in range(i + 1, self.n_partitions):
                a = np.array(self.rate_history[self.partition_names[i]])
                b = np.array(self.rate_history[self.partition_names[j]])
                mi = self._timeseries_mi(a, b)
                mi_sum += mi
                n_pairs += 1

        raw_phi = mi_sum / max(n_pairs, 1)
        # Normalize: MI of time-series is typically small; scale up
        # Max possible MI with MI_BINS bins = log2(MI_BINS) ≈ 3 bits
        self.phi = min(1.0, raw_phi / 1.0)  # 1 bit MI → phi=1.0
        self.history.append(self.phi)
        return self.phi

    def _timeseries_mi(self, a, b):
        """Compute MI between two 1-D time-series using binned histogram."""
        n = len(a)

        a_min, a_max = a.min(), a.max()
        b_min, b_max = b.min(), b.max()

        # If either series is constant, MI = 0
        a_range = a_max - a_min
        b_range = b_max - b_min
        if a_range < 1e-12 or b_range < 1e-12:
            return 0.0

        # Bin into MI_BINS bins
        a_bin = np.clip(
            ((a - a_min) / (a_range + 1e-12) * (MI_BINS - 1)).astype(int),
            0, MI_BINS - 1)
        b_bin = np.clip(
            ((b - b_min) / (b_range + 1e-12) * (MI_BINS - 1)).astype(int),
            0, MI_BINS - 1)

        # Joint histogram
        joint = np.zeros((MI_BINS, MI_BINS))
        for k in range(n):
            joint[a_bin[k], b_bin[k]] += 1

        # Marginals and MI
        p_joint = joint / n
        p_a = p_joint.sum(axis=1)
        p_b = p_joint.sum(axis=0)

        mi = 0.0
        for ai in range(MI_BINS):
            for bi in range(MI_BINS):
                if p_joint[ai, bi] > 1e-10 and p_a[ai] > 1e-10 and p_b[bi] > 1e-10:
                    mi += p_joint[ai, bi] * np.log2(
                        p_joint[ai, bi] / (p_a[ai] * p_b[bi]))

        return max(mi, 0.0)


# ============================================================================
# Global Workspace — broadcast coverage of high fan-out hub neurons
# ============================================================================

class GlobalWorkspace:
    """
    GWT-inspired measure: identifies high fan-out hub neurons from sparse
    weight matrix. Measures broadcast coverage = fraction of partitions
    receiving activity from hub neurons over a rolling window.

    Uses rolling accumulation (no reset) so rare hub spikes are captured.
    """

    def __init__(self, hub_indices, hub_partition_reach, partition_indices,
                 device='cpu'):
        self.device = device
        self.hub_indices = hub_indices
        self.hub_partition_reach = hub_partition_reach
        self.partition_names = list(partition_indices.keys())
        self.n_partitions = len(self.partition_names)
        self.n_hubs = len(hub_indices)

        # Rolling window: track which hubs fired recently
        self.hub_recent = deque(maxlen=PHI_WINDOW)  # list of sets of hub indices
        self.broadcast = 0.0
        self.history = []

    def accumulate(self, spikes):
        """Record which hub neurons fired this step."""
        if self.n_hubs == 0:
            return
        hub_spikes = spikes[0, self.hub_indices]
        active = hub_spikes.nonzero(as_tuple=True)[0]
        fired_set = set(int(self.hub_indices[a]) for a in active.cpu())
        self.hub_recent.append(fired_set)

    def compute(self):
        """Compute broadcast coverage from rolling hub activity."""
        if len(self.hub_recent) < 5 or self.n_hubs == 0:
            return self.broadcast

        # Union of all hubs that fired in the recent window
        all_active = set()
        for s in self.hub_recent:
            all_active.update(s)

        if not all_active:
            self.broadcast = 0.0
            self.history.append(self.broadcast)
            return self.broadcast

        # Count partitions reached by active hubs
        reached = set()
        for hub_idx in all_active:
            if hub_idx in self.hub_partition_reach:
                reached.update(self.hub_partition_reach[hub_idx])

        # Coverage: fraction of partitions reached
        coverage = len(reached) / max(self.n_partitions, 1)

        # Hub activity: fraction of hubs that fired at least once
        hub_fraction = len(all_active) / self.n_hubs

        self.broadcast = min(1.0, coverage * 0.6 + hub_fraction * 0.4)
        self.history.append(self.broadcast)
        return self.broadcast


# ============================================================================
# Self-Model — sensorimotor prediction correlation
# ============================================================================

class SelfModel:
    """
    Metzinger-inspired: sliding window correlation between proprioceptive
    signals (JO touch/sound neurons) and subsequent behavior changes.

    Measures whether the brain predicts its own body state.
    """

    def __init__(self, sensory_indices, motor_indices, device='cpu'):
        """
        Args:
            sensory_indices: tensor of JO neuron indices (proprioceptive)
            motor_indices: tensor of motor/DN neuron indices
            device: torch device
        """
        self.device = device
        self.sensory_indices = sensory_indices
        self.motor_indices = motor_indices

        # Sliding windows
        self.sensory_history = deque(maxlen=SELF_WINDOW + SELF_LAG)
        self.motor_history = deque(maxlen=SELF_WINDOW + SELF_LAG)

        self.self_score = 0.0
        self.history = []

    def accumulate(self, spikes):
        """Record sensory and motor activity for one brain step."""
        sensory_rate = float(spikes[0, self.sensory_indices].mean())
        motor_rate = float(spikes[0, self.motor_indices].mean())
        self.sensory_history.append(sensory_rate)
        self.motor_history.append(motor_rate)

    def compute(self):
        """Compute correlation between lagged sensory and motor signals."""
        if len(self.sensory_history) < SELF_WINDOW + SELF_LAG:
            return self.self_score

        # Sensory from SELF_LAG steps ago, motor from now
        sensory = np.array(list(self.sensory_history)[:SELF_WINDOW])
        motor = np.array(list(self.motor_history)[SELF_LAG:SELF_LAG + SELF_WINDOW])

        # Pearson correlation (absolute value — anticorrelation also counts)
        s_std = sensory.std()
        m_std = motor.std()
        if s_std < 1e-10 or m_std < 1e-10:
            self.self_score = 0.0
        else:
            corr = np.corrcoef(sensory, motor)[0, 1]
            self.self_score = float(abs(corr)) if not np.isnan(corr) else 0.0

        self.history.append(self.self_score)
        return self.self_score


# ============================================================================
# Perturbation Complexity — cascade richness after spike injection
# ============================================================================

class PerturbationComplexity:
    """
    Koch-inspired: inject spikes into random neurons, observe cascade.

    Complexity = regions_affected × temporal_entropy of the cascade.
    """

    def __init__(self, num_neurons, partition_indices, device='cpu'):
        """
        Args:
            num_neurons: total neurons in the model
            partition_indices: dict {name: tensor of indices}
            device: torch device
        """
        self.device = device
        self.num_neurons = num_neurons
        self.partition_indices = partition_indices
        self.partition_names = list(partition_indices.keys())

        self.complexity = 0.0
        self.history = []

        # State for ongoing perturbation observation
        self._observing = False
        self._obs_steps = 0
        self._baseline_rates = None
        self._cascade_bins = []  # spike counts per time bin during observation
        self._cascade_partitions = set()

    def should_perturb(self, brain_step):
        """Check if it's time for a new perturbation."""
        return (not self._observing and
                brain_step > 0 and
                brain_step % PERTURB_INTERVAL == 0)

    def start_perturbation(self, brain, baseline_spikes):
        """Inject spikes into random neurons and begin observation."""
        targets = torch.randint(0, self.num_neurons, (PERTURB_N_NEURONS,),
                                device=self.device)

        # Record baseline: mean spike rate per partition over recent history
        self._baseline_rates = {}
        for name, idx in self.partition_indices.items():
            self._baseline_rates[name] = float(baseline_spikes[0, idx].mean())

        # Inject spikes by boosting rates (kept for PERTURB_INJECT_STEPS)
        brain.rates[0, targets] += 500.0

        self._observing = True
        self._obs_steps = 0
        self._cascade_bins = []
        self._cascade_partitions = set()
        self._inject_targets = targets

    def observe(self, spikes, brain=None):
        """Observe cascade effects after perturbation."""
        if not self._observing:
            return

        self._obs_steps += 1

        # Remove injection after PERTURB_INJECT_STEPS
        if self._obs_steps == PERTURB_INJECT_STEPS and brain is not None:
            brain.rates[0, self._inject_targets] -= 500.0
            brain.rates.clamp_(min=0.0)

        # Record total spike count for this step
        total = float(spikes.sum())
        self._cascade_bins.append(total)

        # Check which partitions show ANY activity above baseline
        for name, idx in self.partition_indices.items():
            rate = float(spikes[0, idx].float().mean())
            baseline = self._baseline_rates.get(name, 0.0)
            # Use additive threshold for near-zero baselines
            threshold = max(baseline * 1.3, baseline + 0.001)
            if rate > threshold:
                self._cascade_partitions.add(name)

        if self._obs_steps >= PERTURB_OBS_STEPS:
            self._finish_observation()

    def _finish_observation(self):
        """Compute complexity from observed cascade."""
        # Regions affected (normalized)
        region_fraction = len(self._cascade_partitions) / max(
            len(self.partition_names), 1)

        # Temporal entropy of cascade bins
        bins = np.array(self._cascade_bins)
        if bins.sum() > 0:
            # Divide into 10 temporal bins
            n_tbins = min(10, len(bins))
            chunk_size = len(bins) // n_tbins
            temporal_counts = []
            for i in range(n_tbins):
                start = i * chunk_size
                end = start + chunk_size if i < n_tbins - 1 else len(bins)
                temporal_counts.append(bins[start:end].sum())
            temporal_counts = np.array(temporal_counts)
            total = temporal_counts.sum()
            if total > 0:
                probs = temporal_counts / total
                probs = probs[probs > 0]
                entropy = -float(np.sum(probs * np.log2(probs)))
                max_entropy = np.log2(n_tbins)
                norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
            else:
                norm_entropy = 0.0
        else:
            norm_entropy = 0.0

        self.complexity = min(1.0, region_fraction * norm_entropy * 2.0)
        self.history.append(self.complexity)

        self._observing = False
        self._obs_steps = 0
        self._baseline_rates = None

    @property
    def is_observing(self):
        return self._observing


# ============================================================================
# Consciousness Timeline — composite index, CSV logging, reports
# ============================================================================

class ConsciousnessTimeline:
    """
    Tracks composite Consciousness Index over time.
    Handles CSV logging, peak detection, and report generation.
    """

    def __init__(self, session_dir):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.timeline = []  # list of dicts
        self.peaks = []     # (step, ci_value) tuples
        self.mode_stats = {}  # {mode: [ci_values]}

        # CSV writers
        tl_path = self.session_dir / 'consciousness_timeline.csv'
        self._tl_file = open(tl_path, 'w', newline='')
        self._tl_writer = csv.writer(self._tl_file)
        self._tl_writer.writerow([
            'step', 't_sim', 'CI', 'phi', 'broadcast', 'self', 'complexity',
            'mode',
        ])

        phi_path = self.session_dir / 'phi_timeseries.csv'
        self._phi_file = open(phi_path, 'w', newline='')
        self._phi_writer = csv.writer(self._phi_file)
        self._phi_writer.writerow(['step', 'phi_proxy'])

    def record(self, step, t_sim, phi, broadcast, self_score, complexity, mode):
        """Record one measurement to timeline."""
        ci = (W_PHI * phi + W_BROADCAST * broadcast +
              W_SELF * self_score + W_COMPLEXITY * complexity)

        entry = {
            'step': step, 't_sim': t_sim, 'ci': ci,
            'phi': phi, 'broadcast': broadcast,
            'self': self_score, 'complexity': complexity, 'mode': mode,
        }
        self.timeline.append(entry)

        # CSV log
        self._tl_writer.writerow([
            step, f'{t_sim:.4f}', f'{ci:.4f}', f'{phi:.4f}',
            f'{broadcast:.4f}', f'{self_score:.4f}', f'{complexity:.4f}',
            mode,
        ])
        try:
            self._tl_file.flush()
        except (PermissionError, OSError):
            pass  # Windows antivirus can block flush

        # Phi timeseries
        self._phi_writer.writerow([step, f'{phi:.4f}'])
        try:
            self._phi_file.flush()
        except (PermissionError, OSError):
            pass

        # Mode stats
        if mode not in self.mode_stats:
            self.mode_stats[mode] = []
        self.mode_stats[mode].append(ci)

        # Peak detection (local maximum in last 5 entries)
        if len(self.timeline) >= 3:
            recent = [e['ci'] for e in self.timeline[-3:]]
            if recent[1] > recent[0] and recent[1] >= recent[2]:
                peak_entry = self.timeline[-2]
                self.peaks.append((peak_entry['step'], peak_entry['ci']))

        return ci

    def get_latest(self):
        """Return latest CI entry or None."""
        return self.timeline[-1] if self.timeline else None

    def get_recent_ci_values(self, n=60):
        """Return last n CI values for timeline display."""
        return [e['ci'] for e in self.timeline[-n:]]

    def generate_report(self):
        """Generate summary report and save to file."""
        report_path = self.session_dir / 'consciousness_report.txt'

        lines = []
        lines.append("=" * 70)
        lines.append("CONSCIOUSNESS PROXY MEASUREMENT REPORT")
        lines.append("Embodied Drosophila — Fly-Brain Simulation")
        lines.append(f"Session: {self.session_dir.name}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")

        if not self.timeline:
            lines.append("No data recorded.")
            with open(report_path, 'w') as f:
                f.write('\n'.join(lines))
            return

        # Summary stats
        ci_values = [e['ci'] for e in self.timeline]
        phi_values = [e['phi'] for e in self.timeline]
        bcast_values = [e['broadcast'] for e in self.timeline]
        self_values = [e['self'] for e in self.timeline]
        cmplx_values = [e['complexity'] for e in self.timeline]

        lines.append("SUMMARY STATISTICS")
        lines.append("-" * 40)
        for label, vals in [('CI', ci_values), ('Phi', phi_values),
                            ('Broadcast', bcast_values),
                            ('Self-Model', self_values),
                            ('Complexity', cmplx_values)]:
            arr = np.array(vals)
            lines.append(f"  {label:12s}: mean={arr.mean():.4f}  "
                         f"std={arr.std():.4f}  "
                         f"min={arr.min():.4f}  max={arr.max():.4f}")
        lines.append("")

        # Per-mode analysis (Experiment A)
        lines.append("EXPERIMENT A: CI BY BEHAVIORAL MODE")
        lines.append("-" * 40)
        for mode, vals in sorted(self.mode_stats.items()):
            arr = np.array(vals)
            lines.append(f"  {mode:12s}: mean={arr.mean():.4f}  "
                         f"std={arr.std():.4f}  n={len(vals)}")
        lines.append("")

        # Experiment B: Phi during mode transitions
        lines.append("EXPERIMENT B: PHI DURING MODE TRANSITIONS")
        lines.append("-" * 40)
        transitions = []
        for i in range(1, len(self.timeline)):
            if self.timeline[i]['mode'] != self.timeline[i - 1]['mode']:
                transitions.append(self.timeline[i])
        if transitions:
            t_phi = [t['phi'] for t in transitions]
            nontrans_phi = [e['phi'] for e in self.timeline
                            if e not in transitions]
            lines.append(f"  During transitions:     "
                         f"mean_phi={np.mean(t_phi):.4f}  n={len(t_phi)}")
            if nontrans_phi:
                lines.append(f"  During stable behavior: "
                             f"mean_phi={np.mean(nontrans_phi):.4f}  "
                             f"n={len(nontrans_phi)}")
        else:
            lines.append("  No mode transitions detected.")
        lines.append("")

        # Experiment C: CI habituation (first vs later measurements)
        lines.append("EXPERIMENT C: CI HABITUATION OVER TIME")
        lines.append("-" * 40)
        n_total = len(ci_values)
        if n_total >= 6:
            first_third = ci_values[:n_total // 3]
            last_third = ci_values[-(n_total // 3):]
            lines.append(f"  First third:  mean_CI={np.mean(first_third):.4f}")
            lines.append(f"  Last third:   mean_CI={np.mean(last_third):.4f}")
            delta = np.mean(last_third) - np.mean(first_third)
            lines.append(f"  Delta:        {delta:+.4f} "
                         f"({'habituation' if delta < 0 else 'sensitization'})")
        else:
            lines.append("  Insufficient data (need >= 6 measurements).")
        lines.append("")

        # Experiment D: GF/escape analysis
        lines.append("EXPERIMENT D: CI vs ESCAPE MODE")
        lines.append("-" * 40)
        escape_ci = self.mode_stats.get('escape', [])
        non_escape_ci = []
        for mode, vals in self.mode_stats.items():
            if mode != 'escape':
                non_escape_ci.extend(vals)
        if escape_ci:
            lines.append(f"  Escape CI:     mean={np.mean(escape_ci):.4f}  "
                         f"n={len(escape_ci)}")
        if non_escape_ci:
            lines.append(f"  Non-escape CI: mean={np.mean(non_escape_ci):.4f}"
                         f"  n={len(non_escape_ci)}")
        lines.append("")

        # Peak events
        lines.append("PEAK EVENTS (top 10)")
        lines.append("-" * 40)
        sorted_peaks = sorted(self.peaks, key=lambda x: x[1], reverse=True)
        for step, ci in sorted_peaks[:10]:
            lines.append(f"  step={step:>8d}  CI={ci:.4f}")
        lines.append("")

        lines.append("=" * 70)
        lines.append("Note: These are proxy measurements of neural integration")
        lines.append("and complexity. They do not constitute evidence of")
        lines.append("subjective experience or phenomenal consciousness.")
        lines.append("=" * 70)

        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"[Consciousness] Report saved: {report_path}")

    def close(self):
        """Close CSV file handles."""
        try:
            self._tl_file.close()
            self._phi_file.close()
        except Exception:
            pass


# ============================================================================
# ConsciousnessDetector — Main orchestrator
# ============================================================================

class ConsciousnessDetector:
    """
    Main orchestrator: loads neuron partitions from annotations,
    initializes all sub-modules, provides update/query interface.
    """

    def __init__(self, brain, label='', sim_timestep=1e-3):
        """
        Args:
            brain: BrainEngine instance with .model.weights, .flyid2i,
                   .num_neurons, .state, .device
            label: optional suffix for session directory (e.g. 'fly0')
            sim_timestep: body simulation timestep in seconds (for time calc)
        """
        self.brain = brain
        self.device = brain.device
        self.num_neurons = brain.num_neurons
        self.brain_step = 0
        self.sim_timestep = sim_timestep
        self._label = label

        tag = f" ({label})" if label else ""
        print(f"[Consciousness{tag}] Initializing consciousness detection...")

        # Load neuron partitions from annotations
        self.partitions = self._build_partitions(brain)

        for name, idx in self.partitions.items():
            print(f"  Partition '{name}': {len(idx)} neurons")

        # Find hub neurons for Global Workspace
        hub_indices, hub_reach = self._find_hub_neurons(brain)
        print(f"  Hub neurons (fan-out > {FANOUT_THRESHOLD}): {len(hub_indices)}")

        # Find sensory/motor indices for Self-Model
        sensory_idx, motor_idx = self._get_sensory_motor_indices(brain)
        print(f"  Sensory (JO) neurons: {len(sensory_idx)}")
        print(f"  Motor (DN) neurons: {len(motor_idx)}")

        # Initialize sub-modules
        self.phi = PhiProxy(self.partitions, device=self.device)
        self.gw = GlobalWorkspace(
            hub_indices, hub_reach, self.partitions, device=self.device)
        self.self_model = SelfModel(
            sensory_idx, motor_idx, device=self.device)
        self.perturbation = PerturbationComplexity(
            self.num_neurons, self.partitions, device=self.device)

        # Session directory
        session_name = datetime.now().strftime('session_%Y%m%d_%H%M%S')
        if label:
            session_name += f'_{label}'
        base_dir = Path(__file__).resolve().parent / 'consciousness_history'
        self.timeline = ConsciousnessTimeline(base_dir / session_name)

        # Current values for display
        self.ci = 0.0
        self.phi_val = 0.0
        self.gw_val = 0.0
        self.self_val = 0.0
        self.cmplx_val = 0.0

        print(f"[Consciousness{tag}] Ready. Composite weights: "
              f"Phi={W_PHI} GW={W_BROADCAST} Self={W_SELF} "
              f"Cmplx={W_COMPLEXITY}")

    def _build_partitions(self, brain):
        """Build neuron partitions from flywire_annotations.tsv."""
        ann_path = (Path(__file__).resolve().parent / 'data' /
                    'flywire_annotations.tsv')

        partitions = {
            'visual': [], 'motor': [], 'olfactory': [], 'integrator': [],
        }

        if not ann_path.exists():
            print(f"  [WARN] Annotations not found: {ann_path}")
            print("  Using fallback: equal random partitions")
            return self._fallback_partitions(brain)

        # Parse annotations
        flyid2i = brain.flyid2i

        with open(ann_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                root_id = int(row.get('root_id', 0))
                if root_id not in flyid2i:
                    continue
                idx = flyid2i[root_id]

                super_class = row.get('super_class', '').lower()
                cell_class = row.get('cell_class', '').lower()
                flow = row.get('flow', '').lower()
                hemibrain_type = row.get('hemibrain_type', '').lower()

                # Visual: optic lobe neurons + visual projection
                if super_class in ('optic', 'visual_projection'):
                    partitions['visual'].append(idx)
                # Motor: efferent / descending neurons
                elif flow in ('efferent', 'descending'):
                    partitions['motor'].append(idx)
                # Olfactory: olfactory, ALPN, ALLN, LHLN classes
                elif cell_class in ('olfactory', 'alpn', 'alln', 'lhln'):
                    partitions['olfactory'].append(idx)
                # Integrator: MBON, CX (central complex), KC, DAN, TuBu
                elif any(tag in hemibrain_type
                         for tag in ('mbon', 'cx', 'kc', 'dan', 'tubu')
                         if tag):
                    partitions['integrator'].append(idx)
                elif any(tag in cell_class
                         for tag in ('mbon', 'kenyon', 'dan')
                         if tag):
                    partitions['integrator'].append(idx)

        # Apply caps
        caps = {
            'visual': VISUAL_CAP, 'motor': MOTOR_CAP,
            'olfactory': OLFACTORY_CAP, 'integrator': INTEGRATOR_CAP,
        }
        result = {}
        for name, indices in partitions.items():
            cap = caps[name]
            if len(indices) > cap:
                indices = indices[:cap]
            if len(indices) == 0:
                # Fallback: grab some random neurons
                n_fallback = min(100, brain.num_neurons)
                indices = list(range(n_fallback))
            result[name] = torch.tensor(indices, dtype=torch.long,
                                        device=self.device)

        return result

    def _fallback_partitions(self, brain):
        """Create simple partitions when annotations unavailable."""
        n = brain.num_neurons
        chunk = n // 4
        return {
            'visual': torch.arange(0, chunk, device=self.device),
            'motor': torch.arange(chunk, 2 * chunk, device=self.device),
            'olfactory': torch.arange(2 * chunk, 3 * chunk, device=self.device),
            'integrator': torch.arange(3 * chunk, n, device=self.device),
        }

    def _find_hub_neurons(self, brain):
        """Find neurons with fan-out > FANOUT_THRESHOLD from weight matrix."""
        weights = brain.model.weights

        try:
            # Convert to COO for column counting
            if weights.is_sparse_csr:
                w_coo = weights.to_sparse_coo()
            elif weights.is_sparse:
                w_coo = weights
            else:
                # Dense fallback (unlikely for 138K neurons)
                return (torch.tensor([], dtype=torch.long, device=self.device),
                        {})

            indices = w_coo.indices()  # (2, nnz): [row_indices, col_indices]
            col_indices = indices[1]   # presynaptic (source) neurons

            # Count outgoing connections per neuron
            fan_out = torch.zeros(self.num_neurons, device=self.device)
            fan_out.scatter_add_(
                0, col_indices,
                torch.ones(col_indices.shape[0], device=self.device))

            hub_mask = fan_out > FANOUT_THRESHOLD
            hub_neuron_indices = hub_mask.nonzero(as_tuple=True)[0]

            # Determine which partitions each hub can reach
            row_indices = indices[0]  # postsynaptic (target) neurons
            hub_reach = {}

            # Build partition membership lookup (on CPU for dict operations)
            neuron_to_partition = {}
            for name, idx_tensor in self.partitions.items():
                for i in idx_tensor.cpu().numpy():
                    neuron_to_partition[int(i)] = name

            # For each hub, find which partitions its targets belong to
            for hub_idx in hub_neuron_indices.cpu().numpy():
                hub_idx = int(hub_idx)
                # Find all targets of this hub
                target_mask = col_indices == hub_idx
                targets = row_indices[target_mask].cpu().numpy()
                reached = set()
                for t in targets[:200]:  # sample up to 200 targets
                    part = neuron_to_partition.get(int(t))
                    if part:
                        reached.add(part)
                hub_reach[hub_idx] = reached

            return hub_neuron_indices, hub_reach

        except Exception as e:
            print(f"  [WARN] Hub detection failed: {e}")
            return (torch.tensor([], dtype=torch.long, device=self.device),
                    {})

    def _get_sensory_motor_indices(self, brain):
        """Get JO/sensory indices and motor/DN indices."""
        flyid2i = brain.flyid2i

        # Motor indices from DN_NEURONS
        from brain_body_bridge import DN_NEURONS
        motor_indices = []
        for name, flyid in DN_NEURONS.items():
            if flyid in flyid2i:
                motor_indices.append(flyid2i[flyid])

        # Sensory: use olfactory + first 100 from visual as proprioceptive proxy
        sensory_indices = []
        if 'olfactory' in self.partitions:
            sensory_indices.extend(
                self.partitions['olfactory'][:200].cpu().numpy().tolist())
        if 'visual' in self.partitions:
            sensory_indices.extend(
                self.partitions['visual'][:100].cpu().numpy().tolist())

        # Ensure we have at least some indices
        if not sensory_indices:
            sensory_indices = list(range(min(100, brain.num_neurons)))
        if not motor_indices:
            motor_indices = list(range(min(20, brain.num_neurons)))

        return (torch.tensor(sensory_indices, dtype=torch.long,
                             device=self.device),
                torch.tensor(motor_indices, dtype=torch.long,
                             device=self.device))

    @torch.no_grad()
    def update(self, body_step, mode='walking'):
        """
        Called every brain step. Routes data to sub-modules on their schedules.

        Args:
            body_step: current body simulation step count
            mode: current behavioral mode string
        """
        spikes = self.brain.state[2]  # (1, N) spike tensor
        self.brain_step += 1
        step = self.brain_step

        # Always accumulate
        self.phi.accumulate(spikes)
        self.gw.accumulate(spikes)
        self.self_model.accumulate(spikes)

        # Perturbation observation (if active)
        if self.perturbation.is_observing:
            self.perturbation.observe(spikes, self.brain)

        # Periodic Phi computation
        if step % PHI_INTERVAL == 0:
            self.phi_val = self.phi.compute()

        # Periodic GW computation
        if step % GW_INTERVAL == 0:
            self.gw_val = self.gw.compute()

        # Periodic Self-Model computation
        if step % SELF_INTERVAL == 0:
            self.self_val = self.self_model.compute()

        # Perturbation trigger
        if self.perturbation.should_perturb(step):
            self.perturbation.start_perturbation(self.brain, spikes)

        # Update complexity from latest perturbation result
        if self.perturbation.history:
            self.cmplx_val = self.perturbation.history[-1]

        # Composite CI (record every CI_RECORD_INTERVAL)
        if step % CI_RECORD_INTERVAL == 0:
            t_sim = body_step * self.sim_timestep
            self.ci = self.timeline.record(
                step, t_sim,
                self.phi_val, self.gw_val, self.self_val, self.cmplx_val,
                mode,
            )

    def get_status_str(self):
        """Return compact status string for console output."""
        return (f"CI={self.ci:.3f} "
                f"Phi={self.phi_val:.2f} GW={self.gw_val:.2f} "
                f"Self={self.self_val:.2f} Cmplx={self.cmplx_val:.2f}")

    def get_monitor_data(self):
        """Return dict of data for brain monitor visualization."""
        recent = self.timeline.get_recent_ci_values(60)
        peaks = [(s, v) for s, v in self.timeline.peaks[-5:]]

        return {
            'consciousness_ci': self.ci,
            'consciousness_phi': self.phi_val,
            'consciousness_gw': self.gw_val,
            'consciousness_self': self.self_val,
            'consciousness_cmplx': self.cmplx_val,
            'consciousness_timeline': recent,
            'consciousness_peaks': peaks,
        }

    def save_session(self):
        """Generate report and close files. Called at simulation end."""
        print("[Consciousness] Generating session report...")
        self.timeline.generate_report()
        self.timeline.close()
        print(f"[Consciousness] Session data saved to: "
              f"{self.timeline.session_dir}")
