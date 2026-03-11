#!/usr/bin/env python3
"""
Somatosensory & Auditory System — Touch (contact forces) and Sound (vibration)
routed through Johnston's Organ (JO) neurons in the connectome.

Biology:
  JO-E/C subtypes → mechanosensory (touch, wind, gravity)
  JO-A/B subtypes → auditory (near-field sound, courtship song)

Architecture:
  MuJoCo contact forces (6 legs × 6 segments) → JO-E/C firing rates
  Virtual vibration sources (position, frequency) → JO-A/B firing rates
  Both injected as Poisson rates into the brain, coexisting with vision.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================================
# Vibration Source
# ============================================================================

class VibrationSource:
    """A virtual sound/vibration emitter in the arena.

    Parameters
    ----------
    position : array-like
        [x, y, z] in mm (arena coordinates).
    frequency : float
        Vibration frequency in Hz. ~200 = courtship, ~400 = alarm.
    amplitude : float
        Source intensity (0-1 scale).
    label : str
        Human-readable name.
    """

    def __init__(self, position, frequency=200.0, amplitude=1.0, label='vibration'):
        self.position = np.array(position, dtype=np.float64)
        self.frequency = frequency
        self.amplitude = amplitude
        self.label = label


# ============================================================================
# JO Neuron Subtypes (cell_type prefixes in FlyWire annotations)
# ============================================================================

# Touch / mechanosensory
JO_TOUCH_TYPES = [
    'JO-E', 'JO-EDC', 'JO-EDM', 'JO-EDP', 'JO-EV', 'JO-EVL', 'JO-EVM',
    'JO-EVP', 'JO-C', 'JO-CA', 'JO-CL', 'JO-CM',
]

# Auditory (near-field sound)
JO_SOUND_TYPES = ['JO-A', 'JO-B']


# ============================================================================
# Somatosensory System
# ============================================================================

class SomatosensorySystem:
    """Processes touch (contact forces) and sound (vibration) into JO rates.

    Contact forces from 6 legs are mapped bilaterally to JO-E/C neurons.
    Vibration sources are mapped bilaterally to JO-A/B neurons based on
    direction relative to the fly's heading.

    Parameters
    ----------
    flyid2i : dict
        FlyWire neuron ID → tensor index mapping.
    annotations_path : str or Path, optional
        Path to flywire_annotations.tsv.
    """

    # Leg layout: indices into (36,3) contact_forces array
    # 6 legs × 6 segments (Tibia + 5 Tarsus)
    LEG_NAMES = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
    SEGMENTS_PER_LEG = 6

    # Force thresholds (Newtons, magnitude)
    FORCE_FLOOR = 0.3     # below = normal walking contact, no JO activation
    FORCE_GROOM = 1.5     # moderate touch → grooming-level JO rate
    FORCE_ESCAPE = 5.0    # strong impact → high JO rate (tactile escape)
    FORCE_SAT = 10.0      # saturation

    # Firing rate limits (Hz)
    TOUCH_MAX_RATE = 250.0
    SOUND_MAX_RATE = 200.0

    # Vibration parameters
    VIBRATION_RANGE = 40.0    # mm — half-power distance
    COURTSHIP_CENTER = 200.0  # Hz — peak frequency for JO-A tuning
    COURTSHIP_BW = 80.0       # Hz — bandwidth (gaussian σ)

    def __init__(self, flyid2i, annotations_path=None):
        self.flyid2i = flyid2i

        if annotations_path is None:
            annotations_path = (
                Path(__file__).resolve().parent / 'data' /
                'flywire_annotations.tsv')
        self._annotations_path = Path(annotations_path)

        # Neuron indices (populated by _load_jo_neurons)
        self.touch_idx_left = np.array([], dtype=np.int64)
        self.touch_idx_right = np.array([], dtype=np.int64)
        self.sound_idx_left = np.array([], dtype=np.int64)
        self.sound_idx_right = np.array([], dtype=np.int64)

        self._load_jo_neurons()

        # Runtime state
        self.leg_forces = np.zeros(6)        # force magnitude per leg
        self.touch_rate_left = 0.0
        self.touch_rate_right = 0.0
        self.sound_rate_left = 0.0
        self.sound_rate_right = 0.0
        self.max_contact_force = 0.0
        self.orientation_bias = 0.0          # +1=source right, -1=source left

    # ── Neuron Loading ────────────────────────────────────────────────────

    def _load_jo_neurons(self):
        """Load JO neuron IDs from FlyWire annotations, split by type/side."""
        path = self._annotations_path
        if not path.exists():
            print(f"[Somatosensory] WARNING: annotations not found at {path}")
            return

        df = pd.read_csv(path, sep='\t', low_memory=False)

        # Find touch neurons
        touch_L, touch_R = self._find_neurons(df, JO_TOUCH_TYPES)
        self.touch_idx_left = self._to_indices(touch_L)
        self.touch_idx_right = self._to_indices(touch_R)

        # Find sound neurons
        sound_L, sound_R = self._find_neurons(df, JO_SOUND_TYPES)
        self.sound_idx_left = self._to_indices(sound_L)
        self.sound_idx_right = self._to_indices(sound_R)

        n_touch = len(self.touch_idx_left) + len(self.touch_idx_right)
        n_sound = len(self.sound_idx_left) + len(self.sound_idx_right)
        print(f"[Somatosensory] JO-touch: {n_touch} neurons "
              f"(L={len(self.touch_idx_left)}, R={len(self.touch_idx_right)})")
        print(f"[Somatosensory] JO-sound: {n_sound} neurons "
              f"(L={len(self.sound_idx_left)}, R={len(self.sound_idx_right)})")

    def _find_neurons(self, df, type_names):
        """Find neurons by exact cell_type match, split left/right."""
        type_col = 'cell_type'
        id_col = 'root_id'
        side_col = 'side'

        mask = df[type_col].astype(str).isin(set(type_names))
        subset = df[mask][[id_col, side_col]].dropna(subset=[id_col])

        left_ids = subset[subset[side_col] == 'left'][id_col].astype(int).tolist()
        right_ids = subset[subset[side_col] == 'right'][id_col].astype(int).tolist()
        return left_ids, right_ids

    def _to_indices(self, flyids):
        """Convert FlyWire IDs to tensor indices."""
        indices = [self.flyid2i[fid] for fid in flyids if fid in self.flyid2i]
        return np.array(indices, dtype=np.int64)

    # ── Contact Force Processing ──────────────────────────────────────────

    def process_contact(self, contact_forces):
        """Convert MuJoCo contact forces to bilateral JO touch rates.

        Parameters
        ----------
        contact_forces : np.ndarray, shape (36, 3)
            3D force vectors for 6 legs × 6 segments.
        """
        # Reshape: (6 legs, 6 segments, 3 axes)
        forces = contact_forces.reshape(6, self.SEGMENTS_PER_LEG, 3)
        magnitudes = np.linalg.norm(forces, axis=2)  # (6, 6)
        self.leg_forces = magnitudes.max(axis=1)      # max per leg (6,)
        self.max_contact_force = float(self.leg_forces.max())

        # Left legs: LF(0), LM(1), LH(2)
        left_force = float(self.leg_forces[:3].max())
        # Right legs: RF(3), RM(4), RH(5)
        right_force = float(self.leg_forces[3:].max())

        self.touch_rate_left = self._force_to_rate(left_force)
        self.touch_rate_right = self._force_to_rate(right_force)

    def _force_to_rate(self, force):
        """Map force magnitude (N) to JO firing rate (Hz)."""
        excess = max(force - self.FORCE_FLOOR, 0.0)
        norm = min(excess / (self.FORCE_SAT - self.FORCE_FLOOR), 1.0)
        return norm * self.TOUCH_MAX_RATE

    # ── Vibration Processing ──────────────────────────────────────────────

    def process_vibration(self, fly_pos, fly_heading, sources):
        """Convert vibration sources to bilateral JO sound rates.

        Parameters
        ----------
        fly_pos : np.ndarray, shape (3,)
            Fly position in mm.
        fly_heading : float
            Fly heading angle in radians (0 = +x).
        sources : list of VibrationSource
            Active vibration sources.
        """
        if not sources:
            self.sound_rate_left = 0.0
            self.sound_rate_right = 0.0
            self.orientation_bias = 0.0
            return

        total_left = 0.0
        total_right = 0.0
        bias_num = 0.0
        bias_den = 0.0

        for src in sources:
            # Distance attenuation
            delta = src.position - fly_pos
            dist = np.linalg.norm(delta[:2])  # horizontal distance
            attenuation = src.amplitude / (1.0 + (dist / self.VIBRATION_RANGE) ** 2)

            if attenuation < 0.01:
                continue

            # Frequency tuning for JO-A (gaussian around courtship frequency)
            freq_gain = np.exp(
                -0.5 * ((src.frequency - self.COURTSHIP_CENTER) /
                         self.COURTSHIP_BW) ** 2)
            # JO-A responds with frequency selectivity, JO-B is broadband
            # Combined rate (weighted: 60% frequency-tuned, 40% broadband)
            effective = attenuation * (0.6 * freq_gain + 0.4)

            # Bilateral split based on source angle relative to fly heading
            source_angle = np.arctan2(delta[1], delta[0])
            relative_angle = source_angle - fly_heading
            # Normalize to [-pi, pi]
            relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi

            # Left antenna: more sensitive to left-side sources
            # Right antenna: more sensitive to right-side sources
            # Cosine-weighted split
            left_weight = 0.5 + 0.4 * np.sin(relative_angle)   # +sin = left
            right_weight = 0.5 - 0.4 * np.sin(relative_angle)  # -sin = right

            rate = effective * self.SOUND_MAX_RATE
            total_left += rate * max(left_weight, 0.0)
            total_right += rate * max(right_weight, 0.0)

            # Orientation bias (for bridge-level courtship turning)
            # Positive = source to the right, negative = source to the left
            bias_num += -np.sin(relative_angle) * effective * freq_gain
            bias_den += effective * freq_gain

        self.sound_rate_left = min(total_left, self.SOUND_MAX_RATE)
        self.sound_rate_right = min(total_right, self.SOUND_MAX_RATE)
        self.orientation_bias = (
            np.clip(bias_num / bias_den, -1.0, 1.0) if bias_den > 0.01
            else 0.0)

    # ── Brain Injection ───────────────────────────────────────────────────

    def get_rates(self):
        """Get combined (indices, rates) arrays for all JO neurons.

        Returns
        -------
        indices : np.ndarray (int64)
            Tensor indices of JO neurons to stimulate.
        rates : np.ndarray (float64)
            Firing rates in Hz for each neuron.
        """
        all_idx = []
        all_rates = []

        # Touch — left
        if len(self.touch_idx_left) > 0 and self.touch_rate_left > 0.1:
            all_idx.append(self.touch_idx_left)
            all_rates.append(
                np.full(len(self.touch_idx_left), self.touch_rate_left))

        # Touch — right
        if len(self.touch_idx_right) > 0 and self.touch_rate_right > 0.1:
            all_idx.append(self.touch_idx_right)
            all_rates.append(
                np.full(len(self.touch_idx_right), self.touch_rate_right))

        # Sound — left
        if len(self.sound_idx_left) > 0 and self.sound_rate_left > 0.1:
            all_idx.append(self.sound_idx_left)
            all_rates.append(
                np.full(len(self.sound_idx_left), self.sound_rate_left))

        # Sound — right
        if len(self.sound_idx_right) > 0 and self.sound_rate_right > 0.1:
            all_idx.append(self.sound_idx_right)
            all_rates.append(
                np.full(len(self.sound_idx_right), self.sound_rate_right))

        if not all_idx:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        return np.concatenate(all_idx), np.concatenate(all_rates)

    # ── Diagnostic Properties ─────────────────────────────────────────────

    @property
    def touch_level(self):
        """Normalized touch intensity [0-1] (max of both sides)."""
        return max(self.touch_rate_left, self.touch_rate_right) / self.TOUCH_MAX_RATE

    @property
    def sound_level(self):
        """Normalized sound intensity [0-1] (max of both sides)."""
        return max(self.sound_rate_left, self.sound_rate_right) / self.SOUND_MAX_RATE

    @property
    def is_tactile_escape(self):
        """True if contact force exceeds escape threshold."""
        return self.max_contact_force > self.FORCE_ESCAPE

    @property
    def is_grooming_touch(self):
        """True if contact force is in grooming range."""
        return (self.FORCE_GROOM < self.max_contact_force <= self.FORCE_ESCAPE)

    def get_status_str(self):
        """One-line diagnostic string."""
        parts = []
        if self.touch_level > 0.01:
            parts.append(
                f"touch={self.touch_level:.2f} "
                f"(L={self.touch_rate_left:.0f} R={self.touch_rate_right:.0f}Hz "
                f"maxF={self.max_contact_force:.1f}N)")
        if self.sound_level > 0.01:
            parts.append(
                f"sound={self.sound_level:.2f} "
                f"(L={self.sound_rate_left:.0f} R={self.sound_rate_right:.0f}Hz "
                f"bias={self.orientation_bias:+.2f})")
        return " | ".join(parts) if parts else "JO=silent"
