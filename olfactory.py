#!/usr/bin/env python3
"""
Olfactory System — Bilateral odor detection via antennal ORNs.

Drosophila detects odors through Olfactory Receptor Neurons (ORNs) on
the antennae. Different receptor types respond to different chemicals:
  - ORN_DM1 (~68 neurons, Or42b equivalent): food odors -> attraction
  - ORN_DA2 (~39 neurons, Or56a equivalent): geosmin -> aversion/escape

Architecture:
  Virtual odor sources emit concentration gradients (inverse-square falloff).
  Left and right antenna positions are computed from fly position + heading.
  Bilateral concentration asymmetry enables chemotaxis (gradient navigation).
"""

import csv
import numpy as np
from pathlib import Path


# ============================================================================
# Odor Source
# ============================================================================

class OdorSource:
    """A point source emitting an odor with distance-dependent concentration.

    Parameters
    ----------
    position : array-like
        [x, y, z] position in mm.
    odor_type : str
        'attractive' or 'repulsive'.
    amplitude : float
        Peak concentration (0-1) at the source.
    spread : float
        Characteristic radius in mm (concentration halves at this distance).
    label : str
        Human-readable name.
    """

    def __init__(self, position, odor_type, amplitude=1.0, spread=25.0,
                 label=''):
        self.position = np.array(position[:3], dtype=np.float64)
        self.odor_type = odor_type   # 'attractive' or 'repulsive'
        self.amplitude = float(amplitude)
        self.spread = float(spread)
        self.label = label or odor_type


# ============================================================================
# Olfactory System
# ============================================================================

class OlfactorySystem:
    """Bilateral olfactory detection using ORN populations from FlyWire.

    Parameters
    ----------
    flyid2i : dict
        FlyWire neuron ID -> tensor index mapping.
    annotations_path : str or Path, optional
        Path to FlyWire annotations TSV.
    """

    # Antenna offset from fly midline (mm). Exaggerated slightly
    # vs real anatomy (~0.15mm) to produce functional chemotaxis
    # gradients at simulation scale.
    ANTENNA_SPREAD = 2.0

    # Firing rates
    ATTRACTIVE_MAX_RATE = 180.0   # Hz (Or42b food)
    REPULSIVE_MAX_RATE = 250.0    # Hz (Or56a danger)

    # Floor: below this concentration, no activation
    CONC_FLOOR = 0.02

    # Repulsive escape threshold (normalized 0-1)
    REPULSION_ESCAPE_THRESH = 0.3

    def __init__(self, flyid2i, annotations_path=None):
        self.flyid2i = flyid2i

        if annotations_path is None:
            annotations_path = Path(__file__).parent / 'data' / 'flywire_annotations.tsv'

        self._load_orn_populations(str(annotations_path))

        # Runtime state
        self.conc_left_att = 0.0
        self.conc_right_att = 0.0
        self.conc_left_rep = 0.0
        self.conc_right_rep = 0.0
        self.attractive_rate_left = 0.0
        self.attractive_rate_right = 0.0
        self.repulsive_rate_left = 0.0
        self.repulsive_rate_right = 0.0
        self.active_source_label = ''

    # ── Population Loading ─────────────────────────────────────────────────

    def _load_orn_populations(self, annotations_path):
        """Load ORN_DM1 (attractive) and ORN_DA2 (repulsive) from annotations."""
        att_left, att_right = [], []
        rep_left, rep_right = [], []

        with open(annotations_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                ct = row.get('cell_type', '')
                if ct not in ('ORN_DM1', 'ORN_DA2'):
                    continue

                rid = int(row['root_id'])
                if rid not in self.flyid2i:
                    continue

                idx = self.flyid2i[rid]
                side = row.get('side', '')

                if ct == 'ORN_DM1':
                    if side == 'left':
                        att_left.append(idx)
                    elif side == 'right':
                        att_right.append(idx)
                elif ct == 'ORN_DA2':
                    if side == 'left':
                        rep_left.append(idx)
                    elif side == 'right':
                        rep_right.append(idx)

        self.att_idx_left = np.array(att_left, dtype=np.int64)
        self.att_idx_right = np.array(att_right, dtype=np.int64)
        self.rep_idx_left = np.array(rep_left, dtype=np.int64)
        self.rep_idx_right = np.array(rep_right, dtype=np.int64)

        n_att = len(att_left) + len(att_right)
        n_rep = len(rep_left) + len(rep_right)
        print(f"[Olfactory] Attractive ORN (DM1/Or42b): {n_att} neurons "
              f"(L={len(att_left)}, R={len(att_right)})")
        print(f"[Olfactory] Repulsive ORN (DA2/Or56a): {n_rep} neurons "
              f"(L={len(rep_left)}, R={len(rep_right)})")

    # ── Concentration Computation ──────────────────────────────────────────

    @staticmethod
    def _concentration_at(pos, sources, odor_type):
        """Compute total concentration at a position.

        Uses inverse-square falloff: c = amplitude / (1 + (d/spread)^2)
        """
        total = 0.0
        for src in sources:
            if src.odor_type != odor_type:
                continue
            dist = np.linalg.norm(pos[:2] - src.position[:2])
            conc = src.amplitude / (1.0 + (dist / src.spread) ** 2)
            total += conc
        return total

    # ── Main Processing ────────────────────────────────────────────────────

    def process(self, fly_pos, fly_heading, odor_sources):
        """Compute bilateral ORN activation from odor sources.

        Parameters
        ----------
        fly_pos : array-like, shape (3,)
            Fly position [x, y, z] in mm.
        fly_heading : float
            Fly yaw angle in radians.
        odor_sources : list of OdorSource
        """
        if not odor_sources:
            self.conc_left_att = 0.0
            self.conc_right_att = 0.0
            self.conc_left_rep = 0.0
            self.conc_right_rep = 0.0
            self._compute_rates()
            return

        # Antenna positions: perpendicular to heading
        # Left antenna: +90 deg from forward
        perp_x = -np.sin(fly_heading) * self.ANTENNA_SPREAD
        perp_y = np.cos(fly_heading) * self.ANTENNA_SPREAD

        pos_left = np.array([
            fly_pos[0] + perp_x, fly_pos[1] + perp_y, fly_pos[2]])
        pos_right = np.array([
            fly_pos[0] - perp_x, fly_pos[1] - perp_y, fly_pos[2]])

        # Compute concentrations at each antenna
        self.conc_left_att = self._concentration_at(
            pos_left, odor_sources, 'attractive')
        self.conc_right_att = self._concentration_at(
            pos_right, odor_sources, 'attractive')
        self.conc_left_rep = self._concentration_at(
            pos_left, odor_sources, 'repulsive')
        self.conc_right_rep = self._concentration_at(
            pos_right, odor_sources, 'repulsive')

        self._compute_rates()

        # Find most active source for labeling
        self.active_source_label = ''
        max_conc = 0.0
        for src in odor_sources:
            if src.odor_type == 'attractive':
                c = self.conc_left_att + self.conc_right_att
            else:
                c = self.conc_left_rep + self.conc_right_rep
            if c > max_conc:
                max_conc = c
                self.active_source_label = src.label

    def _compute_rates(self):
        """Map concentrations to ORN firing rates."""
        # Attractive ORN_DM1
        for attr_name, conc_attr in [
            ('attractive_rate_left', 'conc_left_att'),
            ('attractive_rate_right', 'conc_right_att'),
        ]:
            c = getattr(self, conc_attr)
            if c > self.CONC_FLOOR:
                setattr(self, attr_name,
                        min(c, 1.0) * self.ATTRACTIVE_MAX_RATE)
            else:
                setattr(self, attr_name, 0.0)

        # Repulsive ORN_DA2
        for attr_name, conc_attr in [
            ('repulsive_rate_left', 'conc_left_rep'),
            ('repulsive_rate_right', 'conc_right_rep'),
        ]:
            c = getattr(self, conc_attr)
            if c > self.CONC_FLOOR:
                setattr(self, attr_name,
                        min(c, 1.0) * self.REPULSIVE_MAX_RATE)
            else:
                setattr(self, attr_name, 0.0)

    # ── Brain Injection ────────────────────────────────────────────────────

    def get_rates(self):
        """Get combined (indices, rates) arrays for brain injection.

        Returns
        -------
        indices : np.ndarray (int64)
        rates : np.ndarray (float64)
        """
        all_idx = []
        all_rates = []

        if self.attractive_rate_left > 0.1 and len(self.att_idx_left) > 0:
            all_idx.append(self.att_idx_left)
            all_rates.append(
                np.full(len(self.att_idx_left), self.attractive_rate_left))

        if self.attractive_rate_right > 0.1 and len(self.att_idx_right) > 0:
            all_idx.append(self.att_idx_right)
            all_rates.append(
                np.full(len(self.att_idx_right), self.attractive_rate_right))

        if self.repulsive_rate_left > 0.1 and len(self.rep_idx_left) > 0:
            all_idx.append(self.rep_idx_left)
            all_rates.append(
                np.full(len(self.rep_idx_left), self.repulsive_rate_left))

        if self.repulsive_rate_right > 0.1 and len(self.rep_idx_right) > 0:
            all_idx.append(self.rep_idx_right)
            all_rates.append(
                np.full(len(self.rep_idx_right), self.repulsive_rate_right))

        if not all_idx:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        return np.concatenate(all_idx), np.concatenate(all_rates)

    # ── Diagnostics ────────────────────────────────────────────────────────

    @property
    def attractive_level(self):
        """Normalized attractive intensity [0-1]."""
        return (max(self.attractive_rate_left, self.attractive_rate_right)
                / self.ATTRACTIVE_MAX_RATE)

    @property
    def repulsive_level(self):
        """Normalized repulsive intensity [0-1]."""
        return (max(self.repulsive_rate_left, self.repulsive_rate_right)
                / self.REPULSIVE_MAX_RATE)

    @property
    def is_repulsive_escape(self):
        """True when repulsive concentration triggers escape."""
        return self.repulsive_level > self.REPULSION_ESCAPE_THRESH

    @property
    def attraction_bias(self):
        """Orientation bias for attractive chemotaxis.

        +1 = more on right antenna -> turn right toward source.
        -1 = more on left -> turn left toward source.
        """
        total = self.conc_left_att + self.conc_right_att
        if total < self.CONC_FLOOR * 2:
            return 0.0
        return (self.conc_right_att - self.conc_left_att) / total

    @property
    def repulsion_bias(self):
        """Orientation bias for repulsive escape.

        +1 = more threat on right antenna.
        -1 = more threat on left.
        Bridge should turn AWAY from this direction.
        """
        total = self.conc_left_rep + self.conc_right_rep
        if total < self.CONC_FLOOR * 2:
            return 0.0
        return (self.conc_right_rep - self.conc_left_rep) / total

    def get_status_str(self):
        """One-line diagnostic string."""
        parts = []
        if self.attractive_level > 0.01:
            parts.append(
                f"FOOD={self.attractive_level:.2f} "
                f"[L={self.conc_left_att:.2f} R={self.conc_right_att:.2f}]")
        if self.repulsive_level > 0.01:
            parts.append(
                f"DANGER={self.repulsive_level:.2f} "
                f"[L={self.conc_left_rep:.2f} R={self.conc_right_rep:.2f}]")
        return " | ".join(parts) if parts else ""
