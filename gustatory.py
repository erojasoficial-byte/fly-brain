#!/usr/bin/env python3
"""
Gustatory System — Taste zone detection via tarsal contact.

Drosophila taste with GRN (Gustatory Receptor Neurons) on tarsi:
  Sugar GRNs (21 neurons) → feeding / approach
  Bitter GRNs (42 neurons) → aversion / escape

Architecture:
  MuJoCo end_effector positions (6 legs) checked against taste zones
  on the arena floor. Legs touching a zone activate the corresponding
  GRN population with rate proportional to contact strength.
"""

import numpy as np
from brain_body_bridge import STIMULI


# ============================================================================
# Taste Zone
# ============================================================================

class TasteZone:
    """A circular zone on the arena floor with a taste.

    Parameters
    ----------
    center : array-like
        [x, y] position in mm (arena coordinates).
    radius : float
        Zone radius in mm.
    taste : str
        'sugar' or 'bitter'.
    label : str
        Human-readable name.
    """

    def __init__(self, center, radius, taste, label=''):
        self.center = np.array(center[:2], dtype=np.float64)
        self.radius = float(radius)
        self.taste = taste
        self.label = label or taste


# ============================================================================
# Gustatory System
# ============================================================================

class GustatorySystem:
    """Detects tarsal contact with taste zones and activates GRN populations.

    Parameters
    ----------
    flyid2i : dict
        FlyWire neuron ID → tensor index mapping.
    zones : list of TasteZone
        Taste zones in the arena.
    """

    LEG_NAMES = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

    # Ground contact threshold: only taste when foot is near ground
    GROUND_Z_THRESH = 0.5  # mm above ground

    # Firing rates
    SUGAR_MAX_RATE = 200.0   # Hz
    BITTER_MAX_RATE = 250.0  # Hz

    def __init__(self, flyid2i, zones):
        self.zones = zones
        self.flyid2i = flyid2i

        # Resolve neuron indices from STIMULI dict
        self.sugar_indices = np.array(
            [flyid2i[nid] for nid in STIMULI['sugar']['neurons']
             if nid in flyid2i], dtype=np.int64)
        self.bitter_indices = np.array(
            [flyid2i[nid] for nid in STIMULI['bitter']['neurons']
             if nid in flyid2i], dtype=np.int64)

        # Runtime state
        self.sugar_legs = []       # list of leg names touching sugar
        self.bitter_legs = []      # list of leg names touching bitter
        self.sugar_rate = 0.0
        self.bitter_rate = 0.0
        self.active_zone_label = ''  # label of most-activated zone

        print(f"[Gustatory] Sugar GRNs: {len(self.sugar_indices)} neurons")
        print(f"[Gustatory] Bitter GRNs: {len(self.bitter_indices)} neurons")
        print(f"[Gustatory] {len(zones)} taste zones:")
        for z in zones:
            print(f"  '{z.label}' ({z.taste}) at "
                  f"[{z.center[0]:.0f},{z.center[1]:.0f}]mm r={z.radius:.0f}mm")

    # ── Zone Detection ────────────────────────────────────────────────────

    def process(self, end_effectors):
        """Detect which legs are in which taste zones.

        Parameters
        ----------
        end_effectors : np.ndarray, shape (6, 3)
            Tarsal tip positions [x, y, z] per leg in mm.
        """
        self.sugar_legs = []
        self.bitter_legs = []
        self.active_zone_label = ''

        best_sugar_count = 0
        best_bitter_count = 0

        for zone in self.zones:
            legs_in = []
            for leg_idx in range(6):
                pos = end_effectors[leg_idx]
                # Only count if foot is on ground
                if pos[2] > self.GROUND_Z_THRESH:
                    continue
                dist = np.linalg.norm(pos[:2] - zone.center)
                if dist < zone.radius:
                    legs_in.append(self.LEG_NAMES[leg_idx])

            if zone.taste == 'sugar' and len(legs_in) > best_sugar_count:
                self.sugar_legs = legs_in
                best_sugar_count = len(legs_in)
                if legs_in:
                    self.active_zone_label = zone.label
            elif zone.taste == 'bitter' and len(legs_in) > best_bitter_count:
                self.bitter_legs = legs_in
                best_bitter_count = len(legs_in)
                if legs_in:
                    self.active_zone_label = zone.label

        # Compute rates: scales with number of legs (2+ legs = full rate)
        self.sugar_rate = (
            self.SUGAR_MAX_RATE * min(len(self.sugar_legs) / 2.0, 1.0))
        self.bitter_rate = (
            self.BITTER_MAX_RATE * min(len(self.bitter_legs) / 2.0, 1.0))

    # ── Brain Injection ───────────────────────────────────────────────────

    def get_rates(self):
        """Get combined (indices, rates) arrays for GRN injection.

        Returns
        -------
        indices : np.ndarray (int64)
        rates : np.ndarray (float64)
        """
        all_idx = []
        all_rates = []

        if self.sugar_rate > 0.1 and len(self.sugar_indices) > 0:
            all_idx.append(self.sugar_indices)
            all_rates.append(
                np.full(len(self.sugar_indices), self.sugar_rate))

        if self.bitter_rate > 0.1 and len(self.bitter_indices) > 0:
            all_idx.append(self.bitter_indices)
            all_rates.append(
                np.full(len(self.bitter_indices), self.bitter_rate))

        if not all_idx:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        return np.concatenate(all_idx), np.concatenate(all_rates)

    # ── Diagnostics ───────────────────────────────────────────────────────

    @property
    def sugar_active(self):
        return len(self.sugar_legs) > 0

    @property
    def bitter_active(self):
        return len(self.bitter_legs) > 0

    @property
    def sugar_level(self):
        """Normalized sugar intensity [0-1]."""
        return self.sugar_rate / self.SUGAR_MAX_RATE

    @property
    def bitter_level(self):
        """Normalized bitter intensity [0-1]."""
        return self.bitter_rate / self.BITTER_MAX_RATE

    def get_status_str(self):
        """One-line diagnostic string."""
        parts = []
        if self.sugar_active:
            legs = ','.join(self.sugar_legs)
            parts.append(f"SUGAR={self.sugar_rate:.0f}Hz [{legs}]")
        if self.bitter_active:
            legs = ','.join(self.bitter_legs)
            parts.append(f"BITTER={self.bitter_rate:.0f}Hz [{legs}]")
        return " | ".join(parts) if parts else ""
