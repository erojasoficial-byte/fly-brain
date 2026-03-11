#!/usr/bin/env python3
"""
Wing Song System — Virtual wing vibration and sound production.

Drosophila males produce courtship songs by extending and vibrating one wing.
Since NeuroMechFly's wings are static (no joints in MJCF), we model song
production virtually: brain activity patterns trigger song generation, which
emits a vibration signal detectable by nearby flies' JO neurons.

Song types (based on real Drosophila):
  - Pulse song (~200Hz): interpulse interval courtship signal
  - Sine song (~160Hz): continuous low-frequency courtship
  - Alarm buzz (~400Hz): high-frequency distress signal

Architecture:
  DN activity → song mode selection → virtual VibrationSource at fly position
  → somatosensory system detects it → JO neurons fire → closed loop
"""

import numpy as np
from somatosensory import VibrationSource


# ============================================================================
# Song Definitions
# ============================================================================

SONG_TYPES = {
    'pulse': {
        'freq': 200.0,
        'amp': 0.7,
        'label': 'courtship pulse',
    },
    'sine': {
        'freq': 160.0,
        'amp': 0.5,
        'label': 'courtship sine',
    },
    'alarm': {
        'freq': 400.0,
        'amp': 0.9,
        'label': 'alarm buzz',
    },
}


# ============================================================================
# Wing Song System
# ============================================================================

class WingSongSystem:
    """Generates virtual wing vibration based on brain DN activity.

    The system monitors descending neuron rates and selects a song type:
      - MN9 (feeding/approach) above threshold -> courtship song
      - GF (escape) above threshold -> alarm buzz
      - Otherwise -> silent

    The song is emitted as a VibrationSource at the fly's position,
    which can be detected by somatosensory systems of nearby flies.

    Parameters
    ----------
    self_hearing_gain : float
        Attenuation factor for self-hearing (0-1). The fly hears its
        own song at this fraction of full amplitude. Default 0.2.
    """

    # Thresholds for song triggering from DN rates
    COURTSHIP_THRESH = 0.03   # MN9 (feed/approach) rate
    ALARM_THRESH = 0.15       # GF (escape) rate

    # Pulse/sine alternation period (seconds)
    SONG_CYCLE = 1.0    # total cycle length
    PULSE_FRAC = 0.6    # fraction of cycle that is pulse (vs sine)

    def __init__(self, self_hearing_gain=0.2):
        self.self_hearing_gain = self_hearing_gain

        # Current state
        self.active_song = None    # 'pulse', 'sine', 'alarm', or None
        self.wing_freq = 0.0      # Hz
        self.wing_amp = 0.0       # 0-1

        # Dynamic vibration source (created when singing)
        self.song_source = None    # VibrationSource or None

        # Timing
        self._song_timer = 0.0    # accumulated time in courtship
        self._prev_song = None    # for transition logging

    # ── Processing ─────────────────────────────────────────────────────────

    def process(self, decoder, fly_pos, dt):
        """Determine song mode from DN activity and update vibration.

        Parameters
        ----------
        decoder : DNRateDecoder
            For reading DN group rates.
        fly_pos : array-like, shape (3,)
            Current fly position in mm.
        dt : float
            Timestep in seconds.
        """
        gf_rate = decoder.get_group_rate('escape')
        mn9_rate = decoder.get_group_rate('feed')

        # Priority: alarm > courtship > silence
        if gf_rate > self.ALARM_THRESH:
            self.active_song = 'alarm'
            self._song_timer = 0.0
        elif mn9_rate > self.COURTSHIP_THRESH:
            self._song_timer += dt
            # Alternate pulse ↔ sine within each cycle
            phase = self._song_timer % self.SONG_CYCLE
            if phase < self.PULSE_FRAC * self.SONG_CYCLE:
                self.active_song = 'pulse'
            else:
                self.active_song = 'sine'
        else:
            self.active_song = None
            self._song_timer = 0.0

        # Update vibration
        if self.active_song and self.active_song in SONG_TYPES:
            song = SONG_TYPES[self.active_song]
            self.wing_freq = song['freq']
            self.wing_amp = song['amp']

            if self.song_source is None:
                self.song_source = VibrationSource(
                    position=fly_pos[:3].copy(),
                    frequency=self.wing_freq,
                    amplitude=self.wing_amp * self.self_hearing_gain,
                    label='wing_song',
                )
            else:
                self.song_source.position[:] = fly_pos[:3]
                self.song_source.frequency = self.wing_freq
                self.song_source.amplitude = (
                    self.wing_amp * self.self_hearing_gain)
        else:
            self.wing_freq = 0.0
            self.wing_amp = 0.0
            self.song_source = None

        # Log transitions
        if self.active_song != self._prev_song:
            if self.active_song:
                label = SONG_TYPES[self.active_song]['label']
                print(f"  >> Wing song: {label} "
                      f"({self.wing_freq:.0f}Hz, amp={self.wing_amp:.1f})")
            elif self._prev_song is not None:
                print("  >> Wing song: silent")
            self._prev_song = self.active_song

    # ── Vibration Sources ──────────────────────────────────────────────────

    def get_vibration_sources(self):
        """Return list of active vibration sources from wing song.

        These should be appended to the somatosensory system's
        vibration_sources list for detection by JO neurons.

        Returns
        -------
        list of VibrationSource
            Empty list if not singing, otherwise [song_source].
        """
        if self.song_source is not None:
            return [self.song_source]
        return []

    # ── Diagnostics ────────────────────────────────────────────────────────

    @property
    def is_singing(self):
        return self.active_song is not None

    @property
    def song_level(self):
        """Normalized song intensity [0-1]."""
        return self.wing_amp

    def get_status_str(self):
        """One-line diagnostic string."""
        if not self.is_singing:
            return ""
        return (f"WING={self.wing_freq:.0f}Hz({self.active_song}) "
                f"amp={self.wing_amp:.1f}")
