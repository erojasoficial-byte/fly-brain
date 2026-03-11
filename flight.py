"""
Virtual Flight System: Simulates Drosophila flight via external forces in MuJoCo.

flygym has no flight mode -- wings are static meshes without joints or actuators.
We implement flight pragmatically:
- Translation: xfrc_applied forces on Thorax (lift, thrust, drag)
- Orientation: direct qpos override on the free joint (rigid heading lock)

This separation guarantees zero spinning regardless of articulated body dynamics.
The Giant Fiber (GF > threshold) triggers emergency takeoff.
Escape flight is ballistic: heading is locked at takeoff.

Flight states: GROUNDED -> TAKEOFF -> FLYING -> LANDING -> GROUNDED
"""

import numpy as np
from enum import IntEnum


class FlightState(IntEnum):
    GROUNDED = 0
    TAKEOFF = 1
    FLYING = 2
    LANDING = 3


class FlightSystem:
    """State machine that computes translational forces for virtual flight.

    Orientation is NOT controlled via torques (they fight the articulated
    body dynamics and cause spinning). Instead, the caller overrides qpos
    quaternion directly after each sim.step() for rigid heading lock.
    """

    # Force multipliers (fractions of mg)
    TAKEOFF_LIFT = 1.4
    LANDING_LIFT = 0.5       # partial lift → net downward, controlled descent
    THRUST_GAIN = 0.3

    # Altitude control
    TARGET_ALT = 5.0      # mm
    ALT_P_GAIN = 0.12

    # Aerodynamic drag (fractions of mg per mm/s)
    DRAG_LINEAR = 0.008
    DRAG_VERTICAL = 0.015

    # Force clamp (fractions of mg)
    MAX_FORCE = 3.0

    # Wing beat frequencies (display only)
    WING_FREQ_FLIGHT = 200.0
    WING_FREQ_TAKEOFF = 250.0

    # Altitude thresholds (mm)
    ALT_AIRBORNE = 3.0
    ALT_LANDED = 2.0       # fly naturally stands at ~1.2-1.8mm
    ALT_MAX = 15.0

    # Post-landing cooldown (seconds) — prevents immediate re-takeoff
    LANDING_COOLDOWN = 3.0

    def __init__(self, total_mass, gravity=9810.0,
                 takeoff_thresh=0.06, land_thresh=0.06):
        self.total_mass = total_mass
        self.gravity = gravity
        self.takeoff_thresh = takeoff_thresh
        self.land_thresh = land_thresh
        self.mg = total_mass * gravity

        self.state = FlightState.GROUNDED
        self.force_torque = np.zeros(6)
        self.altitude = 0.0
        self.wing_freq = 0.0
        self._takeoff_time = 0.0
        self._landing_time = 0.0
        self._cooldown = 99.0  # start high so first takeoff is allowed
        self._prev_pos = None
        self._velocity = np.zeros(3)

        # Ballistic escape heading (locked at takeoff)
        self._escape_heading = np.array([1.0, 0.0])
        self._escape_yaw = 0.0

    def get_desired_quat(self):
        """Return target quaternion: upright body + escape heading yaw."""
        half = self._escape_yaw / 2.0
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])

    def update(self, decoder, fly_pos, fly_forward, dt):
        """Update state machine and compute translational forces only."""
        d = decoder
        fly_pos = np.asarray(fly_pos, dtype=np.float64)

        gf = np.mean([d.get_normalized('GF_1'), d.get_normalized('GF_2')])
        p9 = np.mean([
            d.get_normalized('P9_left'), d.get_normalized('P9_right'),
            d.get_normalized('P9_oDN1_left'), d.get_normalized('P9_oDN1_right'),
        ])

        self.altitude = max(float(fly_pos[2]), 0.0)

        if self._prev_pos is not None and dt > 0:
            self._velocity = (fly_pos - self._prev_pos) / dt
        self._prev_pos = fly_pos.copy()

        # ── State transitions ──
        if self.state == FlightState.GROUNDED:
            self._cooldown += dt
            if gf > self.takeoff_thresh and self._cooldown > self.LANDING_COOLDOWN:
                self.state = FlightState.TAKEOFF
                self._takeoff_time = 0.0
                # Lock escape heading (ballistic flight)
                fwd = np.array(fly_forward[:2], dtype=np.float64)
                n = np.linalg.norm(fwd)
                if n > 1e-6:
                    self._escape_heading = fwd / n
                else:
                    self._escape_heading = np.array([1.0, 0.0])
                self._escape_yaw = np.arctan2(
                    self._escape_heading[1], self._escape_heading[0])

        elif self.state == FlightState.TAKEOFF:
            self._takeoff_time += dt
            if self.altitude > self.ALT_AIRBORNE:
                self.state = FlightState.FLYING

        elif self.state == FlightState.FLYING:
            if gf < self.land_thresh:
                self.state = FlightState.LANDING
                self._landing_time = 0.0

        elif self.state == FlightState.LANDING:
            self._landing_time += dt
            if self.altitude <= self.ALT_LANDED or self._landing_time > 3.0:
                self.state = FlightState.GROUNDED
                self._cooldown = 0.0  # start cooldown timer

        # ── Compute forces (no torques — orientation via qpos) ──
        self.force_torque[:] = 0.0

        if self.state == FlightState.TAKEOFF:
            self.force_torque[2] = self.mg * self.TAKEOFF_LIFT
            thrust = self.mg * self.THRUST_GAIN * max(gf, 0.2)
            self.force_torque[0] += thrust * self._escape_heading[0]
            self.force_torque[1] += thrust * self._escape_heading[1]
            self.wing_freq = self.WING_FREQ_TAKEOFF

        elif self.state == FlightState.FLYING:
            alt_error = self.TARGET_ALT - self.altitude
            hover = 1.0 + self.ALT_P_GAIN * alt_error
            if self.altitude > self.ALT_MAX:
                hover = min(hover, 0.5)
            hover = np.clip(hover, 0.3, 1.5)
            self.force_torque[2] = self.mg * hover

            thrust_cmd = max(p9, gf * 0.5)
            thrust = self.mg * self.THRUST_GAIN * thrust_cmd
            self.force_torque[0] += thrust * self._escape_heading[0]
            self.force_torque[1] += thrust * self._escape_heading[1]
            self.wing_freq = self.WING_FREQ_FLIGHT

        elif self.state == FlightState.LANDING:
            # Partial lift that decreases with altitude → controlled descent
            alt_factor = max(0.0, 1.0 - self.altitude / self.ALT_MAX)
            self.force_torque[2] = self.mg * self.LANDING_LIFT * alt_factor
            self.wing_freq = self.WING_FREQ_FLIGHT * 0.8 * alt_factor

        else:
            self.wing_freq = 0.0
            return

        # Aerodynamic drag
        v = self._velocity
        self.force_torque[0] -= self.DRAG_LINEAR * self.mg * v[0]
        self.force_torque[1] -= self.DRAG_LINEAR * self.mg * v[1]
        self.force_torque[2] -= self.DRAG_VERTICAL * self.mg * v[2]

        # Clamp forces
        fmax = self.MAX_FORCE * self.mg
        self.force_torque[:3] = np.clip(self.force_torque[:3], -fmax, fmax)

    @property
    def is_airborne(self):
        return self.state != FlightState.GROUNDED

    @property
    def flight_level(self):
        if self.state == FlightState.TAKEOFF:
            return 1.0
        elif self.state == FlightState.FLYING:
            return 0.7
        elif self.state == FlightState.LANDING:
            return 0.3
        return 0.0

    def get_status_str(self):
        return (f"FLT={self.state.name.lower()} "
                f"ALT={self.altitude:.1f}mm "
                f"WING={self.wing_freq:.0f}Hz")
