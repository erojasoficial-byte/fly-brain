"""
Two Flies — One World, Independent Minds.

Two Drosophila sharing one physical world (single MuJoCo scene).
Each has its own 138,639-neuron brain with independent Hebbian plasticity.
They perceive each other naturally — through vision, touch, sound — because
they exist as physical bodies in the same simulation.

Usage:
    python two_flies.py --visual --somatosensory --vocalize
    python two_flies.py --visual --somatosensory --olfactory --gustatory --vocalize --duration 30
"""

import sys
import argparse
import time as _time
from types import SimpleNamespace

import numpy as np
import mujoco
import mujoco.viewer
from scipy.interpolate import interp1d

from flygym import Fly
from flygym.simulation import Simulation
from flygym.examples.locomotion import PreprogrammedSteps, CPGNetwork
from flygym.arena import FlatTerrain
from dm_control import mjcf

from brain_body_bridge import (
    BrainEngine, DNRateDecoder, BrainBodyBridge, STIMULI, DN_GROUPS,
)
from visual_system import VisualSystem
from looming_arena import LoomingArena
from procedural_arena import ProceduralArena
from somatosensory import SomatosensorySystem, VibrationSource
from gustatory import GustatorySystem, TasteZone
from olfactory import OlfactorySystem, OdorSource
from vocalization import WingSongSystem
from consciousness import ConsciousnessDetector


# ============================================================================
# CrossFlySimulation — adds explicit inter-fly contact pairs before compile
# ============================================================================

# Body segments for inter-fly collision.
# Core body (7) + coxa (6) = 13 per fly → 13×13 = 169 cross pairs.
# Keeps physics overhead manageable while covering the main collision volume.
_CROSS_COLLISION_GEOMS = [
    'Thorax', 'A1A2', 'A3', 'A4', 'A5', 'A6', 'Head',
    'LFCoxa', 'LMCoxa', 'LHCoxa',
    'RFCoxa', 'RMCoxa', 'RHCoxa',
]


class CrossFlySimulation(Simulation):
    """Simulation with inter-fly collision via explicit contact pairs.

    MuJoCo mesh-mesh collision only works through predefined <contact><pair>
    elements (not via contype/conaffinity filtering). This subclass injects
    cross-fly contact pairs into the MJCF before compilation.
    """

    def __init__(self, flies, cameras=None, arena=None, timestep=1e-4,
                 gravity=(0, 0, -9.81e3)):
        if isinstance(flies, list):
            self.flies = flies
        else:
            self.flies = list(flies)

        from flygym.camera import Camera
        if cameras is None:
            self.cameras = [
                Camera(
                    attachment_point=self.flies[0].model.worldbody,
                    camera_name="camera_left",
                )
            ]
        elif isinstance(cameras, list):
            self.cameras = cameras
        else:
            self.cameras = [cameras]

        self.arena = arena if arena is not None else FlatTerrain()
        self.timestep = timestep
        self.curr_time = 0.0
        self._floor_height = self.arena._get_max_floor_height()

        for fly in self.flies:
            self.arena.spawn_entity(
                fly.model, fly.spawn_pos, fly.spawn_orientation)

        arena_root = self.arena.root_element
        arena_root.option.timestep = timestep

        for fly in self.flies:
            fly.init_floor_contacts(self.arena)

        # === INJECT CROSS-FLY CONTACT PAIRS ===
        n_pairs = 0
        if len(self.flies) >= 2:
            n_pairs = self._add_cross_fly_contacts(arena_root)

        self.physics = mjcf.Physics.from_mjcf_model(arena_root)

        for camera in self.cameras:
            camera.init_camera_orientation(self.physics)

        self.gravity = gravity
        self._set_init_pose()

        for fly in self.flies:
            fly.post_init(self)

        print(f"  Inter-fly collision: {n_pairs} contact pairs injected")

    def _add_cross_fly_contacts(self, arena_root):
        """Add explicit contact pairs between fly0 and fly1 body geoms."""
        fly0, fly1 = self.flies[0], self.flies[1]
        n = 0
        for g0 in _CROSS_COLLISION_GEOMS:
            for g1 in _CROSS_COLLISION_GEOMS:
                pair_name = f"cross_{fly0.name}_{g0}_{fly1.name}_{g1}"
                geom1_name = f"{fly0.name}/{g0}"
                geom2_name = f"{fly1.name}/{g1}"
                arena_root.contact.add(
                    "pair",
                    name=pair_name,
                    geom1=geom1_name,
                    geom2=geom2_name,
                    solref=[2e-4, 1e3],
                    solimp=[9.99e-1, 9.999e-1, 1e-3, 5e-1, 2.0],
                    margin=0.0,
                )
                n += 1
        return n


# ============================================================================
# CPG Constants (from flygym.examples.locomotion.turning_controller)
# ============================================================================

_tripod_phase_biases = np.pi * np.array(
    [
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ]
)
_tripod_coupling_weights = (_tripod_phase_biases > 0) * 10

_default_correction_vectors = {
    "F": np.array([-0.03, 0, 0, -0.03, 0, 0.03, 0.03]),
    "M": np.array([-0.015, 0.001, 0.025, -0.02, 0, -0.02, 0.0]),
    "H": np.array([0, 0, 0, -0.02, 0, 0.01, -0.02]),
}

_default_correction_rates = {"retraction": (800, 700), "stumbling": (2200, 1800)}


# ============================================================================
# WalkingCPG — Extracted from HybridTurningController
# ============================================================================

def _find_stumbling_sensors(fly):
    """Find stumbling sensor indices from a Fly's contact placements."""
    stumble_segments = ("Tibia", "Tarsus1", "Tarsus2")
    steps = PreprogrammedSteps()
    stumbling_sensors = {leg: [] for leg in steps.legs}
    for i, sensor_name in enumerate(fly.contact_sensor_placements):
        # sensor_name: e.g. "Animat/LFTarsus1" or just "LFTarsus1"
        base = sensor_name.split("/")[-1]
        leg = base[:2]
        segment = base[2:]
        if segment in stumble_segments and leg in stumbling_sensors:
            stumbling_sensors[leg].append(i)
    stumbling_sensors = {k: np.array(v) for k, v in stumbling_sensors.items()}
    return stumbling_sensors


def _init_phasic_gain(preprogrammed_steps, swing_extension=np.pi / 4):
    """Create phase-dependent correction gain interpolators."""
    phasic_multiplier = {}
    for leg in preprogrammed_steps.legs:
        swing_start, swing_end = preprogrammed_steps.swing_period[leg]
        step_points = [
            swing_start,
            np.mean([swing_start, swing_end]),
            swing_end + swing_extension,
            np.mean([swing_end, 2 * np.pi]),
            2 * np.pi,
        ]
        preprogrammed_steps.swing_period[leg] = (
            swing_start,
            swing_end + swing_extension,
        )
        increment_vals = [0, 0.8, 0, -0.1, 0]
        phasic_multiplier[leg] = interp1d(
            step_points, increment_vals, kind="linear", fill_value="extrapolate"
        )
    return phasic_multiplier


class WalkingCPG:
    """Per-fly CPG locomotion controller.

    Same CPG network + retraction/stumbling correction rules as
    HybridTurningController, but decoupled from Simulation.
    """

    def __init__(self, fly, timestep, seed=0):
        self.timestep = timestep
        self.preprogrammed_steps = PreprogrammedSteps()
        self.cpg_network = CPGNetwork(
            timestep=timestep,
            intrinsic_freqs=np.ones(6) * 12,
            intrinsic_amps=np.ones(6) * 1,
            coupling_weights=_tripod_coupling_weights,
            phase_biases=_tripod_phase_biases,
            convergence_coefs=np.ones(6) * 20,
            seed=seed,
        )
        self.intrinsic_freqs = np.ones(6) * 12
        self.right_leg_inversion = [1, -1, -1, 1, -1, 1, 1]

        self.retraction_correction = np.zeros(6)
        self.stumbling_correction = np.zeros(6)
        self.retraction_persistence_counter = np.zeros(6)

        self.stumbling_sensors = _find_stumbling_sensors(fly)
        self.phasic_multiplier = _init_phasic_gain(self.preprogrammed_steps)

        self.stumbling_force_threshold = -1
        self.max_increment = 80 / 1e-4 * timestep
        self.retraction_persistence_duration = 20 / 1e-4 * timestep
        self.retraction_persistence_initiation_threshold = 20 / 1e-4 * timestep
        self.correction_vectors = _default_correction_vectors
        self.correction_rates = _default_correction_rates

    def _retraction_rule_find_leg(self, obs):
        end_effector_z_pos = obs["fly"][0][2] - obs["end_effectors"][:, 2]
        end_effector_z_pos_sorted_idx = np.argsort(end_effector_z_pos)
        end_effector_z_pos_sorted = end_effector_z_pos[end_effector_z_pos_sorted_idx]
        if end_effector_z_pos_sorted[-1] > end_effector_z_pos_sorted[-3] + 0.05:
            leg_to_correct_retraction = end_effector_z_pos_sorted_idx[-1]
            if (
                self.retraction_correction[leg_to_correct_retraction]
                > self.retraction_persistence_initiation_threshold
            ):
                self.retraction_persistence_counter[leg_to_correct_retraction] = 1
        else:
            leg_to_correct_retraction = None
        return leg_to_correct_retraction

    def _update_persistence_counter(self):
        self.retraction_persistence_counter[
            self.retraction_persistence_counter > 0
        ] += 1
        self.retraction_persistence_counter[
            self.retraction_persistence_counter > self.retraction_persistence_duration
        ] = 0

    def _stumbling_rule_check_condition(self, obs, leg):
        contact_forces = obs["contact_forces"][self.stumbling_sensors[leg], :]
        fly_orientation = obs["fly_orientation"]
        force_proj = np.dot(contact_forces, fly_orientation)
        return (force_proj < self.stumbling_force_threshold).any()

    def _get_net_correction(self, retraction_correction, stumbling_correction):
        if retraction_correction > 0:
            return retraction_correction, True
        return stumbling_correction, False

    def _update_correction_amount(self, condition, curr_amount, correction_rates):
        if condition:
            new_amount = curr_amount + correction_rates[0] * self.timestep
        else:
            new_amount = max(0, curr_amount - correction_rates[1] * self.timestep)
        return new_amount, condition

    def compute_action(self, obs, left_drive, right_drive):
        """Convert [left_drive, right_drive] to joints+adhesion action dict."""
        action = np.array([left_drive, right_drive])

        # Modulate CPG amps/freqs from drives
        amps = np.repeat(np.abs(action[:, np.newaxis]), 3, axis=1).ravel()
        freqs = self.intrinsic_freqs.copy()
        freqs[:3] *= 1 if action[0] > 0 else -1
        freqs[3:] *= 1 if action[1] > 0 else -1
        self.cpg_network.intrinsic_amps = amps
        self.cpg_network.intrinsic_freqs = freqs

        # Retraction rule
        leg_to_correct_retraction = self._retraction_rule_find_leg(obs)
        self._update_persistence_counter()
        persistent_retraction = self.retraction_persistence_counter > 0

        self.cpg_network.step()

        joints_angles = []
        adhesion_onoff = []
        for i, leg in enumerate(self.preprogrammed_steps.legs):
            # Retraction correction
            retraction_correction, _ = self._update_correction_amount(
                condition=(
                    (i == leg_to_correct_retraction) or persistent_retraction[i]
                ),
                curr_amount=self.retraction_correction[i],
                correction_rates=self.correction_rates["retraction"],
            )
            self.retraction_correction[i] = retraction_correction

            # Stumbling correction
            self.stumbling_correction[i], _ = self._update_correction_amount(
                condition=self._stumbling_rule_check_condition(obs, leg),
                curr_amount=self.stumbling_correction[i],
                correction_rates=self.correction_rates["stumbling"],
            )

            # Net correction (retraction has priority)
            net_correction, reset_stumbling = self._get_net_correction(
                self.retraction_correction[i], self.stumbling_correction[i]
            )
            if reset_stumbling:
                self.stumbling_correction[i] = 0.0

            net_correction = np.clip(net_correction, 0, self.max_increment)
            if leg[0] == "R":
                net_correction *= self.right_leg_inversion[i]

            net_correction *= self.phasic_multiplier[leg](
                self.cpg_network.curr_phases[i] % (2 * np.pi)
            )

            my_joints_angles = self.preprogrammed_steps.get_joint_angles(
                leg,
                self.cpg_network.curr_phases[i],
                self.cpg_network.curr_magnitudes[i],
            )
            my_joints_angles += net_correction * self.correction_vectors[leg[1]]
            joints_angles.append(my_joints_angles)

            my_adhesion_onoff = self.preprogrammed_steps.get_adhesion_onoff(
                leg, self.cpg_network.curr_phases[i]
            )
            adhesion_onoff.append(my_adhesion_onoff)

        return {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff).astype(int),
        }


# ============================================================================
# Grooming Controller (from fly_embodied.py)
# ============================================================================

class GroomingController:
    """Generates front-leg oscillation for antennal grooming behavior."""

    def __init__(self, preprogrammed_steps=None, freq_hz=4.0):
        self.steps = preprogrammed_steps or PreprogrammedSteps()
        self.freq = freq_hz
        self.neutral = np.zeros(42)
        for i, leg in enumerate(self.steps.legs):
            self.neutral[i * 7:(i + 1) * 7] = self.steps.get_joint_angles(
                leg, np.pi, 0.0
            )

    def get_action(self, time_s):
        """Return joints+adhesion action dict for grooming at given time."""
        joints = self.neutral.copy()
        phase = 2 * np.pi * self.freq * time_s
        femur_offset = 0.3 * np.sin(phase)
        tibia_offset = 0.4 * np.sin(phase + np.pi / 2)
        for base in (0, 21):  # LF and RF
            joints[base + 3] += femur_offset
            joints[base + 5] += tibia_offset
        adhesion = np.array([0, 1, 1, 0, 1, 1])
        return {"joints": joints, "adhesion": adhesion}


# ============================================================================
# Vision Rendering Helper
# ============================================================================

def render_fly_eyes(pipe, eye_renderer, sim, fly_obs):
    """Render compound eyes for one fly and inject visual rates into brain."""
    model_ptr = sim.physics.model.ptr
    data_ptr = sim.physics.data.ptr
    retina = pipe.fly.retina

    # Hide self-geoms to avoid occlusion
    saved_alpha = []
    for gid in pipe.geom_hide_ids:
        saved_alpha.append(model_ptr.geom_rgba[gid, 3].copy())
        model_ptr.geom_rgba[gid, 3] = 0.0

    readouts = []
    for side in ["L", "R"]:
        cid = pipe.eye_cam_ids.get(side, -1)
        if cid < 0:
            readouts.append(np.zeros((721, 2), dtype=np.float32))
            continue
        eye_renderer.update_scene(data_ptr, camera=cid)
        raw_img = eye_renderer.render()
        fish_img = retina.correct_fisheye(raw_img)
        hex_pxls = retina.raw_image_to_hex_pxls(fish_img)
        readouts.append(hex_pxls)

    # Restore self-geoms
    for i, gid in enumerate(pipe.geom_hide_ids):
        model_ptr.geom_rgba[gid, 3] = saved_alpha[i]

    vision_obs = np.array(readouts, dtype=np.float32)

    # Inject all visual layers into brain
    vis_idx, vis_rates = pipe.visual.process_visual_layers(vision_obs)
    pipe.cached_visual = (vis_idx, vis_rates)
    pipe.brain.set_visual_rates(vis_idx, vis_rates)


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Two Flies — One World, Independent Minds')
    parser.add_argument('--no-viewer', action='store_true',
                        help='Run headless (no MuJoCo viewer)')
    parser.add_argument('--duration', type=float, default=0.0,
                        help='Max sim duration in seconds (0 = unlimited)')
    parser.add_argument('--visual', action='store_true',
                        help='Enable compound eye vision -> connectome')
    parser.add_argument('--somatosensory', action='store_true',
                        help='Enable touch + sound via JO neurons')
    parser.add_argument('--gustatory', action='store_true',
                        help='Enable taste zones (sugar/bitter)')
    parser.add_argument('--olfactory', action='store_true',
                        help='Enable olfactory system')
    parser.add_argument('--vocalize', action='store_true',
                        help='Enable wing song production')
    parser.add_argument('--separation', type=float, default=5.0,
                        help='Initial distance between flies in mm')
    parser.add_argument('--ball', action='store_true',
                        help='Add looming ball (LoomingArena)')
    parser.add_argument('--consciousness', action='store_true',
                        help='Enable per-fly consciousness detection (CI)')
    parser.add_argument('--fresh', action='store_true',
                        help='Ignore saved plastic weights (start fresh)')
    parser.add_argument('--flat', action='store_true',
                        help='Use flat terrain (no procedural world)')
    parser.add_argument('--world-seed', type=int, default=42,
                        help='Procedural world seed')
    return parser.parse_args()


# ============================================================================
# Physics Watchdog — detect NaN/freeze, recover flies
# ============================================================================

class PhysicsWatchdog:
    """Immediate divergence detection + rollback to last known good state.

    Checks qacc AND qvel EVERY step for NaN, Inf, or huge values (>1e10).
    Also runs inside the exception handler when sim.step() throws.
    On divergence:
    1. mj_resetData to clear solver caches
    2. Restore qpos from the snapshot taken BEFORE the bad step
    3. Zero velocities, mj_forward
    => Fly stays exactly where it was 0.2ms ago. No teleport, no jumps.

    Brain state (PyTorch on CUDA) is untouched by mj_resetData.
    """

    MAX_RESETS = 20

    def __init__(self, sim, pipes):
        self.sim = sim
        self.pipes = pipes
        self.total_resets = 0
        self.resets_per_fly = {p.fly.name: 0 for p in pipes}
        # Pre-step snapshot buffer
        self._saved_qpos = sim.physics.data.qpos.copy()

    def save_state(self):
        """Call BEFORE sim.step() — snapshot qpos for rollback."""
        self._saved_qpos[:] = self.sim.physics.data.qpos

    def check_and_recover(self, body_step):
        """Call AFTER sim.step(). Returns True if divergence was detected and fixed."""
        if self.total_resets >= self.MAX_RESETS:
            return False
        data = self.sim.physics.data
        qacc = data.qacc
        qvel = data.qvel
        diverged = (not np.all(np.isfinite(qacc))
                    or not np.all(np.isfinite(qvel))
                    or np.any(np.abs(qacc) > 1e10)
                    or np.any(np.abs(qvel) > 1e8))
        if not diverged:
            return False

        # NaN detected — immediate rollback
        model_ptr = self.sim.physics.model.ptr
        data_ptr = data.ptr

        mujoco.mj_resetData(model_ptr, data_ptr)
        data.qpos[:] = self._saved_qpos   # state from 0.2ms ago
        data.qvel[:] = 0.0
        data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(model_ptr, data_ptr)

        self.total_resets += 1
        for p in self.pipes:
            self.resets_per_fly[p.fly.name] += 1
        reason = ("NaN/Inf" if not np.all(np.isfinite(qacc))
                  else f"huge qacc ({np.max(np.abs(qacc)):.0e})"
                  if np.any(np.abs(qacc) > 1e10)
                  else f"huge qvel ({np.max(np.abs(qvel)):.0e})")
        print(f"  WATCHDOG [{body_step}]: {reason} — "
              f"rolled back (reset #{self.total_resets})")
        return True


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    # ── Timestep & timing constants ──
    # 2e-4 timestep = half the MuJoCo steps vs 1e-4 (physics is 77% of cost)
    TIMESTEP = 2e-4
    BRAIN_RATIO = 50        # brain every 10ms (50 × 0.2ms)
    VISION_RATIO = 500      # vision every 100ms (500 × 0.2ms)
    STEPS_PER_FRAME = 84    # viewer ~60fps (84 × 0.2ms ≈ 16.8ms)
    STATUS_INTERVAL = 5000  # status every 1.0s (5000 × 0.2ms)

    max_steps = int(args.duration / 1e-4) if args.duration > 0 else 0

    # ── Contact sensors (same as fly_embodied.py) ──
    contact_sensors = [
        f"{leg}{seg}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for seg in ["Tibia", "Tarsus1", "Tarsus2",
                     "Tarsus3", "Tarsus4", "Tarsus5"]
    ]

    # ── Create two flies ──
    print("Creating two flies...")
    half_sep = args.separation / 2.0
    fly0 = Fly(
        name="fly0",
        spawn_pos=(0, 0, 0.6),
        spawn_orientation=(0, 0, 0),          # facing +x → toward fly1
        enable_adhesion=True,
        draw_adhesion=False,
        contact_sensor_placements=contact_sensors,
        enable_vision=args.visual,
    )
    fly1 = Fly(
        name="fly1",
        spawn_pos=(args.separation, 0, 0.6),
        spawn_orientation=(0, 0, np.pi),      # facing -x → toward fly0
        enable_adhesion=True,
        draw_adhesion=False,
        contact_sensor_placements=contact_sensors,
        enable_vision=args.visual,
    )

    # Disable flygym's internal vision rendering (Windows GL workaround)
    if args.visual:
        fly0.enable_vision = False
        fly1.enable_vision = False

    # ── Arena ──
    taste_zones = []
    odor_sources = []
    if args.gustatory:
        taste_zones = [
            TasteZone(center=[15.0, 5.0], radius=8.0,
                      taste='sugar', label='sugar_patch'),
            TasteZone(center=[20.0, -10.0], radius=6.0,
                      taste='bitter', label='bitter_patch'),
        ]
    if args.olfactory:
        odor_sources = [
            OdorSource(position=[25.0, 10.0, 1.0],
                       odor_type='attractive', amplitude=0.9, spread=25.0,
                       label='food'),
        ]

    if args.ball:
        arena = LoomingArena(
            ball_radius=6.0,
            approach_speed=15.0,
            start_distance=120.0,
            ball_height=1.5,
            approach_angle=0.0,
            taste_zones=taste_zones,
            odor_sources=odor_sources,
            ground_size=100,
        )
    elif args.flat:
        arena = FlatTerrain()
    else:
        arena = ProceduralArena(
            world_seed=args.world_seed, ground_size=500)

    # ── Unified simulation with inter-fly collision ──
    print("Initializing simulation (two flies, one MuJoCo scene)...")
    sim = CrossFlySimulation(
        flies=[fly0, fly1], cameras=None, arena=arena, timestep=TIMESTEP)
    obs, info = sim.reset()

    # ── Per-fly brain + sensory pipelines ──
    print("Initializing brains (2 × 138,639 neurons on GPU)...")
    vibration_sources = []
    if args.somatosensory:
        vibration_sources = [
            VibrationSource(position=[30.0, 20.0, 1.0],
                            frequency=200.0, amplitude=0.8,
                            label='courtship'),
        ]

    pipes = []
    for i, fly in enumerate([fly0, fly1]):
        plastic_path = f'data/plastic_weights_fly{i}.pt'
        if args.fresh:
            # Use temp path that won't exist, so no weights are loaded
            plastic_path = f'data/.fresh_plastic_fly{i}.pt'
        brain = BrainEngine(device='cuda', plastic_path=plastic_path)

        visual = None
        if args.visual:
            visual = VisualSystem(brain.flyid2i, brain.i2flyid)

        somato = None
        if args.somatosensory:
            somato = SomatosensorySystem(brain.flyid2i)

        gusto = None
        if args.gustatory:
            gusto = GustatorySystem(brain.flyid2i, taste_zones)

        olfact = None
        if args.olfactory:
            olfact = OlfactorySystem(brain.flyid2i)

        song = None
        if args.vocalize:
            song = WingSongSystem(self_hearing_gain=0.2)

        decoder = DNRateDecoder(window_ms=50.0, dt_ms=0.1, max_rate=200.0)
        bridge = BrainBodyBridge(decoder, escape_threshold=0.3,
                                 groom_threshold=0.02)
        cpg = WalkingCPG(fly, sim.timestep, seed=i)
        groom = GroomingController()

        # Register lateralized populations for directional escape
        if visual is not None:
            lplc2_idx = visual.get_lplc2_indices(brain.flyid2i)
            lc4_idx = visual.get_lc4_indices(brain.flyid2i)
            for name, indices in {**lplc2_idx, **lc4_idx}.items():
                brain.register_population(name, indices)
                decoder.register_population(name)

        # Register JO populations
        if somato is not None:
            for pop_name, pop_idx in [
                ('JO_touch_L', somato.touch_idx_left),
                ('JO_touch_R', somato.touch_idx_right),
                ('JO_sound_L', somato.sound_idx_left),
                ('JO_sound_R', somato.sound_idx_right),
            ]:
                if len(pop_idx) > 0:
                    brain.register_population(pop_name, pop_idx)
                    decoder.register_population(pop_name)

        # Set initial stimulus (P9 forward walking)
        brain.set_stimulus('p9')

        # Per-fly consciousness detection
        consciousness = None
        if args.consciousness:
            consciousness = ConsciousnessDetector(
                brain, label=f'fly{i}', sim_timestep=TIMESTEP)

        pipes.append(SimpleNamespace(
            fly=fly, brain=brain, decoder=decoder, bridge=bridge,
            cpg=cpg, groom=groom, visual=visual, somato=somato,
            olfact=olfact, gusto=gusto, song=song,
            consciousness=consciousness,
            drive=np.array([0.0, 0.0]),
            prev_mode='walking',
            cached_visual=(None, None),
            eye_cam_ids={},
            geom_hide_ids=[],
        ))
        print(f"  [fly{i}] Brain + sensory pipeline initialized")

    # ── Physics watchdog ──
    watchdog = PhysicsWatchdog(sim, pipes)

    # ── Cache body IDs for self-righting reflex ──
    _model_ptr = sim.physics.model.ptr
    for pipe in pipes:
        pipe.thorax_body_id = mujoco.mj_name2id(
            _model_ptr, mujoco.mjtObj.mjOBJ_BODY,
            f'{pipe.fly.name}/Thorax')
        _jid = mujoco.mj_name2id(
            _model_ptr, mujoco.mjtObj.mjOBJ_JOINT,
            f'{pipe.fly.name}/freejoint')
        pipe.freejoint_qposadr = (
            _model_ptr.jnt_qposadr[_jid] if _jid >= 0 else -1)
        pipe.fly_mass = float(
            _model_ptr.body_subtreemass[pipe.thorax_body_id])

    # ── Vision rendering setup ──
    eye_renderer = None
    if args.visual:
        model_ptr = sim.physics.model.ptr
        eye_renderer = mujoco.Renderer(model_ptr, height=512, width=450)
        for pipe in pipes:
            for side in ["L", "R"]:
                cam_name = f"{pipe.fly.name}/{side}Eye_cam"
                cid = mujoco.mj_name2id(
                    model_ptr, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                pipe.eye_cam_ids[side] = cid
            for geom_name in getattr(pipe.fly, '_geoms_to_hide', []):
                full_name = f"{pipe.fly.name}/{geom_name}"
                gid = mujoco.mj_name2id(
                    model_ptr, mujoco.mjtObj.mjOBJ_GEOM, full_name)
                if gid >= 0:
                    pipe.geom_hide_ids.append(gid)
            print(f"  [{pipe.fly.name}] Vision cameras ready, "
                  f"{len(pipe.geom_hide_ids)} geoms hidden")

    # ── MuJoCo viewer ──
    viewer = None
    if not args.no_viewer:
        print("Launching MuJoCo viewer...")
        viewer = mujoco.viewer.launch_passive(
            sim.physics.model.ptr, sim.physics.data.ptr,
            show_left_ui=False,
            show_right_ui=False,
        )
        if viewer is not None:
            viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
            for g in range(3):
                viewer.opt.sitegroup[g] = 0
            viewer.opt.sitegroup[4] = 1
            # Track midpoint between flies
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.distance = 60.0
            viewer.cam.azimuth = -120.0
            viewer.cam.elevation = -25.0

    # ── Banner ──
    print()
    print("=" * 70)
    print("  TWO FLIES — ONE WORLD, INDEPENDENT MINDS")
    print(f"  2 × 138,639 neurons | separation={args.separation:.1f} mm")
    if args.visual:
        print("  *** REAL VISION: each fly sees the other ***")
    if args.somatosensory:
        print("  *** SOMATOSENSORY: touch + sound cross-perception ***")
    if args.gustatory:
        print("  *** GUSTATORY: shared taste zones ***")
    if args.olfactory:
        print("  *** OLFACTORY: shared odor sources ***")
    if args.vocalize:
        print("  *** VOCALIZATION: wing song heard by other fly ***")
    if args.consciousness:
        print("  *** CONSCIOUSNESS: independent CI per fly ***")
    if isinstance(arena, ProceduralArena):
        print(f"  *** PROCEDURAL WORLD: seed={args.world_seed}, "
              f"40-body obstacle pool ***")
    print("  *** PHYSICS WATCHDOG: NaN/freeze recovery active ***")
    print("  Close viewer to exit")
    print("=" * 70)
    print()

    # ── Timing ──
    body_step = 0
    physics_errors = 0
    _frame_target = 1.0 / 60.0
    _next_viewer_sync = _time.perf_counter()
    _fps_counter = 0
    _fps_timer = _time.perf_counter()
    _measured_fps = 0.0

    # ── Profiling accumulators ──
    _prof_physics = 0.0
    _prof_vision = 0.0
    _prof_brain = 0.0
    _prof_sensory = 0.0
    _prof_cpg = 0.0
    _prof_viewer = 0.0
    _prof_steps = 0

    # ── Main loop ──
    try:
        while True:
            # Check exit conditions
            if viewer is not None:
                if not viewer.is_running():
                    break
            elif max_steps > 0 and body_step >= max_steps:
                break

            # ── Sensory + brain (staggered between flies) ──
            # fly0 brain ticks: 0, 100, 200, ...
            # fly1 brain ticks: 50, 150, 250, ...
            # fly0 vision:      0, 1000, 2000, ...
            # fly1 vision:      500, 1500, 2500, ...
            # => never two brain GPU steps or two vision renders on same tick
            for i, pipe in enumerate(pipes):
                brain_offset = i * (BRAIN_RATIO // 2)
                if (body_step - brain_offset) % BRAIN_RATIO != 0:
                    continue

                fly_obs = obs[pipe.fly.name]

                # ── Vision (staggered within brain ticks) ──
                vision_offset = i * (VISION_RATIO // 2)
                if (pipe.visual is not None
                        and (body_step - vision_offset) % VISION_RATIO == 0
                        and eye_renderer is not None):
                    _t0 = _time.perf_counter()
                    render_fly_eyes(pipe, eye_renderer, sim, fly_obs)
                    _prof_vision += _time.perf_counter() - _t0

                # ── Sensory processing ──
                _t0 = _time.perf_counter()

                if pipe.somato is not None:
                    contact_forces = fly_obs.get(
                        'contact_forces', np.zeros((36, 3)))
                    pipe.somato.process_contact(contact_forces)

                    fly_pos = fly_obs['fly'][0]
                    fly_orient = fly_obs.get(
                        'fly_orientation', np.zeros(3))
                    fly_heading = float(
                        np.arctan2(fly_orient[1], fly_orient[0]))
                    active_vib = getattr(
                        arena, 'all_vibration_sources', vibration_sources)
                    pipe.somato.process_vibration(
                        fly_pos, fly_heading, active_vib)

                    # Cross-perception: other fly's wing song
                    other = pipes[1 - i]
                    if (other.song is not None
                            and other.song.is_singing):
                        other_obs = obs[other.fly.name]
                        other_pos = other_obs['fly'][0]
                        wing_vib = other.song.get_vibration_sources()
                        for vs in wing_vib:
                            vs.position = other_pos
                        pipe.somato.process_vibration(
                            fly_pos, fly_heading, wing_vib)

                    jo_idx, jo_rates = pipe.somato.get_rates()
                    pipe.brain.set_sensory_rates(jo_idx, jo_rates)
                    pipe.bridge.tactile_force = (
                        pipe.somato.max_contact_force)
                    pipe.bridge.sound_orientation_bias = (
                        pipe.somato.orientation_bias)

                if pipe.gusto is not None:
                    if hasattr(arena, 'all_taste_zones'):
                        pipe.gusto.zones = arena.all_taste_zones
                    end_effectors = fly_obs.get(
                        'end_effectors', np.zeros((6, 3)))
                    pipe.gusto.process(end_effectors)
                    grn_idx, grn_rates = pipe.gusto.get_rates()
                    pipe.brain.set_sensory_rates(grn_idx, grn_rates)
                    pipe.bridge.bitter_active = pipe.gusto.bitter_active

                if pipe.olfact is not None:
                    fly_pos = fly_obs['fly'][0]
                    fly_orient = fly_obs.get(
                        'fly_orientation', np.zeros(3))
                    fly_heading = float(
                        np.arctan2(fly_orient[1], fly_orient[0]))
                    active_odor = getattr(
                        arena, 'all_odor_sources', odor_sources)
                    pipe.olfact.process(
                        fly_pos, fly_heading, active_odor)
                    or_idx, or_rates = pipe.olfact.get_rates()
                    pipe.brain.set_sensory_rates(or_idx, or_rates)
                    pipe.bridge.olfactory_attraction_bias = (
                        pipe.olfact.attraction_bias)
                    pipe.bridge.olfactory_repulsive = (
                        pipe.olfact.is_repulsive_escape)
                    pipe.bridge.olfactory_repulsion_bias = (
                        pipe.olfact.repulsion_bias)

                if pipe.song is not None:
                    fly_pos_ws = fly_obs['fly'][0]
                    pipe.song.process(
                        pipe.decoder, fly_pos_ws,
                        BRAIN_RATIO * sim.timestep)
                    if pipe.somato is not None and pipe.song.is_singing:
                        wing_sources = pipe.song.get_vibration_sources()
                        fly_orient_ws = fly_obs.get(
                            'fly_orientation', np.zeros(3))
                        heading_ws = float(
                            np.arctan2(fly_orient_ws[1],
                                       fly_orient_ws[0]))
                        base_vib = getattr(
                            arena, 'all_vibration_sources', vibration_sources)
                        all_vib = base_vib + wing_sources
                        pipe.somato.process_vibration(
                            fly_pos_ws, heading_ws, all_vib)
                        jo_idx, jo_rates = pipe.somato.get_rates()
                        pipe.brain.set_sensory_rates(jo_idx, jo_rates)

                if (pipe.visual is not None
                        and pipe.cached_visual[0] is not None):
                    vis_eye = pipe.visual._T2_eye
                    vis_rates_arr = pipe.cached_visual[1]
                    if (vis_rates_arr is not None
                            and len(vis_rates_arr) > 0):
                        mask_L = vis_eye == 0
                        mask_R = vis_eye == 1
                        t2_left = (float(vis_rates_arr[mask_L].mean())
                                   if mask_L.any() else 0.0)
                        t2_right = (float(vis_rates_arr[mask_R].mean())
                                    if mask_R.any() else 0.0)
                        pipe.bridge.visual_threat_bias = (
                            (t2_right - t2_left)
                            / (t2_left + t2_right + 1e-6))

                _prof_sensory += _time.perf_counter() - _t0

                # ── Brain step ──
                _t0 = _time.perf_counter()
                pipe.brain.step()
                dn_spikes = pipe.brain.get_dn_spikes()
                pop_spikes = (pipe.brain.get_population_spikes()
                              if pipe.brain.populations else None)
                pipe.decoder.update(dn_spikes, pop_spikes)
                _prof_brain += _time.perf_counter() - _t0

                # ── Consciousness update (per brain tick) ──
                if pipe.consciousness is not None:
                    pipe.consciousness.update(body_step, pipe.bridge.mode)

                # ── Mode transitions ──
                if pipe.bridge.mode != pipe.prev_mode:
                    print(f"  [fly{i}] {pipe.prev_mode} -> "
                          f"{pipe.bridge.mode}")
                    pipe.prev_mode = pipe.bridge.mode

            # ── Compute drive every body step (matches fly_embodied.py) ──
            for pipe in pipes:
                pipe.drive = pipe.bridge.compute_drive(
                    dt=BRAIN_RATIO * sim.timestep)

            # ── Build actions (CPG) ──
            _t0 = _time.perf_counter()
            actions = {}
            for pipe in pipes:
                fly_obs = obs[pipe.fly.name]
                if pipe.bridge.mode == 'grooming':
                    action = pipe.groom.get_action(
                        body_step * sim.timestep)
                else:
                    action = pipe.cpg.compute_action(
                        fly_obs, pipe.drive[0], pipe.drive[1])
                actions[pipe.fly.name] = action
            _prof_cpg += _time.perf_counter() - _t0

            # ── Self-righting reflex (wing-beat uprighting) ──
            # Gentle forces: 1.3x weight lift, torque at 0.2% of weight.
            # The old torque (mass*g) caused ~10,000 rad/s² → NaN at DOF 6.
            _data = sim.physics.data
            _data.xfrc_applied[:] = 0
            for pipe in pipes:
                adr = pipe.freejoint_qposadr
                if adr < 0 or pipe.fly_mass <= 0:
                    continue
                quat = _data.qpos[adr + 3:adr + 7]
                if not np.all(np.isfinite(quat)):
                    continue
                w, qx, qy, qz = quat
                up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
                if up_z < 0.0:
                    # Fly is inverted — gentle wing-beat righting
                    up_x = 2.0 * (qx * qz + w * qy)
                    up_y = 2.0 * (qy * qz - w * qx)
                    torque = np.array([up_y, -up_x, 0.0])
                    if np.linalg.norm(torque) < 0.01:
                        torque = np.array([1.0, 0.0, 0.0])
                    bid = pipe.thorax_body_id
                    weight = pipe.fly_mass * 9.81e3
                    _data.xfrc_applied[bid, 2] = weight * 1.3    # gentle lift
                    _data.xfrc_applied[bid, 3:6] = torque * weight * 0.002  # ~50x gentler

            # ── Update arena with fly positions ──
            if hasattr(arena, 'set_fly_positions'):
                arena.set_fly_positions(
                    [obs[p.fly.name]['fly'][0] for p in pipes])

            # ── Single physics step (with immediate NaN rollback) ──
            watchdog.save_state()
            _prev_obs = obs
            _t0 = _time.perf_counter()
            try:
                obs, reward, terminated, truncated, info = sim.step(actions)
                physics_errors = 0
            except Exception as e:
                physics_errors += 1
                if physics_errors >= 50:
                    print(f"  Physics unstable ({physics_errors} errors): {e}")
                    break
                # Attempt recovery — don't skip watchdog on exception
                if watchdog.check_and_recover(body_step):
                    obs = _prev_obs
                    physics_errors = 0
                continue
            _prof_physics += _time.perf_counter() - _t0

            # ── Immediate NaN check (every step, not periodic) ──
            if watchdog.check_and_recover(body_step):
                obs = _prev_obs  # restore last good observations
                physics_errors = 0
                continue         # retry from the same body_step

            body_step += 1
            _prof_steps += 1

            # ── Sync viewer ──
            if viewer is not None and body_step % STEPS_PER_FRAME == 0:
                _t0 = _time.perf_counter()
                _now = _time.perf_counter()
                _sleep = _next_viewer_sync - _now
                if _sleep > 0.001:
                    _time.sleep(_sleep)
                viewer.sync()
                _prof_viewer += _time.perf_counter() - _t0
                _next_viewer_sync = max(
                    _next_viewer_sync, _now) + _frame_target
                _fps_counter += 1
                if _now - _fps_timer >= 1.0:
                    _measured_fps = _fps_counter / (_now - _fps_timer)
                    _fps_counter = 0
                    _fps_timer = _now

            # ── Status + profiling print ──
            if body_step % STATUS_INTERVAL == 0:
                sim_time = body_step * sim.timestep
                pos0 = obs[fly0.name]['fly'][0]
                pos1 = obs[fly1.name]['fly'][0]
                dist = np.linalg.norm(pos0 - pos1)
                # Compute ms/step breakdown
                _total_prof = (_prof_physics + _prof_vision + _prof_brain
                               + _prof_sensory + _prof_cpg + _prof_viewer)
                if _prof_steps > 0 and _total_prof > 0:
                    _ms = lambda v: v / _prof_steps * 1000
                    _pct = lambda v: v / _total_prof * 100
                    print(f"  t={sim_time:.1f}s  fps={_measured_fps:.0f}  "
                          f"fly0=[{pos0[0]:.1f},{pos0[1]:.1f}] "
                          f"mode={pipes[0].bridge.mode}  "
                          f"fly1=[{pos1[0]:.1f},{pos1[1]:.1f}] "
                          f"mode={pipes[1].bridge.mode}  "
                          f"dist={dist:.1f}mm")
                    print(f"  PROFILE (ms/step): "
                          f"physics={_ms(_prof_physics):.3f} ({_pct(_prof_physics):.0f}%)  "
                          f"vision={_ms(_prof_vision):.3f} ({_pct(_prof_vision):.0f}%)  "
                          f"brain={_ms(_prof_brain):.3f} ({_pct(_prof_brain):.0f}%)  "
                          f"sensory={_ms(_prof_sensory):.3f} ({_pct(_prof_sensory):.0f}%)  "
                          f"cpg={_ms(_prof_cpg):.3f} ({_pct(_prof_cpg):.0f}%)  "
                          f"viewer={_ms(_prof_viewer):.3f} ({_pct(_prof_viewer):.0f}%)")
                else:
                    print(f"  t={sim_time:.1f}s  fps={_measured_fps:.0f}  "
                          f"fly0=[{pos0[0]:.1f},{pos0[1]:.1f}] "
                          f"mode={pipes[0].bridge.mode}  "
                          f"fly1=[{pos1[0]:.1f},{pos1[1]:.1f}] "
                          f"mode={pipes[1].bridge.mode}  "
                          f"dist={dist:.1f}mm")

                # World + watchdog status
                if hasattr(arena, 'n_active_chunks'):
                    print(f"  WORLD: {arena.n_active_chunks} chunks, "
                          f"{arena.n_pool_used}/{40} pool used")
                wd = watchdog
                print(f"  WATCHDOG: {wd.total_resets} resets ("
                      f"{', '.join(f'{k}={v}' for k, v in wd.resets_per_fly.items())})")

                # Consciousness comparison
                if args.consciousness:
                    for i, pipe in enumerate(pipes):
                        if pipe.consciousness is not None:
                            c = pipe.consciousness
                            print(f"  fly{i} {c.get_status_str()}")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        # Save independent plastic weights
        for i, pipe in enumerate(pipes):
            try:
                pipe.brain.save_plastic_weights()
            except (RuntimeError, OSError) as e:
                print(f"  [fly{i}] Could not save weights: {e}")
        # Save consciousness sessions
        for i, pipe in enumerate(pipes):
            if pipe.consciousness is not None:
                try:
                    pipe.consciousness.save_session()
                except (PermissionError, OSError) as e:
                    print(f"  [fly{i}] Could not save consciousness: {e}")
        if viewer is not None:
            viewer.close()
        print(f"\nSimulation ended after {body_step} steps "
              f"({body_step * sim.timestep:.2f}s sim time).")
        if body_step > 0:
            for i, pipe in enumerate(pipes):
                pos = obs[pipe.fly.name]['fly'][0]
                ci_str = ""
                if pipe.consciousness is not None:
                    ci_str = f"  CI={pipe.consciousness.ci:.3f}"
                print(f"  fly{i} final pos: "
                      f"[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] mm"
                      f"{ci_str}")


if __name__ == '__main__':
    main()
