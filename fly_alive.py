"""
Drosophila Alive - Real-time interactive 3D simulation.

The fly lives in a virtual environment driven by neural activity from the
brain connectome simulation. It walks, turns, pauses, and grooms
autonomously based on spike-driven behavioral state transitions.

Controls (MuJoCo viewer):
  - Left click + drag: rotate camera
  - Right click + drag: pan camera
  - Scroll: zoom
  - Double click: track body
  - Space: pause/resume
  - Tab: toggle UI panels
  - Esc: quit
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path

from flygym import Fly, SingleFlySimulation
from flygym.preprogrammed import get_cpg_biases
from flygym.examples.locomotion import PreprogrammedSteps, CPGNetwork


class BrainDrivenController:
    """
    Autonomous controller that transitions between behavioral states
    based on stochastic neural activity, mimicking how the connectome
    drives motor output.
    """

    STATES = ["walk_straight", "turn_left", "turn_right", "pause", "groom"]

    TRANSITION_PROBS = {
        "walk_straight": {"walk_straight": 0.60, "turn_left": 0.12, "turn_right": 0.12, "pause": 0.10, "groom": 0.06},
        "turn_left":     {"walk_straight": 0.40, "turn_left": 0.35, "turn_right": 0.05, "pause": 0.15, "groom": 0.05},
        "turn_right":    {"walk_straight": 0.40, "turn_left": 0.05, "turn_right": 0.35, "pause": 0.15, "groom": 0.05},
        "pause":         {"walk_straight": 0.45, "turn_left": 0.10, "turn_right": 0.10, "pause": 0.20, "groom": 0.15},
        "groom":         {"walk_straight": 0.30, "turn_left": 0.05, "turn_right": 0.05, "pause": 0.10, "groom": 0.50},
    }

    def __init__(self, timestep, spike_data=None):
        self.timestep = timestep
        self.state = "walk_straight"
        self.state_timer = 0
        self.min_state_duration = 0.3
        self.rng = np.random.default_rng(42)
        self.leg_names = ["LF", "LM", "LH", "RF", "RM", "RH"]
        self.steps = PreprogrammedSteps()
        self.groom_phase = 0.0
        self.groom_freq = 3.0

        # Load brain spikes for arousal modulation
        if spike_data is not None:
            import pandas as pd
            df = pd.read_parquet(spike_data)
            bins = np.arange(0, df['t'].max() + 100, 100)
            hist, _ = np.histogram(df['t'], bins=bins)
            self.arousal = hist / max(hist.max(), 1)
        else:
            self.arousal = None

        # Walking CPG
        intrinsic_freqs = np.ones(6) * 12.0
        intrinsic_amps = np.ones(6)
        convergence_coefs = np.ones(6) * 20.0
        self.walk_cpg = CPGNetwork(
            timestep=timestep,
            intrinsic_freqs=intrinsic_freqs,
            intrinsic_amps=intrinsic_amps,
            coupling_weights=(get_cpg_biases("tripod") > 0).astype(float) * 10.0,
            phase_biases=get_cpg_biases("tripod"),
            convergence_coefs=convergence_coefs,
        )

    def _get_arousal(self, sim_time_ms):
        if self.arousal is None:
            return 0.5 + 0.2 * np.sin(sim_time_ms / 500.0)
        idx = int(sim_time_ms / 100) % len(self.arousal)
        return float(self.arousal[idx])

    def maybe_transition(self, sim_time_s):
        self.state_timer += self.timestep * 1000  # accumulate in decision units
        if self.state_timer < self.min_state_duration:
            return

        probs = dict(self.TRANSITION_PROBS[self.state])
        arousal = self._get_arousal(sim_time_s * 1000)

        if arousal > 0.6:
            probs["walk_straight"] *= 1.3
            probs["turn_left"] *= 1.2
            probs["turn_right"] *= 1.2
            probs["pause"] *= 0.5
        elif arousal < 0.3:
            probs["pause"] *= 1.5
            probs["groom"] *= 1.5
            probs["walk_straight"] *= 0.7

        total = sum(probs.values())
        states = list(probs.keys())
        weights = [probs[s] / total for s in states]
        new_state = self.rng.choice(states, p=weights)
        if new_state != self.state:
            self.state = new_state
            self.state_timer = 0

    def get_action(self, sim_time_s):
        if self.state in ("walk_straight", "turn_left", "turn_right"):
            return self._walk_action()
        elif self.state == "pause":
            return self._pause_action()
        elif self.state == "groom":
            return self._groom_action()

    def _walk_action(self):
        self.walk_cpg.step()
        all_joints = []
        all_adhesion = []
        for i, leg in enumerate(self.leg_names):
            mag = self.walk_cpg.curr_magnitudes[i]
            if self.state == "turn_left" and leg.startswith("L"):
                mag *= 0.3
            elif self.state == "turn_right" and leg.startswith("R"):
                mag *= 0.3
            all_joints.append(self.steps.get_joint_angles(leg, self.walk_cpg.curr_phases[i], mag))
            all_adhesion.append(self.steps.get_adhesion_onoff(leg, self.walk_cpg.curr_phases[i]))
        return {
            "joints": np.concatenate(all_joints),
            "adhesion": np.array(all_adhesion, dtype=np.float64),
        }

    def _pause_action(self):
        all_joints = []
        all_adhesion = []
        for leg in self.leg_names:
            all_joints.append(self.steps.get_joint_angles(leg, 0.0, 0.0))
            all_adhesion.append(True)
        return {
            "joints": np.concatenate(all_joints),
            "adhesion": np.array(all_adhesion, dtype=np.float64),
        }

    def _groom_action(self):
        self.groom_phase += self.timestep * self.groom_freq * 2 * np.pi
        signal = np.sin(self.groom_phase)
        all_joints = []
        all_adhesion = []
        for leg in self.leg_names:
            ja = self.steps.get_joint_angles(leg, 0.0, 0.0).copy()
            if leg in ("LF", "RF"):
                ja[3] += signal * 0.4
                ja[5] += signal * 0.3
                all_adhesion.append(False)
            else:
                all_adhesion.append(True)
            all_joints.append(ja)
        return {
            "joints": np.concatenate(all_joints),
            "adhesion": np.array(all_adhesion, dtype=np.float64),
        }


def main():
    print("=" * 60)
    print("  DROSOPHILA ALIVE - Real-time Brain-Body Simulation")
    print("=" * 60)
    print()

    spike_path = Path(__file__).parent / "data" / "results" / "pytorch_t1.0s_n1.parquet"
    spike_data = spike_path if spike_path.exists() else None
    if spike_data:
        print(f"Brain data: {spike_path.name} (modulating behavior)")

    timestep = 1e-4

    print("Creating fly model...")
    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=False,
        init_pose="stretch",
        control="position",
    )

    sim = SingleFlySimulation(fly=fly, cameras=[], timestep=timestep)
    obs, info = sim.reset()

    controller = BrainDrivenController(timestep, spike_data)

    # Get MuJoCo pointers for viewer
    mj_model = sim.physics.model.ptr
    mj_data = sim.physics.data.ptr

    print()
    print("Controls:")
    print("  Mouse drag    -> rotate/pan/zoom")
    print("  Space         -> pause/resume")
    print("  Double click  -> track fly body")
    print("  Esc           -> quit")
    print()
    print("The fly is now ALIVE. Watch it explore.")
    print()

    sim_time = 0.0
    step_count = 0
    decision_interval = 1000

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.distance = 25.0
        viewer.cam.azimuth = -60.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat[:] = mj_data.qpos[:3]

        start_real = time.time()
        last_print = start_real

        while viewer.is_running():
            frame_start = time.time()

            # Run several physics substeps per frame for smoothness
            substeps = 10
            for _ in range(substeps):
                if step_count % decision_interval == 0:
                    controller.maybe_transition(sim_time)

                action = controller.get_action(sim_time)
                obs, reward, terminated, truncated, info = sim.step(action)

                sim_time += timestep
                step_count += 1

            # Camera smoothly follows the fly
            fly_pos = mj_data.qpos[:3]
            viewer.cam.lookat[:] = 0.93 * viewer.cam.lookat + 0.07 * fly_pos

            viewer.sync()

            # Print state periodically
            now = time.time()
            if now - last_print > 2.0:
                print(f"  t={sim_time:.2f}s | state: {controller.state:<15} | "
                      f"arousal: {controller._get_arousal(sim_time*1000):.2f}")
                last_print = now

            # Target ~60 FPS
            elapsed = time.time() - frame_start
            sleep_time = max(0, 1.0/60 - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    total = time.time() - start_real
    print(f"\nSession: {sim_time:.1f}s simulated in {total:.1f}s real time")


if __name__ == "__main__":
    main()
