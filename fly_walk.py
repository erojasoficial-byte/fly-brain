"""
Drosophila 3D Walking Simulation using NeuroMechFly v2 (FlyGym).
Renders an anatomically accurate fruit fly walking with tripod gait.
"""

import numpy as np
from pathlib import Path

from flygym import Fly, SingleFlySimulation, Camera
from flygym.preprogrammed import get_cpg_biases
from flygym.examples.locomotion import PreprogrammedSteps, CPGNetwork


def main():
    output_dir = Path(__file__).parent / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Simulation parameters ---
    timestep = 1e-4
    run_time = 2.0
    num_steps = int(run_time / timestep)

    # --- CPG controller for tripod gait ---
    intrinsic_freqs = np.ones(6) * 12.0
    intrinsic_amps = np.ones(6)
    phase_biases = get_cpg_biases("tripod")
    coupling_weights = (phase_biases > 0).astype(float) * 10.0
    convergence_coefs = np.ones(6) * 20.0

    cpg = CPGNetwork(
        timestep=timestep,
        intrinsic_freqs=intrinsic_freqs,
        intrinsic_amps=intrinsic_amps,
        coupling_weights=coupling_weights,
        phase_biases=phase_biases,
        convergence_coefs=convergence_coefs,
    )

    preprogrammed_steps = PreprogrammedSteps()

    leg_names = ["LF", "LM", "LH", "RF", "RM", "RH"]

    # --- Create fly ---
    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        init_pose="stretch",
        control="position",
    )

    # --- Camera: side view ---
    cam = Camera(
        attachment_point=fly.model.worldbody,
        camera_name="camera_right",
        targeted_fly_names=[fly.name],
        play_speed=0.2,
        window_size=(1280, 720),
        fps=60,
        play_speed_text=True,
        timestamp_text=True,
        draw_contacts=True,
    )

    # --- Build simulation ---
    sim = SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        timestep=timestep,
    )

    print(f"Simulation: {num_steps} steps ({run_time}s)")
    print("Running tripod gait...")

    obs, info = sim.reset()

    for step_i in range(num_steps):
        cpg.step()

        # Build joint angle array: 6 legs x 7 DOFs = 42 joints
        all_joint_angles = []
        all_adhesion = []
        for i, leg in enumerate(leg_names):
            joint_angles = preprogrammed_steps.get_joint_angles(
                leg, cpg.curr_phases[i], cpg.curr_magnitudes[i]
            )
            all_joint_angles.append(joint_angles)
            adhesion_on = preprogrammed_steps.get_adhesion_onoff(leg, cpg.curr_phases[i])
            all_adhesion.append(adhesion_on)

        action = {
            "joints": np.concatenate(all_joint_angles),
            "adhesion": np.array(all_adhesion, dtype=np.float64),
        }

        obs, reward, terminated, truncated, info = sim.step(action)
        sim.render()

        if (step_i + 1) % 5000 == 0:
            pct = (step_i + 1) / num_steps * 100
            print(f"  {pct:.0f}% ({step_i+1}/{num_steps})")

    video_path = str(output_dir / "fly_walking_tripod.mp4")
    print(f"Saving video: {video_path}")
    cam.save_video(video_path)
    print("Done!")
    return video_path


if __name__ == "__main__":
    path = main()
    import subprocess
    subprocess.Popen(["cmd", "/c", "start", "", path], shell=True)
