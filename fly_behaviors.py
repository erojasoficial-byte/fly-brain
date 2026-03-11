"""
Drosophila 3D Behavior Simulation - Multiple gaits and camera angles.
Generates videos of the fly walking with tripod, tetrapod, and wave gaits.
"""

import numpy as np
from pathlib import Path
from flygym import Fly, SingleFlySimulation, Camera
from flygym.preprogrammed import get_cpg_biases
from flygym.examples.locomotion import PreprogrammedSteps, CPGNetwork


def run_gait(gait_name, camera_name, run_time=1.5, output_dir=None):
    """Run a simulation with a specific gait and camera angle."""
    timestep = 1e-4
    num_steps = int(run_time / timestep)

    intrinsic_freqs = np.ones(6) * 12.0
    intrinsic_amps = np.ones(6)
    phase_biases = get_cpg_biases(gait_name)
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

    steps = PreprogrammedSteps()
    leg_names = ["LF", "LM", "LH", "RF", "RM", "RH"]

    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        init_pose="stretch",
        control="position",
    )

    cam = Camera(
        attachment_point=fly.model.worldbody,
        camera_name=camera_name,
        targeted_fly_names=[fly.name],
        play_speed=0.2,
        window_size=(960, 540),
        fps=60,
        play_speed_text=True,
        timestamp_text=True,
        draw_contacts=True,
    )

    sim = SingleFlySimulation(fly=fly, cameras=[cam], timestep=timestep)
    obs, info = sim.reset()

    for step_i in range(num_steps):
        cpg.step()
        all_joints = []
        all_adhesion = []
        for i, leg in enumerate(leg_names):
            all_joints.append(steps.get_joint_angles(leg, cpg.curr_phases[i], cpg.curr_magnitudes[i]))
            all_adhesion.append(steps.get_adhesion_onoff(leg, cpg.curr_phases[i]))

        action = {
            "joints": np.concatenate(all_joints),
            "adhesion": np.array(all_adhesion, dtype=np.float64),
        }
        obs, reward, terminated, truncated, info = sim.step(action)
        sim.render()

    video_path = str(output_dir / f"fly_{gait_name}_{camera_name}.mp4")
    cam.save_video(video_path)
    return video_path


def main():
    output_dir = Path(__file__).parent / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("tripod", "camera_right"),
        ("tripod", "camera_top"),
        ("tripod", "camera_front"),
        ("tetrapod", "camera_right"),
        ("wave", "camera_right"),
    ]

    videos = []
    for i, (gait, cam_name) in enumerate(configs):
        label = f"[{i+1}/{len(configs)}] {gait} gait - {cam_name}"
        print(f"Rendering {label}...")
        path = run_gait(gait, cam_name, run_time=1.5, output_dir=output_dir)
        videos.append((gait, cam_name, path))
        print(f"  Saved: {path}")

    print("\nAll videos generated!")
    for gait, cam, path in videos:
        print(f"  {gait:>10} {cam:>20} -> {path}")

    # Open first video
    import subprocess
    subprocess.Popen(["cmd", "/c", "start", "", videos[0][2]], shell=True)


if __name__ == "__main__":
    main()
