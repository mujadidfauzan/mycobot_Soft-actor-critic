# evaluate_to_csv.py
import csv
import os
from datetime import datetime

import numpy as np
from stable_baselines3 import SAC

# Sesuaikan import env kamu
from source.envs.grasping_env import GraspingEnv


def main():
    model_path = "/home/fauzan/Mujoco/Skripsi/logs/models/SAC_25_02_2026_15_13_02/sac_lift_final.zip"
    xml_path = "/home/fauzan/Mujoco/Skripsi/source/robot/object_lift.xml"

    out_dir = "./eval_logs"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"eval_{ts}.csv")

    env = GraspingEnv(
        render_mode="human",
        frame_skip=5,
        xml_file=xml_path,
    )

    model = SAC.load(model_path)

    obs, info = env.reset()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "step",
                "obj_pos",
                "target_pos",
                "dist_ee_obj",
                "dist_obj_target",
                "joint1_qpos",
                "joint2_qpos",
                "joint3_qpos",
                "joint4_qpos",
                "joint5_qpos",
                "joint6_qpos",
                "gripL_qpos",
                "gripR_qpos",
                "gripper_state",
            ]
        )

        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            state = env.get_physics_state()
            dist_ee_obj = np.linalg.norm(state["ee_pos"] - state["obj_pos"])
            dist_obj_target = np.linalg.norm(state["obj_pos"] - state["target_pos"])

            row = [
                state["step"],
                state["obj_pos"],
                state["target_pos"],
                dist_ee_obj,
                dist_obj_target,
                *state["qpos"][:8],
                state["gripper_state"],
            ]

            print(
                f"Step: {state['step']}, Obj Pos: {state['obj_pos']}, Target Pos: {state['target_pos']}, Dist EE-Obj: {dist_ee_obj:.4f}, Dist Obj-Target: {dist_obj_target:.4f}, Gripper State: {state['gripper_state']}"
            )

            writer.writerow(row)

            env.render()

            if terminated or truncated:
                break

    env.close()
    print(f"[OK] Saved evaluation CSV -> {csv_path}")


if __name__ == "__main__":
    main()
