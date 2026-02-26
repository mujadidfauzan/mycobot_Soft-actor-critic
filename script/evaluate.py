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
                "reward",
                "terminated",
                "truncated",
                "dist_ee_obj",
                "dist_obj_target",
                "ee_x",
                "ee_y",
                "ee_z",
                "obj_x",
                "obj_y",
                "obj_z",
                "target_x",
                "target_y",
                "target_z",
                "gripL_qpos",
                "gripR_qpos",
                "gripper_opening",
            ]
        )

        for step in range(500):
            print(step)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            ee_pos = env.data.site("attachment_site").xpos.copy()
            obj_pos = env.data.body("obj").xpos.copy()
            target_pos = env.data.site("target").xpos.copy()

            dist_ee_obj = float(np.linalg.norm(ee_pos - obj_pos))
            dist_obj_target = float(np.linalg.norm(obj_pos - target_pos))

            gripL = float(env.data.qpos[env.gripL_qadr])
            gripR = float(env.data.qpos[env.gripR_qadr])
            opening = float(gripR - gripL)

            writer.writerow(
                [
                    step,
                    float(reward),
                    bool(terminated),
                    bool(truncated),
                    dist_ee_obj,
                    dist_obj_target,
                    float(ee_pos[0]),
                    float(ee_pos[1]),
                    float(ee_pos[2]),
                    float(obj_pos[0]),
                    float(obj_pos[1]),
                    float(obj_pos[2]),
                    float(target_pos[0]),
                    float(target_pos[1]),
                    float(target_pos[2]),
                    gripL,
                    gripR,
                    opening,
                ]
            )

            env.render()

            if terminated or truncated:
                break

    env.close()
    print(f"[OK] Saved evaluation CSV -> {csv_path}")


if __name__ == "__main__":
    main()
