import argparse

import numpy as np
from stable_baselines3 import SAC

from source.envs.grasping_env import GraspingEnv
from source.envs.reaching_env import ReachingEnv


def is_grasped(env: GraspingEnv) -> bool:
    ee_pos = env.data.site("attachment_site").xpos.copy()
    obj_pos = env.data.body(env.obj_body_name).xpos.copy()
    dist = float(np.linalg.norm(ee_pos - obj_pos))

    grip_l = float(env.data.qpos[env.gripL_qadr])
    grip_r = float(env.data.qpos[env.gripR_qadr])
    gripper_closed = grip_l < 0.0 and grip_r > 0.0

    lifted = float(obj_pos[2]) > 0.07
    return dist < 0.03 and gripper_closed and lifted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp-model", required=True, help="Path to SAC grasping .zip")
    parser.add_argument("--reach-model", required=True, help="Path to SAC reaching .zip")
    parser.add_argument("--xml", required=True, help="Path to MuJoCo XML")
    parser.add_argument("--grasp-steps", type=int, default=500)
    parser.add_argument("--reach-steps", type=int, default=500)
    args = parser.parse_args()

    grasp_env = GraspingEnv(xml_file=args.xml, frame_skip=5, render_mode="human")
    grasp_model = SAC.load(args.grasp_model)

    obs, _ = grasp_env.reset()
    for _ in range(args.grasp_steps):
        action, _ = grasp_model.predict(obs, deterministic=True)
        obs, _reward, _terminated, truncated, _info = grasp_env.step(action)
        if is_grasped(grasp_env) or truncated:
            break

    qpos = grasp_env.data.qpos.copy()
    qvel = grasp_env.data.qvel.copy()
    grasp_env.close()

    reach_env = ReachingEnv(xml_file=args.xml, frame_skip=5, render_mode="human")
    reach_model = SAC.load(args.reach_model)

    reach_env.reset()
    reach_env.set_state(qpos, qvel)

    obs = reach_env._get_obs()
    for _ in range(args.reach_steps):
        action, _ = reach_model.predict(obs, deterministic=True)
        obs, _reward, _terminated, truncated, _info = reach_env.step(action)
        if truncated:
            break

    reach_env.close()


if __name__ == "__main__":
    main()

