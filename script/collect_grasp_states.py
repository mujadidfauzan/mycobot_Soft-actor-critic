import argparse
import os
from datetime import datetime

import numpy as np
from stable_baselines3 import SAC

from source.envs.grasping_env import GraspingEnv


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
    parser.add_argument("--model", required=True, help="Path to SAC grasping .zip")
    parser.add_argument("--xml", required=True, help="Path to MuJoCo XML")
    parser.add_argument(
        "--out",
        default=None,
        help="Output .npz path (defaults to ./logs/grasp_states_<timestamp>.npz)",
    )
    parser.add_argument("--num-states", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-episodes", type=int, default=5000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("logs", f"grasp_states_{ts}.npz")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print("XML file :", args.xml)
    env = GraspingEnv(
        xml_file=args.xml,
        frame_skip=5,
        render_mode="human" if args.render else "rgb_array",
    )
    model = SAC.load(args.model)

    qpos_list: list[np.ndarray] = []
    qvel_list: list[np.ndarray] = []
    object_keys: list[str] = []

    episodes = 0
    while len(qpos_list) < args.num_states and episodes < args.max_episodes:
        episodes += 1
        obs, _info = env.reset()

        for _step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=bool(args.deterministic))
            obs, _reward, _terminated, truncated, _info = env.step(action)

            if is_grasped(env):
                qpos_list.append(env.data.qpos.copy())
                qvel_list.append(env.data.qvel.copy())
                object_keys.append(str(env.current_object_key))
                break

            if truncated:
                break

    env.close()

    if len(qpos_list) == 0:
        raise RuntimeError("No grasp states collected. Try increasing --max-episodes.")

    qpos = np.stack(qpos_list, axis=0)
    qvel = np.stack(qvel_list, axis=0)
    np.savez_compressed(
        out_path,
        qpos=qpos,
        qvel=qvel,
        object_keys=np.asarray(object_keys, dtype="<U16"),
        model_path=str(args.model),
        xml_path=str(args.xml),
        object_body=str(getattr(env, "obj_body_name", "")),
        collected=int(len(qpos_list)),
    )
    print(f"[OK] Saved {len(qpos_list)} states -> {out_path}")


if __name__ == "__main__":
    main()
