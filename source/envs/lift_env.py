import time

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class LiftEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = "object_lift.xml",
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        reward_dist_weight: float = 1,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_dist_weight,
            **kwargs,
        )

        self._reward_dist_weight = reward_dist_weight

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=None,
            camera_name="watching",
            **kwargs,
        )

        obj_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_joint"
        )
        self.obj_qposadr = int(self.model.jnt_qposadr[obj_joint_id])
        self.obj_dofadr = int(self.model.jnt_dofadr[obj_joint_id])

        # print(f"Object joint qposadr: {self.obj_qposadr}")
        robot_dof = self.obj_qposadr
        obs_dim = robot_dof * 2 + 6
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.observation_space = observation_space

        self.max_episode_steps = 200
        self.current_step = 0

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        self.current_step += 1

        action = action.copy()

        scale = 0.05
        current_qpos = self.data.qpos[: self.model.nu]

        target = current_qpos + scale * action
        low, high = self.action_space.low, self.action_space.high
        target = np.clip(target, low, high)

        # Fix gripper control
        target[-2] = -0.7  # left open
        target[-1] = 0.7  # right open
        self.do_simulation(target, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        info = reward_info
        terminated = False
        truncated = self.current_step >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def _get_rew(self, action):
        ee_pos = self.data.site("attachment_site").xpos.copy()
        obj_pos = self.data.body("obj").xpos.copy()

        dist = np.linalg.norm(ee_pos - obj_pos)
        reward_dist = -dist * self._reward_dist_weight
        touch_threshold = 0.02
        touch_bonus = 1.0 if dist < touch_threshold else 0.0

        control_penalty = -0.001 * np.sum(np.square(action))
        reward_info = {
            "dist": float(dist),
            "reward_dist": float(reward_dist),
            "touch_bonus": float(touch_bonus),
            "is_touch": bool(dist < touch_threshold),
            "control_penalty": float(control_penalty),
        }

        reward = reward_dist + touch_bonus + control_penalty

        return reward, reward_info

    def reset_model(self):

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        adr = self.obj_qposadr
        vadr = self.obj_dofadr

        # Random XY
        x = self.np_random.uniform(0.22, 0.38)
        y = self.np_random.uniform(-0.12, 0.12)
        z = 0.025

        # Set position
        qpos[adr + 0] = x
        qpos[adr + 1] = y
        qpos[adr + 2] = z

        # Identity quaternion
        qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]

        # Zero object velocity
        qvel[vadr : vadr + 6] = 0.0

        self.set_state(qpos, qvel)

        self.current_step = 0

        return self._get_obs()

    def _get_obs(self):

        qpos = self.data.qpos
        qvel = self.data.qvel

        # Robot joint positions & velocities
        robot_qpos = qpos[: self.obj_qposadr]
        robot_qvel = qvel[: self.obj_dofadr]

        # Object position
        obj_pos = qpos[self.obj_qposadr : self.obj_qposadr + 3]

        # End effector position
        ee_pos = self.data.site("attachment_site").xpos

        # Relative position
        rel_pos = obj_pos - ee_pos

        obs = np.concatenate([robot_qpos, robot_qvel, obj_pos, rel_pos])

        return obs.astype(np.float32)
