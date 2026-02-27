import time

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class ReachingEnv(MujocoEnv, utils.EzPickle):

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
        reward_dist_weight: float = 2,
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

        self.target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target"
        )
        self.target_default_pos = self.model.site_pos[self.target_site_id].copy()
        self.gripL_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "Slider_10"
        )
        self.gripR_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "Slider_11"
        )

        self.gripL_qadr = int(self.model.jnt_qposadr[self.gripL_jid])
        self.gripR_qadr = int(self.model.jnt_qposadr[self.gripR_jid])

        self.gripL_dadr = int(self.model.jnt_dofadr[self.gripL_jid])
        self.gripR_dadr = int(self.model.jnt_dofadr[self.gripR_jid])

        dummy_obs = self._get_obs()
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=dummy_obs.shape, dtype=np.float32
        )

        self.max_episode_steps = 500
        self.current_step = 0
        self.frame_skip = frame_skip
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

        scale_arm = 0.01

        current_ctrl = self.data.ctrl.copy()
        target = current_ctrl.copy()

        target[:-2] += scale_arm * action[:-2]
        target[6] = -0.02
        target[7] = 0.02

        # target = np.clip(target, self.action_space.low, self.action_space.high)
        # print("Target after clipping:", target)

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
        obj_pos = self.data.body("obj").xpos.copy()
        target_pos = self.data.site("target").xpos.copy()

        dist = np.linalg.norm(obj_pos - target_pos)
        reward_dist = -dist * self._reward_dist_weight
        reward_dist_tanh = 1.0 - float(np.tanh(float(dist) / 0.10))

        control_penalty = -0.001 * np.sum(np.square(action))

        target_xmat = self.data.site("target").xmat.copy()
        target_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(target_quat, target_xmat)
        obj_quat = self.data.qpos[self.obj_qposadr + 3 : self.obj_qposadr + 7].copy()

        orient_err = self._quat_orientation_error(obj_quat, target_quat)
        reward_orient = -orient_err * 0.5

        reward_info = {
            "dist": float(dist),
            "reward_dist": float(reward_dist),
            "reward_dist_tanh": float(reward_dist_tanh),
            "control_penalty": float(control_penalty),
            "reward_orient": float(reward_orient),
        }

        reward = reward_dist + reward_dist_tanh + control_penalty + reward_orient

        return reward, reward_info

    def reset_model(self):

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # === Robot pose: Lift position ===
        joint_lift = np.array(
            [0.19913068, -0.63058867, -1.75069009, 1.33782282, 0.61951687, 2.67788494]
        )

        qpos[:6] = joint_lift

        # === Close gripper ===
        qpos[self.gripL_qadr] = -0.02
        qpos[self.gripR_qadr] = 0.02

        # === Object position (10 cm height) ===
        x, y, z = 0.23644799, 0.05533779, 0.10097997
        qpos[self.obj_qposadr : self.obj_qposadr + 3] = [x, y, z]
        qpos[self.obj_qposadr + 3 : self.obj_qposadr + 7] = [1, 0, 0, 0]

        qvel[self.obj_dofadr : self.obj_dofadr + 6] = 0.0

        # === Random target ===
        tx = self.np_random.uniform(0.10, 0.27)
        ty = self.np_random.uniform(-0.20, 0.20)
        tz = 0.025

        self.model.site_pos[self.target_site_id] = np.array([tx, ty, tz])

        self.set_state(qpos, qvel)
        # mujoco.mj_forward(self.model, self.data)

        self.data.ctrl[:6] = joint_lift
        self.data.ctrl[6] = -0.0
        self.data.ctrl[7] = 0.0

        self.model.opt.gravity[:] = [0, 0, 0]
        for i in range(100):
            mujoco.mj_step(self.model, self.data)

            if i == 50:
                self.data.ctrl[6] = -0.02
                self.data.ctrl[7] = 0.02

        self.model.opt.gravity[:] = [0, 0, -9.81]
        self.current_step = 0

        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        return self._get_obs()

    def _get_obs(self):

        qpos = self.data.qpos
        qvel = self.data.qvel

        # Robot joint positions & velocities
        robot_qpos = qpos[: self.obj_qposadr]
        robot_qvel = qvel[: self.obj_dofadr]

        # Gripper state (2 finger joint terakhir)
        gripper_state = robot_qpos[-2:]

        # Object
        obj_pos = qpos[self.obj_qposadr : self.obj_qposadr + 3]
        obj_quat = qpos[self.obj_qposadr + 3 : self.obj_qposadr + 7]

        # End effector
        ee_pos = self.data.site("attachment_site").xpos
        ee_xmat = self.data.site("attachment_site").xmat
        ee_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(ee_quat, ee_xmat)

        # Target
        target_pos = self.data.site("target").xpos

        # Relative positions
        rel_obj_target = obj_pos - target_pos

        obs = np.concatenate(
            [
                robot_qpos,
                robot_qvel,
                gripper_state,
                obj_pos,
                obj_quat,
                target_pos,
                rel_obj_target,
                ee_pos,
                ee_quat,
            ]
        )

        return obs.astype(np.float32)

    def _quat_orientation_error(self, q1: np.ndarray, q2: np.ndarray) -> float:

        # Normalize
        q1 = q1 / (np.linalg.norm(q1) + 1e-8)
        q2 = q2 / (np.linalg.norm(q2) + 1e-8)

        # Dot product
        dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)

        # Angle error
        angle_err = 2.0 * np.arccos(dot)
        return float(angle_err)
