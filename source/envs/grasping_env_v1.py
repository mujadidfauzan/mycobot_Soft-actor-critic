import time

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class GraspingEnvV1(MujocoEnv, utils.EzPickle):

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
        xml_file: str = "object_lift_v1.xml",
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        reward_dist_weight: float = 2,
        reward_dist_target_weight: float = 1,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_dist_weight,
            reward_dist_target_weight,
            **kwargs,
        )

        self._reward_dist_weight = reward_dist_weight
        self._reward_dist_target_weight = reward_dist_target_weight

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=None,
            camera_name="watching",
            **kwargs,
        )

        self.object_configs = [
            {
                "body": "cube_obj",
                "joint": "cube",
                "site": "cube_frame",
            },
            {
                "body": "triangle_obj",
                "joint": "triangle_joint",
                "site": "triangle_frame",
            },
            {
                "body": "cylinder_obj",
                "joint": "cylinder_joint",
                "site": "cylinder_frame",
            },
        ]

        self.object_meta = []
        for cfg in self.object_configs:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, cfg["joint"]
            )
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, cfg["body"]
            )
            self.object_meta.append(
                {
                    "body": cfg["body"],
                    "joint": cfg["joint"],
                    "site": cfg["site"],
                    "joint_id": joint_id,
                    "body_id": body_id,
                    "qposadr": int(self.model.jnt_qposadr[joint_id]),
                    "dofadr": int(self.model.jnt_dofadr[joint_id]),
                }
            )

        self.active_object = self.object_meta[0]

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

        # self.gripper_state = "open"

        self.max_episode_steps = 500
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

    # HELPER
    def enable_frame_visualization(self):
        if self.render_mode != "human":
            return

        self.render()

        viewer = self.mujoco_renderer._get_viewer("human")
        viewer.vopt.frame = mujoco.mjtFrame.mjFRAME_SITE

    def gripper_ctrl(self, close: bool, target):
        if close:
            # self.gripper_state = "closed"
            target[6] = -0.02
            target[7] = 0.02
        else:
            # self.gripper_state = "open"
            target[6] = 0.02
            target[7] = -0.02

    def _site_xquat(self, site_name: str) -> np.ndarray:
        xmat = self.data.site(site_name).xmat
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, xmat)
        return quat

    def _get_active_obj_pos(self):
        return self.data.body(self.active_object["body"]).xpos.copy()

    def _get_active_obj_quat(self):
        return self._site_xquat(self.active_object["site"])

    # MAIN ENV
    def step(self, action):
        self.current_step += 1
        action = action.copy()

        scale_arm = 0.01

        current_ctrl = self.data.ctrl.copy()
        # print("Current control:", current_ctrl)
        target = current_ctrl.copy()
        delta_deg = np.rad2deg(action[:6] * scale_arm)
        # print("Delta degrees:", delta_deg)

        target[:-2] += scale_arm * action[:-2]
        # print("Target after adding action:", target)
        # print("Target in degrees:", np.rad2deg(target[:-2]))
        target[6] = 0.02
        target[7] = -0.02
        # print("Target before gripper control:", target)
        if (
            np.linalg.norm(
                self.data.site("attachment_site").xpos.copy()
                - self._get_active_obj_pos()
            )
            < 0.03
        ):
            self.gripper_ctrl(close=True, target=target)
        else:
            self.gripper_ctrl(close=False, target=target)

        target = np.clip(target, self.action_space.low, self.action_space.high)
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
        ee_pos = self.data.site("attachment_site").xpos.copy()
        obj_pos = self._get_active_obj_pos()
        target_pos = self.data.site("target").xpos.copy()

        dist = np.linalg.norm(ee_pos - obj_pos)
        reward_dist = -dist * self._reward_dist_weight
        reward_dist_tanh = 1.0 - float(np.tanh(float(dist) / 0.10))

        target_dist = np.linalg.norm(target_pos - obj_pos)
        reward_target = -target_dist * self._reward_dist_target_weight
        reward_target_tanh = 1.0 - float(np.tanh(float(target_dist) / 0.10))

        obj_quat = self._get_active_obj_quat()
        ee_quat = self._site_xquat("attachment_site")

        # ee quat
        obj_ee_quat_dot = np.abs(np.dot(obj_quat, ee_quat))
        obj_ee_quat_dot = np.clip(obj_ee_quat_dot, -1.0, 1.0)
        angle_error = 2.0 * np.arccos(obj_ee_quat_dot)
        reward_orientation_ee_error = (1.0 - float(np.tanh(angle_error / 0.5))) * 0.5

        # obj quat
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        quat_dot = np.abs(np.dot(obj_quat, target_quat))
        quat_dot = np.clip(quat_dot, -1.0, 1.0)
        orientation_error = 1.0 - quat_dot
        reward_orient = -orientation_error * 0.5

        control_penalty = -0.001 * np.sum(np.square(action))
        reward_info = {
            "dist": float(dist),
            "target_dist": float(target_dist),
            "reward_dist": float(reward_dist),
            "reward_dist_tanh": float(reward_dist_tanh),
            "control_penalty": float(control_penalty),
            "reward_target": float(reward_target),
            "reward_target_tanh": float(reward_target_tanh),
            # "reward_orient": float(reward_orient),
            # "reward_ee_orient": float(reward_orientation_ee_error),
        }

        reward = (
            reward_dist
            + reward_dist_tanh
            # + touch_bonus
            + control_penalty
            + reward_target
            # + reward_lift
            + reward_target_tanh
            # + reward_orient
            # + reward_orientation_ee_error
        )

        return reward, reward_info

    def reset_model(self):

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Random Object
        obj_idx = int(self.np_random.integers(0, len(self.object_meta)))
        self.active_object = self.object_meta[obj_idx]
        print(f"Selected active object: {self.active_object['body']}")
        for meta in self.object_meta:
            adr = meta["qposadr"]
            vadr = meta["dofadr"]

            # move object far below the floor
            qpos[adr + 0] = 10.0
            qpos[adr + 1] = 10.0
            qpos[adr + 2] = 10.0

            # identity quaternion
            qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]

            # zero freejoint velocity
            qvel[vadr : vadr + 6] = 0.0

        adr = self.active_object["qposadr"]
        vadr = self.active_object["dofadr"]

        # Random XY
        x = self.np_random.uniform(0.15, 0.27)
        y = self.np_random.uniform(-0.10, 0.10)
        z = 0.050998

        # Set position
        qpos[adr + 0] = x
        qpos[adr + 1] = y
        qpos[adr + 2] = z

        # random rotation around Z-axis
        yaw = self.np_random.uniform(-np.pi, np.pi)

        cy = np.cos(yaw / 2.0)
        sy = np.sin(yaw / 2.0)

        # quaternion MuJoCo: [w, x, y, z]
        qpos[adr + 3 : adr + 7] = [cy, 0.0, 0.0, sy]

        # Zero object velocity
        qvel[vadr : vadr + 6] = 0.0

        # Zero object velocity
        qvel[vadr : vadr + 6] = 0.0

        tz = 0.1

        self.model.site_pos[self.target_site_id] = np.array(
            [x, y, tz], dtype=np.float64
        )

        # Open Gripper
        qpos[self.gripL_qadr] = 0
        qpos[self.gripR_qadr] = 0

        qvel[self.gripL_dadr] = 0.0
        qvel[self.gripR_dadr] = 0.0

        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data)
        self.current_step = 0
        if self.render_mode == "human":
            self.enable_frame_visualization()

        return self._get_obs()

    def _get_obs(self):

        qpos = self.data.qpos
        qvel = self.data.qvel

        # Robot joint positions & velocities
        robot_qpos = qpos[: self.gripL_qadr]
        robot_qvel = qvel[: self.gripL_dadr]

        # Gripper state (2 finger joint terakhir)
        gripper_state = np.array(
            [qpos[self.gripL_qadr], qpos[self.gripR_qadr]], dtype=np.float64
        )

        # Object
        if self.active_object is None:
            obj_pos = np.zeros(3, dtype=np.float64)
            obj_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            obj_pos = self._get_active_obj_pos()
            obj_quat = self._get_active_obj_quat()

        # End effector
        ee_pos = self.data.site("attachment_site").xpos.copy()
        ee_xmat = self.data.site("attachment_site").xmat
        ee_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(ee_quat, ee_xmat)

        # Target
        target_pos = self.data.site("target").xpos

        # Relative positions
        rel_obj_ee = obj_pos - ee_pos
        rel_obj_target = obj_pos - target_pos

        obs = np.concatenate(
            [
                robot_qpos,
                robot_qvel,
                gripper_state,
                obj_pos,
                obj_quat,
                target_pos,
                rel_obj_ee,
                rel_obj_target,
                ee_pos,
                ee_quat,
            ]
        )

        return obs.astype(np.float32)

    def get_physics_state(self):
        return {
            "step": self.current_step,
            "obj_pos": self.data.body("obj").xpos.copy(),
            "target_pos": self.data.site("target").xpos.copy(),
            "ee_pos": self.data.site("attachment_site").xpos.copy(),
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "gripper_state": self.gripper_state,
            "joint_angles": self.data.qpos[: self.obj_qposadr].copy(),
        }
