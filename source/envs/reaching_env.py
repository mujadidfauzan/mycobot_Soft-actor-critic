import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from .mj_utils import resolve_known_objects

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
        reward_axis_weight: float = 1.0,
        reward_orient_weight: float = 0.75,
        grasp_state_dataset_path: str | None = None,
        target_above_place_m: float = 0.04,
        randomize_object_and_place: bool = True,
        place_position_jitter_xy: float = 0.03,
        place_yaw_jitter_rad: float = np.pi,
        min_object_place_dist_xy: float = 0.10,
        success_pos_tolerance_m: float = 0.025,
        success_orient_tolerance_rad: float = 0.30,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_dist_weight,
            reward_axis_weight,
            reward_orient_weight,
            grasp_state_dataset_path,
            target_above_place_m,
            randomize_object_and_place,
            place_position_jitter_xy,
            place_yaw_jitter_rad,
            min_object_place_dist_xy,
            success_pos_tolerance_m,
            success_orient_tolerance_rad,
            **kwargs,
        )

        self._reward_dist_weight = reward_dist_weight
        self._reward_axis_weight = reward_axis_weight
        self._reward_orient_weight = reward_orient_weight
        self._grasp_state_dataset_path = grasp_state_dataset_path
        self._grasp_qpos: np.ndarray | None = None
        self._grasp_qvel: np.ndarray | None = None
        self._target_above_place_m = float(target_above_place_m)
        self._randomize_object_and_place = bool(randomize_object_and_place)
        self._place_position_jitter_xy = float(place_position_jitter_xy)
        self._place_yaw_jitter_rad = float(place_yaw_jitter_rad)
        self._min_object_place_dist_xy = float(min_object_place_dist_xy)
        self._success_pos_tolerance_m = float(success_pos_tolerance_m)
        self._success_orient_tolerance_rad = float(success_orient_tolerance_rad)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=None,
            camera_name="watching",
            **kwargs,
        )

        self._objects = resolve_known_objects(self.model)
        if len(self._objects) == 0:
            raise ValueError("No supported object bodies/joints found in the XML.")

        self.current_object_key: str = next(iter(self._objects.keys()))
        self.obj_body_name = self._objects[self.current_object_key].body_name
        self.obj_joint_name = self._objects[self.current_object_key].joint_name
        self.obj_qposadr = self._objects[self.current_object_key].qpos_adr
        self.obj_dofadr = self._objects[self.current_object_key].dof_adr

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

        self.robot_qpos_len = max(self.gripL_qadr, self.gripR_qadr) + 1
        self.robot_dof_len = max(self.gripL_dadr, self.gripR_dadr) + 1

        # Mapping between object type and its placement "hole" site
        self._place_site_by_object: dict[str, str] = {
            "cube": "cube_place_site",
            "triangle": "tri_place_site",
            "cylinder": "cyl_place_site",
        }
        self._place_body_by_object: dict[str, str] = {
            "cube": "cube_place",
            "triangle": "tri_place",
            "cylinder": "cyl_place",
        }
        self._available_tasks: list[tuple[str, str]] = []
        for obj_key, site_name in self._place_site_by_object.items():
            if obj_key in self._objects and mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, site_name
            ) != -1:
                self._available_tasks.append((obj_key, site_name))

        self._place_defaults: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for obj_key, body_name in self._place_body_by_object.items():
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id != -1:
                self._place_defaults[obj_key] = (
                    self.model.body_pos[body_id].copy(),
                    self.model.body_quat[body_id].copy(),
                )

        if self._grasp_state_dataset_path is not None:
            data = np.load(self._grasp_state_dataset_path)
            qpos = np.asarray(data["qpos"])
            qvel = np.asarray(data["qvel"])
            if qpos.ndim != 2 or qvel.ndim != 2:
                raise ValueError(
                    "Grasp state dataset must contain 2D arrays 'qpos' and 'qvel'."
                )
            if qpos.shape[1] != self.model.nq or qvel.shape[1] != self.model.nv:
                raise ValueError(
                    "Grasp state dataset shapes do not match the loaded model: "
                    f"qpos {qpos.shape} vs model.nq={self.model.nq}, "
                    f"qvel {qvel.shape} vs model.nv={self.model.nv}."
                )
            self._grasp_qpos = qpos
            self._grasp_qvel = qvel

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

    def _set_active_object(self, object_key: str) -> None:
        if object_key not in self._objects:
            raise KeyError(f"Unknown object key: {object_key}")
        self.current_object_key = object_key
        resolved = self._objects[object_key]
        self.obj_body_name = resolved.body_name
        self.obj_joint_name = resolved.joint_name
        self.obj_qposadr = resolved.qpos_adr
        self.obj_dofadr = resolved.dof_adr

    def step(self, action):
        self.current_step += 1
        action = action.copy()

        scale_arm = 0.01

        current_ctrl = self.data.ctrl.copy()
        target = current_ctrl.copy()

        target[:-2] += scale_arm * action[:-2]
        target[6] = -0.02
        target[7] = 0.02

        target = np.clip(target, self.action_space.low, self.action_space.high)

        self.do_simulation(target, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        reward_info["object_key"] = self.current_object_key
        info = reward_info
        terminated = bool(info["is_success"])
        truncated = self.current_step >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def _get_rew(self, action):
        obj_pos = self.data.body(self.obj_body_name).xpos.copy()
        target_pos = self.data.site("target").xpos.copy()

        pos_err = obj_pos - target_pos
        dist = np.linalg.norm(pos_err)
        reward_dist = -dist * self._reward_dist_weight
        reward_dist_tanh = 1.0 - float(np.tanh(float(dist) / 0.10))
        reward_axis = -self._reward_axis_weight * float(np.sum(np.abs(pos_err)))

        control_penalty = -0.001 * np.sum(np.square(action))

        target_xmat = self.data.site("target").xmat.copy()
        target_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(target_quat, target_xmat)
        obj_quat = self.data.qpos[self.obj_qposadr + 3 : self.obj_qposadr + 7].copy()

        orient_err = self._quat_orientation_error(obj_quat, target_quat)
        reward_orient = -orient_err * self._reward_orient_weight
        reward_orient_tanh = 1.0 - float(np.tanh(float(orient_err) / 0.50))
        is_success = (
            dist < self._success_pos_tolerance_m
            and orient_err < self._success_orient_tolerance_rad
        )
        success_bonus = 5.0 if is_success else 0.0

        reward_info = {
            "dist": float(dist),
            "pos_err_x": float(pos_err[0]),
            "pos_err_y": float(pos_err[1]),
            "pos_err_z": float(pos_err[2]),
            "reward_dist": float(reward_dist),
            "reward_dist_tanh": float(reward_dist_tanh),
            "reward_axis": float(reward_axis),
            "control_penalty": float(control_penalty),
            "orient_err": float(orient_err),
            "reward_orient": float(reward_orient),
            "reward_orient_tanh": float(reward_orient_tanh),
            "success_bonus": float(success_bonus),
            "is_success": bool(is_success),
        }

        reward = (
            reward_dist
            + reward_dist_tanh
            + reward_axis
            + reward_orient
            + reward_orient_tanh
            + control_penalty
            + success_bonus
        )

        return reward, reward_info

    def _quat_mul(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )

    def _yaw_to_quat(self, yaw: float) -> np.ndarray:
        half = 0.5 * yaw
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)

    def _randomize_place_pose(
        self, object_key: str, obj_xy: np.ndarray | None = None
    ) -> np.ndarray | None:
        if object_key not in self._place_body_by_object:
            return None
        if object_key not in self._place_defaults:
            return None

        body_name = self._place_body_by_object[object_key]
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            return None

        default_pos, default_quat = self._place_defaults[object_key]
        new_pos = default_pos.copy()
        for _ in range(50):
            jitter_xy = self.np_random.uniform(
                low=-self._place_position_jitter_xy,
                high=self._place_position_jitter_xy,
                size=2,
            )
            candidate_pos = default_pos.copy()
            candidate_pos[:2] += jitter_xy
            if obj_xy is None:
                new_pos = candidate_pos
                break
            dist_xy = np.linalg.norm(candidate_pos[:2] - obj_xy[:2])
            if dist_xy >= self._min_object_place_dist_xy:
                new_pos = candidate_pos
                break
        else:
            new_pos[:2] += np.array([self._min_object_place_dist_xy, 0.0])

        yaw = float(
            self.np_random.uniform(
                low=-self._place_yaw_jitter_rad, high=self._place_yaw_jitter_rad
            )
        )
        new_quat = self._quat_mul(default_quat, self._yaw_to_quat(yaw))

        self.model.body_pos[body_id] = new_pos
        self.model.body_quat[body_id] = new_quat
        return new_quat

    def reset_model(self):

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Choose which object (and matching place) to train on this episode
        place_site_name: str | None = None
        if self._grasp_qpos is None:
            if self._randomize_object_and_place and len(self._available_tasks) > 0:
                idx = int(self.np_random.integers(0, len(self._available_tasks)))
                obj_key, place_site_name = self._available_tasks[idx]
                self._set_active_object(obj_key)
            elif len(self._available_tasks) > 0:
                obj_key, place_site_name = self._available_tasks[0]
                self._set_active_object(obj_key)
        else:
            # If starting from a grasp dataset, keep the active object fixed and
            # only set the matching place/target if available.
            place_site_name = self._place_site_by_object.get(self.current_object_key)

        if self._grasp_qpos is not None and self._grasp_qvel is not None:
            idx = int(self.np_random.integers(0, self._grasp_qpos.shape[0]))
            qpos = self._grasp_qpos[idx].copy()
            qvel = self._grasp_qvel[idx].copy()

        # === Robot pose: Lift position ===
        joint_lift = np.array(
            [0.19913068, -0.63058867, -1.75069009, 1.33782282, 0.61951687, 2.67788494]
        )

        if self._grasp_qpos is None:
            qpos[:6] = joint_lift

        # === Close gripper ===
        qpos[self.gripL_qadr] = -0.02
        qpos[self.gripR_qadr] = 0.02

        # Move non-active objects far away (multi-object XML)
        far = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        for key, obj in self._objects.items():
            if key == self.current_object_key:
                continue
            qpos[obj.qpos_adr : obj.qpos_adr + 3] = far
            qpos[obj.qpos_adr + 3 : obj.qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
            qvel[obj.dof_adr : obj.dof_adr + 6] = 0.0

        self.set_state(qpos, qvel)

        # If not starting from a grasp dataset, place the selected object in the gripper.
        if self._grasp_qpos is None:
            ee_pos = self.data.site("attachment_site").xpos.copy()
            ee_xmat = self.data.site("attachment_site").xmat.copy().reshape(3, 3)
            obj_pos = ee_pos + ee_xmat @ np.array([0.0, 0.0, 0.02], dtype=np.float64)
            qpos[self.obj_qposadr : self.obj_qposadr + 3] = obj_pos
            qpos[self.obj_qposadr + 3 : self.obj_qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
            qvel[self.obj_dofadr : self.obj_dofadr + 6] = 0.0
            self.set_state(qpos, qvel)

        # Keep actuators consistent with the state
        self.data.ctrl[:6] = qpos[:6]
        self.data.ctrl[6] = 0.0
        self.data.ctrl[7] = 0.0

        # Target: 4cm above the chosen place site (or fallback to random XY).
        if place_site_name is not None:
            obj_xy = self.data.body(self.obj_body_name).xpos.copy()[:2]
            place_quat = self._randomize_place_pose(
                self.current_object_key, obj_xy=obj_xy
            )
            mujoco.mj_forward(self.model, self.data)
            place_pos = self.data.site(place_site_name).xpos.copy()
            target_pos = place_pos + np.array([0.0, 0.0, self._target_above_place_m])
            self.model.site_pos[self.target_site_id] = target_pos
            if place_quat is not None:
                self.model.site_quat[self.target_site_id] = place_quat
        else:
            tx = self.np_random.uniform(0.10, 0.27)
            ty = self.np_random.uniform(-0.20, 0.20)
            tz = 0.025
            self.model.site_pos[self.target_site_id] = np.array([tx, ty, tz])
            self.model.site_quat[self.target_site_id] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
        mujoco.mj_forward(self.model, self.data)

        # Stabilize initial grasp / contacts
        self.model.opt.gravity[:] = [0, 0, 0]
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        self.data.ctrl[6] = -0.02
        self.data.ctrl[7] = 0.02
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        self.model.opt.gravity[:] = [0, 0, -9.81]
        self.current_step = 0

        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        return self._get_obs()

    def _get_obs(self):

        qpos = self.data.qpos
        qvel = self.data.qvel

        # Robot joint positions & velocities
        robot_qpos = qpos[: self.robot_qpos_len]
        robot_qvel = qvel[: self.robot_dof_len]

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
        target_xmat = self.data.site("target").xmat
        target_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(target_quat, target_xmat)

        # Relative positions
        rel_obj_target = obj_pos - target_pos
        rel_quat = obj_quat - target_quat

        obs = np.concatenate(
            [
                robot_qpos,
                robot_qvel,
                gripper_state,
                obj_pos,
                obj_quat,
                target_pos,
                target_quat,
                rel_obj_target,
                rel_quat,
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
