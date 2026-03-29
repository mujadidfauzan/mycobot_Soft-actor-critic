import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from .mj_utils import resolve_known_objects, resolve_object

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class GraspingEnv(MujocoEnv, utils.EzPickle):

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
        reward_dist_target_weight: float = 1,
        randomize_object: bool = True,
        target_above_object_m: float = 0.05,
        place_position_jitter_xy: float = 0.03,
        place_yaw_jitter_rad: float = np.pi,
        min_object_place_dist_xy: float = 0.10,
        place_move_penalty_weight: float = 2.0,
        object_spawn_radius_min_m: float = 0.15,
        object_spawn_radius_max_m: float = 0.27,
        object_spawn_height_m: float = 0.050998,
        object_yaw_jitter_rad: float = np.pi,
        task_selection_mode: str | None = None,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_dist_weight,
            reward_dist_target_weight,
            randomize_object,
            target_above_object_m,
            place_position_jitter_xy,
            place_yaw_jitter_rad,
            min_object_place_dist_xy,
            place_move_penalty_weight,
            object_spawn_radius_min_m,
            object_spawn_radius_max_m,
            object_spawn_height_m,
            object_yaw_jitter_rad,
            task_selection_mode,
            **kwargs,
        )

        self._reward_dist_weight = reward_dist_weight
        self._reward_dist_target_weight = reward_dist_target_weight
        self._randomize_object = bool(randomize_object)
        self._target_above_object_m = float(target_above_object_m)
        self._place_position_jitter_xy = float(place_position_jitter_xy)
        self._place_yaw_jitter_rad = float(place_yaw_jitter_rad)
        self._min_object_place_dist_xy = float(min_object_place_dist_xy)
        self._place_move_penalty_weight = float(place_move_penalty_weight)
        self._object_spawn_radius_min_m = float(object_spawn_radius_min_m)
        self._object_spawn_radius_max_m = float(object_spawn_radius_max_m)
        self._object_spawn_height_m = float(object_spawn_height_m)
        self._object_yaw_jitter_rad = float(object_yaw_jitter_rad)
        if self._object_spawn_radius_min_m < 0.0:
            raise ValueError("object_spawn_radius_min_m must be >= 0.")
        if self._object_spawn_radius_max_m <= self._object_spawn_radius_min_m:
            raise ValueError(
                "object_spawn_radius_max_m must be greater than object_spawn_radius_min_m."
            )
        if task_selection_mode is None:
            task_selection_mode = "cycle" if self._randomize_object else "fixed"
        if task_selection_mode not in {"cycle", "random", "fixed"}:
            raise ValueError(
                "task_selection_mode must be one of {'cycle', 'random', 'fixed'}."
            )
        self._task_selection_mode = str(task_selection_mode)
        self._next_task_index = 0

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
            resolved_obj = resolve_object(self.model)
            self._objects = {
                "obj": resolved_obj,
            }

        self.current_object_key: str = next(iter(self._objects.keys()))
        resolved_obj = self._objects[self.current_object_key]
        self.obj_body_name = resolved_obj.body_name
        self.obj_joint_name = resolved_obj.joint_name
        self.obj_frame_site_name = resolved_obj.frame_site_name
        self.obj_qposadr = resolved_obj.qpos_adr
        self.obj_dofadr = resolved_obj.dof_adr

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
        self._place_defaults: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._place_body_ids: dict[str, int] = {}
        self._available_tasks: list[tuple[str, str]] = []
        for obj_key, site_name in self._place_site_by_object.items():
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            body_name = self._place_body_by_object.get(obj_key)
            body_id = (
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if body_name is not None
                else -1
            )
            if body_id != -1:
                self._place_body_ids[obj_key] = body_id
                self._place_defaults[obj_key] = (
                    self.model.body_pos[body_id].copy(),
                    self.model.body_quat[body_id].copy(),
                )
            if obj_key in self._objects and site_id != -1 and body_id != -1:
                self._available_tasks.append((obj_key, site_name))

        self._active_place_site_name: str | None = None
        self._active_place_body_name: str | None = None
        self._active_place_body_id: int = -1
        self._active_place_reset_pos: np.ndarray | None = None
        self._active_place_reset_quat: np.ndarray | None = None
        self._body_visual_state = self._build_body_visual_state()

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

        dummy_obs = self._get_obs()
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=dummy_obs.shape, dtype=np.float32
        )

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

    def _set_active_object(self, object_key: str) -> None:
        if object_key not in self._objects:
            raise KeyError(f"Unknown object key: {object_key}")
        self.current_object_key = object_key
        resolved = self._objects[object_key]
        self.obj_body_name = resolved.body_name
        self.obj_joint_name = resolved.joint_name
        self.obj_frame_site_name = resolved.frame_site_name
        self.obj_qposadr = resolved.qpos_adr
        self.obj_dofadr = resolved.dof_adr

    def _set_active_place(self, object_key: str) -> str | None:
        site_name = self._place_site_by_object.get(object_key)
        if site_name is not None:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id == -1:
                site_name = None

        body_name = self._place_body_by_object.get(object_key)
        body_id = self._place_body_ids.get(object_key, -1)
        if body_id == -1:
            body_name = None

        self._active_place_site_name = site_name
        self._active_place_body_name = body_name
        self._active_place_body_id = body_id
        return site_name

    def _select_episode_object_key(self, object_keys: list[str]) -> str:
        if len(object_keys) == 0:
            raise ValueError("No object keys available for episode selection.")

        if self._task_selection_mode == "random":
            idx = int(self.np_random.integers(0, len(object_keys)))
            return object_keys[idx]

        if self._task_selection_mode == "cycle":
            idx = self._next_task_index % len(object_keys)
            self._next_task_index += 1
            return object_keys[idx]

        if self.current_object_key in object_keys:
            return self.current_object_key
        return object_keys[0]

    # HELPER
    def enable_frame_visualization(self):
        if self.render_mode != "human":
            return

        self.render()

        viewer = self.mujoco_renderer._get_viewer("human")
        viewer.vopt.frame = mujoco.mjtFrame.mjFRAME_SITE

    def gripper_ctrl(self, close: bool, target):
        if close:
            target[6] = -0.02
            target[7] = 0.02
        else:
            target[6] = 0.02
            target[7] = -0.02

    def _site_xquat(self, site_name: str) -> np.ndarray:
        xmat = self.data.site(site_name).xmat
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, xmat)
        return quat

    def _obj_quat(self) -> np.ndarray:
        return self.data.qpos[self.obj_qposadr + 3 : self.obj_qposadr + 7].copy()

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

    def _quat_orientation_error(self, q1: np.ndarray, q2: np.ndarray) -> float:
        q1 = q1 / (np.linalg.norm(q1) + 1e-8)
        q2 = q2 / (np.linalg.norm(q2) + 1e-8)
        dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
        return float(2.0 * np.arccos(dot))

    def _build_body_visual_state(self) -> dict[str, dict[str, np.ndarray]]:
        state: dict[str, dict[str, np.ndarray]] = {}
        tracked_body_names = set(self._place_body_by_object.values()) | {
            resolved.body_name for resolved in self._objects.values()
        }
        for body_name in tracked_body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                continue
            geom_adr = int(self.model.body_geomadr[body_id])
            geom_num = int(self.model.body_geomnum[body_id])
            geom_ids = np.arange(geom_adr, geom_adr + geom_num, dtype=np.int32)
            state[body_name] = {
                "geom_ids": geom_ids,
                "rgba": self.model.geom_rgba[geom_ids].copy(),
                "contype": self.model.geom_contype[geom_ids].copy(),
                "conaffinity": self.model.geom_conaffinity[geom_ids].copy(),
            }
        return state

    def _set_body_enabled(self, body_name: str, enabled: bool) -> None:
        body_state = self._body_visual_state.get(body_name)
        if body_state is None:
            return
        geom_ids = body_state["geom_ids"]
        if enabled:
            self.model.geom_rgba[geom_ids] = body_state["rgba"]
            self.model.geom_contype[geom_ids] = body_state["contype"]
            self.model.geom_conaffinity[geom_ids] = body_state["conaffinity"]
        else:
            hidden_rgba = body_state["rgba"].copy()
            hidden_rgba[:, 3] = 0.0
            self.model.geom_rgba[geom_ids] = hidden_rgba
            self.model.geom_contype[geom_ids] = 0
            self.model.geom_conaffinity[geom_ids] = 0

    def _sample_object_spawn_position(self) -> np.ndarray:
        radius = float(
            self.np_random.uniform(
                self._object_spawn_radius_min_m, self._object_spawn_radius_max_m
            )
        )
        yaw = float(self.np_random.uniform(-np.pi / 2.0, np.pi / 2.0))
        return np.array(
            [
                radius * np.cos(yaw),
                radius * np.sin(yaw),
                self._object_spawn_height_m,
            ],
            dtype=np.float64,
        )

    def _sample_object_quat(self) -> np.ndarray:
        yaw = float(
            self.np_random.uniform(
                low=-self._object_yaw_jitter_rad, high=self._object_yaw_jitter_rad
            )
        )
        return self._yaw_to_quat(yaw)

    def _prepare_place_visibility(self) -> None:
        for key, body_id in self._place_body_ids.items():
            body_name = self._place_body_by_object[key]
            if key == self.current_object_key:
                default_pos, default_quat = self._place_defaults[key]
                self.model.body_pos[body_id] = default_pos
                self.model.body_quat[body_id] = default_quat
                self._set_body_enabled(body_name, True)
            else:
                self._set_body_enabled(body_name, False)

    def _randomize_place_pose(
        self, object_key: str, obj_xy: np.ndarray | None = None
    ) -> np.ndarray | None:
        if object_key not in self._place_body_ids:
            return None
        if object_key not in self._place_defaults:
            return None

        body_id = self._place_body_ids[object_key]
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

    def _capture_active_place_pose(self) -> None:
        if self._active_place_body_name is None:
            self._active_place_reset_pos = None
            self._active_place_reset_quat = None
            return

        self._active_place_reset_pos = self.data.body(
            self._active_place_body_name
        ).xpos.copy()
        self._active_place_reset_quat = self.data.body(
            self._active_place_body_name
        ).xquat.copy()

    def _get_place_move_penalty(self) -> tuple[float, float, float]:
        if (
            self._active_place_body_name is None
            or self._active_place_reset_pos is None
            or self._active_place_reset_quat is None
        ):
            return 0.0, 0.0, 0.0

        place_pos = self.data.body(self._active_place_body_name).xpos.copy()
        place_quat = self.data.body(self._active_place_body_name).xquat.copy()
        place_move_dist = float(
            np.linalg.norm(place_pos - self._active_place_reset_pos)
        )
        place_move_angle = self._quat_orientation_error(
            place_quat, self._active_place_reset_quat
        )
        reward_place_move = -self._place_move_penalty_weight * (
            place_move_dist + 0.25 * place_move_angle
        )
        return reward_place_move, place_move_dist, place_move_angle

    # MAIN ENV
    def step(self, action):
        self.current_step += 1
        action = action.copy()

        scale_arm = 0.01

        current_ctrl = self.data.ctrl.copy()
        target = current_ctrl.copy()

        target[:-2] += scale_arm * action[:-2]
        target[6] = 0.02
        target[7] = -0.02
        if (
            np.linalg.norm(
                self.data.site("attachment_site").xpos.copy()
                - self.data.body(self.obj_body_name).xpos.copy()
            )
            < 0.03
        ):
            self.gripper_ctrl(close=True, target=target)
        else:
            self.gripper_ctrl(close=False, target=target)

        target = np.clip(target, self.action_space.low, self.action_space.high)

        self.do_simulation(target, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        reward_info["object_key"] = self.current_object_key
        reward_info["place_site_name"] = self._active_place_site_name
        reward_info["place_body_name"] = self._active_place_body_name
        info = reward_info
        terminated = False
        truncated = self.current_step >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def _get_rew(self, action):
        ee_pos = self.data.site("attachment_site").xpos.copy()
        obj_pos = self.data.body(self.obj_body_name).xpos.copy()
        target_pos = self.data.site("target").xpos.copy()

        dist = np.linalg.norm(ee_pos - obj_pos)
        reward_dist = -dist * self._reward_dist_weight
        reward_dist_tanh = 1.0 - float(np.tanh(float(dist) / 0.10))

        target_dist = np.linalg.norm(target_pos - obj_pos)
        reward_target = -target_dist * self._reward_dist_target_weight
        reward_target_tanh = 1.0 - float(np.tanh(float(target_dist) / 0.10))

        obj_quat = (
            self._site_xquat(self.obj_frame_site_name)
            if self.obj_frame_site_name is not None
            else self._obj_quat()
        )
        ee_quat = self._site_xquat("attachment_site")

        obj_ee_quat_dot = np.abs(np.dot(obj_quat, ee_quat))
        obj_ee_quat_dot = np.clip(obj_ee_quat_dot, -1.0, 1.0)
        angle_error = 2.0 * np.arccos(obj_ee_quat_dot)
        reward_orientation_ee_error = (1.0 - float(np.tanh(angle_error / 0.5))) * 0.5

        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        quat_dot = np.abs(np.dot(obj_quat, target_quat))
        quat_dot = np.clip(quat_dot, -1.0, 1.0)
        orientation_error = 1.0 - quat_dot
        reward_orient = -orientation_error * 0.5

        reward_place_move, place_move_dist, place_move_angle = (
            self._get_place_move_penalty()
        )
        control_penalty = -0.001 * np.sum(np.square(action))
        reward_info = {
            "dist": float(dist),
            "reward_dist": float(reward_dist),
            "reward_dist_tanh": float(reward_dist_tanh),
            "control_penalty": float(control_penalty),
            "reward_target": float(reward_target),
            "reward_target_tanh": float(reward_target_tanh),
            "reward_orient": float(reward_orient),
            "reward_ee_orient": float(reward_orientation_ee_error),
            "place_move_dist": float(place_move_dist),
            "place_move_angle": float(place_move_angle),
            "reward_place_move": float(reward_place_move),
        }

        reward = (
            reward_dist
            + reward_dist_tanh
            + control_penalty
            + reward_target
            + reward_target_tanh
            + reward_orient
            + reward_orientation_ee_error
            + reward_place_move
        )

        return reward, reward_info

    def reset_model(self):

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        candidate_keys = (
            [object_key for object_key, _ in self._available_tasks]
            if len(self._available_tasks) > 0
            else list(self._objects.keys())
        )
        object_key = self._select_episode_object_key(candidate_keys)
        self._set_active_object(object_key)
        self._set_active_place(object_key)

        adr = self.obj_qposadr
        vadr = self.obj_dofadr

        obj_pos = self._sample_object_spawn_position()
        obj_quat = self._sample_object_quat()

        far = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        for key, obj in self._objects.items():
            self._set_body_enabled(obj.body_name, key == self.current_object_key)
            if key == self.current_object_key:
                continue
            qpos[obj.qpos_adr : obj.qpos_adr + 3] = far
            qpos[obj.qpos_adr + 3 : obj.qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
            qvel[obj.dof_adr : obj.dof_adr + 6] = 0.0

        self._prepare_place_visibility()
        if self._active_place_site_name is not None:
            self._randomize_place_pose(self.current_object_key, obj_xy=obj_pos[:2])

        qpos[adr + 0] = obj_pos[0]
        qpos[adr + 1] = obj_pos[1]
        qpos[adr + 2] = obj_pos[2]
        qpos[adr + 3 : adr + 7] = obj_quat
        qvel[vadr : vadr + 6] = 0.0

        qpos[self.gripL_qadr] = 0.0
        qpos[self.gripR_qadr] = 0.0
        qvel[self.gripL_dadr] = 0.0
        qvel[self.gripR_dadr] = 0.0

        self.set_state(qpos, qvel)
        self.data.ctrl[:6] = qpos[:6]
        self.data.ctrl[6] = 0.02
        self.data.ctrl[7] = -0.02
        mujoco.mj_forward(self.model, self.data)

        target_pos = obj_pos + np.array(
            [0.0, 0.0, self._target_above_object_m], dtype=np.float64
        )
        self.model.site_pos[self.target_site_id] = target_pos
        self.model.site_quat[self.target_site_id] = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )

        mujoco.mj_forward(self.model, self.data)
        self._capture_active_place_pose()

        self.current_step = 0
        if self.render_mode == "human":
            self.enable_frame_visualization()

        return self._get_obs()

    def _get_obs(self):

        qpos = self.data.qpos
        qvel = self.data.qvel

        robot_qpos = qpos[: self.robot_qpos_len]
        robot_qvel = qvel[: self.robot_dof_len]

        gripper_state = robot_qpos[-2:]

        obj_pos = qpos[self.obj_qposadr : self.obj_qposadr + 3]
        obj_quat = (
            self._site_xquat(self.obj_frame_site_name)
            if self.obj_frame_site_name is not None
            else self._obj_quat()
        )

        ee_pos = self.data.site("attachment_site").xpos
        ee_xmat = self.data.site("attachment_site").xmat
        ee_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(ee_quat, ee_xmat)

        target_pos = self.data.site("target").xpos

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
        grip_l = float(self.data.qpos[self.gripL_qadr])
        grip_r = float(self.data.qpos[self.gripR_qadr])
        gripper_state = "closed" if (grip_l < 0.0 and grip_r > 0.0) else "open"
        return {
            "step": self.current_step,
            "object_key": self.current_object_key,
            "place_body_name": self._active_place_body_name,
            "obj_pos": self.data.body(self.obj_body_name).xpos.copy(),
            "target_pos": self.data.site("target").xpos.copy(),
            "ee_pos": self.data.site("attachment_site").xpos.copy(),
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "gripper_state": gripper_state,
            "joint_angles": self.data.qpos[: self.obj_qposadr].copy(),
        }
