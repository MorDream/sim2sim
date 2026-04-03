from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Optional

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
import glfw
import cv2
import os

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

try:
    import pygame
except Exception:
    pygame = None

try:
    import keyboard
except Exception:
    keyboard = None


CURRENT_FILE = Path(__file__).resolve()
LEGGED_GYM_ROOT_DIR = CURRENT_FILE.parents[2]
DEFAULT_CONFIG_PATH = LEGGED_GYM_ROOT_DIR / "deploy" / "deploy_mujoco" / "configs" 

GO2_OBS_COMPONENTS = [
    "lin_vel",
    "ang_vel",
    "projected_gravity",
    "commands",
    "dof_pos",
    "dof_vel",
    "last_actions",
    "height_measurements",
]

GO2_MEASURED_POINTS_X = [float(x) for x in np.arange(-0.5, 1.51, 0.1)]
GO2_MEASURED_POINTS_Y = [float(y) for y in np.arange(-0.5, 0.51, 0.1)]
GO2_HEIGHT_MEASUREMENTS_OFFSET = -0.2
GO2_NUM_ACTIONS = 12
GO2_NUM_HEIGHT_POINTS = len(GO2_MEASURED_POINTS_X) * len(GO2_MEASURED_POINTS_Y)
GO2_NUM_OBS = 48 + GO2_NUM_HEIGHT_POINTS
GO2_MUJOCO_JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]


def resolve_config_path(path_str: str, config_dir: Path) -> Path:
    path_str = path_str.replace("{LEGGED_GYM_ROOT_DIR}", str(LEGGED_GYM_ROOT_DIR))
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    config_relative = (config_dir / candidate).resolve()
    if config_relative.exists():
        return config_relative
    return (LEGGED_GYM_ROOT_DIR / candidate).resolve()


def quat_rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64)
    q_w = quat_wxyz[0]
    q_vec = quat_wxyz[1:]
    a = vec * (2.0 * q_w * q_w - 1.0)
    b = np.cross(q_vec, vec) * (2.0 * q_w)
    c = q_vec * (2.0 * np.dot(q_vec, vec))
    return (a - b + c).astype(np.float32)


def quat_to_yaw(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in quat_wxyz]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    return (target_q - q) * kp + (target_dq - dq) * kd


def extract_action_tensor(policy_output: Any) -> torch.Tensor:
    if torch.is_tensor(policy_output):
        return policy_output

    if isinstance(policy_output, (tuple, list)):
        for item in policy_output:
            try:
                return extract_action_tensor(item)
            except TypeError:
                continue

    if isinstance(policy_output, dict):
        for key in ("actions", "action", "mu", "mean"):
            if key in policy_output:
                return extract_action_tensor(policy_output[key])
        for value in policy_output.values():
            try:
                return extract_action_tensor(value)
            except TypeError:
                continue

    raise TypeError(f"Unsupported policy output type: {type(policy_output)}")


def try_init_joystick(enable_joystick: bool):
    if not enable_joystick:
        return None
    if pygame is None:
        print("[Info] pygame not available, fallback to fixed command.")
        return None

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() <= 0:
        print("[Info] No joystick detected, fallback to fixed command.")
        return None

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"[Info] Joystick connected: {joystick.get_name()}")
    return joystick


def read_xbox_command(joystick, max_cmd: np.ndarray) -> np.ndarray:
    pygame.event.pump()
    dead_zone = 0.10
    lx = joystick.get_axis(0)
    ly = joystick.get_axis(1)
    rx = joystick.get_axis(3)

    if abs(lx) < dead_zone:
        lx = 0.0
    if abs(ly) < dead_zone:
        ly = 0.0
    if abs(rx) < dead_zone:
        rx = 0.0

    cmd_x = -ly * max_cmd[0]
    cmd_y = -lx * max_cmd[1]
    cmd_yaw = -rx * max_cmd[2]
    return np.array([cmd_x, cmd_y, cmd_yaw], dtype=np.float32)


def read_keyboard_command(current_cmd: np.ndarray, max_cmd: np.ndarray) -> np.ndarray:
    """Handle keyboard input for robot control.
    
    Controls:
    - Space: Set command to [0, 0, 0] (stop robot)
    - Up Arrow: Increase forward velocity by 0.4
    - Down Arrow: Decrease forward velocity by 0.4
    - Left Arrow: Decrease yaw velocity (turn left)
    - Right Arrow: Increase yaw velocity (turn right)
    """
    cmd = current_cmd.copy()
    
    if keyboard is not None:
        # Space key - stop robot
        if keyboard.is_pressed('space'):
            cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            # Up/Down arrows - control forward velocity
            if keyboard.is_pressed('up'):
                cmd[0] = min(cmd[0] + 0.4, max_cmd[0])
            elif keyboard.is_pressed('down'):
                cmd[0] = max(cmd[0] - 0.4, -max_cmd[0])
            
            # Left/Right arrows - control yaw velocity
            if keyboard.is_pressed('left'):
                cmd[2] = max(cmd[2] - 0.4, -max_cmd[2])
            elif keyboard.is_pressed('right'):
                cmd[2] = min(cmd[2] + 0.4, max_cmd[2])
    
    return cmd


class HeightMeasurementSampler:
    """
    Approximate legged_gym height measurements in MuJoCo:
    1. rotate local sampling grid by base yaw
    2. cast downward rays
    3. use clip(base_z + offset - measured_heights, -1, 1) * scale
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        base_body_id: int,
        points_x,
        points_y,
        height_offset: float,
        height_scale: float,
        ray_start_height: float = 1.0,
    ) -> None:
        self.model = model
        self.data = data
        self.base_body_id = int(base_body_id)
        self.height_offset = float(height_offset)
        self.height_scale = float(height_scale)
        self.ray_start_height = float(ray_start_height)

        grid_x, grid_y = np.meshgrid(
            np.asarray(points_x, dtype=np.float64),
            np.asarray(points_y, dtype=np.float64),
            indexing="ij",
        )
        self.local_points_xy = np.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1)],
            axis=-1,
        )
        self.num_points = self.local_points_xy.shape[0]

    def sample_heights(self, base_pos: np.ndarray, base_quat_wxyz: np.ndarray) -> np.ndarray:
        yaw = quat_to_yaw(base_quat_wxyz)
        c = math.cos(yaw)
        s = math.sin(yaw)
        yaw_rot = np.array([[c, -s], [s, c]], dtype=np.float64)

        world_xy = self.local_points_xy @ yaw_rot.T
        world_xy[:, 0] += float(base_pos[0])
        world_xy[:, 1] += float(base_pos[1])

        ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        geomid = np.array([-1], dtype=np.int32)
        heights = np.empty(self.num_points, dtype=np.float32)

        for i in range(self.num_points):
            ray_start = np.array(
                [
                    world_xy[i, 0],
                    world_xy[i, 1],
                    float(base_pos[2]) + self.ray_start_height,
                ],
                dtype=np.float64,
            )
            dist = mujoco.mj_ray(
                self.model,
                self.data,
                ray_start,
                ray_dir,
                None,
                1,
                self.base_body_id,
                geomid,
            )
            if np.isfinite(dist) and dist >= 0.0:
                heights[i] = float(ray_start[2] - dist)
            else:
                heights[i] = float(base_pos[2] - 10.0)

        return heights

    def get_height_observation(self, base_pos: np.ndarray, base_quat_wxyz: np.ndarray) -> np.ndarray:
        measured_heights = self.sample_heights(base_pos, base_quat_wxyz)
        heights = np.clip(base_pos[2] + self.height_offset - measured_heights, -1.0, 1.0)
        heights = heights * self.height_scale
        return heights.astype(np.float32)


class MujocoSim2SimEvalJit:
    def __init__(
        self,
        config_path: Path,
        policy_path: Optional[Path] = None,
        xml_path: Optional[Path] = None,
        simulation_duration: Optional[float] = None,
        save_video: bool = False,
        video_path: Optional[Path] = None,
    ) -> None:
        self.config_path = config_path.resolve()
        self.config_dir = self.config_path.parent

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.policy_path = policy_path or resolve_config_path(self.cfg["policy_path"], self.config_dir)
        print(f"[Info] Using policy from: {self.policy_path}")
        self.xml_path = resolve_config_path(self.cfg["xml_path"], self.config_dir)
        self.simulation_duration = float(
            simulation_duration if simulation_duration is not None else self.cfg.get("simulation_duration", 60.0)
        ) 

        self.simulation_dt = float(self.cfg.get("simulation_dt", 0.005))
        self.control_decimation = int(self.cfg.get("control_decimation", 4))

        self.kps = np.asarray(self.cfg["kps"], dtype=np.float32)
        self.kds = np.asarray(self.cfg["kds"], dtype=np.float32)
        self.default_angles_mj = np.asarray(self.cfg["default_angles"], dtype=np.float32)

        self.lin_vel_scale = float(self.cfg.get("lin_vel_scale", 1.0))
        self.ang_vel_scale = float(self.cfg.get("ang_vel_scale", 1.0))
        self.dof_pos_scale = float(self.cfg.get("dof_pos_scale", 1.0))
        self.dof_vel_scale = float(self.cfg.get("dof_vel_scale", 1.0))
        self.action_scale = float(self.cfg.get("action_scale", 1.0))
        self.cmd_scale = np.asarray(self.cfg.get("cmd_scale", [1.0, 1.0, 1.0]), dtype=np.float32)
        self.max_cmd = np.asarray(self.cfg.get("max_cmd", [2.0, 1.0, 2.5]), dtype=np.float32)
        self.cmd = np.asarray(self.cfg.get("cmd_init", [0.5, 0.0, 0.0]), dtype=np.float32)

        self.use_lin_vel = bool(self.cfg.get("use_lin_vel", True))
        if bool(self.cfg.get("set_lin_zero", False)):
            self.use_lin_vel = False

        self.clip_observations = float(self.cfg.get("clip_observations", 100.0))
        self.clip_actions = float(self.cfg.get("clip_actions", 100.0))

        self.num_actions = int(self.cfg.get("num_actions", GO2_NUM_ACTIONS))
        self.num_obs = int(self.cfg.get("num_obs", GO2_NUM_OBS))
        if self.num_actions != GO2_NUM_ACTIONS:
            raise ValueError(f"Expected 12 actions for Go2, got {self.num_actions}")
        if self.num_obs != GO2_NUM_OBS:
            raise ValueError(f"Expected 279 observations, got {self.num_obs}")

        self.obs_components = list(self.cfg.get("obs_components", GO2_OBS_COMPONENTS))
        if self.obs_components != GO2_OBS_COMPONENTS:
            raise ValueError(f"obs_components must match IsaacGym order: {GO2_OBS_COMPONENTS}")

        self.measured_points_x = list(self.cfg.get("measured_points_x", GO2_MEASURED_POINTS_X))
        self.measured_points_y = list(self.cfg.get("measured_points_y", GO2_MEASURED_POINTS_Y))
        self.height_offset = float(self.cfg.get("height_measurements_offset", GO2_HEIGHT_MEASUREMENTS_OFFSET))
        self.height_scale = float(
            self.cfg.get("height_measurements_scale", self.cfg.get("height_measurements", 1.0))
        )
        self.ray_start_height = float(self.cfg.get("ray_start_height", 1.0))

        if len(self.measured_points_x) * len(self.measured_points_y) != GO2_NUM_HEIGHT_POINTS:
            raise ValueError(
                f"Height measurement grid must be 21x11=231, got {len(self.measured_points_x)}x{len(self.measured_points_y)}"
            )

        self.mj_model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.simulation_dt

        self.base_body_name = self.cfg.get("base_body_name", "base")
        self.base_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.base_body_name)
        if self.base_body_id < 0:
            raise ValueError(f"Base body '{self.base_body_name}' not found in {self.xml_path}")

        self.mujoco_joint_names = list(self.cfg.get("mujoco_joint_names", GO2_MUJOCO_JOINT_NAMES))
        self.model_joint_names = list(self.cfg.get("model_joint_names", self.mujoco_joint_names))
        self._build_joint_mappings()
        self._reset_robot_pose()

        self.height_sampler = HeightMeasurementSampler(
            model=self.mj_model,
            data=self.mj_data,
            base_body_id=self.base_body_id,
            points_x=self.measured_points_x,
            points_y=self.measured_points_y,
            height_offset=self.height_offset,
            height_scale=self.height_scale,
            ray_start_height=self.ray_start_height,
        )

        self.policy = torch.jit.load(str(self.policy_path), map_location="cpu")
        self.policy.eval()
        self._try_reset_policy_memory()

        self.last_action_model = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos_mj = self.default_angles_mj.copy()
        self.obs = np.zeros(self.num_obs, dtype=np.float32)

        self.save_video = save_video
        self.video_path = video_path
        self.renderer = None

        # Initialize depth camera rendering (offscreen)
        # Resolution: 120x160 (height x width)
        self.depth_camera_name = "depth_camera"
        self.depth_resolution = (160, 120)  # (width, height)
        self.depth_camera_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, self.depth_camera_name)
        if self.depth_camera_id != -1:
            print(f"[Info] Found depth camera: {self.depth_camera_name}, camera_id = {self.depth_camera_id}")
            # Create OpenGL context for offscreen rendering
            glfw.init()
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            self.depth_window = glfw.create_window(self.depth_resolution[0], self.depth_resolution[1], "DepthOffscreen", None, None)
            glfw.make_context_current(self.depth_window)
            # Create scene and context for depth rendering
            self.depth_scene = mujoco.MjvScene(self.mj_model, maxgeom=10000)
            self.depth_context = mujoco.MjrContext(self.mj_model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.depth_context)
            # Pre-allocate buffers
            self.depth_rgb = np.zeros((self.depth_resolution[1], self.depth_resolution[0], 3), dtype=np.uint8)
            self.depth_buffer = np.zeros((self.depth_resolution[1], self.depth_resolution[0], 1), dtype=np.float32)
            self.depth_viewport = mujoco.MjrRect(0, 0, self.depth_resolution[0], self.depth_resolution[1])
            # Setup fixed camera
            self.depth_mj_camera = mujoco.MjvCamera()
            self.depth_mj_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.depth_mj_camera.fixedcamid = self.depth_camera_id
        else:
            print(f"[Warn] Depth camera '{self.depth_camera_name}' not found in XML, depth capture disabled.")
            self.depth_camera_id = -1

    def _build_joint_mappings(self) -> None:
        if len(self.mujoco_joint_names) != self.num_actions:
            raise ValueError("mujoco_joint_names length must equal num_actions")
        if len(self.model_joint_names) != self.num_actions:
            raise ValueError("model_joint_names length must equal num_actions")
        if set(self.mujoco_joint_names) != set(self.model_joint_names):
            raise ValueError("mujoco_joint_names and model_joint_names must contain the same joints")

        self.idx_model2mj = np.asarray(
            [self.model_joint_names.index(name) for name in self.mujoco_joint_names],
            dtype=np.int64,
        )
        self.idx_mj2model = np.asarray(
            [self.mujoco_joint_names.index(name) for name in self.model_joint_names],
            dtype=np.int64,
        )

        self.joint_ids_mj = []
        self.joint_qposadr = []
        self.joint_dofadr = []
        for name in self.mujoco_joint_names:
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Joint '{name}' not found in MuJoCo model")
            self.joint_ids_mj.append(joint_id)
            self.joint_qposadr.append(int(self.mj_model.jnt_qposadr[joint_id]))
            self.joint_dofadr.append(int(self.mj_model.jnt_dofadr[joint_id]))

        self.joint_ids_mj = np.asarray(self.joint_ids_mj, dtype=np.int32)
        self.joint_qposadr = np.asarray(self.joint_qposadr, dtype=np.int32)
        self.joint_dofadr = np.asarray(self.joint_dofadr, dtype=np.int32)

        joint_id_to_actuator = {}
        for actuator_id in range(self.mj_model.nu):
            joint_id = int(self.mj_model.actuator_trnid[actuator_id, 0])
            joint_id_to_actuator[joint_id] = actuator_id

        self.actuator_ids = np.asarray(
            [joint_id_to_actuator[joint_id] for joint_id in self.joint_ids_mj],
            dtype=np.int32,
        )
        ctrl_range = self.mj_model.actuator_ctrlrange[self.actuator_ids]
        self.torque_min = ctrl_range[:, 0].astype(np.float32)
        self.torque_max = ctrl_range[:, 1].astype(np.float32)

    def _reset_robot_pose(self) -> None:
        self.mj_data.qvel[:] = 0.0
        self.mj_data.ctrl[:] = 0.0
        self.mj_data.qpos[self.joint_qposadr] = self.default_angles_mj
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _try_reset_policy_memory(self) -> None:
        reset_fn = getattr(self.policy, "reset_memory", None)
        if callable(reset_fn):
            try:
                reset_fn()
                print("[Info] JIT policy memory reset successfully.")
            except Exception as exc:
                print(f"[Warn] reset_memory() failed, continue anyway: {exc}")

    def _get_joint_state_mj(self):
           q = self.mj_data.qpos[self.joint_qposadr].copy().astype(np.float32)
           dq = self.mj_data.qvel[self.joint_dofadr].copy().astype(np.float32)
           return q, dq

    def _get_base_pos(self) -> np.ndarray:
           return self.mj_data.qpos[0:3].copy().astype(np.float32)

    def _get_base_quat(self) -> np.ndarray:
           return self.mj_data.qpos[3:7].copy().astype(np.float32)

    def _get_lin_vel(self) -> np.ndarray:
           base_quat = self._get_base_quat()
           base_lin_vel = quat_rotate_inverse(base_quat, self.mj_data.qvel[0:3])
           if not self.use_lin_vel:
               base_lin_vel[:] = 0.0
           return (base_lin_vel * self.lin_vel_scale).astype(np.float32)

    def _get_ang_vel(self) -> np.ndarray:
           base_quat = self._get_base_quat()
           base_ang_vel = quat_rotate_inverse(base_quat, self.mj_data.qvel[3:6])
           return (base_ang_vel * self.ang_vel_scale).astype(np.float32)

    def _get_projected_gravity(self) -> np.ndarray:
           base_quat = self._get_base_quat()
           projected_gravity = quat_rotate_inverse(
               base_quat,
               np.array([0.0, 0.0, -1.0], dtype=np.float32),
           )
           return projected_gravity.astype(np.float32)

    def _get_commands(self) -> np.ndarray:
           return (self.cmd * self.cmd_scale).astype(np.float32)

    def _get_dof_pos(self) -> np.ndarray:
           q_mj, _ = self._get_joint_state_mj()
           dof_pos = (q_mj - self.default_angles_mj) * self.dof_pos_scale
           return dof_pos[self.idx_mj2model].astype(np.float32)

    def _get_dof_vel(self) -> np.ndarray:
           _, dq_mj = self._get_joint_state_mj()
           dof_vel = dq_mj * self.dof_vel_scale
           return dof_vel[self.idx_mj2model].astype(np.float32)

    def _get_last_actions(self) -> np.ndarray:
           # 保持和 IsaacGym/legged_gym 语义一致：这里返回原始 action，不乘 action_scale
           return self.last_action_model.astype(np.float32)

    def _get_height_measurements(self) -> np.ndarray:
           base_pos = self._get_base_pos()
           base_quat = self._get_base_quat()
           return self.height_sampler.get_height_observation(base_pos, base_quat).astype(np.float32)

    def _visualize_height_points(self, viewer) -> None:
        """Visualize terrain height measurements around the robot with color coding:
        - Green: terrain higher than flat ground (z=0)
        - Gray: flat ground (z=0)
        - Red: terrain lower than flat ground (z=0)
        Filters out robot body collisions by checking ray intersection from above.
        """
        if not hasattr(viewer, 'user_scn'):
            return
        
        # Get height observation (normalized)
        height_obs = self._get_height_measurements()
        
        # Get base position for reference
        base_pos = self._get_base_pos()
        base_quat = self._get_base_quat()
        
        # Get world positions of sampling points
        yaw = quat_to_yaw(base_quat)
        c, s = math.cos(yaw), math.sin(yaw)
        yaw_rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        
        world_xy = self.height_sampler.local_points_xy @ yaw_rot.T
        world_xy[:, 0] += float(base_pos[0])
        world_xy[:, 1] += float(base_pos[1])
        
        # Calculate actual terrain heights from observation
        # height_obs = clip(base_z + offset - measured_height, -1, 1) * scale
        # So: measured_height = base_z + offset - (height_obs / scale)
        measured_heights = base_pos[2] + self.height_offset - (height_obs / self.height_scale)
        
        # Add visualization markers
        max_markers = min(len(measured_heights), 231)
        viewer.user_scn.ngeom = 0
        
        for i in range(max_markers):
            terrain_z = measured_heights[i]
            pos_xy = world_xy[i]
            
            # Filter out robot body: check if point is too close to robot parts
            # Skip if this point likely hit the robot body (not terrain)
            # Detect by checking if the point is within robot body bounding volume
            # Simple heuristic: if point is close to base z and within robot body radius
            dx = pos_xy[0] - base_pos[0]
            dy = pos_xy[1] - base_pos[1]
            dist_from_base = math.sqrt(dx*dx + dy*dy)
            
            # Skip points that are likely on robot body (close to base horizontally and vertically)
            # Robot body approx radius ~0.3m, height around base
            if dist_from_base < 0.35 and abs(terrain_z - base_pos[2]) < 0.2:
                continue
            
            # Compare to flat ground (z=0)
            height_diff = terrain_z
            
            # Determine color based on height relative to flat ground
            if height_diff > 0.05:  # Higher than flat ground
                rgba = [0.0, 0.8, 0.0, 0.8]  # Green
            elif height_diff < -0.05:  # Lower than flat ground
                rgba = [0.8, 0.0, 0.0, 0.8]  # Red
            else:  # Flat ground
                rgba = [0.5, 0.5, 0.5, 0.6]  # Gray
            
            pos = np.array([pos_xy[0], pos_xy[1], terrain_z], dtype=np.float64)
            
            idx = viewer.user_scn.ngeom
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[idx],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0, 0],
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=rgba,
            )
            viewer.user_scn.ngeom += 1

    def _get_obs_component(self, name: str) -> np.ndarray:
           getter = getattr(self, f"_get_{name}", None)
           if getter is None:
               raise AttributeError(f"Observation getter '_get_{name}' is not implemented.")
           obs = getter()
           obs = np.asarray(obs, dtype=np.float32).reshape(-1)
           return obs

    def get_observations(self) -> np.ndarray:
           obs_parts = []
           for name in self.obs_components:
               obs_parts.append(self._get_obs_component(name))
           obs = np.concatenate(obs_parts, axis=0).astype(np.float32)
           obs = np.clip(obs, -self.clip_observations, self.clip_observations)

           if obs.shape[0] != self.num_obs:
               raise RuntimeError(
                   f"Observation dimension mismatch: expected {self.num_obs}, got {obs.shape[0]}"
               )
           return obs

    def _run_policy(self) -> None:
           self.obs = self.get_observations()
           with torch.inference_mode():
               obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
               policy_output = self.policy(obs_tensor)
               action_tensor = extract_action_tensor(policy_output)
               action_model = action_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)

           if action_model.shape[0] != self.num_actions:
               raise RuntimeError(f"Action dimension mismatch: expected {self.num_actions}, got {action_model.shape[0]}")

           action_model = np.clip(action_model, -self.clip_actions, self.clip_actions)

           self.last_action_model = action_model
           action_mj = action_model[self.idx_model2mj]
           self.target_dof_pos_mj = self.default_angles_mj + action_mj * self.action_scale


    def _compute_torque(self) -> np.ndarray:
        q_mj, dq_mj = self._get_joint_state_mj()
        tau = pd_control(
            target_q=self.target_dof_pos_mj,
            q=q_mj,
            kp=self.kps,
            target_dq=np.zeros_like(dq_mj, dtype=np.float32),
            dq=dq_mj,
            kd=self.kds,
        ).astype(np.float32)
        tau = np.clip(tau, self.torque_min, self.torque_max)
        return tau

    def _configure_viewer(self, viewer) -> None:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = self.base_body_id
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 60.0

    def run(self, enable_joystick: bool = False, enable_keyboard: bool = False) -> None:
        joystick = try_init_joystick(enable_joystick)
        
        # Initialize keyboard control
        if enable_keyboard and keyboard is None:
            print("[Info] keyboard library not available, keyboard control disabled.")
            enable_keyboard = False

        writer = None
        frame_skip = 1
        if self.save_video:
            if imageio is None:
                raise RuntimeError("save_video=True but imageio is not available.")
            output_path = self.video_path
            if output_path is None:
                video_dir = LEGGED_GYM_ROOT_DIR / "deploy" / "deploy_mujoco" / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                output_path = video_dir / f"{self.policy_path.stem}_sim2sim_eval.mp4"

            self.renderer = mujoco.Renderer(self.mj_model, height=720, width=1280)
            sim_fps = 1.0 / self.mj_model.opt.timestep
            video_fps = 50
            frame_skip = max(1, int(sim_fps / video_fps))
            writer = imageio.get_writer(str(output_path), fps=video_fps)
            print(f"[Info] Recording video to: {output_path}")

        self._run_policy()
        last_status_time = 0.0

        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            self._configure_viewer(viewer)
            start_wall_time = time.time()
            sim_step = 0

            while viewer.is_running() and (time.time() - start_wall_time < self.simulation_duration):
                step_wall_time = time.time()

                if sim_step % self.control_decimation == 0:
                    # Handle input sources with priority: joystick > keyboard > fixed
                    if joystick is not None:
                        self.cmd = read_xbox_command(joystick, self.max_cmd)
                    elif enable_keyboard:
                        self.cmd = read_keyboard_command(self.cmd, self.max_cmd)
                    self._run_policy()

                    if step_wall_time - last_status_time > 0.1:
                        base_quat = self.mj_data.qpos[3:7].copy().astype(np.float32)
                        base_lin_vel = quat_rotate_inverse(base_quat, self.mj_data.qvel[0:3])
                        base_ang_vel = quat_rotate_inverse(base_quat, self.mj_data.qvel[3:6])
                        print(
                            f"\r[Eval] cmd=({self.cmd[0]: .2f}, {self.cmd[1]: .2f}, {self.cmd[2]: .2f}) "
                            f"vel=({base_lin_vel[0]: .2f}, {base_lin_vel[1]: .2f}, {base_ang_vel[2]: .2f})",
                            end="",
                            flush=True,
                        )
                        last_status_time = step_wall_time

                tau = self._compute_torque()
                self.mj_data.ctrl[:] = 0.0
                self.mj_data.ctrl[self.actuator_ids] = tau
                mujoco.mj_step(self.mj_model, self.mj_data)

                if writer is not None and sim_step % frame_skip == 0:
                    self.renderer.update_scene(self.mj_data, camera=viewer.cam)
                    frame = self.renderer.render()
                    writer.append_data(frame)

                # Visualize terrain height points
                self._visualize_height_points(viewer)
                
                # Capture depth image from camera if available
                if self.depth_camera_id != -1:
                    # Update scene from depth camera perspective
                    mujoco.mjv_updateScene(
                        self.mj_model, self.mj_data, mujoco.MjvOption(),
                        None, self.depth_mj_camera,
                        mujoco.mjtCatBit.mjCAT_ALL, self.depth_scene
                    )
                    # Render and read pixels
                    mujoco.mjr_render(self.depth_viewport, self.depth_scene, self.depth_context)
                    mujoco.mjr_readPixels(self.depth_rgb, self.depth_buffer, self.depth_viewport, self.depth_context)
                    
                    # Flip image vertically (OpenGL coordinates vs OpenCV)
                    bgr = cv2.cvtColor(np.flipud(self.depth_rgb), cv2.COLOR_RGB2BGR)
                    depth_image = np.flip(self.depth_buffer, axis=0).squeeze()
                    
                    # Normalize and display depth as grayscale
                    if np.max(depth_image) > np.min(depth_image):
                        depth_normalized = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))
                    else:
                        depth_normalized = np.zeros_like(depth_image)
                    depth_grayscale = np.uint8(depth_normalized * 255)
                    
                    # Display images
                    cv2.imshow('Depth Camera RGB', bgr)
                    cv2.imshow('Depth Camera Depth', depth_grayscale)
                    
                    # Check for ESC exit
                    if cv2.waitKey(1) == 27:
                        print("\n[Info] ESC pressed, exiting...")
                        break
                
                viewer.sync()

                sleep_time = self.mj_model.opt.timestep - (time.time() - step_wall_time)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

                sim_step += 1

        print()
        if writer is not None:
            writer.close()
            print("[Info] Video saved.")
        if pygame is not None and joystick is not None:
            pygame.quit()
        if keyboard is not None and enable_keyboard:
            keyboard.unhook_all()
        # Clean up depth camera resources
        if hasattr(self, 'depth_window'):
            cv2.destroyAllWindows()
            glfw.destroy_window(self.depth_window)
            glfw.terminate()
            del self.depth_context


def parse_args():
    parser = argparse.ArgumentParser(description="MuJoCo sim2sim JIT deployment evaluator for Go2.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH)+"g2.yaml", help="YAML config path.")
    parser.add_argument("--policy-path", type=str, default=None, help="Override policy_path from YAML.")
    parser.add_argument("--xml-path", type=str, default=None, help="Override xml_path from YAML.")
    parser.add_argument("--duration", type=float, default=None, help="Override simulation_duration from YAML.")
    parser.add_argument("--joystick", action="store_true", help="Enable optional joystick command input.")
    parser.add_argument("--keyboard", action="store_true", help="Enable keyboard command input.")
    parser.add_argument("--save-video", action="store_true", help="Record evaluation video.")
    parser.add_argument("--video-path", type=str, default=None, help="Optional output video path.")
    return parser.parse_args()


def main():
    args = parse_args()
    evaluator = MujocoSim2SimEvalJit(
        config_path=Path(os.path.join(DEFAULT_CONFIG_PATH, Path(args.config))),
        policy_path=Path(args.policy_path).resolve() if args.policy_path else None,
        xml_path=Path(args.xml_path).resolve() if args.xml_path else None,
        simulation_duration=args.duration,
        save_video=args.save_video,
        video_path=Path(args.video_path).resolve() if args.video_path else None,
    )
    evaluator.run(enable_joystick=args.joystick, enable_keyboard=args.keyboard)


if __name__ == "__main__":
    main()
