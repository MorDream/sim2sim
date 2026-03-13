
import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import os

# 假设 LEGGED_GYM_ROOT_DIR 已定义，或者手动指定
LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MujocoTerrainSensor:
    def __init__(self, model, data, num_points_x=21, num_points_y=11, spacing=0.1):
        """
        参数:
            model: mujoco.MjModel
            data: mujoco.MjData
            num_points_x/y: 采样点阵规模 (11x11=121个点)
            spacing: 点与点之间的间距 (米)
        """
        self.model = model
        self.data = data

        # 1. 预定义相对坐标网格 (以机器人为中心的局部坐标)
        x = np.linspace(-(num_points_x-1)*spacing/2, (num_points_x-1)*spacing/2, num_points_x)
        y = np.linspace(-(num_points_y-1)*spacing/2, (num_points_y-1)*spacing/2, num_points_y)
        xv, yv = np.meshgrid(x, y)

        # 存储为 (N, 3) 矢量，初始 Z 设为 0
        self.points_local = np.stack([xv.flatten(), yv.flatten(), np.zeros_like(xv.flatten())], axis=-1)
        self.num_points = self.points_local.shape[0]

        # 2. 设置碰撞过滤掩码 (Geom Group Mask)
        # 掩码数组长度为 5，对应 MuJoCo 的 5 个 geom groups
        # [Group0, Group1, Group2, Group3, Group4]
        # 我们只开启 Group 0 (地形)，关闭 Group 1 (机器人)
        self.terrain_mask = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)

        # 3. 归一化偏移量 (对应 legged_gym 的 height_measurements_offset)
        self.height_offset = -0.2
        self.x_range = x
        self.y_range = y
        self.points_local = np.stack([xv.flatten(), yv.flatten(), np.zeros_like(xv.flatten())], axis=-1)

    def get_heights(self, robot_pos, robot_quat):
        """
        获取机器人周围地形的绝对物理高度
        """
        # 将四元数转换为偏航角 (Yaw)，确保采样点随机器人转动
        # 这里简化处理，只提取偏航角
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat([robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]]) # MuJoCo是[w,x,y,z]
        yaw = r.as_euler('zyx')[0]

        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        R_yaw = np.array([[cos_y, -sin_y, 0],
                          [sin_y,  cos_y, 0],
                          [0,      0,     1]])

        # 将局部采样点转换到世界坐标系
        # 射线起点设在机器人位置上方 1.0m，确保能覆盖到高处地形
        points_world = (R_yaw @ self.points_local.T).T + robot_pos + np.array([0, 0, 1.0])

        heights = np.zeros(self.num_points)
        direction = np.array([0, 0, -1.0]) # 垂直向下发射
        geomid_out = np.zeros(1, dtype=np.int32)
        # 核心：使用 mj_ray 进行射线探测
        for i in range(self.num_points):
            # geomgroup=self.terrain_mask 确保射线忽略机器人本体（Group 1）
            dist = mujoco.mj_ray(self.model, self.data, points_world[i], direction,
                                 geomgroup=self.terrain_mask,
                                 flg_static=1, # 仅检测静态几何体（可选）
                                 bodyexclude=-1,
                                 geomid=geomid_out)

            if dist > 0:
                # 实际高度 = 射线起点 Z - 飞行距离
                heights[i] = points_world[i][2] - dist
            else:
                # 没打中地形（可能出界或坑极深）
                heights[i] = -10.0

        return heights

    def get_observation(self):
        """
        生成最终输入给神经网络的观测向量 (Obs)
        """
        # 获取机器人当前状态
        # 假设机器人 root body 名为 "base"
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        robot_pos = self.data.xpos[base_id]
        robot_quat = self.data.xquat[base_id]

        # 获取物理高度
        measured_heights = self.get_heights(robot_pos, robot_quat)

        # 转化为相对高度并归一化 (核心逻辑)
        # Obs = Clip( Base_Z + Offset - Measured_Heights, -1, 1 )
        heights_obs = robot_pos[2] + self.height_offset - measured_heights
        heights_obs = np.clip(heights_obs, -1.0, 1.0)

        return torch.from_numpy(heights_obs).float()

class ObservationSensor:
    def __init__(self, model, data, config, actions):
        self.model = model
        self.data = data
        self.terrain_sensor = MujocoTerrainSensor(model, data)
        self.config = config
        self.quat = d.qpos[3:7].copy()
        self.inv_quat = np.zeros(4)
        mujoco.mju_negQuat(self.inv_quat, self.quat)
        self.default_angles = np.array(self.config["default_angles"], dtype=np.float32)
        self.actions = actions
        self.obs_list = config["obs_list"]

    def _get_height_measurements(self):
        heights = self.terrain_sensor.get_observation()
        height = torch.tensor(heights, dtype=torch.float32).clip(-1, 1)
        return height
    def _get_lin_vel(self):
        local_lin_vel = np.zeros(3)
        mujoco.mju_rotVecQuat(local_lin_vel, self.data.qvel[:3], self.inv_quat)
        if self.config["set_lin_zero"]:
            local_lin_vel = np.zeros(3)
        return local_lin_vel * self.config.get("lin_vel_scale", 2.0)
    def _get_ang_vel(self):
        return self.data.qvel[3:6] * self.config.get("ang_vel_scale", 0.25)

    def _get_projected_gravity(self):
        world_gravity = np.array([0, 0, -1], dtype=np.float64)
        local_gravity = np.zeros(3)
        mujoco.mju_rotVecQuat(local_gravity, world_gravity, self.inv_quat)
        return local_gravity * self.config.get("gravity_scale", 1.0)

    def _get_commands(self):
        return np.array(self.config["cmd_init"]) * np.array(self.config["cmd_scale"])

    def _get_dof_pos(self):
        return (self.data.qpos[7:] - self.default_angles) * self.config.get("dof_pos_scale", 1.0)

    def _get_dof_vel(self):
        return self.data.qvel[6:] * self.config.get("dof_vel_scale", 0.05)

    def _get_last_actions(self):
        return self.actions * self.config.get("action_scale", 1.0)

    def get_observations(self):
        obs = []
        for ob in self.obs_list:
            obs.append(getattr(self, "_get_" + ob)())
        obs = np.concatenate(obs)
        return obs


# --- 2. 部署主程序 ---

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    # 加载配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "configs", args.config_file)
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    # 初始化 MuJoCo
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = config["simulation_dt"]

    # 加载模型
    device = "cuda"
    policy = torch.jit.load(policy_path).to(device)
    policy.eval()

    # 地形采样点
    measured_points_x = np.linspace(-0.5, 1.5, 21)
    measured_points_y = np.linspace(-0.5, 0.5, 11)

    # 控制参数
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)

    obs = np.zeros(279, dtype=np.float32)
    action = np.zeros(12, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    counter = 0

    # 启动查看器
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # 初始稳定：让狗子落到地上
        for _ in range(100):
            mujoco.mj_step(m, d)
            viewer.sync()

        # --- 1. 初始化阶段 (在 while 循环外) ---
        obs_sensor = ObservationSensor(m, d, config, action)
        # 预先提取一些不变的常量
        points_local = obs_sensor.terrain_sensor.points_local
        num_points = len(points_local)
        # 初始化一个全零的 obs 向量
        obs = np.zeros(279, dtype=np.float32)

        # 为了视觉平滑，我们记录上一次探测到的世界坐标点
        cached_world_points = np.zeros((num_points, 3))
        cached_colors = np.zeros((num_points, 4))

        while viewer.is_running():
            step_start = time.time()

            # --- 2. 物理步 (必须保证高频) ---
            tau = (target_dof_pos - d.qpos[7:]) * kps - d.qvel[6:] * kds
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            # --- 3. 策略/传感器更新 (分频执行，例如 50Hz) ---
            counter += 1
            if counter % config["control_decimation"] == 0:
                # 核心优化：不再重新创建对象，只更新内部数据
                obs_sensor.actions = action
                obs = obs_sensor.get_observations()

                # 推理
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    action = policy(obs_tensor).cpu().squeeze().numpy()

                # 处理 action 和 target_dof_pos...
                action = np.clip(action, -5.0, 5.0)
                target_dof_pos = action * config.get("action_scale", 0.25) + default_angles

                # --- 4. 可视化坐标预计算 (只在采样时计算) ---
                # 这样就不需要每物理帧都去算一遍射线和旋转了
                base_pos = d.qpos[0:3]
                base_quat = d.qpos[3:7]
                from scipy.spatial.transform import Rotation as R
                r = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
                yaw = r.as_euler('zyx')[0]
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)
                R_yaw = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])

                for c in range(num_points):
                    h_obs = obs[48 + c]
                    world_sample_xy = R_yaw @ points_local[c] + base_pos
                    z_ground = base_pos[2] - 0.2 - h_obs
                    cached_world_points[c] = [world_sample_xy[0], world_sample_xy[1], z_ground]

                    # 颜色预存
                    if h_obs < -0.05: cached_colors[c] = [0, 1, 0, 1]
                    elif h_obs > 0.05: cached_colors[c] = [1, 0, 0, 1]
                    else: cached_colors[c] = [0.8, 0.8, 0.8, 0.6]

            # --- 5. 渲染输出 (抽样渲染，减少绘图调用) ---
            # 没必要每个物理步都同步 viewer，渲染频率能到 60FPS 即可
            if counter % 10 == 0: # 假设物理是 1000Hz，这里就是 100Hz 刷新率
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    for c in range(num_points):
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=[0.012, 0, 0],
                            pos=cached_world_points[c],
                            mat=np.eye(3).flatten(),
                            rgba=cached_colors[c]
                        )
                        viewer.user_scn.ngeom += 1
                viewer.sync()

            # 帧率同步
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
