import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import time

# ==========================================
# 1. 神经网络结构 (保持与之前对齐)
# ==========================================
class ParkourPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.Sequential(
            nn.Linear(231, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 32)
        )
        self.memory_a = nn.GRU(input_size=80, hidden_size=256, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(256, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 12)
        )
        self.rnn_hidden = torch.zeros(1, 1, 256)

    def forward(self, obs, rnn_state):
        # 根据你的 obs 排列：前 48 位是本体感受，后 231 位是高度测量
        proprio = obs[:, :48] 
        heights = obs[:, 48:]
        
        encoded = self.encoders(heights)
        rnn_input = torch.cat([proprio, encoded], dim=-1).unsqueeze(1)
        
        output, rnn_state = self.memory_a(rnn_input, rnn_state)
        actions = self.actor(output.squeeze(1))
        return actions, rnn_state

# ==========================================
# 2. 仿真主类
# ==========================================
class Go2ParkourSim:
    def __init__(self, xml_path, model_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 控制参数
        self.kp = 80.0
        self.kd = 1.0
        self.action_scale = 0.5
        self.torque_limits = 23.7
        self.last_action = torch.zeros(12)
        self.commands = torch.tensor([1.0, 0.0, 0.0]) # 默认指令：向前走 1m/s
        
        # 默认关节角度 (对齐 Go2 站立姿态)
        self.default_dof_pos = torch.tensor([0.1, 0.7, -1.5, -0.1, 0.7, -1.5, 
                                            0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
        
        # 模型加载
        self.policy = self._load_policy(model_path)
        self.scales = {
            "lin_vel": 1.0,           # 覆盖后的值为 1.0
            "ang_vel": 0.25,
            "commands": np.array([2.0, 2.0, 0.25]),
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "height_measurements": 5.0,
            "height_offset": -0.2
        }
        self.clip_obs = 100.

    def _load_policy(self, path):
        policy = ParkourPolicy()
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt['model_state_dict']
        new_dict = {}
        for k, v in state_dict.items():
            name = k.replace("encoders.0.model.", "encoders.").replace("memory_a.rnn.", "memory_a.")
            if name in policy.state_dict(): new_dict[name] = v
        policy.load_state_dict(new_dict, strict=False)
        return policy

    def _get_obs(self):
        """ 获取并缩放观测值 """
        
        # 1. 坐标变换 (World -> Body)
        quat = self.data.qpos[3:7]
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, quat)
        rot_mat = rot_mat.reshape(3, 3)
        
        # 2. 提取原始数据 (Raw Data)
        raw_lin_vel = rot_mat.T @ self.data.qvel[:3]
        raw_ang_vel = self.data.qvel[3:6]
        raw_gravity = rot_mat.T @ np.array([0, 0, -1])
        raw_dof_pos = self.data.qpos[7:] - self.default_dof_pos.numpy()
        raw_dof_vel = self.data.qvel[6:]
        
        # 3. 模拟高度测量 (高度图也需要 Offset 和 Scale)
        # heights_scaled = (raw_heights + offset) * scale
        raw_heights = np.zeros(231) # 暂假设平地
        scaled_heights = (raw_heights + self.scales["height_offset"]) * self.scales["height_measurements"]
        
        # 4. 应用缩放 (Apply Obs Scales)
        obs_components = [
            raw_lin_vel * self.scales["lin_vel"],          # 3
            raw_ang_vel * self.scales["ang_vel"],          # 3
            raw_gravity,                                    # 3 (通常重力不缩放)
            self.commands.numpy() * self.scales["commands"], # 3
            raw_dof_pos * self.scales["dof_pos"],           # 12
            raw_dof_vel * self.scales["dof_vel"],           # 12
            self.last_action.numpy(),                       # 12 (Action 通常已经是 -1~1)
            scaled_heights                                  # 231
        ]
        
        # 5. 拼接、截断并转换为 Tensor
        obs_array = np.concatenate(obs_components)
        obs_array = np.clip(obs_array, -self.clip_obs, self.clip_obs).round(4)
        
        return torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0)

    def step(self):
        # 获取缩放后的 Obs
        obs = self._get_obs()
        print(f"Obs: {obs}")
        
        # 推理获取 Action (通常在 -1 到 1 之间)
        with torch.no_grad():
            action, self.policy.rnn_hidden = self.policy(obs, self.policy.rnn_hidden)
        
        torch.clamp(action, -1, 1)
        # 注意：这里可能需要对 action 进行处理
        # 你的配置提到 clip_actions_method = None，说明直接使用推理结果
        current_action = action[0]

        # PD 控制：Target = Action * Scale + Default
        q_now = torch.tensor(self.data.qpos[7:])
        dq_now = torch.tensor(self.data.qvel[6:])
        
        target_q = current_action * self.action_scale + self.default_dof_pos
        tau = self.kp * (target_q - q_now) - self.kd * dq_now
        tau = torch.clip(tau, -self.torque_limits, self.torque_limits)
        print(f"Action: {current_action}, Target: {target_q}, Current: {q_now}")
        print(f"Torques: {tau}")
        
        self.data.ctrl[:] = tau.numpy()
        self.last_action = current_action # 存入供下一步 Obs 使用
        
        mujoco.mj_step(self.model, self.data)

# ==========================================
# 3. 运行
# ==========================================
if __name__ == "__main__":
    sim = Go2ParkourSim("/root/mym/parkour-main/legged_gym/resources/robots/go2/urdf/go2_mjcf copy.xml", "/root/mym/parkour-main/legged_gym/logs/field_go2/Feb04_02-53-08_Go2_10skills_pEnergy2.e-07_pTorques-1.e-07_pLazyStop-3.e+00_pPenD5.e-02_penEasier200_penHarder100_leapHeight2.e-01_motorTorqueClip_fromFeb03_07-44-57/model_37500.pt")
    
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            sim.step()
            viewer.sync()
            time.sleep(sim.model.opt.timestep)
            print(sim.model.opt.timestep)