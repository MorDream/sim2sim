import torch
import torch.nn as nn

class MUJOCO_NET(nn.Module):
    def __init__(self):
        super(MUJOCO_NET, self).__init__()
        
        # 1. 视觉编码器 (Height Scan Encoder)
        # 输入 231 -> 输出 32
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(231, 128), nn.ELU(),
                nn.Linear(128, 64),  nn.ELU(),
                nn.Linear(64, 32),   nn.ELU()
            )
        ])
        
        # 2. 策略记忆单元 (Actor RNN)
        # 输入 80 (48本体 + 32视觉) -> 隐藏层 256
        self.memory_a = nn.ModuleDict({
            "rnn": nn.GRU(input_size=80, hidden_size=256, batch_first=True)
        })
        
        # 3. 策略网络 (Actor MLP)
        self.actor = nn.Sequential(
            nn.Linear(256, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 12)
        )
        
        # 4. 动作标准差 (Std)
        self.std = nn.Parameter(torch.zeros(12))

        # --- 以下是 State Estimator 部分 (可选，但为了完整加载权重需保留) ---
        self.memory_s = nn.ModuleDict({
            "rnn": nn.GRU(input_size=45, hidden_size=256, batch_first=True)
        })
        self.state_estimator = nn.Sequential(
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64),  nn.ELU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        """
        x: [Batch, 279]
        前 48 维: Proprioception
        后 231 维: Height Measurements
        """
        # 数据切分
        proprio = x[:, :48]    # [Batch, 48]
        scan = x[:, 48:]       # [Batch, 231]
        
        # 1. 编码扫描点
        visual_latent = self.encoders[0](scan) # [Batch, 32]
        
        # 2. 拼接
        rnn_input = torch.cat([proprio, visual_latent], dim=-1) # [Batch, 80]
        
        # 3. 经过 RNN (增加序列维度)
        rnn_input = rnn_input.unsqueeze(1) 
        rnn_out, _ = self.memory_a["rnn"](rnn_input)
        rnn_out = rnn_out.squeeze(1) # [Batch, 256]
        
        # 4. 输出动作
        actions = self.actor(rnn_out)
        return actions

# --- 辅助函数：手动转换 Key 以适配 ModuleList/ModuleDict ---
def load_compat_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # 修正 state_dict 中的键名以匹配当前模型
    # 例如：将 'encoders.0.model.0.weight' 映射到 'encoders.0.0.weight'
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('.model.', '.') # 移除中间的 .model.
        new_state_dict[new_key] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    print("✅ 权重加载完成！")
    return checkpoint.get('iter', 'N/A')

# 1. 实例化新网络
mujoco_net = MUJOCO_NET()

# 2. 指定权重路径 (你提供的路径)
ckpt_path = r'/root/mym/parkour-main/legged_gym/logs/field_go2/Feb04_02-53-08_Go2_10skills_pEnergy2.e-07_pTorques-1.e-07_pLazyStop-3.e+00_pPenD5.e-02_penEasier200_penHarder100_leapHeight2.e-01_motorTorqueClip_fromFeb03_07-44-57/model_30000.pt'

# 3. 加载权重
iteration = load_compat_weights(mujoco_net, ckpt_path)

# 4. 初始化全零测试输入 (Batch=1, Dim=279)
test_input = torch.zeros(1, 279)

# 5. 推理
mujoco_net.eval()
with torch.no_grad():
    output_actions = mujoco_net(test_input)

# --- 结果展示 ---
print("\n" + "-"*50)
print(f"模型迭代步数: {iteration}")
print(f"输入形状: {test_input.shape}")
print(f"输出 (12关节动作值):")
print(output_actions[0].numpy())
print("-"*50)

# 如果想看 std (学习到的动作探索噪声)
print(f"当前动作 Std (动作置信度): \n{torch.exp(mujoco_net.std).detach().numpy()}")