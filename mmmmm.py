import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path('/root/mym/parkour-main/legged_gym/resources/robots/go2/urdf/go2_mjcf copy.xml')
data = mujoco.MjData(model)

# 调整后的 PD：KD 必须加大，用来吸收 motor 产生的狂暴能量
KP = 45.0    
KD = 10.0    # 增加阻尼
MAX_TORQUE = 10.0 

def get_target_angles(t):
    # 频率降低，动作变慢，防止冲击限位
    wave = 0.3 * np.sin(2 * np.pi * 0.8 * t)
    
    # 严格匹配你的 XML Range:
    # Thigh: 前腿 [-1.5, 3.4], 后腿 [-0.5, 4.5] -> 取 0.8 很安全
    # Calf: 全部为 [-2.7, -0.8] -> 取 -1.8 很安全
    return {
        "FL": {"thigh": 0.8 + wave, "calf": -1.8 + wave},
        "FR": {"thigh": 0.8 - wave, "calf": -1.8 - wave},
        "RL": {"thigh": 0.8 + wave, "calf": -1.8 + wave},
        "RR": {"thigh": 0.8 - wave, "calf": -1.8 - wave},
    }

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        targets = get_target_angles(data.time)

        for i in range(model.nu):
            # 获取执行器控制的关节 ID
            joint_id = model.actuator_trnid[i, 0]
            # 获取该关节在 qpos 中的起始地址 (非常重要！)
            qpos_address = model.jnt_qposadr[joint_id]
            qvel_address = model.jnt_dofadr[joint_id]

            q_current = data.qpos[qpos_address]
            v_current = data.qvel[qvel_address]
            
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            prefix = actuator_name[:2] 
            
            # 设定目标值
            q_target = 0.0
            if "thigh" in actuator_name:
                q_target = targets.get(prefix, {}).get("thigh", 0.8)
            elif "calf" in actuator_name:
                q_target = targets.get(prefix, {}).get("calf", -1.8)
            
            # PD 计算
            torque = KP * (q_target - q_current) - KD * v_current
            torque = np.clip(torque, -MAX_TORQUE, MAX_TORQUE)
            
            data.ctrl[i] = torque

            # 调试：检查目标和当前的偏差
            if i == 0: # 只看第一个执行器，防止刷屏
                print(f"Name: {actuator_name}, Target: {q_target:.2f}, Cur: {q_current:.2f}, Torque: {torque:.2f}")

        mujoco.mj_step(model, data)
        viewer.sync()

        # 保持 500Hz 左右的控制频率
        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)