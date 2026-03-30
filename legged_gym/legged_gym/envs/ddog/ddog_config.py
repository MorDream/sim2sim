""" Basic model configs for Unitree Ddog """
import numpy as np
import os.path as osp

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

ddog_action_scale = 0.5
ddog_const_dof_range = dict(
    Hip_max= 1.0472,
    Hip_min= -1.0472,
    Front_Thigh_max= 3.4907,
    Front_Thigh_min= -1.5708,
    Rear_Thingh_max= 4.5379,
    Rear_Thingh_min= -0.5236,
    Calf_max= -0.83776,
    Calf_min= -2.7227,
)

class Ddog2RoughCfg( LeggedRobotCfg ):
    class env:
        num_envs = 4096
        num_observations = None # No use, use obs_components
        num_privileged_obs = None # No use, use privileged_obs_components

        use_lin_vel = False # to be decided
        num_actions = 12
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

        obs_components = [
            "lin_vel",#线速度3
            "ang_vel",#角速度3
            "projected_gravity",#重力向量投影3
            "commands",#指令3
            "dof_pos",#关节角度12
            "dof_vel",#关节速度12
            "last_actions",#上一动作12
            "height_measurements",#高度测量21*11
        ]
        history_len = 10

    class sensor:
        class proprioception:
            obs_components = ["ang_vel", "projected_gravity", "commands", "dof_pos", "dof_vel"]
            latency_range = [0.005, 0.045] # [s]
            latency_resampling_time = 5.0 # [s]

    class terrain:
        selected = "TerrainPerlin"
        mesh_type = None
        measure_heights = True
        # x: [-0.5, 1.5], y: [-0.5, 0.5] range for ddog
        measured_points_x = [i for i in np.arange(-0.5, 1.51, 0.1)]
        measured_points_y = [i for i in np.arange(-0.5, 0.51, 0.1)]
        horizontal_scale = 0.025 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 4.
        terrain_width = 4.
        num_rows= 16 # number of terrain rows (levels)
        num_cols = 16 # number of terrain cols (types)
        slope_treshold = 1.

        TerrainPerlin_kwargs = dict(
            zScale= 0.07,
            frequency= 10,
        )

    class commands( LeggedRobotCfg.commands ):
        heading_command = False
        resampling_time = 10 # [s]
        lin_cmd_cutoff = 0.2
        ang_cmd_cutoff = 0.2
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-1.0, 3.0]
            lin_vel_y = [-1., 1.]
            ang_vel_yaw = [-2., 2.]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0., 0., 0.5] # [m]
        default_joint_angles = { # 12 joints in the order of simulation
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.95,
            "FL_calf_joint": -1.8,
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": 0.95,
            "FR_calf_joint": -1.8,
            "RL_hip_joint": 0.1,
            "RL_thigh_joint": 0.95,
            "RL_calf_joint": -1.8,
            "RR_hip_joint": -0.1,
            "RR_thigh_joint": 0.95,
            "RR_calf_joint": -1.8,
        }

    class control( LeggedRobotCfg.control ):
        stiffness = {'joint': 30.}
        damping = {'joint': 1.}
        action_scale = ddog_action_scale
        computer_clip_torque = False
        motor_clip_torque = True

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/Littledog/urdf/Ddog.urdf'
        name = "Dgod"
        foot_name = "foot"
        front_hip_names = ["FL_hip_joint", "FR_hip_joint"]
        rear_hip_names = ["RL_hip_joint", "RR_hip_joint"]
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        sdk_dof_range = ddog_const_dof_range
        dof_velocity_override = 35.
        flip_visual_attachments = False

    class termination:
        termination_terms = [
            "roll",
            "pitch",
        ]

        roll_kwargs = dict(
            threshold= 3.0, # [rad]
        )
        pitch_kwargs = dict(
            threshold= 3.0, # [rad] # for leap, jump
        )

    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_com = True
        class com_range:
            x = [-0.2, 0.2]
            y = [-0.1, 0.1]
            z = [-0.05, 0.05]

        randomize_motor = True
        leg_motor_strength_range = [0.8, 1.2]

        randomize_base_mass = True
        added_mass_range = [1.0, 3.0]

        randomize_friction = True
        friction_range = [0., 2.]

        init_base_pos_range = {"x": [0.0, 0.0], "y": [0.0, 0.0]}
        init_base_vel_range = [0.0, 0.0]
        init_base_rot_range = {"roll": [0.0, 0.0], "pitch": [0.0, 0.0], "yaw": [0.0, 0.0]}
        # init_base_rot_range = dict(
        #     roll= [-0.75, 0.75],
        #     pitch= [-0.75, 0.75],
        # )
        # init_base_vel_range = dict(
        #     x= [-0.2, 1.5],
        #     y= [-0.2, 0.2],
        #     z= [-0.2, 0.2],
        #     roll= [-1., 1.],
        #     pitch= [-1., 1.],
        #     yaw= [-1., 1.],
        # )
        init_dof_vel_range = [-5, 5]

        push_robots = True
        max_push_vel_xy = 0.5 # [m/s]
        push_interval_s = 2

    class rewards( LeggedRobotCfg.rewards ):
        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 2.          # 线速度跟踪奖励：鼓励机器人跟随命令的线速度
            tracking_ang_vel = 1.          # 角速度跟踪奖励：鼓励机器人跟随命令的偏航角速度
            lin_vel_z = -4                 # Z轴线速度惩罚：惩罚基座Z方向速度，鼓励保持水平运动
            ang_vel_xy =  -0.1             # XY轴角速度惩罚：惩罚翻滚和俯仰，鼓励稳定姿态
            orientation = -0.5             # 姿态惩罚：惩罚非水平基座，鼓励保持平衡
            torques = -0.00001             # 扭矩惩罚：惩罚过大扭矩，鼓励节能
            dof_vel = -0.                  # 关节速度惩罚：当前未使用，系数为0
            dof_acc = -2.5e-7              # 关节加速度惩罚：惩罚大加速度，鼓励平滑运动
            base_height = -1.0             # 基座高度惩罚：惩罚偏离目标高度，保持稳定行走高度
            feet_air_time = 1.0            # 脚悬空时间奖励：鼓励脚在空中停留更长时间，走出更大步幅
            collision = -1                 # 碰撞惩罚：惩罚身体非足部部位碰撞地面
            stumble = -0.05                # 绊倒惩罚：惩罚脚撞击垂直表面
            stand_still = -2.0             # 原地站立惩罚：零命令时惩罚关节偏离默认位置，鼓励站稳
            action_rate = -0.1             # 动作变化率惩罚：惩罚前后步动作突变，鼓励平滑输出
            action_smoothness = -0.1       # 动作平滑性惩罚：惩罚二阶差分，鼓励轨迹平滑

            feet_contact_forces = -0.00015 # 接触力惩罚：惩罚过大足端接触力，避免硬冲击
            foot_clearance = -0.5          # 脚间隙惩罚：惩罚脚抬高度偏离目标，鼓励适当抬脚
            foot_mirror = -0.1            # 脚步镜像惩罚：惩罚左右腿不对称，鼓励对称步态
            foot_slide = -0.05             # 脚滑动惩罚：惩罚接触脚水平滑动，鼓励脚踏实地
            has_contact = 1.0              # 接触奖励：零命令时奖励更多脚着地，鼓励稳定站立
            hip_pos = -5.0                 # 髋部位置惩罚：惩罚髋部偏离零位置，鼓励髋部保持中立
            powers = -2e-5                 # 功率惩罚：惩罚总能量消耗，功率=|扭矩|×|角速度|
            exceed_dof_pos_limits = -0.4   # 关节位置超限惩罚：统计超过位置限制的关节数量
            exceed_torque_limits_l1norm = -0.4 # 超扭矩L1惩罚：对超过软极限的部分累加惩罚
            dof_vel_limits = -0.4          # 关节速度超限惩罚：惩罚关节速度超过软极限
        
        dof_error_names = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"]
        only_positive_rewards = False
        soft_dof_vel_limit = 0.9
        soft_dof_pos_limit = 0.9
        soft_torque_limit = 0.9
        clearance_height_target = -0.22
        base_height_target = 0.32

    class normalization( LeggedRobotCfg.normalization ):
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            lin_vel = 1.
        height_measurements_offset = -0.2
        clip_actions_method = None # let the policy learns not to exceed the limits

    class noise( LeggedRobotCfg.noise ):
        add_noise = False

    class viewer( LeggedRobotCfg.viewer ):
        pos = [-1., -1., 0.4]
        lookat = [0., 0., 0.3]

    class sim( LeggedRobotCfg.sim ):
        body_measure_points = { # transform are related to body frame
            "base": dict(
                x= [i for i in np.arange(-0.24, 0.41, 0.03)],
                y= [-0.08, -0.04, 0.0, 0.04, 0.08],
                z= [i for i in np.arange(-0.061, 0.071, 0.03)],
                transform= [0., 0., 0.005, 0., 0., 0.],
            ),
            "thigh": dict(
                x= [
                    -0.16, -0.158, -0.156, -0.154, -0.152,
                    -0.15, -0.145, -0.14, -0.135, -0.13, -0.125, -0.12, -0.115, -0.11, -0.105, -0.1, -0.095, -0.09, -0.085, -0.08, -0.075, -0.07, -0.065, -0.05,
                    0.0, 0.05, 0.1,
                ],
                y= [-0.015, -0.01, 0.0, -0.01, 0.015],
                z= [-0.03, -0.015, 0.0, 0.015],
                transform= [0., 0., -0.1,   0., 1.57079632679, 0.],
            ),
            "calf": dict(
                x= [i for i in np.arange(-0.13, 0.111, 0.03)],
                y= [-0.015, 0.0, 0.015],
                z= [-0.015, 0.0, 0.015],
                transform= [0., 0., -0.11,   0., 1.57079632679, 0.],
            ),
        }

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Ddog2RoughPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        clip_min_std = 0.2
        learning_rate = 1e-4
        optimizer_class_name = "AdamW"

    class policy( LeggedRobotCfgPPO.policy ):
        # configs for estimator module
        estimator_obs_components = [
            "ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos",
            "dof_vel",
            "last_actions",
        ]
        estimator_target_components = ["lin_vel"]
        replace_state_prob = 1.0
        class estimator_kwargs:
            hidden_sizes = [128, 64]
            nonlinearity = "CELU"
        # configs for (critic) encoder
        encoder_component_names = ["height_measurements"]
        encoder_class_name = "MlpModel"
        class encoder_kwargs:
            hidden_sizes = [128, 64]
            nonlinearity = "CELU"
        encoder_output_size = 32
        critic_encoder_component_names = ["height_measurements"]
        init_noise_std = 0.5
        # configs for policy: using recurrent policy with GRU
        rnn_type = 'gru'
        mu_activation = None

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "EncoderStateAcRecurrent"
        algorithm_class_name = "EstimatorPPO"
        experiment_name = "rough_ddog"

        resume = True
        load_run = "/root/mym/parkour-main/legged_gym/logs/rough_ddog/Mar27_05-51-00_DdogRough"

        run_name = "".join(["DdogRough"])

        max_iterations = 10000
        save_interval = 200
        log_interval = 100
