import numpy as np
import os.path as osp

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

ddog_action_scale = 0.5
ddog_const_dof_range = dict(
    Hip_max= 0.5,
    Hip_min= -0.5,
    Front_Thigh_max= 3.4,
    Front_Thigh_min= -1.85,
    Rear_Thigh_max= 3.4,
    Rear_Thigh_min= -1.85,
    Calf_max= -0.66,
    Calf_min= -2.757,
)

class DdogRoughCfg( LeggedRobotCfg ):
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

        history_len = 10 # add for cal smoothness

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
        resampling_time = 5 # [s]
        lin_cmd_cutoff = 0.2
        ang_cmd_cutoff = 0.2
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-1.0, 1.5 ]
            lin_vel_y = [-1., 1.]
            ang_vel_yaw = [-2., 2.]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0., 0., 0.38] # [m]
        default_joint_angles = { # 12 joints in the order of simulation
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.95,
            "FL_calf_joint": -1.8,
            "FR_hip_joint": -0.0,
            "FR_thigh_joint": 0.95,
            "FR_calf_joint": -1.8,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.95,
            "RL_calf_joint": -1.8,
            "RR_hip_joint": -0.0,
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
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/Ddog/urdf/Ddog.urdf'
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
            x = [-0.02, 0.02]
            y = [-0.01, 0.01]
            z = [-0.05, 0.05]

        randomize_motor = True
        leg_motor_strength_range = [0.8, 1.2]

        randomize_base_mass = True
        added_mass_range = [1.0, 1.5]

        randomize_friction = True
        friction_range = [0., 2.]

        # init_base_pos_range = {"x": [0.0, 0.0], "y": [0.0, 0.0]}
        # init_base_vel_range = [0.0, 0.0]
        init_base_pos_range = dict(
            x= [0.05, 0.6],
            y= [-0.25, 0.25],
        )
        #init_base_rot_range = {"roll": [0.0, 0.0], "pitch": [0.0, 0.0], "yaw": [0.0, 0.0]}
        init_base_rot_range = dict(
            roll= [-0.75, 0.75],
            pitch= [-0.75, 0.75],
        )
        init_base_vel_range = dict(
            x= [-0.2, 1.5],
            y= [-0.2, 0.2],
            z= [-0.2, 0.2],
            roll= [-1., 1.],
            pitch= [-1., 1.],
            yaw= [-1., 1.],
        )
        init_dof_vel_range = [-5, 5]

        push_robots = True 
        max_push_vel_xy = 0.5 # [m/s]
        push_interval_s = 2

    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            action_rate = -0.01                # 惩罚动作变化率，抑制动作突变
            action_smoothness = -0.01          # 惩罚动作平滑度，鼓励平滑运动
            ang_vel_xy = -0.1                  # 惩罚base在xy方向的角速度，抑制不必要的摇晃
            # base_height = -1e-3             # 惩罚base高度偏离目标值
            # collision = -1.0                # 惩罚身体部位碰撞地面
            dof_acc = -2.5e-7               # 惩罚关节加速度过大
            feet_air_time = -2.0               # 惩罚脚离地时间偏离目标值
            feet_air_time_var = -8.0           # 惩罚双脚离地时间方差，鼓励步态对称性
            # feet_contact_forces = -0.00015  # 惩罚脚接触力过大
            # foot_clearance = -0.5           # 惩罚脚离地高度不达标
            foot_mirror = -0.05                # 惩罚左右脚位置不对称，鼓励镜像步态
            foot_slide = -0.1              # 惩罚脚滑动
            has_contact = 1.0                  # 奖励脚接触地面，鼓励支撑
            hip_pos = -1.0                     # 惩罚髋关节偏离默认位置，鼓励保持原位
            lin_vel_z = -1.0                # 惩罚base在z方向速度，抑制上下弹跳
            # orientation = -0.2              # 惩罚base不水平，鼓励保持水平姿态
            powers = -2e-5                     # 惩罚功率/能量消耗，鼓励节能
            stand_still = -5.0                 # 惩罚零指令下的运动，鼓励静止
            # stumble = -0.05                 # 惩罚绊倒/失稳
            tracking_ang_vel = 1.0            # 奖励跟踪角速度指令误差越小越好
            tracking_lin_vel = 2.0            # 奖励跟踪线速度指令误差越小越好

            exceed_dof_pos_limits = -0.4       # 惩罚关节位置超出软限制
            exceed_torque_limits_l1norm = -0.4 # L1范数惩罚扭矩超出限制
            dof_vel_limits = -0.4              # 惩罚关节速度超出限制

            energy_substeps = -2e-5            # 惩罚子步骤能量消耗，多步积分版本
            dof_error_named = -1.              # 惩罚特定关节位置误差（加权更大）
            dof_error = -0.01                  # 惩罚所有关节位置误差
            
        dof_error_names = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"]
        only_positive_rewards = False
        soft_dof_vel_limit = 0.9
        soft_dof_pos_limit = 0.9
        soft_torque_limit = 0.9

        feet_air_time_target = 0.30
        clearance_height_target = -0.18
        base_height_target = 0.32

    class normalization( LeggedRobotCfg.normalization ):
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            lin_vel = 1.
        height_measurements_offset = -0.2
        clip_actions_method = None # let the policy learns not to exceed the limits

    class noise( LeggedRobotCfg.noise ):
        add_noise = True

    class viewer( LeggedRobotCfg.viewer ):
        pos = [-1., -1., 0.4]
        lookat = [0., 0., 0.3]

    class sim( LeggedRobotCfg.sim ):
        body_measure_points = { # transform are related to body frame
            "base": dict(
                x= [i for i in np.arange(-0.14, 0.141, 0.03)],
                y= [-0.1, -0.05, 0.0, 0.05, 0.1],
                z= [i for i in np.arange(-0.05, 0.051, 0.03)],
                transform= [0., 0., 0., 0., 0., 0.],
            ),
            "thigh": dict(
                x= [
                    -0.18, -0.178, -0.175, -0.17, -0.16,
                    -0.15, -0.14, -0.12, -0.1, -0.08, -0.06 , -0.04, -0.02,
                    0.0, 0.02, 0.04 ,
                ],
                y= [-0.02, 0.0, 0.02 ],
                z= [-0.02, 0.0, 0.02 ],
                transform= [0., 0., -0.1,   0., 1.57079632679, 0.],
            ),
            "calf": dict(
                x= [i for i in np.arange(-0.15, 0.05, 0.03)],
                y= [-0.015, 0.0, 0.015],
                z= [-0.015, 0.0, 0.015],
                transform= [0., 0., -0.11,   0., 1.57079632679, 0.],
            ),
        }

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class DdogRoughCfgPPO( LeggedRobotCfgPPO ):
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
        load_run = "/root/mym/pk_zrg/parkour/legged_gym/logs/rough_ddog/first"

        run_name = "".join(["DdogRough"])
           
        max_iterations = 10000
        save_interval = 500
        log_interval = 10
