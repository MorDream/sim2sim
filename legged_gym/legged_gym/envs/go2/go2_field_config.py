""" Config to train the whole parkour oracle policy """
import numpy as np
from os import path as osp
from collections import OrderedDict

from legged_gym.envs.go2.go2_config import Go2RoughCfg, Go2RoughCfgPPO

class Go2FieldCfg( Go2RoughCfg ):
    class init_state( Go2RoughCfg.init_state ):
        pos = [0.0, 0.0, 0.5]
        zero_actions = False

    class sensor( Go2RoughCfg.sensor):
        class proprioception( Go2RoughCfg.sensor.proprioception ):
            # latency_range = [0.0, 0.0]
            latency_range = [0.005, 0.045] # [s]

    class terrain( Go2RoughCfg.terrain ):
        num_rows = 10
        num_cols = 40
        selected = "BarrierTrack"
        slope_treshold = 20.

        max_init_terrain_level = 2
        curriculum = True
        
        pad_unavailable_info = True
        BarrierTrack_kwargs = dict(
            options= [
                "jump",
                "leap",
                "hurdle",
                "down",
                "tilted_ramp",
                "stairsup",
                "stairsdown",
                "discrete_rect",
                "slope",
                "wave",
                "leap",
            ], # each race track will permute all the options
            jump= dict(
                height= [0.05, 0.5],
                depth= [0.1, 0.3],
                # fake_offset= 0.1,
            ),
            leap= dict(
                length= [0.1, 0.7],
                depth= [0.5, 0.7],
                height= 0.2, # expected leap height over the gap
                # fake_offset= 0.1,
            ),
            hurdle= dict(
                height= [0.05, 0.5],
                depth= [0.2, 0.5],
                # fake_offset= 0.1,
                curved_top_rate= 0.1,
            ),
            down= dict(
                height= [0.1, 0.6],
                depth= [0.3, 0.5],
            ),
            tilted_ramp= dict(
                tilt_angle= [0.2, 0.5],
                switch_spacing= 0.,
                spacing_curriculum= False,
                overlap_size= 0.2,
                depth= [-0.1, 0.1],
                length= [0.6, 1.2],
            ),
            slope= dict(
                slope_angle= [0.2, 0.42],
                length= [1.2, 2.2],
                use_mean_height_offset= True,
                face_angle= [-3.14, 0, 1.57, -1.57],
                no_perlin_rate= 0.2,
                length_curriculum= True,
            ),
            slopeup= dict(
                slope_angle= [0.2, 0.42],
                length= [1.2, 2.2],
                use_mean_height_offset= True,
                face_angle= [-0.2, 0.2],
                no_perlin_rate= 0.2,
                length_curriculum= True,
            ),
            slopedown= dict(
                slope_angle= [0.2, 0.42],
                length= [1.2, 2.2],
                use_mean_height_offset= True,
                face_angle= [-0.2, 0.2],
                no_perlin_rate= 0.2,
                length_curriculum= True,
            ),
            stairsup= dict(
                height= [0.1, 0.3],
                length= [0.3, 0.5],
                residual_distance= 0.05,
                num_steps= [3, 19],
                num_steps_curriculum= True,
            ),
            stairsdown= dict(
                height= [0.1, 0.3],
                length= [0.3, 0.5],
                num_steps= [3, 19],
                num_steps_curriculum= True,
            ),
            discrete_rect= dict(
                max_height= [0.05, 0.2],
                max_size= 0.6,
                min_size= 0.2,
                num_rects= 10,
            ),
            wave= dict(
                amplitude= [0.1, 0.15], # in meter
                frequency= [0.6, 1.0], # in 1/meter
            ),
            track_width= 3.2,
            track_block_length= 2.4,
            wall_thickness= (0.01, 0.6),
            wall_height= [-0.5, 2.0],
            add_perlin_noise= True,
            border_perlin_noise= True,
            border_height= 0.,
            virtual_terrain= False,
            draw_virtual_terrain= True,
            engaging_next_threshold= 0.8,
            engaging_finish_threshold= 0.,
            curriculum_perlin= False,
            no_perlin_threshold= 0.1,
            randomize_obstacle_order= True,
            n_obstacles_per_track= 1,
        )

    class commands( Go2RoughCfg.commands ):
        # a mixture of command sampling and goal_based command update allows only high speed range
        # in x-axis but no limits on y-axis and yaw-axis
        lin_cmd_cutoff = 0.2
        class ranges( Go2RoughCfg.commands.ranges ):
            # lin_vel_x = [0.6, 1.8]
            lin_vel_x = [-0.6, 2.0]
        
        is_goal_based = True
        class goal_based:
            # the ratios are related to the goal position in robot frame
            x_ratio = None # sample from lin_vel_x range
            y_ratio = 1.2
            yaw_ratio = 1.
            follow_cmd_cutoff = True
            x_stop_by_yaw_threshold = 1. # stop when yaw is over this threshold [rad]

    class asset( Go2RoughCfg.asset ):
        terminate_after_contacts_on = []
        penalize_contacts_on = ["thigh", "calf", "base"]

    class termination( Go2RoughCfg.termination ):
        roll_kwargs = dict(
            threshold= 1.4, # [rad]
        )
        pitch_kwargs = dict(
            threshold= 1.6, # [rad]
        )
        timeout_at_border = True
        timeout_at_finished = False

    class rewards( Go2RoughCfg.rewards ):
        # class scales:
        #     tracking_lin_vel = 4.
        #     tracking_ang_vel = 1.
        #     energy_substeps = -2e-7
        #     torques = -1e-7
        #     stand_still = -2.
        #     dof_error_named = -2.
        #     dof_error = -0.005
        #     collision = -1e-4
        #     lazy_stop = -1.
        #     # penalty for hardware safety
        #     exceed_dof_pos_limits = -0.1
        #     exceed_torque_limits_l1norm = -0.1
        #     # penetration penalty
        #     penetrate_depth = -0.001
        #     #add
        #     leap_bonous_cond = 1.0
        #     powers = -1e-7
            
        #     jump_x_vel_cond = 0.5 #这个奖励函数是为了鼓励机器人在跳跃障碍时，具有一定的前进速度并且有一个适当的俯仰角（pitch）。
        #     sync_legs_cond = 0.5#跳跃时，强制机器人前后腿同步运动
        #     dof_error_cond = -5#惩罚机器人在未接触障碍物时的关节误差
            
        #     action_rate = -0.1 # 惩罚动作变化率过大
        #     action_smoothness = -0.01
        #     feet_air_time = 1e-5 # 奖励足部离地时间，鼓励跳跃动作
        #     leap_x_vel_cond = 1.0
            
        #     hip_pos = -1.  
        #     leap_pit_exploration=1.0
        class scales:
            # --- 基础行走约束（增强平地稳定性） ---
            tracking_lin_vel = 5.0
            tracking_ang_vel = 1.0
            # 增加平地足部接触惩罚（可选），如果不在坑边，鼓励四足轮流接触地面（Gait）
            # reference: 降低 feet_air_time 在平地的权重
            feet_air_time_mask = 0.1  # 显著降低，或者只在交互时给这个奖励

            # --- 跳跃核心奖励（必须是条件触发） ---
            # 1. 只有在坑附近（engaging）才给前进速度奖励
            leap_x_vel_cond = 5.0  # 提高权重，给足冲刺动力
            
            # 2. 修正跳跃姿态：跳跃时需要的是爆发力
            # 建议添加一个基于高度变化的奖励，仅在坑边触发
            # jump_up_vel_cond = 1.0 

            # --- 惩罚项调整 ---
            # 关键：加大平地的 dof_error_cond 惩罚
            dof_error_cond = -10.0 # 强制平地必须走得像正常的狗
            
            # 动作平滑性：蹦跳通常伴随剧烈的 action 变化
            action_rate = -0.05 
            action_smoothness = -0.02
            
            # --- 解决跳不过去的问题：碰撞与穿透 ---
            # 如果它经常掉进坑里，可能是因为掉进去后的惩罚不够重，或者掉进去后任务没结束
            collision = -1.0 # 加大碰撞惩罚
            penetrate_depth = 5.0 # 严厉惩罚身体部位进入坑内
            
            lazy_stop = -0.1
        
        tracking_sigma = 0.35
        soft_dof_pos_limit = 0.7

    class noise( Go2RoughCfg.noise ):
        add_noise = False

    class curriculum:
        penetrate_depth_threshold_harder = 100
        penetrate_depth_threshold_easier = 200
        no_moveup_when_fall = True
    
logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go2FieldCfgPPO( Go2RoughCfgPPO ):
    class algorithm( Go2RoughCfgPPO.algorithm ):
        entropy_coef = 0.0

    class runner( Go2RoughCfgPPO.runner ):
        experiment_name = "field_go2"

        resume = True
        load_run = osp.join(logs_root, "field_go2",
            "其它项目是目前最好的，但是在坑面前不动，基线2",
        )

        run_name = "".join(["Go2_",
            ("{:d}skills".format(len(Go2FieldCfg.terrain.BarrierTrack_kwargs["options"])))
        ])

        max_iterations = 1000
        save_interval = 200
        log_interval = 100
