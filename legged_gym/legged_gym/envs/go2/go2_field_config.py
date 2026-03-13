""" Config to train the whole parkour oracle policy """
import numpy as np
from os import path as osp
from collections import OrderedDict

from legged_gym.envs.go2.go2_config import Go2RoughCfg, Go2RoughCfgPPO

class Go2FieldCfg( Go2RoughCfg ):
    class init_state( Go2RoughCfg.init_state ):
        pos = [0.0, 0.0, 0.7]
        zero_actions = False

    class sensor( Go2RoughCfg.sensor):
        class proprioception( Go2RoughCfg.sensor.proprioception ):
            # latency_range = [0.0, 0.0]write
            latency_range = [0.005, 0.045] # [s]

    class terrain( Go2RoughCfg.terrain ):
        #明确定义，地形分为列cols和行rows，列之间有墙相连，同一列可视为一个赛道，num_rows和num_cols决定了赛道的大小
        # rows控制一个赛道中障碍（地形）的数量
        # （num_rows 只影响 difficulty，不影响哪些障碍出现）
        num_rows = 12

        num_cols = 40
        selected = "BarrierTrack"
        slope_treshold = 20.

        max_init_terrain_level = 3
        curriculum = True

        pad_unavailable_info = True
        #地形类型的列表，如果想增大某一种地形的比例，可以重复添加该类型
        BarrierTrack_kwargs = dict(
            options= [
                "jump",
                "leap",
                "leap",
                "leap",
                "hurdle",
                "down",
                "tilted_ramp",
                "stairsup",
                "stairsdown",
                "discrete_rect",
                "slope",
                "wave",
            ], # each race track will permute all the options
            jump= dict(
                height= [0.05, 0.5],
                depth= [0.1, 0.3],
                # fake_offset= 0.1,
            ),
            leap= dict(
                length= [0.2, 1.5],
                depth= [0.5, 0.8],
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
            randomize_obstacle_order= True,#随机抽取障碍
            n_obstacles_per_track= 1,
        )

    class commands( Go2RoughCfg.commands ):
        # a mixture of command sampling and goal_based command update allows only high speed range
        # in x-axis but no limits on y-axis and yaw-axis
        lin_cmd_cutoff = 0.2
        class ranges( Go2RoughCfg.commands.ranges ):
            # lin_vel_x = [0.6, 1.8]
            lin_vel_x = [-1, 2.0]

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
        class scales(Go2RoughCfg.rewards.scales):
            # ===== 核心目标：通过障碍，而不是单纯平地走稳 =====
            tracking_lin_vel = 0.0
            tracking_world_vel = 6.0
            tracking_ang_vel = 1.0

            # ===== 姿态稳定，但不要过度限制跨越动作 =====
            lin_vel_z = -0.6
            ang_vel_xy = -0.15
            orientation = -0.35
            base_height = -0.10
            yaw_abs = -0.25
            lin_pos_y = -0.25

            # ===== 障碍安全 / 鲁棒性 =====
            collision = -3.0
            penetrate_depth = -0.02
            penetrate_volume = -0.002
            stumble = -0.20
            exceed_dof_pos_limits = -0.20
            exceed_torque_limits_l1norm = -0.12
            feet_contact_forces = -0.00005

            # ===== 动作质量 / 硬件友好 =====
            energy_substeps = -5e-7
            powers = -1e-5
            action_rate = -0.12
            action_smoothness = -0.08

            # ===== 姿态先验：非障碍阶段尽量保持接近平地稳定解 =====
            hip_pos = -1.2
            dof_error_named = -0.02
            dof_error = -0.002
            dof_error_cond = -0.15

            # ===== 障碍特化奖励 =====
            jump_x_vel_cond = 0.8
            leap_bonous_cond = 2.5
            down_cond = 0.10
            sync_all_legs_cond = -0.20

            # ===== 不再强调平地步态细节 =====
            feet_air_time = 0.0
            has_contact = 0.0
            stand_still = 0.0
            foot_mirror = 0.0
            foot_slide = -0.01
            lazy_stop = 0.0

        only_positive_rewards = False
        base_height_target = 0.32


    class noise( Go2RoughCfg.noise ):
        add_noise = False

    class curriculum:
        penetrate_volume_threshold_harder = 8000
        penetrate_volume_threshold_easier = 16000
        penetrate_depth_threshold_harder = 800
        penetrate_depth_threshold_easier = 1600
        no_moveup_when_fall = True

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go2FieldCfgPPO( Go2RoughCfgPPO ):
    class algorithm( Go2RoughCfgPPO.algorithm ):
        entropy_coef = 0.0

    class runner( Go2RoughCfgPPO.runner ):
        experiment_name = "field_go2"

        resume = True
        load_run = osp.join(logs_root, "rough_go2",
            "Mar12_06-30-27_Go2Rough",
        )

        run_name = "".join(["Go2_"])

        max_iterations = 20000
        save_interval = 1000
        log_interval = 100
