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
            ], # each race track will permute all the options
            jump= dict(
                height= [0.05, 0.5],
                depth= [0.1, 0.3],
                # fake_offset= 0.1,
            ),
            leap= dict(
                length= [0.05, 0.8],
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
        class scales:
            tracking_lin_vel = 1.
            tracking_ang_vel = 1.
            energy_substeps = -2e-7
            torques = -1e-7
            stand_still = -1.
            dof_error_named = -1.
            dof_error = -0.005
            collision = -0.05
            lazy_stop = -3.
            # penalty for hardware safety
            exceed_dof_pos_limits = -0.1
            exceed_torque_limits_l1norm = -0.1
            # penetration penalty
            penetrate_depth = -0.05

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
        load_run = osp.join(logs_root, "rough_go2",
            "Mar12_06-30-27_Go2Rough",
        )

        run_name = "".join(["Go2_",
            ("{:d}skills".format(len(Go2FieldCfg.terrain.BarrierTrack_kwargs["options"]))),
        ])

        max_iterations = 2000
        save_interval = 500
        log_interval = 100
        