# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def is_primitive_type(obj):
    return not hasattr(obj, '__dict__')

def class_to_dict(obj) -> dict:
    if not hasattr(obj,"__dict__") or isinstance(obj, dict):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict_, strict= False):
    """ If strict, attributes that are not in dict_ will be removed from obj """
    attr_names = [n for n in obj.__dict__.keys() if not (n.startswith("__") and n.endswith("__"))]
    for attr_name in attr_names:
        if not attr_name in dict_:
            delattr(obj, attr_name)
    for key, val in dict_.items():
        attr = getattr(obj, key, None)
        if attr is None or is_primitive_type(attr):
            if isinstance(val, dict):
                setattr(obj, key, copy.deepcopy(val))
                update_class_from_dict(getattr(obj, key), val)
            else:
                setattr(obj, key, val)
        else:
            update_class_from_dict(attr, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    if load_run==-1:
        try:
            runs = os.listdir(root)
            #TODO sort by date to handle change of month
            runs.sort()
            if 'exported' in runs: runs.remove('exported')
            last_run = os.path.join(root, runs[-1])
        except:
            raise ValueError("No runs in this directory: " + root)
        load_run = last_run
    elif os.path.isabs(load_run):
        print("Loading load_run as absolute path:", load_run)
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args(custom_args=[]):
    custom_parameters = [
        {"name": "--task", "type": str, "default": "anymal_c_flat", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
    ] + custom_args
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        rnn_module = actor_critic.memory_a.rnn
        
        # 自动判断类型
        if isinstance(rnn_module, torch.nn.LSTM):
            print("检测到 LSTM 模型，正在导出...")
            exporter = PolicyExporterLSTM(actor_critic)
        elif isinstance(rnn_module, torch.nn.GRU):
            print("检测到 GRU 模型，正在导出...")
            exporter = PolicyExporterGRU(actor_critic)
        else:
            raise TypeError(f"不支持的 RNN 类型: {type(rnn_module)}。仅支持 LSTM 或 GRU。")
            
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        # 在 traced_script_module = torch.jit.script(self) 之前添加：
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class PolicyExporterGRU(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.use_state_estimator = hasattr(actor_critic, "state_estimator") #先判断是否使用线速度估计
        self.use_estimator_memory = hasattr(actor_critic, "memory_s") #先判断是否使用估计器记忆
        if self.use_state_estimator:
            self.state_estimator = copy.deepcopy(actor_critic.state_estimator)
        else:
            self.state_estimator = torch.nn.Identity()
        if self.use_estimator_memory:
            self.estimator_memory = copy.deepcopy(actor_critic.memory_s.rnn)
        else:
            self.estimator_memory = torch.nn.Identity()
        
        if hasattr(actor_critic, 'encoders'):
            self.encoders = copy.deepcopy(actor_critic.encoders)
            self.encoders.cpu()
        else:
            self.encoders = None

        self.actor.cpu()
        self.memory.cpu()
        self.state_estimator.cpu()
        self.estimator_memory.cpu()

        # --- 新增：彻底清洗所有子模块中的 NumPy 类型 ---
        for module in self.modules():
            # 修复 Linear 层
            if isinstance(module, torch.nn.Linear):
                module.in_features = int(module.in_features)
                module.out_features = int(module.out_features)
            # 修复 RNN 层 (GRU/LSTM)
            if isinstance(module, (torch.nn.GRU, torch.nn.LSTM)):
                module.input_size = int(module.input_size)
                module.hidden_size = int(module.hidden_size)
                module.num_layers = int(module.num_layers)
            # 修复部分模块中可能残留的 numpy.int64 属性，避免 TorchScript 报错
            if hasattr(module, "_output_size"):
                output_size = getattr(module, "_output_size")
                if isinstance(output_size, np.generic):
                    setattr(module, "_output_size", int(output_size))

        # 注册 buffer
        self.register_buffer('hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        if self.use_estimator_memory:
            self.register_buffer('estimator_hidden_state', torch.zeros(self.estimator_memory.num_layers, 1, self.estimator_memory.hidden_size))
        else:
            self.register_buffer('estimator_hidden_state', torch.zeros(1, 1, 1))

        # 导出时确认是否走线速度估计分支（训练里 replace lin_vel 时应有 state_estimator + memory_s）
        print("[PolicyExporterGRU] lin_vel path:")
        print(f"  use_state_estimator (hasattr actor_critic.state_estimator) = {self.use_state_estimator}")
        print(f"  state_estimator module = {type(self.state_estimator).__name__}")
        if self.use_state_estimator:
            print("  -> forward 会用 45 维 obs 估计 lin_vel 并替换 obs_direct[:, :3]")
        else:
            print("  -> forward 不会估计线速度（Identity），obs 里 lin_vel 原样进 GRU")
        print(f"  use_estimator_memory (hasattr actor_critic.memory_s) = {self.use_estimator_memory}")
        print(f"  estimator_memory module = {type(self.estimator_memory).__name__}")
        if self.use_estimator_memory:
            print("  -> 估计器侧带 RNN 隐藏状态 estimator_hidden_state")
        else:
            print("  -> 估计器侧无 RNN（Identity），仅 MLP 直接映射 45 维 -> 3 维（若上面不是 Identity）")

    def forward(self, x):
        # x 的输入维度是 279
        # 1. 切片：前 48 维直接使用，后 231 维进入 encoder
        obs_direct = x[:, :48]       # 形状: [batch, 48]
        obs_to_encode = x[:, 48:]    # 形状: [batch, 231]

        # 训练时 lin_vel 由 estimator 预测，输入是 45 维非 lin_vel 观测。
        if self.use_state_estimator:
            estimator_obs = obs_direct[:, 3:48]  # 45 dims
            if self.use_estimator_memory:
                #GRU层进行处理，结合45维观测和之前的记忆，得到特征
                estimator_feat, estimator_h = self.estimator_memory(estimator_obs.unsqueeze(0), self.estimator_hidden_state)
                self.estimator_hidden_state[:] = estimator_h #更新记忆
                estimated_lin_vel = self.state_estimator(estimator_feat.squeeze(0))
            else:
                estimated_lin_vel = self.state_estimator(estimator_obs)
            obs_direct = obs_direct.clone()
            obs_direct[:, :3] = estimated_lin_vel

        # 2. 编码过程
        if self.encoders is not None:
            encoded_part = obs_to_encode
            for encoder in self.encoders:
                encoded_part = encoder(encoded_part)
            # 编码后的 encoded_part 应该是 32 维
        else:
            encoded_part = obs_to_encode # 降级处理

        # 3. 拼接：将直接观测 (48) 和 编码特征 (32) 拼接成 80 维
        # dim=-1 确保在特征维度拼接
        x_combined = torch.cat([obs_direct, encoded_part], dim=-1)

        # 4. 经过 GRU (需要增加序列维度)
        out, h = self.memory(x_combined.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h
        
        # 5. 经过 Actor MLP
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        """用于在 JIT 部署环境中重置机器人记忆"""
        self.hidden_state[:] = 0.
        if self.use_estimator_memory:
            self.estimator_hidden_state[:] = 0.

    def _infer_obs_dim_for_self_check(self):
        """Infer obs dim for a quick exporter sanity check."""
        if self.encoders is None or len(self.encoders) == 0:
            return None
        first_encoder = self.encoders[0]
        model = getattr(first_encoder, "model", None)
        if model is None or len(model) == 0:
            return None
        first_layer = model[0]
        if not isinstance(first_layer, torch.nn.Linear):
            return None
        # Current exporter assumes 48 direct obs + encoded raw obs.
        return 48 + int(first_layer.in_features)

    def _run_export_sanity_check(self):
        """Print estimator output shape and replaced lin_vel once."""
        obs_dim = self._infer_obs_dim_for_self_check()
        if obs_dim is None:
            print("[PolicyExporterGRU self-check] skipped (cannot infer obs_dim).")
            return

        x = torch.zeros((1, obs_dim), dtype=torch.float32)
        with torch.no_grad():
            obs_direct = x[:, :48].clone()
            if self.use_state_estimator:
                estimator_obs = obs_direct[:, 3:48]
                if self.use_estimator_memory:
                    feat, est_h = self.estimator_memory(estimator_obs.unsqueeze(0), self.estimator_hidden_state)
                    estimated_lin_vel = self.state_estimator(feat.squeeze(0))
                    # Do not keep check-time memory state.
                    self.estimator_hidden_state[:] = 0.
                else:
                    estimated_lin_vel = self.state_estimator(estimator_obs)
                obs_direct[:, :3] = estimated_lin_vel
                print(
                    "[PolicyExporterGRU self-check] "
                    f"estimator_out_shape={tuple(estimated_lin_vel.shape)}, "
                    f"replaced_obs_0_3={obs_direct[0, :3].tolist()}"
                )
            else:
                print("[PolicyExporterGRU self-check] skipped (no state_estimator).")

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_gru_0330.pt')
        self.to('cpu')
        self._run_export_sanity_check()
        self.reset_memory()
        
        # 导出为 TorchScript 模块
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
        print(f"成功导出 GRU 策略模型至: {path}")

def merge_dict(this: dict, other: dict):
    """ Merging two dicts. if a key exists in both dict, the other's value will take priority
    NOTE: This method is implemented in python>=3.9
    """
    output = this.copy()
    output.update(other)
    return output
