# 机器人观测值 (Observations) 处理逻辑全流程

---

## 1. 物理状态提取与空间变换
从仿真引擎获取原始张量，并将其从全局世界坐标系转换至机器人局部坐标系，以实现运动学的解耦。

* **线速度 (Linear Velocity)(3,)**：
  将质心线速度投影至机体系。包含**掩蔽逻辑**：若非特权模式且配置关闭，则强制置零以模拟真实传感器缺失。

<pre><code class="language-python">
    # 若非特权模式，则置0
    def _get_lin_vel_obs(self, privileged= False):
        # backward compatibile for proprioception obs components and use_lin_vel related args
        obs_buf = self.base_lin_vel.clone()
        if (not privileged) and (not getattr(self.cfg.env, "use_lin_vel", True)):
            obs_buf[:, :3] = 0.
        if privileged and (not getattr(self.cfg.env, "privileged_use_lin_vel", True)):
            obs_buf[:, :3] = 0.
        return obs_buf
    self.base_lin_vel = quat_rotate_inverse(self.base_quat,	self.root_states[:, 7:10])
</code></pre>

* **角速度 (Angular Velocity)(3,)**：
  直接获取机体系下的角速度向量。

<pre><code class="language-python">
    def _get_ang_vel_obs(self, privileged= False):
        return self.base_ang_vel
    self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
</code></pre>

* **重力投影 (Projected Gravity)(3,)**：
  计算标准化重力向量在机体系下的朝向，表征机身相对于重力方向的倾斜度。
<pre><code class="language-python">
    def _get_projected_gravity_obs(self, privileged= False):
        return self.projected_gravity
    self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
</code></pre>


---

## 2. 细分观测分量处理逻辑
针对不同类型的传感器数据，进行偏移修正、坐标对齐及范围裁剪。
<pre><code class="language-python">
    self.base_quat = self.root_states[:, 3:7]
</code></pre>

### A. 本体感知数据 (Proprioceptive Data)
* **关节位置 (Joint Positions)(12,)**：
  计算当前关节角度与默认姿态（Default Pose）的偏差量，而非绝对角度。
<pre><code class="language-python">
    self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[..., :self.num_dof, 0]
    def _get_dof_pos_obs(self, privileged= False):
        return (self.dof_pos - self.default_dof_pos)
</code></pre>
* **关节速度 (Joint Velocities)(12,)**：
  直接记录各电机当前的旋转速度。
<pre><code class="language-python">
    self.dof_state.view(self.num_envs, -1, 2)[..., :self.num_dof, 1]
</code></pre>
* **历史动作 (Last Actions)(12,)**：
  保留上一时刻输出给控制器的动作张量，提供运动连续性参考。
<pre><code class="language-python">
    def _get_dof_pos_obs(self, privileged= False):
        return (self.dof_pos - self.default_dof_pos)
</code></pre>
* **控制指令 (User Commands)(3,)**：
  提取用户输入的前进、侧移、转向目标速度（前 3 维）。

### B. 机身位姿数据 (Base Pose Data)
* **位置处理**：计算机器人相对于当前环境原点（Environment Origins）的相对位移。
* **姿态角处理**：将四元数转换为欧拉角（Roll/Pitch/Yaw），并统一映射至 $(-\pi, \pi)$ 区间，确保角度表示的唯一性与连续性。

### C. 环境感知数据 (Exteroceptive Data)
* **高度扫描 (Height Measurements)**：
    1.  获取机器人质心高度并叠加修正偏移量（Offset）。
    2.  减去采样点处的绝对地形高度，获得相对高度差。
    3.  **数值裁剪**：使用 `clip` 将结果限制在 $[-1, 1]$ 米，消除极端地形对网络的异常干扰。
<pre><code class="language-python">
    def _get_height_measurements_obs(self, privileged= False):
        # not tested
        height_offset = getattr(self.cfg.normalization, "height_measurements_offset", -0.5)
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) + height_offset - self.measured_heights, -1, 1.)
        return heights
</pre></code>

---

## 3. 动态组装与标准化 (Aggregation & Scaling)


1.  **多模态拼接**：根据配置列表，按顺序将上述处理后的分量（速度、角度、高度图等）拼接为高维观测张量。
<pre><code class="language-python">
    def _get_obs_from_components(self, components: list, privileged= False):
        obs_segments = self.get_obs_segment_from_components(components)
        obs = []
        for k, v in obs_segments.items():
            # get the observation from specific component name
            # such as "_get_lin_vel_obs", "_get_ang_vel_obs", "_get_dof_pos_obs", "_get_forward_depth_obs"
            obs.append(
                getattr(self, "_get_" + k + "_obs")(privileged) * \
                getattr(self.obs_scales, k, 1.)
            )
        obs = torch.cat(obs, dim= 1)
        return obs
</pre></code>
2.  **特权属性判定**：区分“普通观测”与“特权观测（Privileged Obs）”，特权观测通常包含真实的线速度等无噪声完美数据，仅用于训练期间的教师网络或状态估计。
3.  **标准化缩放**：对拼接后的向量乘以对应的 `obs_scales`（如关节速度缩放系数），使不同物理单位的数据处于同一数量级。
<pre><code class="language-python">
    # base_config
    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            commands = [2., 2., 0.25] # matching lin_vel and ang_vel scales
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.
    
    # go2_config
    class normalization( LeggedRobotCfg.normalization ):
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            lin_vel = 1.
        height_measurements_offset = -0.2
        clip_actions_method = None # let the policy learns not to exceed the limits
</pre></code>

---

## 4. 特征降维与策略输入


1.  **环境特征压缩**：将高维的高度图扫描数据送入 **Encoder (MLP)**，压缩为低维的隐含向量（Latent Embedding）(32,)。
<pre><code class="language-python">
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
</pre></code>
2.  **最终融合**：将低维的本体特征与编码后的环境特征合并。
   <!-- <pre><code class="language-python">
   # 获取obs中关于height_measurement的数据组，输入mlp输出32维，并与本体的48维感知拼接
       def embed_encoders_latent(self, observations, obs_slices, encoders, latents_order):
        leading_dims = observations.shape[:-1]
        latents = []
        for encoder_i, encoder in enumerate(encoders):
            # This code is not clean enough, need to sort out later
            if isinstance(encoder, MlpModel):
                latents.append(encoder(
                    observations[..., obs_slices[encoder_i][0]].reshape(-1, np.prod(obs_slices[encoder_i][1]))
                ).reshape(*leading_dims, -1))
            elif isinstance(encoder, Conv2dHeadModel):
                latents.append(encoder(
                    observations[..., obs_slices[encoder_i][0]].reshape(-1, *obs_slices[encoder_i][1])
                ).reshape(*leading_dims, -1))
            else:
                raise NotImplementedError(f"Encoder for {type(encoder)} not implemented")
        # replace the obs vector with the latent vector in eace obs_slice[0] (the slice of obs)
        embedded_obs = []
        embedded_obs.append(observations[..., :obs_slices[latents_order[0]][0].start])
        for order_i in range(len(latents)- 1):
            current_idx = latents_order[order_i]
            next_idx = latents_order[order_i + 1]
            embedded_obs.append(latents[current_idx])
            embedded_obs.append(observations[..., obs_slices[current_idx][0].stop: obs_slices[next_idx][0].start])
        current_idx = latents_order[-1]
        next_idx = None
        embedded_obs.append(latents[current_idx])
        embedded_obs.append(observations[..., obs_slices[current_idx][0].stop:])
        
        return torch.cat(embedded_obs, dim= -1)
   </pre></code> -->
3.  **动作生成**：融合后的特征向量输入 **Actor 网络**，最终映射为 12 维的关节目标位置增量。
   <!-- <pre><code class="langauge-python">
    #

    from .actor_critic import ActorCritic
    class EncoderActorCritic(EncoderActorCriticMixin, ActorCritic):
        pass

    from .actor_critic_recurrent import ActorCriticRecurrent
    class EncoderActorCriticRecurrent(EncoderActorCriticMixin, ActorCriticRecurrent):
        pass
   </pre></code> -->