# InternVLA 导航循环分析与海马体机制评估

## 概述

本文档详细分析 InternVLA-N1 的导航循环机制，包括：
1. 输入方式（是否使用360度图像）
2. 数据集与训练方法
3. 是否具备类似海马体的"新奇/重复"(Novel/Repeat) 记忆机制
4. Grid Map 作为注意力机制的可行性
5. 眼动序列规划（Saccade Sequence）能力评估

---

## 1. InternVLA 是否使用360度图像作为输入？

### 答案：**否，InternVLA 不使用360度全景图像**

InternVLA-N1 使用的是**多视角定向摄像头图像**，而非360度全景图像：

### 当前输入方式

```python
# 来自 internnav/dataset/internvla_n1_lerobot_dataset.py

# 摄像头配置示例
R2R_125CM_0_30 = {
    "data_path": "traj_data/r2r",
    "height": 125,      # 摄像头高度 125cm
    "pitch_1": 0,       # 水平视角 0度
    "pitch_2": 30,      # 向下俯视 30度
}

R2R_60CM_15_15 = {
    "data_path": "traj_data/r2r",
    "height": 60,       # 低位摄像头 60cm  
    "pitch_1": 15,      # 两个角度都是15度
    "pitch_2": 15,
}
```

### 视觉输入结构

| 组件 | 描述 | 360度支持 |
|------|------|----------|
| 主摄像头 | RGB图像 (pitch_1角度) | ❌ 单一方向 |
| 辅助摄像头 | RGB图像 (pitch_2角度，用于look_down) | ❌ 单一方向 |
| 历史帧 | 过去N帧的RGB图像序列 | ❌ 时序记忆，非空间全景 |

### 为什么不使用360度图像？

1. **VLN-CE基准设计**：Room-to-Room (R2R) 和 RxR 数据集基于 Matterport3D，使用离散视点而非连续360度扫描
2. **计算效率**：360度图像处理需要更大的模型容量
3. **实际机器人部署**：多数机器人使用定向摄像头而非全景相机

---

## 2. 数据集与训练方法

### 2.1 使用的数据集

```python
# 来自 internnav/dataset/internvla_n1_lerobot_dataset.py

data_dict = {
    # VLN 导航数据集
    "r2r_125cm_0_30": R2R_125CM_0_30,      # Room-to-Room
    "rxr_125cm_0_30": RxR_125CM_0_30,      # Multilingual R2R
    "scalevln_125cm_0_30": SCALEVLN_125CM_0_30,  # Large-scale VLN
    
    # 视觉-语言预训练数据集
    "cambrian_737k": CAMBRIAN_737K,         # 737K VL样本
    "mp_doc": MP_DOC,                       # 文档理解
    "clevr_mc": CLEVR_MC,                   # 视觉推理
    "videochatgpt": VIDEOCHATGPT,           # 视频理解
}
```

### 2.2 训练循环代码片段

```python
# 来自 internnav/trainer/internvla_n1_trainer.py

def train(attn_implementation="flash_attention_2"):
    # 1. 解析参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 2. 数据增强（可选）
    if data_args.data_augmentation:
        data_args.transform_train = v2.Compose([
            v2.ToImage(),
            v2.ColorJitter(brightness=0.2, saturation=0.2),
            v2.RandomPosterize(bits=4),
            v2.RandomAdjustSharpness(sharpness_factor=1.5),
            v2.RandomAutocontrast(),
            v2.ToPILImage(),
            v2.Resize((data_args.resize_h, data_args.resize_w)),
        ])
    
    # 3. 加载模型
    model = InternVLAN1ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    
    # 4. 初始化导航模块 (System 1: NavDP 或 NextDiT)
    model.get_model().initialize_vision_modules(model_args=model_args)
    
    # 5. 设置可训练参数
    set_model(model_args, model)  # 冻结/解冻特定层
    
    # 6. 准备数据模块
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # 7. 创建Trainer并训练
    trainer = Trainer(model=model, args=training_args, **data_module)
    trainer.train()
```

### 2.3 双系统训练损失

```python
# 来自 internnav/model/basemodel/internvla_n1/internvla_n1.py

def forward(self, ...):
    # System 2: VLM语言建模损失（隐式）
    outputs = self.model(inputs_embeds=inputs_embeds, ...)
    logits = self.lm_head(hidden_states)
    
    # System 1: 轨迹扩散损失
    if 'nextdit' in self.get_system1_type():
        # NextDiT 扩散模型
        noisy_trajectory = (1 - sigmas) * relative_poses + sigmas * noise
        noise_pred = self.get_model().traj_dit(x=action_features, timestep=timesteps, z_latents=latents)
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        
    elif 'navdp' in self.get_system1_type():
        # NavDP 策略
        pred_pg, noise = self.model.navdp.forward_vlm_traj(traj_hidden_states, images_dp, depths_dp)
        loss = (pred_pg - noise).square()
```

---

## 3. InternVLA 现有的记忆机制

### 3.1 MemoryEncoder（记忆编码器）

```python
# 来自 internnav/model/basemodel/internvla_n1/internvla_n1_arch.py

class MemoryEncoder(nn.Module):
    """
    Transformer-based 记忆聚合模块
    - 类似于海马体的序列记忆编码
    """
    def __init__(self, hidden_size=384, num_heads=6, num_layers=3, max_len=512):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads,  # 6个注意力头
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.memory_pos = nn.Parameter(torch.randn(max_len, hidden_size))  # 位置编码
    
    def forward(self, memory, memory_mask=None):
        B, N, C = memory.shape
        pos = self.memory_pos[:N, :].unsqueeze(0).expand(B, -1, -1)
        memory = memory + pos
        encoded_memory = self.encoder(memory, src_key_padding_mask=memory_mask)
        return encoded_memory
```

### 3.2 QFormer（查询式重采样器）

```python
# 来自 internnav/model/basemodel/internvla_n1/internvla_n1_arch.py

class QFormer(nn.Module):
    """
    Query-based attention 机制
    - 32个可学习的查询token
    - 类似于海马体的"检索"机制
    """
    def __init__(self, num_query=32, hidden_size=768, num_layers=3, num_heads=12):
        self.query_tokens = nn.Parameter(torch.randn(num_query, hidden_size))
        self.query_pos = nn.Parameter(torch.randn(num_query, hidden_size))
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
    
    def forward(self, visual_feats, visual_attn_mask=None):
        query_tokens = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        query_tokens = query_tokens + self.query_pos.unsqueeze(0)
        out = self.decoder(query_tokens, visual_feats, memory_key_padding_mask=visual_attn_mask)
        return out
```

### 3.3 历史帧采样（时序记忆）

```python
# 来自 internnav/model/basemodel/internvla_n1/internvla_n1_policy.py

def s2_step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
    # 历史帧采样 - 均匀分布的历史观察
    if self.episode_idx == 0:
        history_id = []
    else:
        history_id = np.unique(
            np.linspace(0, self.episode_idx - 1, self.num_history, dtype=np.int32)
        ).tolist()
        placeholder = (self.DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
        sources[0]["value"] += f' These are your historical observations: {placeholder}.'
    
    # 组合历史帧和当前帧
    self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
```

---

## 4. 是否具备 Novel/Repeat 区分机制？

### 当前状态：**部分具备，但不显式**

| 机制 | 海马体功能 | InternVLA 实现 | 差距 |
|------|------------|----------------|------|
| Novel检测 | CA1区对新环境高频激活 | ❌ 无显式新奇检测 | **需添加** |
| Repeat抑制 | 抑制重复扫描(IOR) | ⚠️ 隐式（VLM内部注意力） | 部分 |
| 空间记忆 | 网格细胞路径积分 | ❌ 仅时序记忆 | **需添加** |
| 场景识别 | 海马体场景匹配 | ⚠️ QFormer查询机制 | 部分 |

### 4.1 缺失的 Novel 检测机制

**建议实现方案：**

```python
class NoveltyDetector(nn.Module):
    """
    基于海马体CA1区的新奇检测机制
    - 比较当前观察与记忆库的相似度
    - 高相似度 = Repeat (抑制扫描)
    - 低相似度 = Novel (激活探索)
    """
    def __init__(self, feature_dim=384, memory_size=100, threshold=0.8):
        super().__init__()
        self.memory_bank = nn.Parameter(torch.randn(memory_size, feature_dim))
        self.threshold = threshold
        self.projection = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, current_obs):
        # 投影当前观察
        obs_proj = self.projection(current_obs)  # (B, D)
        
        # 计算与记忆库的相似度
        similarity = F.cosine_similarity(
            obs_proj.unsqueeze(1),  # (B, 1, D)
            self.memory_bank.unsqueeze(0),  # (1, M, D)
            dim=-1
        )  # (B, M)
        
        max_similarity = similarity.max(dim=-1).values  # (B,)
        
        # Novel: max_similarity < threshold
        is_novel = max_similarity < self.threshold
        
        # 更新记忆库（如果是Novel）
        if is_novel.any():
            self._update_memory(obs_proj[is_novel])
        
        return is_novel, max_similarity
    
    def _update_memory(self, new_obs):
        # FIFO更新记忆库
        self.memory_bank.data = torch.cat([
            self.memory_bank.data[1:],
            new_obs.mean(dim=0, keepdim=True)
        ], dim=0)
```

### 4.2 缺失的 Inhibition of Return (IOR) 机制

**建议实现方案：**

```python
class InhibitionOfReturn(nn.Module):
    """
    空间注意力抑制机制
    - 记录已访问位置
    - 抑制对已访问位置的重复注意
    """
    def __init__(self, grid_size=32, decay_rate=0.9):
        super().__init__()
        self.inhibition_map = nn.Parameter(
            torch.zeros(grid_size, grid_size), 
            requires_grad=False
        )
        self.decay_rate = decay_rate
    
    def update(self, visited_position):
        """标记已访问位置"""
        x, y = visited_position
        self.inhibition_map[x, y] = 1.0
        self.inhibition_map *= self.decay_rate  # 时间衰减
    
    def get_attention_mask(self):
        """返回注意力抑制掩码"""
        return 1.0 - self.inhibition_map  # 已访问位置被抑制
```

---

## 5. Grid Map 作为注意力机制的可行性

### 5.1 当前架构分析

根据 [cognitive_mapping_analysis.md](./cognitive_mapping_analysis.md) 的分析：

**当前状态：**
- Grid Map (Occupancy Map) **不是**模型输入
- 仅用于碰撞检测和可视化
- System 1 和 System 2 不直接使用 Grid Map

### 5.2 Grid Map 注意力集成方案

```python
class GridMapAttention(nn.Module):
    """
    将 Grid Map 作为空间注意力机制集成到模型中
    
    灵感：海马体的空间表征与视觉注意力的结合
    - Grid Map 提供空间上下文
    - 摄像头特征提供视觉内容
    - 交叉注意力融合两种表征
    """
    def __init__(self, visual_dim=384, grid_dim=64, num_heads=8):
        super().__init__()
        
        # Grid Map 编码器
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, grid_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),  # (B, grid_dim, 8, 8)
        )
        
        self.grid_proj = nn.Linear(grid_dim, visual_dim)
        
        # 交叉注意力：视觉特征关注Grid Map
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
    def forward(self, visual_features, grid_map):
        """
        Args:
            visual_features: (B, N_vis, D) 来自摄像头的视觉特征
            grid_map: (B, H, W) 占用栅格图
        
        Returns:
            fused_features: (B, N_vis, D) 融合了空间信息的视觉特征
        """
        # 编码 Grid Map
        B = grid_map.shape[0]
        grid_feat = self.grid_encoder(grid_map.unsqueeze(1))  # (B, D, 8, 8)
        grid_feat = grid_feat.flatten(2).permute(0, 2, 1)  # (B, 64, D)
        grid_feat = self.grid_proj(grid_feat)  # (B, 64, visual_dim)
        
        # 交叉注意力：视觉特征查询Grid Map
        fused, attn_weights = self.cross_attn(
            query=visual_features,
            key=grid_feat,
            value=grid_feat
        )
        
        # 残差连接
        output = visual_features + fused
        
        return output, attn_weights
```

### 5.3 集成到 InternVLA

```python
# 修改 internnav/model/basemodel/internvla_n1/internvla_n1_arch.py

class InternVLAN1MetaModel:
    def __init__(self, config):
        super().__init__(config)
        
        # 现有模块
        self.latent_queries = nn.Parameter(torch.randn(1, config.n_query, config.hidden_size))
        
        # 新增：Grid Map 注意力
        if hasattr(config, 'use_grid_attention') and config.use_grid_attention:
            self.grid_attention = GridMapAttention(
                visual_dim=config.hidden_size,
                grid_dim=64,
                num_heads=8
            )
            
        # 新增：新奇检测
        if hasattr(config, 'use_novelty_detection') and config.use_novelty_detection:
            self.novelty_detector = NoveltyDetector(
                feature_dim=config.hidden_size,
                memory_size=100
            )
```

---

## 6. 眼动序列规划 (Saccade Sequence) 能力评估

### 6.1 当前状态

**InternVLA 已具备部分序列规划能力：**

```python
# 来自 internnav/model/basemodel/internvla_n1/internvla_n1.py

def generate_traj(self, traj_latents, images_dp, depths_dp, predict_step_nums=32, ...):
    """
    生成32步的轨迹序列
    - 使用扩散模型(NextDiT)进行序列生成
    - 可以看作是"运动规划序列"
    """
    latent_size = predict_step_nums  # 32步
    latents = randn_tensor(shape=(batch_size * num_sample_trajs, latent_size, 3))
    
    # 迭代去噪生成轨迹
    for t in scheduler.timesteps:
        noise_pred = self.get_model().traj_dit(x=latent_model_input, timestep=t, z_latents=hidden_states)
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    return latents
```

### 6.2 与海马体眼动规划的对比

| 特性 | 海马体眼动规划 | InternVLA当前实现 | 差距 |
|------|--------------|-------------------|------|
| 多步规划 | ✅ 规划眼动序列 | ✅ 32步轨迹生成 | 匹配 |
| 目标导向 | ✅ 基于认知地图 | ⚠️ 基于语言指令 | 可增强 |
| 空间结构利用 | ✅ "厨房在客厅右边" | ❌ 无显式空间关系 | **需添加** |
| 子目标分解 | ✅ 客厅→厨房→冰箱 | ⚠️ 隐式在VLM中 | 部分 |

### 6.3 建议：显式子目标规划

```python
class SubgoalPlanner(nn.Module):
    """
    基于海马体的分层导航规划
    - 将长距离导航分解为子目标序列
    - 每个子目标可以是房间、地标或航点
    """
    def __init__(self, hidden_size=768, num_subgoals=5):
        super().__init__()
        self.num_subgoals = num_subgoals
        
        # 子目标预测器
        self.subgoal_predictor = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8),
            num_layers=2
        )
        
        # 可学习的子目标查询
        self.subgoal_queries = nn.Parameter(torch.randn(num_subgoals, hidden_size))
        
        # 子目标到动作的映射
        self.subgoal_to_action = nn.Linear(hidden_size, 3)  # (dx, dy, dyaw)
    
    def forward(self, instruction_embedding, current_obs, spatial_memory=None):
        """
        Args:
            instruction_embedding: (B, L, D) 指令嵌入
            current_obs: (B, D) 当前观察嵌入
            spatial_memory: (B, M, D) 空间记忆（如Grid Cell编码）
        
        Returns:
            subgoals: (B, num_subgoals, D) 子目标序列
            subgoal_actions: (B, num_subgoals, 3) 到达每个子目标的动作
        """
        B = instruction_embedding.shape[0]
        
        # 准备上下文
        if spatial_memory is not None:
            context = torch.cat([instruction_embedding, spatial_memory], dim=1)
        else:
            context = instruction_embedding
        
        # 预测子目标序列
        queries = self.subgoal_queries.unsqueeze(0).expand(B, -1, -1)
        subgoals = self.subgoal_predictor(queries, context)
        
        # 转换为动作
        subgoal_actions = self.subgoal_to_action(subgoals)
        
        return subgoals, subgoal_actions
```

---

## 7. 总结与建议

### 7.1 InternVLA 现有机制总结

| 功能 | 状态 | 说明 |
|------|------|------|
| 360度输入 | ❌ 不使用 | 使用定向多视角摄像头 |
| 时序记忆 | ✅ 具备 | MemoryEncoder + 历史帧采样 |
| 空间记忆 | ❌ 缺失 | 无Grid Cell或认知地图 |
| Novel检测 | ❌ 缺失 | 无显式新奇检测 |
| Repeat抑制 | ⚠️ 部分 | VLM注意力隐式包含 |
| 序列规划 | ✅ 具备 | NextDiT 32步轨迹生成 |
| 子目标规划 | ⚠️ 部分 | VLM隐式分解，无显式模块 |

### 7.2 建议的增强方向

1. **添加 Grid Cell 编码器**
   - 参考 [cognitive_mapping_analysis.md](./cognitive_mapping_analysis.md) 中的 GridCellEncoder 方案
   - 提供精确的路径积分和空间自定位

2. **实现 Novelty Detection**
   - 基于 CA1 区的新奇检测机制
   - 驱动探索性扫描 vs 高效利用模式

3. **添加 Grid Map Attention**
   - 将占用栅格图作为空间注意力先验
   - 融合视觉特征和空间结构

4. **显式子目标规划器**
   - 分层导航：指令 → 子目标序列 → 动作序列
   - 类似海马体的"厨房在客厅右边"空间推理

### 7.3 实现优先级

| 优先级 | 模块 | 预估工作量 | 预期收益 |
|--------|------|------------|----------|
| 高 | Grid Cell Encoder | 3-5天 | 精确空间定位 |
| 高 | Novelty Detection | 2-3天 | 探索效率提升 |
| 中 | Grid Map Attention | 2-3天 | 避障能力增强 |
| 中 | Subgoal Planner | 3-5天 | 长距离导航改善 |
| 低 | IOR 机制 | 1-2天 | 减少冗余扫描 |

---

## 附录：关键代码文件位置

| 功能 | 文件路径 |
|------|----------|
| 模型架构 | `internnav/model/basemodel/internvla_n1/internvla_n1_arch.py` |
| 前向传播 | `internnav/model/basemodel/internvla_n1/internvla_n1.py` |
| 导航策略 | `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py` |
| 训练循环 | `internnav/trainer/internvla_n1_trainer.py` |
| 数据集 | `internnav/dataset/internvla_n1_lerobot_dataset.py` |
| 认知地图分析 | `docs/cognitive_mapping_analysis.md` |
