# Cognitive Mapping Integration Analysis for InternVLN

## Overview

This document analyzes the feasibility of integrating cognitive mapping based on grid cells and head direction cells (Zeng et al. 2017) into the InternVLN (DualVLN) codebase. The analysis focuses on understanding the current occupancy map usage, identifying integration points for grid cells, and providing actionable recommendations.

**Reference Paper**: "Cognitive Mapping Based on Conjunctive Representation of Grid Cells and Head Direction Cells" (Zeng et al. 2017)

---

## Task 1: Occupancy Map Usage Analysis

### Usage Locations

```json
{
  "usage_locations": [
    {
      "file": "internnav/evaluator/utils/common.py",
      "line": "43-60",
      "function": "freemap_to_accupancy_map()",
      "note": "Function name has typo in original codebase (accupancy vs occupancy)",
      "context": "Converts freespace map to occupancy map with dilation for visualization and path planning"
    },
    {
      "file": "internnav/evaluator/utils/path_plan.py", 
      "line": "63-104",
      "function": "vis_nav_path()",
      "context": "Visualization of planned path on occupancy map"
    },
    {
      "file": "internnav/env/utils/internutopia_extension/controllers/vln_move_by_flash_with_collision_controller.py",
      "line": "103-137",
      "function": "get_map_info()",
      "context": "CRITICAL: Generates binary free-space map from top-down depth camera for collision checking"
    },
    {
      "file": "internnav/env/utils/internutopia_extension/controllers/vln_move_by_flash_with_collision_controller.py",
      "line": "139-150",
      "function": "check_collision()",
      "context": "Uses depth-derived occupancy map to check if robot position is occupied"
    }
  ],
  "is_model_input": false,
  "depth_dependency": "critical-for-collision",
  "summary": "Occupancy map is NOT a direct model input during training/inference. It is used primarily for:\n1. Collision detection during robot control (CRITICAL)\n2. Path planning visualization\n3. Evaluation metrics\n\nSystem 1 (NavDP) and System 2 (InternVLA-N1) do NOT receive occupancy map as input - they work with RGB/depth images and latent goals."
}
```

### Key Findings

1. **Not a Model Input**: The occupancy map is not fed into the neural network during training or inference
2. **Collision Detection**: Primary use is in `VlnMoveByFlashCollisionController` for real-time obstacle avoidance
3. **Depth Dependency**: Requires top-down depth camera to generate the free-space map
4. **Visualization**: Used for path planning visualization in evaluation

### Depth Usage Breakdown

| Component | Depth Usage | Replaceability |
|-----------|-------------|----------------|
| NavDP (System 1) | RGB+Depth encoder (DepthAnything V2) | Medium - used for goal-conditioned trajectory |
| InternVLA-N1 (System 2) | Async mode uses depth for trajectory generation | Low - core to VLM reasoning |
| Collision Controller | Top-down depth for occupancy map | **High - potential grid cell replacement** |
| Evaluation | Depth for visualization | Low priority |

---

## Task 2: Integration Points for Grid Cells

### System 1 (NavDP/Diffusion Policy) Conditioning

```json
{
  "injection_points": [
    {
      "location": "internnav/model/basemodel/navdp/navdp_policy.py:159-170",
      "current_code": "def predict_noise(self, last_actions, timestep, goal_embed, rgbd_embed):\n    ...\n    cond_embedding = torch.cat([time_embeds, goal_embed, goal_embed, goal_embed, rgbd_embed], dim=1)",
      "proposed_modification": "Add grid_code to conditioning:\ncond_embedding = torch.cat([time_embeds, goal_embed, goal_embed, goal_embed, rgbd_embed, grid_code], dim=1)",
      "feasibility": "medium",
      "reason": "Requires modifying positional encoding size and decoder architecture"
    },
    {
      "location": "internnav/model/basemodel/rdp/rdp_policy.py:199-209",
      "current_code": "# Init the IMU encoder\nif self.model_config.imu_encoder.use:\n    self.imu_linear = nn.Linear(...)",
      "proposed_modification": "Add GridCellEncoder similar to IMU encoder:\nif self.model_config.grid_cell_encoder.use:\n    self.grid_cell_network = GridCellEncoder(n_units=512, ...)",
      "feasibility": "easy",
      "reason": "RDP already has IMU encoder infrastructure - grid cells can follow same pattern"
    },
    {
      "location": "internnav/model/basemodel/internvla_n1/internvla_n1.py:349-432",
      "current_code": "def generate_traj(self, traj_latents, images_dp, depths_dp, ...):\n    # Uses traj_latents from VLM as conditioning",
      "proposed_modification": "Add grid_code to latent conditioning:\nhidden_states = torch.cat([memory_tokens, traj_latents, grid_code], dim=1)",
      "feasibility": "medium",
      "reason": "Trajectory generation conditioned on VLM latents - grid code adds path integration info"
    }
  ],
  "velocity_availability": true,
  "velocity_sources": [
    {
      "file": "internnav/model/utils/vln_utils.py:181-182",
      "info": "S1Output dataclass has linear_velocity and angular_velocity fields (Optional)"
    },
    {
      "file": "internnav/env/utils/agilex_extensions/control.py:6,16",
      "info": "ROS odometry subscriber: /ranger_base_node/odom"
    },
    {
      "file": "internnav/dataset/rdp_lerobot_dataset.py:267-273",
      "info": "IMU data computed from global positions with local coordinate transformation"
    }
  ],
  "estimated_code_changes": "~300 lines / 5-7 files"
}
```

### RDP IMU Encoder as Template

The existing IMU encoder in `rdp_policy.py` provides an excellent template for grid cell integration:

```python
# Current IMU encoder pattern (lines 199-209):
if self.model_config.imu_encoder.use:
    self.imu_linear = nn.Linear(
        self.model_config.imu_encoder.input_size,  # 2 or 3 (x, y, yaw)
        self.model_config.imu_encoder.encoding_size,
    )
    concat_size += self.model_config.imu_encoder.encoding_size

# Proposed Grid Cell encoder (analogous):
if self.model_config.grid_cell_encoder.use:
    self.grid_cell_network = GridCellNetwork(
        n_hd_cells=self.model_config.grid_cell_encoder.n_hd_cells,  # 1275
        n_grid_cells=self.model_config.grid_cell_encoder.n_grid_cells,  # 512
        velocity_dim=self.model_config.grid_cell_encoder.velocity_dim,  # 3
    )
    concat_size += self.model_config.grid_cell_encoder.encoding_size
```

---

## Task 3: Replacement Feasibility Assessment

```json
{
  "replacement_scenarios": [
    {
      "name": "Full Replacement",
      "description": "Remove occupancy map entirely, use grid cells for all spatial reasoning",
      "feasibility_score": 0.3,
      "pros": [
        "No depth sensor dependency",
        "Works with IMU/odometry only",
        "Mathematically elegant path integration"
      ],
      "cons": [
        "CRITICAL: No explicit obstacle information",
        "Cannot detect dynamic obstacles",
        "Collision avoidance severely impacted",
        "Requires complete controller rewrite"
      ],
      "recommended": false
    },
    {
      "name": "Partial Replacement - Localization Only",
      "description": "Grid cells for self-localization and progress tracking, keep occupancy for obstacles",
      "feasibility_score": 0.75,
      "pros": [
        "Precise path integration for distance traveled",
        "Robust to visual aliasing",
        "Maintains collision safety",
        "Incremental integration possible"
      ],
      "cons": [
        "Still requires depth for obstacles",
        "Moderate code complexity",
        "Need to fuse two spatial representations"
      ],
      "recommended": true
    },
    {
      "name": "Parallel Enhancement",
      "description": "Add grid cells alongside occupancy map without removing anything",
      "feasibility_score": 0.85,
      "pros": [
        "Lowest risk approach",
        "Can A/B test improvement",
        "No breaking changes",
        "Grid code as auxiliary conditioning signal"
      ],
      "cons": [
        "Increased computation (minimal)",
        "Code complexity increase",
        "May have redundant representations"
      ],
      "recommended": true
    }
  ],
  "recommended_approach": "Parallel Enhancement → Partial Replacement",
  "critical_blockers": [
    "Grid cells cannot replace collision detection - they have no obstacle awareness",
    "Depth sensor still needed for dynamic obstacle detection in real robot deployment",
    "System 1's diffusion policy was trained without grid cell conditioning - fine-tuning required"
  ]
}
```

### Function Mapping: Occupancy Map vs Grid Cells

| Function | Occupancy Map | Grid Cells | Feasibility |
|----------|---------------|------------|-------------|
| Collision detection | ✅ Explicit obstacle map | ❌ No obstacle info | **Low** |
| Self-localization | ❌ Requires visual matching | ✅ Path integration | **High** |
| Progress tracking | ❌ Indirect via GPS | ✅ Precise distance | **High** |
| Depth-free operation | ❌ Requires depth | ✅ Only velocity/IMU | **High** |
| Vector navigation | ❌ Not supported | ✅ Goal vector computation | **High** |
| Loop closure | ❌ Not supported | ✅ Visual calibration | **Medium** |

---

## Task 4: Implementation Complexity Estimate

```json
{
  "implementation_options": [
    {
      "approach": "Zeng's Explicit Formula (CAN-based, no training)",
      "complexity": "Medium",
      "estimated_lines": 400,
      "estimated_files": 4,
      "components": [
        "GridCellNetwork class (~150 lines)",
        "HeadDirectionNetwork class (~100 lines)", 
        "PathIntegrator class (~100 lines)",
        "Config additions (~50 lines)"
      ],
      "pros": [
        "No training required",
        "Mathematically interpretable",
        "Deterministic behavior",
        "Lightweight (512 + 1275 units)"
      ],
      "cons": [
        "Fixed architecture",
        "May not adapt to novel environments",
        "Requires careful parameter tuning"
      ],
      "reference": "Equations in Zeng et al. 2017 Section III"
    },
    {
      "approach": "DeepMind's Learned Grid Cells (LSTM + Dropout)",
      "complexity": "High", 
      "estimated_lines": 800,
      "estimated_files": 6,
      "components": [
        "GridCellLSTM class (~200 lines)",
        "PlaceCellDecoder class (~100 lines)",
        "HeadDirectionDecoder class (~100 lines)",
        "Training loop (~300 lines)",
        "Dataset preparation (~100 lines)"
      ],
      "pros": [
        "End-to-end trainable",
        "Adaptive to environment",
        "Can emerge optimal representations"
      ],
      "cons": [
        "Requires trajectory dataset for training",
        "Black box (less interpretable)",
        "Higher computational cost"
      ],
      "reference": "Banino et al. 2018 'Vector-based navigation using grid-like representations'"
    },
    {
      "approach": "Hybrid: Explicit HD + Learned Grid",
      "complexity": "Medium-High",
      "estimated_lines": 600,
      "estimated_files": 5,
      "pros": [
        "HD cells from formula (stable)",
        "Grid cells learned (adaptive)",
        "Best of both worlds"
      ],
      "cons": [
        "Architecture complexity",
        "Partial training requirement"
      ],
      "recommended": true
    }
  ],
  "recommended_option": "Zeng's Explicit Formula for initial integration, with option to upgrade to learned grid cells later",
  "estimated_timeline": {
    "phase1_grid_cell_module": "3-5 days",
    "phase2_rdp_integration": "2-3 days", 
    "phase3_testing_validation": "2-3 days",
    "phase4_navdp_integration": "3-5 days",
    "total": "2-3 weeks"
  }
}
```

---

## Proposed Implementation: GridCellEncoder Module

### Core Architecture

```python
# Proposed: internnav/model/encoder/grid_cell_encoder.py

import torch
import torch.nn as nn
import numpy as np

class GridCellNetwork(nn.Module):
    """
    Grid Cell Network based on Continuous Attractor Network (CAN).
    Implements path integration using velocity signals.
    
    Reference: Zeng et al. 2017 - Cognitive Mapping Based on Conjunctive 
    Representation of Grid Cells and Head Direction Cells
    
    Architecture:
    - N grid cells with multi-scale hexagonal firing patterns
    - M head direction cells for orientation encoding  
    - Path integration via velocity-weighted connections
    """
    
    def __init__(
        self,
        n_grid_cells: int = 512,
        n_hd_cells: int = 1275,  # 255 * 5 scales
        grid_scales: list = [0.5, 1.0, 2.0, 4.0],
        velocity_dim: int = 3,  # (linear_vel, angular_vel, heading)
        output_dim: int = 128,
    ):
        super().__init__()
        self.n_grid_cells = n_grid_cells
        self.n_hd_cells = n_hd_cells
        self.grid_scales = grid_scales
        
        # Head direction attractor network (ring attractor)
        self.hd_weights = self._init_hd_weights()
        
        # Grid cell weights per scale (2D sheet attractor)
        self.grid_weights = nn.ParameterList([
            nn.Parameter(self._init_grid_weights(scale), requires_grad=False)
            for scale in grid_scales
        ])
        
        # Velocity to neural activity projection
        self.velocity_encoder = nn.Sequential(
            nn.Linear(velocity_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_hd_cells + n_grid_cells)
        )
        
        # Output projection for conditioning
        self.output_proj = nn.Linear(n_grid_cells + n_hd_cells, output_dim)
        
        # Internal state
        self.register_buffer('hd_state', torch.zeros(1, n_hd_cells))
        self.register_buffer('grid_state', torch.zeros(1, n_grid_cells))
        
    def _init_hd_weights(self):
        """Initialize ring attractor weights for head direction cells."""
        # Cosine connectivity for ring attractor
        n = self.n_hd_cells
        i = torch.arange(n).unsqueeze(0).float()
        j = torch.arange(n).unsqueeze(1).float()
        phase_diff = 2 * np.pi * (i - j) / n
        weights = torch.cos(phase_diff) - 0.5  # Shifted cosine
        return nn.Parameter(weights, requires_grad=False)
    
    def _init_grid_weights(self, scale):
        """Initialize 2D sheet attractor weights for grid cells."""
        n = int(np.sqrt(self.n_grid_cells // len(self.grid_scales)))
        # Hexagonal connectivity pattern
        weights = torch.zeros(n * n, n * n)
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = (i + di) % n, (j + dj) % n
                        nidx = ni * n + nj
                        dist = np.sqrt(di**2 + dj**2)
                        weights[idx, nidx] = np.exp(-dist / scale)
        return weights
        
    def update(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Update grid cell and HD cell states based on velocity.
        
        Args:
            velocity: (batch, 3) tensor of [linear_vel, angular_vel, heading]
            
        Returns:
            grid_code: (batch, output_dim) encoded spatial representation
        """
        batch_size = velocity.shape[0]
        
        # Expand state for batch
        if self.hd_state.shape[0] != batch_size:
            self.hd_state = self.hd_state.expand(batch_size, -1).clone()
            self.grid_state = self.grid_state.expand(batch_size, -1).clone()
        
        # Encode velocity to neural activity shifts
        vel_encoded = self.velocity_encoder(velocity)
        hd_shift = vel_encoded[:, :self.n_hd_cells]
        grid_shift = vel_encoded[:, self.n_hd_cells:]
        
        # Update HD state (ring attractor dynamics)
        hd_input = torch.matmul(self.hd_state, self.hd_weights) + hd_shift
        self.hd_state = torch.softmax(hd_input, dim=-1)
        
        # Update grid state (sheet attractor dynamics)
        grid_input = self.grid_state.clone()
        for weights in self.grid_weights:
            grid_input = grid_input + torch.matmul(grid_input, weights)
        grid_input = grid_input + grid_shift
        self.grid_state = torch.sigmoid(grid_input)
        
        # Combine and project
        combined = torch.cat([self.grid_state, self.hd_state], dim=-1)
        grid_code = self.output_proj(combined)
        
        return grid_code
    
    def reset(self, batch_size: int = 1):
        """Reset internal states for new episode."""
        self.hd_state = torch.zeros(batch_size, self.n_hd_cells, 
                                     device=self.hd_state.device)
        self.grid_state = torch.zeros(batch_size, self.n_grid_cells,
                                       device=self.grid_state.device)
        
    def get_position_estimate(self) -> torch.Tensor:
        """Decode estimated position from grid cell population."""
        # Population vector decoding
        # Returns (batch, 2) estimated [x, y] position
        grid_2d = self.grid_state.view(-1, 
            int(np.sqrt(self.n_grid_cells // len(self.grid_scales))),
            int(np.sqrt(self.n_grid_cells // len(self.grid_scales))),
            len(self.grid_scales)
        )
        # Compute center of mass for position estimate
        positions = []
        for scale_idx, scale in enumerate(self.grid_scales):
            activity = grid_2d[..., scale_idx]
            n = activity.shape[1]
            x_coords = torch.arange(n, device=activity.device).float()
            y_coords = torch.arange(n, device=activity.device).float()
            x_pos = (activity * x_coords.view(1, 1, -1)).sum(dim=(1, 2)) / activity.sum(dim=(1, 2))
            y_pos = (activity * y_coords.view(1, -1, 1)).sum(dim=(1, 2)) / activity.sum(dim=(1, 2))
            positions.append(torch.stack([x_pos * scale, y_pos * scale], dim=-1))
        return torch.stack(positions, dim=1).mean(dim=1)
```

### Integration with RDP Policy

```python
# Modified: internnav/model/basemodel/rdp/rdp_policy.py

# Add to imports:
from internnav.model.encoder.grid_cell_encoder import GridCellNetwork

# Add to __init__ (after IMU encoder, ~line 209):
if self.model_config.grid_cell_encoder.use:
    self.grid_cell_network = GridCellNetwork(
        n_grid_cells=self.model_config.grid_cell_encoder.n_grid_cells,
        n_hd_cells=self.model_config.grid_cell_encoder.n_hd_cells,
        velocity_dim=self.model_config.grid_cell_encoder.velocity_dim,
        output_dim=self.model_config.grid_cell_encoder.encoding_size,
    )
    self.grid_linear = nn.Linear(
        self.model_config.grid_cell_encoder.encoding_size,
        self.model_config.grid_cell_encoder.encoding_size,
    )
    concat_size += self.model_config.grid_cell_encoder.encoding_size

# Add to forward (after IMU processing, ~line 413):
if self.model_config.grid_cell_encoder.use:
    velocity = observations['velocity']  # (batch, 3)
    grid_code = self.grid_cell_network.update(velocity)
    grid_embeds = self.grid_linear(grid_code)
    concat_embeds = torch.cat([concat_embeds, grid_embeds], dim=1)
```

---

## Risk Assessment

### What Could Go Wrong

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Grid cells drift without visual correction | High | Medium | Implement periodic visual recalibration using place cell matching |
| Collision accidents in real robot | High | **Critical** | Never remove occupancy-based collision detection; grid cells are supplementary |
| Incompatible with existing checkpoints | Medium | Medium | Implement as optional module with backward compatibility |
| Training instability | Medium | Low | Use frozen explicit formulas initially, train only projection layers |
| Velocity signal noise | Medium | Medium | Low-pass filtering, use odometry + IMU fusion |

### Dependencies

1. **Velocity Signals**: Requires linear and angular velocity from:
   - Simulation: Available from environment state
   - Real Robot: ROS odometry (`/ranger_base_node/odom`) already subscribed

2. **Configuration System**: Extend `ModelCfg` in `base_encoders.py` with `GridCellEncoder` config

3. **Dataset**: If using learned grid cells, need trajectory dataset with velocity annotations

### Fallback Options

1. **Feature Flag**: `model.grid_cell_encoder.use = False` disables entire module
2. **Gradual Integration**: Start with grid cells as auxiliary loss, not primary conditioning
3. **A/B Testing**: Run with and without grid cells in evaluation to measure impact

---

## Summary and Recommendations

### Key Answers

✅ **Can cognitive mapping replace occupancy map?**
- **Partial Yes**: For self-localization, progress tracking, and vector navigation
- **No**: For collision detection - grid cells have no obstacle awareness

✅ **Concrete integration points identified?**
- RDP policy has IMU encoder template (easiest integration)
- NavDP conditioning can be extended
- InternVLA-N1 trajectory generation supports additional conditioning

✅ **Implementation complexity?**
- Medium: ~400-600 lines, 4-5 files
- Timeline: 2-3 weeks for full integration with testing

### Recommended Next Steps

1. **Phase 1 (Week 1)**: Implement `GridCellEncoder` module with explicit formulas
2. **Phase 2 (Week 1-2)**: Integrate with RDP policy using IMU encoder pattern
3. **Phase 3 (Week 2)**: Add velocity preprocessing and test in simulation
4. **Phase 4 (Week 3)**: Extend to NavDP and InternVLA-N1 conditioning
5. **Phase 5 (Future)**: Optional upgrade to learned grid cells

### Critical Constraints

- ⚠️ **Never remove depth-based collision detection** for real robot safety
- ⚠️ **Grid cells are supplementary**, not replacement for spatial reasoning
- ⚠️ **Velocity signal quality** is crucial - ensure proper filtering

---

## Appendix: Configuration Schema

```python
# Add to internnav/configs/model/base_encoders.py

class GridCellEncoder(BaseModel, extra='allow'):
    """Configuration for Grid Cell Network encoder."""
    use: bool = False
    n_grid_cells: int = 512
    n_hd_cells: int = 1275
    velocity_dim: int = 3  # (linear_vel, angular_vel, heading)
    encoding_size: int = 128
    grid_scales: list = [0.5, 1.0, 2.0, 4.0]
    use_visual_calibration: bool = True
    calibration_interval: int = 100  # steps between visual corrections
```

---

*Analysis completed: 2026-01-30*
*Author: Copilot Agent*
*Reference: Zeng et al. 2017, Banino et al. 2018*
