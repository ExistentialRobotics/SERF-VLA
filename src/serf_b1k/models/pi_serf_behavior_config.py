"""PiSerfBehavior model configuration.

Configuration for the PiSerfBehavior model on BEHAVIOR-1K challenge.
"""

import dataclasses
import json
import pathlib
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

from serf_b1k.models.observation import Observation

if TYPE_CHECKING:
    from serf_b1k.models.pi_serf_behavior import PiSerfBehavior

from b1k.models.pi_behavior_config import TASK_NUM_STAGES, MAX_NUM_STAGES, TOTAL_TASK_STAGE_EMBEDDINGS, TASK_STAGE_OFFSETS

@dataclasses.dataclass(frozen=True)
class PiSerfBehaviorConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 30
    max_token_len: int = 200  # Only used for compatibility, not for actual tokenization
    
    # Number of tasks in the behavior dataset
    num_tasks: int = 50
    # Task embedding dimension - will match the paligemma width
    task_embedding_dim: int = None  # type: ignore
    # Maximum number of subtask states across all tasks
    max_num_subtask_states: int = MAX_NUM_STAGES
    
    # Path to task data JSON file for initialization
    task_data_path: str = "BEHAVIOR-1K/docs/challenge/task_data.json"
    
    # Whether to use correlated noise matching action covariance structure
    # Requires correlation matrix in norm_stats (computed by compute_norm_stats.py)
    use_correlated_noise: bool = True
    
    # Shrinkage parameter for correlation regularization
    # Applied as: S_regularized = beta * S + (1-beta) * I
    # beta=1.0 means full correlation (no shrinkage)
    # beta=0.7 means 70% correlation + 30% independence (recommended for robustness)
    # beta=0.0 means independence (no correlation)
    correlation_beta: float = 0.5
    
    # FAST auxiliary training configuration
    use_fast_auxiliary: bool = False  # Enable FAST during training
    fast_loss_weight: float = 0.1  # Weight for FAST loss (vs flow loss)
    
    # Action dimensions to encode with FAST (default: 0:6, 7:23 = 22 dims)
    # Format: "0:6,7:23" or list of tuples [(0, 6), (7, 23)]
    fast_encoded_dims: str | list[tuple[int, int]] = "0:6,7:23"
    
    # FAST tokenizer vocab size
    fast_vocab_size: int = 1024
    
    # Max FAST tokens to predict (truncate if exceeded)
    max_fast_tokens: int = 32
    
    # FAST tokenizer path (set during initialization, relative to assets_dir/asset_id)
    fast_tokenizer_path: str | None = None
    
    # KV cache transformation for cross-layer attention between VLM and action expert
    # Allows each action expert layer to attend to a learned combination of all VLM layers
    use_kv_transform: bool = True
    
    # Knowledge insulation: stop action expert gradients from flowing to VLM backbone
    # VLM trains on FAST tokens only, action expert on flow matching with frozen VLM features
    # Implements approach from https://www.physicalintelligence.company/research/knowledge_insulation
    use_knowledge_insulation: bool = True
    
    # Subtask/stage prediction auxiliary loss weight (relative to action loss)
    # Higher values emphasize stage prediction accuracy at the expense of action quality
    subtask_loss_weight: float = 0.1
    
    # Time threshold for inpainting during inference
    # Stop enforcing inpainting constraint when t < threshold (let model be free in final steps)
    time_threshold_inpaint: float = 0.3
    
    # Vision backbone finetuning control
    freeze_vision_backbone: bool = True

    # Number of input points for point map inputs
    num_input_points: int = 24576

    # 3D input type ['3d_env_feat_map' | '4d_env_feat_map' | '4d_env_robot_feat_map']
    map_input_type: str | None = None

    # Map tokenizer config
    # True: 8 branches (3 robot-center + 1 global + 2 EE + 1 robot-only + 1 env-only)
    # False: 6 branches (3 robot-center + 2 EE + 1 env-only) — global/robot-only removed
    use_robot_points: bool = True
    robot_center_radii: tuple[float, ...] = (1.0, 2.0, 4.0)
    ee_radius: float = 0.5
    robot_center_z_offset: float = 0.8
    local_branch_planes: int = 256
    local_branch_nsample: int = 16
    # Map tokenizer backbone config
    ptl_enc_planes: tuple[int, ...] = (128, 256)
    ptl_enc_blocks: tuple[int, ...] = (2, 2)
    ptl_enc_strides: tuple[int, ...] = (4, 4)
    ptl_enc_nsample: tuple[int, ...] = (16, 16)

    def __post_init__(self):
        if self.task_embedding_dim is None:
            paligemma_config = _gemma.get_config(self.paligemma_variant)
            object.__setattr__(self, "task_embedding_dim", paligemma_config.width)
    
    def get_fast_dim_ranges(self) -> list[tuple[int, int]]:
        """Parse fast_encoded_dims into list of ranges."""
        if isinstance(self.fast_encoded_dims, str):
            ranges = []
            for range_str in self.fast_encoded_dims.split(','):
                start, end = map(int, range_str.strip().split(':'))
                ranges.append((start, end))
            return ranges
        return self.fast_encoded_dims
    
    def get_total_fast_dims(self) -> int:
        """Get total number of dimensions encoded by FAST."""
        return sum(end - start for start, end in self.get_fast_dim_ranges())

    @property
    @override
    def model_type(self):
        return "pi_serf_behavior"

    @override
    def create(self, rng: at.KeyArrayLike) -> _model.BaseModel:
        from serf_b1k.models.pi_serf_behavior import PiSerfBehavior
        return PiSerfBehavior(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple["Observation", _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            obs_kwargs = {
                "images": {
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                "image_masks": {
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                "state": jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                "tokenized_prompt": jax.ShapeDtypeStruct([batch_size, 2], jnp.int32),
                "tokenized_prompt_mask": jax.ShapeDtypeStruct([batch_size, 2], bool),
            }
            
            if self.use_fast_auxiliary:
                obs_kwargs["fast_tokens"] = jax.ShapeDtypeStruct([batch_size, self.max_fast_tokens], jnp.int32)
                obs_kwargs["fast_token_mask"] = jax.ShapeDtypeStruct([batch_size, self.max_fast_tokens], bool)
            
            n_pts = self.num_input_points
            obs_kwargs["points_xyz"] = jax.ShapeDtypeStruct([batch_size, n_pts, 3], jnp.float32)
            if self.map_input_type in ("3d_env_feat_map", "4d_env_feat_map", "4d_env_robot_feat_map"):
                obs_kwargs["points_rgb"] = None
                obs_kwargs["points_feat"] = jax.ShapeDtypeStruct([batch_size, n_pts, 64], jnp.float32)
            else:
                raise ValueError(f"Unsupported model type: {self.map_input_type}")
            
            obs_kwargs["proprio_state"] = jax.ShapeDtypeStruct([batch_size, 256], jnp.float32)
            obs_kwargs["pc_norm_offset"] = jax.ShapeDtypeStruct([batch_size, 3], jnp.float32)
            obs_kwargs["pc_norm_scale"] = jax.ShapeDtypeStruct([batch_size, 1], jnp.float32)
            if self.use_robot_points:
                obs_kwargs["points_robot_mask"] = jax.ShapeDtypeStruct(
                    [batch_size, n_pts], jnp.bool_
                )

            observation_spec = Observation(**obs_kwargs)
        
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []

        # Do not freeze map tokenizer
        filters.append(
            nnx.Not(nnx_utils.PathRegex(".*map_tokenizer.*")),
        )

        if "lora" in self.paligemma_variant or "lora" in self.action_expert_variant:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )

        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
