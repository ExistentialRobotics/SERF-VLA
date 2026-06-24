"""Training configuration for BEHAVIOR-1K challenge.

Reference: https://github.com/Physical-Intelligence/openpi
"""

import abc
import dataclasses
import difflib
import logging
import pathlib
from typing import TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

# Import from OpenPI
import openpi.models.model as _model
import openpi.training.optimizer as _optimizer
import openpi.transforms as _transforms

# Import from B1K custom modules
from b1k import transforms as b1k_transforms
from b1k.training.config import (
    DataConfig,
    DataConfigFactory,
    LeRobotB1KDataConfig,
    ModelTransformFactory,
    TrainConfig as BaseTrainConfig,
)

# Import from Map B1K custom modules
from serf_b1k.models import pi_serf_behavior_config
from serf_b1k.policies import b1k_policy
from serf_b1k.training import weight_loaders
from serf_b1k.utils.task_metadata import (
    extract_task_id_from_config_name,
    get_task_point_category_settings,
    normalize_task_id,
)

ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter


def _create_repack_mapping(model_config: _model.BaseModelConfig) -> dict[str, str]:
    """Create train-time observation repack mapping for map inputs."""
    repack_mapping = {
        "observation/egocentric_camera": "observation.images.rgb.head",
        "observation/wrist_image_left": "observation.images.rgb.left_wrist",
        "observation/wrist_image_right": "observation.images.rgb.right_wrist",
        "observation/state": "observation.state",
        "actions": "action",
        "task_index": "task_index",  # Always preserve task_index
        "timestamp": "timestamp",    # Preserve timestamp for subtask state computation
        "episode_index": "episode_index",  # Preserve episode_index for episode length lookup
        "index": "index",           # Preserve index
        # Add 3d point inputs
        "observation/points/xyz": "observation.points.xyz",
        "observation/points/rgb": "observation.points.rgb",
        "observation/points/feat": "observation.points.feat",
        "observation/points/pc_norm_offset": "observation.points.pc_norm_offset",
        "observation/points/pc_norm_scale": "observation.points.pc_norm_scale",
    }

    if getattr(model_config, "map_input_type", None) == "4d_env_robot_feat_map":
        repack_mapping["observation/points/robot_mask"] = (
            "observation.points.robot_mask"
        )

    return repack_mapping


@dataclasses.dataclass(frozen=True)
class CustomDataConfig(DataConfig):
    """Custom Data configuration for BEHAVIOR-1K dataset."""

    # Custom data config options
    map_input_type: str | None = None
    map_dataset_root_path: str | None = None
    robot_model_root_path: str | None = None

    # Instance filtering: two-group sampling.
    # keep_all: ALL points from these categories are kept (no subsampling).
    # budget: remaining point budget filled by equal sampling from these categories.
    # If None, all instances are used with equal sampling (default behavior).
    instance_keep_all_categories: tuple[str, ...] | None = None
    instance_budget_categories: tuple[str, ...] | None = None

@dataclasses.dataclass(frozen=True)
class CustomLeRobotB1KDataConfig(LeRobotB1KDataConfig):

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack transforms for B1K observations
        repack_mapping = _create_repack_mapping(model_config)
            
        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_mapping)]
        )

        # Prepare data for policy training
        data_transforms = _transforms.Group(
            inputs=[b1k_policy.B1kInputs(model_type=model_config.model_type)],
            outputs=[b1k_policy.B1kOutputs()],
        )

        # Delta action transforms
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(-3, 3, -1, 7, -1, 7, -1)
        else:
            delta_action_mask = _transforms.make_bool_mask(-23)
        
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms (subtask state, task ID, padding)
        model_transforms = ModelTransformFactory()(model_config)
        
        # FAST tokenization (if enabled for PiSerfBehavior)
        if self.use_fast_tokenization and hasattr(model_config, 'use_fast_auxiliary') and model_config.use_fast_auxiliary:
            asset_id = self.assets.asset_id or self.repo_id
            tokenizer_path = assets_dirs / asset_id / "fast_tokenizer"
            
            # Get base config to access norm_stats
            base_config = self.create_base_config(assets_dirs, model_config)
            
            # Only add transform if tokenizer directory exists
            if tokenizer_path.exists():
                model_transforms = model_transforms.push(
                    inputs=[b1k_transforms.TokenizeFASTActions(
                        tokenizer_path=str(tokenizer_path),
                        encoded_dim_ranges=model_config.get_fast_dim_ranges(),
                        max_fast_tokens=model_config.max_fast_tokens,
                        norm_stats=base_config.norm_stats,
                        use_per_timestamp=base_config.use_per_timestamp_norm,
                    )],
                )
            else:
                logging.warning(
                    f"FAST tokenizer not found at {tokenizer_path}. "
                    "FAST auxiliary training will be disabled (inference mode)."
                )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig(BaseTrainConfig):
    task_id: str = tyro.MISSING

    def __post_init__(self) -> None:
        super().__post_init__()
        canonical_task_id = normalize_task_id(self.task_id)
        get_task_point_category_settings(canonical_task_id)

        if self.task_ids is not None:
            raise ValueError(
                "Map training configs are task-scoped and do not accept top-level "
                "`task_ids` overrides. Use a task-scoped preset name instead."
            )

        config_task_id = extract_task_id_from_config_name(self.name)
        if config_task_id is None:
            raise ValueError(
                f"Map config name {self.name!r} must be task-scoped and end with "
                "`--task-XXXX`."
            )
        if config_task_id != canonical_task_id:
            raise ValueError(
                f"Map config name suffix {config_task_id!r} does not match "
                f"task_id={canonical_task_id!r}."
            )

# B1K Training Configurations
def _make_lr_schedule() -> _optimizer.CosineDecaySchedule:
    return _optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=2.5e-6,
        decay_steps=20_000,
        decay_lr=1e-6,
    )


def _make_weight_loader() -> weight_loaders.PiSerfBehaviorWeightLoader:
    return weight_loaders.PiSerfBehaviorWeightLoader(
        "checkpoints/behavior-1k-solution/behavior_50t_checkpoint/params"
    )

def _make_3d_env_feat_map_config(task_id: str) -> TrainConfig:
    return TrainConfig(
        name=f"pi_serf_behavior_b1k_fast--3d_env_feat_map--50t_lora--{task_id}",
        exp_name="openpi",
        project_name="B1K",
        task_id=task_id,
        model=pi_serf_behavior_config.PiSerfBehaviorConfig(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora",
            action_horizon=30,
            action_dim=32,
            use_correlated_noise=True,
            correlation_beta=0.5,
            use_fast_auxiliary=False,
            fast_loss_weight=0.0,
            fast_encoded_dims="0:6,7:23",
            fast_vocab_size=1024,
            max_fast_tokens=200,
            use_kv_transform=True,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.0,
            freeze_vision_backbone=True,
            map_input_type="3d_env_feat_map",
            num_input_points=12421,
            use_robot_points=False,
            robot_center_radii=(1.0, 2.0, 4.0),
            ee_radius=0.5,
            robot_center_z_offset=0.8,
            local_branch_planes=256,
            local_branch_nsample=16,
        ),
        freeze_filter=pi_serf_behavior_config.PiSerfBehaviorConfig(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora",
            action_horizon=30,
            action_dim=32,
            use_correlated_noise=True,
            correlation_beta=0.5,
            use_fast_auxiliary=False,
            fast_loss_weight=0.0,
            fast_encoded_dims="0:6,7:23",
            fast_vocab_size=1024,
            max_fast_tokens=200,
            use_kv_transform=True,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.0,
            freeze_vision_backbone=True,
            map_input_type="3d_env_feat_map",
            num_input_points=12421,
            use_robot_points=False,
        ).get_freeze_filter(),
        data=CustomLeRobotB1KDataConfig(
            repo_id="IliaLarchenko/behavior_224_rgb",
            base_config=CustomDataConfig(
                prompt_from_task=False,
                use_per_timestamp_norm=True,
                behavior_dataset_root="datasets/2025-BEHAVIOR-1K-CHALLENGE",
                map_dataset_root_path="datasets/SERF-BEHAVIOR-1K-MAP",
            ),
            use_delta_joint_actions=True,
            use_fast_tokenization=True,
        ),
        lr_schedule=_make_lr_schedule(),
        num_train_steps=20_000,
        save_interval=10_000,
        num_workers=4,
        batch_size=16,
        ema_decay=None,
        num_flow_samples=15,
        weight_loader=_make_weight_loader(),
        assets_base_dir="checkpoints/behavior-1k-solution/behavior_50t_checkpoint/assets",
        checkpoint_base_dir="exps",
    )


def _make_4d_env_feat_map_config(task_id: str) -> TrainConfig:
    return TrainConfig(
        name=f"pi_serf_behavior_b1k_fast--4d_env_feat_map--50t_lora--{task_id}",
        exp_name="openpi",
        project_name="B1K",
        task_id=task_id,
        model=pi_serf_behavior_config.PiSerfBehaviorConfig(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora",
            action_horizon=30,
            action_dim=32,
            use_correlated_noise=True,
            correlation_beta=0.5,
            use_fast_auxiliary=False,
            fast_loss_weight=0.0,
            fast_encoded_dims="0:6,7:23",
            fast_vocab_size=1024,
            max_fast_tokens=200,
            use_kv_transform=True,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.0,
            freeze_vision_backbone=True,
            map_input_type="4d_env_feat_map",
            num_input_points=12421,
            use_robot_points=False,
            robot_center_radii=(1.0, 2.0, 4.0),
            ee_radius=0.5,
            robot_center_z_offset=0.8,
            local_branch_planes=256,
            local_branch_nsample=16,
        ),
        freeze_filter=pi_serf_behavior_config.PiSerfBehaviorConfig(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora",
            action_horizon=30,
            action_dim=32,
            use_correlated_noise=True,
            correlation_beta=0.5,
            use_fast_auxiliary=False,
            fast_loss_weight=0.0,
            fast_encoded_dims="0:6,7:23",
            fast_vocab_size=1024,
            max_fast_tokens=200,
            use_kv_transform=True,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.0,
            freeze_vision_backbone=True,
            map_input_type="4d_env_feat_map",
            num_input_points=12421,
            use_robot_points=False,
        ).get_freeze_filter(),
        data=CustomLeRobotB1KDataConfig(
            repo_id="IliaLarchenko/behavior_224_rgb",
            base_config=CustomDataConfig(
                prompt_from_task=False,
                use_per_timestamp_norm=True,
                behavior_dataset_root="datasets/2025-BEHAVIOR-1K-CHALLENGE",
                map_dataset_root_path="datasets/SERF-BEHAVIOR-1K-MAP",
            ),
            use_delta_joint_actions=True,
            use_fast_tokenization=True,
        ),
        lr_schedule=_make_lr_schedule(),
        num_train_steps=20_000,
        save_interval=10_000,
        num_workers=4,
        batch_size=16,
        ema_decay=None,
        num_flow_samples=15,
        weight_loader=_make_weight_loader(),
        assets_base_dir="checkpoints/behavior-1k-solution/behavior_50t_checkpoint/assets",
        checkpoint_base_dir="exps",
    )



def _make_4d_env_robot_feat_map_config(task_id: str) -> TrainConfig:
    return TrainConfig(
        name=f"pi_serf_behavior_b1k_fast--4d_env_robot_feat_map--50t_lora--{task_id}",
        exp_name="openpi",
        project_name="B1K",
        task_id=task_id,
        model=pi_serf_behavior_config.PiSerfBehaviorConfig(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora",
            action_horizon=30,
            action_dim=32,
            use_correlated_noise=True,
            correlation_beta=0.5,
            use_fast_auxiliary=False,
            fast_loss_weight=0.0,
            fast_encoded_dims="0:6,7:23",
            fast_vocab_size=1024,
            max_fast_tokens=200,
            use_kv_transform=True,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.0,
            freeze_vision_backbone=True,
            map_input_type="4d_env_robot_feat_map",
            num_input_points=25000,
            robot_center_radii=(1.0, 2.0, 4.0),
            ee_radius=0.5,
            robot_center_z_offset=0.8,
            local_branch_planes=256,
            local_branch_nsample=16,
        ),
        freeze_filter=pi_serf_behavior_config.PiSerfBehaviorConfig(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora",
            action_horizon=30,
            action_dim=32,
            use_correlated_noise=True,
            correlation_beta=0.5,
            use_fast_auxiliary=False,
            fast_loss_weight=0.0,
            fast_encoded_dims="0:6,7:23",
            fast_vocab_size=1024,
            max_fast_tokens=200,
            use_kv_transform=True,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.0,
            freeze_vision_backbone=True,
            map_input_type="4d_env_robot_feat_map",
            num_input_points=25000,
        ).get_freeze_filter(),
        data=CustomLeRobotB1KDataConfig(
            repo_id="IliaLarchenko/behavior_224_rgb",
            base_config=CustomDataConfig(
                prompt_from_task=False,
                use_per_timestamp_norm=True,
                behavior_dataset_root="datasets/2025-BEHAVIOR-1K-CHALLENGE",
                map_dataset_root_path="datasets/SERF-BEHAVIOR-1K-MAP",
                robot_model_root_path="datasets/SERF-BEHAVIOR-1K-MAP/map_models",
            ),
            use_delta_joint_actions=True,
            use_fast_tokenization=True,
        ),
        lr_schedule=_make_lr_schedule(),
        num_train_steps=20_000,
        save_interval=10_000,
        num_workers=4,
        batch_size=16,
        ema_decay=None,
        num_flow_samples=15,
        weight_loader=_make_weight_loader(),
        assets_base_dir="checkpoints/behavior-1k-solution/behavior_50t_checkpoint/assets",
        checkpoint_base_dir="exps",
    )

def _make_task_configs(task_id: str) -> tuple[TrainConfig, ...]:
    return (        
        _make_3d_env_feat_map_config(task_id),
        _make_4d_env_feat_map_config(task_id),
        _make_4d_env_robot_feat_map_config(task_id),
    )


_MAP_TASK_IDS = ("task-0021", "task-0022", "task-0026")

_CONFIGS = [
    config
    for task_id in _MAP_TASK_IDS
    for config in _make_task_configs(task_id)
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if (
        config_name.startswith("pi_serf_behavior_b1k_fast--")
        and extract_task_id_from_config_name(config_name) is None
    ):
        raise ValueError(
            f"Map config {config_name!r} must be task-scoped and end with "
            "`--task-XXXX`."
        )

    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
