"""Data loading for BEHAVIOR-1K dataset.

Reference: https://github.com/wensi-ai/openpi/tree/behavior

++ by Byeonghyun Pak
- Add support for feature-map inputs
"""

import logging
import time

# Import all base data loading from OpenPI
from openpi.training.data_loader import (
    Dataset,
    IterableDataset,
    DataLoader,
    TransformedDataset,
    IterableTransformedDataset,
    FakeDataset,
    TorchDataLoader,
    RLDSDataLoader,
    create_torch_dataset,
    create_rlds_dataset,
    transform_iterable_dataset,
    create_data_loader,
    create_torch_data_loader,
    create_rlds_data_loader,
)

import openpi.training.config as _config
import openpi.transforms as _transforms

# from b1k.models.observation import Observation
from serf_b1k.models.observation import Observation
from serf_b1k.utils.task_catalog import select_behavior_task_names
from serf_b1k.utils.task_metadata import get_task_point_category_settings
from b1k.transforms_normalize import NormalizeWithPerTimestamp

from b1k.training.data_loader import transform_dataset, extract_episode_lengths_from_dataset

class DataLoaderImpl(DataLoader):
    """Custom DataLoader using our Observation with fast_tokens."""
    
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield Observation.from_dict(batch), batch["actions"]


def create_behavior_dataset(data_config: _config.DataConfig, action_horizon: int, seed: int | None = None, num_points: int | None = None) -> Dataset:
    """Create a BEHAVIOR-1K dataset for training.
    
    Uses OmniGibson's BehaviorLeRobotDataset for efficient loading of BEHAVIOR-1K data.
    
    Args:
        data_config: Data configuration
        action_horizon: Action horizon for delta timestamps
        seed: Random seed for shuffling. If None, uses random seed based on current time.
    
    Returns:
        Dataset instance with BEHAVIOR-1K data
    """
    from serf_b1k.datas.lerobot_serf_dataset import (
        BehaviorLeRobotDataset_3D_EnvFeatMap,
        BehaviorLeRobotDataset_4D_EnvFeatMap,
        BehaviorLeRobotDataset_4D_EnvRobotFeatMap,
    )
    
    # Use random seed if not provided
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed for BehaviorLeRobotDataset: {seed}")
    if data_config.task_ids is None:
        raise ValueError(
            "Map training requires exactly one resolved dataset task. "
            "Use a task-scoped map preset instead of loading all tasks."
        )
    if len(data_config.task_ids) != 1:
        raise ValueError(
            f"Map training requires exactly one dataset task, got {data_config.task_ids!r}."
        )

    target_tasks = select_behavior_task_names(data_config.task_ids)

    dataset_cls = None
    args = {}

    if data_config.map_input_type == "3d_env_feat_map":
        dataset_cls = BehaviorLeRobotDataset_3D_EnvFeatMap
        args["map_input_type"] = "3d_env_feat_map"

    elif data_config.map_input_type == "4d_env_feat_map":
        dataset_cls = BehaviorLeRobotDataset_4D_EnvFeatMap

    elif data_config.map_input_type == "4d_env_robot_feat_map":
        dataset_cls = BehaviorLeRobotDataset_4D_EnvRobotFeatMap
        args["robot_model_root_path"] = data_config.robot_model_root_path

    else:
        raise ValueError(f"Unsupported map_input_type: {data_config.map_input_type}")

    if num_points is not None:
        args["num_points"] = num_points

    if getattr(data_config, "instance_keep_all_categories", None) is not None:
        args["instance_keep_all_categories"] = data_config.instance_keep_all_categories
    if getattr(data_config, "instance_budget_categories", None) is not None:
        args["instance_budget_categories"] = data_config.instance_budget_categories

    dataset = dataset_cls(
        repo_id=data_config.repo_id,
        root=data_config.behavior_dataset_root,
        tasks=target_tasks,
        modalities=["rgb"],
        local_only=True,
        delta_timestamps={
            key: [t / 30.0 for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        episodes=data_config.episodes_index,
        chunk_streaming_using_keyframe=True,
        shuffle=True,
        seed=seed,
        map_dataset_root_path=data_config.map_dataset_root_path,
        **args,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset.meta.tasks)])

    return dataset


def create_behavior_data_loader(
    config: _config.TrainConfig,
    *,
    sharding=None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader:
    """Create a data loader for BEHAVIOR-1K training."""
    import jax
    import time
    from dataclasses import replace
    
    data_config = config.data.create(config.assets_dirs, config.model)
    
    # Use random seed if not provided
    seed = config.seed
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed: {seed}")
    
    task_id = getattr(config, "task_id", None)
    if task_id is not None:
        if config.task_ids is not None:
            raise ValueError(
                "Map training configs are task-scoped and do not accept top-level "
                "`task_ids` overrides."
            )
        task_settings = get_task_point_category_settings(task_id)
        data_config = replace(
            data_config,
            task_ids=[task_settings.task_index],
            instance_keep_all_categories=task_settings.instance_keep_all_categories,
            instance_budget_categories=task_settings.instance_budget_categories,
        )
    elif config.task_ids is not None:
        data_config = replace(data_config, task_ids=config.task_ids)
    if config.model.map_input_type is not None:
        data_config = replace(data_config, map_input_type=config.model.map_input_type)

    dataset = create_behavior_dataset(
        data_config,
        action_horizon=config.model.action_horizon,
        seed=seed,
        num_points=config.model.num_input_points,
    )
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=seed,
    )
    
    return DataLoaderImpl(data_config, data_loader)
