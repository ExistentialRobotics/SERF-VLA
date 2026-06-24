"""Weight loaders for PiSerfBehavior model initialization from Pi05 checkpoints.

Reference: https://github.com/Physical-Intelligence
"""

import dataclasses
import logging
import pathlib
import re

import flax.traverse_util
import numpy as np
import orbax.checkpoint as ocp

import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at
import openpi.shared.download as download

# Re-export base loaders from OpenPI
from openpi.training.weight_loaders import (
    WeightLoader,
    NoOpWeightLoader,
    CheckpointWeightLoader,
    _merge_params,
)

logger = logging.getLogger(__name__)

def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = download.maybe_download(params_path)

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Load directly with PyTreeCheckpointer (handles both old and new checkpoint formats)
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype),
                    item,
                ),
            ),
        )["params"]

    return params

@dataclasses.dataclass(frozen=True)
class PiSerfBehaviorWeightLoader(WeightLoader):
    """Loads checkpoints for PiSerfBehavior model.
    
    Automatically detects:
    - Pi05 checkpoint: Loads weights, preserves new PiSerfBehavior parameters
    - PiSerfBehavior checkpoint: Loads all weights directly
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = restore_params(self.params_path)
        
        # Remove 'value' suffixes (from nnx.State format)
        flat_params = flax.traverse_util.flatten_dict(loaded_params)
        if all(kp[-1] == "value" for kp in flat_params if len(kp) > 0):
            flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
            loaded_params = flax.traverse_util.unflatten_dict(flat_params)
        
        # Detect checkpoint type
        has_task_embeddings = 'task_embeddings' in loaded_params
        
        if has_task_embeddings:
            # Loading PiSerfBehavior checkpoint - load ALL weights from checkpoint
            logging.info("Loading PiSerfBehavior checkpoint (all weights will be loaded)")
            # Use _merge_params with empty missing_regex to validate shapes
            return _merge_params(loaded_params, params, missing_regex="^$")
        else:
            # Loading Pi05 checkpoint - preserve new PiSerfBehavior-specific parameters
            logging.info("Loading Pi05 checkpoint (new PiSerfBehavior parameters will use random init)")
            
            # These parameters are NEW in PiSerfBehavior (not in Pi05), so keep them from params (random init)
            missing_regex = (
                ".*task_embeddings.*|"
                ".*stage_pred_from_vlm.*|"
                ".*task_stage_embeddings.*|"
                ".*gate_sincos.*|"
                ".*gate_task_stage.*|"
                ".*gate_task.*|"
                ".*fusion_layer.*|"
                ".*stage_projection.*|"
                ".*task_subtask_fusion.*|"
                ".*fast_token_embedding.*|"
                ".*fast_token_proj.*|"
                ".*kv_transform.*|"
                # Exclude optional LoRA parameters
                ".*lora.*|"
                # Map tokenizer parameters are new in PiSerfBehavior and should keep their random initialization
                ".*map_tokenizer.*"
            )
            return _merge_params(loaded_params, params, missing_regex=missing_regex)
