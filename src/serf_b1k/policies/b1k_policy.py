"""B1K Policy Transforms

Transforms BEHAVIOR-1K observations to model format.

Reference: https://github.com/wensi-ai/openpi/blob/behavior/src/openpi/policies/b1k_policy.py

++ by Byeonghyun Pak
- Add support for feature-map inputs
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES

from b1k.policies.b1k_policy import extract_state_from_proprio, _parse_image, B1kOutputs

def make_b1k_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/egocentric_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(23),
        "prompt": "do something",
        "points_xyz": np.random.rand(1000, 3).astype(np.float32),
        "points_rgb": np.random.rand(1000, 3).astype(np.float32),
        "points_feat": np.random.rand(1000, 64).astype(np.float32),
    }

@dataclasses.dataclass(frozen=True)
class B1kInputs(transforms.DataTransformFn):
    # Determines which model will be used (not actually used in B1K, kept for compatibility)
    model_type: _model.ModelType | str = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:

        proprio_data = data["observation/state"]
        # extract joint position
        state = extract_state_from_proprio(proprio_data)
        if "actions" in data:
            action =  data["actions"]

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/egocentric_camera"])
        wrist_image_left = _parse_image(data["observation/wrist_image_left"])
        wrist_image_right = _parse_image(data["observation/wrist_image_right"])

        # For B1K, always use 3 cameras (base, left_wrist, right_wrist)
        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (base_image, wrist_image_left, wrist_image_right)
        image_masks = (np.True_, np.True_, np.True_)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = action

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
            
        # Preserve task_index for PiSerfBehavior model
        if "task_index" in data:
            inputs["task_index"] = data["task_index"]
            
        # Preserve tokenized_prompt for PiSerfBehavior model
        if "tokenized_prompt" in data:
            inputs["tokenized_prompt"] = data["tokenized_prompt"]
        if "tokenized_prompt_mask" in data:
            inputs["tokenized_prompt_mask"] = data["tokenized_prompt_mask"]
            
        # Preserve subtask_state for PiSerfBehavior model
        if "subtask_state" in data:
            inputs["subtask_state"] = data["subtask_state"]
            
        # Preserve timestamp and episode_index for subtask state computation
        if "timestamp" in data:
            inputs["timestamp"] = data["timestamp"]
        if "episode_index" in data:
            inputs["episode_index"] = data["episode_index"]
            
        # Preserve initial_actions for inpainting
        if "initial_actions" in data:
            initial_actions = data["initial_actions"]
            # Pad initial_actions from 23 dimensions to 32 dimensions (model's action_dim)
            if initial_actions.shape[-1] < 32:
                padding_dim = 32 - initial_actions.shape[-1]
                padding = np.zeros(initial_actions.shape[:-1] + (padding_dim,))
                initial_actions = np.concatenate([initial_actions, padding], axis=-1)
            inputs["initial_actions"] = initial_actions

        # Add 3d point inputs
        if "observation/points/xyz" in data:
            inputs["points_xyz"] = data["observation/points/xyz"]
        if "observation/points/rgb" in data:
            inputs["points_rgb"] = data["observation/points/rgb"]
        if "observation/points/feat" in data:
            inputs["points_feat"] = data["observation/points/feat"]
        if "observation/points/robot_mask" in data:
            inputs["points_robot_mask"] = data["observation/points/robot_mask"]

        # Raw 256D proprioceptive state for FK (used by the map tokenizer).
        # Set unconditionally — small overhead (256 floats) avoids coupling
        # data transforms to model config.
        inputs["proprio_state"] = data["observation/state"]

        # Map normalization stats (used by the map tokenizer).
        if "observation/points/pc_norm_offset" in data:
            inputs["pc_norm_offset"] = data["observation/points/pc_norm_offset"]
        if "observation/points/pc_norm_scale" in data:
            inputs["pc_norm_scale"] = data["observation/points/pc_norm_scale"]

        return inputs
