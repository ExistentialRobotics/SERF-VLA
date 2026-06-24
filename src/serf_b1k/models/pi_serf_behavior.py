"""The main model for BEHAVIOR-1K challenge.

Based on Pi0.5 implementation from PhysicalIntelligence/openpi

++ Extends to support feature-map inputs.
"""

import logging
import pathlib

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import gemma as _gemma
from openpi.models import siglip as _siglip
from openpi.models.pi0 import make_attn_mask, posemb_sincos
from openpi.shared import array_typing as at

from b1k.models.pi_behavior import PiBehavior

from serf_b1k.models.observation import Observation, preprocess_observation
from serf_b1k.models.point_transformer_local import MapTokenizer
from serf_b1k.models import pi_serf_behavior_config
from serf_b1k.models.pi_serf_behavior_config import (
    TASK_NUM_STAGES, 
    MAX_NUM_STAGES, 
    TOTAL_TASK_STAGE_EMBEDDINGS, 
    TASK_STAGE_OFFSETS
)


logger = logging.getLogger("b1k")

# Module-level FK singleton (not stored on model to avoid NNX pytree issues)
_ROBOT_FK_INSTANCE = None

def _get_robot_fk(z_offset: float = 0.8):
    """Lazy-init module-level FK singleton."""
    global _ROBOT_FK_INSTANCE
    if _ROBOT_FK_INSTANCE is None:
        from serf_b1k.models.robot_fk_jax import RobotFKJax
        urdf_candidates = [
            pathlib.Path("BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf"),
        ]
        for p in urdf_candidates:
            if p.exists():
                _ROBOT_FK_INSTANCE = RobotFKJax(p, z_offset=z_offset)
                break
        if _ROBOT_FK_INSTANCE is None:
            raise FileNotFoundError(
                "R1Pro URDF not found. Searched: " + ", ".join(str(p) for p in urdf_candidates)
            )
    else:
        assert _ROBOT_FK_INSTANCE.z_offset == z_offset, (
            f"FK singleton z_offset mismatch: existing={_ROBOT_FK_INSTANCE.z_offset}, "
            f"requested={z_offset}. Cannot change z_offset after first init."
        )
    return _ROBOT_FK_INSTANCE


class PiSerfBehavior(PiBehavior):

    @override
    def __init__(self, config: pi_serf_behavior_config.PiSerfBehaviorConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        logger.info("Initialized PiSerfBehavior model")

        self.map_input_type = config.map_input_type

        if config.map_input_type not in (
            "3d_env_feat_map",
            "4d_env_feat_map",
            "4d_env_robot_feat_map",
        ):
            raise ValueError(f"Unsupported model_name: {config.map_input_type}")

        # Eagerly init the module-level FK singleton.
        _get_robot_fk(z_offset=config.robot_center_z_offset)

        map_tokenizer = MapTokenizer(
            map_input_dim=64,
            token_dim=2048,
            enc_planes=config.ptl_enc_planes,
            enc_blocks=config.ptl_enc_blocks,
            enc_strides=config.ptl_enc_strides,
            enc_nsample=config.ptl_enc_nsample,
            robot_center_radii=config.robot_center_radii,
            ee_radius=config.ee_radius,
            local_branch_planes=config.local_branch_planes,
            local_branch_nsample=config.local_branch_nsample,
            num_input_points=config.num_input_points,
            use_robot_points=config.use_robot_points,
        )

        self.map_tokenizer = nnx_bridge.ToNNX(map_tokenizer)

        # Lazy init for Linen -> NNX bridge.
        fake_obs = config.fake_obs()
        lazy_kwargs = {}
        if config.use_robot_points:
            lazy_kwargs["robot_mask"] = jnp.zeros(
                (1, config.num_input_points), dtype=jnp.bool_
            )
        self.map_tokenizer.lazy_init(
            fake_obs.points_xyz,
            fake_obs.points_feat,
            jnp.zeros((1, 3)),   # robot_center
            jnp.zeros((1, 3)),   # left_ee
            jnp.zeros((1, 3)),   # right_ee
            jnp.zeros((1, 3)),   # pc_norm_offset
            jnp.ones((1, 1)),    # pc_norm_scale
            rngs=rngs, training=(not self.deterministic),
            **lazy_kwargs,
        )

    @override
    def embed_prefix(
        self, 
        obs: Observation
    ) -> tuple[
        at.Float[at.Array, "b s emb"], 
        at.Bool[at.Array, "b s"], 
        at.Bool[at.Array, " s"]
    ]:
        """
        Embed prefix: images + task + state + FAST_tokens (if provided).
        
        Args:
            obs: Observation (may include fast_tokens and fast_token_mask)
            
        Returns:
            tokens, input_mask, ar_mask
        """
        input_mask = []
        ar_mask = []
        tokens = []
        
        # Embed images
        image_token_list = []
        # Respect freeze_vision_backbone config: if frozen, always use train=False
        # If not frozen, use the model's training state (self.deterministic)
        vision_train_mode = (not self.deterministic) and (not self.config.freeze_vision_backbone)
        
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=vision_train_mode)
            image_token_list.append(image_tokens)  # Store for subtask prediction

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # Image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # Add task embeddings with subtask state fusion
        if obs.tokenized_prompt is not None:
            # obs.tokenized_prompt now contains task_ids (shape: [batch_size, 2])
            task_ids = obs.tokenized_prompt[:, 0]  # Extract task_id: [batch_size]
            base_task_embedding = self.task_embeddings(task_ids)  # shape: [batch_size, embed_dim]
            
            # ALWAYS use the input subtask state - never use predicted state inside model
            if obs.tokenized_prompt.shape[1] > 1:  # If we have [task_id, subtask_state]
                subtask_state = obs.tokenized_prompt[:, 1]  # Use input subtask state
            else:
                raise ValueError("subtask_state must be provided in tokenized_prompt for PiSerfBehavior model")
            
            # Fuse task embedding with subtask state - returns [b, 4, d] with multiple representations
            fused_task_embeddings = self.fuse_task_and_subtask(base_task_embedding, task_ids, subtask_state)
            
            # Create task token sequence: [base_task, task_gated, balanced_fusion, stage_dominant, pure_stage]
            task_sequence = jnp.concatenate([
                base_task_embedding[:, None, :],  # [b, 1, d] - base task token
                fused_task_embeddings              # [b, 4, d] - stage-conditioned tokens
            ], axis=1)  # [b, 5, d]
            
            tokens.append(task_sequence)
            # All task tokens are valid
            task_mask = jnp.ones((obs.tokenized_prompt.shape[0], 5), dtype=jnp.bool_)
            input_mask.append(task_mask)
            # Hierarchical attention: base task (False) then stage tokens (True, False, False, False)
            # Base task attends to images bidirectionally
            # Stage tokens attend to images+task but not vice versa
            ar_mask += [False] + [True, False, False, False]
            
        # Add state as discrete tokens (Pi05 style)
        # Discretize state into bins
        discretized_state = jnp.digitize(obs.state, bins=jnp.linspace(-1, 1, 256 + 1)[:-1]) - 1
        discretized_state = jnp.clip(discretized_state, 0, 255)  # Ensure valid range
        
        # Embed each dimension of the discretized state
        state_tokens = []
        for i in range(obs.state.shape[-1]):
            state_dim_tokens = self.PaliGemma.llm(discretized_state[:, i:i+1], method="embed")
            state_tokens.append(state_dim_tokens)
        
        if state_tokens:
            state_tokens = jnp.concatenate(state_tokens, axis=1)  # shape: [batch_size, state_dim, embed_dim]
            tokens.append(state_tokens)
            input_mask.append(jnp.ones((obs.state.shape[0], obs.state.shape[-1]), dtype=jnp.bool_))
            # State tokens have full bidirectional attention with all prefix tokens
            # (images, task, stages, and other state tokens)
            ar_mask += [False] * state_tokens.shape[1]
        
        # Map tokens (before FAST tokens)
        points_xyz = getattr(obs, "points_xyz", None)
        points_feat = getattr(obs, "points_feat", None)
        
        if self.map_input_type in ("3d_env_feat_map", "4d_env_feat_map", "4d_env_robot_feat_map"):
            points_input = points_feat
        else:
            raise ValueError(f"Unsupported model_name: {self.config.map_input_type}")

        # Compute FK positions from full proprioceptive state (not the truncated 23D state).
        robot_center, left_ee, right_ee = _get_robot_fk(
            z_offset=self.config.robot_center_z_offset
        )(obs.proprio_state)
        pc_norm_offset = obs.pc_norm_offset  # [B, 3]
        pc_norm_scale = obs.pc_norm_scale    # [B, 1]
        robot_mask = getattr(obs, "points_robot_mask", None)
        map_tokens = self.map_tokenizer(
            points_xyz, points_input,
            robot_center, left_ee, right_ee,
            pc_norm_offset, pc_norm_scale,
            training=(not self.deterministic),
            robot_mask=robot_mask,
        )  # [B, 6|8, 2048] depending on use_robot_points
        tokens.append(map_tokens)
        input_mask.append(jnp.ones((map_tokens.shape[0], map_tokens.shape[1]), dtype=jnp.bool_))
        ar_mask += [False] * map_tokens.shape[1]

        # FAST tokens (from observation if provided)
        if self.config.use_fast_auxiliary and obs.fast_tokens is not None:
            fast_tokens = obs.fast_tokens  # [B, T]
            fast_token_mask = obs.fast_token_mask  # [B, T]
            
            # Teacher forcing: shift right [BOS, tok0, tok1, ..., tok_{T-1}]
            bos_token = jnp.zeros((fast_tokens.shape[0], 1), dtype=jnp.int32)
            shifted_tokens = jnp.concatenate([bos_token, fast_tokens[:, :-1]], axis=1)
            
            # Shift mask too: [True, mask_0, mask_1, ..., mask_{T-1}]
            bos_mask = jnp.ones((fast_tokens.shape[0], 1), dtype=jnp.bool_)
            shifted_mask = jnp.concatenate([bos_mask, fast_token_mask[:, :-1]], axis=1)
            
            # Embed using FAST embedding layer (NOT Paligemma!)
            fast_token_emb = self.fast_token_embedding(shifted_tokens)  # [B, T, D]
            
            tokens.append(fast_token_emb)
            input_mask.append(shifted_mask)  # Use the actual token mask
            # Causal for FAST: ALL tokens are causal (pure autoregressive)
            ar_mask += [True] * shifted_tokens.shape[1]
        
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"],
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        
        # Pi05 style: no explicit state token in suffix (it's in prefix as discrete tokens)
        
        action_tokens = self.action_in_proj(noisy_actions)
        # Embed timestep using sine-cosine positional encoding
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        
        # Pi05 style: time MLP for adaRMS
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        action_expert_tokens = action_tokens
        adarms_cond = time_emb
        
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        
        # image/task/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_detailed_loss(
        self, rng: at.KeyArrayLike, observation: Observation, actions: _model.Actions, *, train: bool = False, num_flow_samples: int = 1
    ) -> dict[str, at.Float[at.Array, "*b"]]:
        """
        Compute detailed loss with multiple flow matching samples.
        
        Simplified approach using KV cache:
        - Compute prefix KV cache once (with FAST tokens)
        - Remove FAST tokens from cache (action expert doesn't attend to FAST)
        - Process N flow samples independently, each reusing the same cached prefix
        - Each sample has different noise and different time
        - Average losses across samples
        """
        losses = {}

        preprocess_rng, rng = jax.random.split(rng)
        observation = preprocess_observation(preprocess_rng, observation, train=train)

        batch_size = actions.shape[0]
        
        # 1. Embed prefix once (includes FAST tokens if provided in observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        
        # 2. Compute prefix KV cache
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache_full = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions_prefix
        )
        
        # 3. Predict stage from VLM output of base task token
        # Base task token is the first token after all image tokens
        # Image tokens all have ar_mask=False, task starts with ar_mask=False (base) then True (stage tokens)
        # Structure: [images (all False)] [base_task (False)] [stages (True, False, False, False)]
        # Find first True (first stage token), base task is at that index - 1
        first_stage_token_idx = jnp.argmax(prefix_ar_mask)  # Returns index of first True
        base_task_token_idx = first_stage_token_idx - 1
        base_task_output = prefix_out[:, base_task_token_idx, :]
        subtask_logits = self.stage_pred_from_vlm(base_task_output)  # [B, MAX_NUM_STAGES]
        
        # Mask out invalid stages for each task (vectorized JAX operations)
        task_ids = observation.tokenized_prompt[:, 0]  # [B]
        task_num_stages_array = jnp.array(TASK_NUM_STAGES, dtype=jnp.int32)
        task_num_stages = task_num_stages_array[task_ids]  # [B] - JAX array indexing
        stage_range = jnp.arange(MAX_NUM_STAGES)  # [15]
        valid_mask = stage_range[None, :] < task_num_stages[:, None]  # [B, 15]
        subtask_logits = jnp.where(valid_mask, subtask_logits, -jnp.inf)  # Mask invalid stages
        
        # 4. Extract FAST loss from prefix output (before removing from cache)
        fast_loss_value = 0.0
        fast_len = 0
        fast_targets = observation.fast_tokens
        fast_token_mask = observation.fast_token_mask
        
        if self.config.use_fast_auxiliary and fast_targets is not None:
            fast_len = fast_targets.shape[1]
            fast_start_idx = prefix_tokens.shape[1] - fast_len
            fast_outputs = prefix_out[:, fast_start_idx:, :]  # [B, T, D]
            
            # Project to FAST vocab
            fast_logits = self.fast_token_proj(fast_outputs)  # [B, T, vocab_size]
            
            # Cross-entropy loss with teacher forcing
            pred_logits = fast_logits  # [B, T, vocab]
            target_tokens = fast_targets  # [B, T]
            loss_mask = fast_token_mask  # [B, T]
            
            log_probs = jax.nn.log_softmax(pred_logits, axis=-1)
            target_log_probs = jnp.take_along_axis(
                log_probs,
                target_tokens[:, :, None],
                axis=-1
            ).squeeze(-1)  # [B, T]
            
            fast_token_loss = -target_log_probs  # [B, T]
            
            # Apply mask and normalize by number of valid tokens
            masked_loss = fast_token_loss * loss_mask  # [B, T]
            num_valid_tokens = jnp.maximum(jnp.sum(loss_mask, axis=-1), 1)  # [B]
            losses["fast_loss"] = jnp.sum(masked_loss, axis=-1) / num_valid_tokens  # [B]
            
            # Accuracy (only on valid tokens)
            pred_tokens = jnp.argmax(pred_logits, axis=-1)
            correct = (pred_tokens == target_tokens) * loss_mask
            losses["fast_accuracy"] = jnp.sum(correct, axis=-1) / num_valid_tokens
            
            fast_loss_value = self.config.fast_loss_weight * jnp.mean(losses["fast_loss"])
        elif fast_targets is not None:
            # FAST auxiliary is disabled but data contains FAST tokens
            raise ValueError(
                "use_fast_auxiliary=False but observation contains fast_tokens. "
                "Either enable use_fast_auxiliary in config or ensure data doesn't contain fast_tokens."
            )
        
        # 5. Remove FAST tokens from KV cache (action expert doesn't attend to FAST)
        # KV cache shape: [layers, batch, seq_len, num_kv_heads, head_dim]
        if fast_len > 0:
            cache_k, cache_v = kv_cache_full
            # Remove last fast_len tokens from sequence dimension
            cache_k = cache_k[:, :, :-fast_len, :, :]
            cache_v = cache_v[:, :, :-fast_len, :, :]
            kv_cache_for_actions = (cache_k, cache_v)
            prefix_len_for_actions = prefix_tokens.shape[1] - fast_len
            # Truncate prefix mask and ar_mask for action expert
            prefix_mask_for_actions = prefix_mask[:, :-fast_len]
            prefix_ar_mask_for_actions = prefix_ar_mask[:-fast_len]
        else:
            kv_cache_for_actions = kv_cache_full
            prefix_len_for_actions = prefix_tokens.shape[1]
            prefix_mask_for_actions = prefix_mask
            prefix_ar_mask_for_actions = prefix_ar_mask
        
        # 6. Knowledge insulation: stop gradients from action expert to VLM
        # This must happen BEFORE kv_transform so transform still receives gradients
        if self.config.use_knowledge_insulation:
            kv_cache_for_actions = jax.tree.map(jax.lax.stop_gradient, kv_cache_for_actions)
        
        # 7. Transform KV cache (after stop_gradient, so it receives action expert gradients)
        if self.kv_transform is not None:
            kv_cache_for_actions = self.kv_transform(kv_cache_for_actions)
        
        # 8. Define single flow sample processing
        def process_one_flow_sample(sample_rng):
            """Process one flow sample using the original cached prefix."""
            noise_rng, time_rng = jax.random.split(sample_rng)
            
            # Generate different noise and time for this sample
            noise = self.generate_correlated_noise(noise_rng, batch_size)
            time = jax.random.beta(time_rng, 1.5, 1, (batch_size,)) * 0.999 + 0.001
            
            # Compute noisy actions and target velocity
            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions
            
            # Embed suffix for this sample
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, time
            )
            
            # Build attention mask: suffix attends to prefix (without FAST) + itself
            # When using KV cache, mask shape should be [batch, suffix_len, prefix_len + suffix_len]
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(
                prefix_mask_for_actions, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            
            # Positions for suffix start after cached prefix
            suffix_positions = prefix_len_for_actions + jnp.cumsum(suffix_mask, axis=-1) - 1
            
            # Forward pass with cached prefix (discard returned cache - don't modify original!)
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache_for_actions,  # Original cache, reused for all samples
                adarms_cond=[None, adarms_cond]
            )
            
            # Compute velocity and loss
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            action_loss = jnp.square(v_t - u_t)  # [B, H, D]
            
            return action_loss
        
        # 9. Vectorize over N flow samples
        # Disable type checking inside vmap (jaxtyping doesn't handle traced values well)
        flow_rngs = jax.random.split(rng, num_flow_samples)
        with at.disable_typechecking():
            all_action_losses = jax.vmap(process_one_flow_sample)(flow_rngs)  # [N, B, H, D]
        
        # 10. Average over flow samples
        action_loss = jnp.mean(all_action_losses, axis=0)  # [B, H, D]
        
        # 11. Build per-dimension action losses
        # Base velocity (x,y,z)
        losses["action_loss_base_vel_x"] = jnp.mean(action_loss[..., 0], axis=-1)
        losses["action_loss_base_vel_y"] = jnp.mean(action_loss[..., 1], axis=-1)
        losses["action_loss_base_vel_z"] = jnp.mean(action_loss[..., 2], axis=-1)
        
        # Trunk joints (4)
        for i in range(4):
            losses[f"action_loss_trunk_{i}"] = jnp.mean(action_loss[..., 3+i], axis=-1)
            
        # Left arm joints (7)
        for i in range(7):
            losses[f"action_loss_left_arm_{i}"] = jnp.mean(action_loss[..., 7+i], axis=-1)
            
        # Left gripper
        losses["action_loss_left_gripper"] = jnp.mean(action_loss[..., 14], axis=-1)
        
        # Right arm joints (7)
        for i in range(7):
            losses[f"action_loss_right_arm_{i}"] = jnp.mean(action_loss[..., 15+i], axis=-1)
            
        # Right gripper
        losses["action_loss_right_gripper"] = jnp.mean(action_loss[..., 22], axis=-1)

        # Total action loss: mean over horizon (H) and action dims (D) -> [B]
        losses["action_loss"] = jnp.mean(action_loss, axis=(-2, -1))

        # 12. Add subtask loss during training
        subtask_loss_value = 0.0
        if train and observation.tokenized_prompt.shape[1] > 1:            
            ground_truth_subtask = observation.tokenized_prompt[:, 1]
            subtask_loss = -jax.nn.log_softmax(subtask_logits)[
                jnp.arange(ground_truth_subtask.shape[0]), ground_truth_subtask
            ]
            losses["subtask_loss"] = jnp.mean(subtask_loss)
            losses["subtask_accuracy"] = jnp.mean(
                jnp.argmax(subtask_logits, axis=-1) == ground_truth_subtask
            )
            subtask_loss_value = self.config.subtask_loss_weight * jnp.mean(subtask_loss)
        
        # 13. Total loss
        losses["total_loss"] = losses["action_loss"] + subtask_loss_value + fast_loss_value
        
        return losses

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 20,
        time_threshold_inpaint: float | at.Float[at.Array, ""] | None = None,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        initial_actions: at.Float[at.Array, "b n ad"] | None = None,
    ) -> _model.Actions:
        observation = preprocess_observation(None, observation, train=False)
        # Note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        time_threshold = (
            self.config.time_threshold_inpaint
            if time_threshold_inpaint is None
            else time_threshold_inpaint
        )
        
        # Generate or constrain noise based on inpainting requirements
        if initial_actions is not None:
            # INPAINTING PATH: Construct constrained noise z that satisfies initial_actions
            num_initial_actions = initial_actions.shape[1]
            input_action_dim = initial_actions.shape[2]
            
            # Pad initial_actions to full model dimensions (32D) and action_horizon (30)
            if input_action_dim < self.action_dim:
                action_padding = jnp.zeros((batch_size, num_initial_actions, self.action_dim - input_action_dim))
                initial_actions_full_dim = jnp.concatenate([initial_actions, action_padding], axis=2)
            else:
                initial_actions_full_dim = initial_actions[:, :, :self.action_dim]
            
            if num_initial_actions < self.action_horizon:
                seq_padding = jnp.zeros((batch_size, self.action_horizon - num_initial_actions, self.action_dim))
                initial_actions_padded = jnp.concatenate([initial_actions_full_dim, seq_padding], axis=1)
            else:
                initial_actions_padded = initial_actions_full_dim[:, :self.action_horizon]
            
            # Compute O and U indices for inpainting (JIT-safe: static list comprehensions)
            flat_dim = self.action_horizon * self.action_dim
            
            # Build O_indices: first num_initial_actions timesteps, first input_action_dim dimensions
            O_indices = jnp.array([
                t * self.action_dim + d 
                for t in range(num_initial_actions) 
                for d in range(input_action_dim)
            ], dtype=jnp.int32)
            
            # Build U_indices: all other indices (JIT-safe: static list comprehension)
            # Python set operations happen at trace time (before JIT), so this is safe
            O_set = {t * self.action_dim + d for t in range(num_initial_actions) for d in range(input_action_dim)}
            U_indices = jnp.array([
                i for i in range(flat_dim) if i not in O_set
            ], dtype=jnp.int32)
            
            # Generate noise
            rng, noise_rng = jax.random.split(rng)
            
            if self.correlation_loaded:
                # CORRELATED NOISE: Sample with correlation matrix
                noise = self.generate_correlated_noise(noise_rng, batch_size)
            else:
                # FALLBACK: Independent noise
                noise = jax.random.normal(noise_rng, (batch_size, self.action_horizon, self.action_dim))
            
            # Extract fixed z_O and x0_O for constraint enforcement
            noise_flat = noise.reshape(batch_size, flat_dim)
            fixed_z_O = noise_flat[:, O_indices]  # [b, |O|] - fixed noise for inpainting
            x0_O = initial_actions_padded.reshape(batch_size, flat_dim)[:, O_indices]  # [b, |O|] - target actions
            
            # Precompute correction matrix for correlation-aware inpainting
            inpainting_cache = None
            if self.correlation_loaded:
                cache_key = (num_initial_actions, input_action_dim)
                if cache_key not in self.inpainting_cache:
                    logger.info(f"Computing correction matrix for {num_initial_actions} steps, {input_action_dim} dims...")
                    self.inpainting_cache[cache_key] = self._precompute_correction_matrix(O_indices, U_indices)
                inpainting_cache = self.inpainting_cache[cache_key]
            
        else:
            # NO INPAINTING: Standard noise generation
            if noise is None:
                rng, noise_rng = jax.random.split(rng)
                noise = self.generate_correlated_noise(noise_rng, batch_size)
            
            fixed_z_O = None
            x0_O = None
            O_indices = None
            inpainting_cache = None

        # Split RNG for step loop
        rng, step_rng = jax.random.split(rng)

        # Ensure FAST tokens are never used during inference
        if observation.fast_tokens is not None:
            raise ValueError(
                "FAST tokens must not be provided during inference (sample_actions). "
                "FAST tokens are only used during training for auxiliary loss. "
                "Set observation.fast_tokens=None before calling sample_actions."
            )

        # First fill KV cache with a forward pass of the prefix (no FAST tokens during inference)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        
        # Predict stage from VLM output of base task token
        # Find base task token position (same logic as in compute_detailed_loss)
        first_stage_token_idx = jnp.argmax(prefix_ar_mask)  # Returns index of first True
        base_task_token_idx = first_stage_token_idx - 1
        base_task_output = prefix_out[:, base_task_token_idx, :]
        subtask_logits = self.stage_pred_from_vlm(base_task_output)  # [B, MAX_NUM_STAGES]
        
        # Mask out invalid stages for each task (vectorized JAX operations)
        task_ids = observation.tokenized_prompt[:, 0]  # [B]
        task_num_stages_array = jnp.array(TASK_NUM_STAGES, dtype=jnp.int32)
        task_num_stages = task_num_stages_array[task_ids]  # [B] - JAX array indexing
        stage_range = jnp.arange(MAX_NUM_STAGES)  # [15]
        valid_mask = stage_range[None, :] < task_num_stages[:, None]  # [B, 15]
        subtask_logits = jnp.where(valid_mask, subtask_logits, -jnp.inf)
        
        # Transform KV cache for cross-layer attention
        if self.kv_transform is not None:
            kv_cache = self.kv_transform(kv_cache)
        
        def step(carry):
            x_t, time, step_rng = carry

            # Model forward pass
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            
            # Euler step: x_{t+dt} = x_t + dt * v_t
            x_t_new = x_t + dt * v_t
            
            # Apply correlation-aware inpainting correction
            # Only enforce when time > TIME_THRESHOLD_INPAINT (let model be free in final steps)
            if fixed_z_O is not None:
                time_new = time + dt
                
                def apply_correlated_correction(x):
                    x_flat = x.reshape(batch_size, -1)
                    
                    # Compute desired state at O: x_t[O] = (1-t)*x0[O] + t*z_O
                    x_desired_O = (1.0 - time_new) * x0_O + time_new * fixed_z_O  # [b, |O|]
                    
                    # Compute correction at O
                    delta_O = x_desired_O - x_flat[:, O_indices]  # [b, |O|]
                    
                    # Apply hard constraint at O
                    x_flat = x_flat.at[:, O_indices].set(x_desired_O)
                    
                    # If correlation matrix available, propagate correction to U
                    if inpainting_cache is not None:
                        correction_matrix = inpainting_cache['correction_matrix']  # [|U|, |O|]
                        U_indices_cached = inpainting_cache['U_indices']
                        
                        # Compute correlated correction: δ_U = Σ_{UO}Σ_{OO}^{-1} @ δ_O
                        delta_U = delta_O @ correction_matrix.T  # [b, |U|]
                        
                        # Skip if correction too large (indicates instability)
                        max_correction = jnp.max(jnp.abs(delta_U))
                        x_flat = jax.lax.cond(
                            # Prevents exploding corrections in case of noisy out of distribution initial actions
                            max_correction <= 1.0,
                            lambda x: x.at[:, U_indices_cached].add(delta_U),
                            lambda x: x,
                            x_flat
                        )
                    
                    # Sanity check: if Σ = I, correction_matrix = 0, so delta_U = 0 that is correct
                    # If the correlation is 1 everywhere we will go to the flat prediction that is correct
                    
                    return x_flat.reshape(batch_size, self.action_horizon, self.action_dim)
                
                # Only apply correction when NEW time > threshold
                x_t_new = jax.lax.cond(
                    time_new > time_threshold,
                    apply_correlated_correction,
                    lambda x: x,
                    x_t_new
                )
            
            return x_t_new, time + dt, step_rng

        def cond(carry):
            x_t, time, step_rng = carry
            # Robust to floating-point error
            return time >= -dt / 2

        x_0, _, _ = jax.lax.while_loop(cond, step, (noise, 1.0, step_rng))
        
        return x_0, subtask_logits
