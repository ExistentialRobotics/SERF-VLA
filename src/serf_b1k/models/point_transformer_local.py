"""Multi-branch local map tokenizer.

Shared backbone uses Point Transformer blocks
to produce downsampled points with feature representations.
Up to 8 parallel branches extract spatially-aware tokens (6 when use_robot_points=False):

    (a) 3× robot-center ball query at r=1,2,4 m → PT×2 → attn pool → 1 token each
    (b) 1× global (all pts) → PT×2 → attn pool → 1 token
    (c) 2× end-effector ball query at r=0.5 m (left, right) → PT×2 → attn pool → 1 token each
    (d) 1× robot-only (mask-based filtering) → PT×2 → attn pool → 1 token
    (e) 1× env-only (inverse mask filtering) → PT×2 → attn pool → 1 token

Output: [B, 8, token_dim] (use_robot_points=True) or [B, 6, token_dim] (False)

All ball-query distances are specified in world-space metres and converted
to normalised point-cloud space using (pc_norm_offset, pc_norm_scale) passed
at forward time.

Reference files:
    robot_fk_jax.py — RobotFKJax (base center & EE FK)
"""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


def square_distance(src: jnp.ndarray, dst: jnp.ndarray) -> jnp.ndarray:
    dist = -2 * jnp.matmul(src, jnp.transpose(dst, (0, 2, 1)))
    dist += jnp.sum(src ** 2, axis=-1, keepdims=True)
    dist += jnp.sum(dst ** 2, axis=-1, keepdims=True).transpose(0, 2, 1)
    return dist


def index_points(points: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    batch_size = points.shape[0]
    batch_indices = jnp.arange(batch_size).reshape(
        -1, *([1] * (len(idx.shape) - 1))
    )
    batch_indices = jnp.broadcast_to(batch_indices, idx.shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: jnp.ndarray, npoint: int) -> jnp.ndarray:
    batch_size, num_points, _ = xyz.shape
    centroids = jnp.zeros((batch_size, npoint), dtype=jnp.int32)
    distance = jnp.ones((batch_size, num_points)) * 1e10
    farthest = jnp.zeros(batch_size, dtype=jnp.int32)
    batch_indices = jnp.arange(batch_size)

    def body_fun(i, carry):
        centroids, distance, farthest = carry
        centroids = centroids.at[:, i].set(farthest)
        centroid = xyz[batch_indices, farthest, :].reshape(batch_size, 1, 3)
        dist = jnp.sum((xyz - centroid) ** 2, axis=-1)
        distance = jnp.minimum(distance, dist)
        farthest = jnp.argmax(distance, axis=-1)
        return centroids, distance, farthest

    centroids, _, _ = jax.lax.fori_loop(
        0, npoint, body_fun, (centroids, distance, farthest)
    )
    return centroids


def knn_query(nsample: int, xyz: jnp.ndarray, new_xyz: jnp.ndarray) -> jnp.ndarray:
    sqrdists = square_distance(new_xyz, xyz)
    _, idx = jax.lax.top_k(-sqrdists, nsample)
    return idx.astype(jnp.int32)


def query_and_group(
    nsample: int,
    xyz: jnp.ndarray,
    new_xyz: jnp.ndarray,
    feat: jnp.ndarray,
    use_xyz: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    idx = knn_query(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz = grouped_xyz - new_xyz[:, :, None, :]
    grouped_feat = index_points(feat, idx)

    if use_xyz:
        return idx, jnp.concatenate([grouped_xyz, grouped_feat], axis=-1)
    return idx, grouped_feat


class PointTransformerLayer(nn.Module):
    out_planes: int
    share_planes: int = 8
    nsample: int = 16

    @nn.compact
    def __call__(self, xyz: jnp.ndarray, feat: jnp.ndarray) -> jnp.ndarray:
        batch_size, num_points, _ = feat.shape
        mid_planes = self.out_planes
        out_planes = self.out_planes

        query = nn.Dense(mid_planes, name="linear_q")(feat)
        key = nn.Dense(mid_planes, name="linear_k")(feat)
        value = nn.Dense(out_planes, name="linear_v")(feat)

        idx = knn_query(self.nsample, xyz, xyz)
        grouped_key = index_points(key, idx)
        grouped_value = index_points(value, idx)
        grouped_xyz = index_points(xyz, idx) - xyz[:, :, None, :]

        pos = nn.Dense(mid_planes, name="pos_fc1")(grouped_xyz)
        pos = nn.RMSNorm(name="pos_norm")(pos)
        pos = jax.nn.relu(pos)
        pos = nn.Dense(out_planes, name="pos_fc2")(pos)

        pos_for_weight = pos.reshape(
            batch_size,
            num_points,
            self.nsample,
            out_planes // mid_planes,
            mid_planes,
        ).sum(axis=3)
        weight = grouped_key - query[:, :, None, :] + pos_for_weight
        weight = nn.RMSNorm(name="w_norm1")(weight)
        weight = jax.nn.relu(weight)
        weight = nn.Dense(mid_planes // self.share_planes, name="w_fc1")(weight)
        weight = nn.RMSNorm(name="w_norm2")(weight)
        weight = jax.nn.relu(weight)
        weight = nn.Dense(out_planes // self.share_planes, name="w_fc2")(weight)
        weight = jax.nn.softmax(weight, axis=2)

        share_planes = self.share_planes
        out_feat = (
            (grouped_value + pos).reshape(
                batch_size,
                num_points,
                self.nsample,
                share_planes,
                out_planes // share_planes,
            )
            * weight[:, :, :, None, :]
        ).sum(axis=2).reshape(batch_size, num_points, out_planes)
        return out_feat


class PointTransformerBlock(nn.Module):
    planes: int
    share_planes: int = 8
    nsample: int = 16

    @nn.compact
    def __call__(self, xyz: jnp.ndarray, feat: jnp.ndarray) -> jnp.ndarray:
        identity = feat

        x = nn.Dense(self.planes, use_bias=False, name="linear1")(feat)
        x = nn.RMSNorm(name="bn1")(x)
        x = jax.nn.relu(x)

        x = PointTransformerLayer(
            out_planes=self.planes,
            share_planes=self.share_planes,
            nsample=self.nsample,
            name="transformer2",
        )(xyz, x)
        x = nn.RMSNorm(name="bn2")(x)
        x = jax.nn.relu(x)

        x = nn.Dense(self.planes, use_bias=False, name="linear3")(x)
        x = nn.RMSNorm(name="bn3")(x)
        return jax.nn.relu(x + identity)


# --------------------------------------------------------------------------- #
#  Local branch (PT × 2 → masked attention pool)
# --------------------------------------------------------------------------- #

class LocalBranch(nn.Module):
    """Two PointTransformerBlocks followed by masked attention pooling.

    Args:
        planes:    Feature dimension (must match input feat dim).
        nsample:   KNN neighbourhood size for PT blocks.
        token_dim: Output token dimension.
    """
    planes: int = 256
    nsample: int = 16
    token_dim: int = 2048

    @nn.compact
    def __call__(
        self,
        xyz: jnp.ndarray,
        feat: jnp.ndarray,
        valid_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Args:
            xyz:        [B, M, 3]
            feat:       [B, M, planes]
            valid_mask: [B, M] bool or None

        Returns:
            [B, 1, token_dim] — single attention-pooled token.
        """
        # Zero out invalid points so they don't corrupt KNN neighbors in PT blocks
        if valid_mask is not None:
            feat = jnp.where(valid_mask[:, :, None], feat, 0.0)
            xyz = jnp.where(valid_mask[:, :, None], xyz, jnp.float32(1e6))

        # Two PointTransformerBlocks
        x = PointTransformerBlock(
            planes=self.planes, nsample=self.nsample, name="pt_block_0",
        )(xyz, feat)
        x = PointTransformerBlock(
            planes=self.planes, nsample=self.nsample, name="pt_block_1",
        )(xyz, x)

        # Attention pooling → 1 token
        scores = nn.Dense(1, name="attn_pool_proj")(x).squeeze(-1)  # [B, M]
        if valid_mask is not None:
            scores = jnp.where(valid_mask, scores, jnp.float32(-1e9))
        weights = jax.nn.softmax(scores, axis=-1)                    # [B, M]
        pooled = jnp.einsum("bm,bmd->bd", weights, x)               # [B, planes]

        token = nn.Dense(self.token_dim, name="token_proj")(pooled)  # [B, token_dim]
        return token[:, None, :]                                     # [B, 1, token_dim]


# --------------------------------------------------------------------------- #
#  Inline TransitionDown step (with aux_mask propagation)
# --------------------------------------------------------------------------- #

def _transition_down_with_aux(
    xyz: jnp.ndarray,
    feat: jnp.ndarray,
    stride: int,
    nsample: int,
    linear: nn.Dense,
    norm: nn.RMSNorm,
    aux_mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
    """TransitionDown that also propagates an auxiliary boolean mask.

    FPS is always unmasked so that centroids represent the full scene
    (robot + environment) without bias.  The aux_mask is propagated
    by gathering at the FPS indices.

    Args:
        xyz:      [B, N, 3]
        feat:     [B, N, C]
        stride:   downsampling ratio
        nsample:  KNN neighbourhood size
        linear:   Dense layer (C_in → out_planes)
        norm:     RMSNorm layer
        aux_mask: [B, N] bool or None

    Returns:
        new_xyz:      [B, M, 3]
        new_feat:     [B, M, out_planes]
        new_aux_mask: [B, M] bool or None
    """
    B, N, _ = feat.shape
    M = N // stride

    fps_idx = farthest_point_sample(xyz, M)

    new_xyz = index_points(xyz, fps_idx)  # [B, M, 3]
    _, grouped = query_and_group(nsample, xyz, new_xyz, feat, use_xyz=True)
    # grouped: [B, M, nsample, 3+C]

    x = linear(grouped)       # [B, M, nsample, out_planes]
    x = norm(x)
    x = jax.nn.relu(x)
    x = jnp.max(x, axis=2)   # [B, M, out_planes]

    new_aux_mask = None
    if aux_mask is not None:
        batch_idx = jnp.arange(B)[:, None]
        new_aux_mask = aux_mask[batch_idx, fps_idx]  # [B, M]

    return new_xyz, x, new_aux_mask


# --------------------------------------------------------------------------- #
#  Main tokenizer
# --------------------------------------------------------------------------- #

class MapTokenizer(nn.Module):
    """Multi-branch local map tokenizer with a Point Transformer backbone.

    TransitionDown + PointTransformerBlock levels produce downsampled points
    → 8 parallel branches → attention-pooled tokens.

    Args:
        map_input_dim:        Input feature dim.
        token_dim:            Output token dimension (should match VLM embed dim).
        enc_planes:           Channel dims at each backbone level.
        enc_blocks:           Number of PT blocks at each backbone level.
        enc_strides:          FPS downsampling ratio at each backbone level.
        enc_nsample:          KNN neighbourhood size at each backbone level.
        robot_center_radii:   Ball query radii for robot-center branches (metres).
        ee_radius:            Ball query radius for end-effector branches (metres).
        local_branch_planes:  Feature dim in local PT blocks.
        local_branch_nsample: KNN neighbourhood in local PT blocks.
        num_input_points:     Deprecated compatibility field; ignored.
    """
    map_input_dim: int = 64
    token_dim: int = 2048
    enc_planes: Sequence[int] = (128, 256)
    enc_blocks: Sequence[int] = (2, 2)
    enc_strides: Sequence[int] = (4, 4)
    enc_nsample: Sequence[int] = (16, 16)
    robot_center_radii: Sequence[float] = (1.0, 2.0, 4.0)
    ee_radius: float = 0.5
    local_branch_planes: int = 256
    local_branch_nsample: int = 16
    num_input_points: int = 24576
    use_robot_points: bool = True  # False removes global & robot-only branches (8→6 tokens)

    def setup(self):
        if not (len(self.enc_planes) == len(self.enc_blocks) == len(self.enc_strides) == len(self.enc_nsample)):
            raise ValueError(
                f"enc_planes, enc_blocks, enc_strides, enc_nsample must have the same length, "
                f"got {len(self.enc_planes)}, {len(self.enc_blocks)}, {len(self.enc_strides)}, {len(self.enc_nsample)}"
            )
        if self.local_branch_planes != self.enc_planes[-1]:
            raise ValueError(
                f"local_branch_planes ({self.local_branch_planes}) must equal "
                f"enc_planes[-1] ({self.enc_planes[-1]}) for PT residual connections"
            )
        # (a) 3 robot-center branches
        self.robot_branches = [
            LocalBranch(
                planes=self.local_branch_planes,
                nsample=self.local_branch_nsample,
                token_dim=self.token_dim,
                name=f"robot_branch_r{i}",
            )
            for i in range(len(self.robot_center_radii))
        ]
        # (b) 1 global branch — only when robot points are used
        if self.use_robot_points:
            self.global_branch = LocalBranch(
                planes=self.local_branch_planes,
                nsample=self.local_branch_nsample,
                token_dim=self.token_dim,
                name="global_branch",
            )
        # (c) 2 EE branches (left, right)
        self.left_ee_branch = LocalBranch(
            planes=self.local_branch_planes,
            nsample=self.local_branch_nsample,
            token_dim=self.token_dim,
            name="left_ee_branch",
        )
        self.right_ee_branch = LocalBranch(
            planes=self.local_branch_planes,
            nsample=self.local_branch_nsample,
            token_dim=self.token_dim,
            name="right_ee_branch",
        )
        # (d) 1 robot-only branch — only when robot points are used
        if self.use_robot_points:
            self.robot_only_branch = LocalBranch(
                planes=self.local_branch_planes,
                nsample=self.local_branch_nsample,
                token_dim=self.token_dim,
                name="robot_only_branch",
            )
        # (e) 1 env-only branch
        self.env_only_branch = LocalBranch(
            planes=self.local_branch_planes,
            nsample=self.local_branch_nsample,
            token_dim=self.token_dim,
            name="env_only_branch",
        )

    def _world_to_norm(
        self,
        pos_world: jnp.ndarray,
        offset: jnp.ndarray,
        scale: jnp.ndarray,
    ) -> jnp.ndarray:
        """Convert world coordinates to normalised point-cloud space.

        pos_norm = (pos_world - offset) / scale - 1

        Args:
            pos_world: [..., 3]
            offset:    [B, 3]  (per-axis min from normalize_pc)
            scale:     [B, 1]  (max_range / 2)

        Returns:
            [..., 3] normalised coordinates.
        """
        return (pos_world - offset) / scale - 1.0

    @nn.compact
    def __call__(
        self,
        xyz: jnp.ndarray,
        feat: jnp.ndarray,
        robot_center: jnp.ndarray,
        left_ee: jnp.ndarray,
        right_ee: jnp.ndarray,
        pc_norm_offset: jnp.ndarray,
        pc_norm_scale: jnp.ndarray,
        training: bool = True,
        robot_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Args:
            xyz:            [B, N, 3]   normalised point coordinates
            feat:           [B, N, D]   point features (RGB / latent)
            robot_center:   [B, 3]      world-space robot center
            left_ee:        [B, 3]      world-space left EE position
            right_ee:       [B, 3]      world-space right EE position
            pc_norm_offset: [B, 3]      normalization offset
            pc_norm_scale:  [B, 1]      normalization scale
            training:       bool
            robot_mask:     [B, N] bool, optional — True for robot points

        Returns:
            [B, 8, token_dim] if use_robot_points else [B, 6, token_dim].
        """
        # --- Shared Point Transformer backbone --- #
        current_feat = feat

        current_xyz = xyz
        current_aux_mask = robot_mask

        for lvl in range(len(self.enc_planes)):
            out_ch = self.enc_planes[lvl]
            td_linear = nn.Dense(
                out_ch, use_bias=False,
                name=f"backbone_td{lvl}_linear",
            )
            td_norm = nn.RMSNorm(name=f"backbone_td{lvl}_norm")

            current_xyz, current_feat, current_aux_mask = _transition_down_with_aux(
                current_xyz, current_feat,
                stride=self.enc_strides[lvl],
                nsample=self.enc_nsample[lvl],
                linear=td_linear,
                norm=td_norm,
                aux_mask=current_aux_mask,
            )

            for blk in range(self.enc_blocks[lvl]):
                current_feat = PointTransformerBlock(
                    planes=out_ch,
                    nsample=self.enc_nsample[lvl],
                    name=f"backbone_pt{lvl}_blk{blk}",
                )(current_xyz, current_feat)

        stage2_xyz = current_xyz
        stage2_feat = current_feat
        stage2_robot_mask = current_aux_mask

        # --- Convert query centres to normalised space --- #
        robot_center_norm = self._world_to_norm(robot_center, pc_norm_offset, pc_norm_scale)
        left_ee_norm  = self._world_to_norm(left_ee,  pc_norm_offset, pc_norm_scale)
        right_ee_norm = self._world_to_norm(right_ee, pc_norm_offset, pc_norm_scale)

        tokens = []

        # --- (a) Robot-center ball queries --- #
        robot_sq_dist = jnp.sum(
            (stage2_xyz - robot_center_norm[:, None, :]) ** 2, axis=-1,
        )  # [B, M]

        for i, radius_world in enumerate(self.robot_center_radii):
            norm_radius = radius_world / pc_norm_scale  # [B, 1]
            mask = robot_sq_dist <= (norm_radius ** 2)
            tokens.append(self.robot_branches[i](stage2_xyz, stage2_feat, mask))

        # --- (b) Global branch (only when robot points are used) --- #
        if self.use_robot_points:
            tokens.append(self.global_branch(stage2_xyz, stage2_feat))

        # --- (c) End-effector ball queries --- #
        for ee_pos, branch in [
            (left_ee_norm,  self.left_ee_branch),
            (right_ee_norm, self.right_ee_branch),
        ]:
            sq_dist = jnp.sum(
                (stage2_xyz - ee_pos[:, None, :]) ** 2, axis=-1,
            )
            norm_radius = self.ee_radius / pc_norm_scale  # [B, 1]
            mask = sq_dist <= (norm_radius ** 2)
            tokens.append(branch(stage2_xyz, stage2_feat, mask))

        # --- (d) Robot-only branch (only when robot points are used) --- #
        if self.use_robot_points:
            if stage2_robot_mask is not None:
                tokens.append(self.robot_only_branch(
                    stage2_xyz, stage2_feat, stage2_robot_mask,
                ))
            else:
                tokens.append(jnp.zeros((stage2_xyz.shape[0], 1, self.token_dim)))

        # --- (e) Environment-only branch (inverse mask filtering) --- #
        if stage2_robot_mask is not None:
            env_mask = ~stage2_robot_mask
            tokens.append(self.env_only_branch(
                stage2_xyz, stage2_feat, env_mask,
            ))
        else:
            # No robot → all points are env points → no mask needed
            tokens.append(self.env_only_branch(stage2_xyz, stage2_feat))

        # --- Concatenate all tokens --- #
        return jnp.concatenate(tokens, axis=1)
