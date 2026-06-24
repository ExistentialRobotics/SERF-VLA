"""JAX forward kinematics for the R1Pro robot.

Computes robot base center and end-effector positions from
``observation.state[256]`` entirely in JAX, enabling gradient flow
and JIT compilation.

FK chain parameters are extracted from the URDF at init time using
``yourdfpy`` (NumPy).  At forward-pass time, the chain is evaluated
in pure JAX.

Reference:
    serf_b1k.mapping.utils.robot — NumPy FK utilities
"""

from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


# --- OmniGibson observation.state[256] Index Layout --- #

_IDX_JOINT_QPOS = slice(0, 28)
_IDX_ROBOT_POS = slice(140, 143)
_IDX_ROBOT_ORI_COS = slice(143, 146)
_IDX_ROBOT_ORI_SIN = slice(146, 149)

# OG joint_qpos indices for left arm (de-interleaved)
_OG_LEFT_ARM_INDICES = [10, 12, 14, 16, 18, 20, 22]
# OG joint_qpos indices for right arm (de-interleaved)
_OG_RIGHT_ARM_INDICES = [11, 13, 15, 17, 19, 21, 23]
# OG joint_qpos indices for torso
_OG_TORSO_INDICES = [6, 7, 8, 9]


# --- Rotation primitives (JAX) --- #


def _rotation_about_axis(axis: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    """Compute 3x3 rotation matrix via Rodrigues' formula.

    Args:
        axis: (3,) unit vector, joint rotation axis.
        angle: scalar, rotation angle in radians.

    Returns:
        (3, 3) rotation matrix.
    """
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    t = 1.0 - c
    x, y, z = axis[0], axis[1], axis[2]
    return jnp.array([
        [t * x * x + c,     t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c,     t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c    ],
    ])


def _euler_xyz_to_rotation(euler_rpy: jnp.ndarray) -> jnp.ndarray:
    """Convert euler XYZ angles to 3x3 rotation matrix.

    Args:
        euler_rpy: (3,) roll, pitch, yaw in radians.

    Returns:
        (3, 3) rotation matrix.
    """
    rx, ry, rz = euler_rpy[0], euler_rpy[1], euler_rpy[2]
    cx, sx = jnp.cos(rx), jnp.sin(rx)
    cy, sy = jnp.cos(ry), jnp.sin(ry)
    cz, sz = jnp.cos(rz), jnp.sin(rz)

    # R = Rz @ Ry @ Rx  (extrinsic XYZ = intrinsic ZYX)
    return jnp.array([
        [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
        [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
        [-sy,     sx * cy,                cx * cy               ],
    ])


def _make_transform(R: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Build 4x4 homogeneous transform from R (3x3) and t (3,)."""
    T = jnp.eye(4)
    T = T.at[:3, :3].set(R)
    T = T.at[:3, 3].set(t)
    return T


# --- FK chain extraction from URDF (init-time, NumPy) --- #


def _extract_fk_chain(urdf_path: str | Path) -> dict:
    """Extract FK chain parameters from URDF for left and right grippers.

    Returns a dict with:
        shared_origins:     (n_shared, 4, 4) — fixed transforms for shared joints (torso)
        shared_axes:        (n_shared, 3) — rotation axes for shared joints
        shared_og_indices:  (n_shared,) — OG joint_qpos indices
        left_fixed_origin:  (4, 4) — left_arm_base_joint fixed transform
        left_origins:       (7, 4, 4) — left arm joint fixed transforms
        left_axes:          (7, 3) — left arm joint axes
        left_og_indices:    (7,) — OG indices for left arm
        left_gripper_origin: (4, 4) — left_gripper_joint fixed transform
        right_fixed_origin: (4, 4) — right_arm_base_joint fixed transform
        right_origins:      (7, 4, 4) — right arm joint fixed transforms
        right_axes:         (7, 3) — right arm joint axes
        right_og_indices:   (7,) — OG indices for right arm
        right_gripper_origin: (4, 4) — right_gripper_joint fixed transform
    """
    import yourdfpy

    urdf = yourdfpy.URDF.load(
        str(urdf_path), build_scene_graph=True, load_meshes=False,
    )

    joint_by_name = {j.name: j for j in urdf.robot.joints}

    def get_origin(joint_name: str) -> np.ndarray:
        j = joint_by_name[joint_name]
        return j.origin.astype(np.float32) if j.origin is not None else np.eye(4, dtype=np.float32)

    def get_axis(joint_name: str) -> np.ndarray:
        j = joint_by_name[joint_name]
        return j.axis.astype(np.float32)

    # Shared torso chain: torso_joint1-4
    shared_names = ["torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4"]
    shared_origins = np.stack([get_origin(n) for n in shared_names])  # (4, 4, 4)
    shared_axes = np.stack([get_axis(n) for n in shared_names])      # (4, 3)
    shared_og_indices = np.array(_OG_TORSO_INDICES, dtype=np.int32)

    # Left arm chain
    left_arm_names = [f"left_arm_joint{i}" for i in range(1, 8)]
    left_origins = np.stack([get_origin(n) for n in left_arm_names])  # (7, 4, 4)
    left_axes = np.stack([get_axis(n) for n in left_arm_names])      # (7, 3)
    left_og_indices = np.array(_OG_LEFT_ARM_INDICES, dtype=np.int32)

    # Right arm chain
    right_arm_names = [f"right_arm_joint{i}" for i in range(1, 8)]
    right_origins = np.stack([get_origin(n) for n in right_arm_names])  # (7, 4, 4)
    right_axes = np.stack([get_axis(n) for n in right_arm_names])      # (7, 3)
    right_og_indices = np.array(_OG_RIGHT_ARM_INDICES, dtype=np.int32)

    return {
        "shared_origins": shared_origins,
        "shared_axes": shared_axes,
        "shared_og_indices": shared_og_indices,
        "left_fixed_origin": get_origin("left_arm_base_joint"),
        "left_origins": left_origins,
        "left_axes": left_axes,
        "left_og_indices": left_og_indices,
        "left_gripper_origin": get_origin("left_gripper_joint"),
        "right_fixed_origin": get_origin("right_arm_base_joint"),
        "right_origins": right_origins,
        "right_axes": right_axes,
        "right_og_indices": right_og_indices,
        "right_gripper_origin": get_origin("right_gripper_joint"),
    }


# --- Main JAX FK class --- #


class RobotFKJax:
    """JAX forward kinematics for R1Pro robot.

    Computes robot base center and end-effector positions from
    ``observation.state[256]``.

    FK chain parameters are extracted once from the URDF (NumPy) and
    stored as JAX arrays for efficient JIT-compiled forward passes.

    Args:
        urdf_path: Path to the R1Pro URDF file.
        z_offset: Height offset above base for robot center (meters).
    """

    def __init__(self, urdf_path: str | Path, z_offset: float = 0.8) -> None:
        # Store everything as plain NumPy arrays.  JAX automatically
        # promotes them when used in computations, so no jnp.array()
        # calls are needed and nothing leaks during JIT tracing.
        self._chain = _extract_fk_chain(urdf_path)
        self.z_offset = z_offset

    def __call__(
        self, state: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute robot center, left EE, and right EE positions.

        Args:
            state: [B, >=149] observation state vector.

        Returns:
            robot_center: [B, 3] base position + z_offset.
            left_ee:      [B, 3] left gripper link position.
            right_ee:     [B, 3] right gripper link position.
        """
        c = self._chain
        return jax.vmap(
            lambda s: _single_fk(s, c, self.z_offset)
        )(state)


def _single_fk(
    state: jnp.ndarray,
    jc: dict,
    z_offset: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """FK for a single state vector (no batch dim).

    Pure function — all chain data passed via *jc* dict so nothing
    is captured from a mutable object.
    """
    joint_qpos = state[_IDX_JOINT_QPOS]

    # --- Base transform --- #
    base_pos = state[_IDX_ROBOT_POS]
    ori_cos = state[_IDX_ROBOT_ORI_COS]
    ori_sin = state[_IDX_ROBOT_ORI_SIN]
    euler_rpy = jnp.arctan2(ori_sin, ori_cos)
    base_R = _euler_xyz_to_rotation(euler_rpy)
    T = _make_transform(base_R, base_pos)

    # --- Shared torso chain --- #
    for i in range(4):
        angle = joint_qpos[jc["shared_og_indices"][i]]
        R_joint = _rotation_about_axis(jc["shared_axes"][i], angle)
        T_joint = _make_transform(R_joint, jnp.zeros(3))
        T = T @ jc["shared_origins"][i] @ T_joint

    T_after_torso = T

    # --- Left arm FK --- #
    T_left = T_after_torso @ jc["left_fixed_origin"]
    for i in range(7):
        angle = joint_qpos[jc["left_og_indices"][i]]
        R_joint = _rotation_about_axis(jc["left_axes"][i], angle)
        T_joint = _make_transform(R_joint, jnp.zeros(3))
        T_left = T_left @ jc["left_origins"][i] @ T_joint
    T_left = T_left @ jc["left_gripper_origin"]
    left_ee = T_left[:3, 3]

    # --- Right arm FK --- #
    T_right = T_after_torso @ jc["right_fixed_origin"]
    for i in range(7):
        angle = joint_qpos[jc["right_og_indices"][i]]
        R_joint = _rotation_about_axis(jc["right_axes"][i], angle)
        T_joint = _make_transform(R_joint, jnp.zeros(3))
        T_right = T_right @ jc["right_origins"][i] @ T_joint
    T_right = T_right @ jc["right_gripper_origin"]
    right_ee = T_right[:3, 3]

    # --- Robot center --- #
    robot_center = base_pos.at[2].add(z_offset)

    return robot_center, left_ee, right_ee
