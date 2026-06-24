"""Shared R1Pro robot state utilities.

Converts OmniGibson observation.state[256] vectors to URDF joint
configurations and base transforms.  All functions are pure numpy
unless noted otherwise.

OmniGibson joint_qpos ordering (n_dof=28):
    [0:6]     virtual base (base_footprint x,y,z,rx,ry,rz)
    [6:10]    trunk (torso_joint1-4)
    [10:24]   left/right arms INTERLEAVED:
              [10,12,14,16,18,20,22] = left_arm_joint1-7
              [11,13,15,17,19,21,23] = right_arm_joint1-7
    [24:26]   left gripper (left_gripper_finger_joint1-2)
    [26:28]   right gripper (right_gripper_finger_joint1-2)

URDF actuated joint order (yourdfpy / viser parse order):
    [0:6]   steer/wheel motors (not in OmniGibson state -> 0)
    [6:10]  torso_joint1-4
    [10:17] left_arm_joint1-7  (consecutive)
    [17:19] left_gripper_finger_joint1-2
    [19:26] right_arm_joint1-7 (consecutive)
    [26:28] right_gripper_finger_joint1-2
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# --- Paths --- #

REPO_ROOT = Path(__file__).resolve().parents[4]
URDF_PATH = REPO_ROOT / "BEHAVIOR-1K" / "datasets" / "omnigibson-robot-assets" / "models" / "r1pro" / "urdf" / "r1pro.urdf"

# --- OmniGibson observation.state[256] Index Layout --- #

N_URDF_JOINTS = 28

IDX_JOINT_QPOS = slice(0, 28)
IDX_ROBOT_POS = slice(140, 143)
IDX_ROBOT_ORI_COS = slice(143, 146)
IDX_ROBOT_ORI_SIN = slice(146, 149)


# --- Data Loading --- #


def load_robot_states(parquet_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load observation states and actions from a LeRobot parquet file.

    Args:
        parquet_path: Path to the parquet file.

    Returns:
        states: (N, 256) float32 observation states.
        actions: (N, 23) float32 actions.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    states = np.stack(df["observation.state"].values).astype(np.float32)
    actions = np.stack(df["action"].values).astype(np.float32)
    print(f"[LOAD] {len(states)} robot frames from {parquet_path.name}")
    return states, actions


# --- State Conversion --- #


def joint_qpos_to_urdf_cfg(joint_qpos: np.ndarray) -> np.ndarray:
    """Convert 28-element OmniGibson joint_qpos to URDF joint ordering.

    De-interleaves left/right arm joints from OmniGibson's interleaved
    ordering to URDF's consecutive ordering.

    Args:
        joint_qpos: (28,) joint positions in OmniGibson ordering
            (from ``robot.get_joint_positions()`` or ``state[0:28]``).

    Returns:
        (28,) float32 array in URDF joint ordering.
    """
    cfg = np.zeros(N_URDF_JOINTS, dtype=np.float32)
    # URDF[0:6]   steer/wheel -> 0 (not in OmniGibson state)
    # URDF[6:10]  trunk       <- qpos[6:10]
    cfg[6:10] = joint_qpos[6:10]
    # URDF[10:17] left arm    <- qpos[10,12,14,16,18,20,22] (de-interleave)
    cfg[10:17] = joint_qpos[10:24:2]
    # URDF[17:19] left grip   <- qpos[24:26]
    cfg[17:19] = joint_qpos[24:26]
    # URDF[19:26] right arm   <- qpos[11,13,15,17,19,21,23] (de-interleave)
    cfg[19:26] = joint_qpos[11:24:2]
    # URDF[26:28] right grip  <- qpos[26:28]
    cfg[26:28] = joint_qpos[26:28]
    return cfg


def state_to_urdf_cfg(state: np.ndarray) -> np.ndarray:
    """Convert observation.state[256] to URDF joint configuration (28,).

    Convenience wrapper around ``joint_qpos_to_urdf_cfg`` that first
    extracts the joint_qpos slice from a full state vector.
    """
    return joint_qpos_to_urdf_cfg(state[IDX_JOINT_QPOS])


def state_to_base_pose(
    state: np.ndarray,
    origin_pos: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract robot base position and euler angles from state.

    Args:
        state: (256,) observation state vector.
        origin_pos: If provided, subtract from position to center the scene.

    Returns:
        position: (3,) world position.
        euler_rpy: (3,) roll, pitch, yaw in radians.
    """
    position = state[IDX_ROBOT_POS].copy()
    if origin_pos is not None:
        position -= origin_pos

    ori_cos = state[IDX_ROBOT_ORI_COS]
    ori_sin = state[IDX_ROBOT_ORI_SIN]
    euler_rpy = np.arctan2(ori_sin, ori_cos)

    return position, euler_rpy


def state_to_base_transform_matrix(
    state: np.ndarray,
    origin_pos: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert state to a 4x4 base transform matrix.

    Like ``state_to_base_transform_viser`` but returns a plain numpy
    matrix suitable for Open3D / trimesh (no viser dependency).
    """
    from scipy.spatial.transform import Rotation

    position, euler_rpy = state_to_base_pose(state, origin_pos)
    R = Rotation.from_euler("xyz", euler_rpy).as_matrix()
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = position
    return T


def state_to_base_transform_viser(
    state: np.ndarray,
    origin_pos: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert state to (position, wxyz) for viser frame handles.

    Like ``state_to_base_pose`` but returns a quaternion (w,x,y,z)
    instead of euler angles.  Requires ``viser`` (lazy import).
    """
    import viser.transforms as vtf

    position, euler_rpy = state_to_base_pose(state, origin_pos)
    wxyz = vtf.SO3.from_rpy_radians(
        roll=float(euler_rpy[0]),
        pitch=float(euler_rpy[1]),
        yaw=float(euler_rpy[2]),
    ).wxyz

    return position, np.array(wxyz)


# --- Robot Surface Sampling --- #

# Dense oversampling density: points per m² before voxel downsampling.
_DENSE_POINTS_PER_M2 = 100_000
_MIN_DENSE_POINTS = 500


class RobotSurfaceSampler:
    """Sample and retrieve consistent surface points across joint configurations.

    Points are densely sampled on each link's visual mesh (area-proportional),
    then voxel-downsampled in geometry-local coordinates for uniform spatial
    density.  Stored as flat CSR arrays for efficient FK and serialization.

    Requires ``open3d`` and ``yourdfpy`` (imported lazily).

    Attributes:
        urdf: The loaded yourdfpy URDF model.
        local_points_homo: (N, 4) float32 — all points in geometry-local homogeneous coords.
        geom_offsets: (M+1,) int32 — CSR offsets: geom i owns points[off[i]:off[i+1]].
        geom_nodes: List of M geometry node names (for scene graph transform lookup).
        link_names: List of M parent link names.
        link_labels: (N,) int32 — per-point geometry index.
        n_total_points: Total number of surface points.
        n_geoms: Number of geometry nodes.
        voxel_size: Voxel size used for downsampling.
    """

    def __init__(
        self,
        urdf_path: Path | str,
        voxel_size: float = 0.02,
        seed: int = 0,
    ) -> None:
        import open3d as o3d
        import yourdfpy

        self.voxel_size = voxel_size

        self.urdf = yourdfpy.URDF.load(
            str(urdf_path),
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=False,
            load_collision_meshes=False,
        )

        # Build visual_name → link_name mapping
        visual_to_link: dict[str, str] = {}
        for link in self.urdf.robot.links:
            for vis in link.visuals:
                if vis.name is not None:
                    visual_to_link[vis.name] = link.name

        # --- Sample + voxel downsample per geometry node --- #
        from typing import List
        geom_nodes: List[str] = []
        link_names: List[str] = []
        all_points: List[np.ndarray] = []
        offsets: List[int] = [0]

        np.random.seed(seed)
        total_dense = 0

        for geom_node in self.urdf.scene.graph.nodes_geometry:
            _, geom_key = self.urdf.scene.graph.get(geom_node)
            mesh = self.urdf.scene.geometry[geom_key]

            if not hasattr(mesh, "sample") or mesh.area < 1e-10:
                continue

            # Dense area-proportional sampling
            n_dense = max(_MIN_DENSE_POINTS, int(mesh.area * _DENSE_POINTS_PER_M2))
            dense_pts = mesh.sample(n_dense, return_index=False)
            total_dense += n_dense

            # Voxel downsample in geometry-local frame
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(dense_pts)
            pcd_down = pcd.voxel_down_sample(voxel_size)
            pts = np.asarray(pcd_down.points, dtype=np.float32)

            geom_nodes.append(geom_node)
            link_names.append(visual_to_link.get(geom_node, geom_node))
            all_points.append(pts)
            offsets.append(offsets[-1] + len(pts))

        # --- Build flat CSR arrays --- #
        self.geom_nodes: list[str] = geom_nodes
        self.link_names: list[str] = link_names
        self.geom_offsets: np.ndarray = np.array(offsets, dtype=np.int32)
        self.n_geoms: int = len(geom_nodes)
        self.n_total_points: int = offsets[-1]

        # (N, 4) homogeneous local coords — single contiguous array
        local_pts = np.concatenate(all_points, axis=0)  # (N, 3)
        ones = np.ones((self.n_total_points, 1), dtype=np.float32)
        self.local_points_homo: np.ndarray = np.hstack([local_pts, ones])

        # Per-point geometry index
        self.link_labels: np.ndarray = np.concatenate(
            [
                np.full(offsets[i + 1] - offsets[i], i, dtype=np.int32)
                for i in range(self.n_geoms)
            ]
        )

        print(
            f"[RobotSurfaceSampler] {total_dense} dense → {self.n_total_points} points "
            f"(voxel {voxel_size}m) across {self.n_geoms} geometry nodes"
        )

    # --- Query --- #

    def get_points(
        self,
        cfg: np.ndarray,
        base_transform: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get world-frame surface points for a given joint configuration.

        Args:
            cfg: (N_URDF_JOINTS,) joint configuration array.
            base_transform: Optional (4, 4) world-frame base transform.

        Returns:
            world_points: (N, 3) float32.
            link_labels: (N,) int32 — index into self.geom_nodes / self.link_names.
        """
        self.urdf.update_cfg(cfg)

        world_points = np.empty((self.n_total_points, 3), dtype=np.float32)

        for i in range(self.n_geoms):
            s, e = self.geom_offsets[i], self.geom_offsets[i + 1]
            T, _ = self.urdf.scene.graph.get(self.geom_nodes[i])

            if base_transform is not None:
                T = base_transform @ T

            # (4, 4) @ (4, K) → (4, K), take first 3 rows → (3, K) → (K, 3)
            world_points[s:e] = (T @ self.local_points_homo[s:e].T).T[:, :3]

        return world_points, self.link_labels

    def get_link_mask(self, link_name: str) -> np.ndarray:
        """Boolean mask for points belonging to a given link.

        Args:
            link_name: Link name (substring match).

        Returns:
            (N,) bool array.
        """
        indices = [i for i, name in enumerate(self.link_names) if link_name in name]
        mask = np.zeros(self.n_total_points, dtype=bool)
        for i in indices:
            mask[self.geom_offsets[i]:self.geom_offsets[i + 1]] = True
        return mask

    def get_geom_slice(self, geom_idx: int) -> slice:
        """Get the slice for a given geometry index into the flat arrays."""
        return slice(self.geom_offsets[geom_idx], self.geom_offsets[geom_idx + 1])

    # --- Serialization --- #

    def save(self, path: Path | str) -> None:
        """Save sampled points to .npz (no URDF data, just geometry)."""
        np.savez_compressed(
            str(path),
            local_points_homo=self.local_points_homo,
            geom_offsets=self.geom_offsets,
            link_labels=self.link_labels,
            geom_nodes=np.array(self.geom_nodes, dtype=object),
            link_names=np.array(self.link_names, dtype=object),
            voxel_size=np.array(self.voxel_size),
        )
        print(f"[RobotSurfaceSampler] Saved {self.n_total_points} points to {path}")

    @classmethod
    def load(cls, path: Path | str, urdf_path: Path | str) -> "RobotSurfaceSampler":
        """Load pre-sampled points from .npz + URDF (for FK).

        Args:
            path: Path to the .npz file.
            urdf_path: Path to the URDF file (needed for FK scene graph).
        """
        import yourdfpy

        obj = object.__new__(cls)

        data = np.load(str(path), allow_pickle=True)
        obj.local_points_homo = data["local_points_homo"]
        obj.geom_offsets = data["geom_offsets"]
        obj.link_labels = data["link_labels"]
        obj.geom_nodes = data["geom_nodes"].tolist()
        obj.link_names = data["link_names"].tolist()
        obj.voxel_size = float(data["voxel_size"])
        obj.n_geoms = len(obj.geom_nodes)
        obj.n_total_points = len(obj.local_points_homo)

        obj.urdf = yourdfpy.URDF.load(
            str(urdf_path),
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=False,
            load_collision_meshes=False,
        )
        print(
            f"[RobotSurfaceSampler] Loaded {obj.n_total_points} points from {path}"
        )
        return obj
