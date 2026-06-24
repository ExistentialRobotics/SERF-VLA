import numpy as np
import open3d as o3d


point_cloud_stats = {
    "task-0021": {
        "x": [20.3, 24.3],
        "y": [5.4, 8.3],
        "z": [-0.1, 1.7],
    },
    "task-0022": {
        "x": [-4.4, 1.1],
        "y": [-1.2, 7.8],
        "z": [-0.1, 2.8],
    },
    "task-0026": {
        "x": [0.9, 8.5],
        "y": [-1.1, 7.7],
        "z": [-0.1, 1.7],
    },
}


def get_pc_norm_params(task_idx: str):
    xyz_range = point_cloud_stats[task_idx]
    ranges = [xyz_range[axis][1] - xyz_range[axis][0] for axis in ["x", "y", "z"]]
    max_rng = max(ranges)
    offset = np.array([xyz_range["x"][0], xyz_range["y"][0], xyz_range["z"][0]], dtype=np.float32)
    scale = np.float32((max_rng + 1e-6) / 2.0)
    return offset, scale


def normalize_pc(positions, task_idx):
    xyz_range = point_cloud_stats[task_idx]
    ranges = [xyz_range[axis][1] - xyz_range[axis][0] for axis in ["x", "y", "z"]]
    max_rng = max(ranges)

    pos_np = positions.cpu().numpy()
    for axis_index, axis_name in enumerate(["x", "y", "z"]):
        p_min, p_max = xyz_range[axis_name]
        pos_np[:, axis_index] = np.clip(pos_np[:, axis_index], p_min, p_max)
        pos_np[:, axis_index] = 2 * (pos_np[:, axis_index] - p_min) / (max_rng + 1e-6) - 1
    return o3d.core.Tensor(pos_np, dtype=o3d.core.float32)


def normalize_rgb(colors):
    color_array = colors.numpy() if hasattr(colors, "numpy") else colors
    color_array = color_array.astype(np.float32)
    if color_array.max() > 1.0:
        color_array = color_array / 255.0
    return 2.0 * color_array - 1.0
