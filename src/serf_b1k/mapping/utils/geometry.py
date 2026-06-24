from typing import Tuple, Union

import numpy as np
import torch

__all__ = [
    "calculate_intrinsics",
    "unproject_depth_to_world",
    "solve_rigid_transform",
    "refine_registration_icp",
]

# --------------------------------------------------------------------------- #
#  Camera intrinsics                                                          #
# --------------------------------------------------------------------------- #

def calculate_intrinsics(
    image_width: int,
    image_height: int,
    horizontal_aperture: float,
    focal_length: float,
) -> np.ndarray:
    """
    Calculates camera intrinsic matrix from camera parameters.

    Args:
        image_width: Width of the output image in pixels
        image_height: Height of the output image in pixels
        horizontal_aperture: Horizontal aperture of the camera sensor
        focal_length: Focal length of the camera lens

    Returns:
        3x3 intrinsic matrix K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    """
    vertical_aperture = horizontal_aperture * (image_height / image_width)

    cx = image_width / 2.0
    cy = image_height / 2.0
    fx = focal_length * image_width / horizontal_aperture
    fy = focal_length * image_height / vertical_aperture

    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# --------------------------------------------------------------------------- #
#  3-D coordinates from depth map                                             #
# --------------------------------------------------------------------------- #

def unproject_depth_to_world(
    depth: torch.Tensor,
    cam_to_world: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    original_height: int = None,
    original_width: int = None,
    original_size: int = None,
):
    """
    Compute 3D world coordinates from depth map.

    Args:
        depth: Depth map tensor (B, H, W) or (B, 1, H, W)
        cam_to_world: Camera-to-world transform matrix (B, 1, 3, 4) or (B, 3, 4)
        fx, fy, cx, cy: Original camera intrinsics (before any resizing)
        original_height, original_width: Original image dimensions that intrinsics correspond to
        original_size: Original image size for square images (alternative to height/width)

    Returns:
        coords_world: World coordinates (B, 3, H, W)
    """
    # Handle original_size for square images
    if original_size is not None:
        original_height = original_size
        original_width = original_size
    elif original_height is None or original_width is None:
        original_height = 480
        original_width = 480

    device = depth.device

    if depth.dim() == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)

    B, H_feat, W_feat = depth.shape

    # Scale intrinsics from original resolution to current depth map resolution
    scale_x = W_feat / float(original_width)
    scale_y = H_feat / float(original_height)
    fx_scaled = fx * scale_x
    fy_scaled = fy * scale_y
    cx_scaled = cx * scale_x
    cy_scaled = cy * scale_y

    u = torch.arange(W_feat, device=device).view(1, -1).expand(H_feat, W_feat) + 0.5
    v = torch.arange(H_feat, device=device).view(-1, 1).expand(H_feat, W_feat) + 0.5
    u = u.unsqueeze(0).expand(B, -1, -1)
    v = v.unsqueeze(0).expand(B, -1, -1)

    x_cam = (u - cx_scaled) * depth / fx_scaled
    y_cam = (v - cy_scaled) * depth / fy_scaled
    z_cam = depth
    ones = torch.ones_like(z_cam)
    coords_hom = torch.stack([x_cam, y_cam, z_cam, ones], dim=1)  # (B, 4, H, W)

    cam_to_world = cam_to_world.squeeze(1)
    ones_row = torch.tensor([0, 0, 0, 1], device=device, dtype=cam_to_world.dtype).view(1, 1, 4)
    ones_row = ones_row.expand(B, 1, 4)
    cam_to_world_4x4 = torch.cat([cam_to_world, ones_row], dim=1)

    coords_hom_flat = coords_hom.view(B, 4, -1)
    world_coords_hom = torch.bmm(cam_to_world_4x4, coords_hom_flat)
    coords_world = world_coords_hom[:, :3, :].view(B, 3, H_feat, W_feat)
    return coords_world


# --------------------------------------------------------------------------- #
#  Rigid registration — Open3D Fast Global Registration (FGR)                 #
# --------------------------------------------------------------------------- #

def solve_rigid_transform(
    A: Union[np.ndarray, torch.Tensor],
    B: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve rigid transform using Open3D Fast Global Registration (FGR).

    Uses artificial per-point features to enforce the known correspondences
    so that FGR acts as a robust rigid-transform solver (replacing RANSAC).

    Args:
        A: (N, 3) source points — numpy array or torch tensor.
        B: (N, 3) target (corresponding) points — numpy array or torch tensor.
        threshold: Maximum correspondence distance for FGR.

    Returns:
        Tuple of ``(R, t_vec)`` — (3, 3) rotation and (3,) translation as
        numpy ``float64`` arrays.
    """
    import open3d as o3d

    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor):
        B = B.detach().cpu().numpy()

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n = A.shape[0]

    pcd_A = o3d.geometry.PointCloud()
    pcd_A.points = o3d.utility.Vector3dVector(A)

    pcd_B = o3d.geometry.PointCloud()
    pcd_B.points = o3d.utility.Vector3dVector(B)

    # Fake FPFH-shaped features (dim 33) — identical for source & target so
    # FGR establishes 1-to-1 correspondences by index.
    # Use a local RNG to avoid corrupting global random state.
    rng = np.random.RandomState(0)
    feats = rng.rand(33, n).astype(np.float64)

    feat_A = o3d.pipelines.registration.Feature()
    feat_A.data = feats

    feat_B = o3d.pipelines.registration.Feature()
    feat_B.data = feats

    option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=threshold,
    )

    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        pcd_A, pcd_B, feat_A, feat_B, option,
    )

    T = np.asarray(result.transformation)
    return T[:3, :3], T[:3, 3]


# --------------------------------------------------------------------------- #
#  ICP refinement — Open3D point-to-point ICP                                 #
# --------------------------------------------------------------------------- #

def refine_registration_icp(
    source_points: Union[np.ndarray, torch.Tensor],
    target_points: Union[np.ndarray, torch.Tensor],
    init_R: Union[np.ndarray, torch.Tensor],
    init_t: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.02,
    max_iterations: int = 50,
    return_metrics: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, float, float]]:
    """Refine alignment using Open3D point-to-point ICP.

    Args:
        source_points: (N, 3) model/particle points — numpy array or torch tensor.
        target_points: (M, 3) observation points — numpy array or torch tensor.
        init_R: (3, 3) initial rotation — numpy array or torch tensor.
        init_t: (3,) initial translation — numpy array or torch tensor.
        threshold: Max correspondence distance for ICP.
        max_iterations: Maximum ICP iterations.
        return_metrics: If True, also return fitness and inlier RMSE.

    Returns:
        ``(R, t_vec)`` when ``return_metrics`` is False.
        ``(R, t_vec, fitness, inlier_rmse)`` when ``return_metrics`` is True.
    """
    import open3d as o3d

    if isinstance(source_points, torch.Tensor):
        source_points = source_points.detach().cpu().numpy()
    if isinstance(target_points, torch.Tensor):
        target_points = target_points.detach().cpu().numpy()
    if isinstance(init_R, torch.Tensor):
        init_R = init_R.detach().cpu().numpy()
    if isinstance(init_t, torch.Tensor):
        init_t = init_t.detach().cpu().numpy()

    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)

    # Build 4×4 initial transformation matrix
    init_trans = np.eye(4, dtype=np.float64)
    init_trans[:3, :3] = np.asarray(init_R, dtype=np.float64)
    init_trans[:3, 3] = np.asarray(init_t, dtype=np.float64).ravel()

    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(source_points)

    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(target_points)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_source,
        pcd_target,
        threshold,
        init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )

    T = np.asarray(reg_p2p.transformation)
    R, t_vec = T[:3, :3], T[:3, 3]
    if return_metrics:
        return R, t_vec, reg_p2p.fitness, reg_p2p.inlier_rmse
    return R, t_vec
