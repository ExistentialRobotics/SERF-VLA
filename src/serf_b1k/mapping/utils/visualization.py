import numpy as np
import torch
from typing import Dict, Optional


class TorchPCA:
    """
    GPU-accelerated PCA following scikit-learn's implementation.

    Uses full SVD via torch.linalg.svd (equivalent to sklearn's svd_solver='full').
    """

    def __init__(self, n_components=None, whiten=False):
        self.n_components = n_components
        self.whiten = whiten
        self.mean_ = None
        self.components_ = None  # (n_components, n_features) like sklearn
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_samples_ = None
        self.n_features_ = None

    def fit(self, X):
        """
        Fit PCA on X.

        Args:
            X: (n_samples, n_features) tensor
        """
        self.n_samples_, self.n_features_ = X.shape

        # Center the data
        self.mean_ = X.mean(dim=0)
        X_centered = X - self.mean_

        # Full SVD
        # X = U @ S @ Vt
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

        # Determine n_components
        max_components = min(self.n_samples_, self.n_features_)
        if self.n_components is None:
            n_components = max_components
        else:
            n_components = min(self.n_components, max_components)

        # SVD sign flip (sklearn convention): ensure deterministic sign
        # by making the largest absolute value in each component positive.
        max_abs_idx = torch.argmax(torch.abs(Vt[:n_components]), dim=1)
        signs = torch.sign(Vt[:n_components][torch.arange(n_components), max_abs_idx])
        Vt[:n_components] *= signs.unsqueeze(1)

        # Components are first n_components rows of Vt
        self.components_ = Vt[:n_components]  # (n_components, n_features)

        # Explained variance (sklearn uses n_samples - 1 for unbiased estimate)
        self.explained_variance_ = (S[:n_components] ** 2) / (self.n_samples_ - 1)

        total_var = (S ** 2).sum() / (self.n_samples_ - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        # Store singular values for whitening
        self._S = S[:n_components]

        return self

    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        Args:
            X: (n_samples, n_features) tensor

        Returns:
            (n_samples, n_components) numpy array
        """
        X_centered = X - self.mean_

        # Project: X_centered @ components.T
        X_transformed = X_centered @ self.components_.T

        if self.whiten:
            X_transformed = X_transformed / torch.sqrt(self.explained_variance_)

        # Pad with zeros if fewer components than requested (for RGB visualization)
        actual_components = X_transformed.shape[1]
        if self.n_components is not None and actual_components < self.n_components:
            padding = torch.zeros(X.shape[0], self.n_components - actual_components, device=X.device)
            X_transformed = torch.cat([X_transformed, padding], dim=1)

        return X_transformed.detach().cpu().numpy()

    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


def run_pca_visualization(server, point_map, decoder, env_name, epoch_label, device, config,
                          filtered_instance_ids=None, include_instance_ids=None):
    """
    Visualize neural point map features using PCA coloring.

    Args:
        server: Viser server instance
        point_map: NeuralPointMap instance
        decoder: Feature decoder
        env_name: Environment name
        epoch_label: Current epoch
        device: torch device
        config: Config dict with scene_min, scene_max, and pca_visualization settings
        filtered_instance_ids: Set of instance IDs to exclude from PCA computation.
            Points with these instance IDs will be filtered out before PCA.
            Typically obtained from dataset.background_instance_ids[env_name].
        include_instance_ids: Set of instance IDs to include (whitelist mode).
            If provided, only points with these instance IDs will be visualized.
            Takes precedence over filtered_instance_ids.
    """
    # Get visualization config (with defaults for backward compatibility)
    vis_config = config.get('visualization', {})
    z_threshold = vis_config.get('z_threshold', 2.5)
    max_points = vis_config.get('max_points', 1000000)
    point_size = vis_config.get('point_size', 0.005)
    side_by_side_offset = vis_config.get('side_by_side_offset', 1.2)
    n_components = vis_config.get('n_components', 3)
    whiten = vis_config.get('whiten', True)
    quantile_low = vis_config.get('quantile_low', 0.02)
    quantile_high = vis_config.get('quantile_high', 0.98)
    filter_background = vis_config.get('filter_background', False)

    x_center = (config['scene_min'][0] + config['scene_max'][0]) / 2.0
    y_center = (config['scene_min'][1] + config['scene_max'][1]) / 2.0

    print(f"\n[VIS] Running PCA visualization for {env_name} (epoch: {epoch_label})...")

    coords = point_map.map_points
    features = point_map.map_features
    instance_ids = point_map.map_instance_ids if hasattr(point_map, 'map_instance_ids') else None
    num_points = coords.shape[0]

    if num_points == 0:
        print("[VIS] No neural points in map; skipping.")
        return

    # Whitelist mode: if include_instance_ids is provided, only include those points
    if include_instance_ids is not None and instance_ids is not None and len(include_instance_ids) > 0:
        include_ids_tensor = torch.tensor(list(include_instance_ids), device=device, dtype=instance_ids.dtype)
        include_mask = torch.isin(instance_ids, include_ids_tensor)
        num_included = include_mask.sum().item()
        print(f"[VIS] Including only {num_included} points from {len(include_instance_ids)} specified instance IDs")

        coords = coords[include_mask]
        features = features[include_mask]
        if hasattr(point_map, 'map_instance_ids'):
            instance_ids = instance_ids[include_mask]
        num_points = coords.shape[0]

        if num_points == 0:
            print("[VIS] No points found for specified instance IDs; skipping.")
            return

    # Filter out background/structural elements before PCA if filter_background is enabled
    elif filter_background and filtered_instance_ids is not None and instance_ids is not None and len(filtered_instance_ids) > 0:
        # Create mask for points to keep (not in filtered_instance_ids)
        filtered_ids_tensor = torch.tensor(list(filtered_instance_ids), device=device, dtype=instance_ids.dtype)
        # Check which instance IDs are NOT in filtered_instance_ids
        filter_mask = ~torch.isin(instance_ids, filtered_ids_tensor)
        num_filtered = (~filter_mask).sum().item()
        print(f"[VIS] Filtering {num_filtered} points from {len(filtered_instance_ids)} background/structural instance IDs")

        coords = coords[filter_mask]
        features = features[filter_mask]
        num_points = coords.shape[0]

        if num_points == 0:
            print("[VIS] No points remaining after filtering; skipping.")
            return

    if num_points > max_points:
        indices = torch.randperm(num_points, device=device)[:max_points]
        coords = coords[indices]
        features = features[indices]

    # PCA on raw neural point features (before decoder)
    pca_before = TorchPCA(n_components=n_components, whiten=whiten)
    colors_before = pca_before.fit_transform(features.detach())

    # PCA on decoded features (after decoder)
    with torch.no_grad():
        decoded = decoder(features)
    pca_after = TorchPCA(n_components=n_components, whiten=whiten)
    colors_after = pca_after.fit_transform(decoded)

    # Robust normalization for both
    def normalize_colors(colors):
        q_low = np.quantile(colors, quantile_low, axis=0)
        q_high = np.quantile(colors, quantile_high, axis=0)
        colors = np.clip(colors, q_low, q_high)
        return (colors - q_low) / (q_high - q_low + 1e-8)

    colors_before = normalize_colors(colors_before)
    colors_after = normalize_colors(colors_after)

    coords_np = coords.cpu().numpy()

    mask = coords_np[:, 2] <= z_threshold
    coords_np = coords_np[mask]
    colors_before = colors_before[mask]
    colors_after = colors_after[mask]

    if len(coords_np) == 0:
        print("[VIS] No neural points after Z filtering; skipping.")
        return

    coords_np[:, 0] -= x_center
    coords_np[:, 1] -= y_center

    # Offset for side-by-side visualization
    x_offset = (config['scene_max'][0] - config['scene_min'][0]) * side_by_side_offset

    coords_before = coords_np.copy()
    coords_after = coords_np.copy()
    coords_after[:, 0] += x_offset

    print(f"[VIS] Displaying {len(coords_np)} neural points with PCA for {env_name}")
    print(f"[VIS]   - Left: before decoder, Right: after decoder")

    server.add_point_cloud(
        name=f"/pca/{env_name}/before_decoder",
        points=coords_before,
        colors=(colors_before * 255).astype(np.uint8),
        point_size=point_size
    )
    server.add_point_cloud(
        name=f"/pca/{env_name}/after_decoder",
        points=coords_after,
        colors=(colors_after * 255).astype(np.uint8),
        point_size=point_size
    )


def run_robot_pca_visualization(server, coords, features, decoder, name, epoch_label, device, config):
    """Visualize robot surface point features using PCA coloring.

    Args:
        server: Viser server instance.
        coords: (N, 3) surface point coordinates.
        features: (N, D) learned per-point features.
        decoder: Feature decoder (nn.Module).
        name: Display name for the point cloud.
        epoch_label: Current epoch number.
        device: Torch device string.
        config: Config dict with optional 'visualization' section.
    """
    vis_config = config.get('visualization', {})
    max_points = vis_config.get('max_points', 1000000)
    point_size = vis_config.get('point_size', 0.005)
    side_by_side_offset = vis_config.get('side_by_side_offset', 1.2)
    n_components = vis_config.get('n_components', 3)
    whiten = vis_config.get('whiten', True)
    quantile_low = vis_config.get('quantile_low', 0.02)
    quantile_high = vis_config.get('quantile_high', 0.98)

    num_points = coords.shape[0]
    if num_points == 0:
        print("[VIS] No robot surface points; skipping.")
        return

    print(f"\n[VIS] Running PCA visualization for {name} (epoch: {epoch_label})...")

    if num_points > max_points:
        indices = torch.randperm(num_points, device=device)[:max_points]
        coords = coords[indices]
        features = features[indices]

    # PCA on raw neural point features (before decoder)
    pca_before = TorchPCA(n_components=n_components, whiten=whiten)
    colors_before = pca_before.fit_transform(features.detach())

    # PCA on decoded features (after decoder)
    with torch.no_grad():
        decoded = decoder(features)
    pca_after = TorchPCA(n_components=n_components, whiten=whiten)
    colors_after = pca_after.fit_transform(decoded)

    # Robust normalization
    def normalize_colors(colors):
        q_low = np.quantile(colors, quantile_low, axis=0)
        q_high = np.quantile(colors, quantile_high, axis=0)
        colors = np.clip(colors, q_low, q_high)
        return (colors - q_low) / (q_high - q_low + 1e-8)

    colors_before = normalize_colors(colors_before)
    colors_after = normalize_colors(colors_after)

    # Center on point cloud mean
    coords_np = coords.detach().cpu().numpy()
    center = coords_np.mean(axis=0)
    coords_np -= center

    # Side-by-side offset from point cloud extent
    x_extent = coords_np[:, 0].max() - coords_np[:, 0].min()
    x_offset = x_extent * side_by_side_offset

    coords_before = coords_np.copy()
    coords_after = coords_np.copy()
    coords_after[:, 0] += x_offset

    print(f"[VIS] Displaying {len(coords_np)} robot surface points with PCA")
    print(f"[VIS]   - Left: before decoder, Right: after decoder")

    server.add_point_cloud(
        name=f"/pca/{name}/before_decoder",
        points=coords_before,
        colors=(colors_before * 255).astype(np.uint8),
        point_size=point_size,
    )
    server.add_point_cloud(
        name=f"/pca/{name}/after_decoder",
        points=coords_after,
        colors=(colors_after * 255).astype(np.uint8),
        point_size=point_size,
    )


def run_joint_pca_visualization(
    server,
    point_maps: dict,
    robot_points,
    surface_cache_0: torch.Tensor,
    decoder,
    epoch_label: int,
    device: str,
    config: dict,
) -> None:
    """Fit PCA jointly on env + robot features and display both.

    Args:
        server: Viser server instance.
        point_maps: Dict[env_name, NeuralPointMap] — env point maps.
        robot_points: Robot neural point map (has .features attribute).
        surface_cache_0: (N, 3) canonical-pose surface points (state 0).
        decoder: Shared decoder.
        epoch_label: Current epoch number.
        device: Torch device string.
        config: Config dict with visualization settings.
    """
    vis_config = config.get("visualization", {})
    max_points = vis_config.get("max_points", 1000000)
    point_size = vis_config.get("point_size", 0.005)
    n_components = vis_config.get("n_components", 3)
    whiten = vis_config.get("whiten", True)
    quantile_low = vis_config.get("quantile_low", 0.02)
    quantile_high = vis_config.get("quantile_high", 0.98)
    z_threshold = vis_config.get("z_threshold", 2.5)
    side_by_side_offset = vis_config.get("side_by_side_offset", 1.2)

    print(f"\n[VIS] Running joint PCA visualization (epoch: {epoch_label})...")

    # --- Collect per-episode data ---
    env_episodes = []  # list of (env_name, coords_tensor, feats_tensor)
    for env_name, pm in point_maps.items():
        if pm.map_points.shape[0] == 0:
            continue
        coords = pm.map_points.detach()
        feats = pm.map_features.detach()
        # Per-episode subsample
        if coords.shape[0] > max_points:
            idx = torch.randperm(coords.shape[0], device=device)[:max_points]
            coords = coords[idx]
            feats = feats[idx]
        env_episodes.append((env_name, coords, feats))

    if not env_episodes:
        print("[VIS] No env points; skipping.")
        return

    # --- Robot features ---
    robot_feats = robot_points.features.detach()
    robot_coords = surface_cache_0.detach()

    # --- Fit PCA jointly on all env + robot features ---
    all_env_feats = torch.cat([feats for _, _, feats in env_episodes], dim=0)
    all_feats = torch.cat([all_env_feats, robot_feats], dim=0)
    pca = TorchPCA(n_components=n_components, whiten=whiten)
    pca.fit(all_feats)

    # --- Joint quantile normalization ---
    all_colors = np.concatenate(
        [pca.transform(feats) for _, _, feats in env_episodes] + [pca.transform(robot_feats)],
        axis=0,
    )
    q_low = np.quantile(all_colors, quantile_low, axis=0)
    q_high = np.quantile(all_colors, quantile_high, axis=0)

    def normalize(c):
        c = np.clip(c, q_low, q_high)
        return (c - q_low) / (q_high - q_low + 1e-8)

    # --- Per-episode display (offset along X axis) ---
    x_center = (config["scene_min"][0] + config["scene_max"][0]) / 2.0
    y_center = (config["scene_min"][1] + config["scene_max"][1]) / 2.0
    x_extent = config["scene_max"][0] - config["scene_min"][0]
    step = x_extent * side_by_side_offset

    total_env_pts = 0
    for i, (env_name, coords_t, feats_t) in enumerate(env_episodes):
        colors = normalize(pca.transform(feats_t))
        coords_np = coords_t.cpu().numpy()
        z_mask = coords_np[:, 2] <= z_threshold
        coords_np = coords_np[z_mask]
        colors = colors[z_mask]
        if len(coords_np) == 0:
            continue
        coords_np[:, 0] -= x_center
        coords_np[:, 1] -= y_center
        coords_np[:, 0] += i * step
        server.add_point_cloud(
            name=f"/pca/joint/env/{env_name}",
            points=coords_np,
            colors=(colors * 255).astype(np.uint8),
            point_size=point_size,
        )
        total_env_pts += len(coords_np)

    # --- Robot display (offset past all episodes) ---
    robot_np = robot_coords.cpu().numpy()
    robot_center = robot_np.mean(axis=0)
    robot_np_display = robot_np - robot_center
    robot_np_display[:, 0] += len(env_episodes) * step

    colors_robot = normalize(pca.transform(robot_feats))
    server.add_point_cloud(
        name="/pca/joint/robot",
        points=robot_np_display,
        colors=(colors_robot * 255).astype(np.uint8),
        point_size=point_size,
    )

    print(f"[VIS] Displayed {total_env_pts} env ({len(env_episodes)} episodes) + {len(robot_np)} robot points with joint PCA")


def apply_camera_extrinsic(vis, camera_extrinsic: Optional[np.ndarray]) -> None:
    """Apply a fixed camera extrinsic to an Open3D visualizer.

    Args:
        vis: Open3D Visualizer instance.
        camera_extrinsic: (4, 4) extrinsic matrix. No-op if None.
    """
    if camera_extrinsic is None:
        return
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    params.extrinsic = camera_extrinsic
    ctr.convert_from_pinhole_camera_parameters(params, False)


def draw_geometries_with_key_callbacks(
    geometries,
    window_name: str = "Open3D",
    camera_extrinsic: np.ndarray = None,
) -> None:
    """Open3D visualizer with key callback to print camera pose (press 'P').

    Args:
        geometries: List of Open3D geometries to display.
        window_name: Window title.
        camera_extrinsic: Optional (4, 4) initial camera extrinsic matrix.
    """
    import open3d as o3d

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name)

    for geometry in geometries:
        vis.add_geometry(geometry)

    def print_camera_pose(vis):
        ctr = vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        extrinsic = np.asarray(params.extrinsic)
        lines = ["  camera_extrinsic:"]
        for row in extrinsic:
            vals = ", ".join(f"{v:.8e}" for v in row)
            lines.append(f"    - [{vals}]")
        print(f"\n[{window_name}] Camera Extrinsic (paste into tracking.yaml):")
        print("\n".join(lines))
        return False

    vis.register_key_callback(ord("P"), print_camera_pose)

    if camera_extrinsic is not None:
        ctr = vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        params.extrinsic = camera_extrinsic
        ctr.convert_from_pinhole_camera_parameters(params, False)

    vis.run()
    vis.destroy_window()


def compute_pca_colors(
    neural_points,
    excluded_train_ids_tensor: torch.Tensor,
) -> tuple["TorchPCA", np.ndarray, np.ndarray, np.ndarray]:
    """Compute PCA-based colors for point cloud visualization.

    Fits a 3-component PCA on foreground point features (excluding background
    categories) and projects all points into the resulting color space.

    Args:
        neural_points: Object with ``map_features`` and ``map_instance_ids``
            attributes (e.g. ``NeuralPointMap``).
        excluded_train_ids_tensor: Tensor of instance IDs to exclude from PCA
            fitting (background / structural categories).

    Returns:
        pca: Fitted TorchPCA object (for coloring new points later).
        pca_colors_arr: (N, 3) float32 normalized colors.
        color_min: (3,) normalization minimum.
        color_max: (3,) normalization maximum.
    """
    print("[INIT] Computing PCA colors (excluding background categories)...")
    all_latents = neural_points.map_features.detach().cpu()
    all_inst_ids = neural_points.map_instance_ids.cpu()
    include_mask = ~torch.isin(all_inst_ids, excluded_train_ids_tensor.cpu())

    pca = TorchPCA(n_components=3)
    pca.fit_transform(all_latents[include_mask])
    pca_colors_all = pca.transform(all_latents)

    pca_fg = pca_colors_all[include_mask]
    color_min = pca_fg.min(axis=0)
    color_max = pca_fg.max(axis=0)
    pca_colors_all = (pca_colors_all - color_min) / (color_max - color_min + 1e-8)
    pca_colors_arr = np.asarray(pca_colors_all, dtype=np.float32)

    return pca, pca_colors_arr, color_min, color_max


def build_link_colors(sampler: "RobotSurfaceSampler") -> np.ndarray:
    """Build per-point RGB colors with a distinct hue per geometry node.

    Uses golden-ratio hue spacing for maximal perceptual separation.

    Args:
        sampler: ``RobotSurfaceSampler`` with ``n_total_points``, ``n_geoms``,
            and ``get_geom_slice()`` method.

    Returns:
        (N, 3) uint8 RGB colors.
    """
    import colorsys

    golden_ratio = (1 + 5**0.5) / 2
    colors = np.empty((sampler.n_total_points, 3), dtype=np.float64)

    for i in range(sampler.n_geoms):
        sl = sampler.get_geom_slice(i)
        h = (i / golden_ratio) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.80)
        colors[sl] = [r, g, b]

    return (colors * 255).clip(0, 255).astype(np.uint8)
