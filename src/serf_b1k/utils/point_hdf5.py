from __future__ import annotations

import json
from typing import Any, Optional, Sequence

import h5py
import numpy as np


INITIAL_RGB_DATASET_KEYS = ("rgbs", "colors", "rgb")
FEATURE_DATASET_KEYS = ("initial_features", "features", "latent_features", "latent")


def read_optional_dataset(
    f: h5py.File,
    keys: Sequence[str],
) -> Optional[np.ndarray]:
    for key in keys:
        if key in f:
            return f[key][:]
    return None


def read_instance_id_to_name(f: h5py.File) -> Optional[dict[str, Any]]:
    if "instance_id_to_name" not in f.attrs:
        return None
    return json.loads(f.attrs["instance_id_to_name"])


def read_initial_point_payload(f: h5py.File) -> dict[str, Any]:
    if "initial_points" not in f:
        raise KeyError("missing required dataset: initial_points")
    if "initial_instance_ids" not in f:
        raise KeyError("missing required dataset: initial_instance_ids")

    xyz = f["initial_points"][:].astype(np.float32)
    ids = f["initial_instance_ids"][:].astype(np.int64)

    rgb = read_optional_dataset(f, INITIAL_RGB_DATASET_KEYS)
    if rgb is not None:
        rgb = rgb.astype(np.float32)

    feat = read_optional_dataset(f, FEATURE_DATASET_KEYS)
    if feat is not None:
        assert feat.shape[0] == xyz.shape[0], "feat and xyz must have same number of points"
        feat = feat.astype(np.float32)

    return {
        "xyz": xyz,
        "ids": ids,
        "rgb": rgb,
        "feat": feat,
        "id_to_name": read_instance_id_to_name(f),
    }
