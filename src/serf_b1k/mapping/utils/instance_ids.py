"""Shared instance ID utilities for mapping and tracking."""

import json
import logging
from typing import Dict, Iterable, List, Set

import h5py

logger = logging.getLogger(__name__)


def load_instance_id_map_from_hdf5(hdf5_file: h5py.File) -> Dict[str, str]:
    """Load instance_id_to_name from HDF5 file attributes.

    Args:
        hdf5_file: Open HDF5 file handle.

    Returns:
        Dict mapping instance ID strings to object names.

    Raises:
        KeyError: If 'instance_id_to_name' attribute is not found.
    """
    if "instance_id_to_name" not in hdf5_file.attrs:
        raise KeyError(
            f"'instance_id_to_name' attr not found in {hdf5_file.filename}"
        )
    return json.loads(hdf5_file.attrs["instance_id_to_name"])


def load_instance_id_map_from_json(json_path: str) -> Dict[str, str]:
    """Load instance_id_to_name from a JSON file.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Dict mapping instance ID strings to object names.
    """
    with open(json_path, "r") as f:
        return json.load(f)


def filter_ids_by_keywords(
    id_to_name: Dict[str, str], keywords: Iterable[str]
) -> Set[int]:
    """Return instance IDs whose names contain any of the given keywords.

    Args:
        id_to_name: Dict mapping instance ID strings to object names.
        keywords: Iterable of lowercase keywords to match against.

    Returns:
        Set of integer instance IDs matching any keyword.
    """
    keywords = set(keywords)
    result = set()
    for id_str, name in id_to_name.items():
        name_lower = name.lower()
        if any(kw in name_lower for kw in keywords):
            result.add(int(id_str))
    return result


def resolve_names_to_ids(
    id_to_name: Dict[str, str], target_names: List[str]
) -> List[int]:
    """Resolve target instance names to IDs by substring match.

    Args:
        id_to_name: Dict mapping instance ID strings to object names.
        target_names: List of name substrings to search for.

    Returns:
        List of matched integer instance IDs.
    """
    matched_ids = []
    for target_name in target_names:
        found = [int(oid) for oid, name in id_to_name.items() if target_name in name]
        if found:
            matched_ids.extend(found)
            logger.info("[ID RESOLVE] '%s' -> IDs %s", target_name, found)
        else:
            logger.warning("[ID RESOLVE] '%s' not found in ID map.", target_name)
    return matched_ids


def build_cross_episode_id_mapping(
    online_id_map: Dict[str, str],
    training_id_map: Dict[str, str],
) -> Dict[int, int]:
    """Build online_id -> training_id mapping by matching object names.

    When the same object has different instance IDs across episodes,
    this maps by name to enable cross-episode tracking.

    Args:
        online_id_map: Dict mapping online instance ID strings to object names.
        training_id_map: Dict mapping training instance ID strings to object names.

    Returns:
        Dict mapping online integer IDs to training integer IDs.
    """
    name_to_train_id = {name: int(tid) for tid, name in training_id_map.items()}
    mapping = {}
    for online_id_str, name in online_id_map.items():
        if name in name_to_train_id:
            mapping[int(online_id_str)] = name_to_train_id[name]
    return mapping
