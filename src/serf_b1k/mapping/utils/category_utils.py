"""
Utility functions for extracting object categories from OmniGibson object names.
"""

import re
from typing import Dict, Set


def extract_category_from_name(name: str) -> str:
    """
    Extract object category from OmniGibson object name.

    OmniGibson naming convention:
        - {category}_{id}: e.g., 'dice_269', 'teddy_bear_267'
        - {category}_{hash}_{id}: e.g., 'wardrobe_sclnqc_0', 'bed_ivdnny_0'

    Examples:
        '/World/scene_0/dice_269/base_link/visuals' -> 'dice'
        '/World/scene_0/teddy_bear_267/base_link/visuals' -> 'teddy_bear'
        '/World/scene_0/wardrobe_sclnqc_0/base_link/visuals' -> 'wardrobe'
        '/World/scene_0/board_game_264/base_link/visuals' -> 'board_game'
        'background' -> 'background'

    Args:
        name: Full object path from OmniGibson

    Returns:
        Category string (e.g., 'dice', 'teddy_bear', 'wardrobe')
    """
    if name == 'background':
        return 'background'

    # Extract object part: 'dice_269', 'teddy_bear_267', 'wardrobe_sclnqc_0'
    match = re.search(r'/World/scene_\d+/([^/]+)/', name)
    if not match:
        return 'unknown'

    obj_part = match.group(1)  # e.g., 'dice_269', 'teddy_bear_267'

    # Remove trailing _<id> or _<hash>_<id>
    # Pattern 1: category_<id> (e.g., 'dice_269' -> 'dice')
    # Pattern 2: category_<hash>_<id> (e.g., 'wardrobe_sclnqc_0' -> 'wardrobe')

    # First, try to match pattern: ends with _<hash>_<number> or _<number>
    # Hash is typically 6 lowercase letters
    match_suffix = re.match(r'^(.+?)_([a-z]{6})_(\d+)$', obj_part)
    if match_suffix:
        # Pattern: category_hash_id
        return match_suffix.group(1)

    match_suffix = re.match(r'^(.+?)_(\d+)$', obj_part)
    if match_suffix:
        # Pattern: category_id
        return match_suffix.group(1)

    # Fallback: return the whole part
    return obj_part


def build_instance_to_category(id_to_name: Dict[str, str]) -> Dict[int, str]:
    """
    Build instance_id -> category mapping.

    Args:
        id_to_name: Dict from HDF5 attrs['instance_id_to_name']

    Returns:
        Dict mapping instance_id (int) -> category (str)
    """
    inst_to_cat = {}
    for inst_id, name in id_to_name.items():
        category = extract_category_from_name(name)
        inst_to_cat[int(inst_id)] = category
    return inst_to_cat


def build_category_to_instances(inst_to_cat: Dict[int, str]) -> Dict[str, Set[int]]:
    """
    Build category -> set of instance_ids mapping.

    Args:
        inst_to_cat: Dict mapping instance_id -> category

    Returns:
        Dict mapping category (str) -> set of instance_ids
    """
    cat_to_insts: Dict[str, Set[int]] = {}
    for inst_id, category in inst_to_cat.items():
        if category not in cat_to_insts:
            cat_to_insts[category] = set()
        cat_to_insts[category].add(inst_id)
    return cat_to_insts
