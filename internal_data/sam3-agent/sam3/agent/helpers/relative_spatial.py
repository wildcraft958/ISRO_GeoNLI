# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Relative spatial relationship utilities for filtering masks based on their position relative to other masks.
"""

from typing import Dict, List, Tuple

import numpy as np

from .spatial_filtering import compute_mask_centroids


def compute_mask_positions(
    pred_boxes: List[List[float]],
    orig_img_h: int,
    orig_img_w: int,
) -> np.ndarray:
    """
    Compute mask positions (centroids) for all masks.
    
    Args:
        pred_boxes: List of normalized boxes in xyxy format [x1, y1, x2, y2]
        orig_img_h: Original image height
        orig_img_w: Original image width
    
    Returns:
        Array of shape (N, 2) with [x, y] centroids in pixel coordinates
    """
    return compute_mask_centroids(pred_boxes, orig_img_h, orig_img_w)


def check_spatial_relationship(
    mask_pos: np.ndarray,
    reference_pos: np.ndarray,
    relationship: str,
    distance_threshold: float = 0.2,
    image_diagonal: float = 1.0,
) -> bool:
    """
    Check if a mask satisfies a spatial relationship to a reference mask.
    
    Args:
        mask_pos: Mask position [x, y] in pixel coordinates
        reference_pos: Reference mask position [x, y] in pixel coordinates
        relationship: One of 'left_of', 'right_of', 'above', 'below', 'near', 'far_from'
        distance_threshold: Distance threshold for 'near'/'far_from' (as ratio of image diagonal)
        image_diagonal: Image diagonal length in pixels
    
    Returns:
        True if relationship is satisfied
    """
    if relationship == "left_of":
        return mask_pos[0] < reference_pos[0]
    elif relationship == "right_of":
        return mask_pos[0] > reference_pos[0]
    elif relationship == "above":
        return mask_pos[1] < reference_pos[1]
    elif relationship == "below":
        return mask_pos[1] > reference_pos[1]
    elif relationship == "near":
        distance = np.sqrt(
            (mask_pos[0] - reference_pos[0]) ** 2 + 
            (mask_pos[1] - reference_pos[1]) ** 2
        )
        threshold_pixels = distance_threshold * image_diagonal
        return distance <= threshold_pixels
    elif relationship == "far_from":
        distance = np.sqrt(
            (mask_pos[0] - reference_pos[0]) ** 2 + 
            (mask_pos[1] - reference_pos[1]) ** 2
        )
        threshold_pixels = distance_threshold * image_diagonal
        return distance > threshold_pixels
    else:
        raise ValueError(f"Unknown relationship: {relationship}")


def apply_relative_filter(
    pred_boxes: List[List[float]],
    pred_masks: List,
    pred_scores: List[float],
    relationship: str,
    reference_mask_indices: List[int],
    distance_threshold: float,
    orig_img_h: int,
    orig_img_w: int,
) -> Tuple[List[int], str]:
    """
    Filter masks based on their position relative to reference masks.
    
    Args:
        pred_boxes: List of normalized boxes in xyxy format [x1, y1, x2, y2]
        pred_masks: List of RLE masks
        pred_scores: List of scores
        relationship: Spatial relationship string
        reference_mask_indices: List of reference mask indices (1-indexed from user, will be converted to 0-indexed)
        distance_threshold: Distance threshold for 'near'/'far_from'
        orig_img_h: Original image height
        orig_img_w: Original image width
    
    Returns:
        Tuple of (list of kept mask indices, description message)
    """
    num_masks = len(pred_boxes)
    
    if num_masks == 0:
        return [], "No masks available to filter."
    
    # Convert 1-indexed to 0-indexed
    reference_indices_0based = [idx - 1 for idx in reference_mask_indices]
    
    # Validate reference indices
    for idx in reference_indices_0based:
        if idx < 0 or idx >= num_masks:
            raise ValueError(f"Reference mask index {idx + 1} is out of range (1-{num_masks})")
    
    # Compute positions for all masks
    positions = compute_mask_positions(pred_boxes, orig_img_h, orig_img_w)
    
    # Compute image diagonal for distance calculations
    image_diagonal = np.sqrt(orig_img_h ** 2 + orig_img_w ** 2)
    
    # Filter masks: keep if relationship is satisfied with ANY reference mask (OR logic)
    kept_indices = []
    
    for i in range(num_masks):
        # Skip reference masks themselves
        if i in reference_indices_0based:
            continue
        
        mask_pos = positions[i]
        
        # Check relationship with any reference mask
        for ref_idx in reference_indices_0based:
            ref_pos = positions[ref_idx]
            if check_spatial_relationship(
                mask_pos, ref_pos, relationship, distance_threshold, image_diagonal
            ):
                kept_indices.append(i)
                break  # Found a match, no need to check other references
    
    # Build description
    ref_indices_str = ", ".join(map(str, reference_mask_indices))
    if len(kept_indices) == 0:
        desc = f"No masks found that are '{relationship}' relative to reference mask(s) {ref_indices_str}."
    else:
        kept_indices_1based = [i + 1 for i in kept_indices]
        desc = f"Found {len(kept_indices)} mask(s) (indices {kept_indices_1based}) that are '{relationship}' relative to reference mask(s) {ref_indices_str}."
    
    return kept_indices, desc


def filter_masks_by_relative_position(
    outputs: Dict,
    relationship: str,
    reference_mask_indices: List[int],
    distance_threshold: float = 0.2,
) -> Dict:
    """
    Filter masks in outputs dict based on relative spatial position.
    
    Args:
        outputs: Dict with pred_boxes, pred_masks, pred_scores, orig_img_h, orig_img_w
        relationship: Spatial relationship string
        reference_mask_indices: List of reference mask indices (1-indexed)
        distance_threshold: Distance threshold for 'near'/'far_from' (default: 0.2)
    
    Returns:
        Updated outputs dict with filtered masks
    """
    pred_boxes = outputs.get("pred_boxes", [])
    pred_masks = outputs.get("pred_masks", [])
    pred_scores = outputs.get("pred_scores", [])
    orig_img_h = int(outputs["orig_img_h"])
    orig_img_w = int(outputs["orig_img_w"])
    
    kept_indices, desc = apply_relative_filter(
        pred_boxes,
        pred_masks,
        pred_scores,
        relationship,
        reference_mask_indices,
        distance_threshold,
        orig_img_h,
        orig_img_w,
    )
    
    # Build filtered outputs
    filtered_outputs = dict(outputs)
    filtered_outputs["pred_boxes"] = [pred_boxes[i] for i in kept_indices]
    filtered_outputs["pred_masks"] = [pred_masks[i] for i in kept_indices]
    filtered_outputs["pred_scores"] = [pred_scores[i] for i in kept_indices]
    filtered_outputs["relative_spatial_filter_description"] = desc
    
    return filtered_outputs

