# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Spatial filtering utilities for filtering masks by their position in the image.
"""

from typing import Dict, List, Tuple

import numpy as np

try:
    from pycocotools import mask as mask_utils
except Exception:
    mask_utils = None


def compute_mask_centroids(
    pred_boxes: List[List[float]], 
    orig_img_h: int, 
    orig_img_w: int
) -> np.ndarray:
    """
    Compute centroids from normalized bounding boxes.
    
    Args:
        pred_boxes: List of normalized boxes in xyxy format [x1, y1, x2, y2]
        orig_img_h: Original image height in pixels
        orig_img_w: Original image width in pixels
    
    Returns:
        Array of shape (N, 2) with [x, y] centroids in pixel coordinates
    """
    if not pred_boxes:
        return np.array([]).reshape(0, 2)
    
    boxes = np.array(pred_boxes)  # (N, 4) in normalized xyxy format [x1, y1, x2, y2]
    # Convert normalized xyxy to pixel coordinates and compute centroids
    # xyxy format: [x1, y1, x2, y2] all normalized [0, 1]
    x1_pixel = boxes[:, 0] * orig_img_w
    y1_pixel = boxes[:, 1] * orig_img_h
    x2_pixel = boxes[:, 2] * orig_img_w
    y2_pixel = boxes[:, 3] * orig_img_h
    
    # Compute centroids: center of bounding box
    centroids_x = (x1_pixel + x2_pixel) / 2.0
    centroids_y = (y1_pixel + y2_pixel) / 2.0
    
    centroids = np.stack([centroids_x, centroids_y], axis=1)  # (N, 2)
    return centroids


def apply_spatial_filter(
    pred_boxes: List[List[float]],
    pred_masks: List,
    pred_scores: List[float],
    spatial_criteria: str,
    orig_img_h: int,
    orig_img_w: int,
) -> Tuple[List[int], str]:
    """
    Filter masks based on spatial criteria.
    
    Args:
        pred_boxes: List of normalized boxes in xyxy format [x1, y1, x2, y2]
        pred_masks: List of RLE masks
        pred_scores: List of scores
        spatial_criteria: One of the spatial criteria strings
        orig_img_h: Original image height
        orig_img_w: Original image width
    
    Returns:
        Tuple of (list of kept mask indices, description message)
    """
    num_masks = len(pred_boxes)
    
    if num_masks == 0:
        return [], "No masks available to filter."
    
    if num_masks == 1:
        return [0], "Only one mask available, keeping it."
    
    # Compute centroids
    centroids = compute_mask_centroids(pred_boxes, orig_img_h, orig_img_w)
    
    # Image center
    img_center_x = orig_img_w / 2.0
    img_center_y = orig_img_h / 2.0
    
    kept_indices = []
    
    if spatial_criteria == "leftmost":
        # Mask with minimum x coordinate
        leftmost_idx = np.argmin(centroids[:, 0])
        kept_indices = [leftmost_idx]
        desc = f"Selected leftmost mask (index {leftmost_idx + 1})."
        
    elif spatial_criteria == "rightmost":
        # Mask with maximum x coordinate
        rightmost_idx = np.argmax(centroids[:, 0])
        kept_indices = [rightmost_idx]
        desc = f"Selected rightmost mask (index {rightmost_idx + 1})."
        
    elif spatial_criteria == "topmost":
        # Mask with minimum y coordinate
        topmost_idx = np.argmin(centroids[:, 1])
        kept_indices = [topmost_idx]
        desc = f"Selected topmost mask (index {topmost_idx + 1})."
        
    elif spatial_criteria == "bottommost":
        # Mask with maximum y coordinate
        bottommost_idx = np.argmax(centroids[:, 1])
        kept_indices = [bottommost_idx]
        desc = f"Selected bottommost mask (index {bottommost_idx + 1})."
        
    elif spatial_criteria == "center":
        # Mask closest to image center
        distances_to_center = np.sqrt(
            (centroids[:, 0] - img_center_x) ** 2 + 
            (centroids[:, 1] - img_center_y) ** 2
        )
        center_idx = np.argmin(distances_to_center)
        kept_indices = [center_idx]
        desc = f"Selected mask closest to center (index {center_idx + 1})."
        
    elif spatial_criteria == "second_from_left":
        # Second mask from left
        sorted_by_x = np.argsort(centroids[:, 0])
        if len(sorted_by_x) >= 2:
            second_idx = sorted_by_x[1]
            kept_indices = [second_idx]
            desc = f"Selected second mask from left (index {second_idx + 1})."
        else:
            kept_indices = [sorted_by_x[0]]
            desc = f"Only one mask available, selected it (index {sorted_by_x[0] + 1})."
            
    elif spatial_criteria == "second_from_right":
        # Second mask from right
        sorted_by_x = np.argsort(centroids[:, 0])[::-1]  # Descending
        if len(sorted_by_x) >= 2:
            second_idx = sorted_by_x[1]
            kept_indices = [second_idx]
            desc = f"Selected second mask from right (index {second_idx + 1})."
        else:
            kept_indices = [sorted_by_x[0]]
            desc = f"Only one mask available, selected it (index {sorted_by_x[0] + 1})."
            
    elif spatial_criteria == "third_from_left":
        # Third mask from left
        sorted_by_x = np.argsort(centroids[:, 0])
        if len(sorted_by_x) >= 3:
            third_idx = sorted_by_x[2]
            kept_indices = [third_idx]
            desc = f"Selected third mask from left (index {third_idx + 1})."
        elif len(sorted_by_x) >= 2:
            second_idx = sorted_by_x[1]
            kept_indices = [second_idx]
            desc = f"Only two masks available, selected second from left (index {second_idx + 1})."
        else:
            kept_indices = [sorted_by_x[0]]
            desc = f"Only one mask available, selected it (index {sorted_by_x[0] + 1})."
            
    elif spatial_criteria == "third_from_right":
        # Third mask from right
        sorted_by_x = np.argsort(centroids[:, 0])[::-1]  # Descending
        if len(sorted_by_x) >= 3:
            third_idx = sorted_by_x[2]
            kept_indices = [third_idx]
            desc = f"Selected third mask from right (index {third_idx + 1})."
        elif len(sorted_by_x) >= 2:
            second_idx = sorted_by_x[1]
            kept_indices = [second_idx]
            desc = f"Only two masks available, selected second from right (index {second_idx + 1})."
        else:
            kept_indices = [sorted_by_x[0]]
            desc = f"Only one mask available, selected it (index {sorted_by_x[0] + 1})."
            
    elif spatial_criteria == "left_of_all":
        # All masks that are to the left of all other masks (x < min of all others)
        min_x = np.min(centroids[:, 0])
        left_indices = np.where(centroids[:, 0] == min_x)[0].tolist()
        kept_indices = left_indices
        desc = f"Selected {len(left_indices)} mask(s) that are leftmost (indices {[i+1 for i in left_indices]})."
        
    elif spatial_criteria == "right_of_all":
        # All masks that are to the right of all other masks (x > max of all others)
        max_x = np.max(centroids[:, 0])
        right_indices = np.where(centroids[:, 0] == max_x)[0].tolist()
        kept_indices = right_indices
        desc = f"Selected {len(right_indices)} mask(s) that are rightmost (indices {[i+1 for i in right_indices]})."
        
    elif spatial_criteria == "above_all":
        # All masks that are above all other masks (y < min of all others)
        min_y = np.min(centroids[:, 1])
        above_indices = np.where(centroids[:, 1] == min_y)[0].tolist()
        kept_indices = above_indices
        desc = f"Selected {len(above_indices)} mask(s) that are topmost (indices {[i+1 for i in above_indices]})."
        
    elif spatial_criteria == "below_all":
        # All masks that are below all other masks (y > max of all others)
        max_y = np.max(centroids[:, 1])
        below_indices = np.where(centroids[:, 1] == max_y)[0].tolist()
        kept_indices = below_indices
        desc = f"Selected {len(below_indices)} mask(s) that are bottommost (indices {[i+1 for i in below_indices]})."
        
    else:
        raise ValueError(f"Unknown spatial criteria: {spatial_criteria}")
    
    return kept_indices, desc


def filter_masks_by_spatial_position(
    outputs: Dict,
    spatial_criteria: str,
) -> Dict:
    """
    Filter masks in outputs dict based on spatial position.
    
    Args:
        outputs: Dict with pred_boxes, pred_masks, pred_scores, orig_img_h, orig_img_w
        spatial_criteria: Spatial filtering criteria string
    
    Returns:
        Updated outputs dict with filtered masks
    """
    pred_boxes = outputs.get("pred_boxes", [])
    pred_masks = outputs.get("pred_masks", [])
    pred_scores = outputs.get("pred_scores", [])
    orig_img_h = int(outputs["orig_img_h"])
    orig_img_w = int(outputs["orig_img_w"])
    
    kept_indices, desc = apply_spatial_filter(
        pred_boxes, pred_masks, pred_scores, spatial_criteria, orig_img_h, orig_img_w
    )
    
    # Build filtered outputs
    filtered_outputs = dict(outputs)
    filtered_outputs["pred_boxes"] = [pred_boxes[i] for i in kept_indices]
    filtered_outputs["pred_masks"] = [pred_masks[i] for i in kept_indices]
    filtered_outputs["pred_scores"] = [pred_scores[i] for i in kept_indices]
    filtered_outputs["spatial_filter_description"] = desc
    
    return filtered_outputs

