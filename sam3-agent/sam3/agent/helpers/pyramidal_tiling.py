# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Pyramidal tiling utilities for segmenting large images or detecting small objects.
"""

import math
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from pycocotools import mask as mask_utils
except Exception:
    mask_utils = None

from .mask_overlap_removal import _decode_single_mask, mask_iom


def create_pyramidal_tiles(
    img_w: int,
    img_h: int,
    tile_size: int = 1024,
    overlap_ratio: float = 0.2,
    min_tile_size: int = 512,
) -> List[Tuple[int, int, int, int]]:
    """
    Create overlapping tiles for pyramidal tiling.
    
    Args:
        img_w: Image width
        img_h: Image height
        tile_size: Target tile size (default: 1024)
        overlap_ratio: Overlap ratio between tiles (0.0-0.5, default: 0.2)
        min_tile_size: Minimum tile size (default: 512)
    
    Returns:
        List of tile bounding boxes as (x_min, y_min, x_max, y_max) in pixel coordinates
    """
    if img_w <= tile_size and img_h <= tile_size:
        # Image is smaller than tile size, return single tile
        return [(0, 0, img_w, img_h)]
    
    overlap_pixels = int(tile_size * overlap_ratio)
    stride = tile_size - overlap_pixels
    
    tiles = []
    
    # Generate tiles
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x_max = min(x + tile_size, img_w)
            y_max = min(y + tile_size, img_h)
            
            # Ensure minimum tile size
            if (x_max - x) >= min_tile_size and (y_max - y) >= min_tile_size:
                tiles.append((x, y, x_max, y_max))
            
            x += stride
            if x >= img_w:
                break
        
        y += stride
        if y >= img_h:
            break
    
    # If no tiles created (very small image), return full image as single tile
    if not tiles:
        tiles = [(0, 0, img_w, img_h)]
    
    return tiles


def transform_mask_coordinates(
    box: List[float],
    mask,
    tile_bbox: Tuple[int, int, int, int],
    orig_img_h: int,
    orig_img_w: int,
) -> Tuple[List[float], Dict]:
    """
    Transform mask coordinates from tile-local to global image coordinates.
    
    Args:
        box: Normalized box in xyxy format [x1, y1, x2, y2] (tile-local)
        mask: RLE mask (tile-local)
        tile_bbox: Tile bounding box (x_min, y_min, x_max, y_max) in pixels
        orig_img_h: Original full image height
        orig_img_w: Original full image width
    
    Returns:
        Tuple of (transformed_box, transformed_mask) in global coordinates
    """
    x_min, y_min, x_max, y_max = tile_bbox
    tile_w = x_max - x_min
    tile_h = y_max - y_min
    
    # Transform box: tile-local normalized xyxy -> global normalized xyxy
    # box is [x1, y1, x2, y2] in normalized tile coordinates
    x1_local = box[0] * tile_w
    y1_local = box[1] * tile_h
    x2_local = box[2] * tile_w
    y2_local = box[3] * tile_h
    
    # Convert to global pixel coordinates
    x1_global = x1_local + x_min
    y1_global = y1_local + y_min
    x2_global = x2_local + x_min
    y2_global = y2_local + y_min
    
    # Convert back to normalized global coordinates
    x1_global_norm = x1_global / orig_img_w
    y1_global_norm = y1_global / orig_img_h
    x2_global_norm = x2_global / orig_img_w
    y2_global_norm = y2_global / orig_img_h
    
    transformed_box = [x1_global_norm, y1_global_norm, x2_global_norm, y2_global_norm]
    
    # Transform mask: tile-local RLE -> global RLE
    if mask_utils is None:
        raise ImportError("pycocotools is required for mask transformation")
    
    # Handle both old and new RLE formats
    if isinstance(mask, dict) and "counts" in mask and "size" in mask:
        rle_local = {
            "counts": mask["counts"],
            "size": mask["size"],
        }
    else:
        # Old format: string counts
        rle_local = {
            "counts": mask if isinstance(mask, (str, bytes)) else str(mask),
            "size": [tile_h, tile_w],
        }
    
    # Decode to binary mask
    mask_binary = mask_utils.decode(rle_local)
    if mask_binary.ndim == 3:
        mask_binary = mask_binary[:, :, 0]
    
    # Create full-size binary mask
    mask_global_binary = np.zeros((orig_img_h, orig_img_w), dtype=np.uint8)
    mask_global_binary[y_min:y_max, x_min:x_max] = mask_binary
    
    # Re-encode to RLE
    rle_global = mask_utils.encode(np.asfortranarray(mask_global_binary))
    if isinstance(rle_global["counts"], bytes):
        rle_global["counts"] = rle_global["counts"].decode("utf-8")
    
    # Convert to dict format
    transformed_mask = {
        "counts": rle_global["counts"],
        "size": list(rle_global["size"]),
    }
    
    return transformed_box, transformed_mask


def compute_iou_boxes(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU between two normalized boxes in xyxy format.
    
    Args:
        box1: [x1, y1, x2, y2] normalized
        box2: [x1, y1, x2, y2] normalized
    
    Returns:
        IoU value
    """
    # Boxes are already in xyxy format
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Compute union (area = (x2 - x1) * (y2 - y1))
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def merge_tile_results(
    all_tile_results: List[Dict],
    orig_img_h: int,
    orig_img_w: int,
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Merge results from multiple tiles, removing duplicates using IoU.
    
    Args:
        all_tile_results: List of tile result dicts, each with 'pred_boxes', 'pred_masks', 'pred_scores', 'tile_bbox'
        orig_img_h: Original image height
        orig_img_w: Original image width
        iou_threshold: IoU threshold for deduplication (default: 0.5)
    
    Returns:
        Merged outputs dict with deduplicated masks
    """
    if not all_tile_results:
        return {
            "orig_img_h": orig_img_h,
            "orig_img_w": orig_img_w,
            "pred_boxes": [],
            "pred_masks": [],
            "pred_scores": [],
        }
    
    # Collect all masks from all tiles (already in global coordinates)
    all_boxes = []
    all_masks = []
    all_scores = []
    
    for tile_result in all_tile_results:
        all_boxes.extend(tile_result["pred_boxes"])
        all_masks.extend(tile_result["pred_masks"])
        all_scores.extend(tile_result["pred_scores"])
    
    if len(all_boxes) == 0:
        return {
            "orig_img_h": orig_img_h,
            "orig_img_w": orig_img_w,
            "pred_boxes": [],
            "pred_masks": [],
            "pred_scores": [],
        }
    
    # Deduplicate using IoU on boxes
    # Sort by score (descending)
    sorted_indices = sorted(
        range(len(all_scores)),
        key=lambda i: all_scores[i],
        reverse=True
    )
    
    kept_indices = []
    kept_boxes = []
    
    for idx in sorted_indices:
        box = all_boxes[idx]
        
        # Check IoU with already kept boxes
        is_duplicate = False
        for kept_box in kept_boxes:
            iou = compute_iou_boxes(box, kept_box)
            if iou > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            kept_indices.append(idx)
            kept_boxes.append(box)
    
    # Build final outputs
    merged_outputs = {
        "orig_img_h": orig_img_h,
        "orig_img_w": orig_img_w,
        "pred_boxes": [all_boxes[i] for i in kept_indices],
        "pred_masks": [all_masks[i] for i in kept_indices],
        "pred_scores": [all_scores[i] for i in kept_indices],
    }
    
    return merged_outputs

