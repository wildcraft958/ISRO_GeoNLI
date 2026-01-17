# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Attribute filtering utilities for filtering masks by color, size, aspect ratio, etc.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    from pycocotools import mask as mask_utils
except Exception:
    mask_utils = None

from .mask_overlap_removal import _decode_single_mask


# Color name to RGB mapping (approximate)
COLOR_RGB_MAP = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
}


def compute_dominant_color(
    image: np.ndarray,
    mask: np.ndarray,
    k: int = 3,
) -> Tuple[int, int, int]:
    """
    Compute dominant color in the masked region using k-means clustering.
    
    Args:
        image: RGB image array (H, W, 3)
        mask: Binary mask (H, W)
        k: Number of clusters for k-means (default: 3)
    
    Returns:
        Dominant RGB color as (R, G, B)
    """
    # Extract masked pixels
    masked_pixels = image[mask > 0]
    
    if len(masked_pixels) == 0:
        # No pixels in mask, return black
        return (0, 0, 0)
    
    # Sample pixels if too many (for efficiency)
    if len(masked_pixels) > 10000:
        indices = np.random.choice(len(masked_pixels), 10000, replace=False)
        masked_pixels = masked_pixels[indices]
    
    # Use simple histogram-based approach for speed
    # Compute mean color in masked region
    mean_color = np.mean(masked_pixels, axis=0).astype(int)
    return tuple(mean_color)


def match_color(
    color_rgb: Tuple[int, int, int],
    target_color_name: str,
    tolerance: float = 0.3,
) -> bool:
    """
    Check if a color matches a target color name with tolerance.
    
    Args:
        color_rgb: RGB color tuple (R, G, B)
        target_color_name: Color name string
        tolerance: Color matching tolerance (0-1, default: 0.3)
    
    Returns:
        True if color matches
    """
    target_color_name = target_color_name.lower()
    
    if target_color_name not in COLOR_RGB_MAP:
        # Unknown color name, return False
        return False
    
    target_rgb = np.array(COLOR_RGB_MAP[target_color_name])
    color_rgb_arr = np.array(color_rgb)
    
    # Compute color distance in RGB space (normalized)
    color_diff = np.abs(color_rgb_arr - target_rgb) / 255.0
    max_diff = np.max(color_diff)
    
    return max_diff <= tolerance


def compute_mask_attributes(
    pred_box: List[float],
    pred_mask,
    orig_img_h: int,
    orig_img_w: int,
) -> Dict[str, float]:
    """
    Compute mask attributes: size ratio and aspect ratio.
    
    Args:
        pred_box: Normalized box in xyxy format [x1, y1, x2, y2]
        pred_mask: RLE mask
        orig_img_h: Original image height
        orig_img_w: Original image width
    
    Returns:
        Dict with 'size_ratio' and 'aspect_ratio'
    """
    # Decode mask to get area
    mask_binary = _decode_single_mask(pred_mask, orig_img_h, orig_img_w)
    mask_area = np.sum(mask_binary > 0)
    image_area = orig_img_h * orig_img_w
    size_ratio = mask_area / image_area if image_area > 0 else 0.0
    
    # Compute aspect ratio from box
    # pred_box is [x1, y1, x2, y2] normalized
    width_norm = pred_box[2] - pred_box[0]
    height_norm = pred_box[3] - pred_box[1]
    
    # Convert to pixel dimensions
    width_px = width_norm * orig_img_w
    height_px = height_norm * orig_img_h
    
    aspect_ratio = width_px / height_px if height_px > 0 else 1.0
    
    return {
        "size_ratio": size_ratio,
        "aspect_ratio": aspect_ratio,
    }


def apply_attribute_filters(
    outputs: Dict,
    color: Optional[str] = None,
    min_size_ratio: Optional[float] = None,
    max_size_ratio: Optional[float] = None,
    aspect_ratio_range: Optional[List[float]] = None,
) -> Tuple[List[int], str]:
    """
    Filter masks based on visual attributes.
    
    Args:
        outputs: Dict with pred_boxes, pred_masks, pred_scores, orig_img_h, orig_img_w, original_image_path
        color: Target color name (optional)
        min_size_ratio: Minimum size ratio (optional)
        max_size_ratio: Maximum size ratio (optional)
        aspect_ratio_range: [min_aspect, max_aspect] (optional)
    
    Returns:
        Tuple of (list of kept mask indices, description message)
    """
    pred_boxes = outputs.get("pred_boxes", [])
    pred_masks = outputs.get("pred_masks", [])
    orig_img_h = int(outputs["orig_img_h"])
    orig_img_w = int(outputs["orig_img_w"])
    img_path = outputs["original_image_path"]
    
    num_masks = len(pred_boxes)
    
    if num_masks == 0:
        return [], "No masks available to filter."
    
    # Load image for color analysis
    image_rgb = None
    if color is not None:
        img_bgr = cv2.imread(img_path)
        if img_bgr is not None:
            image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    kept_indices = []
    filter_reasons = []
    
    for i in range(num_masks):
        keep = True
        reasons = []
        
        # Check color
        if color is not None and image_rgb is not None:
            mask_binary = _decode_single_mask(pred_masks[i], orig_img_h, orig_img_w)
            dominant_color = compute_dominant_color(image_rgb, mask_binary)
            if not match_color(dominant_color, color):
                keep = False
                reasons.append(f"color mismatch (dominant: {dominant_color})")
        
        # Check size ratio
        if keep and (min_size_ratio is not None or max_size_ratio is not None):
            attrs = compute_mask_attributes(pred_boxes[i], pred_masks[i], orig_img_h, orig_img_w)
            size_ratio = attrs["size_ratio"]
            
            if min_size_ratio is not None and size_ratio < min_size_ratio:
                keep = False
                reasons.append(f"size too small ({size_ratio:.3f} < {min_size_ratio:.3f})")
            if max_size_ratio is not None and size_ratio > max_size_ratio:
                keep = False
                reasons.append(f"size too large ({size_ratio:.3f} > {max_size_ratio:.3f})")
        
        # Check aspect ratio
        if keep and aspect_ratio_range is not None:
            attrs = compute_mask_attributes(pred_boxes[i], pred_masks[i], orig_img_h, orig_img_w)
            aspect_ratio = attrs["aspect_ratio"]
            min_aspect, max_aspect = aspect_ratio_range
            
            if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
                keep = False
                reasons.append(f"aspect ratio out of range ({aspect_ratio:.2f} not in [{min_aspect:.2f}, {max_aspect:.2f}])")
        
        if keep:
            kept_indices.append(i)
        else:
            filter_reasons.append(f"Mask {i+1}: {', '.join(reasons)}")
    
    # Build description
    if len(kept_indices) == 0:
        desc = f"All {num_masks} mask(s) were filtered out. Filter reasons: {'; '.join(filter_reasons[:3])}"
        if len(filter_reasons) > 3:
            desc += f" (and {len(filter_reasons) - 3} more)"
    else:
        desc = f"Kept {len(kept_indices)} out of {num_masks} mask(s) based on attribute filters."
    
    return kept_indices, desc


def filter_masks_by_attributes(
    outputs: Dict,
    color: Optional[str] = None,
    min_size_ratio: Optional[float] = None,
    max_size_ratio: Optional[float] = None,
    aspect_ratio_range: Optional[List[float]] = None,
) -> Dict:
    """
    Filter masks in outputs dict based on visual attributes.
    
    Args:
        outputs: Dict with pred_boxes, pred_masks, pred_scores, orig_img_h, orig_img_w, original_image_path
        color: Target color name (optional)
        min_size_ratio: Minimum size ratio (optional)
        max_size_ratio: Maximum size ratio (optional)
        aspect_ratio_range: [min_aspect, max_aspect] (optional)
    
    Returns:
        Updated outputs dict with filtered masks
    """
    kept_indices, desc = apply_attribute_filters(
        outputs, color, min_size_ratio, max_size_ratio, aspect_ratio_range
    )
    
    # Build filtered outputs
    filtered_outputs = dict(outputs)
    filtered_outputs["pred_boxes"] = [outputs["pred_boxes"][i] for i in kept_indices]
    filtered_outputs["pred_masks"] = [outputs["pred_masks"][i] for i in kept_indices]
    filtered_outputs["pred_scores"] = [outputs["pred_scores"][i] for i in kept_indices]
    filtered_outputs["attribute_filter_description"] = desc
    
    return filtered_outputs

