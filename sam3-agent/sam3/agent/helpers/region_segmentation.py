# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Region-based segmentation utilities for segmenting within a specific bounding box region.
"""

from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

try:
    from pycocotools import mask as mask_utils
except Exception:
    mask_utils = None


def crop_image_to_region(
    image: Image.Image,
    bbox: List[float],
    use_normalized: bool = True,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Crop image to the specified bounding box region.
    
    Args:
        image: PIL Image
        bbox: Bounding box [x_min, y_min, x_max, y_max]
        use_normalized: Whether bbox is in normalized coordinates (0-1)
    
    Returns:
        Tuple of (cropped_image, pixel_bbox_tuple)
    """
    img_w, img_h = image.size
    
    if use_normalized:
        # Convert normalized [0, 1] to pixel coordinates
        x_min = int(bbox[0] * img_w)
        y_min = int(bbox[1] * img_h)
        x_max = int(bbox[2] * img_w)
        y_max = int(bbox[3] * img_h)
    else:
        x_min = int(bbox[0])
        y_min = int(bbox[1])
        x_max = int(bbox[2])
        y_max = int(bbox[3])
    
    # Clamp to image boundaries
    x_min = max(0, min(x_min, img_w - 1))
    y_min = max(0, min(y_min, img_h - 1))
    x_max = max(x_min + 1, min(x_max, img_w))
    y_max = max(y_min + 1, min(y_max, img_h))
    
    # Crop image
    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    pixel_bbox = (x_min, y_min, x_max, y_max)
    
    return cropped_image, pixel_bbox


def transform_masks_to_global(
    pred_boxes: List[List[float]],
    pred_masks: List,
    region_bbox: Tuple[int, int, int, int],
    orig_img_h: int,
    orig_img_w: int,
) -> Tuple[List[List[float]], List]:
    """
    Transform masks from region-local coordinates to global image coordinates.
    
    Args:
        pred_boxes: List of boxes in normalized xyxy format [x1, y1, x2, y2] (region-local)
        pred_masks: List of RLE masks (region-local)
        region_bbox: Pixel coordinates (x_min, y_min, x_max, y_max) of the region
        orig_img_h: Original full image height
        orig_img_w: Original full image width
    
    Returns:
        Tuple of (transformed_boxes, transformed_masks) in global coordinates
    """
    x_min, y_min, x_max, y_max = region_bbox
    region_w = x_max - x_min
    region_h = y_max - y_min
    
    # Transform boxes: region-local normalized xyxy -> global normalized xyxy
    transformed_boxes = []
    for box in pred_boxes:
        # box is [x1, y1, x2, y2] in normalized region coordinates
        # Convert to pixel coordinates in region
        x1_local = box[0] * region_w
        y1_local = box[1] * region_h
        x2_local = box[2] * region_w
        y2_local = box[3] * region_h
        
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
        
        transformed_boxes.append([x1_global_norm, y1_global_norm, x2_global_norm, y2_global_norm])
    
    # Transform masks: region-local RLE -> global RLE
    transformed_masks = []
    for mask in pred_masks:
        # Decode mask to get binary array
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
                "size": [region_h, region_w],
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
        transformed_masks.append({
            "counts": rle_global["counts"],
            "size": list(rle_global["size"]),
        })
    
    return transformed_boxes, transformed_masks


def segment_in_region(
    image: Image.Image,
    bbox: List[float],
    use_normalized: bool = True,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Prepare image for region-based segmentation.
    
    Args:
        image: Full image
        bbox: Bounding box [x_min, y_min, x_max, y_max]
        use_normalized: Whether bbox is normalized
    
    Returns:
        Tuple of (cropped_image, region_bbox_pixels)
    """
    cropped_image, region_bbox = crop_image_to_region(image, bbox, use_normalized)
    return cropped_image, region_bbox

