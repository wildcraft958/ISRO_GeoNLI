# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from typing import Dict, List

import numpy as np
import torch

try:
    from pycocotools import mask as mask_utils
except Exception:
    mask_utils = None


def bbox_iou(box1, box2):
    """
    Calculate IoU between two boxes in [x1, y1, x2, y2] format (normalized or absolute).
    
    Args:
        box1: [x1, y1, x2, y2] bounding box
        box2: [x1, y1, x2, y2] bounding box
    
    Returns:
        IoU value between 0 and 1
    """
    # Validate boxes have 4 elements
    if len(box1) != 4 or len(box2) != 4:
        return 0.0
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def bbox_nms(boxes, scores, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression on bounding boxes.
    Returns indices of boxes to keep.
    """
    if len(boxes) == 0:
        return []
    
    # Sort by score descending
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    keep = []
    while order:
        i = order[0]
        keep.append(i)
        order = order[1:]
        
        # Filter out boxes with high IoU
        remaining = []
        for j in order:
            if bbox_iou(boxes[i], boxes[j]) < iou_threshold:
                remaining.append(j)
        order = remaining
    
    return keep


def mask_intersection(
    masks1: torch.Tensor, masks2: torch.Tensor, block_size: int = 16
) -> torch.Tensor:
    assert masks1.shape[1:] == masks2.shape[1:]
    assert masks1.dtype == torch.bool and masks2.dtype == torch.bool
    N, M = masks1.shape[0], masks2.shape[0]
    out = torch.zeros(N, M, device=masks1.device, dtype=torch.long)
    for i in range(0, N, block_size):
        for j in range(0, M, block_size):
            a = masks1[i : i + block_size]
            b = masks2[j : j + block_size]
            inter = (a[:, None] & b[None, :]).flatten(-2).sum(-1)
            out[i : i + block_size, j : j + block_size] = inter
    return out


def mask_iom(masks1: torch.Tensor, masks2: torch.Tensor) -> torch.Tensor:
    assert masks1.shape[1:] == masks2.shape[1:]
    assert masks1.dtype == torch.bool and masks2.dtype == torch.bool
    inter = mask_intersection(masks1, masks2)
    area1 = masks1.flatten(-2).sum(-1)  # (N,)
    area2 = masks2.flatten(-2).sum(-1)  # (M,)
    min_area = torch.min(area1[:, None], area2[None, :]).clamp_min(1)
    return inter.float() / (min_area.float() + 1e-8)


def _decode_single_mask(mask_repr, h: int, w: int) -> np.ndarray:
    if isinstance(mask_repr, (list, tuple, np.ndarray)):
        arr = np.array(mask_repr)
        if arr.ndim != 2:
            raise ValueError("Mask array must be 2D (H, W).")
        return (arr > 0).astype(np.uint8)

    if mask_utils is None:
        raise ImportError(
            "pycocotools is required to decode RLE mask strings. pip install pycocotools"
        )

    # Handle new format: dict with counts and size
    if isinstance(mask_repr, dict) and "counts" in mask_repr and "size" in mask_repr:
        rle = {
            "counts": mask_repr["counts"],
            "size": mask_repr["size"],
        }
    # Handle old format: string counts (use provided h, w)
    elif isinstance(mask_repr, (str, bytes)):
        rle = {
            "counts": mask_repr if isinstance(mask_repr, (str, bytes)) else str(mask_repr),
            "size": [h, w],
        }
    else:
        raise ValueError(f"Unsupported mask representation type: {type(mask_repr)}")
    
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return (decoded > 0).astype(np.uint8)


def _decode_masks_to_torch_bool(pred_masks: List, h: int, w: int) -> torch.Tensor:
    bin_masks = [_decode_single_mask(m, h, w) for m in pred_masks]
    masks_np = np.stack(bin_masks, axis=0).astype(np.uint8)  # (N, H, W)
    return torch.from_numpy(masks_np > 0)


def remove_overlapping_masks(sample: Dict, bbox_iou_thresh: float = 0.5) -> Dict:
    """
    Bounding Box NMS deduplication for batch segmentation results.
    Removes boxes with IoU > bbox_iou_thresh (keeps higher confidence detections).
    
    If pred_masks has length 0 or 1, returns sample unchanged (no extra keys).
    """
    # Basic presence checks
    if "pred_masks" not in sample or not isinstance(sample["pred_masks"], list):
        return sample  # nothing to do / preserve as-is

    pred_masks = sample["pred_masks"]
    N = len(pred_masks)

    # --- Early exit: 0 or 1 mask -> do NOT modify the JSON at all ---
    if N <= 1:
        return sample

    pred_scores = sample.get("pred_scores", [1.0] * N)  # fallback if scores missing
    pred_boxes = sample.get("pred_boxes", None)

    assert N == len(pred_scores), "pred_masks and pred_scores must have same length"
    if pred_boxes is not None:
        assert N == len(pred_boxes), "pred_masks and pred_boxes must have same length"

    # ============================================================
    # Bounding Box NMS - removes duplicate detections
    # ============================================================
    if pred_boxes is not None and len(pred_boxes) > 1:
        bbox_keep_indices = bbox_nms(pred_boxes, pred_scores, iou_threshold=bbox_iou_thresh)
        
        # Filter to only boxes that passed bbox NMS
        final_masks = [pred_masks[i] for i in bbox_keep_indices]
        final_scores = [pred_scores[i] for i in bbox_keep_indices]
        final_boxes = [pred_boxes[i] for i in bbox_keep_indices]
        final_count = len(final_masks)
        
        print(f"    📦 BBox NMS (IoU>{bbox_iou_thresh}): {N} → {final_count} masks")
        
        # Build filtered output
        out = dict(sample)
        out["pred_masks"] = final_masks
        out["pred_scores"] = final_scores
        out["pred_boxes"] = final_boxes
        out["kept_indices"] = bbox_keep_indices
        out["removed_indices"] = [i for i in range(N) if i not in set(bbox_keep_indices)]
        out["bbox_iou_threshold"] = float(bbox_iou_thresh)
        return out
    
    # No boxes to filter - return as is
    return sample
