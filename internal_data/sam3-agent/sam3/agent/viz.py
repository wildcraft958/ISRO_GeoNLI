# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import logging

import cv2
import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

from .helpers.visualizer import Visualizer
from .helpers.zoom_in import render_zoom_in

logger = logging.getLogger(__name__)


def visualize(
    input_json: dict,
    zoom_in_index: int | None = None,
    mask_alpha: float = 0.15,
    label_mode: str = "1",
    font_size_multiplier: float = 1.2,
    boarder_width_multiplier: float = 0,
):
    """
    Unified visualization function.

    If zoom_in_index is None:
        - Render all masks in input_json (equivalent to visualize_masks_from_result_json).
        - Returns: PIL.Image

    If zoom_in_index is provided:
        - Returns two PIL.Images:
            1) Output identical to zoom_in_and_visualize(input_json, index).
            2) The same instance rendered via the general overlay using the color
               returned by (1), equivalent to calling visualize_masks_from_result_json
               on a single-mask json_i with color=color_hex.
    """
    # Common fields
    orig_h = int(input_json["orig_img_h"])
    orig_w = int(input_json["orig_img_w"])
    img_path = input_json.get("original_image_path")
    if img_path is None:
        raise KeyError(
            "original_image_path is required in input_json but was not found. "
            "Please ensure the JSON output includes 'original_image_path' when saving SAM3 results."
        )

    # ---------- Mode A: Full-scene render ----------
    if zoom_in_index is None:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        target_h, target_w = img_rgb.shape[:2]

        boxes = np.array(input_json["pred_boxes"])
        # Handle both old format (string) and new format (dict with counts/size)
        rle_masks = []
        binary_masks = []
        resized_mask = False

        for mask_entry in input_json["pred_masks"]:
            if isinstance(mask_entry, dict) and "counts" in mask_entry and "size" in mask_entry:
                # New format: full RLE dict already has counts and size
                rle = {"size": tuple(mask_entry["size"]), "counts": mask_entry["counts"]}
                rle_masks.append(rle)
            else:
                # Old format: string counts, reconstruct size from orig_img_h/w
                rle_masks.append({"size": (orig_h, orig_w), "counts": mask_entry})
        
        # Decode all RLE masks to binary
        for rle in rle_masks:
            mask = mask_utils.decode(rle)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            binary_masks.append(mask)
        
        # Resize masks to match original image dimensions if needed (pyramidal processing fix)
        resized_binary_masks = []
        for mask in binary_masks:
            if mask.shape[0] != orig_h or mask.shape[1] != orig_w:
                # Resize mask to match original image dimensions
                resized_mask = True
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (orig_w, orig_h),  # cv2.resize uses (width, height)
                    interpolation=cv2.INTER_NEAREST
                )
                resized_binary_masks.append((mask > 0.5).astype(np.uint8))
            else:
                resized_binary_masks.append(mask)
        binary_masks = resized_binary_masks

        if resized_mask:
            logger.warning(
                "Resized one or more masks to match image dimensions (%d, %d) for visualization",
                target_h,
                target_w,
            )

        viz = Visualizer(
            img_rgb,
            font_size_multiplier=font_size_multiplier,
            boarder_width_multiplier=boarder_width_multiplier,
        )
        viz.overlay_instances(
            boxes=boxes,
            masks=rle_masks,
            binary_masks=binary_masks,
            assigned_colors=None,
            alpha=mask_alpha,
            label_mode=label_mode,
        )
        pil_all_masks = Image.fromarray(viz.output.get_image())
        return pil_all_masks

    # ---------- Mode B: Zoom-in pair ----------
    else:
        idx = int(zoom_in_index)
        num_masks = len(input_json.get("pred_masks", []))
        if idx < 0 or idx >= num_masks:
            raise ValueError(f"zoom_in_index {idx} is out of range (0..{num_masks-1}).")

        # (1) Replicate zoom_in_and_visualize
        # Handle both old format (string) and new format (dict with counts/size)
        mask_data = input_json["pred_masks"][idx]
        if isinstance(mask_data, dict) and "counts" in mask_data and "size" in mask_data:
            # New format: full RLE dict
            segmentation = {
                "counts": mask_data["counts"],
                "size": mask_data["size"],
            }
        else:
            # Old format: string counts, reconstruct size
            segmentation = {
                "counts": mask_data,
                "size": [orig_h, orig_w],
            }
        object_data = {
            "labels": [{"noun_phrase": f"mask_{idx}"}],
            "segmentation": segmentation,
        }
        pil_img = Image.open(img_path)
        pil_mask_i_zoomed, color_hex = render_zoom_in(
            object_data, pil_img, mask_alpha=mask_alpha
        )

        # (2) Single-instance render with the same color
        boxes_i = np.array([input_json["pred_boxes"][idx]])
        # Handle both old format (string) and new format (dict with counts/size)
        mask_data = input_json["pred_masks"][idx]
        if isinstance(mask_data, dict) and "counts" in mask_data and "size" in mask_data:
            # New format: full RLE dict
            rle_i = {"size": tuple(mask_data["size"]), "counts": mask_data["counts"]}
        else:
            # Old format: string counts, reconstruct size
            rle_i = {"size": (orig_h, orig_w), "counts": mask_data}
        bin_i = mask_utils.decode(rle_i)
        
        # Resize mask if needed (pyramidal processing fix)
        if bin_i.shape[0] != orig_h or bin_i.shape[1] != orig_w:
            bin_i = cv2.resize(
                bin_i.astype(np.float32),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST
            )
            bin_i = (bin_i > 0.5).astype(np.uint8)

        img_bgr_i = cv2.imread(img_path)
        if img_bgr_i is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img_rgb_i = cv2.cvtColor(img_bgr_i, cv2.COLOR_BGR2RGB)
        target_h_i, target_w_i = img_rgb_i.shape[:2]

        bin_i = mask_utils.decode(rle_i)
        if bin_i.ndim == 3:
            bin_i = bin_i[:, :, 0]
        if bin_i.shape[:2] != (target_h_i, target_w_i):
            bin_i = cv2.resize(
                bin_i, (target_w_i, target_h_i), interpolation=cv2.INTER_NEAREST
            )
            bin_i = (bin_i > 0).astype("uint8")
            encoded = mask_utils.encode(np.asfortranarray(bin_i))
            counts = encoded["counts"]
            if isinstance(counts, bytes):
                counts = counts.decode("utf-8")
            rle_i = {"size": list(encoded["size"]), "counts": counts}

        viz_i = Visualizer(
            img_rgb_i,
            font_size_multiplier=font_size_multiplier,
            boarder_width_multiplier=boarder_width_multiplier,
        )
        viz_i.overlay_instances(
            boxes=boxes_i,
            masks=[rle_i],
            binary_masks=[bin_i],
            assigned_colors=[color_hex],
            alpha=mask_alpha,
            label_mode=label_mode,
        )
        pil_mask_i = Image.fromarray(viz_i.output.get_image())

        return pil_mask_i, pil_mask_i_zoomed
