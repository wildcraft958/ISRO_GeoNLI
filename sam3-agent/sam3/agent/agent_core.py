# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import copy
import json
import os
import time

import cv2
from PIL import Image

from .client_llm import send_generate_request
from .client_sam3 import call_sam_service
from .viz import visualize
# from .helpers.spatial_filtering import filter_masks_by_spatial_position  # Disabled - spatial filtering done in LLM thinking
from .helpers.region_segmentation import segment_in_region, transform_masks_to_global
from .helpers.attribute_filtering import filter_masks_by_attributes
# from .helpers.relative_spatial import filter_masks_by_relative_position  # Disabled - relative filtering done in LLM thinking
from .helpers.pyramidal_tiling import (
    create_pyramidal_tiles,
    transform_mask_coordinates,
    merge_tile_results,
)
import re
import base64
import numpy as np
from io import BytesIO
import torch


def visualize_mask_subset(input_json: dict, mask_indices: list, output_path: str) -> str:
    """
    Render only a subset of masks (specified by indices) on the image.
    Returns the path to the saved visualization.
    
    Handles errors gracefully - returns original image if visualization fails.
    """
    from PIL import Image
    
    try:
        # Filter valid indices
        num_boxes = len(input_json.get("pred_boxes", []))
        valid_indices = [i for i in mask_indices if 0 <= i < num_boxes]
        
        if not valid_indices:
            # No valid masks - just save the original image
            orig_path = input_json.get("original_image_path", "")
            if orig_path and os.path.exists(orig_path):
                img = Image.open(orig_path)
                img.save(output_path)
            return output_path
        
        # Create a subset JSON with only the specified masks
        subset_json = {
            "orig_img_h": input_json["orig_img_h"],
            "orig_img_w": input_json["orig_img_w"],
            "original_image_path": input_json["original_image_path"],
            "pred_boxes": [input_json["pred_boxes"][i] for i in valid_indices],
            "pred_masks": [input_json["pred_masks"][i] for i in valid_indices],
            "pred_scores": [input_json.get("pred_scores", [1.0] * len(input_json["pred_boxes"]))[i] for i in valid_indices],
        }
        
        # Use the visualize function to render
        viz_img = visualize(subset_json)
        viz_img.save(output_path)
        return output_path
        
    except Exception as e:
        # If visualization fails, try to save the original image as fallback
        print(f"    ⚠️ Visualization failed ({e}), using original image")
        try:
            orig_path = input_json.get("original_image_path", "")
            if orig_path and os.path.exists(orig_path):
                img = Image.open(orig_path)
                img.save(output_path)
        except:
            pass
        return output_path


def _process_single_batch(
    batch_idx: int,
    batch_mask_indices: list,
    combined_output: dict,
    img_path: str,
    initial_query: str,
    sam_output_dir: str,
    llm_generate_fn,
) -> dict:
    """
    Process a single batch of masks for confidence assessment.
    Returns dict mapping mask index (0-based) to confidence.
    """
    start_idx = batch_mask_indices[0]
    end_idx = batch_mask_indices[-1]
    
    # Create visualization for this batch (use process ID for uniqueness)
    import os as os_module
    process_id = os_module.getpid()
    batch_viz_path = os.path.join(sam_output_dir, f"batch_{batch_idx}_p{process_id}_viz.png")
    visualize_mask_subset(combined_output, batch_mask_indices, batch_viz_path)
    
    # Create the confidence assessment prompt
    mask_list = ", ".join([str(i + 1) for i in batch_mask_indices])
    confidence_prompt = f"""TASK: Assess if each mask matches the query "{initial_query}".

Masks shown: {mask_list} (numbered {start_idx + 1} to {end_idx + 1}).

For EACH mask, output ONE line:
Mask N: HIGH/MEDIUM/LOW

- HIGH = clearly matches "{initial_query}"
- MEDIUM = probably matches but uncertain
- LOW = doesn't match or wrong type

Assess each mask now:"""

    # Build messages
    batch_messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": img_path},
                {"type": "image", "image": batch_viz_path},
                {"type": "text", "text": confidence_prompt},
            ]
        }
    ]
    
    batch_confidences = {}
    
    try:
        response = llm_generate_fn(
            messages=batch_messages,
            max_tokens=512,
        )
        
        if response:
            # Parse the response to extract confidences
            for line in response.split("\n"):
                line = line.strip()
                if not line:
                    continue
                match = re.search(r'(?:Mask\s*)?(\d+)\s*[:\-]\s*(HIGH|MEDIUM|LOW)', line, re.IGNORECASE)
                if match:
                    mask_num = int(match.group(1))
                    confidence = match.group(2).upper()
                    # Convert 1-based mask number to 0-based index and validate bounds
                    mask_idx = mask_num - 1
                    if 0 <= mask_idx < len(batch_mask_indices):
                        # Map to actual mask index in combined_output
                        actual_mask_idx = batch_mask_indices[mask_idx]
                        batch_confidences[actual_mask_idx] = confidence
                    else:
                        print(f"    ⚠️ Mask number {mask_num} out of range for batch (1-{len(batch_mask_indices)})")
            
            return {"batch_idx": batch_idx, "confidences": batch_confidences, "success": True}
        else:
            return {"batch_idx": batch_idx, "confidences": {}, "success": False, "error": "No response"}
            
    except Exception as e:
        return {"batch_idx": batch_idx, "confidences": {}, "success": False, "error": str(e)}


def batch_assess_confidence(
    combined_output: dict,
    img_path: str,
    initial_query: str,
    sam_output_dir: str,
    llm_generate_fn,  # The configured send_generate_request function
    batch_size: int = 10,
    max_workers: int = 4,  # Number of parallel API calls
) -> dict:
    """
    Assess confidence for ALL masks using PARALLEL API calls.
    
    - If total masks <= batch_size: single call (no parallelism needed)
    - If total masks > batch_size: splits into batches, sends ALL batches in PARALLEL
    
    Args:
        llm_generate_fn: The configured send_generate_request function (with server_url, api_key)
        batch_size: Number of masks per batch
        max_workers: Number of parallel API calls (default 4)
    
    Returns a dict mapping mask index (0-based) to confidence: {"HIGH", "MEDIUM", "LOW"}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    num_masks = len(combined_output.get("pred_boxes", []))
    
    if num_masks == 0:
        return {}
    
    num_batches = (num_masks + batch_size - 1) // batch_size
    
    # Prepare all batches
    batch_infos = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_masks)
        batch_mask_indices = list(range(start_idx, end_idx))
        batch_infos.append((batch_idx, batch_mask_indices))
    
    # Single batch case - no parallelism needed
    if num_batches == 1:
        print(f"\n📊 Confidence assessment: {num_masks} masks (single batch)")
        result = _process_single_batch(
            batch_idx=0,
            batch_mask_indices=batch_infos[0][1],
            combined_output=combined_output,
            img_path=img_path,
            initial_query=initial_query,
            sam_output_dir=sam_output_dir,
            llm_generate_fn=llm_generate_fn,
        )
        
        confidences = result["confidences"] if result["success"] else {}
        
        if confidences:
            high_count = sum(1 for c in confidences.values() if c == "HIGH")
            med_count = sum(1 for c in confidences.values() if c == "MEDIUM")
            low_count = sum(1 for c in confidences.values() if c == "LOW")
            print(f"  ✅ Assessed {len(confidences)}/{num_masks} masks")
            print(f"     HIGH: {high_count}, MEDIUM: {med_count}, LOW: {low_count}")
        
        return confidences
    
    # Multiple batches - use parallel execution
    print(f"\n📊 Batched confidence assessment: {num_masks} masks in {num_batches} batches (PARALLEL)")
    print(f"  🚀 Launching {num_batches} parallel API calls (max_workers={max_workers})...")
    
    # Execute all batches in parallel
    confidences = {}
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batch jobs
        futures = {
            executor.submit(
                _process_single_batch,
                batch_idx,
                batch_mask_indices,
                combined_output,
                img_path,
                initial_query,
                sam_output_dir,
                llm_generate_fn,
            ): batch_idx
            for batch_idx, batch_mask_indices in batch_infos
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                if result["success"]:
                    print(f"  ✓ Batch {result['batch_idx'] + 1}/{num_batches} completed ({len(result['confidences'])} masks)")
                else:
                    print(f"  ❌ Batch {result['batch_idx'] + 1}/{num_batches} failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"  ❌ Batch {batch_idx + 1}/{num_batches} exception: {e}")
    
    # Combine all results
    successful_results = 0
    for result in results:
        if result["success"]:
            confidences.update(result["confidences"])
            successful_results += 1
    
    # Validate that at least some results succeeded
    if successful_results == 0 and num_batches > 0:
        print(f"\n  ⚠️ WARNING: All {num_batches} batches failed. No confidence assessments available.")
    elif successful_results < num_batches:
        print(f"\n  ⚠️ WARNING: Only {successful_results}/{num_batches} batches succeeded.")
    
    print(f"\n  ✅ All {num_batches} batches completed. Assessed {len(confidences)}/{num_masks} masks")
    
    # Summarize
    high_count = sum(1 for c in confidences.values() if c == "HIGH")
    med_count = sum(1 for c in confidences.values() if c == "MEDIUM")
    low_count = sum(1 for c in confidences.values() if c == "LOW")
    print(f"     HIGH: {high_count}, MEDIUM: {med_count}, LOW: {low_count}")
    
    return confidences


def cluster_boxes_by_proximity(boxes: list, distance_threshold: float = 0.15) -> list:
    """
    Cluster bounding boxes by spatial proximity using simple greedy clustering.
    
    Args:
        boxes: List of [x1, y1, x2, y2] bounding boxes
        distance_threshold: Maximum normalized distance between box centers to be in same cluster
    
    Returns:
        List of clusters, where each cluster is a list of box indices
    """
    if not boxes:
        return []
    
    # Calculate centroids of each box
    centroids = []
    for box in boxes:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        centroids.append((cx, cy))
    
    # Find image dimensions from boxes for normalization
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    img_diagonal = (max_x**2 + max_y**2) ** 0.5
    
    # Greedy clustering
    assigned = [False] * len(boxes)
    clusters = []
    
    for i in range(len(boxes)):
        if assigned[i]:
            continue
        
        # Start new cluster with this box
        cluster = [i]
        assigned[i] = True
        
        # Find all nearby unassigned boxes
        for j in range(i + 1, len(boxes)):
            if assigned[j]:
                continue
            
            # Calculate distance between centroids
            dist = ((centroids[i][0] - centroids[j][0])**2 + 
                    (centroids[i][1] - centroids[j][1])**2) ** 0.5
            normalized_dist = dist / img_diagonal if img_diagonal > 0 else 0
            
            if normalized_dist < distance_threshold:
                cluster.append(j)
                assigned[j] = True
        
        clusters.append(cluster)
    
    return clusters


def create_cluster_crop(
    img_path: str,
    boxes: list,
    cluster_indices: list,
    output_path: str,
    padding_ratio: float = 0.1,
) -> tuple:
    """
    Create a cropped image containing only the boxes in the cluster.
    
    Handles:
    - Normalized coordinates (0-1) vs pixel coordinates
    - Invalid/negative coordinates
    - Inverted coordinates (right < left)
    - Coordinates outside image bounds
    
    Returns:
        (crop_path, crop_box) where crop_box is [x1, y1, x2, y2] in original image coords
    """
    from PIL import Image
    
    # Load original image first to get dimensions
    img = Image.open(img_path)
    img_w, img_h = img.size
    
    # Get bounding box that contains all boxes in cluster
    cluster_boxes = [boxes[i] for i in cluster_indices]
    
    # Collect all coordinates, handling both normalized and pixel coords
    all_x1, all_y1, all_x2, all_y2 = [], [], [], []
    
    for box in cluster_boxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        
        # Check if coordinates are normalized (0-1 range)
        if all(0 <= c <= 1.0 for c in [x1, y1, x2, y2]):
            # Convert normalized to pixel coordinates
            x1 = x1 * img_w
            y1 = y1 * img_h
            x2 = x2 * img_w
            y2 = y2 * img_h
        
        # Handle inverted coordinates (ensure x1 < x2, y1 < y2)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Clamp to valid range
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))
        
        all_x1.append(x1)
        all_y1.append(y1)
        all_x2.append(x2)
        all_y2.append(y2)
    
    # Get bounding box of all cluster boxes
    min_x = min(all_x1)
    min_y = min(all_y1)
    max_x = max(all_x2)
    max_y = max(all_y2)
    
    # Ensure we have a valid region
    if max_x <= min_x:
        max_x = min_x + 10  # Minimum width
    if max_y <= min_y:
        max_y = min_y + 10  # Minimum height
    
    # Add padding
    width = max_x - min_x
    height = max_y - min_y
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    
    # Calculate crop coordinates with padding, clamped to image bounds
    crop_x1 = int(max(0, min_x - pad_x))
    crop_y1 = int(max(0, min_y - pad_y))
    crop_x2 = int(min(img_w, max_x + pad_x))
    crop_y2 = int(min(img_h, max_y + pad_y))
    
    # Final validation - ensure valid crop region
    if crop_x2 <= crop_x1:
        crop_x2 = min(crop_x1 + 10, img_w)
    if crop_y2 <= crop_y1:
        crop_y2 = min(crop_y1 + 10, img_h)
    
    # Crop and save
    try:
        crop = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        crop.save(output_path)
    except Exception as e:
        # If crop fails, save the entire image as fallback
        print(f"    ⚠️ Crop failed ({e}), using full image")
        img.save(output_path)
        crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, img_w, img_h
    
    return output_path, (crop_x1, crop_y1, crop_x2, crop_y2)


def assess_cluster_with_crop(
    combined_output: dict,
    img_path: str,
    initial_query: str,
    cluster_indices: list,
    cluster_id: int,
    sam_output_dir: str,
    llm_generate_fn,  # The configured send_generate_request function
) -> dict:
    """
    Assess confidence for masks in a cluster using a cropped image.
    
    Args:
        llm_generate_fn: The configured send_generate_request function (with server_url, api_key)
    
    Returns:
        dict mapping mask index (0-based) to confidence: {"HIGH", "MEDIUM", "LOW"}
    """
    boxes = combined_output.get("pred_boxes", [])
    
    # Create crop for this cluster (use process ID for uniqueness)
    import os as os_module
    process_id = os_module.getpid()
    crop_path = os.path.join(sam_output_dir, f"cluster_{cluster_id}_p{process_id}_crop.png")
    try:
        crop_path, crop_box = create_cluster_crop(img_path, boxes, cluster_indices, crop_path)
    except Exception as e:
        print(f"    ⚠️ Crop creation failed for cluster {cluster_id}: {e}")
        # Use original image as fallback
        crop_path = img_path
    
    # Create a mini-visualization showing the cluster masks
    cluster_viz_path = os.path.join(sam_output_dir, f"cluster_{cluster_id}_p{process_id}_viz.png")
    try:
        visualize_mask_subset(combined_output, cluster_indices, cluster_viz_path)
    except Exception as e:
        print(f"    ⚠️ Visualization failed for cluster {cluster_id}: {e}")
        cluster_viz_path = img_path  # Use original image as fallback
    
    # Build the assessment prompt - simple task instruction only
    mask_nums = [i + 1 for i in cluster_indices]
    mask_list = ", ".join(str(m) for m in mask_nums)
    
    prompt = f"""TASK: Assess if each mask matches "{initial_query}".

Masks in this cluster: {mask_list}

For EACH mask, output ONE line:
Mask N: HIGH/MEDIUM/LOW

- HIGH = clearly matches "{initial_query}"
- MEDIUM = probably matches but uncertain  
- LOW = doesn't match or wrong type

Be strict - only HIGH if confident. Assess now:"""

    # Build messages - use original image if crop/viz failed
    image_content = []
    if os.path.exists(crop_path):
        image_content.append({"type": "image", "image": crop_path})
    else:
        image_content.append({"type": "image", "image": img_path})
        
    if os.path.exists(cluster_viz_path) and cluster_viz_path != img_path:
        image_content.append({"type": "image", "image": cluster_viz_path})
    
    image_content.append({"type": "text", "text": prompt})
    
    messages = [
        {
            "role": "user",
            "content": image_content,
        }
    ]
    
    confidences = {}
    
    try:
        response = llm_generate_fn(messages=messages, max_tokens=256)
        
        if response:
            for line in response.split("\n"):
                line = line.strip()
                if not line:
                    continue
                match = re.search(r'(?:Mask\s*)?(\d+)\s*[:\-]\s*(HIGH|MEDIUM|LOW)', line, re.IGNORECASE)
                if match:
                    mask_num = int(match.group(1))
                    confidence = match.group(2).upper()
                    # Only store if mask is in this cluster and index is valid
                    if mask_num in mask_nums:
                        mask_idx = mask_num - 1
                        # Find the actual mask index in cluster_indices
                        try:
                            cluster_local_idx = mask_nums.index(mask_num)
                            actual_mask_idx = cluster_indices[cluster_local_idx]
                            confidences[actual_mask_idx] = confidence
                        except (ValueError, IndexError):
                            print(f"    ⚠️ Mask number {mask_num} out of range for cluster")
    except Exception as e:
        print(f"    ❌ Error assessing cluster {cluster_id}: {e}")
    
    return confidences


def _process_single_cluster(
    cluster_id: int,
    cluster_mask_indices: list,
    combined_output: dict,
    img_path: str,
    initial_query: str,
    sam_output_dir: str,
    llm_generate_fn,
    initial_confidences: dict,
) -> dict:
    """
    Process a single cluster for confidence assessment.
    Returns dict with cluster results.
    
    On error, falls back to using initial confidences for masks in this cluster.
    """
    try:
        cluster_confidences = assess_cluster_with_crop(
            combined_output=combined_output,
            img_path=img_path,
            initial_query=initial_query,
            cluster_indices=cluster_mask_indices,
            cluster_id=cluster_id,
            sam_output_dir=sam_output_dir,
            llm_generate_fn=llm_generate_fn,
        )
    except Exception as e:
        # On error, fall back to initial confidences
        print(f"    ⚠️ Cluster {cluster_id} crop/viz error: {e}, using initial confidences")
        cluster_confidences = {}
    
    # For masks not assessed, keep their initial confidence or mark as LOW
    result_confidences = {}
    for mask_idx in cluster_mask_indices:
        if mask_idx in cluster_confidences:
            result_confidences[mask_idx] = cluster_confidences[mask_idx]
        elif mask_idx in initial_confidences:
            result_confidences[mask_idx] = initial_confidences[mask_idx]
        else:
            result_confidences[mask_idx] = "LOW"
    
    return {
        "cluster_id": cluster_id,
        "confidences": result_confidences,
        "mask_indices": cluster_mask_indices,
    }


def smart_confidence_assessment(
    combined_output: dict,
    img_path: str,
    initial_query: str,
    sam_output_dir: str,
    initial_confidences: dict,
    llm_generate_fn,  # The configured send_generate_request function
    uncertain_threshold: int = 10,
    max_workers: int = 4,  # Number of parallel API calls
) -> dict:
    """
    Smart confidence assessment using clustering and crops for MEDIUM/LOW confidence masks.
    Uses PARALLEL API calls to assess all clusters simultaneously.
    
    When there are too many MEDIUM/LOW confidence masks:
    1. Cluster them by spatial proximity
    2. Create crops for each cluster
    3. Run VLM on ALL clusters in PARALLEL
    4. Wait for all results, then combine
    5. Return final confidences
    
    Args:
        combined_output: The SAM output with boxes, masks, scores
        img_path: Path to original image
        initial_query: User's query
        sam_output_dir: Directory to save intermediate files
        initial_confidences: Dict of initial confidence assessments
        llm_generate_fn: The configured send_generate_request function (with server_url, api_key)
        uncertain_threshold: If MEDIUM+LOW count > this, use clustering
        max_workers: Number of parallel API calls (default 4)
    
    Returns:
        Final confidence dict mapping mask index (0-based) to confidence
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    boxes = combined_output.get("pred_boxes", [])
    num_masks = len(boxes)
    
    # Separate masks by confidence
    high_masks = [i for i, c in initial_confidences.items() if c == "HIGH"]
    medium_masks = [i for i, c in initial_confidences.items() if c == "MEDIUM"]
    low_masks = [i for i, c in initial_confidences.items() if c == "LOW"]
    
    # Also handle masks not in initial_confidences (treat as MEDIUM)
    all_assessed = set(initial_confidences.keys())
    unassessed = [i for i in range(num_masks) if i not in all_assessed]
    medium_masks.extend(unassessed)
    
    uncertain_masks = medium_masks + low_masks
    
    print(f"\n🎯 Smart confidence assessment:")
    print(f"   HIGH: {len(high_masks)}, MEDIUM: {len(medium_masks)}, LOW: {len(low_masks)}")
    
    # If not too many uncertain masks, just return initial confidences
    if len(uncertain_masks) <= uncertain_threshold:
        print(f"   ✓ Uncertain masks ({len(uncertain_masks)}) <= threshold ({uncertain_threshold}), using initial assessment")
        return initial_confidences
    
    print(f"   ⚠️ Uncertain masks ({len(uncertain_masks)}) > threshold ({uncertain_threshold})")
    print(f"   🔍 Clustering and PARALLEL crop-based assessment...")
    
    # Get boxes for uncertain masks only
    uncertain_boxes = [boxes[i] for i in uncertain_masks]
    
    # Validate that we have boxes to cluster
    if not uncertain_boxes:
        print(f"   ⚠️ No uncertain boxes to cluster. Returning initial confidences.")
        return initial_confidences
    
    # Cluster the uncertain boxes
    clusters = cluster_boxes_by_proximity(uncertain_boxes, distance_threshold=0.15)
    
    # Map cluster indices back to original mask indices
    cluster_to_original = []
    for cluster in clusters:
        original_indices = [uncertain_masks[i] for i in cluster]
        cluster_to_original.append(original_indices)
    
    num_clusters = len(cluster_to_original)
    print(f"   📦 Found {num_clusters} clusters from {len(uncertain_masks)} uncertain masks")
    print(f"   🚀 Launching {num_clusters} parallel API calls (max_workers={max_workers})...")
    
    # Start with HIGH confidence masks (they're already confirmed)
    final_confidences = {i: "HIGH" for i in high_masks}
    
    # Assess ALL clusters in PARALLEL
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all cluster jobs
        futures = {
            executor.submit(
                _process_single_cluster,
                cluster_id,
                cluster_mask_indices,
                combined_output,
                img_path,
                initial_query,
                sam_output_dir,
                llm_generate_fn,
                initial_confidences,
            ): cluster_id
            for cluster_id, cluster_mask_indices in enumerate(cluster_to_original)
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            cluster_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"   ✓ Cluster {result['cluster_id'] + 1}/{num_clusters} completed ({len(result['confidences'])} masks)")
            except Exception as e:
                print(f"   ❌ Cluster {cluster_id + 1}/{num_clusters} exception: {e}")
    
    # Combine all results
    successful_clusters = 0
    for result in results:
        if result.get("confidences"):
            final_confidences.update(result["confidences"])
            successful_clusters += 1
    
    # Validate that at least some clusters succeeded
    if successful_clusters == 0 and num_clusters > 0:
        print(f"\n   ⚠️ WARNING: All {num_clusters} clusters failed. Using initial confidences as fallback.")
        return initial_confidences
    elif successful_clusters < num_clusters:
        print(f"\n   ⚠️ WARNING: Only {successful_clusters}/{num_clusters} clusters succeeded.")
    
    # Summarize final results
    final_high = sum(1 for c in final_confidences.values() if c == "HIGH")
    final_med = sum(1 for c in final_confidences.values() if c == "MEDIUM")
    final_low = sum(1 for c in final_confidences.values() if c == "LOW")
    
    print(f"\n   ✅ All {num_clusters} clusters completed. Final assessment:")
    print(f"      HIGH: {final_high}, MEDIUM: {final_med}, LOW: {final_low}")
    
    return final_confidences


def parse_tool_calls_from_text(text: str) -> list:
    """
    Parse tool calls from text response when not using OpenAI function calling API.
    
    Supports multiple formats:
    1. JSON blocks: {"name": "function_name", "arguments": {...}}
    2. Function call syntax: function_name(arg1="value1", arg2="value2")
    3. XML-like tags: <function_call>{"name": "...", "arguments": {...}}</function_call>
    
    Returns list of tool call dicts in format:
    [{"id": "...", "function": {"name": "...", "arguments": "..."}}]
    """
    tool_calls = []
    
    # Known function names (matching active tools in TOOLS list)
    known_functions = [
        "segment_phrase_batch", "examine_each_mask", "select_masks_and_return", 
        "report_no_mask", 
        "filter_masks_by_attributes",
    ]
    
    # Pattern 1: JSON object with name/arguments structure
    # Match patterns like: {"name": "segment_phrase", "arguments": {"text_prompt": "car"}}
    json_pattern = r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        func_name = match.group(1)
        args_str = match.group(2)
        if func_name in known_functions:
            tool_calls.append({
                "id": f"call_{len(tool_calls)}",
                "type": "function",  # Required by OpenAI/vLLM format
                "function": {
                    "name": func_name,
                    "arguments": args_str
                }
            })
    
    if tool_calls:
        return tool_calls
    
    # Pattern 2: Function call syntax like segment_phrase(text_prompt="car")
    # or segment_phrase_batch(text_prompts=["a", "b", "c"])
    for func_name in known_functions:
        # Match function_name followed by parentheses with arguments
        # Use a more robust pattern that handles nested brackets
        pattern = rf'{func_name}\s*\(([^)]*(?:\[[^\]]*\][^)]*)*)\)'
        for match in re.finditer(pattern, text):
            args_text = match.group(1).strip()
            
            # Parse arguments into dict
            args_dict = {}
            if args_text:
                # Special handling for text_prompts array (segment_phrase_batch)
                text_prompts_match = re.search(r'text_prompts\s*=\s*(\[[^\]]+\])', args_text)
                if text_prompts_match:
                    try:
                        args_dict["text_prompts"] = json.loads(text_prompts_match.group(1))
                    except json.JSONDecodeError:
                        # Try to extract strings manually
                        array_str = text_prompts_match.group(1)
                        strings = re.findall(r'"([^"]+)"|\'([^\']+)\'', array_str)
                        args_dict["text_prompts"] = [s[0] or s[1] for s in strings]
                
                # Handle keyword arguments like: text_prompt="car", other_arg=123
                kwarg_pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\[[^\]]*\])|(\{[^}]*\})|([^,\s\]]+))'
                for kwmatch in re.finditer(kwarg_pattern, args_text):
                    key = kwmatch.group(1)
                    if key == "text_prompts" and "text_prompts" in args_dict:
                        continue  # Already handled above
                    # Get the first non-None value from the capture groups
                    value = kwmatch.group(2) or kwmatch.group(3) or kwmatch.group(4) or kwmatch.group(5) or kwmatch.group(6)
                    if value:
                        # Try to parse as JSON for arrays/objects/numbers
                        try:
                            args_dict[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            args_dict[key] = value
            
            tool_calls.append({
                "id": f"call_{len(tool_calls)}",
                "type": "function",  # Required by OpenAI/vLLM format
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(args_dict)
                }
            })
    
    if tool_calls:
        return tool_calls
    
    # Pattern 3: Look for function names followed by JSON arguments
    # e.g., segment_phrase {"text_prompt": "car"}
    for func_name in known_functions:
        pattern = rf'{func_name}\s*(\{{[^{{}}]*\}})'
        for match in re.finditer(pattern, text):
            args_str = match.group(1)
            try:
                # Validate it's valid JSON
                json.loads(args_str)
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",  # Required by OpenAI/vLLM format
                    "function": {
                        "name": func_name,
                        "arguments": args_str
                    }
                })
            except json.JSONDecodeError:
                pass
    
    if tool_calls:
        return tool_calls
    
    # Pattern 4: Look for simple mentions of functions with text_prompt(s) in quotes nearby
    # This is a fallback for less structured outputs
    
    # First check for segment_phrase_batch with text_prompts array
    if "segment_phrase_batch" in text:
        # Look for text_prompts array patterns
        batch_patterns = [
            r'"text_prompts"\s*:\s*(\[[^\]]+\])',
            r"'text_prompts'\s*:\s*(\[[^\]]+\])",
            r'text_prompts\s*=\s*(\[[^\]]+\])',
        ]
        for pattern in batch_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    text_prompts = json.loads(match.group(1))
                    tool_calls.append({
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "segment_phrase_batch",
                            "arguments": json.dumps({"text_prompts": text_prompts})
                        }
                    })
                    return tool_calls
                except json.JSONDecodeError:
                    # Try to extract strings manually
                    array_str = match.group(1)
                    strings = re.findall(r'"([^"]+)"|\'([^\']+)\'', array_str)
                    text_prompts = [s[0] or s[1] for s in strings]
                    if text_prompts:
                        tool_calls.append({
                            "id": "call_0",
                            "type": "function",
                            "function": {
                                "name": "segment_phrase_batch",
                                "arguments": json.dumps({"text_prompts": text_prompts})
                            }
                        })
                        return tool_calls
    
    # Convert segment_phrase to segment_phrase_batch (ONE-SHOT system)
    # If the model tries to call segment_phrase, convert it to a batch call with the single prompt
    if "segment_phrase" in text and "segment_phrase_batch" not in text:
        # Look for text_prompt patterns
        prompt_patterns = [
            r'"text_prompt"\s*:\s*"([^"]+)"',
            r"'text_prompt'\s*:\s*'([^']+)'",
            r'text_prompt\s*=\s*"([^"]+)"',
            r"text_prompt\s*=\s*'([^']+)'",
        ]
        for pattern in prompt_patterns:
            match = re.search(pattern, text)
            if match:
                text_prompt = match.group(1)
                # Convert to batch format with single prompt
                print(f"⚠️ Converting segment_phrase('{text_prompt}') → segment_phrase_batch(['{text_prompt}'])")
                tool_calls.append({
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "segment_phrase_batch",
                        "arguments": json.dumps({"text_prompts": [text_prompt]})
                    }
                })
                return tool_calls
    
    # Pattern 5: Check for other tools without arguments
    for func_name in ["examine_each_mask", "report_no_mask"]:
        if func_name in text and func_name not in [tc["function"]["name"] for tc in tool_calls]:
            # These tools don't require arguments
            tool_calls.append({
                "id": f"call_{len(tool_calls)}",
                "type": "function",  # Required by OpenAI/vLLM format
                "function": {
                    "name": func_name,
                    "arguments": "{}"
                }
            })
            return tool_calls
    
    # Pattern 6: select_masks_and_return with final_answer_masks
    if "select_masks_and_return" in text:
        # Look for array of integers
        masks_pattern = r'"final_answer_masks"\s*:\s*\[([^\]]+)\]'
        match = re.search(masks_pattern, text)
        if match:
            masks_str = match.group(1)
            try:
                masks = json.loads(f"[{masks_str}]")
                tool_calls.append({
                    "id": "call_0",
                    "type": "function",  # Required by OpenAI/vLLM format
                    "function": {
                        "name": "select_masks_and_return",
                        "arguments": json.dumps({"final_answer_masks": masks})
                    }
                })
                return tool_calls
            except json.JSONDecodeError:
                pass
        
        # Alternative: look for array directly after function name
        alt_pattern = r'select_masks_and_return[^[]*\[([^\]]+)\]'
        match = re.search(alt_pattern, text)
        if match:
            masks_str = match.group(1)
            try:
                masks = json.loads(f"[{masks_str}]")
                tool_calls.append({
                    "id": "call_0",
                    "type": "function",  # Required by OpenAI/vLLM format
                    "function": {
                        "name": "select_masks_and_return",
                        "arguments": json.dumps({"final_answer_masks": masks})
                    }
                })
                return tool_calls
            except json.JSONDecodeError:
                pass
    
    return tool_calls


# OpenAI function calling format for Qwen3
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "segment_phrase_batch",
            "description": "THE ONLY segmentation tool. Provide ALL possible prompts (3-5 synonyms/variations) at once. SAM3 runs on ALL prompts simultaneously, combines results, and removes duplicates via NMS. This is a ONE-SHOT operation - no retries allowed. After this, you MUST analyze the masks and call select_masks_and_return or report_no_mask.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text_prompts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of 3-5 simple noun phrases (synonyms/variations). Examples: ['windmill', 'wind turbine', 'turbine', 'tower', 'rotor'] or ['ship', 'boat', 'vessel', 'barge', 'cargo ship']"
                    }
                },
                "required": ["text_prompts"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "examine_each_mask",
            "description": "Use this tool when the segment_phrase tool generates multiple small or overlapping mask(s), making it difficult to distinguish the correct mask(s). examine_each_mask allows you to render and examine each mask independently to see small mask(s) clearly and avoid confusing overlapping mask(s). (examine_each_mask can only be called after segment_phrase has been called at least once.)",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "select_masks_and_return",
            "description": "Call this tool to select a subset of or all of the mask(s) rendered on the most recent image as your final output. When calling select_masks_and_return, you cannot select any mask(s) generated by previous rounds other than the most recent round in your 'final_answer_masks'. You can only use mask(s) from the most recent image in your message history. (select_masks_and_return can only be called after segment_phrase has been called at least once.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "final_answer_masks": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "An array of integers representing the selected mask(s) you want to choose as your final output, e.g., [1, 4, 5]"
                    }
                },
                "required": ["final_answer_masks"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "report_no_mask",
            "description": "Call this tool when you are absolutely sure that there are no object(s) in the image that match or answer the initial user input query.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    # DISABLED: Spatial filtering is now done in LLM thinking process, not as a tool
    # The LLM reasons about mask positions and directly selects appropriate masks
    # using select_masks_and_return instead of calling filter_masks_by_spatial_position
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "filter_masks_by_spatial_position",
    #         "description": "Filter existing masks by their spatial position in the image.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "spatial_criteria": {
    #                     "type": "string",
    #                     "description": "Spatial position criteria"
    #                 }
    #             },
    #             "required": ["spatial_criteria"]
    #         }
    #     }
    # },
    # DISABLED: segment_phrase_in_region - Commented out as per user request
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "segment_phrase_in_region",
    #         "description": "Segment a phrase within a specific bounding box region. Useful when you know the approximate location or want to focus on a specific area.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "text_prompt": {
    #                     "type": "string",
    #                     "description": "A short and simple noun phrase"
    #                 },
    #                 "bbox": {
    #                     "type": "array",
    #                     "items": {"type": "number"},
    #                     "description": "Bounding box [x_min, y_min, x_max, y_max] in normalized coordinates (0-1)",
    #                     "minItems": 4,
    #                     "maxItems": 4
    #                 },
    #                 "use_normalized": {
    #                     "type": "boolean",
    #                     "description": "Whether bbox is in normalized coordinates (default: true)",
    #                     "default": True
    #                 }
    #             },
    #             "required": ["text_prompt", "bbox"]
    #         }
    #     }
    # },
    {
        "type": "function",
        "function": {
            "name": "filter_masks_by_attributes",
            "description": "Filter existing masks by visual attributes like color, size, or shape. Use after segment_phrase to narrow down results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "description": "Filter by dominant color (e.g., 'red', 'blue', 'white', 'black')"
                    },
                    # "min_size_ratio": {
                    #     "type": "number",
                    #     "description": "Minimum size as ratio of image (0-1)"
                    # },
                    # "max_size_ratio": {
                    #     "type": "number",
                    #     "description": "Maximum size as ratio of image (0-1)"
                    # },
                    # "aspect_ratio_range": {
                    #     "type": "array",
                    #     "items": {"type": "number"},
                    #     "description": "Min and max aspect ratio [min, max]",
                    #     "minItems": 2,
                    #     "maxItems": 2
                    # }
                }
            }
        }
    },
    # DISABLED: Relative position filtering is now done in LLM thinking process
    # The LLM reasons about mask positions relative to other objects/masks
    # and directly selects appropriate masks using select_masks_and_return
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "filter_masks_by_relative_position",
    #         "description": "Filter masks based on their position relative to other masks.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "relationship": {"type": "string", "description": "Spatial relationship"},
    #                 "reference_mask_indices": {"type": "array", "items": {"type": "integer"}}
    #             },
    #             "required": ["relationship", "reference_mask_indices"]
    #         }
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "segment_phrase_with_tiling",
    #         "description": "Segment a phrase using pyramidal tiling for better detection of small objects or handling large images. Splits image into overlapping tiles, segments each, and merges results intelligently.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "text_prompt": {
    #                     "type": "string",
    #                     "description": "A short and simple noun phrase"
    #                 },
    #                 "tile_size": {
    #                     "type": "integer",
    #                     "description": "Size of each tile (default: 1024)",
    #                     "default": 1024
    #                 },
    #                 "overlap_ratio": {
    #                     "type": "number",
    #                     "description": "Overlap ratio between tiles (0.0-0.5, default: 0.2)",
    #                     "default": 0.2
    #                 },
    #                 "min_tile_size": {
    #                     "type": "integer",
    #                     "description": "Minimum tile size for recursive splitting (default: 512)",
    #                     "default": 512
    #                 }
    #             },
    #             "required": ["text_prompt"]
    #         }
    #     }
    # }
]


def save_debug_messages(messages_list, debug, debug_folder_path, debug_jsonl_path):
    """Save messages to debug jsonl file if debug is enabled"""
    if debug and debug_jsonl_path:
        # Ensure the debug directory exists before writing
        os.makedirs(debug_folder_path, exist_ok=True)
        with open(debug_jsonl_path, "w") as f:
            for msg in messages_list:
                f.write(json.dumps(msg, indent=4) + "\n")


def cleanup_debug_files(debug, debug_folder_path, debug_jsonl_path):
    """Clean up debug files when function successfully returns"""
    if debug and debug_folder_path:
        try:
            if os.path.exists(debug_jsonl_path):
                os.remove(debug_jsonl_path)
            if os.path.exists(debug_folder_path):
                os.rmdir(debug_folder_path)
        except Exception as e:
            print(f"Warning: Could not clean up debug files: {e}")


def count_images(messages):
    """Count the total number of images present in the messages history."""
    total = 0
    for message in messages:
        # Check if message has content (should be a list)
        if "content" in message and isinstance(message["content"], list):
            # Iterate through each content item
            for content_item in message["content"]:
                # Check if content item is a dict with type "image"
                if (
                    isinstance(content_item, dict)
                    and content_item.get("type") == "image"
                ):
                    total += 1
    return total


def _prune_messages_for_next_round(
    messages_list,
    used_text_prompts,
    latest_sam3_text_prompt,
    img_path,
    initial_text_prompt,
):
    """Return a new messages list that contains only:
    1) messages[:2] (with optional warning text added to the second message's content)
    2) the latest assistant message (and everything after it) that contains a segment_phrase tool call
    """
    # Warn if messages list is getting large, but don't crash - pruning will handle it
    if len(messages_list) >= 10:
        print(f"Warning: messages_list has {len(messages_list)} messages before pruning")

    # Part 1: always keep the first two message JSONs
    part1 = copy.deepcopy(messages_list[:2])

    # Part 2: search backwards for the latest assistant message containing a segment_phrase_batch tool call
    part2_start_idx = None
    for idx in range(len(messages_list) - 1, 1, -1):
        msg = messages_list[idx]
        # We only consider assistant messages
        if msg.get("role") != "assistant":
            continue
        # Check for tool_calls field (structured format)
        if "tool_calls" in msg and msg["tool_calls"]:
            for tool_call in msg["tool_calls"]:
                function_name = tool_call.get("function", {}).get("name", "")
                if function_name == "segment_phrase_batch":
                    part2_start_idx = idx
                    break
        # Fallback: check for <tool> tags in content (backward compatibility)
        elif "content" in msg and isinstance(msg["content"], list):
            for content in msg["content"]:
                if (
                    isinstance(content, dict)
                    and content.get("type") == "text"
                    and "<tool>" in content.get("text", "")
                    and "segment_phrase_batch" in content.get("text", "")
                ):
                    part2_start_idx = idx
                    break
        if part2_start_idx is not None:
            break

    part2 = messages_list[part2_start_idx:] if part2_start_idx is not None else []

    # Part 3: decide whether to add warning text to the second message in part1
    # Note: In ONE-SHOT system, we don't allow retries, but keep this for informational purposes
    previously_used = list(used_text_prompts) if used_text_prompts else []
    if part2 and len(previously_used) > 0:
        warning_text = f'Note: segment_phrase_batch has been called (ONE-SHOT). You must now analyze the masks and select them.'
        # Replace the second message entirely to keep exactly 2 content items
        part1[1] = {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {
                    "type": "text",
                    "text": f"The above image is the raw input image. The initial user input query is: '{initial_text_prompt}'."
                    + " "
                    + warning_text,
                },
            ],
        }
        assert len(part1[1]["content"]) == 2

    # Build the new messages list: part1 (with optional warning), then part2
    new_messages = list(part1)
    new_messages.extend(part2)
    return new_messages


def agent_inference(
    img_path: str,
    initial_text_prompt: str,
    debug: bool = False,
    send_generate_request=send_generate_request,
    call_sam_service=call_sam_service,
    max_generations: int = 100,
    output_dir="../../sam3_agent_out",
):
    """
    Given a text prompt and an image, this tool will perform all aspects of agentic problem solving,
    while saving sam3 and MLLM outputs to their respective directories.

    Args:
        img_path: Path to the input image
        initial_text_prompt: Initial text prompt from the user
        debug: Whether to enable debug mode
        max_generations: Maximum number of send_generate_request calls allowed (default: 100)
    """
    # Validate input image path
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Input image not found: {img_path}")
    if not os.path.isfile(img_path):
        raise ValueError(f"Image path is not a file: {img_path}")
    
    # setup dir
    sam_output_dir = os.path.join(output_dir, "sam_out")
    error_save_dir = os.path.join(output_dir, "none_out")
    debug_save_dir = os.path.join(output_dir, "agent_debug_out")
    os.makedirs(sam_output_dir, exist_ok=True)
    os.makedirs(error_save_dir, exist_ok=True)
    os.makedirs(debug_save_dir, exist_ok=True)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    MLLM_SYSTEM_PROMPT_PATH = os.path.join(
        current_dir, "system_prompts/system_prompt_remote_sensing.txt"
    )
    ITERATIVE_CHECKING_SYSTEM_PROMPT_PATH = os.path.join(
        current_dir, "system_prompts/system_prompt_iterative_checking_remote_sensing.txt"
    )
    # init variables
    PATH_TO_LATEST_OUTPUT_JSON = ""
    LATEST_SAM3_TEXT_PROMPT = ""
    USED_TEXT_PROMPTS = (
        set()
    )  # Track all previously used text prompts for segment_phrase
    generation_count = 0  # Counter for number of send_generate_request calls
    
    # Track tool call history for fallback logic
    recent_tool_calls = []  # Track last 3 tool calls
    sam3_segmentation_count = 0  # Count SAM3 segmentation attempts
    
    # Flag to track if masks have been found - once True, no more segment calls allowed
    MASKS_FOUND = False
    MASKS_FOUND_COUNT = 0

    # debug setup
    debug_folder_path = None
    debug_jsonl_path = None
    if debug:
        debug_folder_path = os.path.join(
            debug_save_dir, f"{img_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]}"
        )
        debug_jsonl_path = os.path.join(debug_folder_path, "debug_history.json")
        os.makedirs(debug_folder_path, exist_ok=True)

    # The helper functions are now defined outside the agent_inference function
    with open(MLLM_SYSTEM_PROMPT_PATH, "r") as f:
        system_prompt = f.read().strip()
    with open(ITERATIVE_CHECKING_SYSTEM_PROMPT_PATH, "r") as f:
        iterative_checking_system_prompt = f.read().strip()

    # Construct the initial message list
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {
                    "type": "text",
                    "text": f"The above image is the raw input image. The initial user input query is: '{initial_text_prompt}'.",
                },
            ],
        },
    ]
    print(f"> Text prompt: {initial_text_prompt}")
    print(f"> Image path: {img_path}")

    print("\n\n")
    print("-" * 30 + f" Round {str(generation_count + 1)}" + "-" * 30)
    print("\n\n")
    # Check before first call
    if generation_count >= max_generations:
        raise ValueError(
            f"Exceeded maximum number of allowed generation requests ({max_generations})"
        )
    generation_count += 1
    
    # Retry logic for LLM calls with exponential backoff
    max_retries = 3
    generated_response = None
    for attempt in range(max_retries):
        generated_response = send_generate_request(messages, tools=TOOLS)
        if generated_response is not None:
            break
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
            print(f"⚠️ Attempt {attempt + 1}/{max_retries}: LLM returned None, retrying in {wait_time}s...")
            time.sleep(wait_time)
        else:
            print(f"❌ All {max_retries} attempts failed: LLM returned None")
    
    # Check if we got a valid response on first call
    if generated_response is None:
        print(f"\n❌ MLLM Response: None (after {max_retries} retry attempts)")
        error_save_path = os.path.join(
            error_save_dir,
            f"{img_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]}_error_history.json",
        )
        with open(error_save_path, "w") as f:
            json.dump(messages, f, indent=4)
        print("Saved messages history that caused error to:", error_save_path)
        
        error_msg = (
            f"Generated text is None after {max_retries} retry attempts on initial LLM call. "
            f"This indicates the LLM/VLM server is not returning valid responses.\n"
            f"Image path: {img_path}\n"
            f"Initial text prompt: {initial_text_prompt}\n\n"
            f"Possible causes:\n"
            f"1. LLM/VLM server is down or unreachable\n"
            f"2. Model is returning structured tool_calls instead of text (check server logs)\n"
            f"3. Model server error or timeout\n"
            f"4. Invalid API configuration or authentication issue\n"
            f"5. Model ran out of context window or tokens\n\n"
            f"Check the server logs above for detailed error messages. "
            f"Error history saved to: {error_save_path}"
        )
        raise ValueError(error_msg)
    
    print(f"\n>>> MLLM Response [start]\n{generated_response}\n<<< MLLM Response [end]\n")
    
    # Handle structured tool calls or plain text response
    while generated_response is not None:
        save_debug_messages(messages, debug, debug_folder_path, debug_jsonl_path)
        
        # Initialize tool_calls_list to avoid NameError
        tool_calls_list = []
        generated_text = ""
        
        # Comprehensive type checking for all possible response types
        if generated_response is None:
            print("❌ Error: generated_response is None")
            print(f"   This indicates the LLM API returned no response")
            break
        
        elif isinstance(generated_response, dict):
            # Check if response contains structured tool calls
            if "tool_calls" in generated_response:
                tool_calls_list = generated_response.get("tool_calls", [])
                generated_text = generated_response.get("content", "")
                
                # Validate tool_calls_list is actually a list
                if not isinstance(tool_calls_list, list):
                    print(f"❌ Error: tool_calls is not a list, got {type(tool_calls_list)}")
                    print(f"   Response: {str(generated_response)[:500]}")
                    break
                
                if not tool_calls_list:
                    print("⚠️ Warning: tool_calls list is empty")
                    break
                    
                print(f"✅ Received {len(tool_calls_list)} tool call(s) in structured format")
            else:
                # Dict response without tool_calls - might be an error response
                print(f"⚠️ Warning: Received dict response without 'tool_calls' field")
                print(f"   Response keys: {list(generated_response.keys())}")
                print(f"   Response: {str(generated_response)[:500]}")
                break
                
        elif isinstance(generated_response, list):
            # Handle unexpected list response
            print(f"❌ Error: Received list response instead of dict/string")
            print(f"   Response type: {type(generated_response)}")
            print(f"   List length: {len(generated_response)}")
            print(f"   First item type: {type(generated_response[0]) if generated_response else 'N/A'}")
            print(f"   Response preview: {str(generated_response)[:500]}")
            print(f"   Expected: dict with 'tool_calls' field or string")
            break
            
        elif isinstance(generated_response, str):
            # Text response - parse tool calls from text
            generated_text = generated_response
            print(f"📝 Received text response, parsing for tool calls...")
            print(f"   Response length: {len(generated_text)}")
            
            # Parse tool calls from text response
            tool_calls_list = parse_tool_calls_from_text(generated_text)
            
            if tool_calls_list:
                print(f"✅ Parsed {len(tool_calls_list)} tool call(s) from text")
            else:
                print(f"⚠️ No tool calls found in text response")
                print(f"   Response preview: {generated_text[:500]}")
                break
            
        else:
            # Unexpected response type
            print(f"❌ Error: Unexpected response type: {type(generated_response)}")
            print(f"   Response value: {str(generated_response)[:500]}")
            print(f"   Expected: dict with 'tool_calls' field, string, or None")
            break
        
        # Validate tool_calls_list before processing
        if not tool_calls_list:
            print("⚠️ Warning: No tool calls to process")
            break
        
        # Validate tool call structure before processing
        if not isinstance(tool_calls_list, list):
            print(f"❌ Error: tool_calls_list is not a list, got {type(tool_calls_list)}")
            break
        
        # Process the first tool call (we handle one at a time)
        tool_call_data = tool_calls_list[0]
        
        # Validate tool_call_data structure
        if not isinstance(tool_call_data, dict):
            print(f"❌ Error: tool_call_data is not a dict, got {type(tool_call_data)}")
            print(f"   Tool call data: {tool_call_data}")
            break
        
        # Extract and validate tool_call_id
        tool_call_id = tool_call_data.get("id")
        if not tool_call_id:
            tool_call_id = f"call_{generation_count}_{hash(str(tool_call_data)) % 10000}"
            print(f"⚠️ Warning: tool_call missing 'id' field, generated: {tool_call_id}")
        
        # Extract and validate function data
        function_data = tool_call_data.get("function", {})
        if not isinstance(function_data, dict):
            print(f"❌ Error: tool_call 'function' field is not a dict, got {type(function_data)}")
            print(f"   Tool call data: {tool_call_data}")
            break
        
        # Extract and validate function name
        function_name = function_data.get("name")
        if not function_name or not isinstance(function_name, str):
            print(f"❌ Error: tool_call missing or invalid 'function.name' field")
            print(f"   Function data: {function_data}")
            print(f"   Tool call data: {tool_call_data}")
            break
        
        # Extract and validate function arguments
        function_arguments_str = function_data.get("arguments", "{}")
        if not isinstance(function_arguments_str, str):
            print(f"⚠️ Warning: function arguments is not a string, converting: {type(function_arguments_str)}")
            try:
                function_arguments_str = json.dumps(function_arguments_str)
            except (TypeError, ValueError) as e:
                print(f"❌ Error: Cannot convert arguments to JSON string: {e}")
                print(f"   Arguments: {function_arguments_str}")
                break
        
        # Parse function arguments with better error handling
        try:
            function_arguments = json.loads(function_arguments_str)
            if not isinstance(function_arguments, dict):
                print(f"⚠️ Warning: Parsed arguments is not a dict, got {type(function_arguments)}")
                print(f"   Arguments: {function_arguments}")
                print(f"   Function: {function_name}")
                print(f"   Tool call ID: {tool_call_id}")
                print(f"   Full generated response type: {type(generated_response)}")
                if isinstance(generated_response, dict):
                    print(f"   Generated response keys: {list(generated_response.keys())}")
                # Try to wrap it in a dict or use empty dict
                function_arguments = {"value": function_arguments} if function_arguments else {}
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in tool call arguments")
            print(f"   Function: {function_name}")
            print(f"   Tool call ID: {tool_call_id}")
            print(f"   Arguments string (full): {function_arguments_str}")
            print(f"   JSON error: {e}")
            print(f"   Error position: line {e.lineno}, column {e.colno}" if hasattr(e, 'lineno') else "")
            print(f"   Full generated response type: {type(generated_response)}")
            if isinstance(generated_response, dict):
                print(f"   Generated response keys: {list(generated_response.keys())}")
                print(f"   Generated response preview: {str(generated_response)[:500]}")
            elif isinstance(generated_response, str):
                print(f"   Generated response preview: {generated_response[:500]}")
            print(f"   Generated text: {generated_text[:500] if generated_text else 'None'}")
            raise ValueError(f"Invalid JSON in tool call arguments for '{function_name}': {function_arguments_str[:200]}..., error: {e}")
        
        # Build tool_call dict in expected format
        tool_call = {
            "name": function_name,
            "parameters": function_arguments,
        }
        
        # Track tool call history (keep last 3)
        recent_tool_calls.append(function_name)
        if len(recent_tool_calls) > 3:
            recent_tool_calls.pop(0)
        
        # Note: segment_phrase_batch is ONE-SHOT, count is set inside the handler
        
        if PATH_TO_LATEST_OUTPUT_JSON == "":
            # The first tool call must be segment_phrase_batch or report_no_mask
            assert (
                tool_call["name"] == "segment_phrase_batch"
                or tool_call["name"] == "report_no_mask"
            ), f"First tool call must be segment_phrase_batch or report_no_mask, got {tool_call['name']}"

        if tool_call["name"] == "segment_phrase_batch":
            print("🔍 Calling segment_phrase_batch tool with multiple prompts...")
            
            # CRITICAL: If masks have already been found, reject further segment calls
            if MASKS_FOUND:
                print(f"⚠️ REJECTED: segment_phrase_batch called but {MASKS_FOUND_COUNT} masks already exist!")
                reject_message = f"""⚠️ ERROR: You called segment_phrase_batch but {MASKS_FOUND_COUNT} masks have ALREADY been detected!

This is a ONE-SHOT system. You CANNOT call segment_phrase_batch again.

You MUST either:
1. Call select_masks_and_return([mask_numbers]) to select the masks that match the query
2. Call report_no_mask() if none of the {MASKS_FOUND_COUNT} masks match

DO NOT try to segment again. Analyze the existing masks shown in the image above."""
                messages.append({
                    "role": "assistant",
                    "content": generated_text if generated_text else None,
                    "tool_calls": [tool_call_data],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps({"message": reject_message, "error": "masks_already_found", "num_masks": MASKS_FOUND_COUNT}),
                })
            else:
                text_prompts = tool_call["parameters"].get("text_prompts", [])
                
                if not text_prompts or not isinstance(text_prompts, list):
                    print("❌ Invalid text_prompts parameter")
                    messages.append({
                        "role": "assistant",
                        "content": generated_text if generated_text else None,
                        "tool_calls": [tool_call_data],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps({"message": "Error: text_prompts must be a non-empty array of strings", "num_masks": 0}),
                    })
                else:
                    print(f"📝 Trying {len(text_prompts)} prompts: {text_prompts}")
                    
                    # Track all masks from all prompts
                    all_boxes = []
                    all_masks = []
                    all_scores = []
                    successful_prompts = []
                    failed_prompts = []
                    orig_img_h = None
                    orig_img_w = None
                    
                    for prompt in text_prompts:
                        USED_TEXT_PROMPTS.add(prompt)
                        print(f"  🔍 Trying prompt: '{prompt}'")
                        
                        try:
                            output_json = call_sam_service(
                                image_path=img_path,
                                text_prompt=prompt,
                                output_folder_path=sam_output_dir,
                            )
                            
                            if os.path.exists(output_json):
                                sam3_out = json.load(open(output_json, "r"))
                                num_found = len(sam3_out.get("pred_boxes", []))
                                
                                if num_found > 0:
                                    successful_prompts.append((prompt, num_found))
                                    all_boxes.extend(sam3_out.get("pred_boxes", []))
                                    all_masks.extend(sam3_out.get("pred_masks", []))
                                    all_scores.extend(sam3_out.get("pred_scores", []))
                                    orig_img_h = sam3_out.get("orig_img_h")
                                    orig_img_w = sam3_out.get("orig_img_w")
                                    PATH_TO_LATEST_OUTPUT_JSON = output_json
                                    print(f"    ✓ Found {num_found} masks")
                                else:
                                    failed_prompts.append(prompt)
                                    print(f"    ✗ No masks")
                            else:
                                failed_prompts.append(prompt)
                        except Exception as e:
                            print(f"    ❌ Error: {e}")
                            failed_prompts.append(prompt)
                    
                    # ONE-SHOT: This is the only attempt
                    sam3_segmentation_count = 1
                    total_masks = len(all_boxes)
                    print(f"\n📊 Batch results: {total_masks} total masks from {len(successful_prompts)} successful prompts")
                    
                    # Append assistant message with tool call
                    messages.append({
                        "role": "assistant",
                        "content": generated_text if generated_text else None,
                        "tool_calls": [tool_call_data],
                    })
                    
                    if total_masks == 0:
                        # No masks found - tell agent to call report_no_mask
                        sam3_output_text_message = f"segment_phrase_batch tried {len(text_prompts)} prompts ({text_prompts}) but found 0 masks.\n\n⚠️ This is a ONE-SHOT system. No retries allowed.\n\nYou MUST call report_no_mask() now since no objects were detected."
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps({"message": sam3_output_text_message, "num_masks": 0}),
                        })
                    else:
                        # MASKS FOUND! Set flag to prevent further segment calls
                        MASKS_FOUND = True
                        MASKS_FOUND_COUNT = total_masks
                        
                        # Apply NMS to remove duplicates across prompts
                        from .helpers.mask_overlap_removal import remove_overlapping_masks
                        
                        # Create combined output
                        combined_output = {
                            "orig_img_h": orig_img_h,
                            "orig_img_w": orig_img_w,
                            "pred_boxes": all_boxes,
                            "pred_masks": all_masks,
                            "pred_scores": all_scores,
                            "original_image_path": img_path,
                        }
                        
                        # Count before NMS
                        masks_before_nms = len(all_boxes)
                        print(f"\n🔄 Applying BBox NMS deduplication to {masks_before_nms} total masks...")
                        
                        # Remove overlaps using Bounding Box NMS only
                        combined_output = remove_overlapping_masks(combined_output, bbox_iou_thresh=0.4)
                        final_masks = len(combined_output.get("pred_boxes", []))
                        MASKS_FOUND_COUNT = final_masks
                        
                        # Calculate duplicates removed
                        duplicates_removed = masks_before_nms - final_masks
                        print(f"✅ Final: {masks_before_nms} → {final_masks} unique masks ({duplicates_removed} duplicates removed)")
                        
                        # Save combined output with unique filename based on batch
                        import time
                        batch_id = int(time.time() * 1000) % 100000
                        combined_output_dir = os.path.join(sam_output_dir, f"batch_combined_{batch_id}")
                        os.makedirs(combined_output_dir, exist_ok=True)
                        combined_json_path = os.path.join(combined_output_dir, "sam3_output.json")
                        combined_image_path = os.path.join(combined_output_dir, "sam3_output.png")
                        combined_output["output_image_path"] = combined_image_path
                        
                        with open(combined_json_path, "w") as f:
                            json.dump(combined_output, f, indent=2)
                        
                        # Render visualization
                        try:
                            viz_img = visualize(combined_output)
                            viz_img.save(combined_image_path)
                        except Exception as e:
                            print(f"⚠️ Viz error: {e}")
                            import shutil
                            shutil.copy(img_path, combined_image_path)
                        
                        PATH_TO_LATEST_OUTPUT_JSON = combined_json_path
                        LATEST_SAM3_TEXT_PROMPT = ", ".join([p[0] for p in successful_prompts])
                        
                        prompt_summary = ", ".join([f"'{p}': {n}" for p, n in successful_prompts])
                        
                        # Show deduplication stats
                        dedup_info = ""
                        if duplicates_removed > 0:
                            dedup_info = f"\n🔄 DEDUPLICATION: {duplicates_removed} duplicate masks removed via NMS (same objects detected by multiple prompts)."
                            print(f"🔄 NMS removed {duplicates_removed} duplicate masks ({masks_before_nms} → {final_masks})")
                        
                        # ============================================================
                        # CONFIDENCE ASSESSMENT (parallel for all masks)
                        # ============================================================
                        confidence_info = ""
                        mask_confidences = {}
                        final_mask_confidences = {}
                        BATCH_SIZE = 10  # Masks per batch for parallel processing
                        UNCERTAIN_THRESHOLD = 10  # If MEDIUM+LOW > this, use smart clustering
                        MAX_WORKERS = 4  # Number of parallel API calls
                        
                        if final_masks > 0:
                            print(f"\n🔍 Running confidence assessment for {final_masks} masks...")
                            
                            # Step 1: Parallel batch assessment (handles both small and large mask counts)
                            mask_confidences = batch_assess_confidence(
                                combined_output=combined_output,
                                img_path=img_path,
                                initial_query=initial_text_prompt,
                                sam_output_dir=sam_output_dir,
                                llm_generate_fn=send_generate_request,
                                batch_size=BATCH_SIZE,
                                max_workers=MAX_WORKERS,
                            )
                            
                            # Early return if no confidence assessments were obtained
                            if not mask_confidences:
                                print(f"\n⚠️ No confidence assessments obtained. Skipping smart clustering.")
                                final_mask_confidences = {}
                            # Step 2: Count MEDIUM/LOW - if too many, use smart clustering (also parallel)
                            elif mask_confidences:
                                med_low_count = sum(1 for c in mask_confidences.values() if c in ["MEDIUM", "LOW"])
                                
                                if med_low_count > UNCERTAIN_THRESHOLD:
                                    print(f"\n⚠️ Too many uncertain masks ({med_low_count} > {UNCERTAIN_THRESHOLD})")
                                    print(f"🔍 Running smart clustering assessment (parallel)...")
                                    
                                    # Use smart clustering for better assessment (also parallel)
                                    final_mask_confidences = smart_confidence_assessment(
                                        combined_output=combined_output,
                                        img_path=img_path,
                                        initial_query=initial_text_prompt,
                                        sam_output_dir=sam_output_dir,
                                        initial_confidences=mask_confidences,
                                        llm_generate_fn=send_generate_request,
                                        uncertain_threshold=UNCERTAIN_THRESHOLD,
                                        max_workers=MAX_WORKERS,
                                    )
                                else:
                                    final_mask_confidences = mask_confidences
                            else:
                                final_mask_confidences = {}
                            
                            if final_mask_confidences:
                                # Build confidence summary for the main LLM
                                high_masks = sorted([i+1 for i, c in final_mask_confidences.items() if c == "HIGH"])
                                med_masks = sorted([i+1 for i, c in final_mask_confidences.items() if c == "MEDIUM"])
                                low_masks = sorted([i+1 for i, c in final_mask_confidences.items() if c == "LOW"])
                                
                                confidence_info = f"\n\n📊 PRE-ASSESSED CONFIDENCES:"
                                if high_masks:
                                    confidence_info += f"\n   ✅ HIGH (confirmed): {high_masks}"
                                if med_masks:
                                    confidence_info += f"\n   ⚠️ MEDIUM (need examination): {med_masks}"
                                if low_masks:
                                    confidence_info += f"\n   ❌ LOW (rejected): {low_masks}"
                                
                                # Guidance based on confidences
                                if high_masks and not med_masks:
                                    confidence_info += f"\n\n🎯 All {len(high_masks)} HIGH confidence masks are confirmed. You can select them directly."
                                elif med_masks:
                                    confidence_info += f"\n\n⚠️ {len(med_masks)} masks need examination. Consider calling examine_each_mask for them."
                                if low_masks:
                                    confidence_info += f"\n❌ {len(low_masks)} LOW confidence masks should be EXCLUDED from selection."
                        
                        sam3_output_text_message = f"✅ SUCCESS: Found {final_masks} UNIQUE masks from prompts: {prompt_summary}.{dedup_info}{confidence_info}"
                        
                        # Build a comprehensive analysis prompt
                        # If we have pre-assessed confidences, include guidance
                        pre_assess_guidance = ""
                        if final_mask_confidences:
                            high_list = sorted([i+1 for i, c in final_mask_confidences.items() if c == "HIGH"])
                            med_list = sorted([i+1 for i, c in final_mask_confidences.items() if c == "MEDIUM"])
                            low_list = sorted([i+1 for i, c in final_mask_confidences.items() if c == "LOW"])
                            
                            if high_list or med_list or low_list:
                                pre_assess_guidance = f"""
📊 PRE-ASSESSED CONFIDENCES (use these to speed up your analysis):
"""
                                if high_list:
                                    pre_assess_guidance += f"   ✅ HIGH (confirmed, select these): {high_list}\n"
                                if med_list:
                                    pre_assess_guidance += f"   ⚠️ MEDIUM (verify before selecting): {med_list}\n"
                                if low_list:
                                    pre_assess_guidance += f"   ❌ LOW (exclude these): {low_list}\n"
                        
                        analysis_instruction = f"""
═══════════════════════════════════════════════════════════════════════════════
🎯 MASKS FOUND - NOW ANALYZE THEM
═══════════════════════════════════════════════════════════════════════════════

{final_masks} masks were detected. The masked image is shown below.
{pre_assess_guidance}
YOUR TASK NOW (follow these steps EXACTLY):

1. If PRE-ASSESSED confidences are provided above:
   - HIGH confidence masks: You can select these directly
   - MEDIUM confidence masks: Verify they match the query before selecting
   - LOW confidence masks: EXCLUDE these from selection

2. For query "{initial_text_prompt}", identify which mask(s) match

3. CALL select_masks_and_return([mask_numbers]) with the matching masks
   OR call report_no_mask() if NONE of the masks match the query

⚠️ DO NOT call segment_phrase_batch again! This is a ONE-SHOT system.
   Analyze the existing masks and SELECT the right ones.
═══════════════════════════════════════════════════════════════════════════════
"""
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps({
                                "message": sam3_output_text_message,
                                "num_masks": final_masks,
                                "output_image_path": combined_image_path,
                                "successful_prompts": successful_prompts,
                                "masks_found": True,
                            }),
                        })
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": analysis_instruction},
                                {"type": "image", "image": combined_image_path},
                            ],
                        })
                    
                    print("\n\n>>> sam3_output_text_message:\n", sam3_output_text_message)


        elif tool_call["name"] == "examine_each_mask":
            print("🔍 Calling examine_each_mask tool...")
            assert LATEST_SAM3_TEXT_PROMPT != ""

            # Make sure that the last message is a image
            assert (
                messages[-1]["content"][1]["type"] == "image"
            ), "Second content element should be an image"
            messages.pop()  # Remove the last user message
            # Add simplified replacement message
            simplified_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "The segment_phrase tool generated several masks. Now you must analyze the mask(s) carefully, compare them against the raw input image and the original user query, and determine your next action.",
                    }
                ],
            }
            messages.append(simplified_message)

            if not os.path.exists(PATH_TO_LATEST_OUTPUT_JSON):
                raise FileNotFoundError(
                    f"SAM3 output file not found: {PATH_TO_LATEST_OUTPUT_JSON}"
                )
            current_outputs = json.load(open(PATH_TO_LATEST_OUTPUT_JSON, "r"))
            num_masks = len(current_outputs.get("pred_masks", []))
            masks_to_keep = []

            # MLLM check the mask one by one
            for i in range(num_masks):
                print(f"🔍 Checking mask {i+1}/{num_masks}...")
                image_w_mask_i, image_w_zoomed_in_mask_i = visualize(current_outputs, i)

                image_w_zoomed_in_mask_i_path = os.path.join(
                    sam_output_dir, rf"{LATEST_SAM3_TEXT_PROMPT}.png".replace("/", "_")
                ).replace(".png", f"_zoom_in_mask_{i + 1}.png")
                image_w_mask_i_path = os.path.join(
                    sam_output_dir, rf"{LATEST_SAM3_TEXT_PROMPT}.png".replace("/", "_")
                ).replace(".png", f"_selected_mask_{i + 1}.png")
                image_w_zoomed_in_mask_i.save(image_w_zoomed_in_mask_i_path)
                image_w_mask_i.save(image_w_mask_i_path)

                # Keep vision requests within 2-image limit for providers like OpenAI/vLLM
                iterative_checking_messages = [
                    {"role": "system", "content": iterative_checking_system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"The initial user input query is: '{initial_text_prompt}'. The raw image is the background of the mask render below (omitted as a separate attachment to respect the 2-image limit).",
                            },
                            {
                                "type": "text",
                                "text": f"Image with the predicted segmentation mask rendered on it: ",
                            },
                            {"type": "image", "image": image_w_mask_i_path},
                            {
                                "type": "text",
                                "text": f"Zoomed-in view of the selected mask: ",
                            },
                            {"type": "image", "image": image_w_zoomed_in_mask_i_path},
                        ],
                    },
                ]
                # Count iterative checking as an LLM call
                generation_count += 1
                if generation_count > max_generations:
                    raise ValueError(
                        f"Exceeded maximum number of allowed generation requests ({max_generations})"
                    )
                checking_response = send_generate_request(
                    iterative_checking_messages,
                    tools=None,  # Iterative checking doesn't use tools
                )

                # Normalize response to string (handle dict/list responses)
                if checking_response is None:
                    raise ValueError(
                        "Generated text is None, which is unexpected. Please check the Qwen server and the input parameters."
                    )
                elif isinstance(checking_response, dict):
                    # Extract content from dict response (might have tool_calls)
                    checking_generated_text = checking_response.get("content", "")
                    if not checking_generated_text:
                        raise ValueError(
                            "Response dict has no 'content' field. This may indicate the model returned tool calls instead of text."
                        )
                elif isinstance(checking_response, list):
                    # Handle unexpected list response
                    print(f"⚠️ Warning: Received list response in iterative checking, converting to string")
                    checking_generated_text = str(checking_response)
                elif not isinstance(checking_response, str):
                    # Convert other types to string
                    checking_generated_text = str(checking_response)
                else:
                    checking_generated_text = checking_response
                
                print(f"Generated text for mask {i+1}: {checking_generated_text}")
                verdict = (
                    checking_generated_text.split("<verdict>")[-1]
                    .split("</verdict>")[0]
                    .strip()
                )
                if "Accept" in verdict:
                    assert not "Reject" in verdict
                    print(f"Mask {i+1} accepted, keeping it in the outputs.")
                    masks_to_keep.append(i)
                elif "Reject" in verdict:
                    assert not "Accept" in verdict
                    print(f"Mask {i+1} rejected, removing it from the outputs.")
                else:
                    raise ValueError(
                        f"Unexpected verdict in generated text: {checking_generated_text}. Expected 'Accept' or 'Reject'."
                    )

            updated_outputs = {
                "original_image_path": current_outputs["original_image_path"],
                "orig_img_h": current_outputs["orig_img_h"],
                "orig_img_w": current_outputs["orig_img_w"],
                "pred_boxes": [current_outputs["pred_boxes"][i] for i in masks_to_keep],
                "pred_scores": [
                    current_outputs["pred_scores"][i] for i in masks_to_keep
                ],
                "pred_masks": [current_outputs["pred_masks"][i] for i in masks_to_keep],
            }

            image_w_check_masks = visualize(updated_outputs)
            image_w_check_masks_path = os.path.join(
                sam_output_dir, rf"{LATEST_SAM3_TEXT_PROMPT}.png"
            ).replace(
                ".png",
                f"_selected_masks_{'-'.join(map(str, [i+1 for i in masks_to_keep]))}.png".replace(
                    "/", "_"
                ),
            )
            image_w_check_masks.save(image_w_check_masks_path)
            # save the updated json outputs and append to message history
            messages.append({
                "role": "assistant",
                "content": generated_text if generated_text else None,
                "tool_calls": [tool_call_data],
            })
            
            if len(masks_to_keep) == 0:
                tool_result_message = f"The original user query was: '{initial_text_prompt}'. The examine_each_mask tool examined and rejected all of the masks generated by the segment_phrase tool. Now, please call the segment_phrase tool again with a different, perhaps more general, or more creative simple noun phrase text_prompt, while adhering to all the rules stated in the system prompt."
                # Per Qwen3 docs: Only add tool result, not duplicate user message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps({"message": tool_result_message, "masks_to_keep": []}),
                })
            else:
                tool_result_message = f"The original user query was: '{initial_text_prompt}'. After calling the examine_each_mask tool on the available masks, the number of available masks is now {len(masks_to_keep)}. All {len(masks_to_keep)} available masks are rendered in this image below, now you must analyze the {len(masks_to_keep)} available mask(s) carefully, compare them against the raw input image and the original user query, and determine your next action."
                # Per Qwen3 docs: Only add tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps({
                        "message": tool_result_message,
                        "masks_to_keep": masks_to_keep,
                        "output_image_path": image_w_check_masks_path,
                    }),
                })
                # For vision models: Add user message with image so model can visually analyze masks
                # Note: Tool result contains the message, user message only provides the image for visual analysis
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please analyze the filtered masks shown in the image below."},
                        {"type": "image", "image": image_w_check_masks_path},
                    ],
                })

            # Create a new filename based on the original path to avoid filename length issues
            base_path = PATH_TO_LATEST_OUTPUT_JSON
            # Remove any existing "masks_" suffix to avoid duplication
            if "masks_" in base_path:
                base_path = base_path.split("masks_")[0] + ".json"
            # Create new filename with current masks; use a clearer suffix when empty
            if len(masks_to_keep) == 0:
                PATH_TO_LATEST_OUTPUT_JSON = base_path.replace(
                    ".json", "masks_none.json"
                )
            else:
                PATH_TO_LATEST_OUTPUT_JSON = base_path.replace(
                    ".json", f"masks_{'_'.join(map(str, masks_to_keep))}.json"
                )
            json.dump(updated_outputs, open(PATH_TO_LATEST_OUTPUT_JSON, "w"), indent=4)

    

        elif tool_call["name"] == "filter_masks_by_attributes":
            print("🔍 Calling filter_masks_by_attributes tool...")
            if not os.path.exists(PATH_TO_LATEST_OUTPUT_JSON):
                raise FileNotFoundError(
                    f"SAM3 output file not found: {PATH_TO_LATEST_OUTPUT_JSON}"
                )
            current_outputs = json.load(open(PATH_TO_LATEST_OUTPUT_JSON, "r"))
            
            # Extract filter parameters (all optional)
            color = tool_call["parameters"].get("color")
            min_size_ratio = tool_call["parameters"].get("min_size_ratio")
            max_size_ratio = tool_call["parameters"].get("max_size_ratio")
            aspect_ratio_range = tool_call["parameters"].get("aspect_ratio_range")
            
            # Validate parameters
            if min_size_ratio is not None and (min_size_ratio < 0 or min_size_ratio > 1):
                raise ValueError(f"min_size_ratio must be in [0, 1], got {min_size_ratio}")
            if max_size_ratio is not None and (max_size_ratio < 0 or max_size_ratio > 1):
                raise ValueError(f"max_size_ratio must be in [0, 1], got {max_size_ratio}")
            if aspect_ratio_range is not None:
                if not isinstance(aspect_ratio_range, list) or len(aspect_ratio_range) != 2:
                    raise ValueError(f"aspect_ratio_range must be a list of 2 numbers [min, max], got {aspect_ratio_range}")
                if aspect_ratio_range[0] < 0 or aspect_ratio_range[1] < 0:
                    raise ValueError(f"aspect_ratio_range values must be non-negative, got {aspect_ratio_range}")
                if aspect_ratio_range[0] > aspect_ratio_range[1]:
                    raise ValueError(f"aspect_ratio_range min must be <= max, got {aspect_ratio_range}")
            
            # Check that at least one filter is provided
            if color is None and min_size_ratio is None and max_size_ratio is None and aspect_ratio_range is None:
                raise ValueError("At least one filter parameter (color, min_size_ratio, max_size_ratio, or aspect_ratio_range) must be provided")
            
            # Apply attribute filtering
            filtered_outputs = filter_masks_by_attributes(
                current_outputs,
                color=color,
                min_size_ratio=min_size_ratio,
                max_size_ratio=max_size_ratio,
                aspect_ratio_range=aspect_ratio_range,
            )
            num_kept = len(filtered_outputs["pred_masks"])
            filter_desc = filtered_outputs.get("attribute_filter_description", "")
            
            # Save updated outputs
            image_w_filtered_masks = visualize(filtered_outputs)
            image_w_filtered_masks_path = os.path.join(
                sam_output_dir, rf"{LATEST_SAM3_TEXT_PROMPT}.png"
            ).replace(
                ".png",
                f"_attribute_filter.png".replace("/", "_"),
            )
            image_w_filtered_masks.save(image_w_filtered_masks_path)
            
            # Update PATH_TO_LATEST_OUTPUT_JSON
            base_path = PATH_TO_LATEST_OUTPUT_JSON
            if "masks_" in base_path:
                base_path = base_path.split("masks_")[0] + ".json"
            if num_kept == 0:
                PATH_TO_LATEST_OUTPUT_JSON = base_path.replace(
                    ".json", "masks_none.json"
                )
            else:
                kept_indices = list(range(num_kept))
                PATH_TO_LATEST_OUTPUT_JSON = base_path.replace(
                    ".json", f"masks_{'_'.join(map(str, kept_indices))}.json"
                )
            json.dump(filtered_outputs, open(PATH_TO_LATEST_OUTPUT_JSON, "w"), indent=4)
            
            # Append assistant message with tool call
            messages.append({
                "role": "assistant",
                "content": generated_text if generated_text else None,
                "tool_calls": [tool_call_data],
            })
            
            if num_kept == 0:
                tool_result_message = f"The original user query was: '{initial_text_prompt}'. The filter_masks_by_attributes tool filtered all masks out. {filter_desc} Now, please call the segment_phrase tool again with a different text_prompt, or try different attribute filters."
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps({"message": tool_result_message, "num_masks": 0}),
                })
            else:
                tool_result_message = f"The original user query was: '{initial_text_prompt}'. After calling the filter_masks_by_attributes tool, the number of available masks is now {num_kept}. {filter_desc} All {num_kept} available masks are rendered in this image below, now you must analyze the {num_kept} available mask(s) carefully, compare them against the raw input image and the original user query, and determine your next action."
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps({
                        "message": tool_result_message,
                        "num_masks": num_kept,
                        "output_image_path": image_w_filtered_masks_path,
                    }),
                })
                # For vision models: Add user message with image so model can visually analyze masks
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please analyze the attribute-filtered masks shown in the image below."},
                        {"type": "image", "image": image_w_filtered_masks_path},
                    ],
                })


        elif tool_call["name"] == "select_masks_and_return":
            print("🔍 Calling select_masks_and_return tool...")
            if not os.path.exists(PATH_TO_LATEST_OUTPUT_JSON):
                raise FileNotFoundError(
                    f"SAM3 output file not found: {PATH_TO_LATEST_OUTPUT_JSON}"
                )
            current_outputs = json.load(open(PATH_TO_LATEST_OUTPUT_JSON, "r"))

            assert list(tool_call["parameters"].keys()) == ["final_answer_masks"]
            masks_to_keep = tool_call["parameters"]["final_answer_masks"]

            # Keep only valid mask indices, remove duplicates, and preserve deterministic ascending order
            available_masks = set(range(1, len(current_outputs["pred_masks"]) + 1))
            masks_to_keep = sorted({i for i in masks_to_keep if i in available_masks})
            # Change this to a update message telling the model to try again along with information about errors made.

            final_outputs = {
                "original_image_path": current_outputs["original_image_path"],
                "orig_img_h": current_outputs["orig_img_h"],
                "orig_img_w": current_outputs["orig_img_w"],
                "pred_boxes": [
                    current_outputs["pred_boxes"][i - 1] for i in masks_to_keep
                ],
                "pred_scores": [
                    current_outputs["pred_scores"][i - 1] for i in masks_to_keep
                ],
                "pred_masks": [
                    current_outputs["pred_masks"][i - 1] for i in masks_to_keep
                ],
            }

            rendered_final_output = visualize(final_outputs)
            messages.append({
                "role": "assistant",
                "content": generated_text if generated_text else None,
                "tool_calls": [tool_call_data],
            })

            # Clean up debug files before successful return
            cleanup_debug_files(debug, debug_folder_path, debug_jsonl_path)
            return messages, final_outputs, rendered_final_output

        elif tool_call["name"] == "report_no_mask":
            print("🔍 Calling report_no_mask tool...")
            height, width = cv2.imread(img_path).shape[:2]
            final_outputs = {
                "original_image_path": img_path,
                "orig_img_h": height,
                "orig_img_w": width,
                "pred_boxes": [],
                "pred_scores": [],
                "pred_masks": [],
            }
            rendered_final_output = Image.open(img_path)
            messages.append({
                "role": "assistant",
                "content": generated_text if generated_text else None,
                "tool_calls": [tool_call_data],
            })
            return messages, final_outputs, rendered_final_output

        else:
            raise ValueError(f"Unknown tool call: {tool_call['name']}")

        # Prune the messages history before the next MLLM generation round according to the 3-part rules.
        # This keeps history compact and ensures the model sees only the allowed parts.
        messages = _prune_messages_for_next_round(
            messages,
            USED_TEXT_PROMPTS,
            LATEST_SAM3_TEXT_PROMPT,
            img_path,
            initial_text_prompt,
        )
        # make sure there can never be more than 2 images in the context
        assert count_images(messages) <= 2
        
        # Continue to next iteration to get next tool call or final response
        generation_count += 1
        if generation_count > max_generations:
            raise ValueError(
                f"Exceeded maximum number of allowed generation requests ({max_generations})"
            )

        print("\n\n")
        print("-" * 30 + f" Round {str(generation_count + 1)}" + "-" * 30)
        print("\n\n")
        
        # Retry logic for LLM calls with exponential backoff
        max_retries = 3
        generated_response = None
        for attempt in range(max_retries):
            generated_response = send_generate_request(messages, tools=TOOLS)
            if generated_response is not None:
                break
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"⚠️ Attempt {attempt + 1}/{max_retries}: LLM returned None, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ All {max_retries} attempts failed: LLM returned None")
        
        # Check if we got a valid response
        if generated_response is None:
            print(f"\n❌ MLLM Response: None (after {max_retries} retry attempts)")
        else:
            print(f"\n>>> MLLM Response [start]\n{generated_response}\n<<< MLLM Response [end]\n")
        
        # If response is still None after retries, break the loop and raise error
        if generated_response is None:
            break

    # If we exit the loop and generated_response is still None, raise error
    if generated_response is None:
        print("\n\n>>> SAM 3 Agent execution ended with error.\n\n")

        error_save_path = os.path.join(
            error_save_dir,
            f"{img_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]}_error_history.json",
        )
        with open(error_save_path, "w") as f:
            json.dump(messages, f, indent=4)
        print("Saved messages history that caused error to:", error_save_path)
        
        # Provide more detailed error message
        error_msg = (
            f"Generated text is None after {max_retries} retry attempts. "
            f"This indicates the LLM/VLM server is not returning valid responses.\n"
            f"Image path: {img_path}\n"
            f"Initial text prompt: {initial_text_prompt}\n\n"
            f"Possible causes:\n"
            f"1. LLM/VLM server is down or unreachable\n"
            f"2. Model is returning structured tool_calls instead of text (check server logs)\n"
            f"3. Model server error or timeout\n"
            f"4. Invalid API configuration or authentication issue\n"
            f"5. Model ran out of context window or tokens\n\n"
            f"Check the server logs above for detailed error messages. "
            f"Error history saved to: {error_save_path}"
        )
        raise ValueError(error_msg)