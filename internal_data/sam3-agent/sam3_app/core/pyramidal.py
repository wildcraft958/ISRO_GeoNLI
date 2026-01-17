
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple, Union

# Helper for mask RLE
def rle_encode(mask_binary):
    """Encode binary mask to RLE"""
    import pycocotools.mask as mask_utils
    # Ensure binary
    mask_binary = np.asfortranarray(mask_binary.astype(np.uint8))
    encoded = mask_utils.encode(mask_binary)
    # Convert bytes to string for JSON serialization
    if isinstance(encoded['counts'], bytes):
        encoded['counts'] = encoded['counts'].decode('utf-8')
    return encoded

class PyramidalInference:
    """
    Handles pyramidal tiling inference for SAM3.
    Decoupled from the main model wrapper for modularity.
    """
    
    def __init__(self, processor):
        self.processor = processor

    def _create_tiles(self, image, tile_size: int, overlap_ratio: float):
        """Generate overlapping tiles from PIL image."""
        img_width, img_height = image.size
        stride = int(tile_size * (1 - overlap_ratio))
        tiles = []
        
        if img_width <= tile_size and img_height <= tile_size:
            return [(image, (0, 0))]
        
        for y in range(0, img_height, stride):
            for x in range(0, img_width, stride):
                x_end = min(x + tile_size, img_width)
                y_end = min(y + tile_size, img_height)
                x_start = max(0, x_end - tile_size)
                y_start = max(0, y_end - tile_size)
                
                tile = image.crop((x_start, y_start, x_end, y_end))
                tiles.append((tile, (x_start, y_start)))
                
                if x_end >= img_width: break
            if y_end >= img_height: break
        
        return tiles

    def _transform_box_to_original(self, box, tile_offset, scale: float, orig_size):
        """Transform box from tile coordinates to original image space."""
        offset_x, offset_y = tile_offset
        orig_w, orig_h = orig_size
        
        if torch.is_tensor(box):
            box = box.cpu().numpy()
        
        box = np.array(box).copy()
        box[0] += offset_x
        box[1] += offset_y
        box[2] += offset_x
        box[3] += offset_y
        
        box = box / scale
        
        # Clip
        box[0] = max(0, min(box[0], orig_w))
        box[1] = max(0, min(box[1], orig_h))
        box[2] = max(0, min(box[2], orig_w))
        box[3] = max(0, min(box[3], orig_h))
        
        return box

    def _transform_mask_to_original(self, mask_binary, tile_offset, scale: float, orig_size, tile_size):
        """Transform binary mask from tile coordinates to original image space."""
        from scipy import ndimage
        
        offset_x, offset_y = tile_offset
        orig_w, orig_h = orig_size
        tile_w, tile_h = tile_size
        
        global_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        
        if scale != 1.0:
            scale_factor = 1.0 / scale
            resized_h = int(tile_h * scale_factor)
            resized_w = int(tile_w * scale_factor)
            
            zoom_factors = (resized_h / mask_binary.shape[0], resized_w / mask_binary.shape[1])
            # Use order=0 for nearest neighbor (binary mask)
            mask_resized = ndimage.zoom(mask_binary.astype(float), zoom_factors, order=0) > 0.5
            mask_resized = mask_resized.astype(np.uint8)
        else:
            mask_resized = mask_binary.astype(np.uint8)
            resized_h, resized_w = tile_h, tile_w
        
        # Map to global
        orig_offset_x = int(offset_x / scale)
        orig_offset_y = int(offset_y / scale)
        
        y_start = max(0, orig_offset_y)
        y_end = min(orig_h, orig_offset_y + resized_h)
        x_start = max(0, orig_offset_x)
        x_end = min(orig_w, orig_offset_x + resized_w)
        
        src_y_start = max(0, -orig_offset_y)
        src_y_end = src_y_start + (y_end - y_start)
        src_x_start = max(0, -orig_offset_x)
        src_x_end = src_x_start + (x_end - x_start)
        
        if src_y_end > mask_resized.shape[0]:
            src_y_end = mask_resized.shape[0]
            y_end = y_start + (src_y_end - src_y_start)
        if src_x_end > mask_resized.shape[1]: 
            src_x_end = mask_resized.shape[1]
            x_end = x_start + (src_x_end - src_x_start)
            
        if y_end > y_start and x_end > x_start:
            global_mask[y_start:y_end, x_start:x_end] = mask_resized[src_y_start:src_y_end, src_x_start:src_x_end]
            
        return global_mask

    def run(self, image, text_prompt: str, tile_size=512, overlap_ratio=0.15, scales=[1.0], confidence_threshold=0.3, batch_size=16):
        """
        Run pyramidal inference.
        """
        orig_w, orig_h = image.size
        
        # 1. Encode text once (Optimization)
        try:
             text_outputs = self.processor.model.backbone.forward_text([text_prompt], device=self.processor.device)
        except Exception as e:
            return {"status": "error", "message": f"Text encoding failed: {str(e)}"}

        all_detections = []
        scales = sorted([s for s in scales if s > 0], reverse=True)
        
        # Lower threshold for candidates
        proc_threshold = max(0.1, confidence_threshold * 0.5)
        orig_proc_threshold = self.processor.confidence_threshold
        self.processor.confidence_threshold = proc_threshold
        
        stats = {"scales": scales, "tile_size": tile_size, "total_tiles": 0}
        
        try:
            for scale in scales:
                # Resize
                if scale != 1.0:
                    sw, sh = int(orig_w * scale), int(orig_h * scale)
                    s_img = image.resize((sw, sh), Image.Resampling.LANCZOS)
                else:
                    s_img = image
                
                tiles = self._create_tiles(s_img, tile_size, overlap_ratio)
                stats["total_tiles"] += len(tiles)
                
                # Batch processing
                for i in range(0, len(tiles), batch_size):
                    chunk_tiles = tiles[i:i+batch_size]
                    chunk_images = [t[0] for t in chunk_tiles]
                    chunk_offsets = [t[1] for t in chunk_tiles]
                    
                    try:
                        # 1. Batch encode images (backbone)
                        inference_state = self.processor.set_image_batch(chunk_images)
                        current_batch_size = len(chunk_images)
                        
                        batched_backbone = inference_state['backbone_out']
                        
                        # Process each tile individually using the batched backbone features
                        for b_idx in range(current_batch_size):
                            # Slice backbone features for this specific image from the batch
                            sliced_backbone = {}
                            
                            for k, v in batched_backbone.items():
                                if k == 'sam2_backbone_out':
                                     # Special handling for SAM2 structure if present
                                     # sam2_backbone_out['backbone_fpn'] is list of tensors
                                     sliced_backbone[k] = {
                                         'backbone_fpn': [t[b_idx:b_idx+1] for t in v['backbone_fpn']],
                                         'vision_pos_enc': [t[b_idx:b_idx+1] for t in v['vision_pos_enc']]
                                     }
                                     if 'vision_features' in v:
                                         sliced_backbone[k]['vision_features'] = v['vision_features'][b_idx:b_idx+1]
                                elif isinstance(v, torch.Tensor):
                                    sliced_backbone[k] = v[b_idx:b_idx+1]
                                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                                     sliced_backbone[k] = [t[b_idx:b_idx+1] for t in v]
                                else:
                                    sliced_backbone[k] = v
                                    
                            # Inject text (singular, as we are now processing a single tile's state)
                            sliced_backbone.update({
                                'language_features': text_outputs['language_features'], # (1, ...)
                                'language_mask': text_outputs['language_mask'],
                                'language_embeds': text_outputs['language_embeds'],
                            })
                            
                            # Create single state for the current tile
                            tile_state = {
                                "original_height": chunk_images[b_idx].height,
                                "original_width": chunk_images[b_idx].width,
                                "backbone_out": sliced_backbone
                            }
                            
                            # Run grounding head (cheap compared to backbone)
                            tile_state = self.processor._forward_grounding(tile_state)
                            
                            # Process results for this tile
                            if 'boxes' in tile_state and len(tile_state['boxes']) > 0:
                                boxes = tile_state['boxes'].cpu().numpy()
                                scores = tile_state['scores'].cpu().numpy()
                                masks = tile_state['masks'].cpu()
                                
                                for k in range(len(boxes)):
                                    if scores[k] < confidence_threshold: continue
                                    
                                    # Transform box
                                    orig_box = self._transform_box_to_original(boxes[k], chunk_offsets[b_idx], scale, (orig_w, orig_h))
                                    
                                    # Skip invalid
                                    if orig_box[2] <= orig_box[0] or orig_box[3] <= orig_box[1]: continue
                                    
                                    # Transform mask
                                    mask_tile = masks[k].squeeze().numpy() > 0.5
                                    mask_global = self._transform_mask_to_original(
                                        mask_tile, chunk_offsets[b_idx], scale, (orig_w, orig_h), (chunk_images[b_idx].width, chunk_images[b_idx].height)
                                    )
                                    
                                    # RLE Encode
                                    mask_rle = rle_encode(mask_global)
                                    
                                    # Calculate area
                                    area_pixels = int(np.sum(mask_global))
                                    
                                    all_detections.append({
                                        'box': orig_box.tolist(),
                                        'mask_rle': mask_rle,
                                        'score': float(scores[k]),
                                        'scale': scale,
                                        'pixel_area': area_pixels
                                    })

                    except Exception as e:
                        print(f"Error processing batch starting at tile {i} (scale {scale}): {e}")
                        continue
                        
            # NMS (Simplified box-based for now, can add mask-IoU later)
            final_detections = self._apply_nms(all_detections, 0.5)
            
            return {
                "status": "success",
                "detections": final_detections,
                "orig_img_w": orig_w,
                "orig_img_h": orig_h,
                "pyramidal_stats": stats
            }
            
        finally:
            self.processor.confidence_threshold = orig_proc_threshold

    def _apply_nms(self, detections, iou_threshold):
        if not detections: return []
        
        # Sort by score desc
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        keep = []
        
        while detections:
            curr = detections.pop(0)
            keep.append(curr)
            
            rem = []
            for d in detections:
                # Box IoU
                iou = self._box_iou(curr['box'], d['box'])
                if iou < iou_threshold:
                    rem.append(d)
            detections = rem
        return keep

    def _box_iou(self, b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0
