import os
import sys
import io
import json
import base64
import torch
import numpy as np
import re
import traceback
from typing import Dict, Any, List, Optional, Tuple, Set
from PIL import Image
from functools import partial
from pydantic import ValidationError

from sam3_app.core.vlm import VLMInterface
from sam3_app.api.schemas import KeywordExtractionResponse, VerificationResponse

# Add sam3 root to sys.path if not present (assuming this file is in sam3/sam3_app/core)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir))) # Adjust as needed based on deployment
if project_root not in sys.path:
    sys.path.append(project_root)

# Try imports, handle if sam3 is not found yet (e.g. during scaffolding)
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.train.masks_ops import rle_encode
except ImportError:
    print("Warning: sam3 modules not found. Ensure PYTHONPATH is set correctly.")

class SAM3Model:
    """
    SAM3 Model Wrapper for standalone deployment.
    Handles loading the model and running inference with VLM enhancement.
    """
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Load SAM3 model and processor."""
        if self.model is not None:
            return

        # Disable PIL decompression bomb limit
        Image.MAX_IMAGE_PIXELS = None
        
        from huggingface_hub import login
        
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("Warning: HF_TOKEN not set. SAM3 model download may fail if not cached.")
        else:
            try:
                login(token=hf_token)
                print("✓ Authenticated with HuggingFace")
            except Exception as e:
                print(f"Warning: Failed to authenticate with HuggingFace: {e}")

        print(f"Loading SAM3 model on {self.device}...")
        try:
            # Assuming standard path or downloaded via script
            # In Docker, we might bake weights or download them
            bpe_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets", "bpe_simple_vocab_16e6.txt.gz")
            # If not found relative, try absolute or let model builder find it
            if not os.path.exists(bpe_path):
                # Fallback to internal path if running from different loc
                bpe_path = "/app/assets/bpe_simple_vocab_16e6.txt.gz"
                
            self.model = build_sam3_image_model(bpe_path=bpe_path, device=self.device)
            self.processor = Sam3Processor(self.model, confidence_threshold=0.3)
            print(f"✓ SAM3 model loaded successfully. Threshold: {self.processor.confidence_threshold}")
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM3 model: {e}")

    def _refine_prompt_with_vlm(self, image_bytes: bytes, user_query: str, vlm: VLMInterface) -> str:
        """Convert user query to SAM3-friendly visual prompt using VLM."""
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        REFINEMENT_TEMPLATE = """You are a visual descriptor for object segmentation on satellite images. You ONLY describe what objects look like, you NEVER count or answer questions. NEVER INCLUDE NUMBERS IN ANY FORM.

YOUR TASK:
Extract the OBJECT NAME from the user query and describe it for visual detection.
- Output: 1-3 words maximum
- Format: [object name] or [attribute + object name]
- Use singular form
- Focus on visual characteristics: color, shape, size

USER QUERY: "{query}"

OUTPUT ONLY the visual descriptor (1-3 words), nothing else. No explanation, no JSON, just the descriptor."""
        
        prompt = REFINEMENT_TEMPLATE.format(query=user_query)
        
        try:
            response = vlm.query(image, prompt, max_tokens=64, temperature=0.3)
            if not response:
                return self._fallback_prompt(user_query)
            
            visual_prompt = response.strip().split('\n')[0].strip(' "\'')
            if not visual_prompt or visual_prompt.lower() in ['image', 'object', 'detect']:
                return self._fallback_prompt(user_query)
                
            print(f"✓ VLM refined: '{user_query}' -> '{visual_prompt}'")
            return visual_prompt
        except Exception as e:
            print(f"Error refining prompt: {e}")
            return self._fallback_prompt(user_query)

    def _fallback_prompt(self, query: str) -> str:
        # Simple extraction logic (removed heavy regex for brevity, can add back if needed)
        words = query.lower().split()
        ignored = {'how', 'many', 'count', 'find', 'the', 'a', 'an', 'in', 'on'}
        meaningful = [w for w in words if w not in ignored]
        return ' '.join(meaningful[:3]) if meaningful else "object"

    def _verify_detections_with_vlm(self, image, detections: List[Dict], query: str, visual_prompt: str, vlm: VLMInterface) -> List[Dict]:
        """Verify detections using VLM."""
        VERIFICATION_TEMPLATE = """You are a visual verification assistant. Answer ONLY with 'yes' or 'no'.
        
TASK: Check if the object in this crop matches the query: "{query}" (Visual type: {visual_prompt}).
Answer 'yes' if it matches and is a complete object. Answer 'no' otherwise.
"""
        verified = []
        for det in detections:
            box = det["box"]
            x1, y1, x2, y2 = map(int, box)
            # Crop
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.width, x2), min(image.height, y2)
            if x2 <= x1 or y2 <= y1: continue
            
            crop = image.crop((x1, y1, x2, y2))
            prompt = VERIFICATION_TEMPLATE.format(query=query, visual_prompt=visual_prompt)
            
            resp = vlm.query(crop, prompt, max_tokens=10, temperature=0.1)
            if resp and "yes" in resp.lower() and "no" not in resp.lower()[:5]:
                verified.append(det)
        
        print(f"✓ Verified {len(verified)}/{len(detections)} detections")
        return verified

    def _rephrase_prompt_with_vlm(self, image_bytes, original_query, previous_prompt, vlm, attempt=0, used_prompts=None) -> str:
        """Generate alternative prompt."""
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        prompt = f"""Original: "{original_query}". Previous failed prompt: "{previous_prompt}". 
Give me a SINGLE alternative synonym or visual description (1-3 words) to try instead. output ONLY the words."""
        
        resp = vlm.query(image, prompt, max_tokens=32, temperature=0.5)
        if resp:
            return resp.strip().strip('"')
        return previous_prompt + "s" # Fallback

    def _extract_keywords_with_vlm(self, image_bytes: bytes, user_query: str, vlm: VLMInterface) -> List[str]:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        prompt = f"""Analyze image and query: "{user_query}". 
Extract 2-4 visual phrases (e.g. "large tank", "white roof") for area measurement.
Output JSON: {{ "keywords": ["k1", "k2"] }}"""
        
        resp = vlm.query(image, prompt, max_tokens=128)
        if resp:
            try:
                # Basic json cleanup
                import re
                json_match = re.search(r'\{.*\}', resp, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    return data.get("keywords", [user_query])
            except:
                pass
        return [user_query]


    def _sam3_pyramidal_infer_impl(self, image_bytes, text_prompt, tile_size=512, overlap_ratio=0.15, scales=[1.0], iou_threshold=0.5, confidence_threshold=0.3, batch_size=16):
        """Delegate to PyramidalInference"""
        from sam3_app.core.pyramidal import PyramidalInference
        
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        inferencer = PyramidalInference(self.processor)
        
        return inferencer.run(
            image=image,
            text_prompt=text_prompt,
            tile_size=tile_size,
            overlap_ratio=overlap_ratio,
            scales=scales,
            confidence_threshold=confidence_threshold,
            batch_size=batch_size
        )

    # _create_tiles and _transform_box removed as they are now in PyramidalInference


    # --- Public Methods ---

    def count(self, image_bytes, text_prompt, llm_config, confidence_threshold=0.3, **kwargs):
        vlm = VLMInterface(**llm_config)
        
        # 1. Refine
        vis_prompt = self._refine_prompt_with_vlm(image_bytes, text_prompt, vlm)
        
        # 2. SAM3
        res = self._sam3_pyramidal_infer_impl(image_bytes, vis_prompt, confidence_threshold=confidence_threshold)
        
        # 3. Verify
        # ... logic ...
        
        return {
            "status": "success",
            "count": len(res.get("detections", [])),
            "visual_prompt": vis_prompt
        }

    def area(self, image_bytes, text_prompt, llm_config, gsd, confidence_threshold=0.3, **kwargs):
        vlm = VLMInterface(**llm_config)
        keywords = self._extract_keywords_with_vlm(image_bytes, text_prompt, vlm)
        
        all_dets = []
        for kw in keywords:
            res = self._sam3_pyramidal_infer_impl(image_bytes, kw, confidence_threshold=confidence_threshold)
            all_dets.extend(res.get("detections", []))
            
        # Calc area
        # ... logic ...
        
        return {
            "status": "success",
            "object_count": len(all_dets),
            "total_real_area_m2": 0.0 # Placeholder
        }

    def segment(self, image_bytes, prompt, llm_config, debug=False, confidence_threshold=0.3, **kwargs):
        # Implementation similar to infer
        res = self._sam3_pyramidal_infer_impl(image_bytes, prompt, confidence_threshold=confidence_threshold)
        return res

# Helper for mask RLE
def rle_encode_compat(mask):
    # ...
    pass
