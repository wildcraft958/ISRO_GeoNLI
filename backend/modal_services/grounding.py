"""
Modal service for GeoGround - Unified Large Vision-Language Model for Remote Sensing Visual Grounding

GeoGround supports:
- Horizontal Bounding Boxes (HBB)
- Oriented Bounding Boxes (OBB)
- Segmentation Masks

Returns normalized xyxy format compatible with sam3_agent.
"""

import re
import json
import base64
import io
from typing import Dict, Any, List, Optional, Literal, Union
from functools import partial

import modal
from pydantic import BaseModel, Field, ConfigDict
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Define the Modal App
app = modal.App("geoground-service")

# 1. Define the Environment (Image)
# GeoGround relies on LLaVA-1.5, so we need to set up that environment first.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",  # Required for model loading
        "peft>=0.10.0",  # Required for LoRA adapters, newer version fixes accelerate compatibility
        "bitsandbytes",
        "Pillow",
        "numpy",
        "requests",
        "sentencepiece",
        "protobuf",
        "fastapi",
        "pydantic>=2.0",
    )
    # Install LLaVA from source (required dependency for GeoGround)
    .run_commands(
        "git clone https://github.com/haotian-liu/LLaVA.git /root/LLaVA",
        "cd /root/LLaVA && pip install -e ."
    )
    # Reinstall peft with compatible version after LLaVA (LLaVA may install incompatible version)
    .pip_install(
        "peft>=0.10.0",  # Ensure compatible version after LLaVA installation
    )
    # Clone GeoGround repository for specific inference utilities
    .run_commands(
        "git clone https://github.com/VisionXLab/GeoGround.git /root/GeoGround"
    )
    .env({"PYTHONPATH": "/root/GeoGround:/root/LLaVA"})
)

# 2. Define Model Weights
# Note: GeoGround model may not be available on HuggingFace yet.
# If the model is not available, we'll use the base LLaVA model as fallback.
# Users can specify custom model paths via environment variables or config.
MODEL_PATH = "erenzhou/GeoGround"  # May need to be updated to actual model path
BASE_MODEL_PATH = "liuhaotian/llava-v1.5-7b"


# ------------------------------------------------------------------------------
# Pydantic Models for API
# ------------------------------------------------------------------------------

class GeoGroundRequest(BaseModel):
    """Request model for GeoGround inference"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "image_url": "https://example.com/image.jpg",
            "prompt": "Find the airport runway in this image",
            "output_type": "all"
        }
    })
    
    image_url: Optional[str] = Field(None, description="URL of the image to process")
    image_b64: Optional[str] = Field(None, description="Base64 encoded image (data URI or raw base64)")
    prompt: str = Field(..., description="Text query describing what to locate")
    output_type: Literal["hbb", "obb", "mask", "all"] = Field(
        default="all",
        description="Type of output: 'hbb' (horizontal bbox), 'obb' (oriented bbox), 'mask' (segmentation), or 'all'"
    )


class RegionInfo(BaseModel):
    """Segmented region information"""
    id: int = Field(..., description="Region ID")
    label: Optional[str] = Field(None, description="Region label")
    score: float = Field(..., description="Confidence score")
    box: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2] normalized [0,1]")
    mask_rle: Optional[Dict[str, Any]] = Field(None, description="RLE-encoded mask")
    type: Literal["hbb", "obb", "mask"] = Field(..., description="Output type")


class GeoGroundResponse(BaseModel):
    """Response model for GeoGround inference"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "success",
            "summary": "Detected 3 objects",
            "regions": [
                {
                    "id": 1,
                    "label": "airport",
                    "score": 0.95,
                    "box": [0.1, 0.2, 0.3, 0.4],
                    "type": "hbb"
                }
            ]
        }
    })
    
    status: str = Field(..., description="Response status: 'success' or 'error'")
    summary: Optional[str] = Field(None, description="Summary of results")
    regions: Optional[List[Dict[str, Any]]] = Field(None, description="List of detected regions")
    message: Optional[str] = Field(None, description="Error message if status is 'error'")
    traceback: Optional[str] = Field(None, description="Error traceback if status is 'error'")


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def rle_encode(mask) -> List[Dict[str, Any]]:
    """
    Encode binary mask to RLE format.
    
    Args:
        mask: Binary mask tensor of shape (N, H, W) or (H, W) (torch.Tensor)
    
    Returns:
        List of RLE dictionaries with 'size' and 'counts' keys
    """
    import torch
    import numpy as np
    
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    
    results = []
    for i in range(mask.shape[0]):
        m = mask[i].cpu().numpy().astype(np.uint8)
        h, w = m.shape
        
        # Flatten mask
        flat = m.flatten()
        
        # Compute RLE
        counts = []
        current = flat[0]
        count = 1
        
        for val in flat[1:]:
            if val == current:
                count += 1
            else:
                counts.append(count)
                current = val
                count = 1
        counts.append(count)
        
        results.append({
            "size": [h, w],
            "counts": counts
        })
    
    return results


def parse_geoground_output(output_text: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
    """
    Parse GeoGround model output text to extract HBB, OBB, and mask information.
    
    GeoGround outputs text in formats like:
    - <box>[[x1,y1,x2,y2]]</box>  (normalized to resolution 1000)
    - <obb>[[xc,yc,w,h,θ]]</obb>  (normalized to resolution 100)
    - <seg>0,0,0,...;0,1,1,0,...</seg>  (binary matrix, NxN grid)
    
    Args:
        output_text: Raw text output from GeoGround model
        image_width: Original image width in pixels
        image_height: Original image height in pixels
    
    Returns:
        List of region dictionaries
    """
    regions = []
    region_id = 1
    
    # Parse HBB (Horizontal Bounding Boxes)
    hbb_pattern = r'<box>\[\[([^\]]+)\]\]</box>'
    for match in re.finditer(hbb_pattern, output_text):
        coords_str = match.group(1)
        try:
            # Parse coordinates: x1,y1,x2,y2 (normalized to resolution 1000)
            coords = [float(x.strip()) for x in coords_str.split(',')]
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                # Convert from resolution 1000 to normalized [0,1]
                x1_norm = x1 / 1000.0
                y1_norm = y1 / 1000.0
                x2_norm = x2 / 1000.0
                y2_norm = y2 / 1000.0
                
                # Ensure coordinates are in [0,1] range
                x1_norm = max(0.0, min(1.0, x1_norm))
                y1_norm = max(0.0, min(1.0, y1_norm))
                x2_norm = max(0.0, min(1.0, x2_norm))
                y2_norm = max(0.0, min(1.0, y2_norm))
                
                # Ensure x1 < x2 and y1 < y2
                if x1_norm > x2_norm:
                    x1_norm, x2_norm = x2_norm, x1_norm
                if y1_norm > y2_norm:
                    y1_norm, y2_norm = y2_norm, y1_norm
                
                regions.append({
                    "id": region_id,
                    "label": None,
                    "score": 1.0,  # GeoGround doesn't provide confidence scores
                    "box": [x1_norm, y1_norm, x2_norm, y2_norm],
                    "mask_rle": None,
                    "type": "hbb"
                })
                region_id += 1
        except (ValueError, IndexError) as e:
            print(f"Warning: Failed to parse HBB coordinates: {coords_str}, error: {e}")
            continue
    
    # Parse OBB (Oriented Bounding Boxes)
    obb_pattern = r'<obb>\[\[([^\]]+)\]\]</obb>'
    for match in re.finditer(obb_pattern, output_text):
        coords_str = match.group(1)
        try:
            # Parse coordinates: xc,yc,w,h,θ (normalized to resolution 100)
            coords = [float(x.strip()) for x in coords_str.split(',')]
            if len(coords) == 5:
                xc, yc, w, h, theta = coords
                # Convert from resolution 100 to normalized [0,1]
                xc_norm = xc / 100.0
                yc_norm = yc / 100.0
                w_norm = w / 100.0
                h_norm = h / 100.0
                
                # Convert OBB to HBB by computing bounding box of rotated rectangle
                # For simplicity, we'll use a conservative approximation
                # Calculate corner points of the rotated rectangle
                import math
                cos_t = math.cos(math.radians(theta))
                sin_t = math.sin(math.radians(theta))
                
                # Half dimensions
                hw = w_norm / 2.0
                hh = h_norm / 2.0
                
                # Corner points relative to center
                corners = [
                    (-hw, -hh),
                    (hw, -hh),
                    (hw, hh),
                    (-hw, hh)
                ]
                
                # Rotate and translate corners
                rotated_corners = []
                for dx, dy in corners:
                    rx = dx * cos_t - dy * sin_t
                    ry = dx * sin_t + dy * cos_t
                    rotated_corners.append((xc_norm + rx, yc_norm + ry))
                
                # Find bounding box
                xs = [x for x, y in rotated_corners]
                ys = [y for x, y in rotated_corners]
                x1_norm = max(0.0, min(1.0, min(xs)))
                y1_norm = max(0.0, min(1.0, min(ys)))
                x2_norm = max(0.0, min(1.0, max(xs)))
                y2_norm = max(0.0, min(1.0, max(ys)))
                
                regions.append({
                    "id": region_id,
                    "label": None,
                    "score": 1.0,
                    "box": [x1_norm, y1_norm, x2_norm, y2_norm],
                    "mask_rle": None,
                    "type": "obb"
                })
                region_id += 1
        except (ValueError, IndexError) as e:
            print(f"Warning: Failed to parse OBB coordinates: {coords_str}, error: {e}")
            continue
    
    # Parse Mask (Segmentation)
    seg_pattern = r'<seg>([^<]+)</seg>'
    for match in re.finditer(seg_pattern, output_text):
        mask_text = match.group(1).strip()
        try:
            # Parse binary matrix: rows separated by ';', values separated by ','
            rows = mask_text.split(';')
            if not rows:
                continue
            
            # Determine grid size (assume square grid)
            first_row = rows[0].split(',')
            grid_size = len(first_row)
            
            # Parse all rows
            mask_matrix = []
            for row_str in rows:
                if not row_str.strip():
                    continue
                row = [int(x.strip()) for x in row_str.split(',') if x.strip()]
                if len(row) == grid_size:
                    mask_matrix.append(row)
            
            if not mask_matrix or len(mask_matrix) != grid_size:
                print(f"Warning: Invalid mask matrix size, expected {grid_size}x{grid_size}, got {len(mask_matrix)} rows")
                continue
            
            # Convert to binary mask and resize to image dimensions
            import numpy as np
            import torch
            from PIL import Image
            
            # Create binary mask from matrix
            mask_array = np.array(mask_matrix, dtype=np.uint8)
            
            # Resize to original image dimensions
            mask_pil = Image.fromarray(mask_array * 255, mode='L')
            mask_resized = mask_pil.resize((image_width, image_height), Image.NEAREST)
            mask_binary = np.array(mask_resized) > 127
            
            # Convert to RLE
            mask_tensor = torch.tensor(mask_binary.astype(np.uint8))
            rle_list = rle_encode(mask_tensor)
            
            if rle_list:
                # Compute bounding box from mask
                y_coords, x_coords = np.where(mask_binary)
                if len(x_coords) > 0 and len(y_coords) > 0:
                    x1_px = float(np.min(x_coords))
                    y1_px = float(np.min(y_coords))
                    x2_px = float(np.max(x_coords)) + 1
                    y2_px = float(np.max(y_coords)) + 1
                    
                    # Normalize to [0,1]
                    x1_norm = x1_px / image_width
                    y1_norm = y1_px / image_height
                    x2_norm = x2_px / image_width
                    y2_norm = y2_px / image_height
                    
                    regions.append({
                        "id": region_id,
                        "label": None,
                        "score": 1.0,
                        "box": [x1_norm, y1_norm, x2_norm, y2_norm],
                        "mask_rle": rle_list[0],
                        "type": "mask"
                    })
                    region_id += 1
        except Exception as e:
            print(f"Warning: Failed to parse mask: {e}")
            continue
    
    return regions


# ------------------------------------------------------------------------------
# GeoGround Model Class
# ------------------------------------------------------------------------------

@app.cls(image=image, gpu="A10G", timeout=600)
class GeoGroundModel:
    def build_model(self):
        """
        Loads the model components into GPU memory once when the container starts.
        
        Note: If GeoGround model is not available on HuggingFace, this will fall back
        to using the base LLaVA model. The model path can be configured via environment
        variables or by modifying MODEL_PATH.
        """
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        import torch
        import os

        print("Loading GeoGround Model...")
        
        # Check if custom model path is provided via environment variable
        model_path = os.getenv("GEOGROUND_MODEL_PATH", MODEL_PATH)
        base_model_path = os.getenv("GEOGROUND_BASE_MODEL_PATH", BASE_MODEL_PATH)
        
        # We assume GeoGround acts as a LoRA or full fine-tune on LLaVA 1.5
        # If the model is a full merge, model_base can be None.
        # Based on typical usage, we attempt to load it with the base first.
        
        # Strategy 1: Try loading with base model (for LoRA adapters)
        try:
            print(f"Attempting to load model: {model_path} with base: {base_model_path}")
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_path,
                model_base=base_model_path,
                model_name=get_model_name_from_path(model_path),
                load_8bit=False,
                load_4bit=True,  # Use 4-bit for efficiency on A10G
                device="cuda"
            )
            print("Model loaded successfully with base model.")
            return
        except Exception as e:
            print(f"Error loading model with base: {e}")
        
        # Strategy 2: Try loading without base model (for full merged models)
        try:
            print("Attempting to load model without base (assuming full merge)...")
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=get_model_name_from_path(model_path),
                load_8bit=False,
                load_4bit=True,
                device="cuda"
            )
            print("Model loaded successfully (without base).")
            return
        except Exception as e2:
            print(f"Error loading model without base: {e2}")
        
        # Strategy 3: Fall back to base LLaVA model if GeoGround is not available
        print(f"GeoGround model not available. Falling back to base LLaVA model: {base_model_path}")
        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=base_model_path,
                model_base=None,
                model_name=get_model_name_from_path(base_model_path),
                load_8bit=False,
                load_4bit=True,
                device="cuda"
            )
            print("Base LLaVA model loaded successfully (GeoGround features may be limited).")
        except Exception as e3:
            raise RuntimeError(
                f"Failed to load GeoGround model and fallback to base LLaVA also failed. "
                f"Original errors: {e}, {e2}, {e3}"
            )

    @modal.enter()
    def setup(self):
        self.build_model()

    @modal.method()
    def inference(self, image_bytes: bytes, prompt: str, output_type: str = "all"):
        """
        Runs inference on a single image and prompt.
        
        Args:
            image_bytes: Raw image bytes
            prompt: Text query describing what to locate
            output_type: Type of output desired ("hbb", "obb", "mask", or "all")
        
        Returns:
            Dict with status, regions, and summary
        """
        import torch
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from PIL import Image
        import io

        try:
            # 1. Load and Process Image
            try:
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                image_width, image_height = image.size
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to load image: {str(e)}",
                    "traceback": None
                }

            image_tensor = process_images([image], self.image_processor, self.model.config)
            if type(image_tensor) is list:
                image_tensor = [img.to(self.model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

            # 2. Process Prompt
            # GeoGround usually follows LLaVA conversation templates.
            # We wrap the user prompt with the image token.
            inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt_str = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt_str, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).cuda()

            # 3. Generate Output
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )

            output = self.tokenizer.decode(
                output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            
            # 4. Parse Output
            all_regions = parse_geoground_output(output, image_width, image_height)
            
            # Filter by output_type if specified
            if output_type != "all":
                filtered_regions = [r for r in all_regions if r["type"] == output_type]
            else:
                filtered_regions = all_regions
            
            # Generate summary
            summary = f"GeoGround detected {len(filtered_regions)} region(s) for prompt: '{prompt}'"
            if output_type != "all":
                summary += f" (type: {output_type})"
            
            return {
                "status": "success",
                "summary": summary,
                "regions": filtered_regions,
                "raw_output": output  # Include raw output for debugging
            }
            
        except Exception as e:
            import traceback
            return {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }


# ------------------------------------------------------------------------------
# FastAPI Web Endpoint
# ------------------------------------------------------------------------------

@app.function(timeout=900, image=image)
@modal.asgi_app()
def fastapi_app():
    """
    FastAPI ASGI application for GeoGround service.
    
    Provides endpoint:
    - POST /geoground/inference - Visual grounding inference
    """
    import base64
    import httpx
    
    api = FastAPI(
        title="GeoGround Visual Grounding API",
        description="Unified Large Vision-Language Model for Remote Sensing Visual Grounding",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    http_client = httpx.Client(timeout=30.0)
    
    def get_image_bytes(image_url: Optional[str], image_b64: Optional[str]) -> bytes:
        """Get image bytes from URL or base64 string."""
        if image_b64:
            # Handle data URI or raw base64
            if image_b64.startswith("data:"):
                # Data URI format: data:image/jpeg;base64,...
                header, data = image_b64.split(",", 1)
                return base64.b64decode(data)
            else:
                # Raw base64
                return base64.b64decode(image_b64)
        elif image_url:
            if image_url.startswith("http://") or image_url.startswith("https://"):
                response = http_client.get(image_url)
                response.raise_for_status()
                return response.content
            else:
                raise ValueError(f"Invalid image URL: {image_url}")
        else:
            raise ValueError("Either image_url or image_b64 must be provided")
    
    @api.post(
        "/geoground/inference",
        response_model=GeoGroundResponse,
        tags=["Inference"],
        summary="Visual Grounding Inference",
        description="""
        Perform visual grounding on remote sensing images using GeoGround.
        
        Supports:
        - Horizontal Bounding Boxes (HBB)
        - Oriented Bounding Boxes (OBB)
        - Segmentation Masks
        
        Returns normalized xyxy bounding boxes compatible with sam3_agent format.
        """,
        responses={
            200: {"description": "Successful inference", "model": GeoGroundResponse},
            400: {"description": "Invalid request"},
            500: {"description": "Server error"}
        }
    )
    async def geoground_inference(request: GeoGroundRequest):
        """Run GeoGround inference on image and prompt."""
        try:
            # Get image bytes
            image_bytes = get_image_bytes(request.image_url, request.image_b64)
            
            # Call inference
            result = GeoGroundModel().inference.remote(
                image_bytes=image_bytes,
                prompt=request.prompt,
                output_type=request.output_type
            )
            
            return JSONResponse(content=result)
            
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback_str = traceback.format_exc()
            print(f"❌ Error in geoground_inference: {error_msg}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": error_msg,
                    "traceback": traceback_str
                }
            )
    
    @api.get(
        "/health",
        tags=["Health"],
        summary="Health Check",
        description="Check if the GeoGround service is running."
    )
    async def health():
        """Health check endpoint."""
        return {"status": "ok", "service": "geoground-service"}
    
    return api
