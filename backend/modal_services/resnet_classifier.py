"""
Modal service for ResNet50-based image modality classification.

This service classifies satellite images into three modalities:
- rgb: Standard RGB images
- infrared: Multispectral/NIR images
- sar: Synthetic Aperture Radar images

Deployed on CPU for cost efficiency.
"""

import io
import base64
from typing import Optional

import modal

# --- CONFIGURATION ---
WEIGHTS_PATH = "/weights/best_resnet.pth"
CLASS_NAMES = ["rgb", "infrared", "sar"]
NUM_CLASSES = 3

# Create Modal app and volume
app = modal.App("resnet-modality-classifier")
vol = modal.Volume.from_name("resnet-modality-weights", create_if_missing=True)

# Define container image with PyTorch (CPU version for efficiency)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "pillow",
        "fastapi",
        "httpx",
        "pydantic>=2.0",
    )
)

# Global model cache
_model_cache = None
_transform_cache = None


def get_model_and_transform():
    """Load and cache the ResNet model and transforms."""
    global _model_cache, _transform_cache
    
    if _model_cache is None:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        
        print("🔄 Loading ResNet50 model...")
        
        # Create model architecture
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        
        # Load trained weights
        model.load_state_dict(
            torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
        )
        model.eval()
        
        _model_cache = model
        print("✅ ResNet50 model loaded")
        
        # Create transform
        _transform_cache = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return _model_cache, _transform_cache


@app.function(
    image=image,
    volumes={"/weights": vol},
    cpu=2,
    memory=4096,  # 4GB RAM
    timeout=60,
    scaledown_window=3000,  # Keep warm for 5 minutes
    min_containers=1,
)
def classify_image(image_bytes: bytes) -> dict:
    """
    Classify image modality using ResNet50.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        dict with modality, confidence, and logits
    """
    import torch
    from PIL import Image
    
    model, transform = get_model_and_transform()
    
    # Load image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Transform and predict
    x = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(dim=1)
    
    modality = CLASS_NAMES[pred_idx.item()]
    
    return {
        "modality": modality,
        "confidence": float(confidence.item()),
        "logits": logits[0].tolist(),
        "probabilities": {
            name: float(probs[0, i].item())
            for i, name in enumerate(CLASS_NAMES)
        }
    }


@app.function(
    image=image,
    volumes={"/weights": vol},
    cpu=2,
    memory=4096,
    timeout=60,
    scaledown_window=300,
)
@modal.asgi_app()
def serve():
    """FastAPI web server for the classifier."""
    import httpx
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    fastapi_app = FastAPI(
        title="ResNet Modality Classifier",
        description="Classify satellite images into rgb/infrared/sar modalities",
        version="1.0.0"
    )
    
    # HTTP client for downloading images
    http_client = httpx.AsyncClient(timeout=30.0)
    
    class ClassifyRequest(BaseModel):
        image_url: Optional[str] = None
        image_base64: Optional[str] = None
    
    class ClassifyResponse(BaseModel):
        success: bool
        modality: Optional[str] = None
        confidence: Optional[float] = None
        probabilities: Optional[dict] = None
        error: Optional[str] = None
    
    @fastapi_app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "resnet-modality-classifier"}
    
    @fastapi_app.post("/classify", response_model=ClassifyResponse)
    async def classify_endpoint(request: ClassifyRequest):
        """
        Classify image modality.
        
        Accepts either image_url or image_base64.
        """
        try:
            # Get image bytes
            image_bytes = None
            
            # Try image_base64 first, but validate it's not a placeholder
            if request.image_base64 and request.image_base64.strip():
                # Skip common placeholder values
                if request.image_base64.lower() not in ["string", "none", "null", ""]:
                    try:
                        # Decode base64
                        if "," in request.image_base64:
                            # Data URI format
                            _, b64_data = request.image_base64.split(",", 1)
                        else:
                            b64_data = request.image_base64
                        image_bytes = base64.b64decode(b64_data)
                    except Exception:
                        # If base64 decode fails, ignore and fall through to image_url
                        image_bytes = None
            
            # Fall back to image_url if image_base64 wasn't valid or wasn't provided
            if image_bytes is None:
                if request.image_url:
                    # Download from URL
                    response = await http_client.get(request.image_url)
                    response.raise_for_status()
                    image_bytes = response.content
                else:
                    return ClassifyResponse(
                        success=False,
                        error="Either image_url or image_base64 must be provided"
                    )
            
            # Classify
            result = classify_image.local(image_bytes)
            
            return ClassifyResponse(
                success=True,
                modality=result["modality"],
                confidence=result["confidence"],
                probabilities=result["probabilities"]
            )
        
        except httpx.HTTPError as e:
            return ClassifyResponse(
                success=False,
                error=f"Failed to download image: {str(e)}"
            )
        except Exception as e:
            return ClassifyResponse(
                success=False,
                error=str(e)
            )
    
    @fastapi_app.on_event("shutdown")
    async def shutdown():
        await http_client.aclose()
    
    return fastapi_app


# CLI for local testing
if __name__ == "__main__":
    # Deploy command: modal deploy resnet_classifier.py
    # Test locally: modal run resnet_classifier.py
    print("ResNet Modality Classifier")
    print("Deploy with: modal deploy resnet_classifier.py")

