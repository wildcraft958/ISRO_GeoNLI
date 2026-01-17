import traceback
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from sam3_app.api.schemas import (
    SAM3CountRequest, SAM3CountResponse,
    SAM3AreaRequest, SAM3AreaResponse,
    SAM3SegmentRequest, SAM3SegmentResponse
)
from sam3_app.core.instances import get_model
from sam3_app.core.model import SAM3Model

router = APIRouter()

def get_image_bytes(image_url: str) -> bytes:
    import httpx
    import base64
    
    if not image_url:
        raise HTTPException(status_code=400, detail="'image_url' is required")
    
    if image_url.startswith('data:'):
        try:
            if ',' in image_url:
                base64_part = image_url.split(',', 1)[1]
            else:
                raise HTTPException(status_code=400, detail="Invalid data URI format")
            return base64.b64decode(base64_part)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")
            
    try:
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            resp = client.get(image_url)
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download image: {resp.status_code}")
            return resp.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")

@router.get("/health")
async def health():
    return {"status": "ok", "service": "sam3-agent"}

@router.post("/sam3/count", response_model=SAM3CountResponse)
async def count_objects(request: SAM3CountRequest, model: SAM3Model = Depends(get_model)):
    try:
        image_bytes = get_image_bytes(request.image_url)
        
        result = model.count(
            image_bytes=image_bytes,
            text_prompt=request.prompt,
            llm_config=request.llm_config.model_dump(),
            confidence_threshold=request.confidence_threshold,
            pyramidal_config=request.pyramidal_config.model_dump() if request.pyramidal_config else None,
            max_retries=request.max_retries,
            include_obb=request.include_obb,
            obb_as_polygon=request.obb_as_polygon
        )
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e), "traceback": traceback.format_exc()}
        )

@router.post("/sam3/area", response_model=SAM3AreaResponse)
async def calculate_area(request: SAM3AreaRequest, model: SAM3Model = Depends(get_model)):
    try:
        image_bytes = get_image_bytes(request.image_url)
        
        result = model.area(
            image_bytes=image_bytes,
            text_prompt=request.prompt,
            llm_config=request.llm_config.model_dump(),
            gsd=request.gsd,
            confidence_threshold=request.confidence_threshold,
            pyramidal_config=request.pyramidal_config.model_dump() if request.pyramidal_config else None,
            max_retries=request.max_retries,
            include_obb=request.include_obb,
            obb_as_polygon=request.obb_as_polygon
        )
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e), "traceback": traceback.format_exc()}
        )

@router.post("/sam3/segment", response_model=SAM3SegmentResponse)
async def segment_image(request: SAM3SegmentRequest, model: SAM3Model = Depends(get_model)):
    try:
        image_bytes = get_image_bytes(request.image_url)
        
        result = model.segment(
            image_bytes=image_bytes,
            prompt=request.prompt,
            llm_config=request.llm_config.model_dump(),
            debug=request.debug,
            confidence_threshold=request.confidence_threshold,
            pyramidal_config=request.pyramidal_config.model_dump() if request.pyramidal_config else None,
            include_obb=request.include_obb,
            obb_as_polygon=request.obb_as_polygon
        )
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e), "traceback": traceback.format_exc()}
        )
