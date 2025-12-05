# app/services/classifier.py
import httpx
import base64
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

async def predict_modality(image_bytes: bytes) -> str:
    """
    Sends image bytes to ResNet API and returns the modality string (IR, SAR, RGB).
    """
    # 1. Encode bytes to Base64 String
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    payload = {
        "image_base64": base64_image
    }

    async with httpx.AsyncClient() as client:
        try:
            # 2. Call External ResNet API
            response = await client.post(
                f'{settings.RESNET_URL}/classify', 
                json=payload, 
                timeout=10.0 # Safety timeout
            )
            response.raise_for_status()
            
            # 3. Parse Response
            data = response.json()
            
            if not data.get("success"):
                logger.error(f"Classifier reported failure: {data.get('error')}")
                raise ValueError("Classification failed")

            # Extract modality (e.g., 'rgb') and normalize to uppercase ('RGB')
            modality = data.get("modality", "RGB").upper()
            return modality

        except httpx.RequestError as e:
            logger.error(f"Network error communicating with ResNet: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected classification error: {e}")
            # Fallback or re-raise depending on strictness. 
            # We re-raise to inform the user.
            raise e