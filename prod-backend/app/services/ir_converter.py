import base64
import httpx
from app.core.config import settings

async def convert_ir_to_rgb(image_bytes: bytes) -> bytes:
    """
    Converts an IR image to RGB by calling the external Modal service (IR_CONVERT_URL).
    Accepts bytes, sends base64 to API, and returns decoded RGB bytes.
    """
    # 1. Encode image bytes to Base64 string for JSON payload
    input_b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    payload = {
        "image_base64": input_b64
    }

    # 2. Call the external IR conversion service
    async with httpx.AsyncClient(timeout=60.0) as client:
        # We assume settings.IR_CONVERT_URL is defined in your config
        response = await client.post(settings.IR_CONVERT_URL, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        # 3. Extract the converted base64 image
        # Expecting format: { "image_base64": "..." }
        output_b64 = data.get("image_base64")
        
        if not output_b64:
            raise ValueError("IR Conversion service returned empty data")
            
        # 4. Decode back to bytes
        return base64.b64decode(output_b64)