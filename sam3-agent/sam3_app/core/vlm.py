import base64
import io
import json
import httpx
from typing import Optional, List, Dict, Any
from PIL import Image

class VLMInterface:
    """
    Interface for interacting with OpenAI-compatible VLM APIs.
    
    Wraps API calls with image support for prompt refinement,
    detection verification, and rephrasing.
    """
    
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.endpoint = f"{self.base_url}/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
    
    @staticmethod
    def encode_image_to_base64(image) -> str:
        """Encode PIL Image to base64 string"""
        buffered = io.BytesIO()
        if isinstance(image, Image.Image):
            # Convert to RGB if needed (JPEG doesn't support alpha)
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            image.save(buffered, format="JPEG")
        else:
            # Assume it's already bytes
            buffered.write(image)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def query(self, image, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> Optional[str]:
        """
        Query the VLM API with an image and text prompt.
        
        Args:
            image: PIL Image or image bytes
            prompt: Text prompt/question
            max_tokens: Maximum response length
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated text response or None on error
        """
        # Handle image input
        if isinstance(image, bytes):
            image_base64 = base64.b64encode(image).decode('utf-8')
        else:
            image_base64 = self.encode_image_to_base64(image)
        
        # OpenAI-compatible message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    }
                ]
            }
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": min(max_tokens, 8192),
            "temperature": temperature
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.endpoint,
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        if not isinstance(result, dict):
                            print(f"❌ API Error: Invalid response format (expected dict)")
                            return None
                        
                        choices = result.get("choices", [])
                        if not choices:
                            print(f"❌ API Error: No choices in response")
                            return None
                        
                        message = choices[0].get("message", {})
                        content = message.get("content")
                        
                        if content is None:
                            print(f"❌ API Error: No content in message")
                            return None
                        
                        return content
                    except json.JSONDecodeError as e:
                        print(f"❌ API Error: Failed to parse response JSON: {e}")
                        return None
                else:
                    print(f"❌ API Error: {response.status_code}")
                    print(f"Response: {response.text[:1000]}")
                    return None
        
        except httpx.TimeoutException:
            print(f"❌ Request timeout after {self.timeout}s")
            return None
        except httpx.ConnectError as e:
            print(f"❌ Connection error: {e}")
            return None
        except Exception as e:
            print(f"❌ API Exception: {e}")
            return None
