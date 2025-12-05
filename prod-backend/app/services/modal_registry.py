from typing import Any, Dict, Optional, Tuple
from app.core.config import settings
# Import your logic handlers
from app.services.modalities.sar_handler import get_sar_payload
from app.services.modalities.ir_handler import get_ir_payload

# --- Base Interface ---
class BaseModelAdapter:
    """
    Every model must implement these three methods.
    This ensures the Orchestrator can treat them all the same way.
    """
    def get_url(self) -> str:
        raise NotImplementedError

    def construct_payload(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes the ChatState and returns the exact JSON payload for the API.
        """
        raise NotImplementedError

    def parse_response(self, data: Dict[str, Any]) -> Tuple[str, Optional[Any]]:
        """
        Takes the raw API JSON response and returns (text_for_chat, metadata).
        """
        raise NotImplementedError

# --- 1. Captioning Model ---
class CaptioningAdapter(BaseModelAdapter):
    def get_url(self):
        return settings.CAPTION_URL

    def construct_payload(self, state):
        # Specific Requirements: Hardcoded prompt, specific model name
        return {
            "model": "qwen-caption-special",
            "temperature": 0.7,
            "max_tokens": 512,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this satellite image."},
                    {"type": "image_url", "image_url": {"url": state["image_url"]}}
                ]
            }]
        }

    def parse_response(self, data):
        # Standard OpenAI format
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"], None
        return data.get("response", "No caption generated."), None

# --- 2. VQA General Model ---
class VQAGeneralAdapter(BaseModelAdapter):
    def get_url(self):
        return settings.VQA_GENERAL_URL

    def construct_payload(self, state):
        full_prompt = self._build_prompt(state)
        return {
            "model": "qwen-vqa-special",
            "temperature": 0.7,
            "max_tokens": 512,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": state["image_url"]}}
                ]
            }]
        }

    def parse_response(self, data):
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"], None
        return data.get("response", "No response."), None

    def _build_prompt(self, state):
        header = f"Image Type: {state['image_type']}\n"
        if state["summary_context"]:
            header += f"History: {state['summary_context']}\n"
        return f"{header}\nQuestion: {state['query_text']}"

# --- 3. VQA Binary Model (Yes/No) ---
class VQABinaryAdapter(VQAGeneralAdapter):
    def get_url(self):
        return settings.VQA_BINARY_URL

    def construct_payload(self, state):
        payload = super().construct_payload(state)
        payload["temperature"] = 0.1 # Lower temp for binary
        return payload

# --- 4. VQA Numerical Model (Counting) ---
class VQANumericalAdapter(VQAGeneralAdapter):
    def get_url(self):
        return settings.VQA_NUMERICAL_URL

    def construct_payload(self, state):
        payload = super().construct_payload(state)
        payload["temperature"] = 0.1 # Lower temp for counting
        return payload

# --- 5. Grounding Model ---
class GroundingAdapter(BaseModelAdapter):
    def get_url(self):
        return settings.GROUNDING_URL

    def construct_payload(self, state):
        # Grounding typically just needs the object name/query
        return {
            "model": "grounding-special",
            "max_tokens": 512,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": state["query_text"]}, # Just the query, no history usually
                    {"type": "image_url", "image_url": {"url": state["image_url"]}}
                ]
            }]
        }

    def parse_response(self, data):
        # We assume the response MIGHT have text, but DEFINITELY has metadata (boxes)
        text_out = "Here are the objects I located."
        if "choices" in data and data["choices"]:
             content = data["choices"][0]["message"]["content"]
             if content: text_out = content
        
        # Return the WHOLE data object as metadata so frontend can parse boxes
        return text_out, data

# --- 6. SAR Direct Model ---
class SARAdapter(BaseModelAdapter):
    def get_url(self):
        return settings.SAR_URL

    def construct_payload(self, state):
        # Delegate specific prompt logic to the handler
        return get_sar_payload(state["image_url"], state["query_text"])

    def parse_response(self, data):
        if "choices" in data:
            return data["choices"][0]["message"]["content"], None
        return data.get("response", "No SAR analysis."), None

# --- 7. IR Direct Model ---
class IRAdapter(BaseModelAdapter):
    def get_url(self):
        return settings.IR_URL

    def construct_payload(self, state):
        # Delegate specific prompt logic to the handler
        return get_ir_payload(state["image_url"], state["query_text"])

    def parse_response(self, data):
        if "choices" in data:
            return data["choices"][0]["message"]["content"], None
        return data.get("response", "No IR analysis."), None

# --- 8. FCC Direct Model (Optional) ---
class FCCAdapter(BaseModelAdapter):
    def get_url(self):
        return settings.FCC_URL

    def construct_payload(self, state):
        # Assuming FCC needs similar structure
        return {
            "model": "fcc-special",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": state["query_text"]},
                    {"type": "image_url", "image_url": {"url": state["image_url"]}}
                ]
            }]
        }

    def parse_response(self, data):
        if "choices" in data:
            return data["choices"][0]["message"]["content"], None
        return data.get("response", "No FCC analysis."), None

# --- FACTORY: The Selector ---
def get_model_adapter(mode: str, subtype: str = "GENERAL") -> BaseModelAdapter:
    """
    Returns the correct adapter class instance based on mode/subtype.
    """
    if mode == "CAPTIONING":
        return CaptioningAdapter()
    elif mode == "GROUNDING":
        return GroundingAdapter()
    elif mode == "SAR_DIRECT":
        return SARAdapter()
    elif mode == "IR_DIRECT":
        return IRAdapter()
    elif mode == "FCC_DIRECT":
        return FCCAdapter()
    elif mode == "VQA":
        if subtype == "BINARY":
            return VQABinaryAdapter()
        elif subtype == "NUMERICAL":
            return VQANumericalAdapter()
        else:
            return VQAGeneralAdapter()
    
    # Fallback
    return VQAGeneralAdapter()