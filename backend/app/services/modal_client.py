import os
import requests
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings


class ModalServiceClient:
    """Client for calling Modal-deployed VLM services with retry logic"""
    
    def __init__(self):
        self.base_url = os.getenv("MODAL_BASE_URL", getattr(settings, "MODAL_BASE_URL", "https://default.modal.run"))
        self.timeout = 120
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8)
    )
    def call_grounding(self, image_url: str, query: str) -> dict:
        """Call grounding service. Returns {bboxes: [...]}"""
        try:
            response = requests.post(
                f"{self.base_url}/grounding",
                json={"image_url": image_url, "query": query},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Grounding service failed: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8)
    )
    def call_vqa(self, image_url: str, query: str) -> dict:
        """Call VQA service. Returns {answer: str, confidence: float}"""
        try:
            response = requests.post(
                f"{self.base_url}/vqa",
                json={"image_url": image_url, "query": query},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"VQA service failed: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8)
    )
    def call_captioning(self, image_url: str) -> dict:
        """Call captioning service. Returns {caption: str}"""
        try:
            response = requests.post(
                f"{self.base_url}/captioning",
                json={"image_url": image_url},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Captioning service failed: {e}")

