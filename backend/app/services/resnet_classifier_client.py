"""
HTTP client for the Modal-deployed ResNet modality classifier service.

This client provides methods to classify images using the ResNet50 model
deployed on Modal. It's used as a fallback when statistical modality
detection is uncertain, especially for SAR images.
"""

import logging
import os
import threading
from typing import Any, Dict, Literal, Optional, Tuple

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)

# Type alias for modality
Modality = Literal["rgb", "infrared", "sar", "unknown"]


class ResNetClassifierError(Exception):
    """Raised when ResNet classification fails."""
    pass


class ResNetClassifierClient:
    """
    HTTP client for the Modal ResNet modality classifier service.
    
    Thread-safe singleton implementation.
    
    Usage:
        client = get_resnet_classifier_client()
        modality, confidence = client.classify_from_url(image_url)
    """
    
    _instance: Optional["ResNetClassifierClient"] = None
    _lock = threading.Lock()
    
    # Default Modal endpoint URL - update after deployment
    DEFAULT_ENDPOINT = "https://maximuspookus--resnet-modality-classifier-serve.modal.run"
    
    def __new__(cls) -> "ResNetClassifierClient":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        
        # Get endpoint URL from environment or settings
        self.base_url = os.getenv(
            "RESNET_CLASSIFIER_URL",
            getattr(settings, "RESNET_CLASSIFIER_URL", self.DEFAULT_ENDPOINT)
        )
        self.timeout = 60  # 60 second timeout
        self._initialized = True
        
        logger.info(f"ResNetClassifierClient initialized with endpoint: {self.base_url}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8)
    )
    def classify_from_url(
        self, 
        image_url: str
    ) -> Tuple[Modality, float, Dict[str, float]]:
        """
        Classify image modality from URL using ResNet.
        
        Args:
            image_url: URL of the image to classify (HTTP/HTTPS or data URI)
            
        Returns:
            Tuple of (modality, confidence, probabilities_dict)
            
        Raises:
            ResNetClassifierError: If classification fails after retries
        """
        try:
            # Determine if it's a data URI or regular URL
            if image_url.startswith("data:"):
                payload = {"image_base64": image_url}
            else:
                payload = {"image_url": image_url}
            
            response = requests.post(
                f"{self.base_url}/classify",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error")
                raise ResNetClassifierError(f"Classification failed: {error_msg}")
            
            modality = result.get("modality", "unknown")
            confidence = result.get("confidence", 0.0)
            probabilities = result.get("probabilities", {})
            
            logger.debug(
                f"ResNet classification: modality={modality}, "
                f"confidence={confidence:.3f}"
            )
            
            return modality, confidence, probabilities
        
        except requests.RequestException as e:
            logger.error(f"ResNet classifier request failed: {e}")
            raise ResNetClassifierError(f"Request failed: {e}") from e
        except Exception as e:
            logger.error(f"ResNet classifier error: {e}")
            raise ResNetClassifierError(str(e)) from e
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4)
    )
    def health_check(self) -> bool:
        """
        Check if the ResNet classifier service is healthy.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=10
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def classify_with_fallback(
        self,
        image_url: str,
        statistical_modality: Optional[str] = None,
        statistical_confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Classify image with smart fallback logic.
        
        If statistical detection already has high confidence for RGB,
        skip ResNet. Otherwise, use ResNet and merge results.
        
        Args:
            image_url: URL of the image
            statistical_modality: Result from statistical detection
            statistical_confidence: Confidence from statistical detection
            
        Returns:
            Dict with final_modality, confidence, source, and details
        """
        # If statistical detection is confident about RGB, trust it
        if (statistical_modality == "rgb" and 
            statistical_confidence is not None and 
            statistical_confidence > 0.8):
            return {
                "final_modality": "rgb",
                "confidence": statistical_confidence,
                "source": "statistical",
                "resnet_used": False,
                "details": {"reason": "high_confidence_rgb_from_statistical"}
            }
        
        # Call ResNet for SAR or uncertain cases
        try:
            modality, confidence, probabilities = self.classify_from_url(image_url)
            
            return {
                "final_modality": modality,
                "confidence": confidence,
                "source": "resnet",
                "resnet_used": True,
                "probabilities": probabilities,
                "details": {
                    "statistical_modality": statistical_modality,
                    "statistical_confidence": statistical_confidence,
                }
            }
        
        except ResNetClassifierError as e:
            # Fallback to statistical result if ResNet fails
            logger.warning(f"ResNet classification failed, using statistical: {e}")
            return {
                "final_modality": statistical_modality or "unknown",
                "confidence": statistical_confidence or 0.0,
                "source": "statistical_fallback",
                "resnet_used": False,
                "error": str(e),
                "details": {"reason": "resnet_failed"}
            }


# =============================================================================
# Global Singleton Accessor
# =============================================================================

_resnet_client: Optional[ResNetClassifierClient] = None


def get_resnet_classifier_client() -> ResNetClassifierClient:
    """
    Get or create the global ResNetClassifierClient instance.
    
    Returns:
        ResNetClassifierClient singleton instance
    """
    global _resnet_client
    
    if _resnet_client is None:
        _resnet_client = ResNetClassifierClient()
    
    return _resnet_client


def is_resnet_classifier_available() -> bool:
    """
    Check if ResNet classifier service is available.
    
    Returns:
        True if service is reachable and healthy
    """
    try:
        client = get_resnet_classifier_client()
        return client.health_check()
    except Exception:
        return False

