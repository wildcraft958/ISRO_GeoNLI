"""
Modality Router Service for automatic image modality detection.

This service classifies images into modalities:
- 'rgb': Standard RGB images
- 'infrared': Multispectral/NIR images requiring IR2RGB conversion
- 'sar': Synthetic Aperture Radar images
- 'unknown': Unclassified images

The router uses metadata analysis, channel count, and statistical features
to determine the most likely modality.
"""

import base64
import logging
import math
import threading
from io import BytesIO
from typing import Any, Dict, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import requests
from PIL import Image
from scipy.stats import kurtosis
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Type alias for modality
Modality = Literal["rgb", "infrared", "sar", "unknown"]


class ImageLoadError(Exception):
    """Raised when image loading fails."""
    pass


class ModalityDetectionError(Exception):
    """Raised when modality detection fails."""
    pass


class ImageLoader:
    """Handle image loading and conversion from various input types.
    
    Supports loading from:
    - Local file paths
    - URLs (http/https)
    - Data URIs (base64 encoded)
    - PIL Images
    - NumPy arrays
    """
    
    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8)
    )
    def download_image(url: str, timeout: int = 30) -> Image.Image:
        """Download image from URL with retry logic.
        
        Args:
            url: HTTP/HTTPS URL to download from.
            timeout: Request timeout in seconds.
            
        Returns:
            PIL Image object.
            
        Raises:
            ImageLoadError: If download fails after retries.
        """
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except requests.RequestException as e:
            logger.error(f"Failed to download image from {url}: {e}")
            raise ImageLoadError(f"Failed to download image: {e}") from e
    
    @staticmethod
    def load_from_data_uri(data_uri: str) -> Image.Image:
        """Load image from base64 data URI.
        
        Args:
            data_uri: Data URI string (e.g., "data:image/png;base64,...")
            
        Returns:
            PIL Image object.
            
        Raises:
            ImageLoadError: If parsing or decoding fails.
        """
        try:
            # Parse data URI
            if "," not in data_uri:
                raise ValueError("Invalid data URI format")
            
            header, b64_data = data_uri.split(",", 1)
            image_bytes = base64.b64decode(b64_data)
            return Image.open(BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Failed to load image from data URI: {e}")
            raise ImageLoadError(f"Failed to load data URI: {e}") from e
    
    @staticmethod
    def load_image_arr(
        path_or_arr: Union[str, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """Load image and return as numpy array.
        
        Preserves original dtype and channels.
        
        Args:
            path_or_arr: File path, PIL Image, or numpy array.
            
        Returns:
            NumPy array of the image.
            
        Raises:
            ImageLoadError: If loading fails.
        """
        try:
            if isinstance(path_or_arr, str):
                # Try OpenCV first (preserves all channels)
                arr = cv2.imread(path_or_arr, -1)
                if arr is None:
                    # Fallback to PIL
                    arr = np.array(Image.open(path_or_arr))
                return arr
            
            if isinstance(path_or_arr, Image.Image):
                return np.array(path_or_arr)
            
            if isinstance(path_or_arr, np.ndarray):
                return path_or_arr
            
            raise ValueError(f"Unsupported image input type: {type(path_or_arr)}")
        except Exception as e:
            logger.error(f"Failed to load image array: {e}")
            raise ImageLoadError(f"Failed to load image: {e}") from e
    
    @staticmethod
    def get_image_metadata(
        path_or_pil: Union[str, Image.Image]
    ) -> Dict[str, Any]:
        """Extract metadata from image.
        
        Args:
            path_or_pil: File path or PIL Image.
            
        Returns:
            Dictionary containing available metadata.
        """
        meta: Dict[str, Any] = {}
        try:
            if isinstance(path_or_pil, Image.Image):
                img = path_or_pil
            else:
                img = Image.open(path_or_pil)
            
            # Get PIL info dict
            if hasattr(img, "info") and img.info:
                meta.update(dict(img.info))
            
            # Try to get TIFF tags
            try:
                tags = getattr(img, "tag_v2", None)
                if tags is not None:
                    for k, v in tags.items():
                        meta[str(k)] = v
            except Exception:
                pass
            
            meta["pil_mode"] = img.mode
            meta["size"] = img.size
            
        except Exception as e:
            logger.debug(f"Could not extract metadata: {e}")
        
        return meta
    
    @staticmethod
    def rgba_to_rgb_pil(
        pil_img: Image.Image, 
        background: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        """Composite RGBA PIL image over background and return RGB PIL image.
        
        Args:
            pil_img: PIL Image (RGBA, LA, or other mode).
            background: RGB background color tuple.
            
        Returns:
            RGB PIL Image.
        """
        if pil_img.mode == "RGBA":
            bg = Image.new("RGB", pil_img.size, background)
            bg.paste(pil_img, mask=pil_img.split()[3])
            return bg
        
        if pil_img.mode == "LA":
            rgba = pil_img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, background)
            bg.paste(rgba, mask=rgba.split()[3])
            return bg
        
        return pil_img.convert("RGB")


class ImageStatsCalculator:
    """Calculate statistical features for images.
    
    Used for SAR detection based on speckle noise characteristics.
    """
    
    @staticmethod
    def collapse_to_single_band(arr: np.ndarray) -> np.ndarray:
        """Return single-band float32 image.
        
        If multi-channel, computes mean across channels.
        
        Args:
            arr: Input image array [H, W] or [H, W, C].
            
        Returns:
            Single-band float32 array [H, W].
        """
        if arr.ndim == 3:
            return arr.mean(axis=2).astype(np.float32)
        return arr.astype(np.float32)
    
    @staticmethod
    def local_speckle_measure(arr: np.ndarray, ksize: int = 7) -> float:
        """Compute mean local coefficient-of-variation (speckle measure).
        
        SAR images typically have high local CV due to speckle noise.
        
        Args:
            arr: Input single-band array [H, W].
            ksize: Kernel size for local statistics.
            
        Returns:
            Mean local coefficient of variation.
        """
        a = arr.astype(np.float32)
        if a.size == 0:
            return 0.0
        
        # Compute local mean and variance using box filters
        mean = cv2.boxFilter(a, ddepth=-1, ksize=(ksize, ksize), normalize=True)
        sq = cv2.boxFilter(a * a, ddepth=-1, ksize=(ksize, ksize), normalize=True)
        var = sq - mean * mean
        std = np.sqrt(np.clip(var, 0, None))
        
        # Local coefficient of variation
        lcv = std / (np.abs(mean) + 1e-9)
        return float(np.nanmean(lcv))


class SARDetector:
    """Detect SAR imagery using threshold-based voting.
    
    Uses three statistical features:
    - Global coefficient of variation (CV)
    - Local coefficient of variation (speckle measure)
    - Kurtosis (heavy-tailed distribution indicator)
    
    Requires at least 2 out of 3 features to exceed thresholds for SAR classification.
    """
    
    DEFAULT_THRESHOLDS: Dict[str, float] = {
        "cv": 0.32624886285281385,
        "local_cv": 0.12780852243304253,
        "kurtosis": 3.8785458505153656
    }
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """Initialize SAR detector.
        
        Args:
            thresholds: Custom thresholds dict. Uses defaults if None.
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
        self.stats_calculator = ImageStatsCalculator()
    
    def is_probably_sar_from_thresholds(
        self, 
        arr: np.ndarray, 
        verbose: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """Compute stats and determine if image is likely SAR.
        
        Uses voting: requires >= 2 features above threshold.
        
        Args:
            arr: Input image array.
            verbose: If True, log diagnostic info.
            
        Returns:
            Tuple of (is_sar, diagnostics_dict).
        """
        band = self.stats_calculator.collapse_to_single_band(arr)
        
        # Compute statistics
        mn, mx = float(band.min()), float(band.max())
        mean = float(band.mean())
        std = float(band.std())
        cv = std / (abs(mean) + 1e-9)
        
        try:
            k = float(kurtosis(band.ravel(), fisher=False, nan_policy="omit"))
        except Exception:
            k = float("nan")
        
        lcv = self.stats_calculator.local_speckle_measure(band, ksize=7)
        
        # Voting
        score = 0
        reasons = []
        
        if not math.isnan(cv) and self.thresholds.get("cv") is not None:
            if cv >= self.thresholds["cv"]:
                score += 1
                reasons.append(f"cv={cv:.3f}>={self.thresholds['cv']:.3f}")
        
        if not math.isnan(lcv) and self.thresholds.get("local_cv") is not None:
            if lcv >= self.thresholds["local_cv"]:
                score += 1
                reasons.append(f"local_cv={lcv:.3f}>={self.thresholds['local_cv']:.3f}")
        
        if not math.isnan(k) and self.thresholds.get("kurtosis") is not None:
            if k >= self.thresholds["kurtosis"]:
                score += 1
                reasons.append(f"kurtosis={k:.3f}>={self.thresholds['kurtosis']:.3f}")
        
        is_sar = score >= 2
        
        diag = {
            "cv": cv,
            "local_cv": lcv,
            "kurtosis": k,
            "score": score,
            "reasons": reasons,
            "mean": mean,
            "min": mn,
            "max": mx
        }
        
        if verbose:
            logger.debug(f"SAR detection diagnostics: {diag}")
        
        return is_sar, diag


class AlphaDetector:
    """Detect alpha-like channels in RGBA images.
    
    Distinguishes true alpha channels from 4th spectral bands.
    """
    
    @staticmethod
    def alpha_like(
        alpha_arr: Optional[np.ndarray], 
        frac_threshold: float = 0.4, 
        uniq_threshold: int = 20
    ) -> bool:
        """Check if array looks like an alpha channel.
        
        Heuristic: many pixels exactly 0 or 255 OR few unique values.
        
        Args:
            alpha_arr: The 4th channel array.
            frac_threshold: Fraction of 0/255 values to consider alpha.
            uniq_threshold: Max unique values to consider alpha.
            
        Returns:
            True if array appears to be an alpha channel.
        """
        if alpha_arr is None:
            return False
        
        flat = alpha_arr.ravel()
        if flat.size == 0:
            return False
        
        # Floating point arrays are unlikely to be alpha
        if np.issubdtype(flat.dtype, np.floating):
            return False
        
        zeros = (flat == 0).sum()
        maxs = (flat == 255).sum()
        frac_end = (zeros + maxs) / (flat.size + 1e-12)
        uniq = np.unique(flat)
        
        # Convert to Python bool for consistency
        return bool((frac_end > frac_threshold) or (uniq.size < uniq_threshold))


class MetadataParser:
    """Parse image metadata for modality hints."""
    
    @staticmethod
    def parse_modality_from_metadata(meta: Dict[str, Any]) -> Optional[Modality]:
        """Extract modality from metadata if present.
        
        Args:
            meta: Metadata dictionary.
            
        Returns:
            Modality string or None if not found.
        """
        if not meta:
            return None
        
        # Check common metadata keys
        for key in ("modality", "MODALITY", "sensor", "SENSOR"):
            v = meta.get(key) or meta.get(key.lower())
            if isinstance(v, str):
                low = v.lower()
                if "sar" in low or "radar" in low:
                    return "sar"
                if any(k in low for k in ("nir", "infra", "ir", "multispectral", "multispec")):
                    return "infrared"
                if any(k in low for k in ("rgb", "truecolor", "vis", "visual")):
                    return "rgb"
        
        return None
    
    @staticmethod
    def parse_bands_from_metadata(meta: Dict[str, Any]) -> Optional[int]:
        """Extract band count from metadata if present.
        
        Args:
            meta: Metadata dictionary.
            
        Returns:
            Number of bands or None if not found.
        """
        if not meta:
            return None
        
        for key in ("bands", "Bands", "SAMPLESPERPIXEL", "samples_per_pixel", "samples"):
            val = meta.get(key) or meta.get(key.lower()) or meta.get(str(key))
            try:
                if val is not None:
                    return int(val)
            except (ValueError, TypeError):
                pass
        
        return None


class ModalityRouterService:
    """Production-ready service for image modality detection.
    
    Thread-safe singleton implementation with URL support.
    
    Usage:
        service = get_modality_router_service()
        modality, diagnostics = service.detect_modality_from_url(image_url)
    """
    
    _instance: Optional["ModalityRouterService"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "ModalityRouterService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        if getattr(self, "_initialized", False):
            return
        
        self.image_loader = ImageLoader()
        self.sar_detector = SARDetector(thresholds)
        self.alpha_detector = AlphaDetector()
        self.metadata_parser = MetadataParser()
        self._initialized = True
        
        logger.info("ModalityRouterService initialized")
    
    def detect_modality_from_url(
        self,
        image_url: str,
        metadata_priority: bool = True
    ) -> Tuple[Modality, Dict[str, Any]]:
        """Detect modality from image URL.
        
        Supports HTTP/HTTPS URLs and data URIs.
        
        Args:
            image_url: URL or data URI of the image.
            metadata_priority: Check metadata before pixel analysis.
            
        Returns:
            Tuple of (modality, diagnostics_dict).
        """
        try:
            # Load image from URL
            if image_url.startswith("data:"):
                pil_img = self.image_loader.load_from_data_uri(image_url)
            else:
                pil_img = self.image_loader.download_image(image_url)
            
            # Route the image
            modality, _, diagnostics = self.route_image(
                pil_img,
                metadata_priority=metadata_priority
            )
            
            return modality, diagnostics
            
        except ImageLoadError as e:
            logger.warning(f"Failed to load image for modality detection: {e}")
            return "unknown", {"error": str(e), "reason": "load_failed"}
        except Exception as e:
            logger.error(f"Unexpected error in modality detection: {e}", exc_info=True)
            return "unknown", {"error": str(e), "reason": "detection_failed"}
    
    def route_image(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        metadata_priority: bool = True,
        force_sar_folder: Optional[str] = None
    ) -> Tuple[Modality, Optional[Image.Image], Dict[str, Any]]:
        """Route image to appropriate modality.
        
        Args:
            image_input: Image as path, PIL Image, or numpy array.
            metadata_priority: Check metadata before pixel analysis.
            force_sar_folder: If path contains this string, force SAR.
            
        Returns:
            Tuple of (modality, rgb_pil_if_applicable, diagnostics).
        """
        # Extract metadata
        meta = self._get_metadata(image_input)
        
        # Check metadata first if priority enabled
        if metadata_priority and meta:
            result = self._route_by_metadata(image_input, meta)
            if result is not None:
                return result
        
        # Check force_sar_folder
        if (isinstance(image_input, str) and 
            force_sar_folder and 
            force_sar_folder in image_input):
            return "sar", None, {"reason": "forced_by_folder"}
        
        # Load array
        try:
            arr = self.image_loader.load_image_arr(image_input)
        except ImageLoadError:
            return "unknown", None, {"error": "load_failed"}
        
        if arr is None:
            return "unknown", None, {"error": "load_failed"}
        
        channels = 1 if arr.ndim == 2 else arr.shape[2]
        dtype = arr.dtype
        
        # Route by channels
        if channels >= 4:
            return self._route_multi_channel(arr, image_input, channels, dtype, meta)
        
        if channels == 3:
            return self._route_three_channel(arr, image_input, meta)
        
        if channels == 1:
            return self._route_single_channel(arr, meta)
        
        return "unknown", None, {"reason": "no_rule_matched", "metadata": meta}
    
    def _get_metadata(
        self, 
        image_input: Union[str, Image.Image, np.ndarray]
    ) -> Dict[str, Any]:
        """Extract metadata from image input."""
        try:
            if isinstance(image_input, (str, Image.Image)):
                return self.image_loader.get_image_metadata(image_input)
        except Exception as e:
            logger.debug(f"Could not extract metadata: {e}")
        return {}
    
    def _route_by_metadata(
        self, 
        image_input: Union[str, Image.Image, np.ndarray], 
        meta: Dict[str, Any]
    ) -> Optional[Tuple[Modality, Optional[Image.Image], Dict[str, Any]]]:
        """Route based on metadata if possible."""
        modality = self.metadata_parser.parse_modality_from_metadata(meta)
        if modality:
            if modality == "rgb":
                try:
                    pil = (Image.open(image_input) 
                           if isinstance(image_input, str) 
                           else image_input)
                    if isinstance(pil, Image.Image):
                        return "rgb", pil.convert("RGB"), {
                            "reason": "metadata_modality", 
                            "metadata": meta
                        }
                except Exception:
                    return "rgb", None, {
                        "reason": "metadata_modality_failed_to_load_pil", 
                        "metadata": meta
                    }
            return modality, None, {"reason": "metadata_modality", "metadata": meta}
        
        bands = self.metadata_parser.parse_bands_from_metadata(meta)
        if bands is not None:
            if bands >= 4:
                return "infrared", None, {
                    "reason": "metadata_bands_count", 
                    "bands": bands, 
                    "metadata": meta
                }
            if bands == 3:
                try:
                    pil = (Image.open(image_input) 
                           if isinstance(image_input, str) 
                           else image_input)
                    if isinstance(pil, Image.Image):
                        return "rgb", pil.convert("RGB"), {
                            "reason": "metadata_bands_count", 
                            "bands": bands, 
                            "metadata": meta
                        }
                except Exception:
                    pass
        
        return None
    
    def _route_multi_channel(
        self, 
        arr: np.ndarray, 
        image_input: Union[str, Image.Image, np.ndarray], 
        channels: int, 
        dtype: np.dtype, 
        meta: Dict[str, Any]
    ) -> Tuple[Modality, Optional[Image.Image], Dict[str, Any]]:
        """Route multi-channel (>=4) images."""
        alpha = arr[..., 3] if arr.ndim == 3 and arr.shape[2] == 4 else None
        
        if alpha is not None and self.alpha_detector.alpha_like(alpha):
            try:
                loaded_cv2 = (cv2.imread(image_input, -1) 
                              if isinstance(image_input, str) 
                              else None)
                if loaded_cv2 is not None:
                    pil = Image.fromarray(arr[..., :3][..., ::-1])
                else:
                    pil = Image.fromarray(arr[..., :3])
            except Exception:
                pil = Image.fromarray(arr[..., :3])
            
            rgb_pil = self.image_loader.rgba_to_rgb_pil(pil)
            return "rgb", rgb_pil, {
                "reason": "rgba_alpha_detected", 
                "channels": channels, 
                "dtype": str(dtype), 
                "metadata": meta
            }
        
        return "infrared", None, {
            "reason": "channels>=4_not_alpha", 
            "channels": channels, 
            "dtype": str(dtype), 
            "metadata": meta
        }
    
    def _route_three_channel(
        self, 
        arr: np.ndarray, 
        image_input: Union[str, Image.Image, np.ndarray], 
        meta: Dict[str, Any]
    ) -> Tuple[Modality, Optional[Image.Image], Dict[str, Any]]:
        """Route 3-channel images."""
        is_sar, diag = self.sar_detector.is_probably_sar_from_thresholds(arr)
        
        if is_sar:
            return "sar", None, {"reason": "3ch_sar_detected", **diag, "metadata": meta}
        
        try:
            loaded_cv2 = (cv2.imread(image_input, -1) 
                          if isinstance(image_input, str) 
                          else None)
            if loaded_cv2 is not None:
                pil_img = Image.fromarray(arr[..., ::-1])
            else:
                pil_img = Image.fromarray(arr)
        except Exception:
            pil_img = Image.fromarray(arr)
        
        return "rgb", pil_img.convert("RGB"), {
            "reason": "3ch_not_sar", 
            **diag, 
            "metadata": meta
        }
    
    def _route_single_channel(
        self, 
        arr: np.ndarray, 
        meta: Dict[str, Any]
    ) -> Tuple[Modality, Optional[Image.Image], Dict[str, Any]]:
        """Route single-channel images."""
        is_sar, diag = self.sar_detector.is_probably_sar_from_thresholds(arr)
        
        if is_sar:
            return "sar", None, {"reason": "1ch_sar_detected", **diag, "metadata": meta}
        
        return "infrared", None, {"reason": "1ch_not_sar", **diag, "metadata": meta}


# =============================================================================
# Global Singleton Accessor
# =============================================================================

_modality_router_service: Optional[ModalityRouterService] = None


def get_modality_router_service() -> ModalityRouterService:
    """Get or create the global ModalityRouterService instance.
    
    Returns:
        ModalityRouterService singleton instance.
    """
    global _modality_router_service
    
    if _modality_router_service is None:
        _modality_router_service = ModalityRouterService()
    
    return _modality_router_service


def is_modality_detection_available() -> bool:
    """Check if modality detection service is available.
    
    Returns:
        True if all dependencies are available.
    """
    try:
        # Check required dependencies
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        from scipy.stats import kurtosis  # noqa: F401
        return True
    except ImportError:
        return False

