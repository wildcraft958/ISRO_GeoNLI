"""
IR2RGB In-Memory Service for False Color Composite (FCC) to RGB conversion.

This service converts multispectral satellite images (with NIR band) to RGB
using pre-trained RPCC (Root Polynomial Color Correction) weights and LUT.

The model is loaded once and cached in memory for efficient processing.
"""

import logging
import threading
from io import BytesIO
from pathlib import Path
from typing import List, Literal, Optional, Union
import base64

import numpy as np
from PIL import Image
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class IR2RGBModel:
    """
    IR to RGB conversion model using RPCC weights and 3D LUT.
    
    Supports synthesizing R, G, or B channel from NIR + other channels.
    """
    
    def __init__(self, npz_path: str):
        """
        Load model weights from NPZ file.
        
        Args:
            npz_path: Path to .npz file containing rpcc_R, rpcc_G, lut_B arrays
        """
        npz = np.load(npz_path, allow_pickle=False)
        
        # Load RPCC weights for R and G channels
        self.rpcc = {
            "R": npz.get("rpcc_R").astype(np.float32) if "rpcc_R" in npz else None,
            "G": npz.get("rpcc_G").astype(np.float32) if "rpcc_G" in npz else None
        }
        
        # Load LUT for B channel
        self.lut = {
            "B": npz.get("lut_B").astype(np.float32) if "lut_B" in npz else None
        }
        
        # Infer LUT grid size if present
        self.grid_size = None
        if self.lut["B"] is not None:
            self.grid_size = self.lut["B"].shape[0]
        
        logger.info(f"IR2RGB model loaded from {npz_path}")
        logger.info(f"  RPCC R weights: {'loaded' if self.rpcc['R'] is not None else 'missing'}")
        logger.info(f"  RPCC G weights: {'loaded' if self.rpcc['G'] is not None else 'missing'}")
        logger.info(f"  LUT B: {'loaded' if self.lut['B'] is not None else 'missing'} (grid_size={self.grid_size})")
    
    @staticmethod
    def _load_img(img: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Load and normalize image to float32 [0,1] with shape (H, W, 3)."""
        if isinstance(img, (str, Path)):
            im = Image.open(str(img)).convert("RGB")
            arr = np.array(im).astype(np.float32) / 255.0
        elif isinstance(img, Image.Image):
            arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        else:
            arr = np.array(img).astype(np.float32)
            if arr.max() > 2.0:  # Likely 0-255 range
                arr = arr / 255.0
        
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Input must be 3-channel image (H, W, 3).")
        
        return arr
    
    @staticmethod
    def _map_channels(arr: np.ndarray, channels: List[str]) -> dict:
        """Map image channels to named bands."""
        if len(channels) != 3:
            raise ValueError("channels must be length 3 (one label per image channel).")
        
        ch_map = {}
        for i, name in enumerate(channels):
            ch_map[name.strip().upper()] = arr[..., i]
        
        return ch_map
    
    @staticmethod
    def _features_root_poly2(X: np.ndarray) -> np.ndarray:
        """
        Compute root polynomial features of degree 2.
        
        X shape: (N, 3) -> returns Psi (N, 10)
        Features: [x1, x2, x3, sqrt(x1*x2), sqrt(x1*x3), sqrt(x2*x3), 
                   sqrt(x1^2), sqrt(x2^2), sqrt(x3^2), 1]
        """
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        eps = 1e-8
        
        m4 = x1 * x2
        m5 = x1 * x3
        m6 = x2 * x3
        m7 = x1 ** 2
        m8 = x2 ** 2
        m9 = x3 ** 2
        
        return np.stack([
            x1, x2, x3,
            np.sqrt(np.abs(m4) + eps),
            np.sqrt(np.abs(m5) + eps),
            np.sqrt(np.abs(m6) + eps),
            np.sqrt(np.abs(m7) + eps),
            np.sqrt(np.abs(m8) + eps),
            np.sqrt(np.abs(m9) + eps),
            np.ones_like(x1)
        ], axis=1).astype(np.float32)
    
    @staticmethod
    def _apply_lut_scalar(X: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """Apply trilinear interpolation on 3D LUT."""
        gs = lut.shape[0]
        scale = gs - 1
        coords = np.clip(X * scale, 0, scale - 1e-6)
        
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        z0 = np.floor(z).astype(int)
        
        x1 = np.clip(x0 + 1, 0, scale).astype(int)
        y1 = np.clip(y0 + 1, 0, scale).astype(int)
        z1 = np.clip(z0 + 1, 0, scale).astype(int)
        
        dx = (x - x0).astype(np.float32)
        dy = (y - y0).astype(np.float32)
        dz = (z - z0).astype(np.float32)
        
        # Get corner values
        c000 = lut[x0, y0, z0]
        c001 = lut[x0, y0, z1]
        c010 = lut[x0, y1, z0]
        c011 = lut[x0, y1, z1]
        c100 = lut[x1, y0, z0]
        c101 = lut[x1, y0, z1]
        c110 = lut[x1, y1, z0]
        c111 = lut[x1, y1, z1]
        
        # Compute weights
        w000 = (1 - dx) * (1 - dy) * (1 - dz)
        w001 = (1 - dx) * (1 - dy) * dz
        w010 = (1 - dx) * dy * (1 - dz)
        w011 = (1 - dx) * dy * dz
        w100 = dx * (1 - dy) * (1 - dz)
        w101 = dx * (1 - dy) * dz
        w110 = dx * dy * (1 - dz)
        w111 = dx * dy * dz
        
        # Interpolate
        out = (w000 * c000 + w001 * c001 + w010 * c010 + w011 * c011 +
               w100 * c100 + w101 * c101 + w110 * c110 + w111 * c111)
        
        return out.astype(np.float32)
    
    @staticmethod
    def _to_pil(rgb_arr: np.ndarray) -> Image.Image:
        """Convert normalized float array to PIL Image."""
        rgb = np.clip(rgb_arr, 0.0, 1.0)
        return Image.fromarray((rgb * 255.0).astype(np.uint8))
    
    def synthesize_R(self, img: Union[str, Path, Image.Image, np.ndarray], 
                     channels: List[str]) -> Image.Image:
        """
        Synthesize R channel from NIR, G, B.
        
        Args:
            img: Input image (path, PIL Image, or numpy array)
            channels: Channel order of input, e.g., ["NIR", "G", "B"]
        
        Returns:
            PIL Image with synthesized RGB
        """
        arr = self._load_img(img)
        ch = self._map_channels(arr, channels)
        
        if not all(k in ch for k in ("NIR", "G", "B")):
            raise ValueError("synthesize_R requires channels to include NIR, G, B in some order.")
        
        nir, G, B = ch["NIR"], ch["G"], ch["B"]
        H, W = nir.shape
        
        X_flat = np.stack([nir.reshape(-1), G.reshape(-1), B.reshape(-1)], axis=1).astype(np.float32)
        
        w = self.rpcc.get("R")
        if w is None:
            raise RuntimeError("RPCC weights for R not loaded.")
        
        Psi = self._features_root_poly2(X_flat)
        y_hat = (Psi @ w).astype(np.float32)
        R_hat = y_hat.reshape(H, W)
        
        rgb_out = np.stack([R_hat, G, B], axis=-1)
        return self._to_pil(rgb_out)
    
    def synthesize_G(self, img: Union[str, Path, Image.Image, np.ndarray], 
                     channels: List[str]) -> Image.Image:
        """
        Synthesize G channel from NIR, R, B.
        
        Args:
            img: Input image (path, PIL Image, or numpy array)
            channels: Channel order of input, e.g., ["NIR", "R", "B"]
        
        Returns:
            PIL Image with synthesized RGB
        """
        arr = self._load_img(img)
        ch = self._map_channels(arr, channels)
        
        if not all(k in ch for k in ("NIR", "R", "B")):
            raise ValueError("synthesize_G requires channels to include NIR, R, B in some order.")
        
        nir, R, B = ch["NIR"], ch["R"], ch["B"]
        H, W = nir.shape
        
        X_flat = np.stack([nir.reshape(-1), R.reshape(-1), B.reshape(-1)], axis=1).astype(np.float32)
        
        w = self.rpcc.get("G")
        if w is None:
            raise RuntimeError("RPCC weights for G not loaded.")
        
        Psi = self._features_root_poly2(X_flat)
        y_hat = (Psi @ w).astype(np.float32)
        G_hat = y_hat.reshape(H, W)
        
        rgb_out = np.stack([R, G_hat, B], axis=-1)
        return self._to_pil(rgb_out)
    
    def synthesize_B(self, img: Union[str, Path, Image.Image, np.ndarray], 
                     channels: List[str]) -> Image.Image:
        """
        Synthesize B channel from NIR, R, G using 3D LUT.
        
        Args:
            img: Input image (path, PIL Image, or numpy array)
            channels: Channel order of input, e.g., ["NIR", "R", "G"]
        
        Returns:
            PIL Image with synthesized RGB
        """
        arr = self._load_img(img)
        ch = self._map_channels(arr, channels)
        
        if not all(k in ch for k in ("NIR", "R", "G")):
            raise ValueError("synthesize_B requires channels to include NIR, R, G in some order.")
        
        nir, R, G = ch["NIR"], ch["R"], ch["G"]
        H, W = nir.shape
        
        X_flat = np.stack([nir.reshape(-1), R.reshape(-1), G.reshape(-1)], axis=1).astype(np.float32)
        
        lut = self.lut.get("B")
        if lut is None:
            raise RuntimeError("LUT for B not loaded.")
        
        y_hat = self._apply_lut_scalar(X_flat, lut)
        B_hat = y_hat.reshape(H, W)
        
        rgb_out = np.stack([R, G, B_hat], axis=-1)
        return self._to_pil(rgb_out)


class IR2RGBService:
    """
    Production-ready IR2RGB service with singleton pattern and thread safety.
    
    Usage:
        service = get_ir2rgb_service()
        result = service.convert_from_url(image_url, channels, synthesize_channel)
    """
    
    _instance = None
    _lock = threading.Lock()
    _model: Optional[IR2RGBModel] = None
    _model_path: Optional[str] = None
    
    def __new__(cls, model_path: Optional[str] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_path: Optional[str] = None):
        if self._model is not None and model_path == self._model_path:
            return  # Already initialized with same path
        
        with self._lock:
            if self._model is not None and model_path == self._model_path:
                return
            
            if model_path is None:
                # Default path relative to project root
                model_path = str(
                    Path(__file__).parent.parent.parent.parent 
                    / "ir2rgb_models" 
                    / "model_weights" 
                    / "models_ir_rgb.npz"
                )
            
            if not Path(model_path).exists():
                raise FileNotFoundError(f"IR2RGB model weights not found at {model_path}")
            
            logger.info(f"Initializing IR2RGB service with model from {model_path}")
            self._model = IR2RGBModel(model_path)
            self._model_path = model_path
    
    @property
    def model(self) -> IR2RGBModel:
        """Get the underlying model instance."""
        if self._model is None:
            raise RuntimeError("IR2RGB model not initialized")
        return self._model
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8)
    )
    def _download_image(self, image_url: str) -> Image.Image:
        """Download image from URL with retry logic."""
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    
    def convert_from_url(
        self,
        image_url: str,
        channels: List[str],
        synthesize_channel: Literal["R", "G", "B"] = "B"
    ) -> dict:
        """
        Convert multispectral image from URL to RGB.
        
        Args:
            image_url: URL of input image
            channels: Channel order, e.g., ["NIR", "R", "G"]
            synthesize_channel: Which channel to synthesize
        
        Returns:
            dict with:
                - rgb_image_base64: Base64 encoded RGB image
                - rgb_image_url: Data URI for direct use
                - format: Image format (PNG)
                - success: True if conversion succeeded
        """
        try:
            # Download image
            img = self._download_image(image_url)
            
            # Convert
            rgb_img = self.convert_pil_image(img, channels, synthesize_channel)
            
            # Encode to base64
            buffer = BytesIO()
            rgb_img.save(buffer, format="PNG", optimize=True)
            img_bytes = buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode()
            
            return {
                "rgb_image_base64": img_base64,
                "rgb_image_url": f"data:image/png;base64,{img_base64}",
                "format": "PNG",
                "size_bytes": len(img_bytes),
                "dimensions": {"width": rgb_img.width, "height": rgb_img.height},
                "success": True
            }
        
        except Exception as e:
            logger.error(f"IR2RGB conversion failed: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    def convert_pil_image(
        self,
        img: Image.Image,
        channels: List[str],
        synthesize_channel: Literal["R", "G", "B"] = "B"
    ) -> Image.Image:
        """
        Convert PIL Image directly.
        
        Args:
            img: Input PIL Image
            channels: Channel order, e.g., ["NIR", "R", "G"]
            synthesize_channel: Which channel to synthesize
        
        Returns:
            PIL Image in RGB format
        """
        if synthesize_channel == "R":
            return self.model.synthesize_R(img, channels)
        elif synthesize_channel == "G":
            return self.model.synthesize_G(img, channels)
        else:  # B
            return self.model.synthesize_B(img, channels)
    
    def convert_numpy(
        self,
        arr: np.ndarray,
        channels: List[str],
        synthesize_channel: Literal["R", "G", "B"] = "B"
    ) -> np.ndarray:
        """
        Convert numpy array directly.
        
        Args:
            arr: Input numpy array (H, W, 3)
            channels: Channel order, e.g., ["NIR", "R", "G"]
            synthesize_channel: Which channel to synthesize
        
        Returns:
            Numpy array in RGB format (H, W, 3), uint8
        """
        pil_img = self.convert_pil_image(Image.fromarray(arr), channels, synthesize_channel)
        return np.array(pil_img)


# Global singleton accessor
_ir2rgb_service: Optional[IR2RGBService] = None


def get_ir2rgb_service(model_path: Optional[str] = None) -> IR2RGBService:
    """
    Get or create the global IR2RGB service instance.
    
    Args:
        model_path: Optional path to model weights. If not provided,
                   uses default path relative to project root.
    
    Returns:
        IR2RGBService singleton instance
    """
    global _ir2rgb_service
    
    if _ir2rgb_service is None:
        _ir2rgb_service = IR2RGBService(model_path)
    
    return _ir2rgb_service


def is_ir2rgb_available() -> bool:
    """Check if IR2RGB service is available (model weights exist)."""
    default_path = (
        Path(__file__).parent.parent.parent.parent 
        / "ir2rgb_models" 
        / "model_weights" 
        / "models_ir_rgb.npz"
    )
    return default_path.exists()

