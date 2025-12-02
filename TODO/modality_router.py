import os
from glob import glob
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2
from PIL import Image
from scipy.stats import kurtosis
import math


class ImageLoader:
    """Handle image loading and conversion from various input types."""
    
    @staticmethod
    def load_image_arr(path_or_arr) -> np.ndarray:
        """Load path (preserve dtype & channels) or pass-through ndarray/PIL.Image."""
        if isinstance(path_or_arr, str):
            arr = cv2.imread(path_or_arr, -1)
            if arr is None:
                arr = np.array(Image.open(path_or_arr))
            return arr
        if isinstance(path_or_arr, Image.Image):
            return np.array(path_or_arr)
        if isinstance(path_or_arr, np.ndarray):
            return path_or_arr
        raise ValueError("Unsupported image input type")
    
    @staticmethod
    def get_image_metadata(path_or_pil) -> Dict[str, Any]:
        """Extract metadata from image (PIL or path)."""
        meta = {}
        try:
            if isinstance(path_or_pil, Image.Image):
                img = path_or_pil
            else:
                img = Image.open(path_or_pil)
            
            info = dict(img.info) if hasattr(img, "info") else {}
            meta.update(info)
            
            try:
                tags = getattr(img, "tag_v2", None)
                if tags is not None:
                    for k, v in tags.items():
                        meta[str(k)] = v
            except Exception:
                pass
            
            meta["pil_mode"] = img.mode
        except Exception:
            pass
        return meta
    
    @staticmethod
    def rgba_to_rgb_pil(pil_img: Image.Image, background=(255, 255, 255)) -> Image.Image:
        """Composite RGBA PIL image over background and return RGB PIL image."""
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
    """Calculate statistical features for images."""
    
    @staticmethod
    def collapse_to_single_band(arr: np.ndarray) -> np.ndarray:
        """Return single-band float32 image (mean across channels if multi-channel)."""
        if arr.ndim == 3:
            return arr.mean(axis=2).astype(np.float32)
        return arr.astype(np.float32)
    
    @staticmethod
    def local_speckle_measure(arr: np.ndarray, ksize: int = 7) -> float:
        """Mean local coefficient-of-variation (speckle-like measure)."""
        a = arr.astype(np.float32)
        if a.size == 0:
            return 0.0
        mean = cv2.boxFilter(a, ddepth=-1, ksize=(ksize, ksize), normalize=True)
        sq = cv2.boxFilter(a * a, ddepth=-1, ksize=(ksize, ksize), normalize=True)
        var = sq - mean * mean
        std = np.sqrt(np.clip(var, 0, None))
        lcv = std / (np.abs(mean) + 1e-9)
        return float(np.nanmean(lcv))


class SARDetector:
    """Detect SAR imagery using threshold-based voting."""
    
    DEFAULT_THRESHOLDS = {
        "cv": 0.32624886285281385,
        "local_cv": 0.12780852243304253,
        "kurtosis": 3.8785458505153656
    }
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self.stats_calculator = ImageStatsCalculator()
    
    def is_probably_sar_from_thresholds(
        self, 
        arr: np.ndarray, 
        verbose: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """Compute stats and return (is_sar, diagnostics). Require >=2 votes for SAR."""
        band = self.stats_calculator.collapse_to_single_band(arr)
        mn, mx = float(band.min()), float(band.max())
        mean = float(band.mean())
        std = float(band.std())
        cv = std / (abs(mean) + 1e-9)
        
        try:
            k = float(kurtosis(band.ravel(), fisher=False, nan_policy="omit"))
        except Exception:
            k = float("nan")
        
        lcv = self.stats_calculator.local_speckle_measure(band, ksize=7)
        
        score = 0
        reasons = []
        
        if not math.isnan(cv) and self.thresholds.get("cv") is not None and cv >= self.thresholds["cv"]:
            score += 1
            reasons.append(f"cv={cv:.3f}>={self.thresholds['cv']:.3f}")
        
        if not math.isnan(lcv) and self.thresholds.get("local_cv") is not None and lcv >= self.thresholds["local_cv"]:
            score += 1
            reasons.append(f"local_cv={lcv:.3f}>={self.thresholds['local_cv']:.3f}")
        
        if not math.isnan(k) and self.thresholds.get("kurtosis") is not None and k >= self.thresholds["kurtosis"]:
            score += 1
            reasons.append(f"kurtosis={k:.3f}>={self.thresholds['kurtosis']:.3f}")
        
        is_sar = score >= 2
        diag = {
            "cv": cv, "local_cv": lcv, "kurtosis": k, 
            "score": score, "reasons": reasons, 
            "mean": mean, "min": mn, "max": mx
        }
        
        if verbose:
            print("SAR check:", diag)
        
        return is_sar, diag


class AlphaDetector:
    """Detect alpha-like channels in RGBA images."""
    
    @staticmethod
    def alpha_like(
        alpha_arr: np.ndarray, 
        frac_threshold: float = 0.4, 
        uniq_threshold: int = 20
    ) -> bool:
        """Heuristic: many pixels exactly 0 or 255 OR few unique values => alpha-like mask."""
        if alpha_arr is None:
            return False
        
        flat = alpha_arr.ravel()
        if flat.size == 0:
            return False
        
        if np.issubdtype(flat.dtype, np.floating):
            return False
        
        zeros = (flat == 0).sum()
        maxs = (flat == 255).sum()
        frac_end = (zeros + maxs) / (flat.size + 1e-12)
        uniq = np.unique(flat)
        
        return (frac_end > frac_threshold) or (uniq.size < uniq_threshold)


class MetadataParser:
    """Parse image metadata for modality hints."""
    
    @staticmethod
    def parse_modality_from_metadata(meta: Dict[str, Any]) -> Optional[str]:
        """Return modality if found in metadata, else None."""
        if not meta:
            return None
        
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
        """Return band count if found in metadata, else None."""
        if not meta:
            return None
        
        for key in ("bands", "Bands", "SAMPLESPERPIXEL", "samples_per_pixel", "samples"):
            val = meta.get(key) or meta.get(key.lower()) or meta.get(str(key))
            try:
                if val is not None:
                    return int(val)
            except Exception:
                pass
        return None


class ImageModalityRouter:
    """Main router for image modality classification."""
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.image_loader = ImageLoader()
        self.sar_detector = SARDetector(thresholds)
        self.alpha_detector = AlphaDetector()
        self.metadata_parser = MetadataParser()
    
    def route_image(
        self,
        image_input,
        metadata_priority: bool = True,
        force_sar_folder: Optional[str] = None,
        verbose: bool = False
    ) -> Tuple[str, Optional[Image.Image], Dict[str, Any]]:
        """
        Return (modality, rgb_pil_if_applicable, diagnostics).
        modality in {'rgb','infrared','sar','unknown'}.
        """
        # Extract metadata
        meta = self._get_metadata(image_input)
        
        # Check metadata first if priority enabled
        if metadata_priority and meta:
            result = self._route_by_metadata(image_input, meta)
            if result is not None:
                return result
        
        # Check force_sar_folder
        if isinstance(image_input, str) and force_sar_folder and force_sar_folder in image_input:
            return "sar", None, {"reason": "forced_by_folder"}
        
        # Load array
        arr = self.image_loader.load_image_arr(image_input)
        if arr is None:
            return "unknown", None, {"error": "load_failed"}
        
        channels = 1 if arr.ndim == 2 else arr.shape[2]
        dtype = arr.dtype
        
        # Route by channels
        if channels >= 4:
            return self._route_multi_channel(arr, image_input, channels, dtype, meta)
        
        if channels == 3:
            return self._route_three_channel(arr, image_input, verbose, meta)
        
        if channels == 1:
            return self._route_single_channel(arr, verbose, meta)
        
        return "unknown", None, {"reason": "no_rule_matched", "metadata": meta}
    
    def _get_metadata(self, image_input) -> Dict[str, Any]:
        """Extract metadata from image input."""
        try:
            if isinstance(image_input, (str, Image.Image)):
                return self.image_loader.get_image_metadata(image_input)
        except Exception:
            pass
        return {}
    
    def _route_by_metadata(
        self, 
        image_input, 
        meta: Dict[str, Any]
    ) -> Optional[Tuple[str, Optional[Image.Image], Dict[str, Any]]]:
        """Route based on metadata if possible."""
        modality = self.metadata_parser.parse_modality_from_metadata(meta)
        if modality:
            if modality == "rgb":
                try:
                    pil = Image.open(image_input) if isinstance(image_input, str) else image_input
                    return "rgb", pil.convert("RGB"), {"reason": "metadata_modality", "metadata": meta}
                except Exception:
                    return "rgb", None, {"reason": "metadata_modality_failed_to_load_pil", "metadata": meta}
            return modality, None, {"reason": "metadata_modality", "metadata": meta}
        
        bands = self.metadata_parser.parse_bands_from_metadata(meta)
        if bands is not None:
            if bands >= 4:
                return "infrared", None, {"reason": "metadata_bands_count", "bands": bands, "metadata": meta}
            if bands == 3:
                try:
                    pil = Image.open(image_input) if isinstance(image_input, str) else image_input
                    return "rgb", pil.convert("RGB"), {"reason": "metadata_bands_count", "bands": bands, "metadata": meta}
                except Exception:
                    pass
        
        return None
    
    def _route_multi_channel(
        self, 
        arr: np.ndarray, 
        image_input, 
        channels: int, 
        dtype, 
        meta: Dict[str, Any]
    ) -> Tuple[str, Optional[Image.Image], Dict[str, Any]]:
        """Route multi-channel (>=4) images."""
        alpha = arr[..., 3] if arr.ndim == 3 and arr.shape[2] == 4 else None
        
        if alpha is not None and self.alpha_detector.alpha_like(alpha):
            try:
                loaded_cv2 = cv2.imread(image_input, -1) if isinstance(image_input, str) else None
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
        image_input, 
        verbose: bool, 
        meta: Dict[str, Any]
    ) -> Tuple[str, Optional[Image.Image], Dict[str, Any]]:
        """Route 3-channel images."""
        is_sar, diag = self.sar_detector.is_probably_sar_from_thresholds(arr, verbose=verbose)
        
        if is_sar:
            return "sar", None, {"reason": "3ch_sar_detected", **diag, "metadata": meta}
        
        try:
            loaded_cv2 = cv2.imread(image_input, -1) if isinstance(image_input, str) else None
            if loaded_cv2 is not None:
                pil_img = Image.fromarray(arr[..., ::-1])
            else:
                pil_img = Image.fromarray(arr)
        except Exception:
            pil_img = Image.fromarray(arr)
        
        return "rgb", pil_img.convert("RGB"), {"reason": "3ch_not_sar", **diag, "metadata": meta}
    
    def _route_single_channel(
        self, 
        arr: np.ndarray, 
        verbose: bool, 
        meta: Dict[str, Any]
    ) -> Tuple[str, Optional[Image.Image], Dict[str, Any]]:
        """Route single-channel images."""
        is_sar, diag = self.sar_detector.is_probably_sar_from_thresholds(arr, verbose=verbose)
        
        if is_sar:
            return "sar", None, {"reason": "1ch_sar_detected", **diag, "metadata": meta}
        
        return "infrared", None, {"reason": "1ch_not_sar", **diag, "metadata": meta}


class FolderRoutingSummarizer:
    """Summarize modality routing for entire folders."""
    
    def __init__(self, router: Optional[ImageModalityRouter] = None):
        self.router = router or ImageModalityRouter()
    
    def folder_routing_summary(
        self, 
        folder: str, 
        force_sar_folder: Optional[str] = None, 
        limit: Optional[int] = None
    ) -> Tuple[Dict[str, int], Dict[str, Dict[str, Any]]]:
        """Return counts and details for all images in folder."""
        paths = sorted(glob(os.path.join(folder, "*")))
        if limit is not None:
            paths = paths[:limit]
        
        counts = {"sar": 0, "infrared": 0, "rgb": 0, "unknown": 0}
        details = {}
        
        for p in paths:
            mod, rgb_img, diag = self.router.route_image(
                p, 
                metadata_priority=True, 
                force_sar_folder=force_sar_folder
            )
            counts[mod] += 1
            details[p] = {"mod": mod, "diag": diag}
        
        return counts, details


# Usage example
if __name__ == "__main__":
    # Basic routing
    router = ImageModalityRouter()
    modality, rgb_pil, diagnostics = router.route_image("path/to/image.png")
    print(f"Modality: {modality}")
    print(f"Diagnostics: {diagnostics}")
    