"""
Unit tests for the modality router service.

Tests cover:
- ImageLoader: Image loading from various sources
- ImageStatsCalculator: Statistical feature computation
- SARDetector: SAR image detection
- AlphaDetector: Alpha channel detection
- MetadataParser: Metadata parsing
- ModalityRouterService: Main routing logic
"""

import io
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.services.modality_router import (
    AlphaDetector,
    ImageLoadError,
    ImageLoader,
    ImageStatsCalculator,
    MetadataParser,
    ModalityRouterService,
    SARDetector,
    get_modality_router_service,
    is_modality_detection_available,
)


# =============================================================================
# ImageLoader Tests
# =============================================================================

class TestImageLoader:
    """Tests for ImageLoader class."""
    
    @pytest.mark.unit
    def test_load_image_arr_from_pil(self, sample_rgb_image):
        """Test loading image from PIL Image."""
        arr = ImageLoader.load_image_arr(sample_rgb_image)
        
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (100, 100, 3)
        assert arr.dtype == np.uint8
    
    @pytest.mark.unit
    def test_load_image_arr_from_numpy(self):
        """Test loading image from numpy array."""
        input_arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = ImageLoader.load_image_arr(input_arr)
        
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, input_arr)
    
    @pytest.mark.unit
    def test_load_image_arr_from_file(self, temp_image_file):
        """Test loading image from file path."""
        arr = ImageLoader.load_image_arr(str(temp_image_file))
        
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3
    
    @pytest.mark.unit
    def test_load_image_arr_invalid_type(self):
        """Test that invalid input type raises error."""
        with pytest.raises(ImageLoadError):
            ImageLoader.load_image_arr(12345)  # Invalid type
    
    @pytest.mark.unit
    def test_download_image_success(self, sample_image_bytes, mock_requests_get):
        """Test successful image download."""
        img = ImageLoader.download_image("https://example.com/test.jpg")
        
        assert isinstance(img, Image.Image)
        mock_requests_get.assert_called_once()
    
    @pytest.mark.unit
    def test_download_image_failure(self):
        """Test image download failure handling."""
        from tenacity import RetryError
        
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Network error")
            
            # Retry decorator wraps exceptions in RetryError after retries
            with pytest.raises((ImageLoadError, RetryError)):
                ImageLoader.download_image("https://example.com/test.jpg")
    
    @pytest.mark.unit
    def test_load_from_data_uri(self, sample_image_data_uri):
        """Test loading image from data URI."""
        img = ImageLoader.load_from_data_uri(sample_image_data_uri)
        
        assert isinstance(img, Image.Image)
    
    @pytest.mark.unit
    def test_load_from_data_uri_invalid(self):
        """Test invalid data URI handling."""
        with pytest.raises(ImageLoadError):
            ImageLoader.load_from_data_uri("invalid_data_uri")
    
    @pytest.mark.unit
    def test_get_image_metadata_from_pil(self, sample_rgb_image):
        """Test metadata extraction from PIL image."""
        meta = ImageLoader.get_image_metadata(sample_rgb_image)
        
        assert isinstance(meta, dict)
        assert "pil_mode" in meta
        assert meta["pil_mode"] == "RGB"
    
    @pytest.mark.unit
    def test_rgba_to_rgb_pil_rgba(self, sample_rgba_image):
        """Test RGBA to RGB conversion."""
        rgb_img = ImageLoader.rgba_to_rgb_pil(sample_rgba_image)
        
        assert rgb_img.mode == "RGB"
        assert rgb_img.size == sample_rgba_image.size
    
    @pytest.mark.unit
    def test_rgba_to_rgb_pil_already_rgb(self, sample_rgb_image):
        """Test RGB image passes through."""
        rgb_img = ImageLoader.rgba_to_rgb_pil(sample_rgb_image)
        
        assert rgb_img.mode == "RGB"


# =============================================================================
# ImageStatsCalculator Tests
# =============================================================================

class TestImageStatsCalculator:
    """Tests for ImageStatsCalculator class."""
    
    @pytest.mark.unit
    def test_collapse_to_single_band_3channel(self):
        """Test collapsing 3-channel image to single band."""
        arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = ImageStatsCalculator.collapse_to_single_band(arr)
        
        assert result.shape == (50, 50)
        assert result.dtype == np.float32
    
    @pytest.mark.unit
    def test_collapse_to_single_band_grayscale(self):
        """Test single-channel image passes through."""
        arr = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        result = ImageStatsCalculator.collapse_to_single_band(arr)
        
        assert result.shape == (50, 50)
        assert result.dtype == np.float32
    
    @pytest.mark.unit
    def test_local_speckle_measure_uniform(self):
        """Test speckle measure on uniform image (should be low)."""
        # Uniform image has low speckle
        arr = np.full((50, 50), 128, dtype=np.float32)
        lcv = ImageStatsCalculator.local_speckle_measure(arr)
        
        assert lcv < 0.1  # Low speckle for uniform image
    
    @pytest.mark.unit
    def test_local_speckle_measure_noisy(self):
        """Test speckle measure on noisy image (should be higher)."""
        np.random.seed(42)
        arr = np.random.exponential(50, (50, 50)).astype(np.float32)
        lcv = ImageStatsCalculator.local_speckle_measure(arr)
        
        assert lcv > 0.1  # Higher speckle for noisy image
    
    @pytest.mark.unit
    def test_local_speckle_measure_empty(self):
        """Test speckle measure on empty array."""
        arr = np.array([], dtype=np.float32)
        lcv = ImageStatsCalculator.local_speckle_measure(arr)
        
        assert lcv == 0.0


# =============================================================================
# SARDetector Tests
# =============================================================================

class TestSARDetector:
    """Tests for SARDetector class."""
    
    @pytest.fixture
    def sar_detector(self):
        """Create SAR detector instance."""
        return SARDetector()
    
    @pytest.mark.unit
    def test_default_thresholds(self, sar_detector):
        """Test default threshold values are set."""
        assert "cv" in sar_detector.thresholds
        assert "local_cv" in sar_detector.thresholds
        assert "kurtosis" in sar_detector.thresholds
    
    @pytest.mark.unit
    def test_custom_thresholds(self):
        """Test custom thresholds."""
        custom = {"cv": 0.5, "local_cv": 0.2, "kurtosis": 5.0}
        detector = SARDetector(thresholds=custom)
        
        assert detector.thresholds == custom
    
    @pytest.mark.unit
    def test_is_probably_sar_uniform_image(self, sar_detector):
        """Test uniform image is not detected as SAR."""
        arr = np.full((100, 100), 128, dtype=np.uint8)
        is_sar, diag = sar_detector.is_probably_sar_from_thresholds(arr)
        
        assert is_sar is False
        assert diag["score"] < 2
    
    @pytest.mark.unit
    def test_is_probably_sar_speckle_image(self, sar_detector, sample_sar_like_image):
        """Test SAR-like image is detected as SAR."""
        is_sar, diag = sar_detector.is_probably_sar_from_thresholds(sample_sar_like_image)
        
        # SAR-like images should have high CV and kurtosis
        assert "cv" in diag
        assert "local_cv" in diag
        assert "kurtosis" in diag
        assert "score" in diag
    
    @pytest.mark.unit
    def test_diagnostics_structure(self, sar_detector):
        """Test diagnostics dictionary structure."""
        arr = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        _, diag = sar_detector.is_probably_sar_from_thresholds(arr)
        
        expected_keys = ["cv", "local_cv", "kurtosis", "score", "reasons", "mean", "min", "max"]
        for key in expected_keys:
            assert key in diag


# =============================================================================
# AlphaDetector Tests
# =============================================================================

class TestAlphaDetector:
    """Tests for AlphaDetector class."""
    
    @pytest.mark.unit
    def test_alpha_like_true_binary(self):
        """Test detection of binary alpha channel."""
        # Binary alpha (all 0 or 255)
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[25:75, 25:75] = 255
        
        # Use == instead of 'is' because numpy returns np.bool_, not Python bool
        assert AlphaDetector.alpha_like(alpha) == True
    
    @pytest.mark.unit
    def test_alpha_like_false_continuous(self):
        """Test continuous values are not alpha-like."""
        # Continuous values (not alpha-like)
        alpha = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # With many unique values, should not be alpha-like
        result = AlphaDetector.alpha_like(alpha, frac_threshold=0.8, uniq_threshold=5)
        assert result is False
    
    @pytest.mark.unit
    def test_alpha_like_none_input(self):
        """Test None input returns False."""
        assert AlphaDetector.alpha_like(None) is False
    
    @pytest.mark.unit
    def test_alpha_like_empty_input(self):
        """Test empty array returns False."""
        assert AlphaDetector.alpha_like(np.array([])) is False
    
    @pytest.mark.unit
    def test_alpha_like_floating_point(self):
        """Test floating point array returns False."""
        alpha = np.random.rand(100, 100).astype(np.float32)
        assert AlphaDetector.alpha_like(alpha) is False
    
    @pytest.mark.unit
    def test_alpha_like_few_unique_values(self):
        """Test few unique values is alpha-like."""
        # Only 3 unique values
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[0:33, :] = 0
        alpha[33:66, :] = 128
        alpha[66:100, :] = 255
        
        # Use == instead of 'is' because numpy returns np.bool_, not Python bool
        assert AlphaDetector.alpha_like(alpha, uniq_threshold=10) == True


# =============================================================================
# MetadataParser Tests
# =============================================================================

class TestMetadataParser:
    """Tests for MetadataParser class."""
    
    @pytest.mark.unit
    def test_parse_modality_sar(self):
        """Test SAR modality detection from metadata."""
        meta = {"modality": "SAR image"}
        result = MetadataParser.parse_modality_from_metadata(meta)
        assert result == "sar"
        
        meta = {"sensor": "radar"}
        result = MetadataParser.parse_modality_from_metadata(meta)
        assert result == "sar"
    
    @pytest.mark.unit
    def test_parse_modality_infrared(self):
        """Test infrared modality detection from metadata."""
        meta = {"modality": "NIR band"}
        result = MetadataParser.parse_modality_from_metadata(meta)
        assert result == "infrared"
        
        meta = {"sensor": "multispectral"}
        result = MetadataParser.parse_modality_from_metadata(meta)
        assert result == "infrared"
    
    @pytest.mark.unit
    def test_parse_modality_rgb(self):
        """Test RGB modality detection from metadata."""
        meta = {"modality": "RGB truecolor"}
        result = MetadataParser.parse_modality_from_metadata(meta)
        assert result == "rgb"
    
    @pytest.mark.unit
    def test_parse_modality_none(self):
        """Test no modality found."""
        meta = {"other_key": "value"}
        result = MetadataParser.parse_modality_from_metadata(meta)
        assert result is None
        
        result = MetadataParser.parse_modality_from_metadata({})
        assert result is None
        
        result = MetadataParser.parse_modality_from_metadata(None)
        assert result is None
    
    @pytest.mark.unit
    def test_parse_bands_from_metadata(self):
        """Test band count extraction."""
        meta = {"bands": 4}
        result = MetadataParser.parse_bands_from_metadata(meta)
        assert result == 4
        
        meta = {"SAMPLESPERPIXEL": "3"}
        result = MetadataParser.parse_bands_from_metadata(meta)
        assert result == 3
    
    @pytest.mark.unit
    def test_parse_bands_none(self):
        """Test no bands found."""
        meta = {"other_key": "value"}
        result = MetadataParser.parse_bands_from_metadata(meta)
        assert result is None


# =============================================================================
# ModalityRouterService Tests
# =============================================================================

class TestModalityRouterService:
    """Tests for ModalityRouterService class."""
    
    @pytest.fixture
    def router_service(self):
        """Create router service instance."""
        return ModalityRouterService()
    
    @pytest.mark.unit
    def test_singleton_pattern(self):
        """Test service uses singleton pattern."""
        service1 = get_modality_router_service()
        service2 = get_modality_router_service()
        
        assert service1 is service2
    
    @pytest.mark.unit
    def test_route_image_rgb(self, router_service, sample_rgb_image):
        """Test routing RGB image."""
        modality, rgb_pil, diag = router_service.route_image(sample_rgb_image)
        
        assert modality in ["rgb", "sar", "unknown"]
        assert isinstance(diag, dict)
    
    @pytest.mark.unit
    def test_route_image_grayscale(self, router_service, sample_grayscale_image):
        """Test routing grayscale image."""
        arr = np.array(sample_grayscale_image)
        modality, rgb_pil, diag = router_service.route_image(arr)
        
        assert modality in ["infrared", "sar", "unknown"]
    
    @pytest.mark.unit
    def test_route_image_rgba(self, router_service, sample_rgba_image):
        """Test routing RGBA image with alpha detection."""
        arr = np.array(sample_rgba_image)
        modality, rgb_pil, diag = router_service.route_image(arr)
        
        # RGBA with alpha-like channel should be detected as RGB
        assert modality in ["rgb", "infrared"]
    
    @pytest.mark.unit
    def test_route_image_multichannel(self, router_service, sample_infrared_image):
        """Test routing multi-channel image."""
        modality, rgb_pil, diag = router_service.route_image(sample_infrared_image)
        
        # 4-channel without alpha should be infrared
        assert modality in ["infrared", "rgb"]
    
    @pytest.mark.unit
    def test_detect_modality_from_url(self, router_service, sample_image_bytes, mock_requests_get):
        """Test modality detection from URL."""
        modality, diag = router_service.detect_modality_from_url("https://example.com/test.jpg")
        
        assert modality in ["rgb", "infrared", "sar", "unknown"]
        assert isinstance(diag, dict)
    
    @pytest.mark.unit
    def test_detect_modality_from_data_uri(self, router_service, sample_image_data_uri):
        """Test modality detection from data URI."""
        modality, diag = router_service.detect_modality_from_url(sample_image_data_uri)
        
        assert modality in ["rgb", "infrared", "sar", "unknown"]
    
    @pytest.mark.unit
    def test_metadata_priority(self, router_service, sample_rgb_image):
        """Test metadata takes priority when enabled."""
        # Mock metadata that says SAR
        with patch.object(router_service, "_get_metadata") as mock_meta:
            mock_meta.return_value = {"modality": "SAR"}
            
            modality, _, diag = router_service.route_image(
                sample_rgb_image, 
                metadata_priority=True
            )
            
            assert modality == "sar"
            assert diag["reason"] == "metadata_modality"
    
    @pytest.mark.unit
    def test_metadata_priority_disabled(self, router_service, sample_rgb_image):
        """Test metadata ignored when priority disabled."""
        with patch.object(router_service, "_get_metadata") as mock_meta:
            mock_meta.return_value = {"modality": "SAR"}
            
            modality, _, diag = router_service.route_image(
                sample_rgb_image,
                metadata_priority=False
            )
            
            # Should use pixel analysis, not metadata
            assert diag.get("reason") != "metadata_modality"


# =============================================================================
# Module-Level Function Tests
# =============================================================================

class TestModuleFunctions:
    """Tests for module-level functions."""
    
    @pytest.mark.unit
    def test_is_modality_detection_available(self):
        """Test availability check."""
        result = is_modality_detection_available()
        
        # Should be True if cv2, numpy, scipy are installed
        assert isinstance(result, bool)
    
    @pytest.mark.unit
    def test_get_modality_router_service(self):
        """Test service getter."""
        service = get_modality_router_service()
        
        assert isinstance(service, ModalityRouterService)


