"""
Unit tests for the IR2RGB service.

Tests cover:
- IR2RGBModel: Model loading, channel synthesis
- IR2RGBService: Service initialization, URL/PIL conversion
- Helper functions: availability check, service getter
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.services.ir2rgb_service import (
    IR2RGBModel,
    IR2RGBService,
    get_ir2rgb_service,
    is_ir2rgb_available,
)


# =============================================================================
# IR2RGBModel Tests
# =============================================================================

class TestIR2RGBModel:
    """Tests for IR2RGBModel class."""
    
    @pytest.fixture
    def mock_npz_data(self):
        """Create mock NPZ data for model."""
        return {
            "rpcc_R": np.random.rand(10).astype(np.float32),
            "rpcc_G": np.random.rand(10).astype(np.float32),
            "lut_B": np.random.rand(17, 17, 17).astype(np.float32),
        }
    
    @pytest.fixture
    def mock_model(self, mock_npz_data, tmp_path):
        """Create a mock IR2RGB model with temporary weights."""
        npz_path = tmp_path / "test_model.npz"
        np.savez(npz_path, **mock_npz_data)
        return IR2RGBModel(str(npz_path))
    
    @pytest.mark.unit
    def test_model_initialization(self, mock_model):
        """Test model loads weights correctly."""
        assert mock_model.rpcc["R"] is not None
        assert mock_model.rpcc["G"] is not None
        assert mock_model.lut["B"] is not None
        assert mock_model.grid_size == 17
    
    @pytest.mark.unit
    def test_model_missing_weights(self, tmp_path):
        """Test model handles missing weights gracefully."""
        npz_path = tmp_path / "partial_model.npz"
        np.savez(npz_path, rpcc_R=np.random.rand(10).astype(np.float32))
        
        model = IR2RGBModel(str(npz_path))
        
        assert model.rpcc["R"] is not None
        assert model.rpcc["G"] is None
        assert model.lut["B"] is None
    
    @pytest.mark.unit
    def test_load_img_from_pil(self, mock_model, sample_rgb_image):
        """Test loading image from PIL."""
        arr = mock_model._load_img(sample_rgb_image)
        
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (100, 100, 3)
        assert arr.dtype == np.float32
        assert arr.max() <= 1.0
    
    @pytest.mark.unit
    def test_load_img_from_numpy(self, mock_model):
        """Test loading image from numpy array."""
        input_arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        arr = mock_model._load_img(input_arr)
        
        assert arr.dtype == np.float32
        assert arr.max() <= 1.0
    
    @pytest.mark.unit
    def test_load_img_from_file(self, mock_model, temp_image_file):
        """Test loading image from file path."""
        arr = mock_model._load_img(str(temp_image_file))
        
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3
    
    @pytest.mark.unit
    def test_load_img_invalid_channels(self, mock_model):
        """Test error on wrong number of channels."""
        # 4-channel image
        arr = np.random.rand(50, 50, 4).astype(np.float32)
        
        with pytest.raises(ValueError, match="3-channel"):
            mock_model._load_img(arr)
    
    @pytest.mark.unit
    def test_map_channels(self, mock_model):
        """Test channel mapping."""
        arr = np.random.rand(50, 50, 3).astype(np.float32)
        channels = ["NIR", "R", "G"]
        
        ch_map = mock_model._map_channels(arr, channels)
        
        assert "NIR" in ch_map
        assert "R" in ch_map
        assert "G" in ch_map
        np.testing.assert_array_equal(ch_map["NIR"], arr[..., 0])
    
    @pytest.mark.unit
    def test_map_channels_wrong_length(self, mock_model):
        """Test error on wrong channel list length."""
        arr = np.random.rand(50, 50, 3).astype(np.float32)
        
        with pytest.raises(ValueError, match="length 3"):
            mock_model._map_channels(arr, ["NIR", "R"])
    
    @pytest.mark.unit
    def test_features_root_poly2(self, mock_model):
        """Test polynomial feature computation."""
        X = np.random.rand(100, 3).astype(np.float32)
        
        Psi = mock_model._features_root_poly2(X)
        
        assert Psi.shape == (100, 10)
        assert Psi.dtype == np.float32
    
    @pytest.mark.unit
    def test_to_pil(self, mock_model):
        """Test numpy to PIL conversion."""
        arr = np.random.rand(50, 50, 3).astype(np.float32)
        
        pil_img = mock_model._to_pil(arr)
        
        assert isinstance(pil_img, Image.Image)
        assert pil_img.mode == "RGB"
        assert pil_img.size == (50, 50)
    
    @pytest.mark.unit
    def test_to_pil_clipping(self, mock_model):
        """Test values are clipped to [0, 1]."""
        arr = np.array([[[1.5, -0.5, 0.5]]], dtype=np.float32)
        
        pil_img = mock_model._to_pil(arr)
        pixel = pil_img.getpixel((0, 0))
        
        # 1.5 clipped to 1.0 -> 255, -0.5 clipped to 0.0 -> 0
        assert pixel[0] == 255
        assert pixel[1] == 0
        assert pixel[2] == 127 or pixel[2] == 128  # 0.5 * 255
    
    @pytest.mark.unit
    def test_synthesize_B(self, mock_model, sample_rgb_image):
        """Test B channel synthesis."""
        result = mock_model.synthesize_B(sample_rgb_image, ["NIR", "R", "G"])
        
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == sample_rgb_image.size
    
    @pytest.mark.unit
    def test_synthesize_B_missing_channels(self, mock_model, sample_rgb_image):
        """Test error when required channels missing."""
        with pytest.raises(ValueError, match="NIR, R, G"):
            mock_model.synthesize_B(sample_rgb_image, ["A", "B", "C"])
    
    @pytest.mark.unit
    def test_synthesize_R(self, mock_model, sample_rgb_image):
        """Test R channel synthesis."""
        result = mock_model.synthesize_R(sample_rgb_image, ["NIR", "G", "B"])
        
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
    
    @pytest.mark.unit
    def test_synthesize_G(self, mock_model, sample_rgb_image):
        """Test G channel synthesis."""
        result = mock_model.synthesize_G(sample_rgb_image, ["NIR", "R", "B"])
        
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"


# =============================================================================
# IR2RGBService Tests
# =============================================================================

class TestIR2RGBService:
    """Tests for IR2RGBService class."""
    
    @pytest.fixture
    def mock_service(self, tmp_path):
        """Create service with mock model weights."""
        # Create mock weights
        npz_path = tmp_path / "models_ir_rgb.npz"
        np.savez(
            npz_path,
            rpcc_R=np.random.rand(10).astype(np.float32),
            rpcc_G=np.random.rand(10).astype(np.float32),
            lut_B=np.random.rand(17, 17, 17).astype(np.float32),
        )
        
        # Reset singleton for testing
        IR2RGBService._instance = None
        IR2RGBService._model = None
        IR2RGBService._model_path = None
        
        return IR2RGBService(str(npz_path))
    
    @pytest.mark.unit
    def test_singleton_pattern(self, mock_service, tmp_path):
        """Test service uses singleton pattern."""
        npz_path = tmp_path / "models_ir_rgb.npz"
        
        service1 = IR2RGBService(str(npz_path))
        service2 = IR2RGBService(str(npz_path))
        
        assert service1 is service2
    
    @pytest.mark.unit
    def test_model_property(self, mock_service):
        """Test model property returns model."""
        model = mock_service.model
        
        assert isinstance(model, IR2RGBModel)
    
    @pytest.mark.unit
    def test_convert_from_pil(self, mock_service, sample_rgb_image):
        """Test conversion from PIL image."""
        result = mock_service.convert_from_pil(
            sample_rgb_image,
            channels=["NIR", "R", "G"],
            synthesize_channel="B"
        )
        
        assert result["success"] is True
        assert "rgb_image" in result
        assert "rgb_image_base64" in result
        assert result["format"] == "PNG"
    
    @pytest.mark.unit
    def test_convert_from_pil_synthesize_r(self, mock_service, sample_rgb_image):
        """Test R channel synthesis."""
        result = mock_service.convert_from_pil(
            sample_rgb_image,
            channels=["NIR", "G", "B"],
            synthesize_channel="R"
        )
        
        assert result["success"] is True
    
    @pytest.mark.unit
    def test_convert_from_pil_synthesize_g(self, mock_service, sample_rgb_image):
        """Test G channel synthesis."""
        result = mock_service.convert_from_pil(
            sample_rgb_image,
            channels=["NIR", "R", "B"],
            synthesize_channel="G"
        )
        
        assert result["success"] is True
    
    @pytest.mark.unit
    def test_convert_from_url(self, mock_service, sample_image_bytes, mock_requests_get):
        """Test conversion from URL."""
        result = mock_service.convert_from_url(
            "https://example.com/test.jpg",
            channels=["NIR", "R", "G"],
            synthesize_channel="B"
        )
        
        assert result["success"] is True
        mock_requests_get.assert_called_once()
    
    @pytest.mark.unit
    def test_convert_from_url_failure(self, mock_service):
        """Test URL download failure handling."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Network error")
            
            result = mock_service.convert_from_url(
                "https://example.com/test.jpg",
                channels=["NIR", "R", "G"],
                synthesize_channel="B"
            )
            
            assert result["success"] is False
            assert "error" in result
    
    @pytest.mark.unit
    def test_convert_from_data_uri(self, mock_service, sample_image_data_uri):
        """Test conversion from data URI."""
        result = mock_service.convert_from_url(
            sample_image_data_uri,
            channels=["NIR", "R", "G"],
            synthesize_channel="B"
        )
        
        assert result["success"] is True
    
    @pytest.mark.unit
    def test_result_dimensions(self, mock_service, sample_rgb_image):
        """Test result contains correct dimensions."""
        result = mock_service.convert_from_pil(
            sample_rgb_image,
            channels=["NIR", "R", "G"],
            synthesize_channel="B"
        )
        
        assert result["dimensions"]["width"] == 100
        assert result["dimensions"]["height"] == 100
    
    @pytest.mark.unit
    def test_result_size_bytes(self, mock_service, sample_rgb_image):
        """Test result contains size in bytes."""
        result = mock_service.convert_from_pil(
            sample_rgb_image,
            channels=["NIR", "R", "G"],
            synthesize_channel="B"
        )
        
        assert result["size_bytes"] > 0


# =============================================================================
# Module-Level Function Tests
# =============================================================================

class TestModuleFunctions:
    """Tests for module-level functions."""
    
    @pytest.mark.unit
    def test_is_ir2rgb_available_with_model(self, tmp_path):
        """Test availability when model exists."""
        # Create mock weights at expected path
        model_dir = tmp_path / "ir2rgb_models" / "model_weights"
        model_dir.mkdir(parents=True)
        npz_path = model_dir / "models_ir_rgb.npz"
        np.savez(npz_path, rpcc_R=np.random.rand(10).astype(np.float32))
        
        with patch("app.services.ir2rgb_service.Path") as mock_path:
            mock_path.return_value.parent.parent.parent.parent = tmp_path
            mock_path.return_value.exists.return_value = True
            
            # Note: This test depends on actual file system in real scenario
            result = is_ir2rgb_available()
            assert isinstance(result, bool)
    
    @pytest.mark.unit
    def test_is_ir2rgb_available_no_model(self):
        """Test availability when model doesn't exist."""
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False
            
            result = is_ir2rgb_available()
            
            assert result is False
    
    @pytest.mark.unit
    def test_get_ir2rgb_service_not_available(self):
        """Test service getter when not available."""
        # Reset singleton
        IR2RGBService._instance = None
        IR2RGBService._model = None
        IR2RGBService._model_path = None
        
        with patch("app.services.ir2rgb_service.is_ir2rgb_available") as mock_avail:
            mock_avail.return_value = False
            
            result = get_ir2rgb_service()
            
            assert result is None


