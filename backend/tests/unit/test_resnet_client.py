"""
Unit tests for the ResNet classifier client.

Tests cover:
- ResNetClassifierClient: Initialization, classification, health check
- Error handling and retry logic
- Fallback classification logic
- Module-level functions
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.services.resnet_classifier_client import (
    ResNetClassifierClient,
    ResNetClassifierError,
    get_resnet_classifier_client,
    is_resnet_classifier_available,
)


# =============================================================================
# ResNetClassifierClient Tests
# =============================================================================

class TestResNetClassifierClient:
    """Tests for ResNetClassifierClient class."""
    
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ResNetClassifierClient._instance = None
        yield
    
    @pytest.fixture
    def client(self):
        """Create client instance."""
        return ResNetClassifierClient()
    
    @pytest.mark.unit
    def test_singleton_pattern(self):
        """Test client uses singleton pattern."""
        client1 = ResNetClassifierClient()
        client2 = ResNetClassifierClient()
        
        assert client1 is client2
    
    @pytest.mark.unit
    def test_initialization(self, client):
        """Test client initializes with correct defaults."""
        assert client.base_url is not None
        assert client.timeout == 60
        assert client._initialized is True
    
    @pytest.mark.unit
    def test_initialization_with_env_var(self):
        """Test client uses environment variable for URL."""
        with patch.dict("os.environ", {"RESNET_CLASSIFIER_URL": "http://custom-url"}):
            ResNetClassifierClient._instance = None
            client = ResNetClassifierClient()
            
            assert client.base_url == "http://custom-url"
    
    @pytest.mark.unit
    def test_classify_from_url_success(self, client):
        """Test successful classification from URL."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "modality": "rgb",
            "confidence": 0.95,
            "probabilities": {"rgb": 0.95, "infrared": 0.03, "sar": 0.02}
        }
        
        with patch("requests.post", return_value=mock_response):
            modality, confidence, probs = client.classify_from_url(
                "https://example.com/test.jpg"
            )
        
        assert modality == "rgb"
        assert confidence == 0.95
        assert probs["rgb"] == 0.95
    
    @pytest.mark.unit
    def test_classify_from_url_data_uri(self, client):
        """Test classification from data URI."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "modality": "infrared",
            "confidence": 0.88,
            "probabilities": {"rgb": 0.05, "infrared": 0.88, "sar": 0.07}
        }
        
        with patch("requests.post", return_value=mock_response) as mock_post:
            modality, confidence, probs = client.classify_from_url(
                "data:image/png;base64,iVBORw0KGgo..."
            )
        
        # Check that data URI was sent as image_base64
        call_args = mock_post.call_args
        assert "image_base64" in call_args.kwargs["json"]
        assert modality == "infrared"
    
    @pytest.mark.unit
    def test_classify_from_url_failure_response(self, client):
        """Test handling of failure response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": False,
            "error": "Invalid image format"
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch("requests.post", return_value=mock_response):
            with pytest.raises(ResNetClassifierError, match="Invalid image format"):
                client.classify_from_url("https://example.com/test.jpg")
    
    @pytest.mark.unit
    def test_classify_from_url_network_error(self, client):
        """Test handling of network errors."""
        with patch("requests.post") as mock_post:
            mock_post.side_effect = requests.RequestException("Connection failed")
            
            with pytest.raises(ResNetClassifierError, match="Request failed"):
                client.classify_from_url("https://example.com/test.jpg")
    
    @pytest.mark.unit
    def test_classify_from_url_http_error(self, client):
        """Test handling of HTTP errors."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        
        with patch("requests.post", return_value=mock_response):
            with pytest.raises(ResNetClassifierError):
                client.classify_from_url("https://example.com/test.jpg")
    
    @pytest.mark.unit
    def test_health_check_success(self, client):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        with patch("requests.get", return_value=mock_response):
            result = client.health_check()
        
        assert result is True
    
    @pytest.mark.unit
    def test_health_check_failure(self, client):
        """Test failed health check."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("Connection refused")
            
            result = client.health_check()
        
        assert result is False
    
    @pytest.mark.unit
    def test_health_check_unhealthy_status(self, client):
        """Test health check with unhealthy status code."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        
        with patch("requests.get", return_value=mock_response):
            result = client.health_check()
        
        assert result is False


# =============================================================================
# Fallback Classification Tests
# =============================================================================

class TestClassifyWithFallback:
    """Tests for classify_with_fallback method."""
    
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ResNetClassifierClient._instance = None
        yield
    
    @pytest.fixture
    def client(self):
        """Create client instance."""
        return ResNetClassifierClient()
    
    @pytest.mark.unit
    def test_skip_resnet_for_high_confidence_rgb(self, client):
        """Test ResNet is skipped for high-confidence RGB."""
        result = client.classify_with_fallback(
            image_url="https://example.com/test.jpg",
            statistical_modality="rgb",
            statistical_confidence=0.9
        )
        
        assert result["final_modality"] == "rgb"
        assert result["source"] == "statistical"
        assert result["resnet_used"] is False
    
    @pytest.mark.unit
    def test_use_resnet_for_low_confidence(self, client):
        """Test ResNet is used for low-confidence results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "modality": "sar",
            "confidence": 0.85,
            "probabilities": {"rgb": 0.10, "infrared": 0.05, "sar": 0.85}
        }
        
        with patch("requests.post", return_value=mock_response):
            result = client.classify_with_fallback(
                image_url="https://example.com/test.jpg",
                statistical_modality="sar",
                statistical_confidence=0.5
            )
        
        assert result["final_modality"] == "sar"
        assert result["source"] == "resnet"
        assert result["resnet_used"] is True
    
    @pytest.mark.unit
    def test_use_resnet_for_sar(self, client):
        """Test ResNet is used for SAR classification."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "modality": "sar",
            "confidence": 0.92,
            "probabilities": {"rgb": 0.05, "infrared": 0.03, "sar": 0.92}
        }
        
        with patch("requests.post", return_value=mock_response):
            result = client.classify_with_fallback(
                image_url="https://example.com/test.jpg",
                statistical_modality="sar",
                statistical_confidence=0.7
            )
        
        assert result["final_modality"] == "sar"
        assert result["source"] == "resnet"
    
    @pytest.mark.unit
    def test_fallback_to_statistical_on_error(self, client):
        """Test fallback to statistical when ResNet fails."""
        with patch("requests.post") as mock_post:
            mock_post.side_effect = requests.RequestException("Service unavailable")
            
            result = client.classify_with_fallback(
                image_url="https://example.com/test.jpg",
                statistical_modality="infrared",
                statistical_confidence=0.6
            )
        
        assert result["final_modality"] == "infrared"
        assert result["source"] == "statistical_fallback"
        assert result["resnet_used"] is False
        assert "error" in result
    
    @pytest.mark.unit
    def test_fallback_unknown_when_no_statistical(self, client):
        """Test fallback to unknown when no statistical result."""
        with patch("requests.post") as mock_post:
            mock_post.side_effect = requests.RequestException("Service unavailable")
            
            result = client.classify_with_fallback(
                image_url="https://example.com/test.jpg"
            )
        
        assert result["final_modality"] == "unknown"
        assert result["confidence"] == 0.0


# =============================================================================
# Module-Level Function Tests
# =============================================================================

class TestModuleFunctions:
    """Tests for module-level functions."""
    
    @pytest.fixture(autouse=True)
    def reset_global(self):
        """Reset global client before each test."""
        import app.services.resnet_classifier_client as rc
        rc._resnet_client = None
        ResNetClassifierClient._instance = None
        yield
    
    @pytest.mark.unit
    def test_get_resnet_classifier_client(self):
        """Test getting client instance."""
        client = get_resnet_classifier_client()
        
        assert isinstance(client, ResNetClassifierClient)
    
    @pytest.mark.unit
    def test_get_resnet_classifier_client_singleton(self):
        """Test client is singleton."""
        client1 = get_resnet_classifier_client()
        client2 = get_resnet_classifier_client()
        
        assert client1 is client2
    
    @pytest.mark.unit
    def test_is_resnet_classifier_available_true(self):
        """Test availability check when service is up."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        with patch("requests.get", return_value=mock_response):
            result = is_resnet_classifier_available()
        
        assert result is True
    
    @pytest.mark.unit
    def test_is_resnet_classifier_available_false(self):
        """Test availability check when service is down."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("Connection refused")
            
            result = is_resnet_classifier_available()
        
        assert result is False
    
    @pytest.mark.unit
    def test_is_resnet_classifier_available_exception(self):
        """Test availability check handles exceptions."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Unexpected error")
            
            result = is_resnet_classifier_available()
        
        assert result is False


# =============================================================================
# Error Class Tests
# =============================================================================

class TestResNetClassifierError:
    """Tests for ResNetClassifierError exception."""
    
    @pytest.mark.unit
    def test_error_message(self):
        """Test error message is preserved."""
        error = ResNetClassifierError("Test error message")
        
        assert str(error) == "Test error message"
    
    @pytest.mark.unit
    def test_error_inheritance(self):
        """Test error inherits from Exception."""
        error = ResNetClassifierError("Test")
        
        assert isinstance(error, Exception)


