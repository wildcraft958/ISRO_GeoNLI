"""
Integration tests for orchestrator API routes.

Tests cover:
- POST /orchestrator/chat - Main chat endpoint
- POST /orchestrator/ir2rgb - IR2RGB conversion endpoint
- GET /orchestrator/modality/status - Modality detection status
- GET /orchestrator/ir2rgb/status - IR2RGB service status
- Session management endpoints
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# =============================================================================
# Chat Endpoint Tests
# =============================================================================

class TestChatEndpoint:
    """Tests for POST /orchestrator/chat endpoint."""
    
    @pytest.mark.integration
    def test_chat_endpoint_minimal_request(self, client, mock_db_dependency):
        """Test chat with minimal required fields."""
        with patch("app.api.routes.orchestrator_routes.get_db_dep", mock_db_dependency):
            with patch("app.api.routes.orchestrator_routes.app") as mock_app:
                # Mock the LangGraph workflow
                mock_app.invoke.return_value = {
                    "session_id": "test-session",
                    "caption_result": "A satellite image",
                    "execution_log": ["Modality detected: rgb"],
                    "detected_modality": "rgb",
                    "modality_confidence": 0.95,
                    "resnet_classification_used": False,
                }
                
                response = client.post(
                    "/orchestrator/chat",
                    json={
                        "image_url": "https://example.com/test.jpg",
                        "mode": "captioning"
                    }
                )
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "execution_log" in data
    
    @pytest.mark.integration
    def test_chat_endpoint_with_query(self, client, mock_db_dependency):
        """Test chat with VQA query."""
        with patch("app.api.routes.orchestrator_routes.get_db_dep", mock_db_dependency):
            with patch("app.api.routes.orchestrator_routes.app") as mock_app:
                mock_app.invoke.return_value = {
                    "session_id": "test-session",
                    "vqa_result": "Yes, there is a building",
                    "execution_log": ["Routed to VQA"],
                    "detected_modality": "rgb",
                }
                
                response = client.post(
                    "/orchestrator/chat",
                    json={
                        "image_url": "https://example.com/test.jpg",
                        "query": "Is there a building?",
                        "mode": "vqa"
                    }
                )
        
        assert response.status_code == 200
    
    @pytest.mark.integration
    def test_chat_endpoint_with_session_id(self, client, mock_db_dependency):
        """Test chat with existing session."""
        with patch("app.api.routes.orchestrator_routes.get_db_dep", mock_db_dependency):
            with patch("app.api.routes.orchestrator_routes.app") as mock_app:
                mock_app.invoke.return_value = {
                    "session_id": "existing-session",
                    "caption_result": "Test result",
                    "execution_log": [],
                }
                
                response = client.post(
                    "/orchestrator/chat",
                    json={
                        "image_url": "https://example.com/test.jpg",
                        "session_id": "existing-session"
                    }
                )
        
        assert response.status_code == 200
        assert response.json()["session_id"] == "existing-session"
    
    @pytest.mark.integration
    def test_chat_endpoint_with_ir2rgb(self, client, mock_db_dependency):
        """Test chat with IR2RGB preprocessing."""
        with patch("app.api.routes.orchestrator_routes.get_db_dep", mock_db_dependency):
            with patch("app.api.routes.orchestrator_routes.app") as mock_app:
                mock_app.invoke.return_value = {
                    "session_id": "test-session",
                    "caption_result": "Converted image",
                    "execution_log": ["IR2RGB applied"],
                    "original_image_url": "https://example.com/original.tif",
                    "image_url": "data:image/png;base64,...",
                }
                
                response = client.post(
                    "/orchestrator/chat",
                    json={
                        "image_url": "https://example.com/multispectral.tif",
                        "needs_ir2rgb": True,
                        "ir2rgb_channels": ["NIR", "R", "G"],
                        "ir2rgb_synthesize": "B"
                    }
                )
        
        assert response.status_code == 200
    
    @pytest.mark.integration
    def test_chat_endpoint_modality_detection_disabled(self, client, mock_db_dependency):
        """Test chat with modality detection disabled."""
        with patch("app.api.routes.orchestrator_routes.get_db_dep", mock_db_dependency):
            with patch("app.api.routes.orchestrator_routes.app") as mock_app:
                mock_app.invoke.return_value = {
                    "session_id": "test-session",
                    "caption_result": "Test",
                    "execution_log": [],
                    "detected_modality": None,
                }
                
                response = client.post(
                    "/orchestrator/chat",
                    json={
                        "image_url": "https://example.com/test.jpg",
                        "modality_detection_enabled": False
                    }
                )
        
        assert response.status_code == 200
    
    @pytest.mark.integration
    def test_chat_endpoint_invalid_mode(self, client):
        """Test chat with invalid mode."""
        response = client.post(
            "/orchestrator/chat",
            json={
                "image_url": "https://example.com/test.jpg",
                "mode": "invalid_mode"
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.integration
    def test_chat_endpoint_missing_image_url(self, client):
        """Test chat without image URL."""
        response = client.post(
            "/orchestrator/chat",
            json={"query": "What is this?"}
        )
        
        assert response.status_code == 422


# =============================================================================
# IR2RGB Endpoint Tests
# =============================================================================

class TestIR2RGBEndpoint:
    """Tests for POST /orchestrator/ir2rgb endpoint."""
    
    @pytest.mark.integration
    def test_ir2rgb_endpoint_success(self, client, sample_image_bytes):
        """Test successful IR2RGB conversion."""
        with patch("app.api.routes.orchestrator_routes.get_ir2rgb_service") as mock_service:
            mock_instance = MagicMock()
            mock_instance.convert_from_url.return_value = {
                "success": True,
                "rgb_image": MagicMock(),
                "rgb_image_base64": "iVBORw0KGgo...",
                "format": "PNG",
                "dimensions": {"width": 100, "height": 100},
                "size_bytes": 12345
            }
            mock_service.return_value = mock_instance
            
            response = client.post(
                "/orchestrator/ir2rgb",
                json={
                    "image_url": "https://example.com/multispectral.tif",
                    "channels": ["NIR", "R", "G"],
                    "synthesize_channel": "B"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    @pytest.mark.integration
    def test_ir2rgb_endpoint_service_unavailable(self, client):
        """Test IR2RGB when service is unavailable."""
        with patch("app.api.routes.orchestrator_routes.get_ir2rgb_service") as mock_service:
            mock_service.return_value = None
            
            response = client.post(
                "/orchestrator/ir2rgb",
                json={
                    "image_url": "https://example.com/test.tif",
                    "channels": ["NIR", "R", "G"]
                }
            )
        
        assert response.status_code == 503
    
    @pytest.mark.integration
    def test_ir2rgb_endpoint_invalid_channels(self, client):
        """Test IR2RGB with invalid channel count."""
        response = client.post(
            "/orchestrator/ir2rgb",
            json={
                "image_url": "https://example.com/test.tif",
                "channels": ["NIR", "R"]  # Only 2 channels
            }
        )
        
        assert response.status_code == 422


# =============================================================================
# Status Endpoint Tests
# =============================================================================

class TestStatusEndpoints:
    """Tests for status check endpoints."""
    
    @pytest.mark.integration
    def test_modality_status_available(self, client):
        """Test modality status when services available."""
        with patch("app.api.routes.orchestrator_routes.is_modality_detection_available") as mock_mod:
            with patch("app.api.routes.orchestrator_routes.is_resnet_classifier_available") as mock_res:
                mock_mod.return_value = True
                mock_res.return_value = True
                
                response = client.get("/orchestrator/modality/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
        assert data["statistical_detection"]["available"] is True
        assert data["resnet_classifier"]["available"] is True
    
    @pytest.mark.integration
    def test_modality_status_partial(self, client):
        """Test modality status when only statistical available."""
        with patch("app.api.routes.orchestrator_routes.is_modality_detection_available") as mock_mod:
            with patch("app.api.routes.orchestrator_routes.is_resnet_classifier_available") as mock_res:
                mock_mod.return_value = True
                mock_res.return_value = False
                
                response = client.get("/orchestrator/modality/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True  # Still available with statistical
        assert data["resnet_classifier"]["available"] is False
    
    @pytest.mark.integration
    def test_ir2rgb_status_available(self, client):
        """Test IR2RGB status when available."""
        with patch("app.api.routes.orchestrator_routes.is_ir2rgb_available") as mock_avail:
            mock_avail.return_value = True
            
            response = client.get("/orchestrator/ir2rgb/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
    
    @pytest.mark.integration
    def test_ir2rgb_status_unavailable(self, client):
        """Test IR2RGB status when unavailable."""
        with patch("app.api.routes.orchestrator_routes.is_ir2rgb_available") as mock_avail:
            mock_avail.return_value = False
            
            response = client.get("/orchestrator/ir2rgb/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is False


# =============================================================================
# Session Management Endpoint Tests
# =============================================================================

class TestSessionEndpoints:
    """Tests for session management endpoints."""
    
    @pytest.mark.integration
    def test_get_session_history(self, client, mock_db_dependency, sample_session_data):
        """Test getting session history."""
        with patch("app.api.routes.orchestrator_routes.get_db_dep", mock_db_dependency):
            with patch("app.services.memory_service.get_session") as mock_get:
                mock_get.return_value = sample_session_data
                
                response = client.get("/orchestrator/session/test-session-123/history")
        
        assert response.status_code == 200
    
    @pytest.mark.integration
    def test_get_session_history_not_found(self, client, mock_db_dependency):
        """Test getting non-existent session history."""
        with patch("app.api.routes.orchestrator_routes.get_db_dep", mock_db_dependency):
            with patch("app.services.memory_service.get_session") as mock_get:
                mock_get.return_value = None
                
                response = client.get("/orchestrator/session/nonexistent/history")
        
        assert response.status_code == 404
    
    @pytest.mark.integration
    def test_delete_session(self, client, mock_db_dependency):
        """Test deleting a session."""
        with patch("app.api.routes.orchestrator_routes.get_db_dep", mock_db_dependency):
            with patch("app.services.memory_service.delete_session") as mock_delete:
                mock_delete.return_value = True
                
                response = client.delete("/orchestrator/session/test-session-123")
        
        assert response.status_code == 200
    
    @pytest.mark.integration
    def test_get_user_sessions(self, client, mock_db_dependency, sample_session_data):
        """Test getting user sessions."""
        with patch("app.api.routes.orchestrator_routes.get_db_dep", mock_db_dependency):
            with patch("app.services.memory_service.get_user_sessions") as mock_get:
                mock_get.return_value = [sample_session_data]
                
                response = client.get(
                    "/orchestrator/sessions",
                    params={"user_id": "test-user"}
                )
        
        assert response.status_code == 200


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in routes."""
    
    @pytest.mark.integration
    def test_internal_server_error(self, client, mock_db_dependency):
        """Test handling of internal server errors."""
        with patch("app.api.routes.orchestrator_routes.get_db_dep", mock_db_dependency):
            with patch("app.api.routes.orchestrator_routes.app") as mock_app:
                mock_app.invoke.side_effect = Exception("Internal error")
                
                response = client.post(
                    "/orchestrator/chat",
                    json={"image_url": "https://example.com/test.jpg"}
                )
        
        assert response.status_code == 500
    
    @pytest.mark.integration
    def test_validation_error_response(self, client):
        """Test validation error response format."""
        response = client.post(
            "/orchestrator/chat",
            json={}  # Missing required fields
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


