"""
Integration tests for orchestrator workflow nodes.

Tests cover:
- detect_modality_node: Modality detection logic
- preprocessing_router: Routing based on modality
- preprocess_ir2rgb_node: IR2RGB preprocessing
- route_to_service_node: Service routing
- Service nodes: grounding, vqa, captioning
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.orchestrator import (
    append_assistant_message,
    append_user_message,
    detect_modality_node,
    preprocess_ir2rgb_node,
    preprocessing_router,
    route_to_service_node,
)
from app.schemas.orchestrator_schema import AgentState


# =============================================================================
# Memory Helper Tests
# =============================================================================

class TestMemoryHelpers:
    """Tests for memory helper functions."""
    
    @pytest.mark.integration
    def test_append_user_message(self, sample_agent_state):
        """Test appending user message to state."""
        state: AgentState = sample_agent_state.copy()
        state["user_query"] = "What is this?"
        state["image_url"] = "https://example.com/test.jpg"
        
        result = append_user_message(state)
        
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "What is this?"
    
    @pytest.mark.integration
    def test_append_assistant_message(self, sample_agent_state):
        """Test appending assistant message to state."""
        state: AgentState = sample_agent_state.copy()
        state["messages"] = [{"role": "user", "content": "Test"}]
        
        result = append_assistant_message(
            state,
            "This is a test response",
            "caption",
            {"model": "gpt-4"}
        )
        
        assert len(result["messages"]) == 2
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][1]["content"] == "This is a test response"
        assert result["messages"][1]["response_type"] == "caption"


# =============================================================================
# Modality Detection Node Tests
# =============================================================================

class TestDetectModalityNode:
    """Tests for detect_modality_node."""
    
    @pytest.mark.integration
    def test_modality_detection_enabled(self, sample_agent_state):
        """Test modality detection when enabled."""
        state: AgentState = sample_agent_state.copy()
        state["modality_detection_enabled"] = True
        state["image_url"] = "https://example.com/test.jpg"
        
        with patch("app.orchestrator.get_modality_router_service") as mock_service:
            mock_instance = MagicMock()
            mock_instance.detect_modality_from_url.return_value = (
                "rgb",
                {"reason": "statistical", "cv": 0.2}
            )
            mock_service.return_value = mock_instance
            
            with patch("app.orchestrator.is_modality_detection_available", return_value=True):
                result = detect_modality_node(state)
        
        assert result["detected_modality"] == "rgb"
        assert "modality_diagnostics" in result
    
    @pytest.mark.integration
    def test_modality_detection_disabled(self, sample_agent_state):
        """Test modality detection when disabled."""
        state: AgentState = sample_agent_state.copy()
        state["modality_detection_enabled"] = False
        
        result = detect_modality_node(state)
        
        assert "detected_modality" not in result or result.get("detected_modality") is None
    
    @pytest.mark.integration
    def test_modality_detection_sar_with_resnet(self, sample_agent_state):
        """Test SAR detection with ResNet fallback."""
        state: AgentState = sample_agent_state.copy()
        state["modality_detection_enabled"] = True
        state["image_url"] = "https://example.com/sar.jpg"
        
        with patch("app.orchestrator.get_modality_router_service") as mock_router:
            mock_router_instance = MagicMock()
            mock_router_instance.detect_modality_from_url.return_value = (
                "sar",
                {"reason": "statistical"}
            )
            mock_router.return_value = mock_router_instance
            
            with patch("app.orchestrator.is_modality_detection_available", return_value=True):
                with patch("app.orchestrator.is_resnet_classifier_available", return_value=True):
                    with patch("app.orchestrator.get_resnet_classifier_client") as mock_resnet:
                        mock_resnet_instance = MagicMock()
                        mock_resnet_instance.classify_from_url.return_value = (
                            "sar",
                            0.92,
                            {"rgb": 0.05, "infrared": 0.03, "sar": 0.92}
                        )
                        mock_resnet.return_value = mock_resnet_instance
                        
                        result = detect_modality_node(state)
        
        assert result["detected_modality"] == "sar"
        assert result.get("resnet_classification_used") is True
        assert result.get("modality_confidence") == 0.92
    
    @pytest.mark.integration
    def test_modality_detection_service_unavailable(self, sample_agent_state):
        """Test when modality detection service is unavailable."""
        state: AgentState = sample_agent_state.copy()
        state["modality_detection_enabled"] = True
        
        with patch("app.orchestrator.is_modality_detection_available", return_value=False):
            result = detect_modality_node(state)
        
        # Should gracefully handle unavailability
        assert "execution_log" in result or result == {}


# =============================================================================
# Preprocessing Router Tests
# =============================================================================

class TestPreprocessingRouter:
    """Tests for preprocessing_router function."""
    
    @pytest.mark.integration
    def test_route_infrared_to_ir2rgb(self, sample_agent_state):
        """Test routing infrared images to IR2RGB."""
        state: AgentState = sample_agent_state.copy()
        state["detected_modality"] = "infrared"
        state["needs_ir2rgb"] = False  # Will be auto-enabled
        
        result = preprocessing_router(state)
        
        assert result == "preprocess_ir2rgb"
    
    @pytest.mark.integration
    def test_route_rgb_to_service(self, sample_agent_state):
        """Test routing RGB images directly to service."""
        state: AgentState = sample_agent_state.copy()
        state["detected_modality"] = "rgb"
        
        result = preprocessing_router(state)
        
        assert result == "route_to_service"
    
    @pytest.mark.integration
    def test_route_sar_to_service(self, sample_agent_state):
        """Test routing SAR images to service."""
        state: AgentState = sample_agent_state.copy()
        state["detected_modality"] = "sar"
        
        result = preprocessing_router(state)
        
        assert result == "route_to_service"
    
    @pytest.mark.integration
    def test_route_unknown_to_service(self, sample_agent_state):
        """Test routing unknown modality to service."""
        state: AgentState = sample_agent_state.copy()
        state["detected_modality"] = "unknown"
        
        result = preprocessing_router(state)
        
        assert result == "route_to_service"
    
    @pytest.mark.integration
    def test_route_with_manual_ir2rgb(self, sample_agent_state):
        """Test routing when IR2RGB is manually enabled."""
        state: AgentState = sample_agent_state.copy()
        state["detected_modality"] = "rgb"
        state["needs_ir2rgb"] = True  # Manually enabled
        
        result = preprocessing_router(state)
        
        assert result == "preprocess_ir2rgb"


# =============================================================================
# IR2RGB Preprocessing Node Tests
# =============================================================================

class TestPreprocessIR2RGBNode:
    """Tests for preprocess_ir2rgb_node."""
    
    @pytest.mark.integration
    def test_ir2rgb_preprocessing_success(self, sample_agent_state):
        """Test successful IR2RGB preprocessing."""
        state: AgentState = sample_agent_state.copy()
        state["image_url"] = "https://example.com/multispectral.tif"
        state["ir2rgb_channels"] = ["NIR", "R", "G"]
        state["ir2rgb_synthesize"] = "B"
        
        with patch("app.orchestrator.get_ir2rgb_service") as mock_service:
            mock_instance = MagicMock()
            mock_instance.convert_from_url.return_value = {
                "success": True,
                "rgb_image_base64": "data:image/png;base64,iVBORw0KGgo...",
                "format": "PNG",
                "dimensions": {"width": 100, "height": 100}
            }
            mock_service.return_value = mock_instance
            
            with patch("app.orchestrator.is_ir2rgb_available", return_value=True):
                result = preprocess_ir2rgb_node(state)
        
        assert "image_url" in result
        assert result["image_url"].startswith("data:image")
        assert result.get("original_image_url") == state["image_url"]
    
    @pytest.mark.integration
    def test_ir2rgb_preprocessing_service_unavailable(self, sample_agent_state):
        """Test when IR2RGB service is unavailable."""
        state: AgentState = sample_agent_state.copy()
        
        with patch("app.orchestrator.is_ir2rgb_available", return_value=False):
            result = preprocess_ir2rgb_node(state)
        
        # Should handle gracefully
        assert "execution_log" in result or result == {}
    
    @pytest.mark.integration
    def test_ir2rgb_preprocessing_failure(self, sample_agent_state):
        """Test IR2RGB conversion failure."""
        state: AgentState = sample_agent_state.copy()
        state["image_url"] = "https://example.com/test.tif"
        
        with patch("app.orchestrator.get_ir2rgb_service") as mock_service:
            mock_instance = MagicMock()
            mock_instance.convert_from_url.return_value = {
                "success": False,
                "error": "Invalid image format"
            }
            mock_service.return_value = mock_instance
            
            with patch("app.orchestrator.is_ir2rgb_available", return_value=True):
                result = preprocess_ir2rgb_node(state)
        
        # Should log error and continue
        assert "execution_log" in result


# =============================================================================
# Service Routing Node Tests
# =============================================================================

class TestRouteToServiceNode:
    """Tests for route_to_service_node."""
    
    @pytest.mark.integration
    def test_route_auto_mode(self, sample_agent_state):
        """Test routing in auto mode."""
        state: AgentState = sample_agent_state.copy()
        state["mode"] = "auto"
        state["user_query"] = "What is this?"
        
        with patch("app.orchestrator.ChatOpenAI") as mock_llm:
            mock_llm_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "VQA"
            mock_llm_instance.invoke.return_value = mock_response
            mock_llm.return_value = mock_llm_instance
            
            result = route_to_service_node(state)
        
        # Should route based on LLM decision
        assert "execution_log" in result
    
    @pytest.mark.integration
    def test_route_explicit_vqa(self, sample_agent_state):
        """Test explicit routing to VQA."""
        state: AgentState = sample_agent_state.copy()
        state["mode"] = "vqa"
        state["user_query"] = "Is there a building?"
        
        with patch("app.orchestrator.modal_client") as mock_modal:
            mock_modal.call_vqa.return_value = {
                "answer": "Yes, there is a building."
            }
            
            result = route_to_service_node(state)
        
        assert result.get("vqa_result") is not None
    
    @pytest.mark.integration
    def test_route_explicit_captioning(self, sample_agent_state):
        """Test explicit routing to captioning."""
        state: AgentState = sample_agent_state.copy()
        state["mode"] = "captioning"
        
        with patch("app.orchestrator.modal_client") as mock_modal:
            mock_modal.call_captioning.return_value = {
                "caption": "A satellite image showing urban areas."
            }
            
            result = route_to_service_node(state)
        
        assert result.get("caption_result") is not None
    
    @pytest.mark.integration
    def test_route_explicit_grounding(self, sample_agent_state):
        """Test explicit routing to grounding."""
        state: AgentState = sample_agent_state.copy()
        state["mode"] = "grounding"
        state["user_query"] = "Find buildings"
        
        with patch("app.orchestrator.modal_client") as mock_modal:
            mock_modal.call_grounding.return_value = {
                "bboxes": [
                    {"x": 10, "y": 20, "w": 50, "h": 50, "label": "building", "conf": 0.9}
                ]
            }
            
            result = route_to_service_node(state)
        
        assert result.get("grounding_result") is not None
        assert "bboxes" in result["grounding_result"]


# =============================================================================
# Workflow Integration Tests
# =============================================================================

class TestWorkflowIntegration:
    """Tests for complete workflow integration."""
    
    @pytest.mark.integration
    def test_complete_workflow_rgb_captioning(self, sample_agent_state):
        """Test complete workflow for RGB image with captioning."""
        state: AgentState = sample_agent_state.copy()
        state["mode"] = "captioning"
        state["image_url"] = "https://example.com/rgb.jpg"
        
        # Step 1: Detect modality
        with patch("app.orchestrator.get_modality_router_service") as mock_router:
            mock_router_instance = MagicMock()
            mock_router_instance.detect_modality_from_url.return_value = (
                "rgb",
                {"reason": "statistical"}
            )
            mock_router.return_value = mock_router_instance
            
            with patch("app.orchestrator.is_modality_detection_available", return_value=True):
                state.update(detect_modality_node(state))
        
        # Step 2: Route (should go to service)
        route = preprocessing_router(state)
        assert route == "route_to_service"
        
        # Step 3: Call service
        with patch("app.orchestrator.modal_client") as mock_modal:
            mock_modal.call_captioning.return_value = {
                "caption": "A satellite image."
            }
            
            state.update(route_to_service_node(state))
        
        assert state.get("caption_result") is not None
        assert state.get("detected_modality") == "rgb"
    
    @pytest.mark.integration
    def test_complete_workflow_infrared_with_ir2rgb(self, sample_agent_state):
        """Test complete workflow for infrared image with IR2RGB."""
        state: AgentState = sample_agent_state.copy()
        state["mode"] = "vqa"
        state["user_query"] = "What is this?"
        state["image_url"] = "https://example.com/infrared.tif"
        
        # Step 1: Detect modality
        with patch("app.orchestrator.get_modality_router_service") as mock_router:
            mock_router_instance = MagicMock()
            mock_router_instance.detect_modality_from_url.return_value = (
                "infrared",
                {"reason": "statistical"}
            )
            mock_router.return_value = mock_router_instance
            
            with patch("app.orchestrator.is_modality_detection_available", return_value=True):
                state.update(detect_modality_node(state))
        
        # Step 2: Route to IR2RGB
        route = preprocessing_router(state)
        assert route == "preprocess_ir2rgb"
        
        # Step 3: Preprocess
        state["ir2rgb_channels"] = ["NIR", "R", "G"]
        state["ir2rgb_synthesize"] = "B"
        
        with patch("app.orchestrator.get_ir2rgb_service") as mock_service:
            mock_instance = MagicMock()
            mock_instance.convert_from_url.return_value = {
                "success": True,
                "rgb_image_base64": "data:image/png;base64,..."
            }
            mock_service.return_value = mock_instance
            
            with patch("app.orchestrator.is_ir2rgb_available", return_value=True):
                state.update(preprocess_ir2rgb_node(state))
        
        # Step 4: Route to service
        route = preprocessing_router(state)
        assert route == "route_to_service"
        
        # Step 5: Call VQA
        with patch("app.orchestrator.modal_client") as mock_modal:
            mock_modal.call_vqa.return_value = {
                "answer": "This is a converted infrared image."
            }
            
            state.update(route_to_service_node(state))
        
        assert state.get("vqa_result") is not None
        assert state.get("original_image_url") is not None


