"""
Unit tests for Pydantic schemas.

Tests cover:
- ChatRequest: Request validation
- ChatResponse: Response structure
- IR2RGBRequest/Response: IR2RGB endpoint models
- AgentState: Workflow state structure
"""

from typing import Dict, List

import pytest
from pydantic import ValidationError

from app.schemas.orchestrator_schema import (
    AgentState,
    ChatRequest,
    ChatResponse,
    IR2RGBRequest,
    IR2RGBResponse,
)


# =============================================================================
# ChatRequest Tests
# =============================================================================

class TestChatRequest:
    """Tests for ChatRequest schema."""
    
    @pytest.mark.unit
    def test_minimal_valid_request(self):
        """Test request with only required fields."""
        request = ChatRequest(image_url="https://example.com/test.jpg")
        
        assert request.image_url == "https://example.com/test.jpg"
        assert request.mode == "auto"  # Default
        assert request.modality_detection_enabled is True  # Default
    
    @pytest.mark.unit
    def test_full_valid_request(self, sample_chat_request):
        """Test request with all fields."""
        request = ChatRequest(**sample_chat_request)
        
        assert request.image_url == sample_chat_request["image_url"]
        assert request.query == sample_chat_request["query"]
        assert request.mode == sample_chat_request["mode"]
        assert request.session_id == sample_chat_request["session_id"]
        assert request.user_id == sample_chat_request["user_id"]
    
    @pytest.mark.unit
    def test_missing_required_field(self):
        """Test validation fails without image_url."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest()
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("image_url",) for e in errors)
    
    @pytest.mark.unit
    def test_invalid_mode(self):
        """Test validation fails with invalid mode."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(image_url="https://example.com/test.jpg", mode="invalid")
        
        errors = exc_info.value.errors()
        assert any("mode" in str(e["loc"]) for e in errors)
    
    @pytest.mark.unit
    def test_valid_modes(self):
        """Test all valid modes."""
        valid_modes = ["auto", "grounding", "vqa", "captioning"]
        
        for mode in valid_modes:
            request = ChatRequest(
                image_url="https://example.com/test.jpg",
                mode=mode
            )
            assert request.mode == mode
    
    @pytest.mark.unit
    def test_ir2rgb_fields(self):
        """Test IR2RGB related fields."""
        request = ChatRequest(
            image_url="https://example.com/test.jpg",
            needs_ir2rgb=True,
            ir2rgb_channels=["NIR", "R", "G"],
            ir2rgb_synthesize="B"
        )
        
        assert request.needs_ir2rgb is True
        assert request.ir2rgb_channels == ["NIR", "R", "G"]
        assert request.ir2rgb_synthesize == "B"
    
    @pytest.mark.unit
    def test_ir2rgb_synthesize_options(self):
        """Test valid synthesize channel options."""
        for channel in ["R", "G", "B"]:
            request = ChatRequest(
                image_url="https://example.com/test.jpg",
                ir2rgb_synthesize=channel
            )
            assert request.ir2rgb_synthesize == channel
    
    @pytest.mark.unit
    def test_modality_detection_toggle(self):
        """Test modality detection can be disabled."""
        request = ChatRequest(
            image_url="https://example.com/test.jpg",
            modality_detection_enabled=False
        )
        
        assert request.modality_detection_enabled is False


# =============================================================================
# ChatResponse Tests
# =============================================================================

class TestChatResponse:
    """Tests for ChatResponse schema."""
    
    @pytest.mark.unit
    def test_minimal_valid_response(self):
        """Test response with required fields."""
        response = ChatResponse(
            session_id="test-session",
            response_type="caption",
            content="A test image",
            execution_log=[]
        )
        
        assert response.session_id == "test-session"
        assert response.response_type == "caption"
        assert response.content == "A test image"
    
    @pytest.mark.unit
    def test_response_with_modality(self):
        """Test response with modality detection results."""
        response = ChatResponse(
            session_id="test-session",
            response_type="answer",
            content="The image shows a building",
            execution_log=["Step 1", "Step 2"],
            detected_modality="rgb",
            modality_confidence=0.95,
            resnet_classification_used=False
        )
        
        assert response.detected_modality == "rgb"
        assert response.modality_confidence == 0.95
        assert response.resnet_classification_used is False
    
    @pytest.mark.unit
    def test_response_with_ir2rgb(self):
        """Test response with IR2RGB conversion results."""
        response = ChatResponse(
            session_id="test-session",
            response_type="caption",
            content="Converted image",
            execution_log=[],
            converted_image_url="data:image/png;base64,...",
            original_image_url="https://example.com/original.tif"
        )
        
        assert response.converted_image_url is not None
        assert response.original_image_url is not None
    
    @pytest.mark.unit
    def test_response_types(self):
        """Test different response types."""
        # Caption response
        response = ChatResponse(
            session_id="test",
            response_type="caption",
            content="A satellite image",
            execution_log=[]
        )
        assert response.response_type == "caption"
        
        # VQA response
        response = ChatResponse(
            session_id="test",
            response_type="answer",
            content="Yes, there is a building",
            execution_log=[]
        )
        assert response.response_type == "answer"
        
        # Grounding response
        response = ChatResponse(
            session_id="test",
            response_type="boxes",
            content={"bboxes": [{"x": 10, "y": 20, "w": 50, "h": 50}]},
            execution_log=[]
        )
        assert response.response_type == "boxes"
    
    @pytest.mark.unit
    def test_execution_log(self):
        """Test execution log field."""
        log = ["Modality detected: rgb", "Routed to VQA", "Response generated"]
        
        response = ChatResponse(
            session_id="test",
            response_type="answer",
            content="Test",
            execution_log=log
        )
        
        assert len(response.execution_log) == 3
        assert "Modality detected: rgb" in response.execution_log


# =============================================================================
# IR2RGBRequest Tests
# =============================================================================

class TestIR2RGBRequest:
    """Tests for IR2RGBRequest schema."""
    
    @pytest.mark.unit
    def test_valid_request(self, sample_ir2rgb_request):
        """Test valid IR2RGB request."""
        request = IR2RGBRequest(**sample_ir2rgb_request)
        
        assert request.image_url == sample_ir2rgb_request["image_url"]
        assert request.channels == sample_ir2rgb_request["channels"]
        assert request.synthesize_channel == sample_ir2rgb_request["synthesize_channel"]
    
    @pytest.mark.unit
    def test_default_synthesize_channel(self):
        """Test default synthesize channel is B."""
        request = IR2RGBRequest(
            image_url="https://example.com/test.tif",
            channels=["NIR", "R", "G"]
        )
        
        assert request.synthesize_channel == "B"
    
    @pytest.mark.unit
    def test_channels_validation_length(self):
        """Test channels must have exactly 3 elements."""
        with pytest.raises(ValidationError):
            IR2RGBRequest(
                image_url="https://example.com/test.tif",
                channels=["NIR", "R"]  # Only 2 channels
            )
        
        with pytest.raises(ValidationError):
            IR2RGBRequest(
                image_url="https://example.com/test.tif",
                channels=["NIR", "R", "G", "B"]  # 4 channels
            )
    
    @pytest.mark.unit
    def test_valid_channel_combinations(self):
        """Test various valid channel combinations."""
        combinations = [
            ["NIR", "R", "G"],
            ["NIR", "G", "B"],
            ["NIR", "R", "B"],
            ["R", "G", "B"],
        ]
        
        for channels in combinations:
            request = IR2RGBRequest(
                image_url="https://example.com/test.tif",
                channels=channels
            )
            assert request.channels == channels


# =============================================================================
# IR2RGBResponse Tests
# =============================================================================

class TestIR2RGBResponse:
    """Tests for IR2RGBResponse schema."""
    
    @pytest.mark.unit
    def test_success_response(self):
        """Test successful IR2RGB response."""
        response = IR2RGBResponse(
            success=True,
            rgb_image_url="data:image/png;base64,...",
            rgb_image_base64="iVBORw0KGgo...",
            format="PNG",
            dimensions={"width": 100, "height": 100},
            size_bytes=12345
        )
        
        assert response.success is True
        assert response.rgb_image_url is not None
        assert response.format == "PNG"
    
    @pytest.mark.unit
    def test_failure_response(self):
        """Test failed IR2RGB response."""
        response = IR2RGBResponse(
            success=False,
            error="Invalid image format"
        )
        
        assert response.success is False
        assert response.error == "Invalid image format"
        assert response.rgb_image_url is None
    
    @pytest.mark.unit
    def test_dimensions_structure(self):
        """Test dimensions field structure."""
        response = IR2RGBResponse(
            success=True,
            dimensions={"width": 1024, "height": 768}
        )
        
        assert response.dimensions["width"] == 1024
        assert response.dimensions["height"] == 768


# =============================================================================
# AgentState Tests
# =============================================================================

class TestAgentState:
    """Tests for AgentState TypedDict."""
    
    @pytest.mark.unit
    def test_valid_state_structure(self, sample_agent_state):
        """Test valid agent state structure."""
        state: AgentState = sample_agent_state
        
        assert state["session_id"] == "test-session-123"
        assert state["image_url"] == "https://example.com/test.jpg"
        assert state["mode"] == "auto"
    
    @pytest.mark.unit
    def test_modality_fields(self, sample_agent_state):
        """Test modality-related fields."""
        state: AgentState = sample_agent_state
        
        assert "modality_detection_enabled" in state
        assert "detected_modality" in state
        assert "modality_confidence" in state
        assert "modality_diagnostics" in state
        assert "resnet_classification_used" in state
    
    @pytest.mark.unit
    def test_ir2rgb_fields(self, sample_agent_state):
        """Test IR2RGB-related fields."""
        state: AgentState = sample_agent_state
        
        assert "needs_ir2rgb" in state
        assert "ir2rgb_channels" in state
        assert "ir2rgb_synthesize" in state
        assert "original_image_url" in state
    
    @pytest.mark.unit
    def test_service_result_fields(self, sample_agent_state):
        """Test service result fields."""
        state: AgentState = sample_agent_state
        
        assert "caption_result" in state
        assert "vqa_result" in state
        assert "grounding_result" in state
    
    @pytest.mark.unit
    def test_session_memory_fields(self, sample_agent_state):
        """Test session memory fields."""
        state: AgentState = sample_agent_state
        
        assert "messages" in state
        assert "session_context" in state
        assert isinstance(state["messages"], list)
        assert isinstance(state["session_context"], dict)
    
    @pytest.mark.unit
    def test_execution_log_field(self, sample_agent_state):
        """Test execution log field."""
        state: AgentState = sample_agent_state
        
        assert "execution_log" in state
        assert isinstance(state["execution_log"], list)
    
    @pytest.mark.unit
    def test_state_modification(self, sample_agent_state):
        """Test state can be modified."""
        state: AgentState = sample_agent_state
        
        state["detected_modality"] = "rgb"
        state["modality_confidence"] = 0.95
        state["execution_log"].append("Test step")
        
        assert state["detected_modality"] == "rgb"
        assert state["modality_confidence"] == 0.95
        assert "Test step" in state["execution_log"]


# =============================================================================
# Schema Serialization Tests
# =============================================================================

class TestSchemaSerialization:
    """Tests for schema serialization."""
    
    @pytest.mark.unit
    def test_chat_request_to_dict(self, sample_chat_request):
        """Test ChatRequest can be serialized to dict."""
        request = ChatRequest(**sample_chat_request)
        data = request.model_dump()
        
        assert isinstance(data, dict)
        assert data["image_url"] == sample_chat_request["image_url"]
    
    @pytest.mark.unit
    def test_chat_response_to_dict(self):
        """Test ChatResponse can be serialized to dict."""
        response = ChatResponse(
            session_id="test",
            response_type="caption",
            content="Test content",
            execution_log=["Step 1"]
        )
        data = response.model_dump()
        
        assert isinstance(data, dict)
        assert data["session_id"] == "test"
    
    @pytest.mark.unit
    def test_chat_request_from_json(self):
        """Test ChatRequest can be created from JSON-like dict."""
        json_data = {
            "image_url": "https://example.com/test.jpg",
            "query": "What is this?",
            "mode": "vqa"
        }
        
        request = ChatRequest.model_validate(json_data)
        
        assert request.image_url == json_data["image_url"]
        assert request.mode == "vqa"
    
    @pytest.mark.unit
    def test_response_json_serialization(self):
        """Test response can be serialized to JSON string."""
        response = ChatResponse(
            session_id="test",
            response_type="caption",
            content="Test",
            execution_log=[]
        )
        
        json_str = response.model_dump_json()
        
        assert isinstance(json_str, str)
        assert "test" in json_str


