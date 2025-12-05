from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import httpx
import uuid
import logging
from typing import Any

from app.core.database import get_db
from app.db.models import ChatSession
from app.schemas.test_structure import TestPipelineRequest, TestPipelineResponse
from app.services.classifier import predict_modality
from app.services.fcc_converter import convert_fcc_to_rgb
from app.services.orchestrator import chat_app

router = APIRouter()
logger = logging.getLogger(__name__)

# Helper to run a single query through the Orchestrator Graph
async def run_query(chat_id: str, user_id: str, text: str, mode: str, session_data: ChatSession) -> Any:
    """
    Runs the query through the standard Orchestrator pipeline.
    The Orchestrator internally handles routing for SAR/IR via determine_mode_node.
    """
    state = {
        "chat_id": chat_id,
        "user_id": user_id,
        "query_text": text,
        "requested_mode": mode,
        "image_url": session_data.image_url,
        "image_type": session_data.image_type,
        "summary_context": session_data.summary_context or ""
    }
    
    # Invoke the Graph
    # determine_mode_node will automatically switch mode to SAR_DIRECT or IR_DIRECT 
    # if image_type is SAR or IR, regardless of 'requested_mode'.
    result = await chat_app.ainvoke(state)
    
    if result.get("response_metadata"):
        return result["response_metadata"]
    return result["response_text"]

@router.post("/test", response_model=TestPipelineResponse, summary="End-to-End Pipeline Test")
async def test_pipeline(payload: TestPipelineRequest, db: Session = Depends(get_db)):
    """
    Parses the generic test JSON, runs classification, creates a session, 
    and executes all queries using the standard Orchestrator pipeline.
    """
    # 1. Setup
    user_id = "test_runner"
    chat_id = str(uuid.uuid4())
    img_url = payload.input_image.image_url
    
    logger.info(f"Starting Test Pipeline for Image: {img_url}")

    # 2. STEP 0: Download & Classify
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(img_url)
            resp.raise_for_status()
            image_bytes = resp.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {e}")

    # Run Classification
    try:
        raw_modality = await predict_modality(image_bytes)
        
        final_modality = raw_modality
        final_image_url = img_url 

        if raw_modality == "FCC":
            # FCC -> Convert to RGB -> Treat as RGB (Orchestrator handles as RGB)
            _ = await convert_fcc_to_rgb(image_bytes)
            # In production, upload converted bytes & update URL here
            final_modality = "RGB"
            
        elif raw_modality == "IR":
            # IR -> Keep as IR (Orchestrator will route to IR_DIRECT)
            final_modality = "IR" 
            
        elif raw_modality == "SAR":
            # SAR -> Keep as SAR (Orchestrator will route to SAR_DIRECT)
            final_modality = "SAR"
            
        else:
            # RGB -> Keep as RGB
            final_modality = "RGB"

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail="Pipeline Step 0 Failed")

    # Create Session in DB
    session = ChatSession(
        chat_id=chat_id,
        user_id=user_id,
        image_url=final_image_url,
        image_type=final_modality,
        summary_context=""
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    # 3. STEP 1: Run Batch Queries
    # We call run_query for each item. The Orchestrator handles the logic.
    
    # A. Captioning
    cap_res = await run_query(chat_id, user_id, payload.queries.caption_query.instruction, "CAPTIONING", session)
    
    # B. Grounding
    gnd_res = await run_query(chat_id, user_id, payload.queries.grounding_query.instruction, "GROUNDING", session)
    
    # C. Attributes (VQA)
    # Passing "VQA" allows the RGB pipeline to use the VQA sub-classifier.
    # If image_type is SAR/IR, the Orchestrator ignores "VQA" and uses SAR_DIRECT/IR_DIRECT logic.
    bin_res = await run_query(chat_id, user_id, payload.queries.attribute_query.binary.instruction, "VQA", session)
    num_res = await run_query(chat_id, user_id, payload.queries.attribute_query.numeric.instruction, "VQA", session)
    sem_res = await run_query(chat_id, user_id, payload.queries.attribute_query.semantic.instruction, "VQA", session)

    # 4. Construct Response
    return TestPipelineResponse(
        input_image=payload.input_image,
        queries={
            "caption_query": {
                "instruction": payload.queries.caption_query.instruction,
                "response": str(cap_res)
            },
            "grounding_query": {
                "instruction": payload.queries.grounding_query.instruction,
                "response": gnd_res 
            },
            "attribute_query": {
                "binary": {
                    "instruction": payload.queries.attribute_query.binary.instruction,
                    "response": str(bin_res)
                },
                "numeric": {
                    "instruction": payload.queries.attribute_query.numeric.instruction,
                    "response": str(num_res)
                },
                "semantic": {
                    "instruction": payload.queries.attribute_query.semantic.instruction,
                    "response": str(sem_res)
                }
            }
        }
    )