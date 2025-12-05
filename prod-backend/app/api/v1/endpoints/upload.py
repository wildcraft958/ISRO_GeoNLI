from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
import uuid
from io import BytesIO

from app.core.database import get_db
from app.db.models import ChatSession
from app.services.s3_service import s3_service
from app.services.classifier import predict_modality
from app.services.fcc_converter import convert_fcc_to_rgb

router = APIRouter()

@router.post("/image/upload")
async def upload_image(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Step 0 Flow:
    1. Read File 
    2. ResNet Classify (SAR, IR, RGB)
    3. Logic Branch:
       - IR: Upload Original -> DB Type: 'IR'
       - SAR: Upload Original -> DB Type: 'SAR'
       - RGB: Upload Original -> DB Type: 'RGB'
       - Other: Upload Original -> FCC -> Convert to RGB -> DB Type: 'RGB'
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing")

    chat_id = str(uuid.uuid4())
    
    try:
        # A. Read content
        original_bytes = await file.read()
        
        # B. Classify
        raw_modality = await predict_modality(original_bytes)
        
        # C. Process based on Modality
        final_image_bytes = original_bytes
        final_modality = raw_modality
        
        if raw_modality == "FCC":
            # CASE 1: FCC -> Convert to RGB
            # We "continue as currently deployed pipeline is" (treat as RGB)
            final_image_bytes = convert_fcc_to_rgb(original_bytes)
            final_modality = "RGB" 
            
        elif raw_modality == "SAR":
            # CASE 2: SAR -> Keep as SAR
            final_modality = "SAR"
            
        elif raw_modality == "RGB":
            # CASE 3: RGB -> Keep as RGB
            final_modality = "RGB"
            
        elif raw_modality == "IR":
            # CASE 4: IR -> Keep as IR
            final_modality = "IR"

        # D. Upload the (potentially converted) image to S3
        file_obj = BytesIO(final_image_bytes)
        file_ext = file.filename.split(".")[-1]
        s3_key = f"{user_id}/{chat_id}.{file_ext}"
        
        image_url = s3_service.upload_file(
            file_obj=file_obj, 
            object_name=s3_key, 
            content_type=file.content_type
        )
        
        # E. Save to DB
        new_session = ChatSession(
            chat_id=chat_id,
            user_id=user_id,
            image_url=image_url,
            image_type=final_modality, # Will be RGB (converted), SAR, or IR
            summary_context=""
        )
        
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        
        return {
            "chat_id": chat_id,
            "image_url": image_url,
            "image_type": final_modality,
            "success": True
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))