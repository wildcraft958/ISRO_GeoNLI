# app/api/v1/endpoints/upload.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
import uuid
from typing import Optional

from app.core.database import get_db
from app.db.models import ChatSession
from app.services.s3_service import s3_service
from app.services.classifier import predict_modality

router = APIRouter()

@router.post("/image/upload")
async def upload_image(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Step 0 Flow:
    1. Read File -> 2. S3 Upload -> 3. ResNet Classify -> 4. Save to DB
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing")

    # Generate unique ID
    chat_id = str(uuid.uuid4())
    
    try:
        # A. Read file content once
        # We need bytes for both S3 upload and Classification
        file_content = await file.read()
        
        # B. Classification (Async call to ResNet)
        # We do this first or in parallel. If this fails, we might not want to save the session.
        # Passing raw bytes to our service helper
        modality = await predict_modality(file_content)

        # C. Upload to S3
        # Reset file cursor for boto3 if we were passing the file object, 
        # but here we can pass the BytesIO or write bytes directly.
        # Since our s3_service expects a file-like object, we wrap the bytes.
        from io import BytesIO
        file_obj = BytesIO(file_content)
        
        file_ext = file.filename.split(".")[-1]
        s3_key = f"{user_id}/{chat_id}.{file_ext}"
        
        image_url = s3_service.upload_file(
            file_obj=file_obj, 
            object_name=s3_key, 
            content_type=file.content_type
        )
        
        # D. Save to Database
        new_session = ChatSession(
            chat_id=chat_id,
            user_id=user_id,
            image_url=image_url,
            image_type=modality, # Populated from ResNet
            summary_context=""
        )
        
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        
        return {
            "chat_id": chat_id,
            "image_url": image_url,
            "image_type": modality,
            "success": True
        }

    except Exception as e:
        # In production, specific error handling (400 vs 500) goes here
        raise HTTPException(status_code=500, detail=str(e))