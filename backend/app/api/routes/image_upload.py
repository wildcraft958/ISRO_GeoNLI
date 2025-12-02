from app.core.s3 import AWS_S3_BUCKET_NAME, get_s3_client, s3_file_url
from botocore.exceptions import ClientError
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(prefix="/image", tags=["image"])


@router.post("/upload")
async def upload_file_to_s3(file: UploadFile = File(...)):
    s3 = get_s3_client()

    key = f"uploads/{file.filename}"

    try:
        s3.upload_fileobj(
            file.file,
            AWS_S3_BUCKET_NAME,
            key,
            ExtraArgs={"ContentType": file.content_type},
        )
    except ClientError as e:
        print("S3 upload error:", e)
        raise HTTPException(status_code=500, detail="Error uploading file to S3")

    return {"file_url": s3_file_url(key)}
