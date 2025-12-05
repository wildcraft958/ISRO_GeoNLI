# app/services/s3_service.py
import boto3
from botocore.exceptions import ClientError
from app.core.config import settings
import logging

# Configure Logger
logger = logging.getLogger(__name__)

class S3Service:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket = settings.S3_BUCKET_NAME

    def upload_file(self, file_obj, object_name, content_type):
        """
        Uploads a file-like object to S3 and returns the public URL.
        """
        try:
            self.s3_client.upload_fileobj(
                file_obj,
                self.bucket,
                object_name,
                ExtraArgs={"ContentType": content_type}
            )
            
            # Construct URL (Virtual-hosted-style access)
            url = f"https://{self.bucket}.s3.{settings.AWS_REGION}.amazonaws.com/{object_name}"
            return url
            
        except ClientError as e:
            logger.error(f"S3 Upload Error: {e}")
            raise e

# Instantiate as singleton if stateless
s3_service = S3Service()