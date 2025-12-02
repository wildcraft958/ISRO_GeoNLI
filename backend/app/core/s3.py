import os
from typing import Any

import boto3
from botocore.client import Config

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


def get_s3_client() -> Any:
    if not AWS_S3_BUCKET_NAME or not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        raise RuntimeError("AWS S3 configuration missing in environment variables")

    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )


def s3_file_url(key: str) -> str:
    """Public S3 URL of the uploaded file"""
    if not AWS_S3_BUCKET_NAME:
        raise RuntimeError("Missing AWS_S3_BUCKET_NAME environment variable")

    return f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"
