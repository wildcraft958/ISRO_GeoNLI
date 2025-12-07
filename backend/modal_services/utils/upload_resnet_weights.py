"""
Script to upload ResNet model weights to Modal Volume.

Usage:
    modal run upload_resnet_weights.py --weights-path /path/to/best_resnet.pth
    
Or with default path:
    modal run upload_resnet_weights.py
"""

import modal
import sys
from pathlib import Path

# Volume name must match the one in resnet_classifier.py
VOLUME_NAME = "resnet-modality-weights"
REMOTE_WEIGHTS_PATH = "best_resnet.pth"

app = modal.App("upload-resnet-weights")


@app.local_entrypoint()
def main(weights_path: str = None):
    """
    Upload ResNet weights to Modal Volume.
    
    Args:
        weights_path: Local path to the .pth file
                     (default: best_resnet.pth in the same directory as this script)
    """
    # Default to weights file in the same directory as this script
    if weights_path is None:
        script_dir = Path(__file__).parent
        weights_path = str(script_dir / "best_resnet.pth")
    
    local_path = Path(weights_path)
    
    if not local_path.exists():
        print(f"❌ Error: Weights file not found at {weights_path}")
        print("Please provide the correct path with --weights-path")
        sys.exit(1)
    
    print(f"📦 Uploading weights from: {weights_path}")
    print(f"📁 Target volume: {VOLUME_NAME}")
    print(f"📄 Remote path: /{REMOTE_WEIGHTS_PATH}")
    
    # Get or create volume
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    
    # Read the weights file
    with open(local_path, "rb") as f:
        weights_data = f.read()
    
    file_size_mb = len(weights_data) / (1024 * 1024)
    print(f"📊 File size: {file_size_mb:.2f} MB")
    
    # Upload to volume using batch_upload
    with vol.batch_upload() as batch:
        batch.put_file(local_path, REMOTE_WEIGHTS_PATH)
    
    print(f"✅ Successfully uploaded weights to volume '{VOLUME_NAME}'")
    print(f"   Remote path: /weights/{REMOTE_WEIGHTS_PATH}")
    print("\n🚀 You can now deploy the classifier with:")
    print("   modal deploy resnet_classifier.py")


# Alternative: Function-based upload for programmatic use
@app.function(
    volumes={"/weights": modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)},
    timeout=300,
)
def upload_weights_remote(weights_bytes: bytes, filename: str = "best_resnet.pth"):
    """
    Upload weights from within a Modal function.
    
    This can be called remotely if you have the weights as bytes.
    """
    import os
    
    target_path = f"/weights/{filename}"
    
    with open(target_path, "wb") as f:
        f.write(weights_bytes)
    
    # Commit the volume changes
    modal.Volume.from_name(VOLUME_NAME).commit()
    
    file_size = os.path.getsize(target_path)
    return {
        "success": True,
        "path": target_path,
        "size_bytes": file_size
    }


if __name__ == "__main__":
    print("Run this script with Modal:")
    print("  modal run upload_resnet_weights.py --weights-path /path/to/best_resnet.pth")

