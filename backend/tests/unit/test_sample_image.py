"""
Test modality detection on sample1.png image.

This test runs the actual modality router on the provided sample image
and displays the detection results.
"""

import pytest
from pathlib import Path
from PIL import Image

from app.services.modality_router import get_modality_router_service


@pytest.mark.unit
def test_sample1_image_modality_detection():
    """Test modality detection on sample1.png."""
    # Path to sample image (in project root)
    sample_image_path = Path(__file__).parent.parent.parent.parent / "sample1.png"
    
    if not sample_image_path.exists():
        pytest.skip(f"Sample image not found at {sample_image_path}")
    
    # Load image to get info
    pil_image = Image.open(sample_image_path)
    image_info = {
        "size": pil_image.size,
        "mode": pil_image.mode,
        "format": pil_image.format,
        "channels": len(pil_image.getbands()) if hasattr(pil_image, 'getbands') else None
    }
    
    # Get the modality router service
    router = get_modality_router_service()
    
    # Route image (accepts file path, PIL Image, or numpy array)
    modality, rgb_pil, diagnostics = router.route_image(
        str(sample_image_path),
        metadata_priority=True
    )
    
    # Print results for visibility
    print("\n" + "="*70)
    print("MODALITY DETECTION RESULTS FOR sample1.png")
    print("="*70)
    print(f"\nImage Information:")
    for key, value in image_info.items():
        print(f"  {key}: {value}")
    
    print(f"\nDetected Modality: {modality.upper()}")
    print(f"\nDiagnostics:")
    for key, value in diagnostics.items():
        if isinstance(value, (int, float, str, bool)):
            print(f"  {key}: {value}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float, str, bool)):
                    print(f"    {sub_key}: {sub_value}")
                elif isinstance(sub_value, dict):
                    print(f"    {sub_key}:")
                    for sub_sub_key, sub_sub_value in sub_value.items():
                        if isinstance(sub_sub_value, (int, float, str, bool)):
                            print(f"      {sub_sub_key}: {sub_sub_value}")
    print("="*70 + "\n")
    
    # Assert that we got a valid modality
    assert modality in ["rgb", "infrared", "sar", "unknown"]
    assert isinstance(diagnostics, dict)
    
    return modality, diagnostics

