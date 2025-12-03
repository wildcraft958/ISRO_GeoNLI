"""
Test modality detection on all sample images.

This test runs the modality router on all images in the sample images folder
and displays the detection results for each.
"""

import pytest
from pathlib import Path
from PIL import Image

from app.services.modality_router import get_modality_router_service
from app.services.resnet_classifier_client import get_resnet_classifier_client, is_resnet_classifier_available


@pytest.mark.unit
def test_all_sample_images_modality_detection():
    """Test modality detection on all sample images."""
    # Path to sample images folder
    sample_images_dir = Path(__file__).parent.parent.parent.parent / "sample images"
    
    if not sample_images_dir.exists():
        pytest.skip(f"Sample images directory not found at {sample_images_dir}")
    
    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    image_files = [
        f for f in sample_images_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        pytest.skip(f"No image files found in {sample_images_dir}")
    
    # Get services
    router = get_modality_router_service()
    resnet_available = is_resnet_classifier_available()
    
    print("\n" + "="*80)
    print("BATCH MODALITY DETECTION RESULTS")
    print("="*80)
    print(f"Total images: {len(image_files)}")
    print(f"ResNet Classifier Available: {resnet_available}")
    print("="*80 + "\n")
    
    results = []
    
    for image_path in sorted(image_files):
        try:
            # Load image to get info
            pil_image = Image.open(image_path)
            image_info = {
                "size": pil_image.size,
                "mode": pil_image.mode,
                "format": pil_image.format,
                "channels": len(pil_image.getbands()) if hasattr(pil_image, 'getbands') else None
            }
            
            # Detect modality using statistical method
            modality, rgb_pil, diagnostics = router.route_image(
                str(image_path),
                metadata_priority=True
            )
            
            # Try ResNet if available and if SAR detected
            resnet_result = None
            if resnet_available and modality == "sar":
                try:
                    resnet_client = get_resnet_classifier_client()
                    # Convert to data URI for ResNet
                    from io import BytesIO
                    import base64
                    buffer = BytesIO()
                    pil_image.save(buffer, format="PNG")
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    data_uri = f"data:image/png;base64,{img_base64}"
                    
                    resnet_modality, resnet_confidence, resnet_probs = resnet_client.classify_from_url(data_uri)
                    resnet_result = {
                        "modality": resnet_modality,
                        "confidence": resnet_confidence,
                        "probabilities": resnet_probs
                    }
                except Exception as e:
                    resnet_result = {"error": str(e)}
            
            result = {
                "filename": image_path.name,
                "image_info": image_info,
                "modality": modality,
                "diagnostics": diagnostics,
                "resnet_result": resnet_result
            }
            results.append(result)
            
            # Print results for each image
            print(f"\n{'─'*80}")
            print(f"Image: {image_path.name}")
            print(f"{'─'*80}")
            print(f"Size: {image_info['size'][0]}x{image_info['size'][1]} | Mode: {image_info['mode']} | Format: {image_info['format']}")
            print(f"Detected Modality: {modality.upper()}")
            print(f"Reason: {diagnostics.get('reason', 'N/A')}")
            
            if 'cv' in diagnostics:
                print(f"CV (Coefficient of Variation): {diagnostics.get('cv', 'N/A'):.4f}" if isinstance(diagnostics.get('cv'), (int, float)) else f"CV: {diagnostics.get('cv', 'N/A')}")
            
            if resnet_result:
                if "error" in resnet_result:
                    print(f"ResNet: Error - {resnet_result['error']}")
                else:
                    print(f"ResNet Classification: {resnet_result['modality'].upper()} (confidence: {resnet_result['confidence']:.3f})")
                    print(f"ResNet Probabilities: {resnet_result['probabilities']}")
            
        except Exception as e:
            print(f"\n{'─'*80}")
            print(f"Image: {image_path.name} - ERROR: {e}")
            print(f"{'─'*80}")
            results.append({
                "filename": image_path.name,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    modality_counts = {}
    for result in results:
        if "error" not in result:
            mod = result["modality"]
            modality_counts[mod] = modality_counts.get(mod, 0) + 1
    
    print(f"\nModality Distribution:")
    for modality, count in sorted(modality_counts.items()):
        print(f"  {modality.upper()}: {count}")
    
    print(f"\nTotal Processed: {len([r for r in results if 'error' not in r])}/{len(results)}")
    print("="*80 + "\n")
    
    # Assertions
    assert len(results) > 0, "No images were processed"
    successful = [r for r in results if "error" not in r]
    assert len(successful) > 0, "No images were successfully processed"
    
    return results

