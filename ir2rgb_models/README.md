# IR2RGB Models

This directory contains the model weights and core model implementation for IR2RGB conversion.

## Contents

- `ir2rgb_model.py` - Core IR2RGBModel class implementation
- `model_weights/models_ir_rgb.npz` - Pre-trained model weights (RPCC + LUT)

## Model Weights

The `models_ir_rgb.npz` file contains:
- `rpcc_R`: Root Polynomial Color Correction weights for R channel synthesis
- `rpcc_G`: Root Polynomial Color Correction weights for G channel synthesis  
- `lut_B`: 3D Look-Up Table for B channel synthesis

## Usage

This model is used by the `IR2RGBService` in the backend (`backend/app/services/ir2rgb_service.py`).

### Standalone Usage

```python
from ir2rgb_model import IR2RGBModel

# Load model
model = IR2RGBModel("model_weights/models_ir_rgb.npz")

# Convert image (synthesize B channel from NIR, R, G)
rgb_image = model.synthesize_B("path/to/image.jpg", channels=["NIR", "R", "G"])
rgb_image.save("output_rgb.png")
```

## Channel Mapping

The model expects 3-channel input images with the following channel orders:

- **For R synthesis**: `["NIR", "G", "B"]` - Synthesizes red channel
- **For G synthesis**: `["NIR", "R", "B"]` - Synthesizes green channel
- **For B synthesis**: `["NIR", "R", "G"]` - Synthesizes blue channel (most common)

## Integration with Backend

The model is integrated into the LangGraph orchestrator workflow:

1. **Preprocessing Node**: Converts multispectral images to RGB before routing to VLM services
2. **Standalone Endpoint**: `/api/orchestrator/ir2rgb` for direct conversion
3. **In-Memory Processing**: Model loads once and stays in memory for efficient processing

### Example API Request

```bash
curl -X POST http://localhost:8000/api/orchestrator/ir2rgb \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/multispectral.jpg",
    "channels": ["NIR", "R", "G"],
    "synthesize_channel": "B"
  }'
```

## Model Architecture

- **RPCC (Root Polynomial Color Correction)**: Used for R and G channel synthesis
  - Computes polynomial features from NIR + other channels
  - Matrix multiplication with pre-trained weights
  
- **3D LUT (Look-Up Table)**: Used for B channel synthesis
  - Trilinear interpolation in 3D color space
  - More accurate but computationally intensive

## License

ISC
