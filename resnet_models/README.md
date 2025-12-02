# ResNet Modality Classifier Models

This directory contains the model weights and core model implementation for ResNet50-based image modality classification.

## Contents

- `resnet_model.py` - Core ResNetModalityClassifier class implementation
- `model_weights/best_resnet.pth` - Pre-trained ResNet50 weights (90 MB)

## Model Weights

The `best_resnet.pth` file contains:
- ResNet50 backbone (ImageNet-pretrained, weights=None)
- Custom classifier head (3 classes: rgb, infrared, sar)
- Trained weights for satellite image modality classification

## Usage

This model is used by the `ResNetClassifierClient` in the backend (`backend/app/services/resnet_classifier_client.py`) for remote classification via Modal service.

### Standalone Usage

```python
from resnet_model import ResNetModalityClassifier

# Load model
model = ResNetModalityClassifier("model_weights/best_resnet.pth", device="cpu")

# Classify image
modality, confidence, probabilities = model.predict("path/to/image.jpg")
print(f"Modality: {modality}, Confidence: {confidence:.3f}")
print(f"Probabilities: {probabilities}")
```

### Batch Prediction

```python
images = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = model.predict_batch(images)

for img_path, (modality, confidence, probs) in zip(images, results):
    print(f"{img_path}: {modality} ({confidence:.3f})")
```

## Integration with Backend

The model is integrated into the LangGraph orchestrator workflow:

1. **Modality Detection Node**: Uses ResNet as fallback when statistical detection is uncertain (especially for SAR)
2. **Two-Stage Detection**: 
   - First: Fast statistical detection (local)
   - Second: ResNet confirmation for SAR or uncertain cases (Modal service)
3. **Remote Deployment**: Model runs on Modal (CPU) for cost efficiency

### Workflow Integration

```
detect_modality_node
  │
  ├─ Statistical Detection (fast, local)
  │
  └─ If SAR or uncertain → ResNet Classifier (Modal)
     │
     └─ Returns: modality, confidence, probabilities
```

## Model Architecture

- **Backbone**: ResNet50 (ImageNet architecture)
- **Classifier Head**: Linear layer (2048 → 3 classes)
- **Input**: RGB images, 224x224 pixels
- **Output**: 3-class classification (rgb, infrared, sar)
- **Normalization**: ImageNet mean/std normalization

## Preprocessing

Images are preprocessed with:
1. Resize to 224x224
2. Convert to tensor
3. Normalize with ImageNet statistics:
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

## Modal Deployment

The model is deployed on Modal as a CPU service:
- **Service**: `resnet-modality-classifier`
- **Endpoint**: `https://maximuspookus--resnet-modality-classifier-serve.modal.run`
- **Volume**: `resnet-modality-weights` (contains best_resnet.pth)

### Deployment Commands

```bash
# Upload weights to Modal Volume
cd modal_services
modal run upload_resnet_weights.py --weights-path ../resnet_models/model_weights/best_resnet.pth

# Deploy service
modal deploy resnet_classifier.py
```

## Performance

- **Inference Time**: ~100-200ms per image (CPU)
- **Accuracy**: High accuracy for SAR detection (primary use case)
- **Confidence Thresholds**: Typically >0.7 for reliable predictions

## License

ISC

