"""
ResNet50 Model for Image Modality Classification.

This module provides the core ResNetModalityClassifier class that classifies
satellite images into three modalities: RGB, infrared, and SAR.

The model is used by the ResNetClassifierClient in the backend for remote
classification via Modal service, and can also be used locally.

Usage:
    from resnet_model import ResNetModalityClassifier
    
    model = ResNetModalityClassifier("model_weights/best_resnet.pth", device="cpu")
    modality, confidence = model.predict("path/to/image.jpg")
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Union, Literal, Optional

# Type alias for modality
Modality = Literal["rgb", "infrared", "sar"]

CLASS_NAMES = ["rgb", "infrared", "sar"]
NUM_CLASSES = 3


class ResNetModalityClassifier:
    """
    ResNet50-based classifier for image modality detection.
    
    Classifies satellite images into:
    - 'rgb': Standard RGB images
    - 'infrared': Multispectral/NIR images
    - 'sar': Synthetic Aperture Radar images
    """
    
    def __init__(
        self, 
        weights_path: Union[str, Path], 
        num_classes: int = NUM_CLASSES,
        device: Optional[str] = None
    ):
        """
        Initialize ResNet50 classifier.
        
        Args:
            weights_path: Path to the .pth file containing trained weights
            num_classes: Number of output classes (default: 3)
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        
        # Create model architecture
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Load trained weights
        self.model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
        
        self.model.to(device)
        self.model.eval()
        
        # Image transforms (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(
        self, 
        image_input: Union[str, Path, Image.Image]
    ) -> Tuple[Modality, float, Dict[str, float]]:
        """
        Predict image modality.
        
        Args:
            image_input: Image path, Path object, or PIL Image
            
        Returns:
            Tuple of (modality, confidence, probabilities_dict)
        """
        # Load and preprocess image
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input)).convert('RGB')
        else:
            img = image_input.convert('RGB')
        
        x = self.transform(img).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = probs.max(dim=1)
        
        modality = CLASS_NAMES[pred_idx.item()]
        confidence_val = float(confidence.item())
        
        # Get probabilities for all classes
        probabilities = {
            name: float(probs[0, i].item())
            for i, name in enumerate(CLASS_NAMES)
        }
        
        return modality, confidence_val, probabilities
    
    def predict_batch(
        self,
        image_inputs: list[Union[str, Path, Image.Image]]
    ) -> list[Tuple[Modality, float, Dict[str, float]]]:
        """
        Predict modality for multiple images.
        
        Args:
            image_inputs: List of image paths, Path objects, or PIL Images
            
        Returns:
            List of (modality, confidence, probabilities_dict) tuples
        """
        results = []
        for img_input in image_inputs:
            results.append(self.predict(img_input))
        return results


if __name__ == "__main__":
    # Example usage
    model_path = "model_weights/best_resnet.pth"
    model = ResNetModalityClassifier(model_path, device="cpu")
    
    # Test prediction
    test_image = "path/to/test_image.jpg"  # Update with actual path
    
    try:
        modality, confidence, probabilities = model.predict(test_image)
        print(f"Predicted modality: {modality}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Probabilities: {probabilities}")
    except FileNotFoundError:
        print(f"Test image not found at {test_image}")
        print("Model loaded successfully. Ready for predictions.")

