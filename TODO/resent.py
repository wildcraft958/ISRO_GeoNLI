import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

def load_resnet_model(weights_path, num_classes=3, device="cuda"):
    # Create the model architecture
    model = models.resnet50(weights=None)   # must match training setup
   
    # Replace classifier head
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load trained weights
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.to(device)
    model.eval()
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "/root/best_resnet.pth"

model = load_resnet_model(model_path, num_classes=3, device=device)

print("Loaded model:", model)

from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class_names = ["rgb", "infrared", "sar"]  # your labels

def predict_image(model, img_path):
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()

    return class_names[pred]

prediction = predict_image(model, "/root/images.jpg")
print("Prediction:", prediction)