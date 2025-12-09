import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  

model.load_state_dict(torch.load("drone_classifier.pth", map_location="cpu"))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

classes = ["drone", "no_drone"]

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)  

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = classes[predicted.item()]
    confidence = confidence.item() * 100

    return label, confidence

if __name__ == "__main__":
    img_path = input("Enter image path: ")
    label, confidence = predict_image(img_path)
    print(f"Prediction: {label}  ({confidence:.2f}% confidence)")
