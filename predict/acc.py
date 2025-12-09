import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


num_classes = 2  
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, num_classes) 


model.load_state_dict(torch.load('drone_classifier.pth', map_location='cpu'))
model.eval()


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

val_dataset = datasets.ImageFolder(
    r'C:\Users\LENOVO\Desktop\Project\dataset\val',
    transform=transform
)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Validation Accuracy: {100 * correct / total:.2f}%')
