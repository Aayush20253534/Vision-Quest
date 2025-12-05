import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  

model.load_state_dict(torch.load("drone classifier.pth", map_location="cpu"))
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

def run_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open webcam!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        img_tensor = preprocess(pil_img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        label = classes[predicted.item()]
        conf_percent = confidence.item() * 100

        color = (0, 255, 0) if label == "drone" else (0, 0, 255)

        frame = cv2.flip(frame, 1)

        text = f"{label.upper()}  ({conf_percent:.2f}%)"
        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        bar_x, bar_y = 20, 70
        bar_width = 250
        filled = int(bar_width * (conf_percent / 100))

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (180, 180, 180), 2)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 20), color, -1)
        cv2.imshow("Drone Detector - Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose mode:")
    print("1 = Predict an image")
    print("2 = Webcam detection")

    mode = input("Enter 1 or 2: ")

    if mode == "1":
        img_path = input("Enter image path: ")
        label, confidence = predict_image(img_path)
        print(f"Prediction: {label}  ({confidence:.2f}% confidence)")

    elif mode == "2":
        run_webcam()

    else:
        print("Invalid input")
