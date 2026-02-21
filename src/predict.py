import torch
from torchvision import transforms
from PIL import Image

from model import get_model
from utils import get_device


def predict(image_path, class_names):
    device = get_device()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model = get_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(torch.load("/Users/bimalkumal/AI Projects/Flower_Classifier/saved_models/best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    print(f"Prediction: {class_names[pred.item()]}")

class_names=['daisy', 'dandelion']
img_path="/Users/bimalkumal/Downloads/flower/train/dandelion/10683189_bd6e371b97_jpg.rf.213ffbff3870cabf5f0701a25c8a1b57.jpg"
predict(img_path,class_names)