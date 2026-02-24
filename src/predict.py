import torch
import torch.nn.functional as F
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

        probs=F.softmax(outputs,dim=1)
        confidence,pred=torch.max(probs,1)
        confidence=confidence.item()
        pred_class=class_names[pred.item()]
    threshold=0.9
    if confidence<threshold:
        print(f"Prediction : unknown (Confidence:{confidence:.2f})")
    else:
        print(f"Prediction:{pred_class}(Confidence:{confidence:.2f})")

class_names=['daisy', 'dandelion']
img_path='/Users/bimalkumal/Desktop/Daisy_flower.jpg'
predict(img_path,class_names)