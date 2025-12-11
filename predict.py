import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse
from torchvision.models import ResNet50_Weights


MODEL_PATH = "best_model.pt" 
CLASS_LIST = ["butterfly","cat","chicken","cow","dog","elephant","horse","sheep","spider","squirrel"] 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASS_LIST))

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict(model, image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return CLASS_LIST[pred.item()], conf.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image for prediction")
    args = parser.parse_args()

    print("Loading model...")
    model = load_model()

    print(f"Predicting: {args.image}")
    pred_class, confidence = predict(model, args.image)

    print(f"\nPrediction: {pred_class}")
    print(f"Confidence: {confidence * 100:.2f}%")
