import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models import ResNet50_Weights


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

IMAGE_SIZE = config["data"]["image_size"]
BATCH_SIZE = config["training"]["batch_size"]
DATA_DIR = config["data"]["data_dir"]
NUM_CLASSES = config["data"]["num_classes"]
EPOCHS = config["training"]["epochs"]
LR = config["optimizer"]["learning_rate"]
CKPT_DIR = config["checkpoint"]["dir"]

os.makedirs(CKPT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


aug_cfg = config["augmentations"]

train_transforms = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ColorJitter(brightness=.5,hue=.3),
    transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
]

train_transform = transforms.Compose(train_transforms)


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)

train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

val_dataset.dataset.transform = eval_transform
test_dataset.dataset.transform = eval_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


if config["model"]["type"] == "resnet50":
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
else:
    raise ValueError(f"Unsupported model: {config['model']['type']}")

for param in model.parameters():
    param.requires_grad = False


model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


writer = SummaryWriter(log_dir="logs")

best_val_loss = float("inf")

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    train_losses=[]
    train_accs=[]
    validation_losses=[]
    validation_accs=[]
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / total
    train_acc = correct / total
    val_loss, val_acc = evaluate(model, val_loader)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    validation_losses.append(val_loss)
    validation_accs.append(val_acc)
    print(
        f"Epoch {epoch}/{EPOCHS} "
        f"- Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
        f"- Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
    )
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_model.pt"))
        print("Saved new best model")

writer.close()

test_loss, test_acc = evaluate(model, test_loader)
print(f"\nFinal Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")