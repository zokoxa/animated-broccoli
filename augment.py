import os
import random
from PIL import Image
from torchvision import transforms


folder = "data/animals/chicken"
target_count = 4863


valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

original_images = [
    f for f in os.listdir(folder)
    if os.path.splitext(f)[1].lower() in valid_ext
]

current_count = len(original_images)
print(f"Current images: {current_count}")

if current_count >= target_count:
    print("Folder already has enough images. No augmentation needed.")
    exit()

augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.ToPILImage()
])

while current_count < target_count:

    img_name = random.choice(original_images)
    img_path = os.path.join(folder, img_name)

    
    img = Image.open(img_path).convert("RGB")

    
    aug_img = augment(img)

   
    out_name = f"augmented_{current_count}.jpg"
    out_path = os.path.join(folder, out_name)

    aug_img.save(out_path)

    current_count += 1
    print(f"Saved: {out_name}  ({current_count}/{target_count})")

print("Done! Folder now contains 4863 images.")
