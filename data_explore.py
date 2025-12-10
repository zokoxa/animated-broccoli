import os

root_dir = "data/animals"

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

folder_counts = {}

for folder, subfolders, files in os.walk(root_dir):
    count = 0
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in image_extensions:
            count += 1
    
    
    folder_counts[folder] = count


for folder, count in folder_counts.items():
    print(f"{folder}: {count} images")

"""
data/animals/squirrel: 1862 images
data/animals/elephant: 1446 images
data/animals/dog: 4863 images
data/animals/spider: 4821 images
data/animals/cow: 1866 images
data/animals/cat: 1668 images
data/animals/butterfly: 2112 images
data/animals/sheep: 1820 images
data/animals/horse: 2623 images
data/animals/chicken: 3098 images
"""