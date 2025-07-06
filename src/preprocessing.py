import os
import shutil

def organize_images(src_dir="train"):
    cat_dir = os.path.join(src_dir, 'cats')
    dog_dir = os.path.join(src_dir, 'dogs')

    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(dog_dir, exist_ok=True)

    for fname in os.listdir(src_dir):
        if fname.startswith('cat') and fname.endswith('.jpg'):
            shutil.move(os.path.join(src_dir, fname), os.path.join(cat_dir, fname))
        elif fname.startswith('dog') and fname.endswith('.jpg'):
            shutil.move(os.path.join(src_dir, fname), os.path.join(dog_dir, fname))

    print("✅ Images organized into cats/ and dogs/ folders.")
