import os
import shutil
import kagglehub
from glob import glob
import time

# Download location (KaggleHub cache)
original_cache_dir = "/media/adnan/Datasets/KaggleHub"
os.environ["KAGGLEHUB_CACHE"] = original_cache_dir
os.makedirs(original_cache_dir, exist_ok=True)

# Where to copy all images (flattened)
full_datasets_dir = "datasets/full_datasets"
os.makedirs(full_datasets_dir, exist_ok=True)

datasets = [
    "jessicali9530/lfw-dataset", # 112 MB
    "jessicali9530/celeba-dataset", # 1.45 GB
    "debarghamitraroy/casia-webface", # 2.81 GB
    "hearfool/vggface2", # 2.5 GB
    "arnaud58/flickrfaceshq-dataset-ffhq" # 20.9 GB
]

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

i = 0
while True:
    try: dataset = datasets[i]
    except: break
    try:
        # Download dataset
        dataset_path = kagglehub.dataset_download(dataset)
        print(f"Path to dataset files for {dataset}:", dataset_path)
        
        # Get dataset name (e.g., lfw-dataset)
        dataset_name = dataset.split('/')[-1]
        
        # Destination folder for all images (flattened)
        dest_folder = os.path.join(full_datasets_dir, dataset_name)
        os.makedirs(dest_folder, exist_ok=True)
        
        # Recursively find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(dataset_path, '**', f'*{ext}'), recursive=True))
        
        print(f"Found {len(image_files)} images in {dataset_name}")
        
        # Copy images to destination folder, flattening structure
        for idx, img_path in enumerate(image_files):
            # Avoid filename collisions by prefixing with index
            filename = f"{idx:06d}_" + os.path.basename(img_path)
            dest_path = os.path.join(dest_folder, filename)
            shutil.copy2(img_path, dest_path)
        i += 1
    except:
        dt = 5
        print(f"connection interrupted, retrying in {dt} seconds", end='\r')
        time.sleep(dt)
        continue