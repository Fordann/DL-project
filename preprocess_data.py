import tensorflow as tf
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

class FastImageFolderSequence(tf.keras.utils.Sequence):
    def __init__(self, folder_path, batch_size=8, target_size=(128, 128), 
                 num_workers=4, prefetch=2):
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_workers = num_workers
        self.prefetch = prefetch
        
        # Get all the images in the folder
        self.image_files = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.image_files.append(os.path.join(folder_path, file))
        
        print(f"✓ Found {len(self.image_files)} images")
        
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))
    
    def _load_single_image(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.target_size, Image.LANCZOS)
            img_array = (np.array(img, dtype=np.float32) - 127.5) / 127.5
            return img_array
        except Exception as e:
            print(f"Error: {img_path} - {e}")
            return np.zeros((*self.target_size, 3), dtype=np.float32)
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.image_files))
        batch_files = self.image_files[start_idx:end_idx]
        
        batch_images = list(self.executor.map(self._load_single_image, batch_files))
        
        return np.array(batch_images, dtype=np.float32)
    
    def on_epoch_end(self):
        np.random.shuffle(self.image_files)
    
    def __del__(self):
        self.executor.shutdown(wait=False)