import kagglehub
import os
from preprocess_data import FastImageFolderSequence
from config import CONFIG

path = kagglehub.dataset_download("rhtsingh/google-universal-image-embeddings-128x128")
folder_path = os.path.join(path, "128x128/cars")

train_generator = FastImageFolderSequence(
    folder_path=folder_path,
    batch_size=CONFIG['batch_size'],
    target_size=(CONFIG['img_size'], CONFIG['img_size']),
    num_workers=4,
    prefetch=2 
)