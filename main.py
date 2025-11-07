from U_net import build_unet
from config import CONFIG
from trainer import train_model
from sampling import generate_single_image, visualize_diffusion_process
from tensorflow.keras.models import load_model
import os
import argparse
from gpu_setup import setup_tensorflow_gpu

if __name__ == "__main__":
    gpu_available = setup_tensorflow_gpu()
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', default=False, action="store_true")
    args = parser.parse_args()

    if args.training or not os.path.exists(CONFIG['best_model_name']):
        unet_model = build_unet(CONFIG['img_size'])
        train_model(unet_model) 
    else:
        unet_model = load_model(CONFIG['best_model_name'])

    generated_img = generate_single_image(unet_model, img_size=CONFIG['img_size'])

    process_images = visualize_diffusion_process(
        unet_model, 
        img_size=CONFIG['img_size'], 
        steps_to_show=10
    )