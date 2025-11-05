from U_net import build_unet
from config import CONFIG
from trainer import train_model
from sampling import generate_single_image, visualize_diffusion_process

if __name__ == "__main__":
    unet_model = build_unet(CONFIG['img_size'])
    train_model(unet_model)

    generated_img = generate_single_image(unet_model, img_size=CONFIG['img_size'])

    process_images = visualize_diffusion_process(
        unet_model, 
        img_size=CONFIG['img_size'], 
        steps_to_show=10
    )