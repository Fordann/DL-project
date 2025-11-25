# main.py
#
# Entry point for:
#   - training the text-conditioned diffusion model
#   - generating images from a text prompt using CLIP + UNet_clip

import os
import argparse

import cv2
import tensorflow as tf
import torch
from transformers import CLIPTokenizer, CLIPTextModel

from config import CONFIG
from gpu_setup import setup_tensorflow_gpu
from UNet_clip import build_unet_clip
from trainer import train_model
from sampling import generate_single_image, visualize_diffusion_process


# ----------------------------------------------------
# CLIP text encoder helper (for inference / sampling)
# ----------------------------------------------------
_CLIP_TOKENIZER = None
_CLIP_TEXT_MODEL = None
_CLIP_DEVICE = "cpu"


def load_clip_text_model(model_name: str = "openai/clip-vit-base-patch32"):
    """
    Lazy-load the CLIP tokenizer and text model (for prompts at inference time).
    """
    global _CLIP_TOKENIZER, _CLIP_TEXT_MODEL, _CLIP_DEVICE

    if _CLIP_TOKENIZER is not None and _CLIP_TEXT_MODEL is not None:
        return _CLIP_TOKENIZER, _CLIP_TEXT_MODEL

    print(f"[CLIP] Loading text model: {model_name}")
    _CLIP_TOKENIZER = CLIPTokenizer.from_pretrained(model_name)
    _CLIP_TEXT_MODEL = CLIPTextModel.from_pretrained(model_name)

    _CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _CLIP_TEXT_MODEL.to(_CLIP_DEVICE)

    return _CLIP_TOKENIZER, _CLIP_TEXT_MODEL


def encode_prompt_to_tf(prompt: str, model_name: str = "openai/clip-vit-base-patch32") -> tf.Tensor:
    """
    Encode a single text prompt into a CLIP embedding and return it
    as a TensorFlow tensor of shape (1, text_dim).

    This is used at inference time to condition the diffusion model.
    """
    tokenizer, text_model = load_clip_text_model(model_name)

    text_model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(_CLIP_DEVICE)

        outputs = text_model(**inputs)
        pooled = outputs.pooler_output  # shape: (1, text_dim), e.g. (1, 512)

    emb_np = pooled.detach().cpu().numpy().astype("float32")
    text_emb_tf = tf.convert_to_tensor(emb_np, dtype=tf.float32)  # (1, text_dim)
    return text_emb_tf


# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main():
    # Setup TensorFlow GPU (your own helper)
    gpu_available = setup_tensorflow_gpu()
    print(f"GPU available: {gpu_available}")

    parser = argparse.ArgumentParser(description="Text-conditioned diffusion (UNet + CLIP)")
    parser.add_argument(
        "--training",
        action="store_true",
        help="Train the diffusion model instead of only loading it."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt to generate an image from (requires a trained model)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sample.png",
        help="Path where the generated image will be saved."
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="If set, also save intermediate diffusion images."
    )
    parser.add_argument(
        "--viz_steps",
        type=int,
        default=10,
        help="Number of diffusion steps to visualize if --viz is used."
    )

    args = parser.parse_args()

    # ------------------------------------------------
    # 1) Build or load the model
    # ------------------------------------------------
    model_path = CONFIG["best_model_name"]  # ensure this matches what trainer.save() uses

    if args.training or not os.path.exists(model_path):
        print("[MAIN] Training mode or no existing model found, building a new UNet_clip.")

        unet_model = build_unet_clip(
            img_size=CONFIG["img_size"],
            channels=3,
            time_dim=256,         # you can move this to CONFIG if you want
            text_dim=512,         # CLIP ViT-B/32 uses 512-dim text embeddings
            max_timesteps=CONFIG["T"]
        )

        train_model(unet_model)
        print(f"[MAIN] Training finished. Model saved (check trainer.py path).")
    else:
        print(f"[MAIN] Loading existing model from {model_path}")
        unet_model = tf.keras.models.load_model(model_path)
        print("[MAIN] Model loaded.")

    # ------------------------------------------------
    # 2) If a prompt is provided, generate an image
    # ------------------------------------------------
    if args.prompt is not None:
        print(f"[MAIN] Generating image for prompt: \"{args.prompt}\"")

        # Encode the text prompt with CLIP
        text_emb = encode_prompt_to_tf(args.prompt)  # shape (1, 512)

        # Generate a single image conditioned on this embedding
        generated_img = generate_single_image(
            unet_model,
            img_size=CONFIG["img_size"],
            text_emb=text_emb
        )

        # Save the generated image (OpenCV expects BGR)
        cv2.imwrite(args.output, generated_img)
        print(f"[MAIN] Generated image saved to: {args.output}")

        # Optionally visualize the reverse diffusion process
        if args.viz:
            print(f"[MAIN] Visualizing diffusion process: {args.viz_steps} snapshots.")
            process_images = visualize_diffusion_process(
                unet_model,
                img_size=CONFIG["img_size"],
                text_emb=text_emb,
                steps_to_show=args.viz_steps
            )

            # Save each intermediate step
            viz_dir = "diffusion_viz"
            os.makedirs(viz_dir, exist_ok=True)

            for i, img in enumerate(process_images):
                out_path = os.path.join(viz_dir, f"step_{i:02d}.png")
                cv2.imwrite(out_path, img)

            print(f"[MAIN] Saved {len(process_images)} diffusion steps in folder: {viz_dir}")

    else:
        print("[MAIN] No prompt provided. Training / loading done, nothing to generate.")


if __name__ == "__main__":
    main()
