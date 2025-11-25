# clip_encode_labels.py

import os
import glob
import numpy as np
import torch
from transformers import CLIPTokenizer, CLIPTextModel

# --------- Config adaptée à ton projet ---------
MODEL_NAME = "openai/clip-vit-base-patch32"

# Get project root directory (one level up from clipEncoder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "train_images")
CAPTIONS_DIR = os.path.join(BASE_DIR, "dataset", "train_images_captions")

BATCH_SIZE = 32
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_EMB_PATH = os.path.join(OUT_DIR, "clip_text_embs.npy")
OUT_IMG_LIST_PATH = os.path.join(OUT_DIR, "clip_image_paths.txt")
# ----------------------------------------------


def main():
    print(f"Loading CLIP tokenizer & text model: {MODEL_NAME}")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
    text_model = CLIPTextModel.from_pretrained(MODEL_NAME)

    # Lister les images .png
    img_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))
    if len(img_paths) == 0:
        raise ValueError(f"Aucune image .png trouvée dans {IMAGES_DIR}")

    captions = []
    valid_img_paths = []

    print(f"Found {len(img_paths)} images. Matching with captions...")

    for img_path in img_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        # Chez toi: image_...png  ->  caption: image_...png_caption.txt
        caption_filename = base + ".png_caption.txt"
        txt_path = os.path.join(CAPTIONS_DIR, caption_filename)

        if not os.path.exists(txt_path):
            print(f"[WARN] No caption for image {img_path}, expected {txt_path}")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if text == "":
            print(f"[WARN] Empty caption in {txt_path}, skipping.")
            continue

        captions.append(text)
        valid_img_paths.append(img_path)

    if len(captions) == 0:
        raise ValueError("Aucun couple (image, caption) valide trouvé.")

    print(f"Will encode {len(captions)} (image, caption) pairs.")

    text_model.eval()
    all_embs = []

    with torch.no_grad():
        for i in range(0, len(captions), BATCH_SIZE):
            batch = captions[i:i + BATCH_SIZE]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            outputs = text_model(**inputs)
            pooled = outputs.pooler_output    # (B, hidden_size)
            all_embs.append(pooled.cpu().numpy())

    embs = np.concatenate(all_embs, axis=0)
    np.save(OUT_EMB_PATH, embs)

    with open(OUT_IMG_LIST_PATH, "w", encoding="utf-8") as f:
        for p in valid_img_paths:
            # Write relative path from project root for portability
            rel_path = os.path.relpath(p, BASE_DIR)
            f.write(rel_path + "\n")

    print(f"Saved embeddings to {OUT_EMB_PATH} with shape {embs.shape}")
    print(f"Saved aligned image list to {OUT_IMG_LIST_PATH}")


if __name__ == "__main__":
    main()
