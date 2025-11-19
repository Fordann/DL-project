# TextGeneration.py

import cv2
import numpy as np
import tensorflow as tf

from U_net import build_unet


# -----------------------------
# Simple CLIP-like text encoder
# -----------------------------

def build_text_vectorizer(max_tokens=10000, max_len=32):
    """
    Text vectorizer that turns raw strings into integer token IDs.
    You should call `adapt_text_vectorizer()` once with a list of prompts
    before using the encoder.
    """
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=max_len
    )
    return vectorizer


def build_clip_text_encoder(vocab_size=10000, max_len=32, embed_dim=256, proj_dim=256):
    """
    Very lightweight CLIP-style text encoder:
      tokens -> embedding -> pooled -> MLP
    This is NOT the real OpenAI CLIP, but behaves similarly as a text embedding model.
    """
    token_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int64, name="token_ids")

    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        name="text_embedding"
    )(token_ids)

    # Simple pooling over token dimension
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Small MLP projection head ("CLIP-style" projection)
    x = tf.keras.layers.Dense(512, activation="swish")(x)
    x = tf.keras.layers.Dense(proj_dim, activation="swish")(x)

    model = tf.keras.Model(token_ids, x, name="clip_text_encoder")
    return model


# ---------------------------------
# Diffusion + U-Net + Text guidance
# ---------------------------------

class TextToImageDiffusion:
    def __init__(
        self,
        img_size=64,
        channels=3,
        timesteps=1000,
        max_tokens=10000,
        max_len=32,
        time_dim=256
    ):
        self.img_size = img_size
        self.channels = channels
        self.timesteps = timesteps

        # U-Net from your separate file
        self.unet = build_unet(img_size=img_size, channels=channels, time_dim=time_dim)

        # Beta schedule (linear)
        beta = np.linspace(1e-4, 0.02, timesteps, dtype=np.float32)
        alpha = 1.0 - beta
        alpha_bar = np.cumprod(alpha)

        self.beta = tf.constant(beta, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.alpha_bar = tf.constant(alpha_bar, dtype=tf.float32)

        # CLIP-style text encoder
        self.text_vectorizer = build_text_vectorizer(
            max_tokens=max_tokens,
            max_len=max_len
        )
        self.text_encoder = build_clip_text_encoder(
            vocab_size=max_tokens,
            max_len=max_len,
            embed_dim=256,
            proj_dim=time_dim  # same dim as time embedding for easier fusion if desired
        )

    # ---------- Text handling ----------

    def adapt_text_vectorizer(self, text_corpus):
        """
        Fit the TextVectorization layer on a list of example prompts.
        Call this once before using encode_text().
        """
        text_ds = tf.data.Dataset.from_tensor_slices(text_corpus).batch(32)
        self.text_vectorizer.adapt(text_ds)

    def encode_text(self, prompt):
        """
        Encode a single text prompt into a CLIP-style embedding.
        Shape: (1, time_dim)
        """
        tokens = self.text_vectorizer(tf.constant([prompt]))
        text_emb = self.text_encoder(tokens)
        return text_emb

    # ---------- Image utilities (OpenCV) ----------

    def _postprocess_image(self, x):
        """
        Convert model output in [-1, 1] to uint8 BGR image for OpenCV.
        """
        x = tf.clip_by_value(x, -1.0, 1.0)
        x = ((x + 1.0) * 127.5)  # [-1,1] -> [0,255]
        img = tf.cast(x[0], tf.uint8).numpy()  # take first in batch
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    # ---------- Sampling (reverse diffusion) ----------

    def p_sample(self, x_t, t_scalar):
        """
        One reverse diffusion step:
          x_t -> x_{t-1}
        Uses your U-Net to predict noise at timestep t.
        This version does NOT yet include gradient-based CLIP guidance
        on the image; the U-Net itself can be trained to use text
        conditioning in its weights.
        """
        # Unet expects a batch of timesteps (int32)
        t_tensor = tf.constant([t_scalar], dtype=tf.int32)

        # Predict noise epsilon_theta(x_t, t)
        eps_theta = self.unet([x_t, t_tensor])

        # Get scalars for this timestep
        beta_t = self.beta[t_scalar]
        alpha_t = self.alpha[t_scalar]
        alpha_bar_t = self.alpha_bar[t_scalar]

        # Predict x0 from x_t and epsilon
        x0_pred = (x_t - tf.sqrt(1.0 - alpha_bar_t) * eps_theta) / tf.sqrt(alpha_bar_t)

        # Compute mean of q(x_{t-1} | x_t, x0)
        mean = (
            tf.sqrt(self.alpha_bar[t_scalar - 1] if t_scalar > 0 else 1.0) * x0_pred +
            tf.sqrt(1.0 - self.alpha_bar[t_scalar - 1] if t_scalar > 0 else 0.0) * eps_theta
        )

        if t_scalar > 0:
            noise = tf.random.normal(tf.shape(x_t))
            x_prev = mean + tf.sqrt(beta_t) * noise
        else:
            x_prev = mean

        return x_prev

    def sample(self, prompt, num_steps=50):
        """
        Generate a single image from a text prompt.
        The text encoder output is computed and (in a full implementation)
        would be used to train/condition the U-Net. Here we focus on
        wiring everything together and running the sampling loop.
        """
        batch_size = 1

        text_emb = self.encode_text(prompt)  # shape (1, time_dim)
        _ = text_emb  # placeholder – prevents unused variable warnings

        # Start from pure Gaussian noise
        x = tf.random.normal((batch_size, self.img_size, self.img_size, self.channels))

        # Choose a subset of timesteps to step through
        step_indices = np.linspace(self.timesteps - 1, 0, num_steps, dtype=int)

        for t in step_indices:
            x = self.p_sample(x, int(t))

        # Convert final latent to OpenCV image
        return self._postprocess_image(x)

    def save_sample(self, prompt, out_path, num_steps=50):
        """
        Generate an image from `prompt` and save it to `out_path` using OpenCV.
        """
        img_bgr = self.sample(prompt, num_steps=num_steps)
        cv2.imwrite(out_path, img_bgr)
        print(f"Saved generated image to: {out_path}")


# -------------------------
# Example usage (main)
# -------------------------

if __name__ == "__main__":
    # Hyperparameters – match these to your U-Net training settings
    IMG_SIZE = 64
    CHANNELS = 3
    TIMESTEPS = 1000
    TIME_DIM = 256

    diffusion = TextToImageDiffusion(
        img_size=IMG_SIZE,
        channels=CHANNELS,
        timesteps=TIMESTEPS,
        time_dim=TIME_DIM
    )

    corpus = [
        "a red pixel art dragon",
        "a small blue house",
        "a green tree with a brown trunk",
        "a pixel art knight with a sword"
    ]
    diffusion.adapt_text_vectorizer(corpus)

    prompt = "a red pixel art dragon"
    diffusion.save_sample(prompt, "sample_dragon.png", num_steps=50)
