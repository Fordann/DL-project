import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import CONFIG
from trainer import betas, alphas, alpha_bars
@tf.function
def p_sample(model, x, t):
    """
    Un step de débruitage (enlève progressivement le bruit)
    """
    # Récupérer les paramètres pour ce timestep
    beta_t = tf.gather(betas, t)
    alpha_t = tf.gather(alphas, t)
    alpha_bar_t = tf.gather(alpha_bars, t)
    
    # Prédire le bruit
    eps_theta = model([x, t], training=False)
    eps_theta = tf.cast(eps_theta, tf.float32)
    
    # Calculer la moyenne
    sqrt_alpha_t = tf.sqrt(alpha_t)
    sqrt_one_minus_alpha_bar_t = tf.sqrt(1.0 - alpha_bar_t)
    
    # Reshape pour broadcasting
    sqrt_alpha_t = tf.reshape(sqrt_alpha_t, [-1, 1, 1, 1])
    sqrt_one_minus_alpha_bar_t = tf.reshape(sqrt_one_minus_alpha_bar_t, [-1, 1, 1, 1])
    beta_t = tf.reshape(beta_t, [-1, 1, 1, 1])
    
    mean = (1.0 / sqrt_alpha_t) * (
        x - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_theta
    )
    
    # Ajouter du bruit (sauf au dernier step)
    if t[0] > 0:
        sigma_t = tf.sqrt(beta_t)
        z = tf.random.normal(shape=tf.shape(x))
        mean = mean + sigma_t * z
    
    return mean


def sample(model, img_size=128, num_images=1, show_progress=True):
    """
    Génère des images à partir de bruit pur
    """
    print(f"🎨 Génération de {num_images} image(s)...")
    
    # Commencer avec du bruit pur
    x = tf.random.normal((num_images, img_size, img_size, 3))
    
    # Débruiter progressivement (de T-1 à 0)
    if show_progress:
        iterator = tqdm(reversed(range(CONFIG['T'])), desc="Sampling", total=CONFIG['T'])
    else:
        iterator = reversed(range(CONFIG['T']))
    
    for t in iterator:
        t_batch = tf.constant([t] * num_images, dtype=tf.int32)
        x = p_sample(model, x, t_batch)
    
    return x

def generate_single_image(model, img_size=128, save_path='generated_car.png'):
    """Génère et sauvegarde une seule image"""
    
    # Générer
    img = sample(model, img_size=img_size, num_images=1)
    
    # Normaliser pour affichage (de [-1, 1] à [0, 1])
    img = tf.clip_by_value((img + 1) / 2, 0, 1)
    
    # Afficher et sauvegarder
    plt.figure(figsize=(6, 6))
    plt.imshow(img[0])
    plt.axis('off')
    plt.title('Voiture Générée par le Modèle', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    
    print(f"✅ Image sauvegardée: {save_path}")
    return img[0]

def visualize_diffusion_process(model, img_size=128, steps_to_show=10, 
                                save_path='diffusion_process.png'):
    """
    Montre comment l'image se forme étape par étape
    """
    print("🎬 Visualisation du processus de diffusion...")
    
    # Commencer avec du bruit pur
    x = tf.random.normal((1, img_size, img_size, 3))
    
    # Choisir les étapes à afficher
    steps_indices = np.linspace(0, CONFIG['T']-1, steps_to_show, dtype=int)
    saved_images = []
    
    # Débruiter progressivement
    for t in tqdm(reversed(range(CONFIG['T'])), desc="Débruitage", total=CONFIG['T']):
        t_batch = tf.constant([t], dtype=tf.int32)
        x = p_sample(model, x, t_batch)
        
        # Sauvegarder certaines étapes
        if t in steps_indices:
            img = tf.clip_by_value((x + 1) / 2, 0, 1)
            saved_images.append((CONFIG['T'] - t, img[0].numpy()))
    
    # Afficher
    n_cols = 5
    n_rows = (len(saved_images) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten()
    
    for i, (step, img) in enumerate(saved_images):
        axes[i].imshow(img)
        axes[i].set_title(f'Step {step}/{CONFIG['T']}', fontsize=10, fontweight='bold')
        axes[i].axis('off')
    
    # Cacher les axes vides
    for i in range(len(saved_images), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Processus de Débruitage (de Bruit → Image)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Processus sauvegardé: {save_path}")
    return saved_images