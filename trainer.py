import numpy as np
import tensorflow as tf
from tqdm import tqdm
from load_dataset import train_generator
from config import CONFIG

#noise for each step
betas = tf.constant( 
    np.linspace(CONFIG['beta_start'], CONFIG['beta_end'], CONFIG['T'], dtype=np.float32)
)
alphas = tf.constant(1.0 - betas.numpy()) # quantity of information still in the image
alpha_bars = tf.constant(np.cumprod(alphas.numpy(), axis=0))


def q_sample(x0, t, noise=None):
    if noise is None:
        noise = tf.random.normal(shape=tf.shape(x0), dtype=x0.dtype)

    sqrt_alpha_bar = tf.gather(tf.sqrt(alpha_bars), t)
    sqrt_one_minus = tf.gather(tf.sqrt(1.0 - alpha_bars), t)

    sqrt_alpha_bar = tf.reshape(sqrt_alpha_bar, (-1, 1, 1, 1))
    sqrt_one_minus = tf.reshape(sqrt_one_minus, (-1, 1, 1, 1))

    sqrt_alpha_bar = tf.cast(sqrt_alpha_bar, x0.dtype)
    sqrt_one_minus = tf.cast(sqrt_one_minus, x0.dtype)
    noise = tf.cast(noise, x0.dtype)

    return sqrt_alpha_bar * x0 + sqrt_one_minus * noise


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = tf.exp(-np.log(10000.0) * tf.range(0, half, dtype=tf.float32) / half)
    args = tf.cast(tf.expand_dims(t, 1), tf.float32) * tf.expand_dims(freqs, 0)
    emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
    return emb


def diffusion_loss(model, x0):
    batch_size = tf.shape(x0)[0]
    t = tf.random.uniform((batch_size,), minval=0, maxval=CONFIG['T'], dtype=tf.int32)
    noise = tf.random.normal(shape=tf.shape(x0), dtype=tf.float32)
    x_t = q_sample(x0, t, noise)
    t_float = tf.cast(t, tf.float32)
    
    pred_noise = model([x_t, t_float], training=True)
    pred_noise = tf.cast(pred_noise, tf.float32)  # conversion float16 to float32 to use mixedprecision 
    
    loss = tf.reduce_mean(tf.square(noise - pred_noise))
    return loss


optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-5)

class CustomEarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float('inf')
        self.wait = 0
        self.best_weights = None
    
    def on_epoch_end(self, epoch, loss, model):
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0
            self.best_weights = model.get_weights()
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                if self.best_weights is not None:
                    model.set_weights(self.best_weights)
                return True
        return False

early_stopping = CustomEarlyStopping(patience=5)

@tf.function
def train_step(x, model):
    with tf.GradientTape() as tape:
        loss = diffusion_loss(model, x)

    grads = tape.gradient(loss, model.trainable_variables)
    grads = [tf.clip_by_norm(g, 1.0) for g in grads if g is not None]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train_model(model):
    losses_history = []

    for epoch in range(CONFIG['epochs']):
        epoch_losses = []
        
        progress_bar = tqdm(
            range(len(train_generator)), 
            desc=f"Epoch {epoch+1}"
        )
 
        for batch_idx in progress_bar:
            x_batch = train_generator[batch_idx]
            x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
            
            loss = train_step(x_batch, model)

            epoch_losses.append(loss.numpy())
            progress_bar.set_postfix({
                'loss': f'{loss.numpy():.4f}',
                'lr': f'{optimizer.learning_rate.numpy():.2e}'
            })

        avg_loss = np.mean(epoch_losses)
        losses_history.append(avg_loss)
        
        print(f"\n Epoch {epoch+1} - Loss: {avg_loss:.4f}")
        train_generator.on_epoch_end()
        should_stop = early_stopping.on_epoch_end(epoch, avg_loss, model)
        if should_stop:
            break
    
    model.save('diffusion_model_complete.keras')
    return losses_history
