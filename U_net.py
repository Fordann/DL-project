import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(x, filters, time_emb, kernel_size=3):
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.GroupNormalization(groups=8)(x)

    #Injecting time embedding
    t = layers.Dense(filters)(time_emb)
    t = layers.Reshape((1, 1, filters))(t)  
    x = x + t 

    # We don't use ReLu: want to learn things from negative weights -> could be interesting pattern
    x = layers.Activation('swish')(x)
    return x

def encoder_block(x, filters, time_emb):
    x = conv_block(x, filters, time_emb)
    x = conv_block(x, filters, time_emb)
    p = layers.MaxPooling2D((2,2))(x)
    return x, p

def decoder_block(x, skip_features, filters, time_emb):
    x = layers.Conv2DTranspose(filters, (2,2), strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, filters, time_emb)
    return x

def build_unet(img_size, channels=3, time_dim=256):
    #define inputs
    img_input = layers.Input(shape=(img_size, img_size, channels), name='image')
    time_input = layers.Input(shape=(), dtype=tf.int32, name='timestep')

    #Transform time_input in an Embedding space 
    time_emb = layers.Embedding(input_dim=1000, output_dim=time_dim)(time_input)
    time_emb = layers.Dense(time_dim, activation='swish')(time_emb)
    time_emb = layers.Dense(time_dim, activation='swish')(time_emb)


    c1, p1 = encoder_block(img_input, 64, time_emb)
    c2, p2 = encoder_block(p1, 128, time_emb)
    c3, p3 = encoder_block(p2, 256, time_emb)
    c4, p4 = encoder_block(p3, 512, time_emb)

    bottleneck = conv_block(p4, 1024, time_emb)
    bottleneck = conv_block(bottleneck, 1024, time_emb)

    d4 = decoder_block(bottleneck, c4, 512, time_emb)
    d3 = decoder_block(d4, c3, 256, time_emb)
    d2 = decoder_block(d3, c2, 128, time_emb)
    d1 = decoder_block(d2, c1, 64, time_emb)

    output = layers.Conv2D(channels, (1, 1), activation='linear')(d1)
    
    model = Model(inputs=[img_input, time_input], outputs=output, name='UNet_Diffusion')

    return model

