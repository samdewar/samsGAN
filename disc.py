import tensorflow as tf
from tensorflow.keras import layers

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

optimiser=tf.keras.optimizers.Adam(1-4)
def make_discriminator(droupoutRate=0.3):
    discriminator=tf.keras.Sequential()
    discriminator.add(layers.Conv2D(64,(5,5),strides=(2,2),padding='same',input_shape=[IMAGE_SIZE,1]))

    discriminator.add(layers.LeakyReLU())
    discriminator.add(layers.Dropout(droupoutRate))

    discriminator.add(layers.Conv2D(128,(5,5),strides=(2,2),padding='same'))
    discriminator.add(layers.LeakyReLU())
    discriminator.add(layers.Dropout(droupoutRate))

    discriminator.add(layers.Flatten())
    discriminator.add(layers.Dense(1))

    return discriminator

def loss(realOut,fakeOut):
    realLoss=cross_entropy(tf.ones_like(realOut),realOut)
    fakeLoss=cross_entropy(tf.ones_like(fakeOut),fakeOut)
    return realLoss+fakeLoss