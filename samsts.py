import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display


dataset_path ="C:\\Users\\Sam\\PycharmProjects\\samsgan\\training_data\\"
BATCH_SIZE=3
IMAGE_SIZE=(571,571)

train_dataset=tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE
)


train_dataset=train_dataset.map(lambda x,y:(tf.cast(x,tf.float32)/127.5-1.0,y))
train_dataset=train_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

noise_dim=100
num_channels=3
#
def make_generator(height=7, width=7,inputChannels=256):
    generator=tf.keras.Sequential()
    generator.add(layers.Dense(height*width*inputChannels,use_bias=False,input_shape=(100,)))
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU())

    generator.add(layers.Reshape(height,width,inputChannels))
    assert generator.output_shape == (None, height, width, inputChannels)
    generator.add(layers.Conv2DTranspose(128,(5,5),strides=(1,1),padding='same', use_bias=False))
    # assert generator.output_shape == (none,2)
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU)

    generator.add(layers.Conv2DTranspose(64,(5,5), strides=(2,2),padding='same',use_bias=False))
    # assert generator
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU)

    generator.add(layers.Conv2DTranspose(1,(5,5),strides=(2,2),padding='same',use_bias=False))
    # assert

    return generator


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

# LOSS FUNCTIONS
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def lossDiscriminator(realOut,fakeOut):
    realLoss=cross_entropy(tf.ones_like(realOut),realOut)
    fakeLoss=cross_entropy(tf.ones_like(fakeOut),fakeOut)
    return realLoss+fakeLoss
def lossGenerator(fakeOut):
    return cross_entropy(tf.ones_like(fakeOut),fakeOut)
