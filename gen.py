import tensorflow as tf
from tensorflow.keras import layers


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