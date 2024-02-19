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

import gen
import disc

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


# CHECKPOINTING GOES HERE


