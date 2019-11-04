import os, sys, pickle, json
from os import makedirs
from os.path import exists, join
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import optimizers, backend
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Concatenate, Conv2D, AveragePooling2D, Lambda

import keras_model_64 as cnn64


SHUFFLE_BUFFER = 512
batch_size = 128
num_classes = 2
block_size = 64
NUM_CHANNELS = 1
epochs = 100
filepath ='./testdata.tfrecord'



def load_tfrecord(filepath):
    features = {'image': tf.io.FixedLenFeature([], tf.float32),
                'label': tf.io.FixedLenFeature([], tf.int64)
                }
    data = []
    for s_example in tf.python_io.tf_record_iterator(filepath):
        example = tf.parse_single_example(s_example, features=features)
        data.append(tf.expand_dims(example['image'], 0))
        data.append(tf.expand_dims(example['label'], 1))
    return tf.concat(0, data)


if __name__ == "__main__":
    data = load_tfrecord(filepath)