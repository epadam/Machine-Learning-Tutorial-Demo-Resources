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
#import keras_model_16 as cnn16
#import keras_model_32 as cnn32
import keras_model_64 as cnn64
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels

np.set_printoptions(threshold=sys.maxsize)

SHUFFLE_BUFFER = 512
batch_size = 128
num_classes = 2
block_size = 64
NUM_CHANNELS = 1
epochs = 100
filepath ='./'
#log_dir='./logs/tf/64'




def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {'image': tf.io.FixedLenFeature([], tf.float32),
                        'label': tf.io.FixedLenFeature([], tf.int64),
                        'qp': tf.io.FixedLenFeature([], tf.int64)
                        }
    
    # Load one example
    image, label, qp = tf.io.parse_single_example(proto, keys_to_features)
    
    # Turn your saved image string into an array
    image = tf.decode_raw(image, tf.float32)
    #image = tf.to_float(image)
    label = tf.decode_raw(label, tf.int64)
    #label = tf.to_uint8(label)
    qp = tf.decode_raw(qp, tf.int64)
    
    #qp = tf.to_uint8(qp)
    print(tf.shape(image))
    print(tf.shape(label))
    print(tf.shape(qp))


    return image, label, qp  


def create_dataset(filepath):
    
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)
    
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function)
    
    # This dataset will go on forever
    dataset = dataset.repeat()
    
    # Set the number of datapoints you want to load and shuffle 
    dataset = dataset.shuffle(SHUFFLE_BUFFER)
    
    # Set the batchsize
    dataset = dataset.batch(batch_size)
    
    # Create an iterator
    iterator = dataset.make_one_shot_iterator()
    
    # Create your tf representation of the iterator
    image, label, qp = iterator.get_next()





    # Bring your picture back in shape
    image = tf.reshape(image, [-1, block_size, block_size, NUM_CHANNELS])
    label = tf.reshape(label, [-1])
    qp = tf.reshape(qp, [-1])

    print(tf.shape(image))
    print(tf.shape(label))
    print(tf.shape(qp))

    print(image.dtype)
    print(label.dtype)
    print(qp.dtype)

    print(np.shape(image))
    print(np.shape(label))
    print(np.shape(qp))
    
    
    # Create a one hot array for your labels
    label = tf.one_hot(label, num_classes)
    
    return image, label, qp


create_dataset(filepath)