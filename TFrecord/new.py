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
log_dir='./logs/tf/64'





def decode(serialized_example):
    """decode the serialized example"""
    features = tf.parse_single_example(serialized_example,
                            features={"image": tf.FixedLenFeature([], tf.string),
                                      "label": tf.FixedLenFeature([], tf.int64)})
    image = tf.decode_raw(features["image"], tf.uint8)
    #image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [block_size, block_size, NUM_CHANNELS])
    label = tf.cast(features["label"], tf.int64)
    return image, label


def normalize(image, label):
    """normalize the image to [-0.5, 0.5]"""
    image = image / 255.0
    return image, label





def create_dataset(filename):
    
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filename)

    dataset = dataset.repeat(epochs)
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset= dataset.map(decode)
    
    # Set the number of datapoints you want to load and shuffle 
    dataset = dataset.shuffle(SHUFFLE_BUFFER)
    
    # Set the batchsize
    dataset = dataset.batch(batch_size)
    
    # Create an iterator
    iterator = dataset.make_one_shot_iterator()
    
    # Create your tf representation of the iterator
    image, label = iterator.get_next()

    # Bring your picture back in shape  
    #image = tf.reshape(image, [-1, block_size, block_size, NUM_CHANNELS])
        

    
    # Create a one hot array for your labels
    label = tf.one_hot(label, num_classes)
    
    return image, label


raw_train, label_train = create_dataset(filepath)

print(tf.shape(raw_train))
print(tf.shape(label_train))


print(raw_train.dtype)
print(label_train.dtype)

print(np.shape(raw_train))
print(np.shape(label_train))



data = Input(shape=(block_size,block_size,NUM_CHANNELS))

#data_norm = Lambda(sub_mean)(data)

data_pooling = AveragePooling2D(pool_size=(4, 4),padding='valid')(data)

conv1 = Conv2D(16, (4, 4), strides =(4,4), activation='relu', padding='valid')(data_pooling)
conv1_d = Dropout(0.5)(conv1)

conv2 = Conv2D(24, (2, 2), strides =(2,2), activation='relu', padding='valid')(conv1_d)
conv2_d = Dropout(0.5)(conv2)
flat2 = Flatten()(conv2_d)

conv3 = Conv2D(32, (2, 2), strides =(2,2), activation='relu', padding='valid')(conv2_d)
conv3_d = Dropout(0.5)(conv3)
flat3 = Flatten()(conv3_d)


# qp = Input(shape=(1,))
# qp_n = Lambda(lambda x: x/255)(qp)

concat = Concatenate(axis=1)([flat2, flat3])

fc1 = Dense(64, activation='relu')(concat)
fc1_d = Dropout(0.5)(fc1)
#fc1_qp = Concatenate(axis=1)([fc1_d, qp_n])

fc2 = Dense(48, activation='relu')(fc1_d)
fc2_d = Dropout(0.5)(fc2)
#fc2_qp = Concatenate(axis=1)([fc2_d, qp_n])
    
output = Dense(num_classes, activation='softmax')(fc2_d)

model = Model(inputs=data, outputs=output)


model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])



SUM_OF_ALL_DATASAMPLES=898

STEPS_PER_EPOCH = SUM_OF_ALL_DATASAMPLES // batch_size

print(type(STEPS_PER_EPOCH))


history = model.fit(raw_train, label_train, epochs=epochs, verbose=1, steps_per_epoch=STEPS_PER_EPOCH)
