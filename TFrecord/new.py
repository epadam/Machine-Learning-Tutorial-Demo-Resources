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


def read_TFRecords_test():
    record_itr = tf.python_io.tf_record_iterator(path=filepath)
    for r in record_itr:
        example = tf.train.Example()
        example.ParseFromString(r)

        label = example.features.feature["label"].int64_list.value[0]
        print("Label", label)
        image_bytes = example.features.feature["image"].bytes_list.value[0]
        img = np.fromstring(image_bytes, dtype=np.uint8).reshape(64, 64, 1)
        #print(img)
        # plt.imshow(img, cmap="gray")
        # plt.show()
        break  # 只读取一个Example