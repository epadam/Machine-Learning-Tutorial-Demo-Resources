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



def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    features = {'image': tf.io.FixedLenFeature([], tf.string),
                 'label': tf.io.FixedLenFeature([], tf.int64)
                        #'qp': tf.io.FixedLenFeature([], tf.int64)
                        }
    
    # Load one example
    parsed_features = tf.io.parse_single_example(proto, features)

    oneimage = tf.decode_raw(parsed_features['image'], tf.uint8)


    #oneimage = tf.cast(parsed_features['image'], tf.float32)


    onelabel = tf.cast(parsed_features['label'], tf.int64)

   
    # Turn your saved image string into an array
    
    #label = tf.cast(label_raw, tf.int32)
    #image = tf.to_float(image_raw)
    #label = tf.decode_raw(parsed_features['label'], tf.int64)
    #label = tf.to_int64(label)
    #qp = tf.decode_raw(qp, tf.int64)
    #print(image.dtype)
    #print(label.dtype)

    
    
    #label = tf.reshape(label, [-1])
    #qp = tf.reshape(qp, [-1])


    
    #qp = tf.to_uint8(qp)
    #print(tf.shape(image))
    #print(tf.shape(label))
    #print(tf.shape(qp))


    return oneimage, onelabel
'''

example = tf.train.Example()
for record in tf.python_io.tf_record_iterator(filepath):
    example.ParseFromString(record)
    f = example.features.feature
    image = f['image'].float_list.value[0]

    image = tf.reshape(image, [-1, block_size, block_size, NUM_CHANNELS])


    label = f['label'].int64_list.value[0]
    # for bytes you might want to represent them in a different way (based on what they were before saving)
    # something like `np.fromstring(f['img'].bytes_list.value[0], dtype=np.uint8
    # Now do something with your v1/v2/v3




dataset = tf.data.TFRecordDataset(filepath)

dataset = dataset.map(_parse_function)

dataset= dataset.shuffle(buffer_size=4000).batch(128)

'''
def create_dataset(filepath):
    
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.repeat(epochs)
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset= dataset.map(_parse_function)

    '''

    dataset = dataset.cache() # This dataset fits in RAM
    dataset = dataset.repeat()
    #dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batch_size, drop_remainder=True) 
    dataset = dataset.prefetch(898)
    '''

    
    # Set the number of datapoints you want to load and shuffle 
    dataset = dataset.shuffle(SHUFFLE_BUFFER)
    
    # Set the batchsize
    dataset = dataset.batch(batch_size)
    
    # Create an iterator
    iterator = dataset.make_one_shot_iterator()
    
    # Create your tf representation of the iterator
    image, label = iterator.get_next()



    # Bring your picture back in shape  
    image = tf.reshape(image, [-1, block_size, block_size, NUM_CHANNELS])
        

    
    # Create a one hot array for your labels
    label = tf.one_hot(label, num_classes)
    
    return image, label


raw_train, label_train = create_dataset(filepath)

raw_val, label_val = create_dataset('./valdata.tfrecord')

print(tf.shape(raw_train))
print(tf.shape(label_train))


print(raw_train.dtype)
print(label_train.dtype)

print(np.shape(raw_train))
print(np.shape(label_train))





# def sub_mean(x):
#     x = x/255
#     x = x - backend.mean(x)   
#     return x



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


score = model.evaluate(raw_val, label_val, verbose=0, steps=2)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


model.save(log_dir+'/m1_qp120_64_HV.h5') 



tf.keras.backend.set_learning_phase(1)

model=tf.keras.models.load_model(log_dir+'/m1_qp120_64_HV.h5')
export_path='./test/1'

with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        session=sess,
        export_dir=export_path,
        inputs={
            'input_image':model.input
        },
        outputs={
            t.name:t for t in model.outputs
        }
    )