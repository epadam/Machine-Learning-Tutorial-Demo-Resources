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
log_dir='./logs/tf/64'

if not exists(log_dir):
    makedirs(log_dir)



class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir=log_dir, **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', 'epoch_'): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
                        'label': tf.FixedLenFeature([], tf.string),
                        'qp': tf.FixedLenFeature([], tf.string)
                        }
    
    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)
    
    # Turn your saved image string into an array
    image = tf.decode_raw(parsed_features['image'], tf.float32)
    image = tf.to_float(image)
    label = tf.decode_raw(parsed_features['label'], tf.uint8)
    #label = tf.to_uint8(label)
    qp = tf.decode_raw(parsed_features['qp'], tf.uint8)
    #qp = tf.to_uint8(qp)
    
    return image, label, qp

  
def create_dataset(filepath):
    
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)
    
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=4)
    
    # This dataset will go on forever
    dataset = dataset.repeat(epochs)
    
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

    print(np.shape(image))
    print(np.shape(label))

    qp = tf.reshape(qp, [-1, 1])
    print(np.shape(qp))
    
    # Create a one hot array for your labels
    label = tf.one_hot(label, num_classes)
    
    return image, label, qp




SUM_OF_ALL_DATASAMPLES=898

STEPS_PER_EPOCH = SUM_OF_ALL_DATASAMPLES // batch_size

print(type(STEPS_PER_EPOCH))

raw_train, label_train, qp_train = create_dataset(filepath)




def sub_mean(x):
    x = x/255
    x = x - backend.mean(x)   
    return x



data = Input(shape=(block_size,block_size,NUM_CHANNELS))

data_norm = Lambda(sub_mean)(data)

data_pooling = AveragePooling2D(pool_size=(4, 4),padding='valid')(data_norm)

conv1 = Conv2D(16, (4, 4), strides =(4,4), activation='relu', padding='valid')(data_pooling)
conv1_d = Dropout(0.5)(conv1)

conv2 = Conv2D(24, (2, 2), strides =(2,2), activation='relu', padding='valid')(conv1_d)
conv2_d = Dropout(0.5)(conv2)
flat2 = Flatten()(conv2_d)

conv3 = Conv2D(32, (2, 2), strides =(2,2), activation='relu', padding='valid')(conv2_d)
conv3_d = Dropout(0.5)(conv3)
flat3 = Flatten()(conv3_d)

qp = Input(tensor=qp_train)
qp_n = Lambda(lambda x: x/255, input_shape=(1,))(qp)

concat = Concatenate(axis=1)([flat2, flat3, qp_n])

fc1 = Dense(64, activation='relu')(concat)
fc1_d = Dropout(0.5)(fc1)
fc1_qp = Concatenate(axis=1)([fc1_d, qp_n])

fc2 = Dense(48, activation='relu')(fc1_qp)
fc2_d = Dropout(0.5)(fc2)
fc2_qp = Concatenate(axis=1)([fc2_d, qp_n])
    
output = Dense(num_classes, activation='softmax')(fc2_qp)

model = Model(inputs=[data,qp], outputs=output)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])


#model = load_model(log_dir+'/m1_qp120_32_sh.h5')
#class_weight = {0: 8.74, 1: 36.4, 2: 33.82, 3: 1, 4: 132.52, 5: 112.28, 6: 188., 7: 109.24, 8: 63.65, 9: 53.18}
#class_weight = {0: 1.55, 1: 6.87, 2: 7.47, 3: 1, 4: 21.73, 5: 21.2, 6: 23.74, 7: 23.61, 8: 9.64, 9: 11.74} #32
#class_weight = {0: 1., 1: 5.77, 2: 6.29, 3: 11.74, 4: 28.27, 5: 37.52, 6: 28.54, 7: 37.04, 8: 14.1, 9: 15.53} #16

history = model.fit([raw_train, qp_train], label_train, epochs=epochs, verbose=1, steps_per_epoch=STEPS_PER_EPOCH)


y_pred = model.predict([raw_test, qp_test])
y_index =np.zeros(len(y_pred))

for i in range(len(y_pred)):
    y_index[i] = np.argmax(y_pred[i])

#print(y_index)
report = classification_report(y_test,y_index)
print(report)


with open(log_dir+'/precision.txt', 'w') as ps:
    ps.write(report)

y_test = y_test.astype(int)
y_index = y_index.astype(int)

report = classification_report(y_test,y_index, output_dict=True)

with open(log_dir+'/classification_m1_64_HV', 'wb') as re:
    pickle.dump(report, re)


class_names = ["None", "Horz", "Vert","Split","Horz A", "Horz B", "Vert A", "Vert B", "Horz 4", "Vert 4"]
print('Confusion Matrix')

plot_confusion_matrix(y_test, y_index, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig(log_dir+'/m1_qp120_64_cm_HV.jpg')
plt.show()

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(log_dir+'/m1_qp120_64_acc_HV.jpg')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(log_dir+'/m1_qp120_64_loss_HV.jpg')
plt.show()



with open(log_dir+'/trainHistoryDict_m1_64_HV', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


model.save(log_dir+'/m1_qp120_16_fw.h5') 

'''
model=load_model('my_model.h5') 

tsample_file = 'training_samples_all_intra_16.txt'
tlabel_file = 'labels_16_intra.txt'
tqp_file = 'qps_16_intra.txt'

#read samples
with open(tsample_file, 'rb') as f:
    pixels = f.read()
    raw = np.frombuffer(pixels, dtype = np.float)
    traw = np.reshape(raw, [-1, block_size, block_size, NUM_CHANNELS])
    print(np.shape(traw))  
   
#read labels
with open(tlabel_file, 'r') as f_single_label:
    single_label = f_single_label.read()    
    single_label =np.fromstring (single_label, dtype=np.int32 ,sep=' ')
    single_label = np.reshape(single_label, [-1])
    tsingle_label = keras.utils.to_categorical(single_label, num_classes)
    print(np.shape(tsingle_label))

#labels_10 = np.loadtxt("labels_10_64.txt", dtype=float)

#read qps
with open(tqp_file, 'r') as f_qp:
    qps = f_qp.read()
    qps =np.fromstring (qps, dtype=np.float, sep=' ')
    tqps = np.reshape(qps, [-1,1]) 
    print(np.shape(tqps)) 


classes = model.predict([traw, tqps])

print(classes)

'''
