import keras
from keras.applications.resnet50 import ResNet50
import numpy as np
from keras.layers import Input, Concatenate
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K


batch_size = 128
num_classes = 10
block_size = 16
NUM_CHANNELS = 1
epochs = 10
sample_file = 'merge_aecf_samples_16_intra.txt'
label_file = 'merge_aecf_labels_16_intra.txt'
qp_file = 'merge_aecf_qps_16_intra.txt'
log_dir='./logs/resnet/16'


#read samples
with open(sample_file, 'rb') as f:
    pixels = f.read()
    raw = np.frombuffer(pixels, dtype = np.float)
    raw = np.reshape(raw, [-1, block_size, block_size, NUM_CHANNELS])
    print(np.shape(raw))
    
    raw = raw /255
    print(len(raw))
    for i in range(len(raw)):
        raw[i]= raw[i]-raw[i].mean()

    #np.kron(raw, np.ones((14,14))) 
   
#read labels
with open(label_file, 'r') as f_single_label:
    single_label = f_single_label.read()    
    single_label =np.fromstring (single_label, dtype=np.int32 ,sep=' ')
    single_label = np.reshape(single_label, [-1])
    single_label = keras.utils.to_categorical(single_label, num_classes)       
    print(np.shape(single_label))


# this could also be the output a different Keras model or layer
#input_tensor = Input(shape=(16, 16, 1))  # this assumes K.image_data_format() == 'channels_last'

img_input = Input(shape=(block_size,block_size,1))
img_conc = Concatenate()([img_input, img_input, img_input])   

base_model = ResNet50(input_tensor=img_conc, weights='imagenet', include_top=False)




# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(10, activation='softmax')(x)



# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)



'''
input_tensor = Input(shape=(64, 64, 1)) 
model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=(64,64,1), pooling=None, classes=10)

'''



# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
#for layer in base_model.layers:
#    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


# train the model on the new data for a few epochs
history = model.fit(raw, single_label, validation_split =0.1, batch_size=batch_size,  epochs=epochs, verbose=1)


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(log_dir+'/m1_qp120_16_acc_dbec.jpg')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(log_dir+'/m1_qp120_16_loss_dbec.jpg')
plt.show()