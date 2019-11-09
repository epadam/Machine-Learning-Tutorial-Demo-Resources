import keras
from keras import backend as K
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import keras_model_16b as cnn16

block_size = 32
NUM_CHANNELS = 1
num_classes = 2
sample_file = 'NS_samples_32_intra.txt'
label_file = 'NS_labels_32_intra.txt'
qp_file = 'NS_qps_32_intra.txt'

with open(sample_file, 'rb') as f:
    pixels = f.read()
    raw = np.frombuffer(pixels, dtype = np.float)
    raw = np.reshape(raw, [-1, block_size, block_size, NUM_CHANNELS])
    print(np.shape(raw))
    rawt = raw[:45000]
    print(np.shape(rawt))
    raw10=raw[-5000:]
    print(np.shape(raw10))

#read labels
with open(label_file, 'r') as f_single_label:
    single_label = f_single_label.read()    
    single_label =np.fromstring (single_label, dtype=np.int32 ,sep=' ')
    single_label = np.reshape(single_label, [-1])
    single_labelt = single_label[:45000]
    last10 = single_label[-5000:]
    print(np.shape(last10))
    single_labelt = keras.utils.to_categorical(single_labelt, num_classes)
    last10hot = keras.utils.to_categorical(last10, num_classes)

print('partition=', last10[23])

model=load_model('m1_qp120_32_NS2.h5')
#查看输入图片
fig1,ax1 = plt.subplots(figsize=(4,4))
ax1.imshow(np.reshape(raw10[23], (32, 32)),cmap='gray')
plt.show()


image_arr=np.reshape(raw10[23], (-1,32, 32,1))
#可视化第一个MaxPooling2D
layer_1 = K.function([model.layers[0].input], [model.layers[1].output])
# 只修改inpu_image
f1 = layer_1([image_arr])[0]
# 第一层卷积后的特征图展示，输出是（1,12,12,6），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
re = np.transpose(f1, (0,3,1,2))

plt.subplot()
plt.imshow(re[0][0], cmap='gray')
plt.show()
#可视化第二个MaxPooling2D
layer_2 = K.function([model.layers[0].input], [model.layers[2].output])
f2 = layer_2([image_arr])[0]
# 第一层卷积后的特征图展示，输出是（1,4,4,16），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
re = np.transpose(f2, (0,3,1,2))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(re[0][i])
    plt.colorbar()
plt.show() 

layer_3 = K.function([model.layers[0].input], [model.layers[4].output])
f3 = layer_3([image_arr])[0]
# 第一层卷积后的特征图展示，输出是（1,4,4,16），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
re = np.transpose(f3, (0,3,1,2))
for i in range(24):
    plt.subplot(4,6,i+1)
    plt.imshow(re[0][i])
    plt.colorbar()
plt.show()

layer_3 = K.function([model.layers[0].input], [model.layers[6].output])
f3 = layer_3([image_arr])[0]
# 第一层卷积后的特征图展示，输出是（1,4,4,16），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
re = np.transpose(f3, (0,3,1,2))
for i in range(32):
    plt.subplot(4,8,i+1)
    plt.imshow(re[0][i])
    plt.colorbar()
plt.show() 
