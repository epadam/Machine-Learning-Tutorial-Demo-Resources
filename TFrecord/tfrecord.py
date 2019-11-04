import numpy as np
import tensorflow as tf

sample_file = 'HV_samples_64_intra.txt'
label_file = 'HV_labels_64_intra.txt'
qp_file = 'HV_qps_64_intra.txt'
block_size = 64
NUM_CHANNELS = 1


#read samples
with open(sample_file, 'rb') as f:
    pixels = f.read()
    raw = np.frombuffer(pixels, dtype = np.float)
    print('raw shape', np.shape(raw))
    raw = np.uint8(raw)
    raw = np.reshape(raw, [-1, block_size, block_size, NUM_CHANNELS])
    print('raw shape', np.shape(raw))
    raw_train = raw[:898]
    raw_val = raw[-100:]
    #raw_train = np.reshape(raw_train, [-1, block_size, block_size, NUM_CHANNELS])
    print('raw_train', np.shape(raw_train))
    print('raw type', raw_train.dtype)
    #raw_train = raw_train.tostring()
    #raw_test = raw[-len(raw)//10:]
    #raw_test = np.reshape(raw_test, [-1, block_size, block_size, NUM_CHANNELS])
    #print(np.shape(raw_test))
    #raw2=raw/255
    #rmean = np.mean(raw[1])
    #print(rmean)
    #raw3=raw2 - rmean
    #print(raw3)  


#read labels
with open(label_file, 'r') as f_single_label:
    single_label = f_single_label.read()    
    single_label =np.fromstring (single_label, dtype=np.int64 ,sep=' ')
    #single_label = np.reshape(single_label, [-1])
    single_label = np.uint8(single_label)
    #single_label = keras.utils.to_categorical(single_label, num_classes)
    label_train = single_label[:898]
    #label_train = label_train.tostring()
    label_val = single_label[-100:]

    print('label shape',np.shape(label_train))
    #print(np.shape(label_test))
        
    #print(np.shape(single_label))


#read qps
with open(qp_file, 'r') as f_qp:
    qps = f_qp.read()
    qps =np.fromstring (qps, dtype=np.int64, sep=' ')
    #qps = np.reshape(qps, [-1,1])
    #print(np.shape(qps))
    	 
    qp_train = qps[:898]
    #qp_train = qp_train.tostring()
    qp_test = qps[-100:]
    #print(np.shape(qp_train))
    #print(np.shape(qp_test))



output_filename = "./valdata.tfrecord"

'''
writer = tf.io.TFRecordWriter(output_filename)


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


#raw_train = raw_train.flatten

#raw_train = np.reshape(raw_train, [block_size*block_size*NUM_CHANNELS])

#print(np.shape(raw_train))

feature_dict = {
    'image': bytes_feature(raw_train),
    'label': bytes_feature(label_train)
    #'qp': int64_feature(qp_train)
}

example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

writer.write(example.SerializeToString())
'''

# int64
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# bytes
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



with tf.io.TFRecordWriter(output_filename) as writer:
        for index in range(100):
            image_bytes = raw_val[index].tostring()
            label = label_val[index]
            example = tf.train.Example(features=tf.train.Features(
                feature={"image": _bytes_feature(image_bytes),
                         "label": _int64_feature(label)}))
            writer.write(example.SerializeToString())

