import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow.contrib.tensorboard.plugins import projector

LOG_DIR = 'projector/HV64'
NAME_TO_VISUALISE_VARIABLE = "mnistembedding"
sample_file = 'HV_samples_64_intra.txt'
label_file = 'HV_labels_64_intra.txt'
block_size = 64

with open(sample_file, 'rb') as f:
    pixels = f.read()
    raw1 = np.frombuffer(pixels, dtype = np.float)
    raw = np.reshape(raw1, [-1, block_size*block_size])
    raw = raw[:2000]
    print(np.shape(raw))
    #raw = raw /255
    print(len(raw))
    #for i in range(len(raw)):
    #    raw[i]= raw[i]-raw[i].mean()
    

with open(label_file, 'r') as f_single_label:
    single_label = f_single_label.read()    
    single_label =np.fromstring (single_label, dtype=np.int32 ,sep=' ')
    single_label = np.reshape(single_label, [-1])
    single_label = single_label[:2000]

print(np.shape(raw))
print(np.shape(single_label))
print(single_label[:10])


path_for_mnist_sprites = os.path.join(LOG_DIR,'mnistdigits.png')
path_for_mnist_metadata = os.path.join(LOG_DIR,'metadata.tsv')

embedding_var = tf.Variable(raw, name=NAME_TO_VISUALISE_VARIABLE)
summary_writer = tf.summary.FileWriter(LOG_DIR)


def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    
    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))
    
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img
    
    return spriteimage

def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits,(-1,block_size,block_size))

def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1-mnist_digits


config  =  projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

# Specify where you find the metadata
embedding.metadata_path = 'metadata.tsv' #'metadata.tsv'

# Specify where you find the sprite (we will create this later)
embedding.sprite.image_path = 'mnistdigits.png' #'mnistdigits.png'
embedding.sprite.single_image_dim.extend([block_size,block_size])

# Say that you want to visualise the embeddings
projector.visualize_embeddings(summary_writer, config)

sess  =  tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 1)

to_visualise = raw
to_visualise = vector_to_matrix_mnist(to_visualise)
to_visualise = invert_grayscale(to_visualise)

sprite_image = create_sprite_image(to_visualise)

plt.imsave(path_for_mnist_sprites,sprite_image,cmap='gray')
plt.imshow(sprite_image,cmap='gray')



#with open(path_for_mnist_metadata, 'w') as f:
#    np.savetxt(f, single_label)

with open(path_for_mnist_metadata,'w') as f:
    f.write("Index\tLabel\n")
    for index,label in enumerate(single_label):
        f.write("%d\t%d\n" % (index,label))
